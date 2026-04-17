"""
train.py — Train MisinfoNet V3 and save to saved_model/

Usage:
    python train.py                        # train with defaults
    python train.py --epochs 100           # more epochs
    python train.py --threshold 0.20       # looser RAG gate
    python train.py --seed 7              # different random seed

After training, run:
    python predict.py --text "Your claim here"
    python evaluate.py
"""

import sys, os, copy, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics import f1_score

from src.dataset import get_splits
from src.rag_module import RAGModule
from src.feature_extractors import (CNNFeatureExtractor,
                                     RNNFeatureExtractor,
                                     RVNNFeatureExtractor)
from src.model_v3_lazy import MisinfoNetV3Lazy
from src.checkpoint import save


# ── argument parsing ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MisinfoNet V3")
    p.add_argument("--epochs",    type=int,   default=60,   help="training epochs (default 60)")
    p.add_argument("--batch",     type=int,   default=128,  help="batch size (default 128)")
    p.add_argument("--lr",        type=float, default=1e-3, help="peak learning rate (default 0.001)")
    p.add_argument("--threshold", type=float, default=0.30, help="RAG coverage threshold (default 0.30)")
    p.add_argument("--lambda_rag",type=float, default=0.6,  help="auxiliary RAG loss weight (default 0.6)")
    p.add_argument("--dropout",   type=float, default=0.25, help="dropout rate (default 0.25)")
    p.add_argument("--seed",      type=int,   default=42,   help="random seed (default 42)")
    p.add_argument("--pretrain_epochs", type=int, default=12,
                   help="curriculum pretrain epochs (default 12)")
    return p.parse_args()


# ── feature extraction ────────────────────────────────────────────────────

def build_extractors(all_texts, threshold):
    print("  Building feature extractors...", end=" ", flush=True)
    rag  = RAGModule(coverage_threshold=threshold)
    cnn  = CNNFeatureExtractor();  cnn.fit(all_texts)
    rnn  = RNNFeatureExtractor();  rnn.fit(all_texts)
    rvnn = RVNNFeatureExtractor(); rvnn.fit(all_texts)
    print("done")
    return rag, cnn, rnn, rvnn


def extract_features(texts, rag, cnn, rnn, rvnn):
    rf, cov = rag.extract_features(texts)
    return rf, cnn.transform(texts), rnn.transform(texts), rvnn.transform(texts), cov


# ── training loop ─────────────────────────────────────────────────────────

def train_loop(model, tr, va, va_t,
               epochs, batch, lr, seed,
               pretrain=False, pretrain_epochs=12, label="V3"):
    """
    Train model for given epochs. Returns best model by val F1.
    tr  : tuple (rf, cf, rn, rv, cov, labels) — pre-computed features
    va  : same for validation
    va_t: raw validation texts (for V3 lazy predict)
    """
    tr_rf, tr_cf, tr_rn, tr_rv, tr_cov, tr_y = tr
    va_rf, va_cf, va_rn, va_rv, va_cov, va_y = va

    rng = np.random.default_rng(seed)

    # ── Stage 1: curriculum pretrain for V3 ──────────────────────────────
    if pretrain and hasattr(model, "T_start"):
        hi = tr_cov >= 0.40
        model.lam = 1.0
        model.T   = model.T_start
        print(f"\n  Stage 1 — curriculum pretrain ({pretrain_epochs} epochs, "
              f"{hi.sum()} high-coverage samples)...")
        for ep in range(pretrain_epochs):
            model.train()
            idx = rng.permutation(hi.sum())
            for s in range(0, hi.sum(), batch):
                b = idx[s:s+batch]
                if len(b) < 2: continue
                model.loss_and_step(
                    tr_rf[hi][b], tr_cf[hi][b], tr_rn[hi][b],
                    tr_rv[hi][b], tr_cov[hi][b], tr_y[hi][b], lr=3e-3)
        model.lam = 0.6
        print("  Pretrain done")

    # ── Stage 2: joint training ───────────────────────────────────────────
    print(f"\n  Stage 2 — joint training ({epochs} epochs)...")
    best_f1 = -1
    best_m  = None
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        if hasattr(model, "anneal_temperature"):
            model.anneal_temperature(ep - 1, epochs)

        # warmup 5 epochs then cosine decay
        lr_ep = lr * ep / 5 if ep <= 5 else \
                lr * 0.5 * (1 + np.cos(np.pi * (ep - 5) / (epochs - 5)))

        idx = rng.permutation(len(tr_y))
        ep_loss, ep_acc = [], []

        for s in range(0, len(tr_y), batch):
            b = idx[s:s+batch]
            if len(b) < 2: continue
            st = model.loss_and_step(
                tr_rf[b], tr_cf[b], tr_rn[b], tr_rv[b],
                tr_cov[b], tr_y[b], lr=lr_ep)
            ep_loss.append(st["loss"])
            ep_acc.append(st["acc"])

        # validation
        model.eval()
        if isinstance(model, MisinfoNetV3Lazy):
            vp = model.predict(va_t)["predictions"]
        else:
            vp = model.predict(va_rf, va_cf, va_rn, va_rv, va_cov)["predictions"]
        vf1  = f1_score(va_y, vp, zero_division=0)
        vacc = (vp == va_y).mean()

        if vf1 > best_f1:
            best_f1 = vf1
            best_m  = copy.deepcopy(model)

        if ep % 10 == 0 or ep == 1:
            elapsed = time.perf_counter() - t_start
            t_gate  = f"  T={model.T:.3f}" if hasattr(model, "T") else ""
            print(f"  [{label} ep {ep:3d}]  loss={np.mean(ep_loss):.4f}  "
                  f"train={np.mean(ep_acc):.3f}  val_f1={vf1:.4f}  "
                  f"val_acc={vacc:.3f}{t_gate}  ({elapsed:.0f}s)")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Best val F1: {best_f1:.4f}  (training took {elapsed:.0f}s)")
    return best_m, best_f1


# ── main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  MisinfoNet V3 — Training")
    print("="*60)
    print(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"  threshold={args.threshold}  lambda_rag={args.lambda_rag}")
    print(f"  dropout={args.dropout}  seed={args.seed}")

    # ── data ──────────────────────────────────────────────────────────────
    (tr_t, tr_y), (va_t, va_y), (te_t, te_y) = get_splits(args.seed)
    all_t = tr_t + va_t + te_t
    print(f"\n  Train={len(tr_t)}  Val={len(va_t)}  Test={len(te_t)}")

    # ── extractors ────────────────────────────────────────────────────────
    rag, cnn, rnn, rvnn = build_extractors(all_t, args.threshold)

    print("  Extracting features...", end=" ", flush=True)
    tr = extract_features(tr_t, rag, cnn, rnn, rvnn) + (tr_y,)
    va = extract_features(va_t, rag, cnn, rnn, rvnn) + (va_y,)
    te = extract_features(te_t, rag, cnn, rnn, rvnn) + (te_y,)
    print("done")

    DIM = tr[1].shape[1] + tr[2].shape[1] + tr[3].shape[1] + tr[0].shape[1]
    print(f"  Neural input dim : {DIM}")
    print(f"  RAG coverage     : {(tr[4] >= args.threshold).mean()*100:.1f}% (train)")
    from src.knowledge_base import KB
    print(f"  KB entries       : {len(KB)}")

    rng_model = np.random.default_rng(args.seed)

    # ── build V3 ──────────────────────────────────────────────────────────
    model = MisinfoNetV3Lazy(
        rng=rng_model,
        rag_module=rag,
        cnn_extractor=cnn,
        rnn_extractor=rnn,
        rvnn_extractor=rvnn,
        neural_in_dim=DIM,
        coverage_threshold=args.threshold,
        lambda_rag=args.lambda_rag,
        dropout=args.dropout,
        temp_start=5.0,
        temp_end=0.05,
    )
    print(f"  Parameters       : {model.param_count():,}")

    # ── train ─────────────────────────────────────────────────────────────
    best_model, best_f1 = train_loop(
        model, tr, va, va_t,
        epochs         = args.epochs,
        batch          = args.batch,
        lr             = args.lr,
        seed           = args.seed,
        pretrain       = True,
        pretrain_epochs= args.pretrain_epochs,
        label          = "V3",
    )

    # ── test set evaluation ───────────────────────────────────────────────
    best_model.eval()
    out   = best_model.predict(te_t)
    preds = out["predictions"]
    acc   = (preds == te_y).mean()
    f1    = f1_score(te_y, preds, zero_division=0)
    fp    = int(((preds == 1) & (te_y == 0)).sum())
    fn    = int(((preds == 0) & (te_y == 1)).sum())

    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  F1 (FAKE)   : {f1:.4f}")
    print(f"  FP={fp}  FN={fn}")
    print(f"  RAG routed  : {out['n_rag_routed']}/{len(te_t)} "
          f"({out['rag_coverage']*100:.1f}%)")
    print(f"  Neural ran  : {out['n_neu_routed']}/{len(te_t)}")

    # ── save ──────────────────────────────────────────────────────────────
    print(f"\n  Saving model...")
    meta = {
        "epochs":    args.epochs,
        "threshold": args.threshold,
        "val_f1":    f"{best_f1:.4f}",
        "test_acc":  f"{acc*100:.2f}%",
        "test_f1":   f"{f1:.4f}",
        "params":    model.param_count(),
        "seed":      args.seed,
    }
    save(best_model, rag, cnn, rnn, rvnn, meta)

    print(f"\n  Done. Run predictions with:")
    print(f'  python predict.py --text "Your claim here"')
    print(f"  python evaluate.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
