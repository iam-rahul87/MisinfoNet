"""
src/checkpoint.py — save and load trained model + extractors.
"""

import os, pickle
import numpy as np

SAVE_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_model")
MODEL_PATH  = os.path.join(SAVE_DIR, "model.pkl")
EXTRAS_PATH = os.path.join(SAVE_DIR, "extractors.pkl")
META_PATH   = os.path.join(SAVE_DIR, "meta.txt")


def _mlp_weights(mlp):
    layers = [{"W": l.W.copy(), "b": l.b.copy()} for l in mlp.linears]
    norms  = [{"g": n.gamma.copy(), "b": n.beta.copy()} for n in mlp.norms]
    return {"layers": layers, "norms": norms, "dropout": mlp.dropout}

def _restore_mlp(mlp, state):
    for lin, s in zip(mlp.linears, state["layers"]):
        lin.W[:] = s["W"]; lin.b[:] = s["b"]
    for ln, s in zip(mlp.norms, state["norms"]):
        ln.gamma[:] = s["g"]; ln.beta[:] = s["b"]


def save(model, rag, cnn, rnn, rvnn, meta=None):
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_state = {
        "rag_head":      _mlp_weights(model.rag_head),
        "neural_head":   _mlp_weights(model.neural_head),
        "tau":           model.tau,
        "lam":           model.lam,
        "T_start":       model.T_start,
        "T_end":         model.T_end,
        "neural_in_dim": model.neural_head.linears[0].W.shape[0],
    }
    with open(MODEL_PATH, "wb") as f: pickle.dump(model_state, f)

    extras = {
        # RAG module
        "rag_scaler":     rag.scaler,
        "rag_vectorizer": rag.vectorizer,
        "rag_real_vecs":  rag.real_vecs,
        "rag_fake_vecs":  rag.fake_vecs,
        "rag_threshold":  rag.threshold,
        # CNN uses ngram_vec
        "cnn_vec":        cnn.ngram_vec,
        "cnn_fitted":     cnn._fitted,
        # RNN uses word_vec
        "rnn_vec":        rnn.word_vec,
        "rnn_fitted":     rnn._fitted,
        # RVNN uses char_vec
        "rvnn_vec":       rvnn.char_vec,
        "rvnn_fitted":    rvnn._fitted,
    }
    with open(EXTRAS_PATH, "wb") as f: pickle.dump(extras, f)

    if meta:
        with open(META_PATH, "w") as f:
            for k, v in meta.items(): f.write(f"{k}: {v}\n")

    print(f"  Saved → {SAVE_DIR}/")
    if meta:
        print("  " + "  ".join(f"{k}={v}" for k, v in meta.items()))


def load():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model at {MODEL_PATH}\nRun: python train.py")

    from src.model_v3_lazy import MisinfoNetV3Lazy
    from src.rag_module import RAGModule
    from src.feature_extractors import (CNNFeatureExtractor,
                                        RNNFeatureExtractor,
                                        RVNNFeatureExtractor)

    with open(MODEL_PATH,  "rb") as f: ms = pickle.load(f)
    with open(EXTRAS_PATH, "rb") as f: ex = pickle.load(f)

    # RAG — bypass __init__, restore fitted state directly
    rag           = RAGModule.__new__(RAGModule)
    rag.threshold = ex["rag_threshold"]
    rag.vectorizer= ex["rag_vectorizer"]
    rag.real_vecs = ex["rag_real_vecs"]
    rag.fake_vecs = ex["rag_fake_vecs"]
    rag.scaler    = ex["rag_scaler"]

    # feature extractors — bypass __init__, restore fitted vectorizers
    cnn           = CNNFeatureExtractor.__new__(CNNFeatureExtractor)
    cnn.ngram_vec = ex["cnn_vec"]
    cnn._fitted   = ex["cnn_fitted"]

    rnn           = RNNFeatureExtractor.__new__(RNNFeatureExtractor)
    rnn.word_vec  = ex["rnn_vec"]
    rnn._fitted   = ex["rnn_fitted"]

    rvnn          = RVNNFeatureExtractor.__new__(RVNNFeatureExtractor)
    rvnn.char_vec = ex["rvnn_vec"]
    rvnn._fitted  = ex["rvnn_fitted"]

    # model — rebuild skeleton then restore weights
    dim   = ms["neural_in_dim"]
    model = MisinfoNetV3Lazy(
        rng=np.random.default_rng(0),
        rag_module=rag, cnn_extractor=cnn,
        rnn_extractor=rnn, rvnn_extractor=rvnn,
        neural_in_dim=dim,
        coverage_threshold=ms["tau"],
        lambda_rag=ms["lam"],
        dropout=0.0,
        temp_start=ms["T_start"],
        temp_end=ms["T_end"],
    )
    _restore_mlp(model.rag_head,    ms["rag_head"])
    _restore_mlp(model.neural_head, ms["neural_head"])
    model.eval()

    print(f"  Loaded from {SAVE_DIR}/  (dim={dim}, tau={ms['tau']})")
    return model, rag, cnn, rnn, rvnn


def model_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(EXTRAS_PATH)

def print_meta():
    if os.path.exists(META_PATH):
        print(open(META_PATH).read().strip())
    else:
        print("  No meta file found.")
