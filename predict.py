"""
predict.py — Load saved model and classify claims.

Usage:
    # Single claim
    python predict.py --text "Vaccines contain microchips"

    # Multiple claims from a text file (one per line)
    python predict.py --file claims.txt

    # Interactive mode (type claims one by one)
    python predict.py --interactive

    # Run built-in demo
    python predict.py --demo
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.checkpoint import load, model_exists, print_meta


# ── demo claims ───────────────────────────────────────────────────────────

DEMO_CLAIMS = [
    # Real — politics, wars, terrorism
    ("Russia launched a full-scale invasion of Ukraine on February 24 2022.", 0),
    ("The 2020 US election was certified by all 50 states with no widespread fraud found.", 0),
    ("Hamas attacked Israel on October 7 2023 killing approximately 1200 people.", 0),
    ("The September 11 attacks were carried out by al-Qaeda killing nearly 3000 people.", 0),
    ("The Holocaust resulted in the systematic murder of approximately six million Jewish people.", 0),
    ("NATO was founded in 1949 and now has 32 member states following Sweden's accession.", 0),
    ("The Bucha massacre involved over 400 civilian deaths confirmed by satellite imagery.", 0),
    ("Over 60 courts dismissed lawsuits challenging the 2020 US election for lack of evidence.", 0),
    ("The 7 July 2005 London bombings were carried out by four British Islamist men.", 0),
    ("The Taliban regained control of Afghanistan in August 2021 after the US withdrawal.", 0),
    # Fake — political disinformation
    ("BREAKING: The 2020 US election was stolen through coordinated deep state fraud.", 1),
    ("EXPOSED: Ukraine is a Nazi regime installed by a CIA-backed coup — Russia is the victim.", 1),
    ("SHOCKING: The Bucha massacre was staged by Ukraine and NATO as anti-Russia propaganda.", 1),
    ("LEAKED: The September 11 attacks were an inside job planned by the US government.", 1),
    ("The Holocaust death tolls are fabricated wartime propaganda — the numbers are exaggerated.", 1),
    ("NATO deliberately provoked Russia into invading Ukraine as a planned geopolitical trap.", 1),
    ("George Soros personally funds migrant caravans to destroy Western culture and elections.", 1),
    ("Every major terrorist attack in the West has been a government false flag operation.", 1),
    ("The January 6 Capitol riot was staged by the FBI and antifa to frame Trump supporters.", 1),
    ("ISIS was secretly created and funded by the CIA and Israeli Mossad.", 1),
]


# ── display helpers ───────────────────────────────────────────────────────

def format_result(text, pred, prob, routed_to_rag):
    label     = "FAKE" if pred == 1 else "REAL"
    conf      = prob[pred] * 100
    route     = "RAG (KB match)" if routed_to_rag else "Neural (novel claim)"
    bar_width = 30
    fake_bar  = int(prob[1] * bar_width)
    real_bar  = bar_width - fake_bar
    bar       = "█" * fake_bar + "░" * real_bar

    print(f"\n  Claim   : {text}")
    print(f"  Verdict : {label}  ({conf:.1f}% confidence)")
    print(f"  Route   : {route}")
    print(f"  FAKE [{bar}] REAL   {prob[1]*100:.0f}% / {prob[0]*100:.0f}%")


def print_separator():
    print("  " + "─" * 58)


# ── classify functions ────────────────────────────────────────────────────

def classify_one(model, text: str):
    """Classify a single claim and print result."""
    model.eval()
    out  = model.predict([text])
    pred = int(out["predictions"][0])
    prob = out["probabilities"][0]
    rag  = bool(out["routed_to_rag"][0])
    format_result(text, pred, prob, rag)
    return pred, prob


def classify_many(model, texts: list[str]):
    """Classify a list of claims, return results."""
    model.eval()
    out   = model.predict(texts)
    preds = out["predictions"]
    probs = out["probabilities"]
    rags  = out["routed_to_rag"]

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text":    text,
            "label":   "FAKE" if preds[i] == 1 else "REAL",
            "pred":    int(preds[i]),
            "conf":    float(probs[i][preds[i]]) * 100,
            "rag":     bool(rags[i]),
        })
    return results


# ── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MisinfoNet V3 — Predict")
    p.add_argument("--text",        type=str,  default=None,
                   help="Single claim to classify")
    p.add_argument("--file",        type=str,  default=None,
                   help="Text file with one claim per line")
    p.add_argument("--interactive", action="store_true",
                   help="Interactive mode — type claims one by one")
    p.add_argument("--demo",        action="store_true",
                   help="Run demo on 10 built-in claims")
    args = p.parse_args()

    if not any([args.text, args.file, args.interactive, args.demo]):
        p.print_help()
        print("\n  Example:")
        print('  python predict.py --text "Vaccines cause autism"')
        print('  python predict.py --demo')
        sys.exit(0)

    # ── load model ────────────────────────────────────────────────────────
    if not model_exists():
        print("\n  No saved model found. Train first:")
        print("  python train.py\n")
        sys.exit(1)

    print("\n  Loading model...", end=" ", flush=True)
    model, *_ = load()
    print("done\n")

    # ── single claim ──────────────────────────────────────────────────────
    if args.text:
        print_separator()
        classify_one(model, args.text)
        print_separator()

    # ── file ──────────────────────────────────────────────────────────────
    elif args.file:
        if not os.path.exists(args.file):
            print(f"  File not found: {args.file}")
            sys.exit(1)
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"  Classifying {len(texts)} claims from {args.file}...\n")
        print_separator()
        results = classify_many(model, texts)
        for r in results:
            route = "RAG" if r["rag"] else "NEU"
            print(f"  [{route}] {r['label']:4s} {r['conf']:5.1f}%  {r['text'][:70]}")
        print_separator()
        n_fake = sum(1 for r in results if r["pred"] == 1)
        n_real = len(results) - n_fake
        print(f"\n  {len(results)} claims: {n_fake} FAKE, {n_real} REAL\n")

    # ── interactive ───────────────────────────────────────────────────────
    elif args.interactive:
        print("  Interactive mode — type a claim and press Enter.")
        print("  Type 'quit' or press Ctrl+C to exit.\n")
        print_separator()
        try:
            while True:
                text = input("  Claim: ").strip()
                if not text:
                    continue
                if text.lower() in ("quit", "exit", "q"):
                    break
                classify_one(model, text)
                print_separator()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.\n")

    # ── demo ──────────────────────────────────────────────────────────────
    elif args.demo:
        texts  = [t for t, _ in DEMO_CLAIMS]
        labels = [l for _, l in DEMO_CLAIMS]
        lmap   = {0: "REAL", 1: "FAKE"}

        print(f"  {'Result':<8} {'Route':<7} {'Conf':>5}  Claim")
        print_separator()

        results = classify_many(model, texts)
        correct = 0
        for r, gt in zip(results, labels):
            sym   = "✓" if r["pred"] == gt else "✗"
            route = "RAG" if r["rag"] else "NEU"
            trunc = r["text"][:65] + "..." if len(r["text"]) > 68 else r["text"]
            if r["pred"] == gt: correct += 1
            print(f"  {sym}{r['label']:<7} [{route:<3}] {r['conf']:>4.0f}%  {trunc}")

        print_separator()
        print(f"\n  Score: {correct}/{len(DEMO_CLAIMS)} correct ({correct/len(DEMO_CLAIMS)*100:.0f}%)\n")


if __name__ == "__main__":
    main()
