"""
evaluate.py — Load saved model and run full accuracy evaluation.

Usage:
    python evaluate.py                  # evaluate on synthetic test set
    python evaluate.py --real           # evaluate on 100 hand-crafted claims
    python evaluate.py --real --verbose # show every error
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.dataset import get_splits
from src.checkpoint import load, model_exists, print_meta


# ── 100 hand-crafted real-world claims ───────────────────────────────────

REAL_WORLD_CLAIMS = [
    # EASY REAL — clear, well-known facts (10)
    ("Russia launched a full-scale invasion of Ukraine on February 24 2022.", 0),
    ("The 2020 US presidential election saw over 155 million votes cast, a record turnout.", 0),
    ("Hamas attacked Israel on October 7 2023 killing approximately 1200 people and taking 250 hostages.", 0),
    ("The September 11 attacks were carried out by nineteen al-Qaeda hijackers killing 2977 people.", 0),
    ("The Holocaust resulted in the systematic murder of approximately six million Jewish people.", 0),
    ("NATO was founded in 1949 with 12 member states and has expanded to 32 members as of 2024.", 0),
    ("The Taliban regained control of Afghanistan in August 2021 following the US withdrawal.", 0),
    ("The Iraq War found no weapons of mass destruction that were cited as its justification.", 0),
    ("North Korea has conducted six nuclear tests between 2006 and 2017.", 0),
    ("The Syrian civil war began in 2011 and has caused over 500000 deaths.", 0),

    # EASY FAKE — obvious sensationalism (10)
    ("BREAKING: The 2020 US election was stolen through coordinated deep state fraud across six swing states.", 1),
    ("EXPOSED: Ukraine is a Nazi regime installed by a CIA-backed coup — Russia is liberating it.", 1),
    ("SHOCKING: The September 11 attacks were an inside job planned and executed by the US government.", 1),
    ("The Holocaust death tolls are fabricated wartime Allied propaganda — the numbers are grossly exaggerated.", 1),
    ("LEAKED: The Bucha massacre was entirely staged by Ukraine and NATO intelligence as anti-Russia propaganda.", 1),
    ("BREAKING: George Soros personally funds and organises migrant caravans to destroy Western culture.", 1),
    ("EXPOSED: NATO deliberately provoked Russia into invading Ukraine as a planned US geopolitical trap.", 1),
    ("SHOCKING: The January 6 Capitol riot was a false flag operation staged by the FBI and antifa.", 1),
    ("ISIS was secretly created and funded by the CIA and Israeli Mossad to destabilise the Middle East.", 1),
    ("LEAKED: Every major terrorist attack in the West since 2001 has been a government false flag.", 1),

    # MEDIUM REAL — specific details required (10)
    ("The Bucha massacre involved over 400 confirmed civilian deaths verified by satellite imagery and forensic evidence.", 0),
    ("Trump-appointed Attorney General William Barr stated in December 2020 there was no widespread election fraud.", 0),
    ("The Cybersecurity and Infrastructure Security Agency called the 2020 election the most secure in American history.", 0),
    ("Dominion Voting Systems settled a defamation lawsuit against Fox News for 787 million dollars in 2023.", 0),
    ("The ICC issued arrest warrants for Vladimir Putin and Maria Lvova-Belova in March 2023.", 0),
    ("The 9/11 Commission found no credible evidence of collaboration between Saddam Hussein and al-Qaeda.", 0),
    ("The Chilcot Inquiry concluded in 2016 that the British government overstated the case for war in Iraq.", 0),
    ("Over 60 lawsuits challenging the 2020 election results were dismissed by courts for lack of evidence.", 0),
    ("Osama bin Laden was killed by US Navy SEALs in Abbottabad Pakistan in May 2011.", 0),
    ("The UN General Assembly voted 141 to 5 in March 2022 demanding Russia withdraw from Ukraine.", 0),

    # MEDIUM FAKE — sounds plausible, uses specific detail (10)
    ("Foreign servers in Frankfurt Germany were used by the CIA to change vote tallies on election night 2020.", 1),
    ("Zelensky is a CIA puppet installed by George Soros to serve as a vehicle for Western destabilisation of Russia.", 1),
    ("Ukraine was secretly developing US-funded bioweapons in laboratories across the country before the Russian invasion.", 1),
    ("The United States secretly blew up the Nord Stream pipelines to force Germany into economic war with Russia.", 1),
    ("Ray Epps was an FBI informant who deliberately instigated the January 6 crowd on federal orders.", 1),
    ("Russia invaded Ukraine solely to protect Russian-speaking civilians from genocide by the Ukrainian government.", 1),
    ("The Democratic Party is running a secret child trafficking network from a Washington DC pizza restaurant.", 1),
    ("NATO deliberately expanded eastward to encircle Russia as preparation for a future military invasion.", 1),
    ("The 7/7 London bombings were a false flag staged by British intelligence to justify the Iraq War.", 1),
    ("The Christchurch mosque attack was staged by the New Zealand government to justify gun confiscation.", 1),

    # HARD REAL — counterintuitive or less well-known (10)
    ("Domestic terrorism from far-right extremists has killed more people in the US since 9/11 than Islamist terrorism.", 0),
    ("Research shows immigrants commit crimes at lower rates than native-born citizens in most Western countries.", 0),
    ("Abu Ghraib prison was the site of documented abuse and torture of Iraqi detainees by US military personnel.", 0),
    ("The Iraq War cost the US over 2 trillion dollars and resulted in between 150000 and 600000 Iraqi civilian deaths.", 0),
    ("Russia's invasion of Ukraine caused the largest refugee crisis in Europe since World War II.", 0),
    ("Saudi Arabia has faced scrutiny over whether government officials had foreknowledge of or links to the 9/11 hijackers.", 0),
    ("The five permanent UN Security Council members each hold individual veto power over Security Council resolutions.", 0),
    ("Global refugee numbers exceeded 100 million for the first time in 2022 according to the UNHCR.", 0),
    ("Economic studies consistently show that immigration increases GDP and fills critical labour market shortages.", 0),
    ("Sweden was formally admitted to NATO in March 2024 after Hungary ratified its membership.", 0),

    # HARD FAKE — subtle, no obvious red flags (10)
    ("Multiple credible cardiologists have privately confirmed the COVID vaccine is causing epidemic-level heart damage.", 1),
    ("Leaked internal documents show Western intelligence agencies pre-planned the Ukraine war years in advance.", 1),
    ("Israeli intelligence had complete foreknowledge of the October 7 Hamas attack but deliberately allowed it to happen.", 1),
    ("Anonymous Pentagon sources confirm the US military has secretly controlled ISIS for strategic purposes since 2014.", 1),
    ("A freedom of information request revealed the UK government knew the Iraq WMD dossier was fabricated before publishing it.", 1),
    ("Internal CIA cables obtained by whistleblowers confirm the agency orchestrated the 2014 Maidan coup in Ukraine.", 1),
    ("EU internal documents reveal a secret migration quota agreement never disclosed to the public or parliaments.", 1),
    ("Classified NSA files show the agency had advance warning of the 2015 Paris attacks and chose not to intervene.", 1),
    ("Senior UN officials privately admit the organisation is effectively controlled by a small group of Western billionaires.", 1),
    ("Intercepted communications show George Soros coordinated directly with European leaders on open-borders migration policy.", 1),

    # SOCIAL MEDIA REAL — casual phrasing (10)
    ("Zelensky was actually elected with 73 percent of the vote in 2019, just so people know", 0),
    ("Fun fact: fifteen of the nineteen September 11 hijackers were Saudi nationals, none were Iraqi or Afghan", 0),
    ("The 2020 election was audited multiple times in Georgia including a full hand recount — Biden won every time", 0),
    ("Worth knowing: the US and Russia together hold about 90 percent of the world's nuclear warheads", 0),
    ("Apparently the Nuremberg Trials convicted Nazi leaders for crimes against humanity — first time in history", 0),
    ("The Korean War technically never ended — there was an armistice in 1953 but no formal peace treaty", 0),
    ("Just a reminder that RT was required to register as a foreign agent in the US before the Ukraine invasion", 0),
    ("The Rwandan genocide killed around 800000 people in roughly 100 days in 1994", 0),
    ("So apparently Sweden joining NATO in 2024 means the alliance now has 32 members", 0),
    ("Turns out the Manchester bombing killed 22 people, most of them children at an Ariana Grande concert", 0),

    # SOCIAL MEDIA FAKE — casual disinformation (10)
    ("anyone else notice how the media stopped talking about election fraud the second Biden took office weird", 1),
    ("my friend who works in intelligence says the whole Ukraine war was planned by the CIA years ago just saying", 1),
    ("not to be that person but zelensky literally banned opposition parties and jailed journalists think about it", 1),
    ("did you know russia actually warned the us about the boston bombing beforehand and they ignored it makes you think", 1),
    ("just saying — countries that didn't sanction russia like hungary are doing way better economically right now", 1),
    ("a former army guy told me the january 6 footage they haven't released proves it was all staged from the start", 1),
    ("apparently there are thousands of hours of exculpatory footage from january 6 that the committee destroyed", 1),
    ("wake up people — the deep state has been running the country for decades and elections are just theatre", 1),
    ("anyone notice how every time there's a mass shooting it conveniently happens right when gun control is being debated", 1),
    ("my cousin in europe says locals are terrified but media won't report it because of the pro-immigration agenda", 1),
]


# ── evaluation helpers ────────────────────────────────────────────────────

def full_eval(model, texts, labels, verbose=False):
    model.eval()
    out   = model.predict(texts)
    preds = out["predictions"]
    rm    = out["routed_to_rag"]
    nm    = ~rm

    acc  = (preds == labels).mean()
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    fp   = int(((preds == 1) & (labels == 0)).sum())
    fn   = int(((preds == 0) & (labels == 1)).sum())

    ra = (preds[rm] == labels[rm]).mean() if rm.sum() > 0 else float("nan")
    na = (preds[nm] == labels[nm]).mean() if nm.sum() > 0 else float("nan")

    print(f"\n  Overall accuracy : {acc*100:.2f}%  ({int(acc*len(labels))}/{len(labels)})")
    print(f"  F1 (FAKE)        : {f1:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  False positives  : {fp}")
    print(f"  False negatives  : {fn}")
    print(f"  RAG routed       : {rm.sum()} ({rm.mean()*100:.0f}%)  accuracy={ra*100:.1f}%")
    print(f"  Neural ran       : {nm.sum()} ({nm.mean()*100:.0f}%)  accuracy={na*100:.1f}%")

    if verbose:
        errs = np.where(preds != labels)[0]
        if len(errs):
            print(f"\n  Errors ({len(errs)}):")
            for i in errs:
                gt   = "REAL" if labels[i] == 0 else "FAKE"
                pred = "REAL" if preds[i]  == 0 else "FAKE"
                rt   = "RAG" if rm[i] else "NEU"
                t    = texts[i][:70] + "..." if len(texts[i]) > 73 else texts[i]
                print(f"    [{rt}] GT:{gt}→{pred}  {t}")

    return acc, f1


# ── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MisinfoNet V3 — Evaluate")
    p.add_argument("--real",    action="store_true",
                   help="Evaluate on 100 hand-crafted real-world claims")
    p.add_argument("--verbose", action="store_true",
                   help="Show all errors")
    args = p.parse_args()

    if not model_exists():
        print("\n  No saved model found. Train first:")
        print("  python train.py\n")
        sys.exit(1)

    print("\n  Loading model...", end=" ", flush=True)
    model, *_ = load()
    print("done")

    print("\n  Training metadata:")
    print_meta()

    if args.real:
        # ── 100 hand-crafted claims ───────────────────────────────────────
        texts  = [t for t, _ in REAL_WORLD_CLAIMS]
        labels = np.array([l for _, l in REAL_WORLD_CLAIMS])

        print(f"\n{'='*58}")
        print(f"  EVALUATION — 80 hand-crafted real-world claims")
        print(f"{'='*58}")

        full_eval(model, texts, labels, verbose=args.verbose)

        # per-category
        cats = [
            ("Easy real (10)",     slice(0, 10)),
            ("Easy fake (10)",     slice(10, 20)),
            ("Medium real (10)",   slice(20, 30)),
            ("Medium fake (10)",   slice(30, 40)),
            ("Hard real (10)",     slice(40, 50)),
            ("Hard fake (10)",     slice(50, 60)),
            ("Social real (10)",   slice(60, 70)),
            ("Social fake (10)",   slice(70, 80)),
        ]
        print(f"\n  Per-category:")
        out   = model.predict(texts)
        preds = out["predictions"]
        for cat, sl in cats:
            sl_y = labels[sl]
            a    = (preds[sl] == sl_y).mean()
            bar  = "█" * int(a * 20) + "░" * (20 - int(a * 20))
            print(f"  {cat:<22} [{bar}] {a:.0%}")

    else:
        # ── synthetic test set ────────────────────────────────────────────
        print(f"\n  Loading synthetic test set...", end=" ", flush=True)
        _, _, (te_t, te_y) = get_splits(42)
        print(f"done ({len(te_t)} claims)")

        print(f"\n{'='*58}")
        print(f"  EVALUATION — synthetic test set (n={len(te_t)})")
        print(f"{'='*58}")
        print(f"  Note: synthetic claims use same topics as training.")
        print(f"  For real-world accuracy, run: python evaluate.py --real")

        full_eval(model, te_t, te_y, verbose=args.verbose)

    print(f"\n{'='*58}\n")


if __name__ == "__main__":
    main()
