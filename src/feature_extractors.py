"""
Neural branch feature extractors.

CNN  — character/word n-gram pattern features (local sensationalism signals)
RNN  — positional sequence features (narrative consistency signals)
RVNN — syntactic structure features (hedging, attribution, negation signals)

Each returns a fixed-dim numpy array per sample.
These feed into a shared MLP classifier in the V3 model.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# ── shared vocabulary built lazily ──────────────────────────────────────────
_SENSATIONAL_PATTERNS = [
    r"\bBREAKING\b", r"\bSHOCKING\b", r"\bCONFIRMED\b", r"\bEXPOSED\b",
    r"\bSECRET\b", r"\bHIDING\b", r"\bCOVER.UP\b", r"\bDON'T WANT YOU\b",
    r"\bWAKE UP\b", r"\bSHEEP\b", r"\bTRUTH\b", r"\bHOAX\b",
    r"\bCONSPIRACY\b", r"\bLIES\b", r"\bAGENDA\b", r"\bELITE\b",
    r"\bCURE\b", r"\bMIRACLE\b", r"\bTOXIC\b", r"\bPOISON\b",
    r"\bPROVEN\b", r"\bGUARANTEED\b", r"\bNATURAL\b", r"\bBIG PHARMA\b",
    r"\bDEPOPULATION\b", r"\bNWO\b", r"\bDEEP STATE\b", r"\bCHIP\b",
    r"\bMINDCONTROL\b", r"\bGENETIC\b",
]

_HEDGE_WORDS = [
    "reportedly", "allegedly", "sources say", "some claim", "many believe",
    "it is said", "rumored", "supposedly", "apparently", "unconfirmed",
    "insiders say", "anonymous sources", "leaked documents", "whistleblower",
]

_NEGATION_WORDS = [
    "no evidence", "not true", "false claim", "debunked", "myth",
    "disproven", "fabricated", "no proof", "never", "impossible",
    "never happened", "scientists confirm", "experts say", "studies show",
]

_AUTHORITATIVE_WORDS = [
    "study", "research", "scientists", "doctors", "evidence", "trial",
    "peer-reviewed", "journal", "CDC", "WHO", "FDA", "clinical", "data",
    "researchers", "analysis", "findings", "published",
]


class CNNFeatureExtractor:
    """
    Simulates TextCNN: detects local n-gram patterns.
    Features: sensationalism score, n-gram TF-IDF projections,
              uppercase ratio, exclamation density, caps-word count.
    Output dim: 32
    """

    def __init__(self):
        self.ngram_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(2, 4),
            max_features=500,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]):
        self.ngram_vec.fit(texts)
        self._fitted = True
        return self

    def _handcrafted(self, text: str) -> np.ndarray:
        upper_text = text.upper()
        feats = []

        # sensationalism pattern hits (normalised)
        hits = sum(1 for p in _SENSATIONAL_PATTERNS if re.search(p, upper_text))
        feats.append(hits / len(_SENSATIONAL_PATTERNS))

        # uppercase ratio
        alpha = [c for c in text if c.isalpha()]
        feats.append(sum(c.isupper() for c in alpha) / max(len(alpha), 1))

        # exclamation density
        words = text.split()
        feats.append(sum(w.endswith("!") for w in words) / max(len(words), 1))

        # all-caps word count (normalised)
        feats.append(sum(w.isupper() and len(w) > 2 for w in words) / max(len(words), 1))

        # authoritative word density
        lower = text.lower()
        auth = sum(w in lower for w in _AUTHORITATIVE_WORDS)
        feats.append(auth / max(len(words), 1))

        # hedge word density
        hedge = sum(h in lower for h in _HEDGE_WORDS)
        feats.append(hedge / max(len(words), 1))

        return np.array(feats, dtype=np.float32)   # 6-dim

    def transform(self, texts: list[str]) -> np.ndarray:
        hand = np.vstack([self._handcrafted(t) for t in texts])  # (N, 6)
        if self._fitted:
            ngram = self.ngram_vec.transform(texts).toarray()     # (N, 500)
            # reduce to 26 dims via column mean across 20-col windows
            chunks = [ngram[:, i:i+20].mean(axis=1, keepdims=True)
                      for i in range(0, 500, 20)]                 # 25 × (N,1)
            ngram_red = np.hstack(chunks[:26])                    # (N, 26)
        else:
            ngram_red = np.zeros((len(texts), 26), dtype=np.float32)
        return np.hstack([hand, ngram_red]).astype(np.float32)    # (N, 32)


class RNNFeatureExtractor:
    """
    Simulates BiLSTM: detects sequential/narrative signals.
    Features: positional word weights, topic-shift indicators,
              sentiment arc, coherence proxies.
    Output dim: 32
    """

    def __init__(self):
        self.word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            max_features=1000,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]):
        self.word_vec.fit(texts)
        self._fitted = True
        return self

    def _positional_features(self, text: str) -> np.ndarray:
        words = text.lower().split()
        n = max(len(words), 1)
        feats = []

        # beginning / middle / end authoritative density
        thirds = [words[:n//3], words[n//3:2*n//3], words[2*n//3:]]
        for segment in thirds:
            auth = sum(w in _AUTHORITATIVE_WORDS for w in segment)
            feats.append(auth / max(len(segment), 1))

        # beginning / end hedge density
        for segment in [thirds[0], thirds[2]]:
            hedge = sum(any(h in " ".join(segment) for h in _HEDGE_WORDS)
                        for _ in [1])
            feats.append(float(hedge))

        # negation in second half vs first half
        first_half = " ".join(words[:n//2])
        second_half = " ".join(words[n//2:])
        neg_first  = sum(n_ in first_half  for n_ in _NEGATION_WORDS)
        neg_second = sum(n_ in second_half for n_ in _NEGATION_WORDS)
        feats.append(float(neg_second - neg_first))

        # sentence length (proxy for narrative complexity)
        feats.append(min(n / 50.0, 1.0))

        # vocabulary richness (type-token ratio)
        feats.append(len(set(words)) / n)

        # question-like structure
        feats.append(float(text.strip().endswith("?")))

        return np.array(feats, dtype=np.float32)   # 10 dims

    def transform(self, texts: list[str]) -> np.ndarray:
        pos = np.vstack([self._positional_features(t) for t in texts])  # (N,10)
        if self._fitted:
            wv = self.word_vec.transform(texts).toarray()  # (N, 1000)
            # compress to 22 dims
            chunks = [wv[:, i:i+46].mean(axis=1, keepdims=True)
                      for i in range(0, 1000, 46) if wv[:, i:i+46].shape[1] > 0]
            wv_red = np.hstack(chunks[:22]) if chunks else np.zeros((len(texts), 22), dtype=np.float32)  # (N, 22)
        else:
            wv_red = np.zeros((len(texts), 22), dtype=np.float32)
        return np.hstack([pos, wv_red]).astype(np.float32)  # (N, 32)


class RVNNFeatureExtractor:
    """
    Simulates Tree-LSTM: detects syntactic structure signals.
    Features: hedge/attribution phrase counts, passive voice indicators,
              negation depth, structural complexity proxies.
    Output dim: 32
    """

    def __init__(self):
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=500,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]):
        self.char_vec.fit(texts)
        self._fitted = True
        return self

    def _syntactic_features(self, text: str) -> np.ndarray:
        lower = text.lower()
        words = lower.split()
        n = max(len(words), 1)
        feats = []

        # hedge phrase density
        hedge_count = sum(h in lower for h in _HEDGE_WORDS)
        feats.append(hedge_count / n)

        # authoritative phrase density
        auth_count = sum(a in lower for a in _AUTHORITATIVE_WORDS)
        feats.append(auth_count / n)

        # negation density
        neg_count = sum(neg in lower for neg in _NEGATION_WORDS)
        feats.append(neg_count / n)

        # passive voice proxy (contains "is/are/was/were + verb-ed")
        passive = len(re.findall(r'\b(is|are|was|were)\s+\w+ed\b', lower))
        feats.append(passive / n)

        # attribution marker ("according to", "said", "claims", "states")
        attribution = len(re.findall(
            r'\b(according to|said|claims|states|alleges|reports)\b', lower))
        feats.append(attribution / n)

        # clause depth proxy (comma count / length)
        feats.append(text.count(",") / n)

        # conjunction density (and/but/however/although)
        conj = len(re.findall(r'\b(and|but|however|although|yet|whereas)\b', lower))
        feats.append(conj / n)

        # number of numeric claims ("100%", "proven", "all", "never")
        absolutes = len(re.findall(
            r'\b(100%|all|never|always|every|none|completely|proven|guaranteed)\b', lower))
        feats.append(absolutes / n)

        # embedded quotation depth (quote marks)
        feats.append((text.count('"') + text.count("'")) / n)

        # comparative structure ("more than", "less than", "better than")
        compar = len(re.findall(r'\b(more than|less than|better than|worse than)\b', lower))
        feats.append(compar / n)

        return np.array(feats, dtype=np.float32)   # 10 dims

    def transform(self, texts: list[str]) -> np.ndarray:
        syn = np.vstack([self._syntactic_features(t) for t in texts])  # (N,10)
        if self._fitted:
            cv = self.char_vec.transform(texts).toarray()  # (N, 500)
            chunks = [cv[:, i:i+23].mean(axis=1, keepdims=True)
                      for i in range(0, 500, 23)]
            cv_red = np.hstack(chunks[:22])               # (N, 22)
        else:
            cv_red = np.zeros((len(texts), 22), dtype=np.float32)
        return np.hstack([syn, cv_red]).astype(np.float32)  # (N, 32)
