"""
RAG Module — Retrieval-Augmented Classification
TF-IDF retrieval over the KB, returns 6 normalized similarity features.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.knowledge_base import REAL_DOCS, FAKE_DOCS, ALL_DOCS, ALL_LABELS


class RAGModule:
    def __init__(self, coverage_threshold: float = 0.15):
        self.threshold = coverage_threshold
        self._build_index()

    def _build_index(self):
        """Fit a single TF-IDF space over all KB docs."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams for richer overlap
            min_df=1,
            sublinear_tf=True,
        )
        self.vectorizer.fit(ALL_DOCS)

        real_vecs = self.vectorizer.transform(REAL_DOCS).toarray()   # (R, V)
        fake_vecs = self.vectorizer.transform(FAKE_DOCS).toarray()   # (F, V)

        # L2-normalize so cosine sim = dot product
        self.real_vecs = self._l2(real_vecs)   # (R, V)
        self.fake_vecs = self._l2(fake_vecs)   # (F, V)

        # Scaler fitted on KB itself for feature normalisation
        self.scaler = StandardScaler()
        kb_feats = np.vstack([self._raw_features(d) for d in ALL_DOCS])
        self.scaler.fit(kb_feats)

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _l2(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return mat / norms

    def _embed(self, text: str) -> np.ndarray:
        vec = self.vectorizer.transform([text]).toarray()[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _raw_features(self, text: str) -> np.ndarray:
        v = self._embed(text)
        real_sims = self.real_vecs @ v          # (R,)
        fake_sims = self.fake_vecs @ v          # (F,)
        real_max  = float(real_sims.max())
        real_mean = float(real_sims.mean())
        fake_max  = float(fake_sims.max())
        fake_mean = float(fake_sims.mean())
        top1      = max(real_max, fake_max)
        gap       = real_max - fake_max         # + → leans REAL, - → FAKE
        return np.array([real_max, real_mean, fake_max, fake_mean, top1, gap],
                        dtype=np.float32)

    # ── public API ────────────────────────────────────────────────────────────
    def extract_features(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        features  : (N, 6)  normalised RAG features
        coverage  : (N,)    top-1 cosine similarity (= coverage score)
        """
        raw = np.vstack([self._raw_features(t) for t in texts])   # (N, 6)
        features = self.scaler.transform(raw)                      # (N, 6)
        coverage = raw[:, 4]                                       # top1_sim
        return features.astype(np.float32), coverage.astype(np.float32)

    def coverage_mask(self, coverage: np.ndarray) -> np.ndarray:
        """Boolean mask: True where RAG has sufficient KB coverage."""
        return coverage >= self.threshold
