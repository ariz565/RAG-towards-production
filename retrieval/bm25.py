"""BM25 — sparse lexical retrieval (pure-Python Okapi BM25).

The lexical workhorse: exact-term matching with TF saturation (k1) and length
normalization (b). Unbeatable for codes, IDs, names, and rare jargon that dense
embeddings smooth away — which is exactly why hybrid (BM25 + dense) wins.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from retrieval.base import Doc, Retriever, RetrievedDoc, matches_filter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[Doc] = []
        self._tf: list[Counter] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def index(self, docs: list[Doc]) -> None:
        self._docs = docs
        self._tf = [Counter(_tokenize(d.text)) for d in docs]
        self._doc_len = [sum(tf.values()) for tf in self._tf]
        self._avgdl = (sum(self._doc_len) / len(docs)) if docs else 0.0

        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(docs)
        # Okapi BM25 idf with +1 smoothing (always positive).
        self._idf = {
            term: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5)) for term, dfi in df.items()
        }

    def search(self, query: str, k: int = 10, filter_metadata: dict | None = None) -> list[RetrievedDoc]:
        q_terms = _tokenize(query)
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            if not matches_filter(self._docs[i].metadata, filter_metadata):
                continue
            score = 0.0
            dl = self._doc_len[i]
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                score += idf * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedDoc(self._docs[i].id, self._docs[i].text, s, rank + 1, "bm25", self._docs[i].metadata)
            for rank, (i, s) in enumerate(scored[:k])
        ]
