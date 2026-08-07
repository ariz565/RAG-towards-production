"""Reranking — stage 2 of two-stage retrieval.

Stage 1 (BM25/dense/hybrid) optimizes recall over the whole corpus fast. Stage 2
rescores the top candidates with a slower, more accurate model for precision:

    retrieve top ~100  →  rerank to top ~10  →  LLM context

- **CrossEncoderReranker** — reads (query, doc) *together* and scores relevance
  directly (BGE-reranker / Cohere Rerank). +5–15 NDCG@10 typically. Lazy model;
  falls back to lexical so the lab runs offline.
- **ColBERTReranker** — late-interaction MaxSim (reuses the embeddings lab).
- **LexicalReranker** — dependency-free token-overlap (offline default/baseline).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter

from retrieval.base import RetrievedDoc

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        ...

    @staticmethod
    def _reorder(docs: list[RetrievedDoc], scores: list[float], top_k: int) -> list[RetrievedDoc]:
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        out = []
        for rank, i in enumerate(order[:top_k]):
            d = docs[i]
            out.append(RetrievedDoc(d.id, d.text, float(scores[i]), rank + 1, f"{d.source}+rerank", d.metadata))
        return out


class LexicalReranker(Reranker):
    """Token-overlap (query terms covered by the doc). Offline baseline."""

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        q = Counter(_TOKEN_RE.findall(query.lower()))
        scores = []
        for d in docs:
            dt = Counter(_TOKEN_RE.findall(d.text.lower()))
            scores.append(sum(min(c, dt.get(t, 0)) for t, c in q.items()))
        return self._reorder(docs, scores, top_k)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._fallback = LexicalReranker()

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        try:
            from sentence_transformers import CrossEncoder
            if self._model is None:
                self._model = CrossEncoder(self.model_name)
            scores = self._model.predict([(query, d.text) for d in docs]).tolist()
            return self._reorder(docs, scores, top_k)
        except Exception:
            return self._fallback.rerank(query, docs, top_k)  # offline degrade


class ColBERTReranker(Reranker):
    """Late-interaction MaxSim rerank (reuses the embeddings lab)."""

    def __init__(self, colbert=None):
        from embeddings.multivector import ColBERTEmbedder
        self.colbert = colbert or ColBERTEmbedder()

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        scores = [self.colbert.score(query, d.text) for d in docs]
        return self._reorder(docs, scores, top_k)
