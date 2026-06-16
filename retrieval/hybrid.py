"""Hybrid retrieval — BM25 + dense, fused with RRF.

The production default: lexical catches exact terms, dense catches paraphrases,
RRF merges them without score-normalization headaches. Tune the per-retriever
prefetch by corpus (jargon-heavy → more BM25 candidates; conversational → more dense).
"""

from __future__ import annotations

from retrieval.base import Doc, Retriever, RetrievedDoc
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever(Retriever):
    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        rrf_k: int = 60,
        prefetch: int = 20,
        weights: list[float] | None = None,   # [bm25_weight, dense_weight]
    ):
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k
        self.prefetch = prefetch
        self.weights = weights
        self._by_id: dict[str, Doc] = {}

    def index(self, docs: list[Doc]) -> None:
        self.bm25.index(docs)
        self.dense.index(docs)
        self._by_id = {d.id: d for d in docs}

    def search(self, query: str, k: int = 10) -> list[RetrievedDoc]:
        bm25_ids = [d.id for d in self.bm25.search(query, self.prefetch)]
        dense_ids = [d.id for d in self.dense.search(query, self.prefetch)]
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=self.rrf_k, weights=self.weights)
        return [
            RetrievedDoc(doc_id, self._by_id[doc_id].text, score, rank + 1, "hybrid")
            for rank, (doc_id, score) in enumerate(fused[:k])
            if doc_id in self._by_id
        ]
