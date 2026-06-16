"""Composable retrieval pipeline: transform → retrieve → fuse → rerank.

This is the production shape of advanced RAG retrieval:
1. (optional) transform the query (multi-query / HyDE / decomposition / step-back),
2. retrieve candidates for each resulting query,
3. fuse the ranked lists with RRF (RAG-Fusion) when there's more than one,
4. (optional) rerank the candidate pool with a cross-encoder,
5. return the top-k.
"""

from __future__ import annotations

from retrieval.base import Retriever, RetrievedDoc
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.query_transform import QueryTransform
from retrieval.rerank import Reranker


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        transform: QueryTransform | None = None,
        reranker: Reranker | None = None,
        k: int = 5,
        prefetch: int = 20,
        rrf_k: int = 60,
    ):
        self.retriever = retriever
        self.transform = transform
        self.reranker = reranker
        self.k = k
        self.prefetch = prefetch
        self.rrf_k = rrf_k

    def search(self, query: str) -> list[RetrievedDoc]:
        queries = self.transform.transform(query) if self.transform else [query]

        by_id: dict[str, RetrievedDoc] = {}
        ranked_lists: list[list[str]] = []
        for q in queries:
            results = self.retriever.search(q, self.prefetch)
            ranked_lists.append([r.id for r in results])
            for r in results:
                by_id.setdefault(r.id, r)

        if len(ranked_lists) > 1:  # RAG-Fusion across transformed queries
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
            candidates = [by_id[i] for i, _ in fused if i in by_id]
        else:
            candidates = [by_id[i] for i in ranked_lists[0] if i in by_id]

        candidates = candidates[: self.prefetch]
        if self.reranker:
            return self.reranker.rerank(query, candidates, self.k)
        return candidates[: self.k]
