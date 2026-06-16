"""Reciprocal Rank Fusion (RRF) — combine ranked lists by RANK, not score.

Why rank-based: BM25 scores are unbounded positives; cosine is [-1, 1]. Averaging
raw scores lets BM25 dominate. RRF sidesteps normalization entirely — it only
uses each item's *position* in each list:

    score(d) = Σ_lists  weight / (k + rank_in_list)

Cormack et al. (SIGIR 2009). Production default: k=60, ~20 candidates per
retriever. Also used to merge multi-query results (RAG-Fusion).
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id-lists → [(doc_id, fused_score)] sorted descending."""
    scores: dict[str, float] = {}
    for li, ranked in enumerate(ranked_lists):
        w = weights[li] if weights else 1.0
        for rank, doc_id in enumerate(ranked):  # rank is 0-based
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
