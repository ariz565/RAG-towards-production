"""Maximal Marginal Relevance (MMR) — diversify the final retrieved set.

Reranking optimizes pure relevance, which can return several near-duplicate
passages. MMR re-selects from the candidate pool by balancing relevance to the
query against novelty versus what's already chosen:

    MMR = argmax_{d in pool}  [ λ · sim(d, query) − (1 − λ) · max_{s in chosen} sim(d, s) ]

λ = 1.0 → pure relevance (no diversity); λ = 0.0 → pure diversity.

It embeds the candidates once (the pool is small — tens of items), so it's an
opt-in step (`settings.mmr_enabled`) rather than always-on.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.services.embeddings import cosine_similarity, embed_query, embed_texts

logger = logging.getLogger(__name__)


async def mmr_select(query: str, candidates: list[dict], *, k: int, lambda_: float | None = None) -> list[dict]:
    """Select up to ``k`` candidates maximizing relevance−redundancy via MMR.

    Args:
        query: The user query (relevance anchor).
        candidates: Retrieved items, each a dict with a "text" key. Assumed to be
            in descending relevance order (e.g. post-rerank) so ties fall back to
            the existing ranking.
        k: Number of items to keep.
        lambda_: Relevance/diversity trade-off; defaults to ``settings.mmr_lambda``.

    Returns:
        The selected items, in MMR-selection order. Degrades to ``candidates[:k]``
        if there's nothing to diversify or embedding fails.
    """
    lambda_ = settings.mmr_lambda if lambda_ is None else lambda_

    if k <= 0 or len(candidates) <= k:
        return candidates[:k] if k > 0 else []

    texts = [c.get("text", "") for c in candidates]
    try:
        query_vec = await embed_query(query)
        doc_vecs = await embed_texts(texts)
    except Exception as exc:  # never let diversity break retrieval
        logger.warning(f"MMR embedding failed ({exc}); falling back to rank order.")
        return candidates[:k]

    relevance = [cosine_similarity(query_vec, dv) for dv in doc_vecs]

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < k:
        best_idx, best_score = None, float("-inf")
        for i in remaining:
            redundancy = max((cosine_similarity(doc_vecs[i], doc_vecs[s]) for s in selected), default=0.0)
            score = lambda_ * relevance[i] - (1 - lambda_) * redundancy
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(best_idx)
        remaining.discard(best_idx)

    logger.info(f"MMR: {len(candidates)} candidates → {len(selected)} diversified (λ={lambda_}).")
    return [candidates[i] for i in selected]
