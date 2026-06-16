"""Reranker — cross-encoder reranking of retrieved candidates (Phase B).

Best practice (2026): cast a wide net at retrieval (BM25 + vector), then rerank
the candidates with a slower, more accurate cross-encoder and keep only the top
few. This is the single biggest precision win after hybrid search.

- Local cross-encoder (sentence-transformers) — no API, no key.
- Lazy-loaded singleton; the model is only downloaded/loaded on first use.
- Degrades gracefully: if disabled or the model can't load, it returns the
  original ordering (trimmed) so the pipeline never hard-fails.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker with lazy model loading and safe fallback."""

    def __init__(self) -> None:
        self._model = None
        self._unavailable = False

    def _ensure_model(self) -> bool:
        """Load the cross-encoder once. Returns False if unavailable."""
        if self._model is not None:
            return True
        if self._unavailable:
            return False
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker: {settings.rerank_model}")
            self._model = CrossEncoder(settings.rerank_model)
            return True
        except Exception as e:  # model missing / offline / import error
            logger.warning(f"Reranker unavailable ({e}); falling back to retrieval order.")
            self._unavailable = True
            return False

    def _rerank_sync(self, query: str, items: list[dict], top_k: int) -> list[dict]:
        """Score (query, item.text) pairs and return the top_k items."""
        candidates = items[: settings.rerank_candidates]
        if not self._ensure_model() or not candidates:
            return candidates[:top_k]

        pairs = [(query, (it.get("text") or "")[: settings.rerank_max_chars]) for it in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Rerank scoring failed ({e}); keeping retrieval order.")
            return candidates[:top_k]

        ranked = sorted(
            zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        out: list[dict] = []
        for item, score in ranked[:top_k]:
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            out.append(enriched)
        return out

    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        top_k: int | None = None,
    ) -> list[dict]:
        """Async wrapper — runs the (CPU-bound) cross-encoder in a thread."""
        top_k = top_k or settings.rerank_top_k
        if not settings.rerank_enabled or len(items) <= 1:
            return items[:top_k]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._rerank_sync, query, items, top_k)

    @property
    def is_active(self) -> bool:
        return settings.rerank_enabled and not self._unavailable


# Singleton
reranker = Reranker()
