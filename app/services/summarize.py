"""Map-reduce summarization for large documents ("summarize this 500-page PDF").

RAG retrieval answers *targeted* questions; summarization needs the *whole*
document, which won't fit a context window. So we map-reduce:

    MAP    : summarize each chunk independently (parallel, bounded concurrency)
    REDUCE : combine the summaries; if still too large, recursively summarize
             groups of summaries (a reduction *tree*) until they fit
    FINAL  : write the document summary from the collapsed summaries

This scales to arbitrarily large docs at O(n) LLM calls in the map step. Chunks
come from the already-built hybrid index (or a fresh ingest), so per-page text is
reused, not re-extracted.
"""

from __future__ import annotations

import asyncio
import logging
import math

from app import prompts
from app.config import settings
from app.services.llm import chat
from app.services.registry import registry
from app.services.storage import get_storage

logger = logging.getLogger(__name__)

# Prompt text lives in the central registry; aliased here for local use.
_MAP_PROMPT = prompts.SUMMARIZE_MAP
_REDUCE_PROMPT = prompts.SUMMARIZE_REDUCE
_FINAL_PROMPT = prompts.SUMMARIZE_FINAL


async def _chunk_texts(tenant_id: str, doc_id: str) -> list[str]:
    """Reuse the doc's indexed chunks; fall back to a fresh ingest."""
    bundle = registry.resolve(tenant_id, doc_id)
    if bundle and bundle.hybrid_index.is_loaded and bundle.hybrid_index._chunks:
        return [c["text"] for c in bundle.hybrid_index._chunks]
    pdf = get_storage().pdf_path(tenant_id, doc_id)
    if not pdf.exists():
        return []
    from app.services.ingestion import ingest_pdf
    chunks = await ingest_pdf(pdf, save_chunks=False)
    return [c.text for c in chunks]


def _merge_to(texts: list[str], n: int) -> list[str]:
    """Merge adjacent chunks into ~n groups to bound the number of map calls."""
    if len(texts) <= n:
        return texts
    size = math.ceil(len(texts) / n)
    return ["\n".join(texts[i:i + size]) for i in range(0, len(texts), size)]


async def summarize_document(tenant_id: str, doc_id: str, *, style: str = "concise") -> dict:
    texts = await _chunk_texts(tenant_id, doc_id)
    if not texts:
        return {"summary": "", "chunks": 0, "reduce_levels": 0}

    sem = asyncio.Semaphore(settings.summarize_max_concurrency)

    async def _summarize(prompt: str) -> str:
        async with sem:
            resp = await chat(prompt, temperature=0)
            return (resp.content or "").strip()

    # Fast path: a small doc fits one call — don't map-reduce a 3-page PDF.
    combined_all = "\n\n".join(texts)
    if len(combined_all) <= settings.summarize_reduce_char_budget:
        final = await _summarize(
            _FINAL_PROMPT.format(style=style, text=combined_all[: settings.summarize_reduce_char_budget])
        )
        return {"summary": final, "chunks": len(texts), "reduce_levels": 0}

    # Bound map cost on very large docs by merging adjacent chunks first.
    map_inputs = _merge_to(texts, settings.summarize_max_map_calls)

    # ── MAP: summarize each (merged) chunk ───────────────────────
    summaries = await asyncio.gather(
        *[_summarize(_MAP_PROMPT.format(text=t[: settings.summarize_map_chars])) for t in map_inputs]
    )
    summaries = [s for s in summaries if s]

    # ── REDUCE: collapse the reduction tree until it fits ────────
    levels = 0
    group = settings.summarize_reduce_group_size
    while len("\n".join(summaries)) > settings.summarize_reduce_char_budget and len(summaries) > 1:
        batches = ["\n".join(f"- {s}" for s in summaries[i:i + group]) for i in range(0, len(summaries), group)]
        summaries = await asyncio.gather(*[_summarize(_REDUCE_PROMPT.format(text=b)) for b in batches])
        summaries = [s for s in summaries if s]
        levels += 1

    # ── FINAL ────────────────────────────────────────────────────
    combined = "\n".join(f"- {s}" for s in summaries)
    final = await _summarize(_FINAL_PROMPT.format(style=style, text=combined))
    return {"summary": final, "chunks": len(texts), "reduce_levels": levels}
