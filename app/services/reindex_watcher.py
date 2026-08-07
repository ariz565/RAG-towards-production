"""Automatic reindex on source-document change — closes the gap where
HybridIndex.update() (incremental content-hash diffing, see hybrid_retrieval.py)
was real but only ever invoked manually via `python -m app.cli update-hybrid`.

Periodically checks each loaded document's source PDF against its recorded
lineage content hash (lineage.py) and, if it changed, re-ingests and
incrementally re-indexes it — the exact same call the CLI command makes, just
on a timer instead of a human remembering to run it.

Pure asyncio (no new dependency): a background task started from the FastAPI
lifespan, cancelled cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.services import lineage
from app.services.lineage import sha256_of
from app.services.registry import registry

logger = logging.getLogger(__name__)


async def _check_and_reindex(bundle) -> None:
    """One document: compare its PDF's current hash to the recorded lineage
    hash, and incrementally re-index if they differ. No-ops quietly if the
    PDF or a lineage record can't be found (e.g. a legacy bundle predating
    lineage tracking) — this only heals documents it can confidently identify."""
    from app.services.ingestion import ingest_pdf
    from app.services.storage import get_storage

    if not bundle.stem:
        return

    pdf = get_storage().pdf_path(bundle.tenant_id, bundle.doc_id)
    if not pdf.exists():
        return

    record = lineage.load(bundle.stem)
    if not record:
        return  # nothing to compare against — don't guess

    current_hash = sha256_of(pdf.read_bytes())
    if current_hash == record.get("content_sha256"):
        return  # unchanged

    logger.info(f"Auto-reindex: '{bundle.stem}' content changed on disk — re-indexing...")
    chunks = await ingest_pdf(pdf)
    stats = await bundle.hybrid_index.update(chunks)
    bundle.hybrid_index.save(bundle.stem)
    lineage.update_after_reindex(
        bundle.stem, content_sha256=current_hash, size_bytes=len(pdf.read_bytes()), chunk_count=len(chunks)
    )

    logger.info(f"Auto-reindex: '{bundle.stem}' updated ({stats}).")


async def watch_loop() -> None:
    """Runs until cancelled: re-check every loaded document every
    auto_reindex_interval_seconds. One document's failure never stops the
    loop or blocks the others."""
    while True:
        await asyncio.sleep(settings.auto_reindex_interval_seconds)
        for bundle in list(registry._bundles.values()):
            try:
                await _check_and_reindex(bundle)
            except Exception as e:
                logger.warning(f"Auto-reindex check failed for '{bundle.stem}': {e}")
