"""Document lineage / source certification — provenance for every indexed document.

2026 RAG governance extends into the knowledge base: every retrievable document
should carry a provenance record so you can answer *"where did this answer come
from, and is the source trusted?"*. Per document we persist:

- a content fingerprint (SHA-256 of the original PDF bytes) — tamper-evidence;
- who uploaded it (hashed) and when;
- the embedding model + dimension it was indexed with (lineage of the vectors);
- which indexes exist (PageIndex tree / hybrid) and the chunk count;
- a ``certified`` flag a reviewer can set once the source is vetted.

Records live at ``data/index/<stem>_lineage.json`` next to the index artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.services.governance import APP_VERSION, _embedding_model_name

logger = logging.getLogger(__name__)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _user_hash(email: str | None) -> str:
    if not email:
        return "anonymous"
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:16]


def _lineage_path(stem: str) -> Path:
    return settings.index_path / f"{stem}_lineage.json"


@dataclass
class LineageRecord:
    """Provenance + certification metadata for one indexed document."""

    tenant_id: str
    doc_id: str
    source_filename: str
    content_sha256: str
    size_bytes: int
    uploaded_by: str            # hashed
    uploaded_at: str            # ISO-8601 UTC
    chunk_count: int = 0
    embedding_model: str = ""
    embedding_dim: int = 0
    has_pageindex: bool = False
    has_hybrid: bool = False
    certified: bool = False
    app_version: str = APP_VERSION
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def record_upload(
    *,
    tenant_id: str,
    doc_id: str,
    stem: str,
    source_filename: str,
    pdf_bytes: bytes,
    uploaded_by: str | None,
    chunk_count: int = 0,
    has_pageindex: bool = False,
    has_hybrid: bool = False,
) -> LineageRecord | None:
    """Create + persist a lineage record for a freshly indexed document."""
    if not settings.lineage_enabled:
        return None
    record = LineageRecord(
        tenant_id=tenant_id,
        doc_id=doc_id,
        source_filename=source_filename,
        content_sha256=sha256_of(pdf_bytes),
        size_bytes=len(pdf_bytes),
        uploaded_by=_user_hash(uploaded_by),
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=chunk_count,
        embedding_model=_embedding_model_name(),
        embedding_dim=settings.embedding_dim,
        has_pageindex=has_pageindex,
        has_hybrid=has_hybrid,
    )
    save(record, stem)
    return record


def save(record: LineageRecord, stem: str) -> None:
    try:
        path = _lineage_path(stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Lineage recorded: {stem} (sha256={record.content_sha256[:12]}…)")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Lineage write failed for {stem}: {e}")


def load(stem: str) -> dict | None:
    path = _lineage_path(stem)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Lineage read failed for {stem}: {e}")
        return None


def certify(stem: str, certified: bool = True) -> dict | None:
    """Mark a document's source as reviewed/certified (or revoke)."""
    data = load(stem)
    if data is None:
        return None
    data["certified"] = certified
    try:
        with open(_lineage_path(stem), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Lineage certify failed for {stem}: {e}")
    return data
