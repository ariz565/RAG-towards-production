"""Pluggable document storage — local now, S3-ready later.

`StorageBackend` abstracts WHERE source PDFs live so the rest of the app never
hard-codes the filesystem. `LocalStorage` isolates each tenant in its own
subdirectory (``data/pdf/<tenant>/``). To move to S3, implement the same
interface as `S3Storage` and flip `STORAGE_BACKEND=s3` — no caller changes.

Backward compatibility: for the default tenant, flat files already in
``data/pdf/`` (e.g. the demo doc) are still found via a fallback.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import settings


def tenant_slug(tenant_id: str | None) -> str:
    base = tenant_id or settings.default_tenant
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_") or "default"


class StorageBackend(ABC):
    @abstractmethod
    def save_pdf(self, tenant_id: str, filename: str, data: bytes) -> str:
        """Persist a PDF for a tenant; return its doc_id (filename stem)."""

    @abstractmethod
    def pdf_path(self, tenant_id: str, doc_id: str) -> Path:
        """Local filesystem path to a tenant's PDF (parsers read from disk)."""

    @abstractmethod
    def list_docs(self, tenant_id: str) -> list[str]:
        """List a tenant's doc_ids."""


class LocalStorage(StorageBackend):
    def __init__(self, base: Path | None = None):
        self.base = base or settings.pdf_path

    def _tenant_dir(self, tenant_id: str) -> Path:
        d = self.base / tenant_slug(tenant_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_pdf(self, tenant_id: str, filename: str, data: bytes) -> str:
        name = Path(filename).name
        if not name.lower().endswith(".pdf"):
            raise ValueError("only .pdf files are supported")
        path = self._tenant_dir(tenant_id) / name
        path.write_bytes(data)
        return Path(name).stem

    def pdf_path(self, tenant_id: str, doc_id: str) -> Path:
        sub = self.base / tenant_slug(tenant_id) / f"{doc_id}.pdf"
        if sub.exists():
            return sub
        flat = self.base / f"{doc_id}.pdf"   # legacy/default flat fallback
        return flat if flat.exists() else sub

    def list_docs(self, tenant_id: str) -> list[str]:
        out: list[str] = []
        sub = self.base / tenant_slug(tenant_id)
        if sub.exists():
            out += [p.stem for p in sub.glob("*.pdf")]
        if tenant_slug(tenant_id) == tenant_slug(settings.default_tenant):
            out += [p.stem for p in self.base.glob("*.pdf")]   # legacy flat = default
        return sorted(set(out))


# A future S3 backend would implement StorageBackend by uploading to a bucket and
# downloading to a temp path in pdf_path(); callers stay identical.

_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _storage
    if _storage is None:
        # if settings.storage_backend == "s3": _storage = S3Storage()
        _storage = LocalStorage()
    return _storage
