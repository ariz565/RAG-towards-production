"""Corpus Registry — per-tenant, per-document index bundles (Phase C).

Replaces the old module-level singletons (one document in global memory) with a
registry of ``DocumentBundle``s keyed by ``tenant_id/doc_id``. Each bundle owns
its own PageIndex, hybrid index (per-document Qdrant collection), and domain
profile.

The active document for a request flows through a **contextvar** (task-isolated,
like the provider override) so the pipeline never stores non-serializable index
objects in LangGraph state — which keeps the checkpointer happy.
"""

from __future__ import annotations

import contextvars
import logging
import re
from dataclasses import dataclass

from app.config import RetrievalStrategy, settings
from app.services.domain_profile import DomainProfile, default_profile, ensure_profile
from app.services.hybrid_retrieval import HybridIndex
from app.services.indexer import DocumentIndex

logger = logging.getLogger(__name__)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_") or "doc"


def collection_for(stem: str) -> str:
    """Per-document Qdrant collection name (from the tenant-namespaced stem)."""
    return f"{settings.qdrant_collection}_{_slug(stem)}"


def namespace(tenant_id: str, doc_id: str) -> str:
    """Filesystem/collection-safe artifact key: '<tenant>__<doc>'."""
    from app.services.storage import tenant_slug
    return f"{tenant_slug(tenant_id)}__{doc_id}"


# ═══════════════════════════════════════════════════════════════════════
# BUNDLE
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DocumentBundle:
    """Everything needed to answer questions about one document."""

    tenant_id: str
    doc_id: str
    document_index: DocumentIndex
    hybrid_index: HybridIndex
    profile: DomainProfile
    indexed_at: str = ""        # data freshness (ISO-8601)

    @property
    def key(self) -> str:
        return f"{self.tenant_id}/{self.doc_id}"

    @property
    def is_ready(self) -> bool:
        return self.document_index.is_loaded or self.hybrid_index.is_loaded


# ═══════════════════════════════════════════════════════════════════════
# ACTIVE-DOCUMENT CONTEXTVAR
# ═══════════════════════════════════════════════════════════════════════

_active_bundle: contextvars.ContextVar[DocumentBundle | None] = contextvars.ContextVar(
    "vision_active_bundle", default=None
)


def use_document(bundle: DocumentBundle | None):
    """Set the active document bundle for the current task. Returns a reset token."""
    return _active_bundle.set(bundle)


def clear_document(token) -> None:
    _active_bundle.reset(token)


def current_bundle() -> DocumentBundle | None:
    return _active_bundle.get()


# ═══════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════


class CorpusRegistry:
    """Holds loaded DocumentBundles keyed by tenant_id/doc_id."""

    def __init__(self) -> None:
        self._bundles: dict[str, DocumentBundle] = {}

    @staticmethod
    def _key(tenant_id: str, doc_id: str) -> str:
        return f"{tenant_id}/{doc_id}"

    def register(self, bundle: DocumentBundle) -> None:
        self._bundles[bundle.key] = bundle

    def get(self, tenant_id: str, doc_id: str) -> DocumentBundle | None:
        return self._bundles.get(self._key(tenant_id, doc_id))

    def bundles_for(self, tenant_id: str | None) -> list[DocumentBundle]:
        tenant_id = tenant_id or settings.default_tenant
        return [b for b in self._bundles.values() if b.tenant_id == tenant_id]

    def resolve(self, tenant_id: str | None, doc_id: str | None) -> DocumentBundle | None:
        """Resolve a bundle, falling back to the single/first loaded document."""
        tenant_id = tenant_id or settings.default_tenant
        if doc_id:
            return self.get(tenant_id, doc_id)
        # No doc specified — prefer a doc in this tenant, else any loaded doc.
        for bundle in self._bundles.values():
            if bundle.tenant_id == tenant_id:
                return bundle
        return next(iter(self._bundles.values()), None)

    def list(self) -> list[dict]:
        return [
            {
                "tenant_id": b.tenant_id,
                "doc_id": b.doc_id,
                "doc_name": b.document_index.doc_name or b.doc_id,
                "total_pages": b.document_index.total_pages,
                "total_chunks": len(b.hybrid_index._chunks) if b.hybrid_index else 0,
                "preview": (b.hybrid_index._chunks[0]["text"][:150] + "...") if b.hybrid_index and b.hybrid_index._chunks else "",
                "pageindex": b.document_index.is_loaded,
                "hybrid": b.hybrid_index.is_loaded,
                "topics": b.profile.topics,
            }
            for b in self._bundles.values()
        ]

    # ── Loading ─────────────────────────────────────────────────────

    async def load_bundle(
        self, tenant_id: str, doc_id: str, stem: str | None = None
    ) -> DocumentBundle | None:
        """Build a bundle for one document from on-disk, tenant-namespaced indexes.

        `stem` is the artifact namespace (``<tenant>__<doc>`` for uploads, or a
        bare doc id for legacy/default files); callers normally let it default.
        """
        from app.services.storage import get_storage

        tenant_id = tenant_id or settings.default_tenant
        stem = stem or namespace(tenant_id, doc_id)

        # PageIndex (tenant-namespaced index file; PDF resolved via storage)
        document_index = DocumentIndex()
        index_file = settings.index_path / f"{stem}_index.json"
        pdf = get_storage().pdf_path(tenant_id, doc_id)
        if index_file.exists() and pdf.exists():
            try:
                document_index.load_index(index_file, pdf)
            except Exception as e:
                logger.warning(f"PageIndex load failed for {stem}: {e}")

        # Hybrid index (per-document, tenant-namespaced collection)
        hybrid_index = HybridIndex(collection_name=collection_for(stem))
        if hybrid_index.load(stem):
            if settings.active_retrieval in (RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_ONLY):
                try:
                    await hybrid_index._build_vectors_from_dicts()
                except Exception as e:
                    logger.warning(f"Vector restore failed for {stem}: {e}")

        # Domain profile (cached per namespace)
        sample = document_index.sample_text() if document_index.is_loaded else ""
        doc_name = document_index.doc_name or doc_id
        try:
            profile = await ensure_profile(doc_name, stem, sample)
        except Exception as e:
            logger.warning(f"Profile load failed for {stem}: {e}")
            profile = default_profile(doc_name)

        # Data freshness: prefer the index's recorded build time, else file mtime.
        indexed_at = document_index.indexed_at
        if not indexed_at:
            import datetime

            for candidate in (index_file, settings.bm25_path / f"{stem}_hybrid_chunks.json"):
                if candidate.exists():
                    indexed_at = datetime.datetime.fromtimestamp(
                        candidate.stat().st_mtime, datetime.timezone.utc
                    ).isoformat()
                    break

        bundle = DocumentBundle(tenant_id, doc_id, document_index, hybrid_index, profile, indexed_at)
        if not bundle.is_ready:
            logger.info(f"No indexes found for '{stem}' — skipping.")
            return None
        self.register(bundle)
        logger.info(f"Registered bundle: {bundle.key} (stem={stem}, "
                    f"pages={document_index.total_pages}, hybrid={hybrid_index.is_loaded})")
        return bundle

    async def load_all(self, tenant_id: str | None = None) -> int:
        """Discover every document on disk and load it under its (parsed) tenant."""
        stems: set[str] = set()
        for f in settings.index_path.glob("*_index.json"):
            stems.add(f.name[: -len("_index.json")])
        for f in settings.bm25_path.glob("*_hybrid_chunks.json"):
            stems.add(f.name[: -len("_hybrid_chunks.json")])

        loaded = 0
        for stem in sorted(stems):
            # "<tenant>__<doc>" → tenant-scoped; bare "<doc>" → default tenant (legacy).
            if "__" in stem:
                tenant, doc = stem.split("__", 1)
            else:
                tenant, doc = settings.default_tenant, stem
            try:
                if await self.load_bundle(tenant, doc, stem=stem):
                    loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load bundle '{stem}': {e}")
        logger.info(f"Registry: {loaded} document(s) loaded across tenants.")
        return loaded


# Singleton registry
registry = CorpusRegistry()
