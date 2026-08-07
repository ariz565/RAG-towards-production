"""Semantic answer cache — reuse a past answer when a new query is close
enough in *meaning* (not just exact text) to one already answered for the
same tenant+document.

Complements embed_query()'s exact-match cache (embeddings.py): that one avoids
re-embedding an identical string; this one avoids re-running retrieval and
generation entirely for a *reworded* repeat question. Built on Qdrant — already
a dependency, already used per-document (hybrid_retrieval.py) — as one extra
collection filtered by tenant_id+doc_id, rather than adding a new vector store.

Invalidation: each entry is tagged with the document's lineage content hash
(lineage.py) at write time. If the document has since changed, the hash no
longer matches and the entry is treated as a miss — reuses the fingerprint
the app already tracks instead of a separate invalidation mechanism.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from app.config import settings
from app.services.hybrid_retrieval import get_qdrant_client

logger = logging.getLogger(__name__)

_COLLECTION = "semantic_cache"


def _point_id(tenant_id: str, doc_id: str, query: str) -> int:
    """Deterministic uint64 id so re-caching the same (tenant, doc, query) upserts in place."""
    return int(hashlib.sha256(f"{tenant_id}/{doc_id}/{query}".encode()).hexdigest()[:15], 16)


def _current_content_hash(doc_id: str) -> str | None:
    from app.services import lineage

    record = lineage.load(doc_id)
    return record.get("content_sha256") if record else None


async def get(tenant_id: str, doc_id: str, query: str) -> dict | None:
    """Return a cached answer dict if a semantically-close question was
    already answered for this tenant+document, else None."""
    if not settings.semantic_cache_enabled:
        return None

    client = get_qdrant_client()
    if not client.collection_exists(_COLLECTION):
        return None

    from qdrant_client import models

    from app.services.embeddings import embed_query

    vector = await embed_query(query)
    hits = client.query_points(
        collection_name=_COLLECTION,
        query=vector,
        query_filter=models.Filter(must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
        ]),
        limit=1,
        score_threshold=settings.semantic_cache_threshold,
    ).points
    if not hits:
        return None

    payload = hits[0].payload
    current_hash = _current_content_hash(doc_id)
    if current_hash and payload.get("content_hash") != current_hash:
        logger.debug(f"Semantic cache hit for doc={doc_id!r} invalidated — document changed since caching.")
        return None

    logger.info(f"Semantic cache hit (score={hits[0].score:.3f}) tenant={tenant_id} doc={doc_id}")
    return json.loads(payload["answer_json"])


async def put(tenant_id: str, doc_id: str, query: str, answer: dict) -> None:
    """Cache a grounded answer for later semantic reuse."""
    if not settings.semantic_cache_enabled:
        return

    from qdrant_client import models

    from app.services.embeddings import embed_query

    vector = await embed_query(query)
    client = get_qdrant_client()
    if not client.collection_exists(_COLLECTION):
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=models.VectorParams(size=len(vector), distance=models.Distance.COSINE),
        )

    client.upsert(collection_name=_COLLECTION, points=[
        models.PointStruct(
            id=_point_id(tenant_id, doc_id, query),
            vector=vector,
            payload={
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "query": query,
                "answer_json": json.dumps(answer),
                "content_hash": _current_content_hash(doc_id) or "",
                "cached_at": time.time(),
            },
        )
    ])
