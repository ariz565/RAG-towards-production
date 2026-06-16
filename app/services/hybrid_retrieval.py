"""Hybrid Retrieval Service — BM25 + Vector with RRF fusion.

Supports four modes (controlled by RetrievalStrategy):
- HYBRID: BM25 + Vector → RRF fusion (best quality)
- BM25_ONLY: Sparse retrieval only (fast, no embeddings needed)
- VECTOR_ONLY: Dense retrieval only (semantic understanding)
- PAGEINDEX: Delegated to tree_search.py (not handled here)

Storage:
- Qdrant for vector storage (in-memory or Docker)
- Pickle for BM25 index (disk-persisted)

Usage (per-document instances are owned by the corpus registry):
    from app.services.hybrid_retrieval import HybridIndex

    index = HybridIndex(collection_name="vision_mydoc")
    await index.build(chunks)
    results = await index.search("What is stored in Cosmos DB?")
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from app.config import RetrievalStrategy, settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SHARED QDRANT CLIENT (Phase C)
# One client process-wide. Local path mode locks the storage dir, so every
# HybridIndex must share a single client and use a per-document collection.
# ═══════════════════════════════════════════════════════════════════════

_shared_qdrant_client = None


def get_qdrant_client():
    """Return a process-wide Qdrant client (server | persistent path | in-memory)."""
    global _shared_qdrant_client
    if _shared_qdrant_client is not None:
        return _shared_qdrant_client

    from qdrant_client import QdrantClient

    if settings.qdrant_use_server:
        _shared_qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        logger.info(f"Qdrant: server {settings.qdrant_host}:{settings.qdrant_port}")
    elif settings.qdrant_in_memory:
        _shared_qdrant_client = QdrantClient(":memory:")
        logger.info("Qdrant: in-memory (ephemeral)")
    else:
        Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
        _shared_qdrant_client = QdrantClient(path=settings.qdrant_path)
        logger.info(f"Qdrant: persistent at {settings.qdrant_path}")
    return _shared_qdrant_client


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""
    chunk_id: str
    text: str
    score: float
    page_numbers: list[int]
    section_title: str = ""
    source: str = ""  # "bm25", "vector", or "hybrid"
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# HYBRID INDEX
# ═══════════════════════════════════════════════════════════════════════


class HybridIndex:
    """Combined BM25 + Qdrant vector index.

    Supports building, saving, loading, and searching with RRF fusion.
    """

    def __init__(self, collection_name: str | None = None):
        self._bm25 = None           # BM25Okapi instance
        self._bm25_corpus = []       # Tokenized corpus for BM25
        self._chunks: list[dict] = []  # chunk_id → chunk data
        self._qdrant_client = None
        self._collection = collection_name or settings.qdrant_collection
        self._collection_ready = False
        self._is_loaded = False

    @staticmethod
    def _record(chunk) -> dict:
        """Normalize a Chunk into the stored dict shape (shared by build + update)."""
        return {
            "id": chunk.id,
            "text": chunk.text,
            "embed_text": getattr(chunk, "embed_text", chunk.text),
            "context": getattr(chunk, "context", ""),
            "page_numbers": chunk.page_numbers,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "content_hash": getattr(chunk, "content_hash", ""),
            "metadata": chunk.metadata,
        }

    def _collection_has_points(self) -> bool:
        """True if this collection exists, is populated, AND matches the active dim.

        The dimension guard forces a rebuild if the persisted collection was built
        with a different embedding model (e.g. switching HF 384-d ↔ OpenAI 1536-d).
        """
        client = self._qdrant_client or get_qdrant_client()
        try:
            if not client.collection_exists(self._collection):
                return False
            if client.count(self._collection).count <= 0:
                return False
            info = client.get_collection(self._collection)
            size = info.config.params.vectors.size
            if size != settings.embedding_dim:
                logger.warning(
                    f"Collection '{self._collection}' dim {size} != active embedding dim "
                    f"{settings.embedding_dim}; will rebuild."
                )
                return False
            return True
        except Exception:
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ── BUILD ───────────────────────────────────────────────────────

    async def build(
        self,
        chunks: list,  # List of ingestion.Chunk objects
        *,
        strategy: RetrievalStrategy | None = None,
    ) -> dict:
        """Build the hybrid index from chunks.

        Steps:
        1. Build BM25 index from chunk texts
        2. Embed all chunks
        3. Upsert to Qdrant

        Args:
            chunks: List of Chunk objects from ingestion.
            strategy: Override active strategy.

        Returns:
            Stats dict with counts.
        """
        strategy = strategy or settings.active_retrieval

        # Store chunk data for retrieval
        self._chunks = [self._record(chunk) for chunk in chunks]

        stats = {"total_chunks": len(chunks)}

        # Build BM25 index (needed for hybrid and bm25_only)
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.BM25_ONLY):
            self._build_bm25()
            stats["bm25_indexed"] = True

        # Build vector index (needed for hybrid and vector_only)
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_ONLY):
            await self._build_vectors(chunks)
            stats["vector_indexed"] = True

        self._is_loaded = True
        logger.info(f"Hybrid index built: {stats}")
        return stats

    def _build_bm25(self) -> None:
        """Build BM25 index from stored chunks."""
        from rank_bm25 import BM25Okapi

        logger.info(f"Building BM25 index over {len(self._chunks)} chunks...")

        # Tokenize: lowercase + split on whitespace/punctuation.
        # Uses embed_text (context + text) so Contextual Retrieval also helps BM25.
        self._bm25_corpus = []
        for chunk in self._chunks:
            tokens = _tokenize(chunk.get("embed_text") or chunk["text"])
            self._bm25_corpus.append(tokens)

        self._bm25 = BM25Okapi(self._bm25_corpus)
        logger.info("BM25 index ready")

    def _recreate_collection(self, dim: int) -> None:
        """Drop and recreate this index's collection on the shared client."""
        from qdrant_client import models

        client = get_qdrant_client()
        self._qdrant_client = client
        try:
            if client.collection_exists(self._collection):
                client.delete_collection(self._collection)
        except Exception:
            pass
        client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def _upsert_chunks(self, vectors: list) -> None:
        """Upsert self._chunks + their vectors into this index's collection."""
        from qdrant_client import models

        client = self._qdrant_client or get_qdrant_client()
        batch_size = 100
        points = []
        for chunk, vector in zip(self._chunks, vectors):
            points.append(models.PointStruct(
                id=_point_id(chunk["id"]),   # stable id → enables incremental upsert/delete
                vector=vector,
                payload={
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "page_numbers": chunk["page_numbers"],
                    "section_title": chunk["section_title"],
                    "chunk_index": chunk["chunk_index"],
                },
            ))
            if len(points) >= batch_size:
                client.upsert(collection_name=self._collection, points=points)
                points = []
        if points:
            client.upsert(collection_name=self._collection, points=points)
        self._collection_ready = True

    async def _build_vectors(self, chunks: list) -> None:
        """Embed chunks (embed_text) and upsert to this index's collection."""
        from app.services.embeddings import embed_texts

        logger.info(f"Embedding {len(chunks)} chunks → '{self._collection}'...")
        texts = [getattr(c, "embed_text", c.text) for c in chunks]
        vectors = await embed_texts(texts)
        dim = len(vectors[0]) if vectors else settings.embedding_dim
        self._recreate_collection(dim)
        self._upsert_chunks(vectors)
        logger.info(f"Qdrant: {len(vectors)} vectors upserted to '{self._collection}'")

    # ── SEARCH ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search using the specified retrieval strategy.

        Args:
            query: The search query.
            strategy: Override active strategy.
            top_k: Number of results to return.

        Returns:
            Sorted list of RetrievalResult objects.
        """
        strategy = strategy or settings.active_retrieval
        top_k = top_k or settings.retrieval_top_k

        if strategy == RetrievalStrategy.BM25_ONLY:
            return self._search_bm25(query, top_k=settings.bm25_top_k)[:top_k]
        elif strategy == RetrievalStrategy.VECTOR_ONLY:
            return await self._search_vector(query, top_k=settings.vector_top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            return await self._search_hybrid(query, top_k=top_k)
        else:
            raise ValueError(f"Strategy {strategy} not handled by HybridIndex")

    def _search_bm25(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """BM25 sparse retrieval."""
        if not self._bm25:
            logger.warning("BM25 index not built")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self._chunks[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    score=float(scores[idx]),
                    page_numbers=chunk["page_numbers"],
                    section_title=chunk["section_title"],
                    source="bm25",
                ))

        return results

    async def _search_vector(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Vector (dense) retrieval via Qdrant."""
        if not self._qdrant_client or not self._collection_ready:
            logger.warning("Qdrant not ready")
            return []

        from app.services.embeddings import embed_query

        query_vector = await embed_query(query)

        hits = self._qdrant_client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
        ).points

        results = []
        for hit in hits:
            payload = hit.payload
            results.append(RetrievalResult(
                chunk_id=payload.get("chunk_id", ""),
                text=payload.get("text", ""),
                score=hit.score,
                page_numbers=payload.get("page_numbers", []),
                section_title=payload.get("section_title", ""),
                source="vector",
            ))

        return results

    async def _search_hybrid(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Hybrid search: BM25 + Vector → RRF fusion.

        RRF formula: score = 1/(rank_i + k) for each system, then sum.
        """
        # Get candidates from both systems. Each system must surface at least
        # `top_k` so the fused pool can actually reach the requested size even
        # when the two systems overlap heavily.
        bm25_results = self._search_bm25(query, top_k=max(settings.bm25_top_k, top_k))
        vector_results = await self._search_vector(query, top_k=max(settings.vector_top_k, top_k))

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, RetrievalResult] = {}
        k = settings.rrf_k

        for rank, result in enumerate(bm25_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (rank + k)
            chunk_data[result.chunk_id] = result

        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (rank + k)
            if result.chunk_id not in chunk_data:
                chunk_data[result.chunk_id] = result

        # Sort by fused score
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            result = chunk_data[cid]
            result.score = rrf_scores[cid]
            result.source = "hybrid"
            results.append(result)

        logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"→ {len(results)} fused results"
        )
        return results

    # ── PERSISTENCE ─────────────────────────────────────────────────

    def save(self, pdf_stem: str) -> None:
        """Save BM25 index and chunk data to disk."""
        bm25_dir = settings.bm25_path
        bm25_dir.mkdir(parents=True, exist_ok=True)

        # Save BM25
        if self._bm25:
            bm25_file = bm25_dir / f"{pdf_stem}_bm25.pkl"
            with open(bm25_file, "wb") as f:
                pickle.dump({
                    "bm25": self._bm25,
                    "corpus": self._bm25_corpus,
                }, f)
            logger.info(f"BM25 index saved: {bm25_file}")

        # Save chunks
        chunks_file = bm25_dir / f"{pdf_stem}_hybrid_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Chunk data saved: {chunks_file}")

    def load(self, pdf_stem: str) -> bool:
        """Load BM25 index and chunk data from disk.

        Returns True if loaded successfully.
        """
        bm25_dir = settings.bm25_path

        # Load chunks
        chunks_file = bm25_dir / f"{pdf_stem}_hybrid_chunks.json"
        if not chunks_file.exists():
            return False

        with open(chunks_file, encoding="utf-8") as f:
            self._chunks = json.load(f)

        # Load BM25
        bm25_file = bm25_dir / f"{pdf_stem}_bm25.pkl"
        if bm25_file.exists():
            with open(bm25_file, "rb") as f:
                data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._bm25_corpus = data["corpus"]

        self._is_loaded = True
        logger.info(f"Hybrid index loaded: {len(self._chunks)} chunks, BM25={self._bm25 is not None}")
        return True

    async def load_and_build_vectors(self, pdf_stem: str) -> bool:
        """Load from disk and rebuild vector index (Qdrant is ephemeral in-memory)."""
        if not self.load(pdf_stem):
            return False

        # Rebuild vectors if needed
        strategy = settings.active_retrieval
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_ONLY):
            from app.services.ingestion import Chunk
            # Convert dicts back to Chunk-like objects for embedding
            chunks = []
            for c in self._chunks:
                chunks.append(type("ChunkLike", (), {"text": c["text"], "id": c["id"]})())

            await self._build_vectors_from_dicts()

        return True

    async def _build_vectors_from_dicts(self) -> None:
        """Build/restore Qdrant vectors from stored chunk dicts.

        With persistent storage, an already-populated collection is reused as-is
        (no re-embedding on every boot — the Phase C fix).
        """
        from app.services.embeddings import embed_texts

        if not self._chunks:
            return

        self._qdrant_client = get_qdrant_client()

        # Persisted collection already has our vectors → skip the expensive rebuild.
        if self._collection_has_points():
            self._collection_ready = True
            logger.info(
                f"Qdrant: reusing persisted collection '{self._collection}' "
                f"({len(self._chunks)} chunks, no re-embed)"
            )
            return

        texts = [(c.get("embed_text") or c["text"]) for c in self._chunks]
        logger.info(f"Re-embedding {len(texts)} chunks → '{self._collection}'...")
        vectors = await embed_texts(texts)
        dim = len(vectors[0])
        self._recreate_collection(dim)
        self._upsert_chunks(vectors)
        logger.info(f"Qdrant rebuilt: {len(vectors)} vectors in '{self._collection}'")

    async def update(self, new_chunks: list) -> dict:
        """Incrementally re-index: embed only added/changed chunks, delete removed.

        Diffs `new_chunks` against the loaded corpus by id + content_hash, so a doc
        edit re-embeds a handful of chunks instead of the whole document. BM25 is
        corpus-global, so it is cheaply rebuilt. Requires the prior index loaded
        (else falls back to a full build).
        """
        if not self._chunks:
            await self.build(new_chunks)
            return {"added": len(new_chunks), "changed": 0, "removed": 0, "full_rebuild": True}

        old_by_id = {c["id"]: c for c in self._chunks}
        new_by_id = {c.id: c for c in new_chunks}
        added = [c for cid, c in new_by_id.items() if cid not in old_by_id]
        changed = [c for cid, c in new_by_id.items()
                   if cid in old_by_id and old_by_id[cid].get("content_hash") != c.content_hash]
        removed_ids = [cid for cid in old_by_id if cid not in new_by_id]
        to_embed = added + changed

        # Replace the stored corpus + rebuild BM25 (global statistics).
        self._chunks = [self._record(c) for c in new_chunks]
        self._build_bm25()

        # Vector delta (only when vectors are in use).
        if settings.active_retrieval in (RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_ONLY):
            from qdrant_client import models

            from app.services.embeddings import embed_texts

            client = get_qdrant_client()
            self._qdrant_client = client

            vectors = await embed_texts([getattr(c, "embed_text", c.text) for c in to_embed]) if to_embed else []
            if to_embed and not client.collection_exists(self._collection):
                self._recreate_collection(len(vectors[0]))

            stale = [_point_id(cid) for cid in removed_ids] + [_point_id(c.id) for c in changed]
            if stale and client.collection_exists(self._collection):
                client.delete(collection_name=self._collection,
                              points_selector=models.PointIdsList(points=stale))

            if to_embed:
                points = [
                    models.PointStruct(
                        id=_point_id(c.id), vector=v,
                        payload={"chunk_id": c.id, "text": c.text,
                                 "page_numbers": c.page_numbers,
                                 "section_title": c.section_title,
                                 "chunk_index": c.chunk_index},
                    )
                    for c, v in zip(to_embed, vectors)
                ]
                for i in range(0, len(points), 100):
                    client.upsert(collection_name=self._collection, points=points[i:i + 100])
            self._collection_ready = True

        stats = {"added": len(added), "changed": len(changed), "removed": len(removed_ids)}
        logger.info(f"Incremental update for '{self._collection}': {stats}")
        return stats

    def get_chunks_for_pages(self, page_numbers: list[int]) -> list[dict]:
        """Get all chunks that come from the specified pages."""
        return [
            c for c in self._chunks
            if any(p in c["page_numbers"] for p in page_numbers)
        ]


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25: lowercase, split, remove short words."""
    import re
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) > 1]


def _point_id(chunk_id: str) -> int:
    """Deterministic uint64 Qdrant point id from a chunk id (stable across re-index)."""
    import hashlib
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:15], 16)
