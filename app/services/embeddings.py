"""Embedding Service — local-first, provider-switchable.

Supports:
- HuggingFace Sentence Transformers (free, local, default)
- OpenAI text-embedding-3-small (cloud)

Usage:
    from app.services.embeddings import embed_texts, embed_query

    vectors = await embed_texts(["chunk 1", "chunk 2"])
    query_vec = await embed_query("What is the GPA requirement?")
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

import numpy as np

from app.config import EmbeddingProvider, settings
from app.services.retry import RetryPolicy, retry_async

logger = logging.getLogger(__name__)

# Small LRU for query embeddings — the same query is embedded several times per
# request (router scoring, vector search, grounding retries). Keyed by
# (provider, model, text) so a provider/model switch never returns a stale vector.
_query_cache: "OrderedDict[tuple[str, str, str], list[float]]" = OrderedDict()


def _provider_model(provider: EmbeddingProvider) -> str:
    """Resolve the active model name for a provider (part of the cache key)."""
    if provider == EmbeddingProvider.HUGGINGFACE:
        return settings.hf_embedding_model
    if provider == EmbeddingProvider.OLLAMA:
        return settings.ollama_embedding_model
    return settings.openai_embedding_model


def _embed_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_retry_base_delay,
        max_delay=settings.llm_retry_max_delay,
        jitter=settings.llm_retry_jitter,
        respect_retry_after=settings.llm_retry_respect_retry_after,
    )


# ═══════════════════════════════════════════════════════════════════════
# HUGGINGFACE (local / free)
# ═══════════════════════════════════════════════════════════════════════

_hf_model = None


def _get_hf_model():
    """Lazy-load the HuggingFace model (heavy import)."""
    global _hf_model
    if _hf_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading HuggingFace embedding model: {settings.hf_embedding_model}")
        _hf_model = SentenceTransformer(settings.hf_embedding_model)
        logger.info(f"Embedding model loaded (dim={_hf_model.get_sentence_embedding_dimension()})")
    return _hf_model


def _hf_embed_sync(texts: list[str]) -> list[list[float]]:
    """Embed texts using HuggingFace Sentence Transformers (CPU-bound, sync)."""
    model = _get_hf_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


async def _hf_embed(texts: list[str]) -> list[list[float]]:
    """Run the CPU-bound HF encode off the event loop so it never blocks serving."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _hf_embed_sync, texts)


# ═══════════════════════════════════════════════════════════════════════
# OPENAI (cloud)
# ═══════════════════════════════════════════════════════════════════════


async def _openai_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ═══════════════════════════════════════════════════════════════════════
# OLLAMA (local)
# ═══════════════════════════════════════════════════════════════════════


async def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using Ollama local models."""
    import ollama

    client = ollama.AsyncClient(host=settings.ollama_base_url)
    embeddings = []
    # Ollama embeds one at a time (batch by sending list)
    for text in texts:
        response = await client.embed(
            model=settings.ollama_embedding_model,
            input=text,
        )
        embeddings.append(response["embeddings"][0])
    return embeddings


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════


async def embed_texts(
    texts: list[str],
    *,
    provider: EmbeddingProvider | None = None,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed a list of texts using the active embedding provider.

    Args:
        texts: List of text strings to embed.
        provider: Override the default provider.
        batch_size: Process texts in batches (for HuggingFace).

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    provider = provider or settings.active_embedding

    if not texts:
        return []

    all_embeddings = []

    if provider == EmbeddingProvider.HUGGINGFACE:
        # CPU-bound encode runs in a thread (does not block the event loop).
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(await _hf_embed(batch))
            if i > 0:
                logger.debug(f"Embedded {i + len(batch)}/{len(texts)} chunks")
    elif provider == EmbeddingProvider.OPENAI:
        policy = _embed_retry_policy()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embedded = await retry_async(
                lambda b=batch: _openai_embed(b), policy=policy, label="embed[openai]"
            )
            all_embeddings.extend(embedded)
    elif provider == EmbeddingProvider.OLLAMA:
        policy = _embed_retry_policy()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embedded = await retry_async(
                lambda b=batch: _ollama_embed(b), policy=policy, label="embed[ollama]"
            )
            all_embeddings.extend(embedded)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    logger.info(f"Embedded {len(texts)} texts → {len(all_embeddings)} vectors [{provider.value}]")
    return all_embeddings


async def embed_query(
    query: str,
    *,
    provider: EmbeddingProvider | None = None,
) -> list[float]:
    """Embed a single query string.

    Args:
        query: The search query to embed.
        provider: Override the default provider.

    Returns:
        Single embedding vector.
    """
    provider = provider or settings.active_embedding

    if not settings.embedding_cache_enabled:
        return (await embed_texts([query], provider=provider))[0]

    key = (provider.value, _provider_model(provider), query)
    cached = _query_cache.get(key)
    if cached is not None:
        _query_cache.move_to_end(key)   # mark as recently used
        return list(cached)

    vector = (await embed_texts([query], provider=provider))[0]
    _query_cache[key] = vector
    if len(_query_cache) > settings.embedding_cache_size:
        _query_cache.popitem(last=False)  # evict least-recently-used
    return list(vector)


def get_embedding_dim(provider: EmbeddingProvider | None = None) -> int:
    """Return the dimension of embeddings for the active provider."""
    provider = provider or settings.active_embedding
    if provider == EmbeddingProvider.HUGGINGFACE:
        return settings.hf_embedding_dim
    if provider == EmbeddingProvider.OLLAMA:
        return settings.ollama_embedding_dim
    return 1536  # OpenAI text-embedding-3-small


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (0.0 if either is a zero vector)."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
