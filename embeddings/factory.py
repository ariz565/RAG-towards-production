"""Provider registry + factory for dense embedders.

    emb = get_embedder("hash", config=EmbedderConfig(dimensions=128, prefix_preset="e5"))
    qv  = emb.embed_query("...")

Multi-vector (ColBERT) and sparse (SPLADE) have different interfaces and are
imported directly from their modules.
"""

from __future__ import annotations

from enum import Enum

from embeddings.base import Embedder, EmbedderConfig
from embeddings.dense import (
    HashEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)


class EmbeddingProvider(str, Enum):
    HASH = "hash"                              # dependency-free, offline
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    OLLAMA = "ollama"


_REGISTRY: dict[EmbeddingProvider, type[Embedder]] = {
    EmbeddingProvider.HASH: HashEmbedder,
    EmbeddingProvider.SENTENCE_TRANSFORMERS: SentenceTransformerEmbedder,
    EmbeddingProvider.OPENAI: OpenAIEmbedder,
    EmbeddingProvider.OLLAMA: OllamaEmbedder,
}


def get_embedder(
    provider: str | EmbeddingProvider,
    config: EmbedderConfig | None = None,
    **kwargs,
) -> Embedder:
    provider = EmbeddingProvider(provider)
    return _REGISTRY[provider](config=config, **kwargs)
