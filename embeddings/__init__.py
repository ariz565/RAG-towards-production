"""Embeddings lab — production techniques for RAG embeddings (standalone).

Pure-Python core (numpy-optional); heavy SDKs imported lazily. Runs offline via
the dependency-free HashEmbedder.

Covers: asymmetric prefixes · pooling · L2 normalization · Matryoshka (MRL)
truncation · int8/binary quantization · dense / sparse (SPLADE) / multi-vector
(ColBERT) · caching.
"""

from embeddings.base import (
    Embedder,
    EmbedderConfig,
    Vector,
    cosine,
    dot,
    euclidean,
    l2_normalize,
    pool,
    truncate_mrl,
)
from embeddings.cache import CachingEmbedder
from embeddings.dense import (
    HashEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from embeddings.factory import EmbeddingProvider, get_embedder
from embeddings.multivector import ColBERTEmbedder, maxsim
from embeddings.sparse import SpladeEmbedder, lexical_sparse, sparse_dot

__all__ = [
    "Embedder",
    "EmbedderConfig",
    "Vector",
    "get_embedder",
    "EmbeddingProvider",
    "cosine",
    "dot",
    "euclidean",
    "l2_normalize",
    "pool",
    "truncate_mrl",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "CachingEmbedder",
    "ColBERTEmbedder",
    "maxsim",
    "SpladeEmbedder",
    "lexical_sparse",
    "sparse_dot",
]
