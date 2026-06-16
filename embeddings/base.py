"""Core types + vector math for the embeddings lab.

Standalone study module (not wired into the app). Pure-Python vector math so the
whole lab runs with **zero dependencies** (numpy is used only if present, for
speed). Every backend implements the same `Embedder` interface, which applies the
production knobs uniformly:

    prefix (asymmetric query/doc) → provider embed → MRL truncate → L2 normalize

Why these knobs matter (the senior talking points):
- **Asymmetric prefixes** put queries and documents in different regions of the
  space → more discriminative retrieval (E5 `query:`/`passage:`, Nomic
  `search_query:`/`search_document:`).
- **Pooling** turns token vectors into one vector (mean is the safe default;
  last-token for LLM-based embedders; CLS for BERT-style).
- **L2 normalization** makes cosine == dot product and stabilizes ANN search.
- **Matryoshka (MRL)** front-loads information into early dims, so you can
  truncate (3072→256) for big storage/latency wins with minimal recall loss.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from embeddings.prefixes import apply_prefix

Vector = list[float]


# ── Pure-Python vector math (numpy-free) ─────────────────────────────

def l2_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def l2_normalize(v: Vector) -> Vector:
    n = l2_norm(v)
    return [x / n for x in v] if n > 0 else list(v)


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a: Vector, b: Vector) -> float:
    na, nb = l2_norm(a), l2_norm(b)
    return dot(a, b) / (na * nb) if na > 0 and nb > 0 else 0.0


def euclidean(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ── Pooling (token matrix → one vector) ──────────────────────────────

def mean_pool(tokens: list[Vector]) -> Vector:
    if not tokens:
        return []
    dim = len(tokens[0])
    return [sum(t[i] for t in tokens) / len(tokens) for i in range(dim)]


def max_pool(tokens: list[Vector]) -> Vector:
    if not tokens:
        return []
    dim = len(tokens[0])
    return [max(t[i] for t in tokens) for i in range(dim)]


def cls_pool(tokens: list[Vector]) -> Vector:
    return list(tokens[0]) if tokens else []


def last_pool(tokens: list[Vector]) -> Vector:
    return list(tokens[-1]) if tokens else []


POOLERS = {"mean": mean_pool, "max": max_pool, "cls": cls_pool, "last": last_pool}


def pool(tokens: list[Vector], strategy: str = "mean") -> Vector:
    return POOLERS[strategy](tokens)


# ── Matryoshka truncation ────────────────────────────────────────────

def truncate_mrl(v: Vector, dim: int | None) -> Vector:
    """Keep the first `dim` dimensions (Matryoshka). Caller re-normalizes."""
    return v[:dim] if dim else v


# ── Config + interface ───────────────────────────────────────────────


@dataclass
class EmbedderConfig:
    normalize: bool = True            # L2-normalize outputs (cosine == dot)
    dimensions: int | None = None     # MRL truncation target (e.g. 256); None = full
    prefix_preset: str = "none"       # none | e5 | nomic | bge
    pooling: str = "mean"             # mean | cls | last | max (raw-token backends)


class Embedder(ABC):
    """Interface: asymmetric `embed_query` / `embed_documents`, shared post-proc."""

    def __init__(self, config: EmbedderConfig | None = None):
        self.config = config or EmbedderConfig()

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[Vector]:
        """Provider-specific raw embedding (no prefix/normalize/MRL)."""

    # Asymmetric entry points -----------------------------------------
    def embed_documents(self, texts: list[str]) -> list[Vector]:
        prefixed = [apply_prefix(t, self.config.prefix_preset, "document") for t in texts]
        return [self._postprocess(v) for v in self._embed(prefixed)]

    def embed_query(self, text: str) -> Vector:
        prefixed = apply_prefix(text, self.config.prefix_preset, "query")
        return self._postprocess(self._embed([prefixed])[0])

    def _postprocess(self, v: Vector) -> Vector:
        v = truncate_mrl(v, self.config.dimensions)   # MRL first…
        if self.config.normalize:                     # …then renormalize
            v = l2_normalize(v)
        return v

    @property
    def dim(self) -> int:
        return self.config.dimensions or self._native_dim()

    def _native_dim(self) -> int:
        return len(self._embed(["dimension probe"])[0])
