"""Core types for the chunking lab.

Standalone study module — NOT wired into the main app. Every strategy is a
small, single-responsibility class implementing the `Chunker` interface, so
they're interchangeable behind a factory/flag.

Design choices that matter at the senior bar:
- Token-accurate sizing via tiktoken (chunk budgets should be in *tokens*, not
  characters, because that's what the model and the context window count).
- A uniform `Chunk` carrying offsets + token count + metadata + parent link, so
  any strategy (including parent-child) shares one shape.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ── Tokenizer (lazy, shared) ─────────────────────────────────────────

_ENCODER = None


def get_encoder():
    """Return a cached tiktoken encoder, or None if tiktoken isn't installed."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = False  # sentinel: tried and unavailable
    return _ENCODER or None


def count_tokens(text: str) -> int:
    """Token count via tiktoken; falls back to a ~1.3×word heuristic."""
    enc = get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, int(len(text.split()) * 1.3))


# ── Sentence splitting (shared by sentence/semantic strategies) ──────

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence segmentation (regex; good enough for chunking)."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


# ── Chunk model ──────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A unit of retrievable text plus provenance."""

    text: str
    index: int = 0
    id: str = ""
    parent_id: str | None = None          # set by hierarchical (parent-child) strategy
    start_char: int | None = None         # offset in the source (when tracked)
    end_char: int | None = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens(self.text)
        if not self.id:
            self.id = f"chunk-{self.index}"


# ── Strategy interface ───────────────────────────────────────────────


class Chunker(ABC):
    """Strategy contract: text in → ordered chunks out."""

    @abstractmethod
    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        """Split `text` into chunks, attaching `metadata` to each."""

    def __call__(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        return self.chunk(text, metadata=metadata)


def stats(chunks: list[Chunk]) -> dict:
    """Summary stats for a chunk set (handy for comparing strategies)."""
    if not chunks:
        return {"count": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0}
    toks = [c.token_count for c in chunks]
    return {
        "count": len(chunks),
        "avg_tokens": round(sum(toks) / len(toks), 1),
        "min_tokens": min(toks),
        "max_tokens": max(toks),
    }
