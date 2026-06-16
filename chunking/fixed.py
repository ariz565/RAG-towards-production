"""Fixed-size chunking with overlap (the baseline everyone starts with).

Sliding window of a fixed token (or char) size, stepping by `size - overlap`.

When to use: uniform/unstructured text, or when you want predictable, model-
budget-aligned chunks. Cheap and deterministic.
Trade-off: blind to meaning — it will happily cut mid-sentence/mid-idea.
"""

from __future__ import annotations

from chunking.base import Chunk, Chunker, count_tokens, get_encoder


class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, unit: str = "token"):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if unit not in ("token", "char"):
            raise ValueError("unit must be 'token' or 'char'")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        encoder = get_encoder() if self.unit == "token" else None
        if encoder is not None:
            return self._chunk_tokens(text, encoder, metadata or {})
        return self._chunk_chars(text, metadata or {})

    # ── token windows (preferred: budgets are in tokens) ──────────────
    def _chunk_tokens(self, text, encoder, metadata) -> list[Chunk]:
        ids = encoder.encode(text)
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(ids), step)):
            window = ids[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(Chunk(
                text=encoder.decode(window).strip(),
                index=i,
                token_count=len(window),
                metadata={**metadata, "strategy": "fixed", "unit": "token"},
            ))
            if start + self.chunk_size >= len(ids):
                break
        return chunks

    # ── char windows (fallback when tiktoken is unavailable) ──────────
    def _chunk_chars(self, text, metadata) -> list[Chunk]:
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(text), step)):
            window = text[start : start + self.chunk_size]
            if not window.strip():
                break
            chunks.append(Chunk(
                text=window.strip(),
                index=i,
                start_char=start,
                end_char=start + len(window),
                metadata={**metadata, "strategy": "fixed", "unit": "char"},
            ))
            if start + self.chunk_size >= len(text):
                break
        return chunks
