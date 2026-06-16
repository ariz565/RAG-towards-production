"""Recursive character chunking (the production default — LangChain's approach).

Split on a *hierarchy* of separators (paragraph → line → sentence → word → char),
descending only when a piece is still over budget, then greedily merge adjacent
pieces back up to `chunk_size` with `chunk_overlap`.

When to use: the sensible default for most documents. Respects natural
boundaries far better than fixed-size while keeping chunks budget-bounded.
Trade-off: still structure-agnostic (doesn't know headings/tables), and meaning
can span a boundary.
"""

from __future__ import annotations

from chunking.base import Chunk, Chunker, count_tokens

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or _DEFAULT_SEPARATORS

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        pieces = self._split(text, self.separators)
        merged = self._merge(pieces)
        return [
            Chunk(text=t, index=i, metadata={**(metadata or {}), "strategy": "recursive"})
            for i, t in enumerate(merged)
            if t.strip()
        ]

    # Recursively break oversized pieces using progressively finer separators.
    def _split(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[-1]
        remaining = separators[1:]
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining = separators[i + 1 :]
                break

        raw = list(text) if separator == "" else text.split(separator)
        out: list[str] = []
        for part in raw:
            if not part:
                continue
            if count_tokens(part) <= self.chunk_size or not remaining:
                out.append(part)
            else:
                out.extend(self._split(part, remaining))
        return out

    # Greedily pack pieces up to chunk_size, carrying a token-bounded overlap tail.
    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece in pieces:
            t = count_tokens(piece)
            if current and current_tokens + t > self.chunk_size:
                chunks.append(" ".join(current).strip())
                current, current_tokens = self._overlap_tail(current)
            current.append(piece)
            current_tokens += t
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _overlap_tail(self, pieces: list[str]) -> tuple[list[str], int]:
        kept: list[str] = []
        total = 0
        for piece in reversed(pieces):
            t = count_tokens(piece)
            if total + t > self.chunk_overlap:
                break
            kept.insert(0, piece)
            total += t
        return kept, total
