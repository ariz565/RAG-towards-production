"""Structure-aware (Markdown header) chunking.

Split on the Markdown header hierarchy (#, ##, ###…), keep the full header path
in metadata, and size-bound each section with a recursive splitter. Chunks
inherit their section context — great for docs, wikis, and technical manuals.

When to use: structured documents where headings carry meaning and you want
section-scoped retrieval / citations.
Trade-off: only as good as the document's structure (useless on unstructured text).
"""

from __future__ import annotations

import re

from chunking.base import Chunk, Chunker
from chunking.recursive import RecursiveChunker

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownHeaderChunker(Chunker):
    def __init__(self, max_tokens: int = 512, overlap: int = 64):
        # Sections still get budget-bounded so a huge section can't blow the window.
        self._inner = RecursiveChunker(chunk_size=max_tokens, chunk_overlap=overlap)

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        for header_path, body in self._split_sections(text):
            if not body.strip():
                continue
            for sub in self._inner.chunk(body):
                chunks.append(Chunk(
                    text=sub.text,
                    index=len(chunks),
                    metadata={
                        **(metadata or {}),
                        "strategy": "markdown",
                        "headers": header_path,
                        "section": header_path[-1] if header_path else "",
                    },
                ))
        return chunks

    # Walk lines, maintaining a header stack; emit (header_path, body) sections.
    def _split_sections(self, text: str) -> list[tuple[list[str], str]]:
        sections: list[tuple[list[str], str]] = []
        stack: list[tuple[int, str]] = []
        buffer: list[str] = []

        def flush() -> None:
            if any(line.strip() for line in buffer):
                sections.append(([h for _, h in stack], "\n".join(buffer)))

        for line in text.splitlines():
            m = _HEADER_RE.match(line)
            if m:
                flush()                       # close the section that just ended
                buffer = []
                level, header = len(m.group(1)), m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()               # pop siblings/deeper headers
                stack.append((level, header))
            else:
                buffer.append(line)
        flush()
        return sections
