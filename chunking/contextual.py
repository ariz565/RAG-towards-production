"""Contextual chunking (Anthropic "Contextual Retrieval").

A *decorator* over any base chunker: for each chunk, an LLM writes a short,
document-aware context ("This section of the 2025 admissions guide explains…")
which is prepended before embedding. Solves the classic failure where an
isolated chunk ("the deadline is March 1st") loses what it refers to.

Anthropic report ~−49% retrieval failures (−67% combined with a re-ranker).

When to use: high-value corpora where retrieval precision matters and you can
afford one LLM call per chunk at index time. Trade-off: indexing cost/latency.
The original text is preserved in metadata for display/grounding.
"""

from __future__ import annotations

from chunking.base import Chunk, Chunker
from chunking.llm_util import LLMFn, default_llm_fn
from chunking.recursive import RecursiveChunker

_PROMPT = """<document>
{doc}
</document>

Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>

Write a short (1-2 sentence) context that situates this chunk within the overall
document to improve search retrieval. Answer with ONLY the context."""


class ContextualChunker(Chunker):
    def __init__(
        self,
        base_chunker: Chunker | None = None,
        llm_fn: LLMFn | None = None,
        doc_context_chars: int = 8000,
    ):
        self.base_chunker = base_chunker or RecursiveChunker()
        self.llm_fn = llm_fn or default_llm_fn
        self.doc_context_chars = doc_context_chars

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        base_chunks = self.base_chunker.chunk(text, metadata=metadata)
        doc = text[: self.doc_context_chars]

        enriched: list[Chunk] = []
        for c in base_chunks:
            context = self._contextualize(doc, c.text)
            enriched.append(Chunk(
                text=f"{context}\n\n{c.text}" if context else c.text,
                index=c.index,
                metadata={
                    **c.metadata,
                    "strategy": "contextual",
                    "context": context,
                    "original_text": c.text,  # keep clean text for display/grounding
                },
            ))
        return enriched

    def _contextualize(self, doc: str, chunk: str) -> str:
        try:
            return self.llm_fn(_PROMPT.format(doc=doc, chunk=chunk)).strip()
        except Exception:
            return ""  # degrade to the bare chunk rather than fail the batch
