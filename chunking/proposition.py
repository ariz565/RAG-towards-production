"""Proposition-based chunking (Chen et al., "Dense X Retrieval", 2023).

Decompose text into **propositions** — atomic, self-contained factual statements
with pronouns/references resolved, each understandable in isolation. Every
proposition becomes a chunk. Maximizes retrieval precision: a query matches the
exact fact, not a paragraph that merely contains it.

When to use: fact-dense corpora (specs, policies, FAQs) where pinpoint retrieval
beats passage retrieval. Trade-off: LLM extraction cost; very small chunks can
lose narrative context (often paired with parent-child or contextual).
"""

from __future__ import annotations

from chunking.base import Chunk, Chunker, split_sentences
from chunking.llm_util import LLMFn, default_llm_fn, parse_json_list

_PROMPT = """Decompose the following text into a list of atomic PROPOSITIONS.
Rules:
- Each proposition states a single, self-contained fact.
- Resolve pronouns/references so each stands alone (no "it", "they", "this").
- Do not add information that isn't in the text.

Text:
{text}

Return ONLY a JSON array of strings."""


class PropositionChunker(Chunker):
    def __init__(self, llm_fn: LLMFn | None = None, max_chars: int = 6000):
        self.llm_fn = llm_fn or default_llm_fn
        self.max_chars = max_chars

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        propositions = self._extract(text[: self.max_chars])
        return [
            Chunk(text=p, index=i, metadata={**(metadata or {}), "strategy": "proposition"})
            for i, p in enumerate(propositions)
        ]

    def _extract(self, text: str) -> list[str]:
        try:
            props = parse_json_list(self.llm_fn(_PROMPT.format(text=text)))
        except Exception:
            props = []
        return props or split_sentences(text)  # fallback: sentences are weak propositions
