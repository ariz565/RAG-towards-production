"""Agentic chunking (LLM-as-editor).

The most "semantic" strategy: extract propositions, then let the LLM act like an
editor — for each proposition decide whether it extends an existing chunk or
starts a new one, so every chunk is a coherent, self-contained idea regardless of
where its sentences sat in the source. (Greg Kamradt's agentic chunking.)

When to use: high-value, topically-uneven corpora (legal, clinical, financial)
where retrieval quality justifies the cost. Trade-off: an LLM call per
proposition — the most expensive strategy; an index-time batch job, never the hot
path.
"""

from __future__ import annotations

import re

from chunking.base import Chunk, Chunker
from chunking.llm_util import LLMFn, default_llm_fn
from chunking.proposition import PropositionChunker

_ROUTE_PROMPT = """You are grouping propositions into coherent, self-contained chunks.

Existing chunks (index: summary):
{summaries}

New proposition:
"{proposition}"

If it belongs with an existing chunk, reply with that index. If it starts a new
topic, reply NEW. Answer with ONLY the index number or the word NEW."""


class AgenticChunker(Chunker):
    def __init__(
        self,
        llm_fn: LLMFn | None = None,
        proposition_chunker: PropositionChunker | None = None,
        max_groups: int = 50,
    ):
        self.llm_fn = llm_fn or default_llm_fn
        self.proposition_chunker = proposition_chunker or PropositionChunker(llm_fn=self.llm_fn)
        self.max_groups = max_groups

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Chunk]:
        propositions = [c.text for c in self.proposition_chunker.chunk(text)]
        groups: list[list[str]] = []
        for prop in propositions:
            idx = self._route(prop, groups)
            if idx is None and len(groups) < self.max_groups:
                groups.append([prop])
            elif idx is None:
                groups[-1].append(prop)  # cap reached → fold into last group
            else:
                groups[idx].append(prop)

        return [
            Chunk(
                text=" ".join(g).strip(),
                index=i,
                metadata={**(metadata or {}), "strategy": "agentic", "propositions": len(g)},
            )
            for i, g in enumerate(groups)
        ]

    def _route(self, proposition: str, groups: list[list[str]]) -> int | None:
        if not groups:
            return None
        summaries = "\n".join(f"{i}: {' '.join(g)[:200]}" for i, g in enumerate(groups))
        try:
            answer = self.llm_fn(
                _ROUTE_PROMPT.format(summaries=summaries, proposition=proposition)
            ).strip()
        except Exception:
            return None
        if answer.upper().startswith("NEW"):
            return None
        digits = re.findall(r"\d+", answer)
        if not digits:
            return None
        idx = int(digits[0])
        return idx if 0 <= idx < len(groups) else None
