"""Core types + helpers for the RAG evaluation lab (RAGAS-style).

Retrieval metrics (recall@k/nDCG/MRR) live in the retrieval lab; this lab covers
the **generation half** — does the answer stay faithful to the context, address
the question, and did retrieval surface the right context?

Every metric takes an optional `llm_fn` (and `embedder`). With an LLM it runs the
true RAGAS-style method (claim decomposition + NLI + question generation); with
none it falls back to a transparent **lexical** heuristic so the whole lab runs
offline and deterministically. Real evals should inject an LLM.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "for", "is", "are", "and", "or", "in", "on",
         "at", "by", "with", "as", "it", "this", "that", "be", "do", "i", "you"}


@dataclass
class Sample:
    """One evaluation example."""

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)   # retrieved chunks
    ground_truth: str | None = None                     # reference answer (optional)


@dataclass
class MetricResult:
    name: str
    score: float            # 0..1
    detail: dict = field(default_factory=dict)


# ── Text helpers ─────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split((text or "").strip()) if s.strip()]


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP}


def lexical_coverage(claim: str, context: str) -> float:
    """Fraction of a claim's content tokens present in the context (0..1)."""
    c = content_tokens(claim)
    if not c:
        return 0.0
    return len(c & content_tokens(context)) / len(c)


# ── Metric interface ─────────────────────────────────────────────────


class Metric(ABC):
    name: str = "metric"

    @abstractmethod
    def score(self, sample: Sample) -> MetricResult:
        ...


def aggregate(results: list[MetricResult]) -> float:
    return round(sum(r.score for r in results) / len(results), 3) if results else 0.0
