"""Query transformations — fix the query before you retrieve.

Retrieval quality is capped by the query. These LLM-driven rewrites attack
different failure modes:
- **MultiQuery / RAG-Fusion** — N paraphrases from different angles → ↑recall.
- **HyDE** — generate a hypothetical *answer* and retrieve with it; bridges the
  short-query↔long-document gap.
- **Decomposition** — break a multi-part question into sub-questions.
- **StepBack** — abstract to a broader question to fetch governing concepts.

Each returns a list of query strings (the pipeline retrieves with all and fuses).
The LLM is injected; without one they degrade to the original query so the lab
still runs offline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chunking.llm_util import LLMFn, default_llm_fn, parse_json_list


class QueryTransform(ABC):
    @abstractmethod
    def transform(self, query: str) -> list[str]:
        ...


class Identity(QueryTransform):
    def transform(self, query: str) -> list[str]:
        return [query]


class MultiQuery(QueryTransform):
    def __init__(self, llm_fn: LLMFn | None = None, n: int = 3):
        self.llm_fn = llm_fn or default_llm_fn
        self.n = n

    def transform(self, query: str) -> list[str]:
        prompt = (f"Generate {self.n} alternative search queries, from different angles, "
                  f"for the question below. Return ONLY a JSON array of strings.\n\n{query}")
        try:
            variants = parse_json_list(self.llm_fn(prompt))
        except Exception:
            variants = []
        return [query, *variants] if variants else [query]


class HyDE(QueryTransform):
    def __init__(self, llm_fn: LLMFn | None = None):
        self.llm_fn = llm_fn or default_llm_fn

    def transform(self, query: str) -> list[str]:
        prompt = (f"Write a short, plausible passage that directly answers the question. "
                  f"It does not need to be factual — it is used only for retrieval.\n\n{query}")
        try:
            hypothetical = self.llm_fn(prompt).strip()
        except Exception:
            hypothetical = ""
        return [hypothetical] if hypothetical else [query]


class Decomposition(QueryTransform):
    def __init__(self, llm_fn: LLMFn | None = None):
        self.llm_fn = llm_fn or default_llm_fn

    def transform(self, query: str) -> list[str]:
        prompt = (f"Decompose the question into the minimal set of self-contained sub-questions "
                  f"needed to answer it. Return ONLY a JSON array of strings.\n\n{query}")
        try:
            subs = parse_json_list(self.llm_fn(prompt))
        except Exception:
            subs = []
        return subs or [query]


class StepBack(QueryTransform):
    def __init__(self, llm_fn: LLMFn | None = None):
        self.llm_fn = llm_fn or default_llm_fn

    def transform(self, query: str) -> list[str]:
        prompt = (f"Write ONE broader, more general 'step-back' question whose answer "
                  f"provides the concept needed to answer the specific question below. "
                  f"Reply with only the question.\n\n{query}")
        try:
            broader = self.llm_fn(prompt).strip()
        except Exception:
            broader = ""
        return [broader, query] if broader else [query]
