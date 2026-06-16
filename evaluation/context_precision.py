"""Context Precision — are the *relevant* contexts ranked near the top?

Penalizes retrievers that bury the useful chunk under noise (the LLM gets
distracted before it reaches the signal). RAGAS computes a rank-weighted average
precision: for each position k, precision@k counted only when item k is relevant,
averaged over the relevant items.

Relevance per context is judged by an LLM (vs question + ground_truth) or, offline,
by lexical overlap with the ground_truth (or question if no ground_truth).
"""

from __future__ import annotations

from chunking.llm_util import LLMFn

from evaluation.base import Metric, MetricResult, Sample, lexical_coverage

_JUDGE_PROMPT = (
    "Question: {question}\nReference answer: {reference}\n\nContext: \"{context}\"\n\n"
    "Is this context useful for answering the question? Answer ONLY 'yes' or 'no'."
)


class ContextPrecision(Metric):
    name = "context_precision"

    def __init__(self, llm_fn: LLMFn | None = None, coverage_threshold: float = 0.3):
        self.llm_fn = llm_fn
        self.coverage_threshold = coverage_threshold

    def score(self, sample: Sample) -> MetricResult:
        if not sample.contexts:
            return MetricResult(self.name, 0.0, {"reason": "no contexts"})

        reference = sample.ground_truth or sample.question
        relevant_flags = [self._relevant(c, sample.question, reference) for c in sample.contexts]

        # Rank-weighted average precision (RAGAS-style).
        num_relevant = sum(relevant_flags)
        if num_relevant == 0:
            return MetricResult(self.name, 0.0, {"relevant": 0})

        hits = 0
        precision_sum = 0.0
        for k, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                hits += 1
                precision_sum += hits / k
        score = precision_sum / num_relevant
        return MetricResult(self.name, round(score, 3),
                            {"relevant_positions": [i + 1 for i, r in enumerate(relevant_flags) if r]})

    def _relevant(self, context: str, question: str, reference: str) -> bool:
        if self.llm_fn:
            try:
                v = self.llm_fn(_JUDGE_PROMPT.format(question=question, reference=reference, context=context[:4000]))
                return v.strip().lower().startswith("y")
            except Exception:
                pass
        return lexical_coverage(reference, context) >= self.coverage_threshold
