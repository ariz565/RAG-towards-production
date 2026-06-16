"""Context Recall — did retrieval surface everything the answer needs?

RAGAS method: decompose the GROUND-TRUTH answer into claims, then check each is
attributable to (supported by) the retrieved contexts.
score = attributable_claims / total_ground_truth_claims.

This is the retrieval-quality metric that needs a reference answer. Offline
fallback: lexical coverage of each ground-truth sentence by the contexts.
"""

from __future__ import annotations

from chunking.llm_util import LLMFn, parse_json_list

from evaluation.base import Metric, MetricResult, Sample, lexical_coverage, split_sentences

_CLAIMS_PROMPT = (
    "Break the REFERENCE ANSWER into a JSON array of atomic factual claims.\n\n"
    "REFERENCE ANSWER:\n{reference}\n\nReturn ONLY a JSON array of strings."
)
_ATTR_PROMPT = (
    "Context:\n{context}\n\nClaim: \"{claim}\"\n\n"
    "Can this claim be attributed to the context? Answer ONLY 'yes' or 'no'."
)


class ContextRecall(Metric):
    name = "context_recall"

    def __init__(self, llm_fn: LLMFn | None = None, coverage_threshold: float = 0.5):
        self.llm_fn = llm_fn
        self.coverage_threshold = coverage_threshold

    def score(self, sample: Sample) -> MetricResult:
        if not sample.ground_truth:
            return MetricResult(self.name, 0.0, {"reason": "context_recall needs ground_truth"})
        if not sample.contexts:
            return MetricResult(self.name, 0.0, {"reason": "no contexts"})

        context = "\n".join(sample.contexts)
        claims = self._claims(sample.ground_truth)
        if not claims:
            return MetricResult(self.name, 0.0, {"reason": "no claims"})

        attributable = [c for c in claims if self._attributable(c, context)]
        return MetricResult(self.name, round(len(attributable) / len(claims), 3),
                            {"claims": len(claims), "attributable": len(attributable)})

    def _claims(self, reference: str) -> list[str]:
        if self.llm_fn:
            try:
                claims = parse_json_list(self.llm_fn(_CLAIMS_PROMPT.format(reference=reference)))
                if claims:
                    return claims
            except Exception:
                pass
        return split_sentences(reference)

    def _attributable(self, claim: str, context: str) -> bool:
        if self.llm_fn:
            try:
                v = self.llm_fn(_ATTR_PROMPT.format(context=context[:6000], claim=claim))
                return v.strip().lower().startswith("y")
            except Exception:
                pass
        return lexical_coverage(claim, context) >= self.coverage_threshold
