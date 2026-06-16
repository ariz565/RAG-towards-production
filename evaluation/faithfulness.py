"""Faithfulness — is every claim in the answer supported by the context?

The anti-hallucination metric. RAGAS method: decompose the answer into atomic
claims, then check each claim is entailed by the retrieved context (NLI).
score = supported_claims / total_claims.

Offline fallback: split into sentences, count a sentence "supported" if enough of
its content tokens appear in the context.
"""

from __future__ import annotations

from chunking.llm_util import LLMFn, parse_json_list

from evaluation.base import Metric, MetricResult, Sample, lexical_coverage, split_sentences

_CLAIMS_PROMPT = (
    "Break the ANSWER into a JSON array of atomic factual claims.\n\nANSWER:\n{answer}\n\n"
    "Return ONLY a JSON array of strings."
)
_NLI_PROMPT = (
    "Context:\n{context}\n\nClaim: \"{claim}\"\n\n"
    "Is the claim fully supported by the context? Answer ONLY 'yes' or 'no'."
)


class Faithfulness(Metric):
    name = "faithfulness"

    def __init__(self, llm_fn: LLMFn | None = None, coverage_threshold: float = 0.6):
        self.llm_fn = llm_fn
        self.coverage_threshold = coverage_threshold

    def score(self, sample: Sample) -> MetricResult:
        context = "\n".join(sample.contexts)
        if not sample.answer.strip():
            return MetricResult(self.name, 0.0, {"reason": "empty answer"})

        claims = self._claims(sample.answer)
        if not claims:
            return MetricResult(self.name, 0.0, {"reason": "no claims"})

        supported = [c for c in claims if self._supported(c, context)]
        score = len(supported) / len(claims)
        return MetricResult(self.name, round(score, 3), {
            "claims": len(claims), "supported": len(supported),
            "unsupported": [c for c in claims if c not in supported],
        })

    def _claims(self, answer: str) -> list[str]:
        if self.llm_fn:
            try:
                claims = parse_json_list(self.llm_fn(_CLAIMS_PROMPT.format(answer=answer)))
                if claims:
                    return claims
            except Exception:
                pass
        return split_sentences(answer)  # fallback: sentences as claims

    def _supported(self, claim: str, context: str) -> bool:
        if self.llm_fn:
            try:
                verdict = self.llm_fn(_NLI_PROMPT.format(context=context[:6000], claim=claim))
                return verdict.strip().lower().startswith("y")
            except Exception:
                pass
        return lexical_coverage(claim, context) >= self.coverage_threshold
