"""Answer Relevancy — does the answer actually address the question?

RAGAS method: ask an LLM to generate N questions that the ANSWER would answer,
embed them and the original question, and take the mean cosine similarity. A
focused, on-topic answer yields questions close to the original; a vague or
padded answer drifts.

Offline fallback: cosine(answer, question) via the embeddings lab's HashEmbedder
(no LLM, no API).
"""

from __future__ import annotations

from chunking.llm_util import LLMFn, parse_json_list
from embeddings.base import EmbedderConfig, cosine
from embeddings.dense import HashEmbedder

from evaluation.base import Metric, MetricResult, Sample

_GEN_Q_PROMPT = (
    "Given the ANSWER, generate {n} questions that this answer would correctly and "
    "directly answer.\n\nANSWER:\n{answer}\n\nReturn ONLY a JSON array of strings."
)


class AnswerRelevancy(Metric):
    name = "answer_relevancy"

    def __init__(self, llm_fn: LLMFn | None = None, embedder=None, n_questions: int = 3):
        self.llm_fn = llm_fn
        self.embedder = embedder or HashEmbedder(EmbedderConfig(normalize=True))
        self.n_questions = n_questions

    def score(self, sample: Sample) -> MetricResult:
        if not sample.answer.strip():
            return MetricResult(self.name, 0.0, {"reason": "empty answer"})

        q_vec = self.embedder.embed_query(sample.question)

        if self.llm_fn:
            generated = self._gen_questions(sample.answer)
            if generated:
                vecs = self.embedder.embed_documents(generated)
                sims = [max(0.0, cosine(q_vec, v)) for v in vecs]
                return MetricResult(self.name, round(sum(sims) / len(sims), 3),
                                    {"generated_questions": generated})

        # Fallback: direct question↔answer similarity.
        a_vec = self.embedder.embed_documents([sample.answer])[0]
        return MetricResult(self.name, round(max(0.0, cosine(q_vec, a_vec)), 3),
                            {"method": "question↔answer cosine (fallback)"})

    def _gen_questions(self, answer: str) -> list[str]:
        try:
            return parse_json_list(self.llm_fn(_GEN_Q_PROMPT.format(n=self.n_questions, answer=answer)))
        except Exception:
            return []
