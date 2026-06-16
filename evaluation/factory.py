"""Metric registry + factory (flag-selectable)."""

from __future__ import annotations

from enum import Enum

from evaluation.answer_relevancy import AnswerRelevancy
from evaluation.base import Metric
from evaluation.context_precision import ContextPrecision
from evaluation.context_recall import ContextRecall
from evaluation.faithfulness import Faithfulness


class MetricName(str, Enum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


_REGISTRY = {
    MetricName.FAITHFULNESS: Faithfulness,
    MetricName.ANSWER_RELEVANCY: AnswerRelevancy,
    MetricName.CONTEXT_PRECISION: ContextPrecision,
    MetricName.CONTEXT_RECALL: ContextRecall,
}


def get_metric(name: str | MetricName, llm_fn=None, embedder=None) -> Metric:
    name = MetricName(name)
    if name is MetricName.ANSWER_RELEVANCY:
        return AnswerRelevancy(llm_fn=llm_fn, embedder=embedder)
    return _REGISTRY[name](llm_fn=llm_fn)


def default_suite(llm_fn=None, embedder=None) -> list[Metric]:
    return [get_metric(n, llm_fn=llm_fn, embedder=embedder) for n in MetricName]
