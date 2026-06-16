"""RAG evaluation lab — RAGAS-style generation metrics (standalone).

faithfulness · answer_relevancy · context_precision · context_recall.
Each runs offline (lexical/embedding fallback) or with an injected LLM for the
true RAGAS-style judgement. Retrieval metrics (recall@k/nDCG/MRR) live in the
retrieval lab.
"""

from evaluation.answer_relevancy import AnswerRelevancy
from evaluation.base import Metric, MetricResult, Sample, aggregate
from evaluation.context_precision import ContextPrecision
from evaluation.context_recall import ContextRecall
from evaluation.evaluate import evaluate
from evaluation.factory import MetricName, default_suite, get_metric
from evaluation.faithfulness import Faithfulness

__all__ = [
    "Sample",
    "Metric",
    "MetricResult",
    "aggregate",
    "evaluate",
    "default_suite",
    "get_metric",
    "MetricName",
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
]
