"""DeepEval CI gate — LLM-judge quality metrics on the golden set.

This is the gate that blocks PRs when RAG quality regresses (2026 best practice:
evals-in-CI, not dashboards-after-the-fact).

Two tiers of tests:

1. ALWAYS-ON (no model needed): golden-set schema validity + structural checks.
2. GATED (RUN_EVALS=1): runs the real pipeline and scores answers with DeepEval
   metrics — Faithfulness, Answer Relevancy, Contextual Precision/Recall — against
   the 2026 production thresholds. Also asserts out-of-scope questions are refused.

Run locally:
    cd agent-backend/admission-guide
    pip install -r evals/requirements-eval.txt
    export RUN_EVALS=1
    export EVAL_JUDGE_PROVIDER=openai   # a hosted judge is strongly recommended
    pytest evals/ -v
"""

from __future__ import annotations

import os

import pytest

from app.config import settings
from evals.harness import bootstrap, load_golden_set, run_case

RUN_EVALS = os.getenv("RUN_EVALS") == "1"
gated = pytest.mark.skipif(not RUN_EVALS, reason="Set RUN_EVALS=1 to run the LLM-judge eval gate.")

GOLDEN = load_golden_set()
IN_SCOPE = [c for c in GOLDEN if c["type"] == "in_scope"]
OUT_OF_SCOPE = [c for c in GOLDEN if c["type"] == "out_of_scope"]


# ─────────────────────────────────────────────────────────────────────
# TIER 1 — always on (structure / no model required)
# ─────────────────────────────────────────────────────────────────────

def test_golden_set_nonempty():
    assert len(GOLDEN) >= 10, "Golden set should have a meaningful number of cases."
    assert IN_SCOPE, "Need in-scope cases."
    assert OUT_OF_SCOPE, "Need out-of-scope negatives."


@pytest.mark.parametrize("case", IN_SCOPE, ids=[c["id"] for c in IN_SCOPE])
def test_in_scope_cases_well_formed(case):
    assert case["expected_answer"].strip(), f"{case['id']} missing expected_answer"
    assert isinstance(case.get("expected_pages"), list)


# ─────────────────────────────────────────────────────────────────────
# TIER 2 — gated LLM-judge metrics (RUN_EVALS=1)
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def predictions():
    """Run the pipeline over the golden set once for the whole session."""
    import asyncio

    async def _run():
        await bootstrap()
        return {c["id"]: await run_case(c) for c in GOLDEN}

    return asyncio.run(_run())


@gated
@pytest.mark.parametrize("case", IN_SCOPE, ids=[c["id"] for c in IN_SCOPE])
def test_answer_quality(case, predictions):
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    from evals.judge import build_judge

    judge = build_judge()
    pred = predictions[case["id"]]

    # An in-scope question must not be wrongly refused.
    assert not pred["out_of_scope"], f"{case['id']} was wrongly routed out of scope"
    assert pred["retrieval_context"], f"{case['id']} retrieved no context"

    tc = LLMTestCase(
        input=case["question"],
        actual_output=pred["actual_output"],
        expected_output=case["expected_answer"],
        retrieval_context=pred["retrieval_context"],
    )

    metrics = [
        FaithfulnessMetric(threshold=settings.eval_faithfulness_threshold, model=judge),
        AnswerRelevancyMetric(threshold=settings.eval_answer_relevancy_threshold, model=judge),
        ContextualPrecisionMetric(threshold=settings.eval_context_precision_threshold, model=judge),
        ContextualRecallMetric(threshold=settings.eval_context_recall_threshold, model=judge),
    ]
    assert_test(tc, metrics)


@gated
@pytest.mark.parametrize("case", OUT_OF_SCOPE, ids=[c["id"] for c in OUT_OF_SCOPE])
def test_out_of_scope_refused(case, predictions):
    """Off-topic questions must be refused (guardrail), never answered with citations."""
    pred = predictions[case["id"]]
    assert pred["out_of_scope"], f"{case['id']} should have been refused as out of scope"
