"""Optional offline Ragas evaluation — for tuning chunking / embeddings / reranking.

DeepEval (test_rag_quality.py) is the CI gate. Ragas is the offline experimentation
tool: run it while iterating on retrieval to see faithfulness / answer relevancy /
context precision / context recall move, before committing a change.

Ragas integrates cleanly with LangChain LLM + embedding wrappers. By default this
uses the project's OpenAI settings as the Ragas judge; adjust as needed.

Usage:
    cd agent-backend/admission-guide
    pip install -r evals/requirements-eval.txt
    python -m evals.ragas_offline
"""

from __future__ import annotations

import asyncio
import logging

from evals.harness import bootstrap, load_golden_set, run_case

logging.basicConfig(level=logging.WARNING)


async def _collect():
    await bootstrap()
    golden = [c for c in load_golden_set() if c["type"] == "in_scope"]
    return [await run_case(c) for c in golden]


def main() -> None:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError:
        raise SystemExit(
            "Ragas not installed. Run: pip install -r evals/requirements-eval.txt"
        )

    preds = asyncio.run(_collect())

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": p["question"],
                "response": p["actual_output"],
                "retrieved_contexts": p["retrieval_context"],
                "reference": p["expected_answer"],
            }
            for p in preds
        ]
    )

    # Ragas judge: configure your preferred LLM/embeddings here.
    # Defaults to env-configured OpenAI via LangChain wrappers.
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from app.config import settings

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.openai_model or "gpt-4o-mini", temperature=0)
    )
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    print("\n── Ragas offline results ─────────────────────────────")
    print(result)
    print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
