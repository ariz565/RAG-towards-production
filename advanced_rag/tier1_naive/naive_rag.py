"""Tier 1 — Naive RAG: single vector search -> stuff -> generate.

WORKING
    query -> embed -> top-k dense similarity -> concat chunks into the prompt ->
    one LLM call -> answer. One retrieval, one generation. No fusion, no rerank,
    no verification.

USE CASES
    FAQ bots, semantic document Q&A where the answer sits in one or two chunks that
    are *lexically/semantically close* to the question. Cheapest and fastest tier
    (~1 LLM call, <1s). Right for ~60-70% of real queries.

LIMITATIONS (this demo shows two on purpose)
    1. Exact terms / codes: "SEC-17" can be diluted by an embedding (geometry, not
       symbols) — sparse/BM25 wins there (see tier2 hybrid).
    2. Vocabulary mismatch when synonym expansion is off — paraphrases miss.
    These motivate the entire Advanced tier.
"""

from __future__ import annotations

from advanced_rag._harness import FakeLLM, banner, dense_search, doc


def naive_rag(question: str, llm: FakeLLM, k: int = 3) -> dict:
    hits = dense_search(question, k=k)
    contexts = [doc(doc_id) for doc_id, _ in hits]
    answer = llm.answer(question, contexts)
    return {
        "question": question,
        "retrieved": [f"{d['id']}:{d['title']}" for d in contexts],
        "answer": answer,
        "llm_calls": llm.calls,
    }


def main() -> None:
    banner("TIER 1 — Naive RAG (single dense retrieval -> generate)")
    for q in [
        "How many days of annual leave do I get?",     # works well — close match
        "What are the exceptions to SEC-17?",          # exact code — dense may dilute
    ]:
        llm = FakeLLM()
        r = naive_rag(q, llm)
        print(f"\nQ: {q}")
        print(f"  retrieved : {r['retrieved']}")
        print(f"  answer    : {r['answer']}")
        print(f"  llm_calls : {r['llm_calls']}")
    print("\nTakeaway: one retrieval + one generation is cheap and often enough — but "
          "exact codes and paraphrases expose its limits. Continue to Tier 2.")


if __name__ == "__main__":
    main()
