"""Tier 3 — Self-RAG: reflection gates that make the model critique its own RAG.

WORKING
    Self-RAG inserts explicit *reflection* decisions around retrieve→generate:
      1. IsRelevant  — for each retrieved passage, keep it only if it's actually
                       relevant (drop distractors before they pollute the prompt).
      2. IsSupported — after drafting, check every claim is grounded in the kept
                       passages; if not, regenerate or decline.
      3. IsUseful    — does the answer actually address the question?
    In the paper these are special "reflection tokens" the model emits; here we make
    each an explicit gate so the control flow is visible.

USE CASES
    Hallucination-intolerant domains — legal, medical, financial — where serving an
    unsupported claim is worse than saying "I can't confirm that."

LIMITATIONS
    More LLM calls (one grade per passage + a support check). Quality of the gates is
    only as good as the critic model.
"""

from __future__ import annotations

from advanced_rag._harness import FakeLLM, banner, bm25_search, dense_search, doc, rrf

RELEVANCE_GATE = 0.20   # IsRelevant: keep passages scoring at/above this


def self_rag(question: str, llm: FakeLLM, *, k: int = 5) -> dict:
    # Retrieve a candidate pool (hybrid).
    bm25 = [d for d, _ in bm25_search(question, k=k)]
    dense = [d for d, _ in dense_search(question, k=k)]
    pool = [doc(d) for d, _ in rrf([bm25, dense])][:k]

    # Gate 1 — IsRelevant: keep only passages the critic deems relevant.
    graded = [(c, llm.grade_relevance(question, c)) for c in pool]
    kept = [c for c, s in graded if s >= RELEVANCE_GATE]
    dropped = [(c["id"], round(s, 2)) for c, s in graded if s < RELEVANCE_GATE]

    if not kept:
        return {"answer": "No relevant passages passed the relevance gate; declining.",
                "kept": [], "dropped": dropped, "supported": False, "useful": False,
                "llm_calls": llm.calls}

    # Generate from the kept (relevant) context only.
    answer = llm.answer(question, kept)

    # Gate 2 — IsSupported: is the drafted answer grounded in the kept passages?
    supported = llm.supported(answer, kept)
    # Gate 3 — IsUseful: trivial heuristic (non-empty, addresses the question).
    useful = bool(answer) and "don't have enough" not in answer

    if not supported:
        answer = "Draft answer was not fully supported by the sources; declining to avoid hallucination."

    return {
        "answer": answer,
        "kept": [c["id"] for c in kept],
        "dropped": dropped,
        "supported": supported,
        "useful": useful,
        "llm_calls": llm.calls,
    }


def main() -> None:
    banner("TIER 3 — Self-RAG (IsRelevant / IsSupported / IsUseful gates)")
    for q in [
        "how many days of annual leave do employees get",   # well-supported
        "what is the company's cryptocurrency policy",       # no support → should decline
    ]:
        llm = FakeLLM()
        r = self_rag(q, llm)
        print(f"\nQ: {q}")
        print(f"  kept (relevant)   : {r['kept']}")
        print(f"  dropped (distractor): {r['dropped']}")
        print(f"  IsSupported       : {r['supported']}   IsUseful: {r['useful']}")
        print(f"  answer            : {r['answer']}")
        print(f"  llm_calls         : {r['llm_calls']}")
    print("\nTakeaway: the relevance gate drops distractors before generation, and the "
          "support gate turns an ungrounded draft into an honest refusal — fewer hallucinations.")


if __name__ == "__main__":
    main()
