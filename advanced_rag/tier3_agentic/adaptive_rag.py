"""Tier 3 — Adaptive RAG: route each query to the cheapest tier that will work.

WORKING
    A lightweight classifier labels the query's complexity, then dispatches:
      • SIMPLE  → one-shot retrieve→generate (Tier 1/2). ~1 LLM call, fast, cheap.
      • COMPLEX → an agentic path: decompose into sub-questions, retrieve per
                  sub-question, then synthesize. Several LLM calls.
    Because ~60–70% of production queries are simple, routing keeps average cost/latency
    low while still handling the hard tail well. This is widely considered the current
    state of the art for real, mixed workloads.

USE CASES
    Any production assistant facing a *mix* of trivial lookups and multi-hop/comparison
    questions — you don't want to pay agentic cost for "what's the leave policy?".

LIMITATIONS
    The router is the weak link: misroute a complex query to the cheap path and you
    under-answer it. Keep the classifier conservative and measure misroutes.
"""

from __future__ import annotations

from advanced_rag._harness import FakeLLM, banner, dense_search, doc, rrf


def _simple_path(question: str, llm: FakeLLM) -> dict:
    contexts = [doc(d) for d, _ in dense_search(question, k=3)]
    return {"answer": llm.answer(question, contexts), "used": [c["id"] for c in contexts]}


def _complex_path(question: str, llm: FakeLLM) -> dict:
    subs = llm.decompose(question)
    used: list[str] = []
    per_sub_lists: list[list[str]] = []
    for sub in subs:
        ids = [d for d, _ in dense_search(sub, k=2)]
        per_sub_lists.append(ids)
    fused = [d for d, _ in rrf(per_sub_lists)][:4]
    contexts = [doc(d) for d in fused]
    used = fused
    return {"answer": llm.answer(question, contexts), "used": used, "sub_questions": subs}


def adaptive_rag(question: str, llm: FakeLLM) -> dict:
    complexity = llm.classify_complexity(question)
    route = _complex_path if complexity == "complex" else _simple_path
    result = route(question, llm)
    result["route"] = complexity
    result["llm_calls"] = llm.calls
    return result


def main() -> None:
    banner("TIER 3 — Adaptive RAG (complexity router: simple→cheap, complex→agentic)")
    for q in [
        "how many public holidays are there",                       # SIMPLE
        "compare annual leave and parental leave entitlements",     # COMPLEX → decompose
    ]:
        llm = FakeLLM()
        r = adaptive_rag(q, llm)
        print(f"\nQ: {q}")
        print(f"  route     : {r['route'].upper()}")
        if r.get("sub_questions"):
            print(f"  sub-qs    : {r['sub_questions']}")
        print(f"  used      : {r['used']}")
        print(f"  answer    : {r['answer']}")
        print(f"  llm_calls : {r['llm_calls']}  (cheap path uses far fewer)")
    print("\nTakeaway: the router spends agentic budget only on hard queries; the simple "
          "majority stays one-shot. That's how Adaptive RAG optimizes average cost/latency.")


if __name__ == "__main__":
    main()
