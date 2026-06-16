"""Tier 3 — Corrective RAG (CRAG): grade the retrieval, then correct it.

WORKING
    After retrieval, a lightweight evaluator grades how well the retrieved context can
    answer the question, producing one of three verdicts:
      • CORRECT   (score high)  → use the context as-is.
      • AMBIGUOUS (score mid)   → refine: rewrite / multi-query and re-retrieve, then
                                   combine.
      • INCORRECT (score low)   → the corpus probably can't answer; fall back to an
                                   external source (web search stand-in here).
    This stops the classic failure where weak retrieval silently yields a confident,
    wrong answer.

USE CASES
    Open-domain assistants and knowledge bases with coverage gaps — CRAG degrades
    gracefully (refine or go external) instead of hallucinating from bad context.

LIMITATIONS
    Needs a decent evaluator and a trustworthy fallback. Extra retrieval rounds add
    latency/cost (bounded here to one correction).
"""

from __future__ import annotations

from advanced_rag._harness import FakeLLM, banner, dense_search, doc, rrf

UPPER, LOWER = 0.50, 0.20   # CORRECT >= UPPER; INCORRECT < LOWER; else AMBIGUOUS


def _context_score(question: str, contexts: list[dict], llm: FakeLLM) -> float:
    if not contexts:
        return 0.0
    return max(llm.grade_relevance(question, c) for c in contexts)


def _web_fallback(question: str) -> dict:
    """Stand-in for an external search tool when the corpus can't answer."""
    return {"id": "web", "title": "web", "text": f"[external] General reference about: {question}."}


def corrective_rag(question: str, llm: FakeLLM, *, k: int = 3) -> dict:
    contexts = [doc(d) for d, _ in dense_search(question, k=k)]
    score = _context_score(question, contexts, llm)

    if score >= UPPER:
        verdict, used = "CORRECT", contexts
    elif score < LOWER:
        verdict = "INCORRECT"
        used = [_web_fallback(question)]                       # go external
    else:
        verdict = "AMBIGUOUS"
        # Refine: multi-query expansion → re-retrieve → union with originals.
        variants = llm.multi_query(question)
        lists = [[d for d, _ in dense_search(v, k=k)] for v in variants]
        refined = [doc(d) for d, _ in rrf(lists)][:k]
        used = refined or contexts

    answer = llm.answer(question, used)
    return {"verdict": verdict, "score": round(score, 2),
            "used": [c["id"] for c in used], "answer": answer, "llm_calls": llm.calls}


def main() -> None:
    banner("TIER 3 — Corrective RAG (grade → correct: as-is / refine / fallback)")
    for q in [
        "how many days of annual paid leave do employees accrue",  # strong → CORRECT
        "meal reimbursement limit on business travel",             # partial → AMBIGUOUS → refine
        "what is the company's stock buyback schedule",            # absent → INCORRECT → fallback
    ]:
        llm = FakeLLM()
        r = corrective_rag(q, llm)
        print(f"\nQ: {q}")
        print(f"  verdict : {r['verdict']} (score={r['score']})")
        print(f"  used    : {r['used']}")
        print(f"  answer  : {r['answer']}")
    print("\nTakeaway: CRAG grades retrieval first, then refines weak context or falls back "
          "externally — so coverage gaps become graceful degradation, not confident hallucination.")


if __name__ == "__main__":
    main()
