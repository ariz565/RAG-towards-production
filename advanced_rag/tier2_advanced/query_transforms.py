"""Tier 2 — Query transforms: reshape the query before retrieving.

The query the user types is often not the best *search* query. Four transforms,
each for a different failure mode:

- REWRITE / NORMALIZE  — expand abbreviations, tidy ("PTO" -> "paid time off (leave)").
  Conservative: aggressive rewriting can *poison* recall, so keep meaning intact.
- HyDE (Hypothetical Document Embeddings) — draft a plausible *answer* and embed THAT.
  An answer looks more like the target passage than a terse question does, so dense
  recall improves on under-specified questions.
- MULTI-QUERY — generate several paraphrases, retrieve for each, union/fuse the hits.
  Widens recall when one phrasing misses.
- DECOMPOSITION — split a multi-part / comparison question into sub-questions,
  retrieve per sub-question. Needed for "compare A and B" style queries.
- STEP-BACK — ask the more general question first to pull in background, then the
  specific one. Helps reasoning-heavy questions.

USE CASES
    Short/ambiguous queries (HyDE), vocabulary mismatch (multi-query), comparison &
    multi-hop (decomposition), reasoning questions (step-back).
"""

from __future__ import annotations

from advanced_rag._harness import FakeLLM, banner, dense_search, rrf


def _ids(pairs):
    return [d for d, _ in pairs]


def with_hyde(question: str, llm: FakeLLM, k: int = 3) -> list[str]:
    pseudo = llm.hyde(question)
    return _ids(dense_search(pseudo, k=k))


def with_multi_query(question: str, llm: FakeLLM, k: int = 3) -> list[str]:
    variants = llm.multi_query(question)
    lists = [_ids(dense_search(v, k=k)) for v in variants]
    return [d for d, _ in rrf(lists)][:k]


def with_decomposition(question: str, llm: FakeLLM, k: int = 2) -> dict:
    subs = llm.decompose(question)
    return {sub: _ids(dense_search(sub, k=k)) for sub in subs}


def with_step_back(question: str, llm: FakeLLM, k: int = 3) -> dict:
    general = llm.step_back(question)
    return {"general": _ids(dense_search(general, k=k)), "specific": _ids(dense_search(question, k=k))}


def main() -> None:
    banner("TIER 2 — Query transforms (HyDE / multi-query / decomposition / step-back)")
    llm = FakeLLM()

    q1 = "pto rules"
    print(f"\nREWRITE   Q: {q1!r}")
    print(f"  rewritten -> {llm.rewrite(q1)!r}")
    print(f"  baseline retrieve : {_ids(dense_search(q1, k=3))}")
    print(f"  rewritten retrieve: {_ids(dense_search(llm.rewrite(q1), k=3))}")

    q2 = "annual leave"
    print(f"\nHyDE      Q: {q2!r}")
    print(f"  hypothetical doc -> {llm.hyde(q2)!r}")
    print(f"  HyDE retrieve     : {with_hyde(q2, llm)}")

    q3 = "reimbursement"
    print(f"\nMULTIQ    Q: {q3!r}")
    print(f"  variants -> {llm.multi_query(q3)}")
    print(f"  fused retrieve    : {with_multi_query(q3, llm)}")

    q4 = "compare annual leave and parental leave"
    print(f"\nDECOMP    Q: {q4!r}")
    for sub, hits in with_decomposition(q4, llm).items():
        print(f"  sub: {sub!r:45} -> {hits}")

    q5 = "can I take parental leave while working remotely"
    print(f"\nSTEPBACK  Q: {q5!r}")
    sb = with_step_back(q5, llm)
    print(f"  general -> {sb['general']}")
    print(f"  specific-> {sb['specific']}")

    print(f"\nTakeaway: transforms reshape the query to the retriever's strengths. "
          f"(llm_calls={llm.calls}) Decomposition is what makes 'compare A and B' work.")


if __name__ == "__main__":
    main()
