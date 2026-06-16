"""Tier 2 — Cross-encoder reranking: retrieve wide, rerank narrow.

WORKING
    First-stage retrieval (hybrid) is recall-oriented and cheap — it casts a wide net
    (e.g. top-20). A cross-encoder then reads each (query, passage) pair *jointly* and
    scores true relevance, and we keep only the top few (e.g. top-3). Bi-encoders
    embed query and doc separately (fast, lossy); cross-encoders attend over both at
    once (slow, accurate) — so you use the bi-encoder to shortlist and the
    cross-encoder to order. It's the single biggest precision win after hybrid.

USE CASES
    Any time first-stage recall is decent but the *order* is wrong — the right passage
    is in the top-20 but not the top-3 the LLM actually reads. Reranking promotes it.

NOTE
    The harness's ``lexical_rerank`` is a stand-in for a real cross-encoder
    (e.g. bge-reranker / ms-marco-MiniLM). The two-stage *shape* is what matters.
"""

from __future__ import annotations

from advanced_rag._harness import (
    banner,
    bm25_search,
    dense_search,
    doc,
    lexical_rerank,
    rrf,
)


def retrieve_then_rerank(question: str, *, wide: int = 20, narrow: int = 3) -> dict:
    # Stage 1 — wide hybrid candidate pool.
    bm25 = [d for d, _ in bm25_search(question, k=wide)]
    dense = [d for d, _ in dense_search(question, k=wide)]
    pool = [d for d, _ in rrf([bm25, dense])][:wide]
    # Stage 2 — cross-encoder reranks the pool down to the few the LLM will read.
    reranked = [d for d, _ in lexical_rerank(question, pool, k=narrow)]
    return {"pool": pool, "reranked": reranked}


def main() -> None:
    banner("TIER 2 — Cross-encoder reranking (retrieve wide -> rerank narrow)")
    for q in [
        "what is reimbursed for meals on travel",
        "who approves security policy exceptions",
    ]:
        r = retrieve_then_rerank(q, wide=8, narrow=3)
        print(f"\nQ: {q}")
        print(f"  stage-1 pool (hybrid, wide): {r['pool']}")
        print(f"  stage-2 reranked (top-3)   : {r['reranked']}")
        top = doc(r["reranked"][0])
        print(f"  reader sees first: {top['id']}:{top['title']}")
    print("\nTakeaway: the reranker re-orders a wide candidate pool so the passage the "
          "LLM reads first is the most relevant — promoting good docs that fusion left mid-list.")


if __name__ == "__main__":
    main()
