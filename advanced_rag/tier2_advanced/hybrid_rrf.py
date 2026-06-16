"""Tier 2 — Hybrid search (BM25 + dense) fused with Reciprocal Rank Fusion.

WORKING
    Run sparse (BM25, exact terms) and dense (embedding, meaning) retrieval in
    parallel, then fuse the two ranked lists with RRF:  score(d) = Σ 1/(rank_i + k),
    k≈60. Rank-based fusion needs no score normalization and rewards documents that
    rank well in *either* system.

USE CASES
    The default Advanced-tier retriever. Sparse catches codes/IDs/rare terms
    ("SEC-17", "order 991"); dense catches paraphrases ("time off" ~ "leave").
    Hybrid is rarely worse than either alone and usually better.

WHY RRF (vs weighted score sum)
    BM25 scores and cosine scores live on different scales; naively adding them lets
    one dominate. RRF uses *ranks*, so it's scale-free and robust.
"""

from __future__ import annotations

from advanced_rag._harness import (
    banner,
    bm25_search,
    dense_search,
    doc,
    recall_at_k,
    rrf,
)


def hybrid_search(question: str, k: int = 5) -> list[str]:
    bm25 = [d for d, _ in bm25_search(question, k=10)]
    dense = [d for d, _ in dense_search(question, k=10)]
    fused = [d for d, _ in rrf([bm25, dense])]
    return fused[:k]


def _ids(pairs):
    return [d for d, _ in pairs]


def main() -> None:
    banner("TIER 2 — Hybrid (BM25 + dense) -> RRF fusion")

    # Two queries chosen so each single retriever has a blind spot.
    cases = [
        # exact code → BM25 should shine, dense may dilute it
        ("What are the exceptions to SEC-17?", {"d4"}),
        # paraphrase ("time off") → dense (synonym-expanded) should shine, BM25 may miss
        ("how do I request time off", {"d1"}),
    ]
    for q, relevant in cases:
        bm25 = _ids(bm25_search(q, k=5))
        dense = _ids(dense_search(q, k=5))
        fused = hybrid_search(q, k=5)
        print(f"\nQ: {q}   (relevant={relevant})")
        print(f"  bm25   : {bm25}   recall@3={recall_at_k(bm25, relevant, 3):.2f}")
        print(f"  dense  : {dense}   recall@3={recall_at_k(dense, relevant, 3):.2f}")
        print(f"  hybrid : {fused}   recall@3={recall_at_k(fused, relevant, 3):.2f}")
        top = doc(fused[0])
        print(f"  hybrid top-1: {top['id']}:{top['title']}")

    print("\nTakeaway: hybrid+RRF matches the better of the two retrievers on each query "
          "type — so you don't have to bet on one. This is the Advanced-tier baseline.")


if __name__ == "__main__":
    main()
