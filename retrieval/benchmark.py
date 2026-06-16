"""Capstone benchmark — BM25 vs dense vs hybrid (+rerank) on recall@k / nDCG / MRR.

This ties the trio together: documents (which you'd produce with the *chunking*
lab) are indexed; the dense path uses the *embeddings* lab; retrieval strategies
are scored against gold labels with the IR metrics from base.py. It's the
empirical way to choose a retrieval stack.

Usage:
    python -m retrieval.benchmark              # offline (hash embedder, lexical rerank)
    python -m retrieval.benchmark --k 5
"""

from __future__ import annotations

import argparse
import sys

from retrieval.base import mrr, ndcg_at_k, recall_at_k
from retrieval.factory import get_reranker, get_retriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.sample_data import CORPUS, QRELS

# (label, strategy, rerank)
CONFIGS = [
    ("bm25", "bm25", "none"),
    ("dense", "dense", "none"),
    ("hybrid", "hybrid", "none"),
    ("hybrid+rerank", "hybrid", "lexical"),
]


def _evaluate(pipeline: RetrievalPipeline, k: int) -> dict:
    recalls, ndcgs, mrrs = [], [], []
    for query, relevant in QRELS:
        ids = [r.id for r in pipeline.search(query)]
        recalls.append(recall_at_k(ids, relevant, k))
        ndcgs.append(ndcg_at_k(ids, relevant, k))
        mrrs.append(mrr(ids, relevant))
    n = len(QRELS)
    return {
        f"recall@{k}": round(sum(recalls) / n, 3),
        f"ndcg@{k}": round(sum(ndcgs) / n, 3),
        "mrr": round(sum(mrrs) / n, 3),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrieval benchmark")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = []
    for label, strategy, rerank in CONFIGS:
        retriever = get_retriever(strategy)
        retriever.index(CORPUS)
        pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=get_reranker(rerank),
            k=args.k,
            prefetch=10,
        )
        metrics = _evaluate(pipeline, args.k)
        rows.append((label, metrics))

    print(f"\nRetrieval benchmark — {len(QRELS)} queries, {len(CORPUS)} docs, k={args.k}\n")
    cols = list(rows[0][1].keys())
    print("strategy".ljust(16) + "".join(c.ljust(12) for c in cols))
    print("-" * (16 + 12 * len(cols)))
    for label, metrics in rows:
        print(label.ljust(16) + "".join(str(metrics[c]).ljust(12) for c in cols))
    print("\n(offline: hash embedder + lexical rerank. With real models the dense/"
          "rerank rows rise substantially — that's the point of measuring.)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
