"""Flag-driven retrieval demo over the sample corpus.

Examples:
    python -m retrieval.cli --strategy hybrid --query "minimum GPA for CS"
    python -m retrieval.cli --strategy bm25 --query "TOEFL"
    python -m retrieval.cli --strategy hybrid --rerank lexical
    python -m retrieval.cli --strategy hybrid --transform multi_query   # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import sys

from retrieval.factory import RetrieverStrategy, get_reranker, get_retriever, get_transform
from retrieval.pipeline import RetrievalPipeline
from retrieval.sample_data import CORPUS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrieval demo")
    p.add_argument("--strategy", default="hybrid", choices=[s.value for s in RetrieverStrategy])
    p.add_argument("--transform", default="none",
                   choices=["none", "multi_query", "hyde", "decomposition", "step_back"])
    p.add_argument("--rerank", default="none",
                   choices=["none", "lexical", "cross_encoder", "colbert"])
    p.add_argument("--query", default="minimum GPA for the computer science program")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--prefetch", type=int, default=20)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    retriever = get_retriever(args.strategy)
    retriever.index(CORPUS)

    pipeline = RetrievalPipeline(
        retriever=retriever,
        transform=get_transform(args.transform),
        reranker=get_reranker(args.rerank),
        k=args.k,
        prefetch=args.prefetch,
    )

    try:
        results = pipeline.search(args.query)
    except Exception as e:
        print(f"failed: {type(e).__name__}: {e}\n"
              f"(transforms need an LLM via OPENAI_API_KEY; cross_encoder needs sentence-transformers)",
              file=sys.stderr)
        return 1

    print(f"\nstrategy={args.strategy}  transform={args.transform}  rerank={args.rerank}")
    print(f"query: {args.query}\n")
    for r in results:
        print(f"  {r.rank}. ({r.score:+.4f}) [{r.id}] {r.text[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
