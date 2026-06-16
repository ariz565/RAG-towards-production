"""Benchmark chunking strategies side-by-side on one input.

Runs each strategy over the same text and prints a comparison table — the
empirical way to *choose* a strategy (count, size distribution, latency) instead
of arguing about it. Classical strategies run by default; embedding/LLM ones are
opt-in (they need models / API keys) and skipped gracefully if unavailable.

Usage:
    python -m chunking.benchmark
    python -m chunking.benchmark --file notes.md --chunk-size 256
    python -m chunking.benchmark --include-embedding   # + semantic, late
    python -m chunking.benchmark --include-llm          # + contextual, proposition, agentic
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from chunking.cli import _SAMPLE
from chunking.factory import ChunkingStrategy, get_chunker

CLASSICAL = ["fixed", "recursive", "sentence", "markdown", "parent_child"]
EMBEDDING = ["semantic", "late"]
LLM = ["contextual", "proposition", "agentic"]


def _kwargs(strategy: ChunkingStrategy, args) -> dict:
    if strategy in (ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE):
        return {"chunk_size": args.chunk_size, "chunk_overlap": args.overlap}
    if strategy is ChunkingStrategy.SENTENCE:
        return {"max_tokens": args.chunk_size}
    if strategy is ChunkingStrategy.MARKDOWN:
        return {"max_tokens": args.chunk_size, "overlap": args.overlap}
    if strategy is ChunkingStrategy.PARENT_CHILD:
        return {
            "parent_chunk_size": args.chunk_size * 4,
            "child_chunk_size": args.chunk_size,
            "child_overlap": min(args.overlap, max(0, args.chunk_size - 1)),
        }
    if strategy is ChunkingStrategy.LATE:
        return {"chunk_tokens": args.chunk_size}
    return {}


def _metrics(strategy: str, chunks, elapsed_ms: float) -> dict:
    toks = [c.token_count for c in chunks] or [0]
    return {
        "strategy": strategy,
        "chunks": len(chunks),
        "avg_tok": round(statistics.mean(toks), 1),
        "min_tok": min(toks),
        "max_tok": max(toks),
        "std_tok": round(statistics.pstdev(toks), 1) if len(toks) > 1 else 0.0,
        "ms": round(elapsed_ms, 1),
    }


_COLS = [
    ("strategy", "Strategy", 14),
    ("chunks", "Chunks", 8),
    ("avg_tok", "Avg tok", 9),
    ("min_tok", "Min", 6),
    ("max_tok", "Max", 6),
    ("std_tok", "Std", 8),
    ("ms", "Time ms", 9),
]


def _print_table(rows: list[dict]) -> None:
    header = "".join(label.ljust(w) for _, label, w in _COLS)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("".join(str(r.get(key, "")).ljust(w) for key, _, w in _COLS))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark chunking strategies")
    p.add_argument("--file", help="Text/markdown file (defaults to a built-in sample)")
    p.add_argument("--chunk-size", type=int, default=256, dest="chunk_size")
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--include-embedding", action="store_true")
    p.add_argument("--include-llm", action="store_true")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    text = open(args.file, encoding="utf-8").read() if args.file else _SAMPLE

    strategies = list(CLASSICAL)
    if args.include_embedding:
        strategies += EMBEDDING
    if args.include_llm:
        strategies += LLM

    rows: list[dict] = []
    for name in strategies:
        strategy = ChunkingStrategy(name)
        try:
            chunker = get_chunker(strategy, **_kwargs(strategy, args))
            start = time.perf_counter()
            chunks = chunker.chunk(text)
            rows.append(_metrics(name, chunks, (time.perf_counter() - start) * 1000))
        except Exception as e:
            print(f"  skipped '{name}': {type(e).__name__}: {e}", file=sys.stderr)

    _print_table(rows)
    print(f"\nInput: {len(text)} chars. Pick the strategy that best fits your "
          f"retrieval evals (recall@k / context precision), not the prettiest table.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
