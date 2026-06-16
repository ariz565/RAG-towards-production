"""Flag-driven demo: choose a chunking strategy and inspect the result.

Examples:
    python -m chunking.cli --strategy recursive --chunk-size 256 --overlap 32
    python -m chunking.cli --strategy semantic --percentile 90 --file notes.md
    python -m chunking.cli --strategy parent_child
    python -m chunking.cli --strategy markdown --file README.md
"""

from __future__ import annotations

import argparse
import sys

from chunking.base import stats
from chunking.factory import ChunkingStrategy, get_chunker
from chunking.parent_child import ParentChildChunker

_SAMPLE = (
    "# Admissions\n\n"
    "The Computer Science program requires a minimum GPA of 3.2. Transfer students "
    "must submit official transcripts. Applications close on March 1st.\n\n"
    "## Financial Aid\n\n"
    "Need-based scholarships are available. The FAFSA deadline is February 15th. "
    "Merit awards are granted automatically based on the application.\n\n"
    "## Campus Life\n\n"
    "On-campus housing is guaranteed for first-year students. The meal plan is "
    "optional for commuters."
)


def _kwargs_for(strategy: ChunkingStrategy, args) -> dict:
    """Map generic CLI flags to the kwargs each strategy actually accepts."""
    if strategy in (ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE):
        return {"chunk_size": args.chunk_size, "chunk_overlap": args.overlap}
    if strategy is ChunkingStrategy.SENTENCE:
        return {"max_tokens": args.max_tokens, "sentence_overlap": args.sentence_overlap}
    if strategy is ChunkingStrategy.SEMANTIC:
        return {"breakpoint_percentile": args.percentile, "buffer_size": args.buffer}
    if strategy is ChunkingStrategy.MARKDOWN:
        return {"max_tokens": args.max_tokens, "overlap": args.overlap}
    if strategy is ChunkingStrategy.PARENT_CHILD:
        return {"parent_chunk_size": args.parent_size, "child_chunk_size": args.child_size,
                "child_overlap": args.overlap}
    if strategy is ChunkingStrategy.LATE:
        return {"chunk_tokens": args.chunk_size}
    # contextual / proposition / agentic use their defaults (need an LLM via OPENAI_API_KEY)
    return {}


def _print_chunks(chunks, *, limit: int, full: bool) -> None:
    for c in chunks[:limit]:
        preview = c.text if full else (c.text[:160] + ("..." if len(c.text) > 160 else ""))
        link = f"  parent={c.parent_id}" if c.parent_id else ""
        print(f"\n[{c.index}] ({c.token_count} tok){link}\n{preview}")
    if len(chunks) > limit:
        print(f"\n... {len(chunks) - limit} more")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chunking strategy demo")
    p.add_argument("--strategy", required=True, choices=[s.value for s in ChunkingStrategy])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", help="Path to a text/markdown file")
    src.add_argument("--text", help="Inline text to chunk")
    # tuning flags (used per-strategy as relevant)
    p.add_argument("--chunk-size", type=int, default=256, dest="chunk_size")
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    p.add_argument("--sentence-overlap", type=int, default=1, dest="sentence_overlap")
    p.add_argument("--percentile", type=float, default=90.0)
    p.add_argument("--buffer", type=int, default=1)
    p.add_argument("--parent-size", type=int, default=1024, dest="parent_size")
    p.add_argument("--child-size", type=int, default=256, dest="child_size")
    p.add_argument("--limit", type=int, default=10, help="Max chunks to print")
    p.add_argument("--full", action="store_true", help="Print full chunk text")
    args = p.parse_args(argv)

    try:  # ensure unicode chunk text prints on any console (e.g. Windows cp1252)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    text = args.text or (open(args.file, encoding="utf-8").read() if args.file else _SAMPLE)
    strategy = ChunkingStrategy(args.strategy)

    try:
        chunker = get_chunker(strategy, **_kwargs_for(strategy, args))
    except Exception as e:
        print(f"Failed to build '{strategy.value}' chunker: {e}", file=sys.stderr)
        return 1

    print(f"\n=== Strategy: {strategy.value} ===")

    try:
        # Parent-child has two granularities — show both.
        if isinstance(chunker, ParentChildChunker):
            parents, children = chunker.chunk_hierarchy(text)
            print(f"Parents: {len(parents)}  |  Children: {len(children)}")
            print(f"\n-- parents -- {stats(parents)}")
            _print_chunks(parents, limit=args.limit, full=args.full)
            print(f"\n-- children -- {stats(children)}")
            _print_chunks(children, limit=args.limit, full=args.full)
            return 0

        chunks = chunker.chunk(text)
    except Exception as e:
        print(
            f"'{strategy.value}' could not run: {type(e).__name__}: {e}\n"
            f"  - semantic → needs sentence-transformers + numpy\n"
            f"  - late → needs transformers + torch + numpy\n"
            f"  - contextual/proposition/agentic → need an LLM (set OPENAI_API_KEY)",
            file=sys.stderr,
        )
        return 1

    print(f"Stats: {stats(chunks)}")
    _print_chunks(chunks, limit=args.limit, full=args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
