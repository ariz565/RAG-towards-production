"""Flag-driven embeddings demo: embed a query + docs, rank, and show the effect
of prefixes, MRL truncation, and quantization.

Examples:
    python -m embeddings.cli                                  # offline (hash provider)
    python -m embeddings.cli --prefix e5 --dim 128
    python -m embeddings.cli --quantize int8
    python -m embeddings.cli --quantize binary
    python -m embeddings.cli --provider sentence_transformers --model BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import sys

from embeddings.base import EmbedderConfig, cosine
from embeddings.factory import EmbeddingProvider, get_embedder
from embeddings.quantize import (
    calibrate_int8,
    dequantize_int8,
    hamming,
    quantize_binary,
    quantize_int8,
    storage_bytes,
)

_QUERY = "What GPA do I need for the computer science program?"
_DOCS = [
    "The Computer Science program requires a minimum GPA of 3.2 for admission.",
    "Need-based scholarships are available; the FAFSA deadline is February 15th.",
    "On-campus housing is guaranteed for all first-year students.",
]


def _provider_kwargs(provider: EmbeddingProvider, args) -> dict:
    if provider is EmbeddingProvider.HASH:
        return {"hash_dim": args.hash_dim}
    return {"model_name": args.model} if args.model else {}


def _rank(scores: list[tuple[int, float]]):
    return sorted(scores, key=lambda x: x[1], reverse=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Embeddings demo")
    p.add_argument("--provider", default="hash", choices=[e.value for e in EmbeddingProvider])
    p.add_argument("--model", help="Model name for ST/OpenAI/Ollama providers")
    p.add_argument("--hash-dim", type=int, default=256, dest="hash_dim")
    p.add_argument("--prefix", default="none", choices=["none", "e5", "nomic", "bge"])
    p.add_argument("--dim", type=int, default=None, help="Matryoshka truncation target")
    p.add_argument("--quantize", default="none", choices=["none", "int8", "binary"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--query", default=_QUERY)
    p.add_argument("--doc", action="append", help="Document (repeatable)")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    config = EmbedderConfig(
        normalize=not args.no_normalize,
        dimensions=args.dim,
        prefix_preset=args.prefix,
    )
    provider = EmbeddingProvider(args.provider)
    try:
        emb = get_embedder(provider, config=config, **_provider_kwargs(provider, args))
        docs = args.doc or _DOCS
        qv = emb.embed_query(args.query)
        dvs = emb.embed_documents(docs)
    except Exception as e:
        print(f"provider '{args.provider}' failed: {type(e).__name__}: {e}\n"
              f"(hash works offline; ST needs sentence-transformers; openai/ollama need a server)",
              file=sys.stderr)
        return 1

    dim = len(qv)
    print(f"\nprovider={args.provider}  prefix={args.prefix}  dim={dim}  "
          f"normalize={not args.no_normalize}  quantize={args.quantize}")
    print(f"query: {args.query}\n")

    # float32 cosine ranking
    ranked = _rank([(i, cosine(qv, dv)) for i, dv in enumerate(dvs)])
    print("-- cosine ranking (float32) --")
    for rank, (i, s) in enumerate(ranked, 1):
        print(f"  {rank}. ({s:+.3f}) {docs[i][:70]}")

    if args.quantize == "int8":
        mins, maxs = calibrate_int8(dvs + [qv])
        qd = [dequantize_int8(quantize_int8(dv, mins, maxs), mins, maxs) for dv in dvs]
        qq = dequantize_int8(quantize_int8(qv, mins, maxs), mins, maxs)
        rq = _rank([(i, cosine(qq, d)) for i, d in enumerate(qd)])
        print("\n-- ranking after int8 quantization --")
        for rank, (i, s) in enumerate(rq, 1):
            print(f"  {rank}. ({s:+.3f}) {docs[i][:70]}")

    if args.quantize == "binary":
        qb = quantize_binary(qv)
        rb = _rank([(i, -hamming(qb, quantize_binary(dv))) for i, dv in enumerate(dvs)])
        print("\n-- ranking after binary quantization (−Hamming) --")
        for rank, (i, s) in enumerate(rb, 1):
            print(f"  {rank}. (dist {-s}) {docs[i][:70]}")

    print(f"\nstorage / vector @ dim={dim}:  "
          f"float32={storage_bytes(dim,'float32')}B  "
          f"int8={storage_bytes(dim,'int8')}B  "
          f"binary={storage_bytes(dim,'binary')}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
