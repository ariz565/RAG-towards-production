"""Benchmark the MRL + quantization trade-off (storage vs retrieval quality).

For a tiny built-in corpus, embed with the offline hash provider at the full
dimension (float32) as the baseline ranking, then measure how well truncated
dimensions (Matryoshka) and quantized dtypes preserve the **top-1 retrieval**
result — alongside the per-vector storage. This is the empirical version of the
"80% storage cut with minimal recall loss" claim.

Usage:
    python -m embeddings.benchmark
    python -m embeddings.benchmark --provider sentence_transformers --model BAAI/bge-small-en-v1.5
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

_CORPUS = [
    "The Computer Science program requires a minimum GPA of 3.2.",
    "Transfer students must submit official transcripts by March 1st.",
    "Need-based scholarships are available via the FAFSA, due February 15th.",
    "On-campus housing is guaranteed for first-year students.",
    "The library is open 24 hours during final exam weeks.",
    "International applicants must submit TOEFL or IELTS scores.",
]
_QUERIES = [
    "minimum GPA for computer science",
    "scholarship and financial aid deadline",
    "do international students need english test scores",
    "where do first year students live",
]


def _top1(query_vec, doc_vecs, score_fn) -> int:
    return max(range(len(doc_vecs)), key=lambda i: score_fn(query_vec, doc_vecs[i]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MRL + quantization benchmark")
    p.add_argument("--provider", default="hash", choices=[e.value for e in EmbeddingProvider])
    p.add_argument("--model", help="Model for non-hash providers")
    p.add_argument("--prefix", default="none", choices=["none", "e5", "nomic", "bge"])
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    kwargs = {} if args.provider == "hash" else ({"model_name": args.model} if args.model else {})

    def embed(dim):
        cfg = EmbedderConfig(normalize=True, dimensions=dim, prefix_preset=args.prefix)
        emb = get_embedder(args.provider, config=cfg, **kwargs)
        return emb.embed_query, emb.embed_documents

    # Baseline: full dimension, float32.
    try:
        q_full, d_full = embed(None)
        base_docs = d_full(_CORPUS)
        base_q = [q_full(q) for q in _QUERIES]
    except Exception as e:
        print(f"provider '{args.provider}' failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    full_dim = len(base_docs[0])
    baseline_top1 = [_top1(qv, base_docs, cosine) for qv in base_q]

    dims = [d for d in (full_dim, full_dim // 2, full_dim // 4) if d >= 8]
    rows = []

    for dim in dims:
        qfn, dfn = embed(dim)
        dvs = dfn(_CORPUS)
        qvs = [qfn(q) for q in _QUERIES]

        # float32
        rows.append(_row("float32", dim, qvs, dvs, baseline_top1, cosine))
        # int8
        mins, maxs = calibrate_int8(dvs)
        dq = [dequantize_int8(quantize_int8(v, mins, maxs), mins, maxs) for v in dvs]
        qq = [dequantize_int8(quantize_int8(v, mins, maxs), mins, maxs) for v in qvs]
        rows.append(_row("int8", dim, qq, dq, baseline_top1, cosine))
        # binary (Hamming → negate so "higher = better")
        db = [quantize_binary(v) for v in dvs]
        qb = [quantize_binary(v) for v in qvs]
        rows.append(_row("binary", dim, qb, db, baseline_top1,
                         lambda a, b: -hamming(a, b)))

    _print(rows, full_dim)
    return 0


def _row(dtype, dim, qvs, dvs, baseline_top1, score_fn) -> dict:
    matches = sum(1 for i, qv in enumerate(qvs)
                  if _top1(qv, dvs, score_fn) == baseline_top1[i])
    return {
        "dtype": dtype,
        "dim": dim,
        "bytes": storage_bytes(dim, dtype),
        "top1_match": round(matches / len(qvs), 3),
    }


def _print(rows, full_dim) -> None:
    base_bytes = storage_bytes(full_dim, "float32")
    print(f"\nBaseline: {full_dim}-d float32 = {base_bytes} B/vector\n")
    print("dtype    dim     bytes/vec   storage_cut   top1_match_vs_baseline")
    print("-" * 64)
    for r in rows:
        cut = f"{(1 - r['bytes'] / base_bytes) * 100:5.1f}%"
        print(f"{r['dtype']:<8} {r['dim']:<7} {r['bytes']:<11} {cut:<13} {r['top1_match']}")
    print("\nNote: MRL truncation only preserves quality on MRL-TRAINED models "
          "(OpenAI-3, BGE-M3, Nomic-v2, Gemini). The offline hash provider is NOT "
          "MRL-trained, so truncation degrades here — that's the point.")
    print("Rule of thumb: keep the cheapest representation that holds top-k recall on YOUR evals.\n")


if __name__ == "__main__":
    raise SystemExit(main())
