"""Strategy benchmark (Phase E) — PageIndex vs Hybrid vs BM25 vs Vector.

Runs the golden set through every retrieval strategy and writes a reproducible
markdown + JSON report. This is the citable artifact: an apples-to-apples
comparison (accuracy, grounding, latency, cost) backed by the eval harness —
the thing most "RAG demo" repos can't produce.

Usage:
    cd agent-backend/admission-guide
    python -m evals.benchmark
"""

from __future__ import annotations

import asyncio
import json
import time

from evals.harness import RESULTS_DIR, bootstrap, load_golden_set, run_case

STRATEGIES = ["pageindex", "hybrid", "bm25_only", "vector_only"]


def _page_metrics(expected: list[int], retrieved: list[int]) -> tuple[float, float]:
    if not expected:
        return (1.0, 1.0)
    exp, ret = set(expected), set(retrieved)
    hit = len(exp & ret)
    return (hit / len(exp) if exp else 1.0, hit / len(ret) if ret else 0.0)


async def _bench_strategy(strategy: str, golden: list[dict]) -> dict:
    in_scope = [c for c in golden if c["type"] == "in_scope"]
    out_scope = [c for c in golden if c["type"] == "out_of_scope"]

    recalls, precisions, confs, latencies = [], [], [], []
    grounded = 0
    tokens = 0
    started = time.time()

    for case in in_scope:
        pred = await run_case(case, strategy=strategy)
        recall, precision = _page_metrics(case["expected_pages"], pred["retrieved_page_numbers"])
        recalls.append(recall)
        precisions.append(precision)
        confs.append(pred["confidence"])
        grounded += 1 if pred["grounded"] else 0
        latencies.append(pred["total_duration_ms"])
        tokens += pred["total_tokens"]

    oos_ok = 0
    for case in out_scope:
        pred = await run_case(case, strategy=strategy)
        oos_ok += 1 if pred["out_of_scope"] else 0
        latencies.append(pred["total_duration_ms"])
        tokens += pred["total_tokens"]

    n = len(in_scope) or 1
    return {
        "strategy": strategy,
        "page_recall": round(sum(recalls) / n, 3),
        "page_precision": round(sum(precisions) / n, 3),
        "grounded_rate": round(grounded / n, 3),
        "avg_confidence": round(sum(confs) / n, 3),
        "out_of_scope_accuracy": round(oos_ok / len(out_scope), 3) if out_scope else 1.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "total_tokens": tokens,
        "wall_seconds": round(time.time() - started, 1),
    }


_COLUMNS = [
    ("strategy", "Strategy"),
    ("page_recall", "Page Recall"),
    ("page_precision", "Page Prec."),
    ("grounded_rate", "Grounded"),
    ("avg_confidence", "Avg Conf."),
    ("out_of_scope_accuracy", "OOS Acc."),
    ("avg_latency_ms", "Latency (ms)"),
    ("total_tokens", "Tokens"),
]


def _to_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    lines = [
        "# Retrieval Strategy Benchmark",
        "",
        "_Golden-set comparison across retrieval strategies. Reproduce with "
        "`python -m evals.benchmark`._",
        "",
        header,
        sep,
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['strategy']} | error: {row['error']} | | | | | | |")
            continue
        cells = [str(row.get(key, "")) for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    await bootstrap()
    golden = load_golden_set()

    rows: list[dict] = []
    for strategy in STRATEGIES:
        print(f"  Benchmarking strategy: {strategy} ...")
        try:
            rows.append(await _bench_strategy(strategy, golden))
        except Exception as e:  # a strategy with no index, etc.
            rows.append({"strategy": strategy, "error": str(e)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    markdown = _to_markdown(rows)
    (RESULTS_DIR / "benchmark.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"  Saved → {RESULTS_DIR / 'benchmark.md'} and benchmark.json\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
