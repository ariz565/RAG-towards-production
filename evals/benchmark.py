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

from evals import attribution
from evals.harness import RESULTS_DIR, bootstrap, load_golden_set, run_case
from retrieval.base import hit_at_k, mrr, ndcg_at_k

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
    hits, mrrs, ndcgs = [], [], []
    verdicts: list[str] = []
    per_case: list[dict] = []
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

        # hit_rate/MRR/nDCG need rank order, which retrieved_page_numbers (a sorted
        # set) discards — ranked_page_numbers preserves it. Reuses retrieval/base.py's
        # formulas rather than re-deriving them for the live harness.
        expected = set(case["expected_pages"])
        ranked = pred.get("ranked_page_numbers") or pred["retrieved_page_numbers"]
        if expected:
            hits.append(hit_at_k(ranked, expected, len(ranked) or 1))
            mrrs.append(mrr(ranked, expected))
            ndcgs.append(ndcg_at_k(ranked, expected, len(ranked) or 1))

        verdict = attribution.classify_case(
            case_type="in_scope", expected_pages=case["expected_pages"],
            retrieved_pages=pred["retrieved_page_numbers"], grounded=pred["grounded"],
            unsupported_claims=pred["unsupported_claims"], out_of_scope=False,
        )
        verdicts.append(verdict)
        per_case.append({"id": case["id"], "question": case["question"], "verdict": verdict,
                          "expected_pages": case["expected_pages"],
                          "retrieved_pages": pred["retrieved_page_numbers"]})

    oos_ok = 0
    for case in out_scope:
        pred = await run_case(case, strategy=strategy)
        oos_ok += 1 if pred["out_of_scope"] else 0
        latencies.append(pred["total_duration_ms"])
        tokens += pred["total_tokens"]

        verdict = attribution.classify_case(
            case_type="out_of_scope", expected_pages=[], retrieved_pages=[],
            grounded=pred["grounded"], unsupported_claims=[], out_of_scope=pred["out_of_scope"],
        )
        verdicts.append(verdict)
        per_case.append({"id": case["id"], "question": case["question"], "verdict": verdict,
                          "expected_pages": [], "retrieved_pages": []})

    n = len(in_scope) or 1
    m = len(hits) or 1  # cases with at least one expected page (excludes n/a cases)
    return {
        "strategy": strategy,
        "page_recall": round(sum(recalls) / n, 3),
        "page_precision": round(sum(precisions) / n, 3),
        "hit_rate": round(sum(hits) / m, 3),
        "mrr": round(sum(mrrs) / m, 3),
        "ndcg": round(sum(ndcgs) / m, 3),
        "grounded_rate": round(grounded / n, 3),
        "avg_confidence": round(sum(confs) / n, 3),
        "out_of_scope_accuracy": round(oos_ok / len(out_scope), 3) if out_scope else 1.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "total_tokens": tokens,
        "wall_seconds": round(time.time() - started, 1),
        # Retrieval-vs-generation attribution: how many failures are retrieval's
        # fault (right pages never found) vs. generation's (right pages found,
        # answer still wrong/ungrounded) — see attribution.py.
        "attribution": attribution.summarize(verdicts),
        "cases": per_case,
    }


_COLUMNS = [
    ("strategy", "Strategy"),
    ("page_recall", "Page Recall"),
    ("page_precision", "Page Prec."),
    ("hit_rate", "Hit Rate"),
    ("mrr", "MRR"),
    ("ndcg", "nDCG"),
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
            blanks = " | ".join("" for _ in range(len(_COLUMNS) - 2))
            lines.append(f"| {row['strategy']} | error: {row['error']} | {blanks} |")
            continue
        cells = [str(row.get(key, "")) for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Retrieval-vs-generation attribution")
    lines.append("")
    lines.append("_Is a bad answer retrieval's fault or generation's? retrieval_miss = the "
                  "right pages were never found; generation_miss = they were found and the "
                  "answer was still wrong/ungrounded._")
    lines.append("")
    for row in rows:
        attr = row.get("attribution")
        if not attr:
            continue
        lines.append(f"**{row['strategy']}**: " + ", ".join(f"{k}={v}" for k, v in attr.items() if v))
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
