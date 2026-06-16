"""Run the golden set through the pipeline and report deterministic metrics.

This is the fast, judge-free pass: retrieval page recall/precision, out-of-scope
routing accuracy, refusal correctness, latency and tokens. It also saves the
predictions to ``results/latest.json`` so the DeepEval CI gate can score the same
run without re-invoking the pipeline.

Usage:
    python -m evals.run_evals                  # active strategy
    python -m evals.run_evals --strategy hybrid
    python -m evals.run_evals --strategy pageindex
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from evals.harness import RESULTS_DIR, bootstrap, load_golden_set, run_case

logging.basicConfig(level=logging.WARNING, format="%(levelname)s │ %(name)s │ %(message)s")


def _page_metrics(expected: list[int], retrieved: list[int]) -> tuple[float, float]:
    """Return (recall, precision) of retrieved pages vs expected pages."""
    if not expected:
        return (1.0, 1.0)
    exp, ret = set(expected), set(retrieved)
    hit = len(exp & ret)
    recall = hit / len(exp) if exp else 1.0
    precision = hit / len(ret) if ret else 0.0
    return (recall, precision)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG golden-set evaluation")
    parser.add_argument("--strategy", default=None, help="pageindex | hybrid | bm25_only | vector_only")
    parser.add_argument("--out", default=str(RESULTS_DIR / "latest.json"))
    args = parser.parse_args()

    await bootstrap()
    golden = load_golden_set()

    started = time.time()
    predictions: list[dict] = []
    for case in golden:
        pred = await run_case(case, strategy=args.strategy)
        predictions.append(pred)

    # ── Deterministic metrics ────────────────────────────────────
    in_scope = [p for p in predictions if p["type"] == "in_scope"]
    out_scope = [p for p in predictions if p["type"] == "out_of_scope"]

    recalls, precisions = [], []
    for p in in_scope:
        r, pr = _page_metrics(p["expected_pages"], p["retrieved_page_numbers"])
        p["page_recall"], p["page_precision"] = r, pr
        recalls.append(r)
        precisions.append(pr)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    oos_correct = sum(1 for p in out_scope if p["out_of_scope"])
    oos_accuracy = oos_correct / len(out_scope) if out_scope else 1.0
    # In-scope answers should NOT be wrongly refused by the guardrail.
    in_scope_not_refused = sum(1 for p in in_scope if not p["out_of_scope"])
    answered_rate = in_scope_not_refused / len(in_scope) if in_scope else 1.0
    avg_latency = sum(p["total_duration_ms"] for p in predictions) / max(len(predictions), 1)
    total_tokens = sum(p["total_tokens"] for p in predictions)

    # Verification spine (Phase B) signals
    grounded_rate = (
        sum(1 for p in in_scope if p.get("grounded")) / len(in_scope) if in_scope else 0.0
    )
    avg_confidence = (
        sum(p.get("confidence", 0.0) for p in in_scope) / len(in_scope) if in_scope else 0.0
    )
    refused_count = sum(1 for p in in_scope if p.get("refused"))

    summary = {
        "strategy": args.strategy or "active-default",
        "cases": len(predictions),
        "in_scope": len(in_scope),
        "out_of_scope": len(out_scope),
        "avg_page_recall": round(avg_recall, 3),
        "avg_page_precision": round(avg_precision, 3),
        "out_of_scope_accuracy": round(oos_accuracy, 3),
        "in_scope_answered_rate": round(answered_rate, 3),
        "grounded_rate": round(grounded_rate, 3),
        "avg_grounding_confidence": round(avg_confidence, 3),
        "refused_in_scope": refused_count,
        "avg_latency_ms": round(avg_latency, 1),
        "total_tokens": total_tokens,
        "wall_seconds": round(time.time() - started, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=False)

    # ── Report ───────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  RAG Golden-Set Evaluation — deterministic pass")
    print("═" * 64)
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    print("═" * 64)
    print("  Per-case page recall:")
    for p in in_scope:
        flag = "✓" if p["page_recall"] >= 0.5 else "✗"
        print(f"   {flag} {p['id']}  recall={p['page_recall']:.2f}  "
              f"pages={p['retrieved_page_numbers']} (exp {p['expected_pages']})")
    print(f"\n  Predictions saved to: {args.out}\n")

    # Non-zero exit if retrieval is clearly broken (useful as a smoke gate).
    return 0 if avg_recall > 0 and oos_accuracy >= 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
