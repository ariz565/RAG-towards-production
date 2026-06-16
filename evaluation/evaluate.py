"""Run a metric suite over a dataset → per-sample + aggregate scores."""

from __future__ import annotations

from evaluation.base import Metric, Sample


def evaluate(samples: list[Sample], metrics: list[Metric]) -> dict:
    per_sample: list[dict] = []
    sums: dict[str, float] = {m.name: 0.0 for m in metrics}

    for s in samples:
        row: dict[str, float] = {}
        for m in metrics:
            res = m.score(s)
            row[m.name] = res.score
            sums[m.name] += res.score
        per_sample.append(row)

    n = len(samples) or 1
    aggregate = {name: round(total / n, 3) for name, total in sums.items()}
    return {"per_sample": per_sample, "aggregate": aggregate, "n": len(samples)}
