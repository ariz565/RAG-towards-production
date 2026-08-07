"""Read-only bridge into evals/results/ -- lets vision-ui render the golden-set
benchmark (evals/benchmark.py) without re-running it, same reasoning as
app/routers/second_brain.py: operator-facing eval output, not per-tenant
document data, so no auth dependency and an honest empty state instead of an
error when the benchmark hasn't been run yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/evals", tags=["evals"])

RESULTS_PATH = Path("evals/results/benchmark.json")


@router.get("/benchmark")
def get_benchmark() -> dict:
    if not RESULTS_PATH.exists():
        return {"available": False, "rows": []}
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"available": True, "rows": rows}
