"""Eval harness bootstrap — load indexes + domain profile, then run queries.

Mirrors the server's startup so evals can run the *real* pipeline from a script
or pytest, without launching FastAPI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings
from app.services.pipeline import ask
from app.services.registry import registry

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

_bootstrapped = False


def load_golden_set(path: Path | str = GOLDEN_SET_PATH) -> list[dict]:
    """Load and validate the golden set JSONL."""
    path = Path(path)
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "id" in obj, f"line {line_no}: missing 'id'"
            assert obj.get("type") in ("in_scope", "out_of_scope"), f"line {line_no}: bad 'type'"
            assert obj.get("question"), f"line {line_no}: missing 'question'"
            if obj["type"] == "in_scope":
                assert obj.get("expected_answer"), f"line {line_no}: in_scope needs 'expected_answer'"
            items.append(obj)
    return items


async def bootstrap() -> None:
    """Load all document bundles from disk via the registry (idempotent)."""
    global _bootstrapped
    if _bootstrapped:
        return
    count = await registry.load_all()
    logger.info(f"Eval harness bootstrap: {count} document(s) loaded.")
    _bootstrapped = True


async def run_case(case: dict, *, strategy: str | None = None) -> dict:
    """Run one golden-set case through the pipeline and return a prediction record."""
    res = await ask(case["question"], strategy=strategy)
    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "expected_answer": case.get("expected_answer", ""),
        "expected_pages": case.get("expected_pages", []),
        "actual_output": res.get("answer", ""),
        "retrieval_context": res.get("retrieval_context", []),
        "retrieved_page_numbers": res.get("retrieved_page_numbers", []),
        "ranked_page_numbers": res.get("ranked_page_numbers", []),
        "out_of_scope": res.get("out_of_scope", False),
        "grounded": res.get("grounded", False),
        "refused": res.get("refused", False),
        "unsupported_claims": res.get("unsupported_claims", []),
        "confidence": res.get("confidence", 0.0),
        "strategy_used": res.get("strategy_used", strategy or settings.active_retrieval.value),
        "total_tokens": res.get("total_tokens", 0),
        "total_duration_ms": res.get("total_duration_ms", 0.0),
    }
