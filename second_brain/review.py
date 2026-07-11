"""The epistemic-integrity mechanism.

Both source documents converge on the same critique. karpathy-second-brain.md's
community insight #2 puts it bluntly: an LLM that both writes the knowledge base
*and* grades its own writing, with no independent check, eventually produces a
"very convincing but low-integrity" wiki — the worst kind, because you trust it.
graphify's own save-result/reflect loop has exactly this shape (confirmed by
reading its source): the same agent that answers a question also self-labels that
answer `useful`, with no review gate, and after two self-marks it becomes a
"preferred source" other sessions are told to trust.

This module is the fix, applied here rather than left as a known gap: every page
write and every graph link proposed by pipeline.py's ingest()/ask() lands here
first, as a pending item on disk. Nothing touches wiki/*.md or graph.db until
pipeline.py explicitly applies it — either immediately, if the operator opted
into `--auto-approve` (a real, named decision, not a silent default), or later,
via `second_brain review approve <id>`, which is a human in the loop by
construction.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_STATES = ("pending", "applied", "rejected")


@dataclass
class ProposedUpdate:
    id: str
    kind: str          # "page" | "graph_link"
    operation: str      # "ingest" | "ask" — which pipeline stage proposed this
    payload: dict
    rationale: str
    created_at: str
    status: str = "pending"
    resolution_note: str = ""


class UnknownUpdate(KeyError):
    pass


class ReviewQueue:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self._dirs = {state: self.base_dir / ".review" / state for state in _STATES}
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def propose(self, kind: str, operation: str, payload: dict, rationale: str) -> ProposedUpdate:
        update = ProposedUpdate(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            operation=operation,
            payload=payload,
            rationale=rationale,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write(update)
        return update

    def _path(self, update: ProposedUpdate) -> Path:
        return self._dirs[update.status] / f"{update.id}.json"

    def _write(self, update: ProposedUpdate) -> None:
        self._path(update).write_text(json.dumps(asdict(update), indent=2), encoding="utf-8")

    def list_pending(self) -> list[ProposedUpdate]:
        items = []
        for f in sorted(self._dirs["pending"].glob("*.json")):
            items.append(ProposedUpdate(**json.loads(f.read_text(encoding="utf-8"))))
        return items

    def get_pending(self, update_id: str) -> ProposedUpdate:
        f = self._dirs["pending"] / f"{update_id}.json"
        if not f.exists():
            raise UnknownUpdate(f"no pending update with id {update_id!r}")
        return ProposedUpdate(**json.loads(f.read_text(encoding="utf-8")))

    def list_rejected(self, limit: int | None = None) -> list[ProposedUpdate]:
        """Most recent first. Sorted by the `created_at` field, not by filename --
        filenames are random uuid hex, not chronological, so glob() order alone
        would silently return an arbitrary-looking recency ordering."""
        items = [
            ProposedUpdate(**json.loads(f.read_text(encoding="utf-8")))
            for f in self._dirs["rejected"].glob("*.json")
        ]
        items.sort(key=lambda u: u.created_at, reverse=True)
        return items[:limit] if limit is not None else items

    def _transition(self, update: ProposedUpdate, new_status: str, note: str = "") -> ProposedUpdate:
        old_path = self._path(update)
        update.status = new_status
        update.resolution_note = note
        old_path.unlink(missing_ok=True)
        self._write(update)
        return update

    def mark_applied(self, update_id: str) -> ProposedUpdate:
        return self._transition(self.get_pending(update_id), "applied")

    def mark_rejected(self, update_id: str, reason: str = "") -> ProposedUpdate:
        return self._transition(self.get_pending(update_id), "rejected", reason)
