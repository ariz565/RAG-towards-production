"""End-to-end tests against OfflineLLMAdapter -- zero API key, zero network.

Uses plain `def test_x(): asyncio.run(...)` rather than `async def test_x()`,
because this repo's pytest config sets `asyncio_mode = "auto"` (needs the
pytest-asyncio plugin) but that plugin isn't installed in every environment this
module might run in -- and per the module's own "zero new dependencies" rule, we
don't get to require it just for our own tests either.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from second_brain.adapters import OfflineLLMAdapter
from second_brain.pipeline import SecondBrain


def _run(coro):
    return asyncio.run(coro)


def _brain(tmp_path: Path, auto_approve: bool = False) -> SecondBrain:
    return SecondBrain(OfflineLLMAdapter(), tmp_path, auto_approve=auto_approve)


def test_ingest_creates_page_and_is_idempotent(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    (brain.store.raw_dir / "note.md").write_text("Widgets are small reusable UI components.")

    results = _run(brain.ingest())
    assert len(results) == 1
    assert results[0]["status"] == "ingested"
    assert results[0]["pages"][0]["applied"] is True
    assert brain.store.list_pages() == ["concepts/note.md"]
    assert "Widgets are small reusable UI components." in brain.store.read_page_body("concepts/note.md")

    # Re-running ingest on an unchanged raw/ file must not re-propose it.
    results2 = _run(brain.ingest())
    assert results2[0]["status"] == "skipped_unchanged"


def test_ingest_without_auto_approve_queues_for_review(tmp_path):
    brain = _brain(tmp_path, auto_approve=False)
    (brain.store.raw_dir / "gizmo.md").write_text("Gizmos are small mechanical devices.")

    results = _run(brain.ingest())
    assert results[0]["pages"][0]["applied"] is False
    assert brain.store.list_pages() == []  # nothing hits disk until approved

    pending = brain.review.list_pending()
    assert len(pending) == 1

    outcome = brain.apply_pending(pending[0].id)
    assert outcome["applied"] is True
    assert brain.store.list_pages() == ["concepts/gizmo.md"]


def test_reject_pending_leaves_wiki_untouched(tmp_path):
    brain = _brain(tmp_path, auto_approve=False)
    (brain.store.raw_dir / "note.md").write_text("Some content.")
    _run(brain.ingest())
    update_id = brain.review.list_pending()[0].id

    outcome = brain.reject_pending(update_id, "not trustworthy")
    assert outcome["rejected"] is True
    assert brain.store.list_pages() == []
    assert brain.review.list_pending() == []


def test_ask_offline_declines_write_back_honestly(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    (brain.store.raw_dir / "widgets.md").write_text(
        "Widgets are small reusable UI components used across the app."
    )
    _run(brain.ingest())

    result = _run(brain.ask("what are widgets?"))
    assert result["pages_read"] == ["concepts/widgets.md"]
    assert result["write_back_ok"] is True
    assert result["nothing_to_file_reason"]  # offline adapter always declines, honestly
    assert result["retried_for_rule_6"] is False
    assert "concepts/widgets.md" in result["answer"]


def test_ask_with_no_matching_pages_still_satisfies_rule_6(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    result = _run(brain.ask("completely unrelated question about astrophysics"))
    assert result["pages_read"] == []
    assert result["write_back_ok"] is True
    assert result["nothing_to_file_reason"]


def test_lint_reports_orphans_and_narrative(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    for name, text in [("a.md", "Alpha concept text."), ("b.md", "Beta concept text.")]:
        (brain.store.raw_dir / name).write_text(text)
    _run(brain.ingest())

    report = _run(brain.lint())
    # offline adapter never proposes graph links, so every ingested page is an orphan node
    assert report["stats"]["orphans"] == 2
    assert report["stats"]["contradictions"] == 0
    assert "orphan" in report["narrative"].lower()
    assert Path(report["report_path"]).exists()


def test_index_is_regenerated_after_ingest(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    (brain.store.raw_dir / "note.md").write_text("Some concept content here.")
    _run(brain.ingest())
    assert "[[concepts/note.md]]" in brain.store.read_index()


def test_context_manager_closes_the_graph_connection(tmp_path):
    """Regression test: GraphStore opens a sqlite connection that nothing used
    to close. On Windows this fails to delete the directory afterward while the
    handle is still open -- caught via a live verification script, not by a
    tmp_path-based test alone, since pytest's own cleanup is more forgiving."""
    with SecondBrain(OfflineLLMAdapter(), tmp_path, auto_approve=True) as brain:
        (brain.store.raw_dir / "note.md").write_text("content")
        _run(brain.ingest())
    # after __exit__, the connection is closed -- a further query must fail
    import sqlite3
    try:
        brain.graph._conn.execute("SELECT 1")
        raised = False
    except sqlite3.ProgrammingError:
        raised = True
    assert raised


def test_index_recent_changes_reflects_the_operation_just_run(tmp_path):
    """Regression test: append_log() must run BEFORE _regenerate_index() in every
    operation, or the "Recent changes" section is always one entry stale (caught
    via manual end-to-end testing, not by the simpler test above)."""
    brain = _brain(tmp_path, auto_approve=True)
    (brain.store.raw_dir / "first.md").write_text("First concept.")
    (brain.store.raw_dir / "second.md").write_text("Second concept.")
    _run(brain.ingest())

    index_text = brain.store.read_index()
    assert "ingest | second.md" in index_text  # the most recent ingest, not just the first

    _run(brain.ask("something with no matches"))
    assert "ask | something with no matches" in brain.store.read_index()

    _run(brain.lint())
    assert "lint | health check" in brain.store.read_index()


def test_apply_page_reports_sandbox_violation_without_raising(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    error = brain._apply_page({"path": "not_allowed.md", "title": "x", "content": "x"}, None)
    assert error is not None


def test_apply_link_reports_invalid_relation_without_raising(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    error = brain._apply_link({"src": "a", "dst": "b", "relation": "loves"})
    assert error is not None


def test_audit_log_records_every_operation(tmp_path):
    brain = _brain(tmp_path, auto_approve=True)
    (brain.store.raw_dir / "note.md").write_text("Some content.")
    _run(brain.ingest())
    _run(brain.ask("what is in the notes?"))
    _run(brain.lint())

    audit_text = brain.audit_log.read_text(encoding="utf-8")
    events = [line for line in audit_text.splitlines() if line.strip()]
    kinds = {json.loads(e)["event"] for e in events}
    assert {"ingest", "ask", "lint"} <= kinds
