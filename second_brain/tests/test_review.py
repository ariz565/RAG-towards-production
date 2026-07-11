from pathlib import Path

import pytest

from second_brain.review import ReviewQueue, UnknownUpdate


def _queue(tmp_path: Path) -> ReviewQueue:
    return ReviewQueue(tmp_path)


def test_propose_lands_in_pending(tmp_path):
    q = _queue(tmp_path)
    update = q.propose("page", "ingest", {"path": "concepts/a.md"}, "because reasons")
    assert update.status == "pending"
    pending = q.list_pending()
    assert len(pending) == 1
    assert pending[0].id == update.id
    assert pending[0].rationale == "because reasons"


def test_approve_moves_out_of_pending(tmp_path):
    q = _queue(tmp_path)
    update = q.propose("graph_link", "ask", {"src": "a", "dst": "b", "relation": "is-a"}, "")
    q.mark_applied(update.id)
    assert q.list_pending() == []
    with pytest.raises(UnknownUpdate):
        q.get_pending(update.id)


def test_reject_records_reason(tmp_path):
    q = _queue(tmp_path)
    update = q.propose("page", "ingest", {"path": "concepts/a.md"}, "")
    q.mark_rejected(update.id, "hallucinated, no source for this")
    assert q.list_pending() == []
    rejected_files = list((tmp_path / ".review" / "rejected").glob("*.json"))
    assert len(rejected_files) == 1
    assert "hallucinated" in rejected_files[0].read_text(encoding="utf-8")


def test_unknown_id_raises(tmp_path):
    q = _queue(tmp_path)
    with pytest.raises(UnknownUpdate):
        q.get_pending("does-not-exist")


def test_multiple_proposals_independent(tmp_path):
    q = _queue(tmp_path)
    a = q.propose("page", "ingest", {"path": "concepts/a.md"}, "")
    b = q.propose("page", "ingest", {"path": "concepts/b.md"}, "")
    q.mark_applied(a.id)
    pending_ids = {u.id for u in q.list_pending()}
    assert pending_ids == {b.id}
