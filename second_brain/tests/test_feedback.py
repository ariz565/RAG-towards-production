from pathlib import Path

from second_brain.feedback import recent_rejections, render_lessons
from second_brain.review import ReviewQueue


def _queue(tmp_path: Path) -> ReviewQueue:
    return ReviewQueue(tmp_path)


def test_render_lessons_empty_when_nothing_rejected(tmp_path):
    assert render_lessons([]) == ""


def test_recent_rejections_excludes_pending_and_applied(tmp_path):
    q = _queue(tmp_path)
    pending = q.propose("page", "ingest", {"path": "concepts/a.md"}, "")
    applied = q.propose("page", "ingest", {"path": "concepts/b.md"}, "")
    rejected = q.propose("page", "ingest", {"path": "concepts/c.md"}, "")
    q.mark_applied(applied.id)
    q.mark_rejected(rejected.id, "duplicate of an existing page")

    rejections = recent_rejections(q)
    assert [u.id for u in rejections] == [rejected.id]
    assert pending.id not in [u.id for u in rejections]


def test_recent_rejections_respects_limit_and_recency(tmp_path):
    q = _queue(tmp_path)
    ids = []
    for i in range(7):
        u = q.propose("page", "ingest", {"path": f"concepts/{i}.md"}, "")
        q.mark_rejected(u.id, f"reason {i}")
        ids.append(u.id)

    top3 = recent_rejections(q, limit=3)
    assert [u.id for u in top3] == list(reversed(ids))[:3]


def test_render_lessons_includes_page_summary_and_reason(tmp_path):
    q = _queue(tmp_path)
    u = q.propose(
        "page", "ingest",
        {"path": "concepts/chunking-notes.md", "title": "Chunking Notes"},
        "ingested verbatim",
    )
    q.mark_rejected(u.id, "duplicated an existing page, should have updated it instead")

    text = render_lessons(recent_rejections(q))
    assert "concepts/chunking-notes.md" in text
    assert "Chunking Notes" in text
    assert "duplicated an existing page" in text
    assert "REJECTED" in text


def test_render_lessons_includes_link_summary(tmp_path):
    q = _queue(tmp_path)
    u = q.propose(
        "graph_link", "ask",
        {"src": "concepts/a", "dst": "concepts/b", "relation": "contradicts"},
        "flagged a disagreement",
    )
    q.mark_rejected(u.id, "these don't actually contradict, misread the source")

    text = render_lessons(recent_rejections(q))
    assert "concepts/a --contradicts--> concepts/b" in text
    assert "misread the source" in text


def test_render_lessons_handles_missing_reason(tmp_path):
    q = _queue(tmp_path)
    u = q.propose("page", "ingest", {"path": "concepts/a.md"}, "")
    q.mark_rejected(u.id)  # no reason given

    text = render_lessons(recent_rejections(q))
    assert "(no reason given)" in text
