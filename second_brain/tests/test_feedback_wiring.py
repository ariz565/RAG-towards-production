"""Proves the feedback loop is actually wired end-to-end, not just correct in
isolation -- test_feedback.py already covers feedback.py's own logic; this file
proves pipeline.py actually calls it, and calls it only where it's supposed to.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from second_brain.pipeline import SecondBrain
from second_brain.tests._helpers import SpyLLMAdapter


def _run(coro):
    return asyncio.run(coro)


def test_no_lessons_block_when_nothing_has_ever_been_rejected(tmp_path: Path):
    spy = SpyLLMAdapter()
    brain = SecondBrain(spy, tmp_path, auto_approve=True)
    (brain.store.raw_dir / "a.md").write_text("Some content.")
    _run(brain.ingest())

    ingest_calls = spy.calls_with_task("ingest")
    assert len(ingest_calls) == 1
    assert "Recently rejected" not in ingest_calls[0][0]


def test_rejected_ingest_proposal_shows_up_as_a_lesson_in_the_next_ingest(tmp_path: Path):
    spy = SpyLLMAdapter()
    brain = SecondBrain(spy, tmp_path, auto_approve=False)

    (brain.store.raw_dir / "a.md").write_text("First source.")
    _run(brain.ingest())
    pending_id = brain.review.list_pending()[0].id
    brain.reject_pending(pending_id, "content was copied verbatim with no synthesis")

    (brain.store.raw_dir / "b.md").write_text("Second source.")
    _run(brain.ingest())

    ingest_calls = spy.calls_with_task("ingest")
    assert len(ingest_calls) == 2
    first_system, _ = ingest_calls[0]
    second_system, _ = ingest_calls[1]

    assert "Recently rejected" not in first_system  # nothing rejected yet at that point
    assert "Recently rejected" in second_system
    assert "content was copied verbatim with no synthesis" in second_system
    assert "concepts/a.md" in second_system  # names the specific rejected proposal


def test_ask_answer_receives_lessons_but_ask_plan_never_does(tmp_path: Path):
    spy = SpyLLMAdapter()
    brain = SecondBrain(spy, tmp_path, auto_approve=False)

    (brain.store.raw_dir / "a.md").write_text("Some content about widgets.")
    _run(brain.ingest())
    pending_id = brain.review.list_pending()[0].id
    brain.reject_pending(pending_id, "too shallow, needs more detail")

    _run(brain.ask("what are widgets?"))

    plan_calls = spy.calls_with_task("ask_plan")
    answer_calls = spy.calls_with_task("ask_answer")
    assert len(plan_calls) == 1
    assert len(answer_calls) == 1

    plan_system, _ = plan_calls[0]
    answer_system, _ = answer_calls[0]
    assert "Recently rejected" not in plan_system  # the retrieval boundary holds
    assert "Recently rejected" in answer_system
    assert "too shallow, needs more detail" in answer_system
