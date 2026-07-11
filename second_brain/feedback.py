"""Closes one gap, deliberately not the other.

`review.py` accumulates a real feedback signal every time a proposal is rejected
-- (what was proposed, why a human said no). Nothing previously read that history
back into anything, so every ingest/ask call started equally ignorant of past
corrections. This module closes that gap, but only halfway on purpose, because
the other half is exactly the thing graphify's own CHANGELOG names as unsafe to
build without more machinery than this project has:

    "Letting verdicts influence query traversal is deliberately deferred --
     it needs propensity correction + exploration to avoid a self-reinforcing
     feedback loop." (graphify, on its reflect.py)

Three design decisions follow directly from that caution:

1. GENERATION, NEVER RETRIEVAL. Lessons are embedded in `ingest_prompt` and
   `ask_answer_prompt` (what the LLM writes) -- never in `ask_plan_prompt`
   (which pages get read). A rejection changes what the model is warned not to
   repeat; it must never change what evidence it's shown, or a bad early
   retrieval choice could entrench itself by looking "safer" over time.
2. REJECTIONS ONLY, NEVER APPROVALS. A rejection carries a clear, actionable
   signal ("don't do X, here's why"). Treating "got approved" as "was good" is
   the subtler version of the same feedback-loop risk -- the first plausible
   proposal for a topic is what gets approved, not necessarily the best one.
3. NO SCORING, NO DECAY, NO AGGREGATION ACROSS RUNS. Just the N most recent
   rejections, re-read fresh every call. At personal-wiki scale there isn't
   enough data for a decayed/corroboration model (graphify's `reflect.py`) to be
   signal rather than noise -- and that machinery is exactly what makes letting
   it touch retrieval risky in the first place. Simpler is the safer choice here,
   not just the cheaper one.
"""

from __future__ import annotations

from second_brain.review import ProposedUpdate, ReviewQueue

DEFAULT_LIMIT = 5


def recent_rejections(review: ReviewQueue, limit: int = DEFAULT_LIMIT) -> list[ProposedUpdate]:
    return review.list_rejected(limit=limit)


def render_lessons(rejections: list[ProposedUpdate]) -> str:
    """Plain text for embedding in a system prompt. Empty string if there's
    nothing to warn about -- callers should skip the section entirely rather
    than embed an empty header."""
    if not rejections:
        return ""
    lines = ["Recently rejected proposals -- do not repeat the same mistake:"]
    for u in rejections:
        reason = u.resolution_note or "(no reason given)"
        lines.append(f"- Proposed {_summarize_payload(u.kind, u.payload)} -- REJECTED: {reason}")
    return "\n".join(lines)


def _summarize_payload(kind: str, payload: dict) -> str:
    if kind == "page":
        return f"page \"{payload.get('path', '?')}\" ({payload.get('title', '')!r})"
    return f"link {payload.get('src', '?')} --{payload.get('relation', '?')}--> {payload.get('dst', '?')}"
