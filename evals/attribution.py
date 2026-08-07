"""Retrieval-vs-generation attribution — is a bad answer retrieval's fault or
generation's?

The data to answer this already exists on every prediction record (see
harness.py's run_case): expected_pages, retrieved_page_numbers, grounded, and
unsupported_claims. Nothing joined them into one diagnostic before — a human
had to read two separate fields and infer it themselves. classify_case() makes
that join explicit and pure (no I/O), so it's usable both from benchmark.py
(aggregate + per-case reporting) and interactively (e.g. debugging one case).
"""

from __future__ import annotations

# Case-level verdicts, roughly in "how bad" order.
RETRIEVAL_MISS = "retrieval_miss"        # expected pages never retrieved at all
PARTIAL_RETRIEVAL = "partial_retrieval"  # some but not all expected pages retrieved
GENERATION_MISS = "generation_miss"      # right pages retrieved, answer still ungrounded
FALSE_POSITIVE = "false_positive"        # out-of-scope case that wasn't refused
OK = "ok"
CORRECT_REFUSAL = "correct_refusal"      # out-of-scope case correctly refused


def classify_case(
    *,
    case_type: str,
    expected_pages: list[int],
    retrieved_pages: list[int],
    grounded: bool,
    unsupported_claims: list[str],
    out_of_scope: bool,
) -> str:
    """One verdict per golden-set case — the join _page_metrics()/grounded/
    unsupported_claims don't do on their own."""
    if case_type == "out_of_scope":
        return CORRECT_REFUSAL if out_of_scope else FALSE_POSITIVE

    if not expected_pages:
        return OK  # nothing to check retrieval against

    expected = set(expected_pages)
    overlap = expected & set(retrieved_pages)

    if not overlap:
        return RETRIEVAL_MISS  # the right material was never even found

    if not grounded or unsupported_claims:
        # The right material WAS retrieved — this failure is generation's, not retrieval's.
        return GENERATION_MISS

    if overlap != expected:
        return PARTIAL_RETRIEVAL  # found some of it, not all — a recall gap, not a zero

    return OK


def summarize(verdicts: list[str]) -> dict[str, int]:
    """Count of each verdict, for the aggregate report."""
    counts = {v: 0 for v in (OK, PARTIAL_RETRIEVAL, RETRIEVAL_MISS, GENERATION_MISS, CORRECT_REFUSAL, FALSE_POSITIVE)}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    return counts
