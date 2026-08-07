"""Pure-function tests for the retrieval-vs-generation attribution join.
No app dependencies — same reasoning as retrieval/base.py's own tests."""

from evals.attribution import classify_case, summarize


def test_retrieval_miss_when_no_expected_page_retrieved():
    v = classify_case(case_type="in_scope", expected_pages=[3], retrieved_pages=[1, 2],
                       grounded=True, unsupported_claims=[], out_of_scope=False)
    assert v == "retrieval_miss"


def test_generation_miss_when_right_page_found_but_ungrounded():
    v = classify_case(case_type="in_scope", expected_pages=[3], retrieved_pages=[3],
                       grounded=False, unsupported_claims=["some claim"], out_of_scope=False)
    assert v == "generation_miss"


def test_partial_retrieval_when_some_but_not_all_pages_found():
    v = classify_case(case_type="in_scope", expected_pages=[3, 4], retrieved_pages=[3],
                       grounded=True, unsupported_claims=[], out_of_scope=False)
    assert v == "partial_retrieval"


def test_ok_when_fully_retrieved_and_grounded():
    v = classify_case(case_type="in_scope", expected_pages=[3], retrieved_pages=[3, 5],
                       grounded=True, unsupported_claims=[], out_of_scope=False)
    assert v == "ok"


def test_correct_refusal_for_out_of_scope_case_that_was_refused():
    v = classify_case(case_type="out_of_scope", expected_pages=[], retrieved_pages=[],
                       grounded=False, unsupported_claims=[], out_of_scope=True)
    assert v == "correct_refusal"


def test_false_positive_for_out_of_scope_case_that_was_not_refused():
    v = classify_case(case_type="out_of_scope", expected_pages=[], retrieved_pages=[],
                       grounded=True, unsupported_claims=[], out_of_scope=False)
    assert v == "false_positive"


def test_summarize_counts_every_verdict_kind_including_zero():
    counts = summarize(["ok", "ok", "retrieval_miss"])
    assert counts["ok"] == 2
    assert counts["retrieval_miss"] == 1
    assert counts["generation_miss"] == 0  # present with 0, not missing
