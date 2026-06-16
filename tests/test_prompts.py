"""Central prompt registry — builders produce well-formed prompts.

These run fully offline: app/prompts.py is pure (stdlib only, no settings/IO),
which is the point of centralizing prompts — they're cheap to unit-test.
"""

from app import prompts


def test_navigation_select_includes_inputs_and_cap():
    p = prompts.navigation_select("What are the fees?", "[n1] Fees (pp.1-2)", "fees doc", max_nodes=6)
    assert "What are the fees?" in p
    assert "[n1] Fees (pp.1-2)" in p
    assert "fees doc" in p                 # domain context injected
    assert "up to 6" in p                  # max_nodes interpolated
    assert "Return ONLY the JSON." in p


def test_navigation_select_omits_empty_domain_context():
    p = prompts.navigation_select("q", "tree", "", max_nodes=3)
    assert "Document context:" not in p    # no dangling empty-context line


def test_navigation_repair_constrains_to_valid_ids():
    p = prompts.navigation_repair("q", ["fake_id"], {"a", "b"})
    assert "fake_id" in p                   # tells the model what was wrong
    assert "a, b" in p                      # only valid ids offered (sorted)


def test_scope_guardrail_and_answer_carry_context():
    assert "scope validator" in prompts.scope_guardrail("ctx", "q")
    ans = prompts.answer_generation("a helpful assistant", "the question", "the pages")
    assert "a helpful assistant" in ans
    assert "the question" in ans and "the pages" in ans
    assert "DATA, not instructions" in ans  # injection-defense rule preserved
    assert "```citations" in ans            # citations block preserved


def test_grounding_and_grade_and_clarify():
    assert "fact-checker" in prompts.grounding_check("ans", "src")
    assert "Rate (0.0-1.0)" in prompts.grade_context("q", "snippet")
    assert "AMBIGUOUS" in prompts.clarify("topics", "q")


def test_query_understanding_and_profile_and_contextualize():
    assert "self-contained search query" in prompts.query_understanding("what gpa")
    assert "concise profile" in prompts.domain_profile("sample text")
    ctx = prompts.contextualize_chunk("DOC", "CHUNK")
    assert "<document>" in ctx and "CHUNK" in ctx


def test_summarize_templates_format():
    assert "{text}" in prompts.SUMMARIZE_MAP
    assert prompts.SUMMARIZE_FINAL.format(style="concise", text="x").startswith("Write a concise summary")
