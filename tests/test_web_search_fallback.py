"""Web search fallback (Tavily) — the node's logic, mocked end-to-end.

No real network/API key involved: web_search.search() and chat() are both
monkeypatched, matching this suite's existing convention (see
test_pageindex.py's monkeypatch.setattr(ts, "chat_json", ...)). What this
proves: the graceful "not configured" / "no results" paths correctly fall
through to insufficient_grounding rather than silently breaking the graph,
and the "results found" path actually produces a labeled, non-grounded answer.
"""

import pytest

from app.config import settings
from app.services import pipeline, web_search


def _base_state() -> dict:
    return {"original_query": "who won the 2026 F1 championship?"}


@pytest.mark.asyncio
async def test_disabled_by_default_falls_through(monkeypatch):
    monkeypatch.setattr(settings, "web_search_fallback_enabled", False)
    result = await pipeline.web_search_fallback_node(_base_state())
    assert "web_search_used" not in result
    assert pipeline.route_after_web_search_fallback(result) == "insufficient_grounding"


@pytest.mark.asyncio
async def test_enabled_but_no_results_falls_through(monkeypatch):
    monkeypatch.setattr(settings, "web_search_fallback_enabled", True)

    async def fake_search(query, max_results=3):
        return []

    monkeypatch.setattr(web_search, "search", fake_search)
    result = await pipeline.web_search_fallback_node(_base_state())
    assert "web_search_used" not in result
    assert pipeline.route_after_web_search_fallback(result) == "insufficient_grounding"


@pytest.mark.asyncio
async def test_enabled_with_results_answers_and_labels_as_web_sourced(monkeypatch):
    monkeypatch.setattr(settings, "web_search_fallback_enabled", True)

    async def fake_search(query, max_results=3):
        return [{"title": "F1 2026 results", "url": "https://example.com/f1", "content": "Driver X won."}]

    class FakeResponse:
        content = "Driver X won the 2026 championship."
        tokens_used = 12

    async def fake_chat(prompt, *, temperature=0):
        return FakeResponse()

    monkeypatch.setattr(web_search, "search", fake_search)
    monkeypatch.setattr(pipeline, "chat", fake_chat)

    result = await pipeline.web_search_fallback_node(_base_state())
    assert result["web_search_used"] is True
    assert result["grounded"] is False  # honest: web-sourced, not document-grounded
    assert "Driver X won" in result["answer"]
    assert "web search" in result["answer"].lower()
    assert pipeline.route_after_web_search_fallback(result) == pipeline.END
