"""Web search fallback (Tavily) — last resort when grounding_check would refuse.

Plain REST call, no SDK: Tavily's API is one POST. Returns [] (never raises) on
any failure — missing key, network error, bad response — so the caller always
has a safe path back to the honest decline in insufficient_grounding_node.
"""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


async def search(query: str, max_results: int = 3) -> list[dict]:
    """Return up to max_results {title, url, content} dicts, or [] on any failure."""
    if not settings.tavily_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                _TAVILY_URL,
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.debug(f"Tavily search failed, no web fallback: {e}")
        return []

    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in data.get("results", [])[:max_results]
        if r.get("content")
    ]
