"""Query understanding — classify → normalize → (conservatively) rewrite.

Runs before routing/retrieval to lift retrieval quality (2026 practice):
- **classify** the intent so the system can branch (factual lookup vs summarize
  vs compare vs chit-chat) instead of forcing everything through retrieval;
- **normalize** to remove noise (whitespace, stray punctuation);
- **rewrite** the query into a clearer, self-contained search query — but
  *conservatively*: aggressive expansion/rewriting is known to **poison** recall,
  so the rewrite must preserve meaning and is independently toggleable + measured.

LLM-driven when available (one JSON call), with a deterministic heuristic
fallback so it runs offline. Gate with QUERY_UNDERSTANDING_ENABLED /
QUERY_REWRITE_ENABLED.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app import prompts
from app.config import settings

logger = logging.getLogger(__name__)

# Allowed intents (kept small + actionable).
INTENTS = {"factual", "summarize", "compare", "chitchat"}

_SUMMARIZE_RE = re.compile(r"\b(summari[sz]e|summary|tl;dr|overview of)\b", re.IGNORECASE)
_COMPARE_RE = re.compile(r"\b(compare|comparison|difference|differences|versus|vs\.?|better)\b", re.IGNORECASE)
_CHITCHAT_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening)|how are you)\b", re.IGNORECASE)


@dataclass
class QueryAnalysis:
    original: str
    normalized: str
    rewritten: str            # the query to actually search with
    intent: str               # factual | summarize | compare | chitchat
    scope: str                # "single" | "multi"
    needs_retrieval: bool
    reasoning: str = ""


def _normalize(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    return q


def _heuristic_intent(q: str) -> str:
    if _CHITCHAT_RE.search(q):
        return "chitchat"
    if _SUMMARIZE_RE.search(q):
        return "summarize"
    if _COMPARE_RE.search(q):
        return "compare"
    return "factual"


async def analyze(query: str) -> QueryAnalysis:
    normalized = _normalize(query)

    # Disabled → pass-through (preserves prior behavior exactly).
    if not settings.query_understanding_enabled:
        return QueryAnalysis(query, normalized, normalized, "factual", "single", True, "disabled")

    intent = _heuristic_intent(normalized)
    rewritten = normalized
    reasoning = "heuristic"

    # LLM classification + conservative rewrite (best-effort; falls back to heuristic).
    try:
        from app.services.llm import chat_json

        data, _ = await chat_json(prompts.query_understanding(normalized), temperature=0)
        if isinstance(data, dict):
            cand = str(data.get("intent", "")).lower().strip()
            if cand in INTENTS:
                intent = cand
            if settings.query_rewrite_enabled:
                rw = str(data.get("rewritten", "")).strip()
                if rw:
                    rewritten = rw
            reasoning = str(data.get("reasoning", "")).strip() or "llm"
    except Exception as e:  # offline / parse failure → keep heuristic
        logger.debug(f"Query understanding fell back to heuristic: {e}")

    scope = "multi" if intent == "compare" else "single"
    needs_retrieval = intent in ("factual", "compare", "summarize")
    return QueryAnalysis(query, normalized, rewritten, intent, scope, needs_retrieval, reasoning)
