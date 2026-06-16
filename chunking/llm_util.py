"""Shared, injectable LLM caller for the LLM-driven chunkers.

Contextual / proposition / agentic chunking all need an LLM. To keep the lab
**standalone and provider-agnostic**, they accept an injected ``llm_fn(prompt)->str``.
The default lazily uses OpenAI (if installed + OPENAI_API_KEY set), else raises a
clear error — so importing the package never requires a model.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

LLMFn = Callable[[str], str]

_client = None


def default_llm_fn(prompt: str) -> str:
    """Minimal OpenAI chat call. Override by injecting your own llm_fn."""
    global _client
    from openai import OpenAI  # lazy: only needed if you actually use an LLM chunker

    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY
    resp = _client.chat.completions.create(
        model=os.getenv("CHUNK_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def parse_json_list(raw: str) -> list[str]:
    """Best-effort extraction of a JSON array of strings from an LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        # Fallback: pull the first [...] block.
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                return [str(x).strip() for x in data if str(x).strip()]
            except Exception:
                pass
    return []
