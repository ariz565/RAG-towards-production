"""Prompt construction — pure functions, no I/O, no LLM calls.

Mirrors the style of app/prompts.py: every prompt is a function that returns
(system, user) and can be unit-tested without a network call. Every prompt that
expects structured output leads with a `TASK: <name>` marker on the first line of
the system prompt — a real backend just treats it as an instruction; the
OfflineLLMAdapter (adapters.py) uses it to route to the right deterministic
generator without brittle prose-sniffing.
"""

from __future__ import annotations

import json

from second_brain.schema import SCHEMA_RULES

_JSON_ONLY = "Output ONLY the JSON object. No markdown fences, no prose outside it."


def ingest_prompt(
    index_summary: str, raw_filename: str, raw_content: str, *, max_chars: int = 8000, lessons: str = ""
) -> tuple[str, str]:
    lessons_block = f"\n{lessons}\n" if lessons else ""
    system = (
        "TASK: ingest\n"
        "You are a knowledge architect maintaining a personal wiki.\n\n"
        f"{SCHEMA_RULES}\n"
        f"{lessons_block}"
        "Given the current wiki index and one new raw source, decide what pages to "
        "create or update and what typed relationships to propose. Output ONLY a "
        "JSON object with this exact shape:\n"
        "{\n"
        '  "pages": [{"path": "concepts/x.md", "title": "...", "tldr": "...", '
        '"content": "...", "type": "concept"}],\n'
        '  "graph_links": [{"src": "concepts/x", "dst": "concepts/y", "relation": '
        '"is-a", "confidence": 0.9, "rationale": "..."}],\n'
        '  "changelog": "one-line description of what changed"\n'
        "}\n"
        '"type" is "concept" or "entity" ("path" must start with the matching '
        'directory). Emit [] for anything you have nothing to add. '
        f"{_JSON_ONLY}"
    )
    user = (
        f"CURRENT WIKI INDEX:\n{index_summary}\n\n"
        f"SOURCE FILE ({raw_filename}):\n{raw_content[:max_chars]}"
    )
    return system, user


def ask_plan_prompt(index_summary: str, question: str) -> tuple[str, str]:
    # Deliberately no `lessons` parameter here. Past rejections are allowed to
    # change what the model is warned not to WRITE (see ask_answer_prompt /
    # ingest_prompt), never what evidence it's shown before answering -- see
    # feedback.py's module docstring for why that boundary matters.
    system = (
        "TASK: ask_plan\n"
        "Given a wiki index and a question, decide which wiki pages are worth "
        'reading before answering. Output ONLY JSON: {"pages": ["concepts/x.md", '
        '...]}. List at most 5 pages, most relevant first. Return an empty list if '
        f"nothing in the index looks relevant. {_JSON_ONLY}"
    )
    user = f"INDEX:\n{index_summary}\n\nQUESTION: {question}"
    return system, user


def ask_answer_prompt(pages: dict[str, str], graph_context: str, question: str, *, lessons: str = "") -> tuple[str, str]:
    lessons_block = f"\n{lessons}\n" if lessons else ""
    system = (
        "TASK: ask_answer\n"
        f"{SCHEMA_RULES}\n"
        f"{lessons_block}"
        "Answer the question using ONLY the wiki pages and graph relationships given "
        "below. Output ONLY a JSON object with this exact shape:\n"
        "{\n"
        '  "answer": "markdown answer, citing which pages it draws on",\n'
        '  "wiki_updates": [{"path": "...", "title": "...", "tldr": "...", '
        '"content": "...", "type": "concept", "why": "what new synthesis this '
        'captures"}],\n'
        '  "graph_links": [{"src": "...", "dst": "...", "relation": "...", '
        '"confidence": 0.0, "rationale": "..."}],\n'
        '  "nothing_to_file_reason": null\n'
        "}\n"
        "House rule 6 is checked mechanically after this call: \"wiki_updates\" must "
        "be non-empty, OR \"nothing_to_file_reason\" must be a real explanation "
        "string. Leaving both empty is a schema violation and you will be asked "
        f"again. {_JSON_ONLY}"
    )
    context_block = "\n\n".join(f"=== {path} ===\n{content}" for path, content in pages.items())
    user = (
        f"WIKI PAGES:\n{context_block or '(none read)'}\n\n"
        f"GRAPH RELATIONSHIPS:\n{graph_context or '(none)'}\n\n"
        f"QUESTION: {question}"
    )
    return system, user


def ask_retry_note() -> str:
    """Appended to the user prompt when the first ask_answer response violated
    house rule 6 (two outputs, always) — see pipeline.py's verification step."""
    return (
        "\n\n[RETRY] Your previous JSON response left both \"wiki_updates\" and "
        '"nothing_to_file_reason" empty/null. Per house rule 6, pick one: a concrete '
        "wiki_updates entry, or a genuine, specific nothing_to_file_reason. Return "
        "the full corrected JSON object."
    )


def lint_narrative_prompt(stats: dict) -> tuple[str, str]:
    system = (
        "TASK: lint_narrative\n"
        "You are health-checking a personal knowledge wiki. You are given "
        "structural findings already computed by code (contradictions, orphan "
        "pages, broken links, stale pages). Write one short, honest paragraph "
        "summarizing overall wiki health, then a bullet list of 0-3 suggested new "
        "article topics the gaps point to. Do not invent findings beyond what's "
        "given — you are narrating, not analyzing."
    )
    user = (
        "STRUCTURAL FINDINGS:\n```json\n" + json.dumps(stats, indent=2) + "\n```\n\n"
        "Write the paragraph, then the bullet list (omit the list if there's "
        "nothing to suggest)."
    )
    return system, user
