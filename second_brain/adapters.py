"""Two concrete LLMPort adapters.

`ProjectLLMAdapter` is the "real" one — it lazily imports app.services.llm (the
project's own multi-provider LLM service) so this module has zero import-time
dependency on the app package; it only needs it if you actually pick this adapter.

`OfflineLLMAdapter` is a deterministic, dependency-free stand-in — same role as
retrieval/dense.py's HashEmbedder or advanced_rag's FakeLLM elsewhere in this repo.
It lets the whole ingest -> ask -> lint pipeline run and be unit-tested with zero
API key and zero network. See its docstring below for exactly what it does and
does not pretend to do.
"""

from __future__ import annotations

import json
import re

_TASK_RE = re.compile(r"^TASK:\s*(\w+)", re.IGNORECASE)
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "how", "what", "why", "does", "do", "with", "about", "this", "that", "into",
}


class ProjectLLMAdapter:
    """Wraps this repo's own app.services.llm — whatever ACTIVE_MODEL is configured
    in .env is what answers ingest/ask/lint calls. Requires running with the `app`
    package importable (i.e. from the repo root) and a provider configured."""

    def __init__(self) -> None:
        self._chat = None
        self._chat_json = None

    @property
    def name(self) -> str:
        return "project"

    def _load(self) -> None:
        if self._chat is not None:
            return
        try:
            from app.services.llm import chat, chat_json
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "ProjectLLMAdapter needs the app package importable (run from the "
                "RAG-towards-production repo root) and a provider configured via "
                "ACTIVE_MODEL in .env (see app/config.py). "
                f"Import failed: {exc}"
            ) from exc
        self._chat, self._chat_json = chat, chat_json

    async def complete_json(self, system: str, user: str) -> dict:
        self._load()
        try:
            data, _tokens = await self._chat_json(user, system=system)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    async def complete_text(self, system: str, user: str) -> str:
        self._load()
        resp = await self._chat(user, system=system)
        return resp.content


class OfflineLLMAdapter:
    """Deterministic, dependency-free stand-in for a real LLM.

    - ingest: one page per raw file, content copied VERBATIM (no summarization —
      per house rule "you propose, you don't commit," a component that can't
      actually understand the text shouldn't pretend to). Proposes zero graph
      links: it has no way to judge is-a/part-of/etc. relationships honestly.
    - ask: naive keyword-overlap page selection, then an extractive (quoting, not
      generative) "answer." Always declines to write back
      (nothing_to_file_reason) — the honest answer for something that cannot
      synthesize.
    - lint: turns the structural stats dict (computed entirely in pipeline.py,
      not by this adapter) into one canned sentence.

    The point is to prove the *pipeline mechanics* — sandboxed writes, the review
    queue, idempotent ingest, the "two outputs" retry, staleness detection — work
    end to end without spending a token or needing an API key. For real synthesis
    quality, use ProjectLLMAdapter.
    """

    @property
    def name(self) -> str:
        return "offline"

    async def complete_text(self, system: str, user: str) -> str:
        if _task_of(system) == "lint_narrative":
            return _offline_lint_narrative(_extract_json_block(user) or {})
        return ""

    async def complete_json(self, system: str, user: str) -> dict:
        task = _task_of(system)
        if task == "ingest":
            return _offline_ingest(user)
        if task == "ask_plan":
            return _offline_ask_plan(user)
        if task == "ask_answer":
            return _offline_ask_answer(user)
        return {}


# ── Offline adapter internals ────────────────────────────────────────────────

def _task_of(system: str) -> str:
    m = _TASK_RE.match(system.strip())
    return m.group(1).lower() if m else ""


def _extract_json_block(text: str) -> dict | None:
    m = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _offline_ingest(user: str) -> dict:
    m = re.search(r"SOURCE FILE \((?P<name>.+?)\):\n(?P<content>.*)", user, re.DOTALL)
    if not m:
        return {"pages": [], "graph_links": [], "changelog": ""}
    filename, content = m.group("name"), m.group("content").strip()
    stem = re.sub(r"\.[^.]+$", "", filename)
    first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), stem)
    tldr = (first_line[:157] + "...") if len(first_line) > 160 else first_line
    page = {
        "path": f"concepts/{_slugify(stem)}.md",
        "title": stem.replace("_", " ").replace("-", " ").title(),
        "tldr": tldr,
        "content": content,
        "type": "concept",
    }
    return {
        "pages": [page],
        "graph_links": [],
        "changelog": f"Ingested `{filename}` verbatim (offline adapter: no synthesis, no inferred links).",
    }


_INDEX_LINE_RE = re.compile(r"\[\[([^\]]+)\]\]\s*(?:—|--)\s*(.*)")


def _parse_index(index_text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in _INDEX_LINE_RE.finditer(index_text)]


def _offline_ask_plan(user: str) -> dict:
    m = re.search(r"INDEX:\n(?P<index>.*?)\n\nQUESTION: (?P<question>.*)", user, re.DOTALL)
    if not m:
        return {"pages": []}
    question_words = _words(m.group("question"))
    scored = []
    for path, tldr in _parse_index(m.group("index")):
        overlap = len(question_words & (_words(path) | _words(tldr)))
        if overlap:
            scored.append((overlap, path))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return {"pages": [path for _, path in scored[:5]]}


def _offline_ask_answer(user: str) -> dict:
    decline = {
        "answer": "",
        "wiki_updates": [],
        "graph_links": [],
        "nothing_to_file_reason": "offline adapter performs no synthesis",
    }
    m = re.search(
        r"WIKI PAGES:\n(?P<pages>.*?)\n\nGRAPH RELATIONSHIPS:\n(?P<graph>.*?)\n\nQUESTION: (?P<question>.*)",
        user, re.DOTALL,
    )
    if not m or m.group("pages").strip() == "(none read)":
        decline["answer"] = "No wiki pages matched this question (nothing indexed yet, or no keyword overlap)."
        return decline

    page_blocks = re.findall(r"=== (.+?) ===\n(.*?)(?=\n=== |\Z)", m.group("pages"), re.DOTALL)
    lines = []
    for path, content in page_blocks:
        first = next((ln.strip() for ln in content.strip().splitlines() if ln.strip()), "")
        lines.append(f"- **{path}**: {first}")
    decline["answer"] = (
        "[offline adapter — extractive, not generated]\n\n"
        f"Read {len(page_blocks)} page(s):\n\n" + "\n".join(lines)
    )
    return decline


def _offline_lint_narrative(stats: dict) -> str:
    counts = {
        "contradiction(s)": stats.get("contradictions", 0),
        "orphan page(s)": stats.get("orphans", 0),
        "broken link(s)": stats.get("broken_links", 0),
        "stale page(s)": stats.get("stale_pages", 0),
    }
    findings = [f"{n} {label}" for label, n in counts.items() if n]
    if not findings:
        return "The wiki looks healthy: no contradictions, orphan pages, broken links, or stale sources detected."
    return (
        "Found " + ", ".join(findings) + ". "
        "[offline adapter: no suggested articles — that needs real synthesis.]"
    )
