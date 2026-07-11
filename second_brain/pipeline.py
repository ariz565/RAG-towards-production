"""The orchestrator — Karpathy's three operations (Ingest / Query / Lint), made
concrete against an LLMPort, a WikiStore, a GraphStore, and a ReviewQueue.

Nothing here knows which LLM backend is in use (see ports.py/adapters.py) or
whether an update is destined for immediate application or a human's later
approval (see review.py) — this module just wires the pieces together and is the
one place that enforces the two structural guarantees the schema *asks for* but
cannot *guarantee* on its own:

  - "two outputs, always" (ask): verified by inspecting the parsed JSON response,
    with exactly one retry if the model skipped it — a verification checkpoint,
    not an infinite loop.
  - "you propose, you don't commit" (ingest + ask): every page write and graph
    link goes through review.py; this module decides *when* to auto-apply
    (only if the caller opted into --auto-approve) but never *whether* a write is
    trustworthy — that judgment stays with review.py's approve/reject, i.e. with
    a human.

Every operation is also independently audited to audit.log (JSON lines) — an
append-only record of what actually ran, not what the model claims it did.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from second_brain import feedback, prompts, schema
from second_brain.graph import GraphStore, InvalidRelation
from second_brain.graph_html import write_graph_html
from second_brain.ports import LLMPort
from second_brain.review import ReviewQueue
from second_brain.store import SandboxViolation, WikiStore

_REQUIRED_PAGE_KEYS = {"path", "title", "content"}


def _node_id(page_path: str) -> str:
    return page_path[:-3] if page_path.endswith(".md") else page_path


def _recent_log_entries(log_text: str, n: int = 5) -> list[str]:
    chunks = [c.strip() for c in re.split(r"(?=^## \[)", log_text, flags=re.MULTILINE) if c.strip().startswith("## [")]
    return list(reversed(chunks[-n:]))


class SecondBrain:
    def __init__(self, llm: LLMPort, base_dir: Path, *, auto_approve: bool = False) -> None:
        self.llm = llm
        self.base_dir = Path(base_dir)
        self.auto_approve = auto_approve
        self.store = WikiStore(self.base_dir)
        self.graph = GraphStore(self.store.wiki_dir / "graph.db")
        self.review = ReviewQueue(self.base_dir)
        self.audit_log = self.base_dir / "audit.log"
        self.store.write_schema_doc(schema.render_schema_doc())

    def close(self) -> None:
        """Releases graph.db's sqlite connection. The CLI never needs this (the
        process exits after each command, which reclaims the handle anyway) --
        this matters for programmatic use: a script or notebook constructing
        multiple SecondBrain instances in one process, or anything that tries to
        delete the underlying directory afterward (Windows locks open file
        handles, unlike POSIX) will fail without it. Use as a context manager
        to get this for free."""
        self.graph.close()

    def __enter__(self) -> "SecondBrain":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ── shared plumbing ──────────────────────────────────────────────────

    def _audit(self, event: str, **fields) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, "backend": self.llm.name, **fields}
        with self.audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _apply_page(self, payload: dict, default_source: str | None) -> str | None:
        """Returns an error string on failure, None on success."""
        missing = _REQUIRED_PAGE_KEYS - payload.keys()
        if missing:
            return f"page proposal missing required keys: {sorted(missing)}"
        sources = payload.get("sources") or ([default_source] if default_source else [])
        try:
            self.store.write_page(
                payload["path"],
                title=payload.get("title", payload["path"]),
                tldr=payload.get("tldr", ""),
                content=payload["content"],
                node_type=payload.get("type", "concept"),
                sources=sources,
            )
        except SandboxViolation as exc:
            return str(exc)
        self.graph.upsert_node(_node_id(payload["path"]), payload.get("title", payload["path"]))
        return None

    def _apply_link(self, payload: dict) -> str | None:
        try:
            self.graph.link(
                payload["src"], payload["dst"], payload["relation"],
                confidence=float(payload.get("confidence", 1.0)),
                rationale=payload.get("rationale", ""),
            )
        except (InvalidRelation, KeyError, ValueError) as exc:
            return str(exc)
        return None

    def _propose_and_maybe_apply(self, kind: str, operation: str, payload: dict, rationale: str,
                                  *, default_source: str | None = None) -> dict:
        update = self.review.propose(kind, operation, payload, rationale)
        result = {"id": update.id, "kind": kind, "applied": False, "error": None}
        if self.auto_approve:
            error = self._apply_page(payload, default_source) if kind == "page" else self._apply_link(payload)
            if error:
                result["error"] = error
            else:
                self.review.mark_applied(update.id)
                result["applied"] = True
        return result

    def apply_pending(self, update_id: str) -> dict:
        """Used by `second_brain review approve <id>` — the human-in-the-loop path."""
        update = self.review.get_pending(update_id)
        error = self._apply_page(update.payload, None) if update.kind == "page" else self._apply_link(update.payload)
        if error:
            return {"id": update_id, "applied": False, "error": error}
        self.review.mark_applied(update_id)
        self._regenerate_index()
        self._audit("review_approve", id=update_id, kind=update.kind)
        return {"id": update_id, "applied": True, "error": None}

    def reject_pending(self, update_id: str, reason: str) -> dict:
        self.review.mark_rejected(update_id, reason)
        self._audit("review_reject", id=update_id, reason=reason)
        return {"id": update_id, "rejected": True}

    def _regenerate_index(self) -> None:
        """index.md is a *view*, not a store — it's fully recomputed from the
        current pages + graph on every change, per karpathy-second-brain.md
        community insight #10 ("the index isn't a file the agent maintains by
        hand — it's a query, always current"). The LLM is never asked to hand-
        write it, so it can't forget to."""
        pages = self.store.list_pages_with_meta()
        concepts = [p for p in pages if p.node_type != "entity"]
        entities = [p for p in pages if p.node_type == "entity"]
        lines = ["# Second Brain — Index", "", f"_Last updated: {datetime.now(timezone.utc).isoformat()}_", ""]

        def _section(title: str, items) -> list[str]:
            if not items:
                return []
            out = [f"## {title}", ""]
            out += [f"- [[{p.path}]] — {p.tldr}" for p in items]
            out.append("")
            return out

        lines += _section("Concepts", concepts)
        lines += _section("Entities", entities)

        connected = sorted(pages, key=lambda p: len(self.graph.neighbors(_node_id(p.path))), reverse=True)
        connected = [p for p in connected if self.graph.neighbors(_node_id(p.path))][:5]
        if connected:
            lines += ["## Most connected", ""]
            lines += [f"- [[{p.path}]] ({len(self.graph.neighbors(_node_id(p.path)))} relationship(s))" for p in connected]
            lines.append("")

        log_text = (self.store.wiki_dir / "log.md")
        if log_text.exists():
            recent = _recent_log_entries(log_text.read_text(encoding="utf-8"))
            if recent:
                lines += ["## Recent changes", ""] + recent

        self.store.write_index("\n".join(lines).rstrip() + "\n")

    # ── Operation 1: Ingest ─────────────────────────────────────────────

    async def ingest(self, target: Path | None = None) -> list[dict]:
        candidates = [target] if target else self.store.raw_files()
        results = []
        for raw_path in candidates:
            rel_source = raw_path.relative_to(self.base_dir).as_posix()
            if self.store.already_ingested(raw_path):
                results.append({"source": rel_source, "status": "skipped_unchanged"})
                continue

            index_summary = self.store.read_index()
            lessons = feedback.render_lessons(feedback.recent_rejections(self.review))
            system, user = prompts.ingest_prompt(
                index_summary, raw_path.name, raw_path.read_text(encoding="utf-8"), lessons=lessons
            )
            response = await self.llm.complete_json(system, user)

            page_results = [
                self._propose_and_maybe_apply("page", "ingest", page, response.get("changelog", ""), default_source=rel_source)
                for page in response.get("pages", []) or []
            ]
            link_results = [
                self._propose_and_maybe_apply("graph_link", "ingest", link, link.get("rationale", ""))
                for link in response.get("graph_links", []) or []
            ]

            self.store.mark_ingested(raw_path)
            # append_log BEFORE regenerate_index: index.md's "Recent changes" section
            # reads log.md, so regenerating first would always be one entry behind.
            self.store.append_log("ingest", raw_path.name, response.get("changelog", ""))
            self._regenerate_index()
            self._audit("ingest", source=rel_source, pages=len(page_results), links=len(link_results))

            results.append({
                "source": rel_source,
                "status": "ingested",
                "pages": page_results,
                "graph_links": link_results,
                "changelog": response.get("changelog", ""),
            })
        return results

    # ── Operation 2: Ask ────────────────────────────────────────────────

    async def ask(self, question: str) -> dict:
        index_summary = self.store.read_index()
        # ask_plan_prompt deliberately never receives `lessons` -- see its call
        # site's comment in prompts.py and feedback.py's module docstring.
        plan_system, plan_user = prompts.ask_plan_prompt(index_summary, question)
        plan = await self.llm.complete_json(plan_system, plan_user)

        known_pages = set(self.store.list_pages())
        page_paths = [p for p in plan.get("pages", []) or [] if p in known_pages]
        pages = {p: (self.store.read_page_body(p) or "") for p in page_paths}
        graph_context = self.graph.render_context([_node_id(p) for p in page_paths])

        lessons = feedback.render_lessons(feedback.recent_rejections(self.review))
        system, user = prompts.ask_answer_prompt(pages, graph_context, question, lessons=lessons)
        response = await self.llm.complete_json(system, user)

        def _satisfies_rule_6(resp: dict) -> bool:
            return bool(resp.get("wiki_updates")) or bool(resp.get("nothing_to_file_reason"))

        retried = False
        if not _satisfies_rule_6(response):
            retried = True
            response = await self.llm.complete_json(system, user + prompts.ask_retry_note())

        page_results = [
            self._propose_and_maybe_apply("page", "ask", page, page.get("why", ""))
            for page in response.get("wiki_updates", []) or []
        ]
        link_results = [
            self._propose_and_maybe_apply("graph_link", "ask", link, link.get("rationale", ""))
            for link in response.get("graph_links", []) or []
        ]

        write_back_ok = _satisfies_rule_6(response)
        self.store.append_log("ask", question[:60], response.get("nothing_to_file_reason") or "wiki updated")
        self._regenerate_index()  # always, even when nothing was written -- reflects the log entry above
        self._audit("ask", question=question, pages_read=len(pages), retried=retried, write_back_ok=write_back_ok)

        return {
            "answer": response.get("answer", ""),
            "pages_read": page_paths,
            "wiki_updates": page_results,
            "graph_links": link_results,
            "nothing_to_file_reason": response.get("nothing_to_file_reason"),
            "retried_for_rule_6": retried,
            "write_back_ok": write_back_ok,
        }

    # ── Operation 3: Lint ───────────────────────────────────────────────

    async def lint(self) -> dict:
        stats = {
            "contradictions": len(self.graph.contradictions()),
            "orphans": len(self.graph.orphans()),
            "broken_links": len(self.store.broken_links()),
            "stale_pages": len(self.store.stale_pages()),
            "low_confidence_edges": len(self.graph.low_confidence()),
        }
        system, user = prompts.lint_narrative_prompt(stats)
        narrative = await self.llm.complete_text(system, user)

        report = {
            "stats": stats,
            "narrative": narrative,
            "contradictions": [e.__dict__ for e in self.graph.contradictions()],
            "orphans": self.graph.orphans(),
            "broken_links": self.store.broken_links(),
            "stale_pages": self.store.stale_pages(),
            "low_confidence_edges": [e.__dict__ for e in self.graph.low_confidence()],
        }
        report_path = self.store.wiki_dir / f".lint-{datetime.now(timezone.utc).date()}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        self.store.append_log("lint", "health check", narrative)
        self._regenerate_index()
        self._audit("lint", **stats)
        report["report_path"] = str(report_path)
        return report

    # ── Graph viewer ─────────────────────────────────────────────────────

    def export_graph_html(self, out_path: Path | None = None) -> Path:
        """Interactive graph.html -- vis.js loaded from a CDN in the browser (see
        graph_html.py), not a wiki page, so it isn't sandboxed through review.py;
        it's a read-only rendering of whatever's already been approved into
        graph.db, regenerated on demand, never a source of new claims."""
        target = out_path or (self.store.wiki_dir / "graph.html")
        written = write_graph_html(self.graph, self.store, target)
        self._audit("export_graph_html", path=str(written))
        return written
