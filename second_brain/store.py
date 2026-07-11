"""Layer 1 (raw/) and Layer 2 (wiki/) file I/O.

Everything that touches disk lives here, and only here — pipeline.py never opens
a file directly. Two things this module enforces that a raw `open()` call
wouldn't:

1. **The sandbox.** `write_page()` refuses to write anywhere except
   `wiki/concepts/*.md` or `wiki/entities/*.md`. In file 116 (the Claude-Agent-SDK
   version of this pattern) this same boundary is enforced by a `PreToolUse` hook
   intercepting the model's own Write/Edit tool calls. There's no tool-calling loop
   here — we call the LLM as a library function and apply its *proposed* JSON
   ourselves — so the boundary is just a plain path check in the one function that
   writes wiki pages. Same guarantee, no hook required, because our own code is
   the only thing that ever touches the filesystem.
2. **Source provenance.** Every page's frontmatter records which raw/ file(s) it
   came from and their content hash *at write time*. `stale_pages()` recomputes
   those hashes later and flags anything whose source has since changed — this is
   community insight #5 from karpathy-second-brain.md ("Source Provenance via
   Content Hashing"), implemented directly rather than left as an open problem.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"\A---\n(?P<meta>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")

PAGE_DIRS = ("concepts", "entities")


class SandboxViolation(ValueError):
    """Raised when a write would land outside wiki/concepts or wiki/entities."""


@dataclass
class PageMeta:
    path: str
    title: str
    tldr: str
    node_type: str
    sources: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    updated_at: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WikiStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.wiki_dir = self.base_dir / "wiki"
        for d in (self.raw_dir, self.wiki_dir, self.wiki_dir / "concepts", self.wiki_dir / "entities"):
            d.mkdir(parents=True, exist_ok=True)
        self._ledger_path = self.wiki_dir / ".ingest_ledger.json"

    # ── raw/ ─────────────────────────────────────────────────────────────

    def raw_files(self) -> list[Path]:
        return sorted(p for p in self.raw_dir.rglob("*") if p.is_file())

    def _load_ledger(self) -> dict[str, str]:
        if self._ledger_path.exists():
            return json.loads(self._ledger_path.read_text(encoding="utf-8"))
        return {}

    def _save_ledger(self, ledger: dict[str, str]) -> None:
        self._ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")

    def already_ingested(self, raw_path: Path) -> bool:
        ledger = self._load_ledger()
        key = raw_path.relative_to(self.base_dir).as_posix()
        return ledger.get(key) == sha256_file(raw_path)

    def mark_ingested(self, raw_path: Path) -> None:
        ledger = self._load_ledger()
        ledger[raw_path.relative_to(self.base_dir).as_posix()] = sha256_file(raw_path)
        self._save_ledger(ledger)

    # ── sandboxing ───────────────────────────────────────────────────────

    def _resolve_in_wiki(self, rel_path: str) -> Path:
        candidate = (self.wiki_dir / rel_path).resolve()
        wiki_root = self.wiki_dir.resolve()
        if candidate != wiki_root and wiki_root not in candidate.parents:
            raise SandboxViolation(f"{rel_path!r} resolves outside wiki/: {candidate}")
        return candidate

    # ── wiki pages (concepts/ + entities/) ──────────────────────────────

    def write_page(self, rel_path: str, *, title: str, tldr: str, content: str,
                    node_type: str = "concept", sources: list[str] | None = None) -> None:
        if not (rel_path.startswith("concepts/") or rel_path.startswith("entities/")) or not rel_path.endswith(".md"):
            raise SandboxViolation(
                f"{rel_path!r} must be under concepts/ or entities/ and end in .md"
            )
        target = self._resolve_in_wiki(rel_path)

        sources = sources or []
        source_hashes: dict[str, str] = {}
        for src in sources:
            src_path = self.base_dir / src
            if src_path.exists():
                source_hashes[src] = sha256_file(src_path)

        meta = {
            "title": title,
            "tldr": tldr,
            "node_type": node_type,
            "sources": sources,
            "source_hashes": source_hashes,
            "updated_at": _now(),
        }
        body = f"> **TLDR**: {tldr}\n\n{content.strip()}\n"
        if sources:
            body += "\n## Sources\n" + "\n".join(f"- {s}" for s in sources) + "\n"

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"---\n{json.dumps(meta, indent=2)}\n---\n{body}", encoding="utf-8")

    def read_page(self, rel_path: str) -> str | None:
        target = self._resolve_in_wiki(rel_path)
        return target.read_text(encoding="utf-8") if target.exists() else None

    def parse_frontmatter(self, text: str) -> tuple[dict, str]:
        m = _FRONTMATTER_RE.match(text)
        if not m:
            return {}, text
        try:
            meta = json.loads(m.group("meta"))
        except json.JSONDecodeError:
            meta = {}
        return meta, m.group("body")

    def read_page_body(self, rel_path: str) -> str | None:
        text = self.read_page(rel_path)
        if text is None:
            return None
        _, body = self.parse_frontmatter(text)
        return body

    def page_meta(self, rel_path: str) -> PageMeta | None:
        text = self.read_page(rel_path)
        if text is None:
            return None
        meta, _ = self.parse_frontmatter(text)
        return PageMeta(
            path=rel_path,
            title=meta.get("title", rel_path),
            tldr=meta.get("tldr", ""),
            node_type=meta.get("node_type", "concept"),
            sources=meta.get("sources", []),
            source_hashes=meta.get("source_hashes", {}),
            updated_at=meta.get("updated_at", ""),
        )

    def list_pages(self) -> list[str]:
        pages = []
        for sub in PAGE_DIRS:
            for p in sorted((self.wiki_dir / sub).glob("*.md")):
                pages.append(f"{sub}/{p.name}")
        return pages

    def list_pages_with_meta(self) -> list[PageMeta]:
        return [m for m in (self.page_meta(p) for p in self.list_pages()) if m is not None]

    # ── code-controlled top-level files (never LLM-proposed) ────────────

    def write_index(self, content: str) -> None:
        (self.wiki_dir / "index.md").write_text(content, encoding="utf-8")

    def read_index(self) -> str:
        path = self.wiki_dir / "index.md"
        return path.read_text(encoding="utf-8") if path.exists() else "(wiki is empty — run ingest first)"

    def write_schema_doc(self, content: str) -> None:
        (self.wiki_dir / "SCHEMA.md").write_text(content, encoding="utf-8")

    def append_log(self, kind: str, title: str, body: str = "") -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry = f"\n## [{ts}] {kind} | {title}\n"
        if body:
            entry += f"{body}\n"
        log_path = self.wiki_dir / "log.md"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    # ── lint helpers ─────────────────────────────────────────────────────

    def extract_wikilinks(self, body: str) -> list[str]:
        return [m.group(1).strip() for m in _WIKILINK_RE.finditer(body)]

    def broken_links(self) -> list[tuple[str, str]]:
        known = set(self.list_pages())
        broken = []
        for page in self.list_pages():
            body = self.read_page_body(page) or ""
            for target in self.extract_wikilinks(body):
                normalized = target if target.endswith(".md") else f"{target}.md"
                if normalized not in known and target not in known:
                    broken.append((page, target))
        return broken

    def stale_pages(self) -> list[str]:
        """Pages whose recorded source(s) have changed content (or vanished)
        since the page was written — karpathy-second-brain.md insight #5."""
        stale = []
        for meta in self.list_pages_with_meta():
            for src, recorded_hash in meta.source_hashes.items():
                src_path = self.base_dir / src
                if not src_path.exists() or sha256_file(src_path) != recorded_hash:
                    stale.append(meta.path)
                    break
        return stale
