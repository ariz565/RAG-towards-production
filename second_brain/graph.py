"""Typed-edge knowledge graph over wiki pages — pure sqlite3 (stdlib), no new
dependency.

This is the community's fix for the crack in Karpathy's original design (see
karpathy-second-brain.md, community insight #9, "Ontology Is the Hardest Unsolved
Problem"): a plain `[[wikilink]]` says two pages are related, never *how*. If
"Fruit" and "iPhone" both link to "Apple," a link can't tell you which Apple. A
typed edge can: `apple-fruit --is-a--> fruit` vs `apple-inc --is-a--> company`.

Five relation types, no more. `InvalidRelation` is raised on anything else —
deliberately, so the ontology can't silently sprawl into a vague folksonomy the
way free-form wikilinks do at scale (see file 116, which this module's design is
closest to; graphify takes the opposite bet — dozens of relation verbs inferred by
an LLM — which is right for code, wrong for a personal wiki where you want to be
able to name every relation type from memory).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VALID_RELATIONS = frozenset({"is-a", "part-of", "contradicts", "supersedes", "depends-on"})

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    node_type  TEXT NOT NULL DEFAULT 'concept',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    relation   TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    rationale  TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (src, dst, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


class InvalidRelation(ValueError):
    pass


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    relation: str
    confidence: float
    rationale: str
    created_at: str


@dataclass(frozen=True)
class Node:
    id: str
    title: str
    node_type: str


class GraphStore:
    """One SQLite file. Every write auto-commits — this is a single-writer personal
    tool, not a service; see README's "known limits" for what that means at scale."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert_node(self, node_id: str, title: str, node_type: str = "concept") -> None:
        now = self._now()
        self._conn.execute(
            """INSERT INTO nodes (id, title, node_type, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                                              node_type=excluded.node_type,
                                              updated_at=excluded.updated_at""",
            (node_id, title, node_type, now, now),
        )
        self._conn.commit()

    def link(self, src: str, dst: str, relation: str, confidence: float = 1.0, rationale: str = "") -> Edge:
        if relation not in VALID_RELATIONS:
            raise InvalidRelation(f"{relation!r} is not one of {sorted(VALID_RELATIONS)}")
        confidence = max(0.0, min(1.0, confidence))
        now = self._now()
        # Auto-create endpoint nodes if they don't exist yet — the caller may be
        # linking a page it's proposing in the same batch, not yet applied.
        for node_id in (src, dst):
            self._conn.execute(
                "INSERT OR IGNORE INTO nodes (id, title, node_type, created_at, updated_at) "
                "VALUES (?, ?, 'concept', ?, ?)",
                (node_id, node_id, now, now),
            )
        self._conn.execute(
            """INSERT INTO edges (src, dst, relation, confidence, rationale, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(src, dst, relation) DO UPDATE SET
                   confidence=excluded.confidence, rationale=excluded.rationale""",
            (src, dst, relation, confidence, rationale, now),
        )
        self._conn.commit()
        return Edge(src, dst, relation, confidence, rationale, now)

    def _rows_to_edges(self, rows) -> list[Edge]:
        return [Edge(r["src"], r["dst"], r["relation"], r["confidence"], r["rationale"], r["created_at"]) for r in rows]

    def neighbors(self, node_id: str, relation: str | None = None) -> list[Edge]:
        query = "SELECT * FROM edges WHERE src = ? OR dst = ?"
        params: list = [node_id, node_id]
        if relation:
            query += " AND relation = ?"
            params.append(relation)
        return self._rows_to_edges(self._conn.execute(query, params).fetchall())

    def contradictions(self) -> list[Edge]:
        """Used by lint() — O(1) index lookup instead of re-reading every page to
        spot disagreements. This is the whole point of a graph over grep."""
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE relation = 'contradicts' ORDER BY created_at DESC"
        ).fetchall()
        return self._rows_to_edges(rows)

    def low_confidence(self, threshold: float = 0.6) -> list[Edge]:
        """Edges worth a human's second look — surfaced in lint(), not silently
        trusted just because they made it into the graph."""
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE confidence < ? ORDER BY confidence ASC", (threshold,)
        ).fetchall()
        return self._rows_to_edges(rows)

    def _outgoing(self, node_id: str, relation: str) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE src = ? AND relation = ?", (node_id, relation)
        ).fetchall()
        return self._rows_to_edges(rows)

    def traverse(self, start: str, relation: str, depth: int = 2) -> list[str]:
        """Directed BFS following src->dst only, e.g. walk every 'part-of'
        ancestor of `start`. Deliberately one-directional, unlike neighbors()
        (which is bidirectional for context-rendering) — "is-a"/"part-of" chains
        only make sense walked forward."""
        seen = {start}
        frontier = [start]
        for _ in range(depth):
            next_frontier: list[str] = []
            for node in frontier:
                for edge in self._outgoing(node, relation):
                    if edge.dst not in seen:
                        seen.add(edge.dst)
                        next_frontier.append(edge.dst)
            frontier = next_frontier
            if not frontier:
                break
        return sorted(seen - {start})

    def all_node_ids(self) -> set[str]:
        return {r["id"] for r in self._conn.execute("SELECT id FROM nodes").fetchall()}

    def all_nodes(self) -> list[Node]:
        rows = self._conn.execute("SELECT id, title, node_type FROM nodes ORDER BY id").fetchall()
        return [Node(r["id"], r["title"], r["node_type"]) for r in rows]

    def all_edges(self) -> list[Edge]:
        rows = self._conn.execute("SELECT * FROM edges ORDER BY created_at").fetchall()
        return self._rows_to_edges(rows)

    def orphans(self) -> list[str]:
        """Nodes with no edges at all — candidates for linking up or archiving."""
        rows = self._conn.execute(
            """SELECT id FROM nodes
               WHERE id NOT IN (SELECT src FROM edges)
                 AND id NOT IN (SELECT dst FROM edges)"""
        ).fetchall()
        return [r["id"] for r in rows]

    def render_context(self, node_ids: list[str]) -> str:
        """Plain-text rendering of a node's relationships, for embedding directly
        into the ask_answer prompt (see prompts.py's graph_context parameter)."""
        lines: list[str] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for node_id in node_ids:
            for edge in self.neighbors(node_id):
                key = (edge.src, edge.dst, edge.relation)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                lines.append(f"{edge.src} --{edge.relation}--> {edge.dst} (confidence {edge.confidence:.2f})")
        return "\n".join(lines)
