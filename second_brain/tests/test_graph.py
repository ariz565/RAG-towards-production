from pathlib import Path

import pytest

from second_brain.graph import GraphStore, InvalidRelation


def _graph(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path / "graph.db")


def test_link_and_neighbors(tmp_path):
    g = _graph(tmp_path)
    g.link("concepts/a", "concepts/b", "is-a", confidence=0.9, rationale="stated directly")
    neighbors = g.neighbors("concepts/a")
    assert len(neighbors) == 1
    assert neighbors[0].dst == "concepts/b"
    assert neighbors[0].relation == "is-a"
    # bidirectional: querying from the other endpoint finds the same edge
    assert g.neighbors("concepts/b")[0].src == "concepts/a"


def test_rejects_invalid_relation(tmp_path):
    g = _graph(tmp_path)
    with pytest.raises(InvalidRelation):
        g.link("a", "b", "loves")


def test_link_auto_creates_endpoint_nodes(tmp_path):
    g = _graph(tmp_path)
    g.link("concepts/a", "concepts/b", "depends-on")
    assert {"concepts/a", "concepts/b"} <= g.all_node_ids()


def test_orphans(tmp_path):
    g = _graph(tmp_path)
    g.upsert_node("concepts/lonely", "Lonely Concept")
    g.link("concepts/a", "concepts/b", "depends-on")
    orphans = g.orphans()
    assert "concepts/lonely" in orphans
    assert "concepts/a" not in orphans


def test_contradictions_and_low_confidence(tmp_path):
    g = _graph(tmp_path)
    g.link("concepts/a", "concepts/b", "contradicts", confidence=1.0)
    g.link("concepts/c", "concepts/d", "is-a", confidence=0.3)
    assert len(g.contradictions()) == 1
    low = g.low_confidence(threshold=0.6)
    assert len(low) == 1
    assert low[0].src == "concepts/c"


def test_traverse_is_directed_not_bidirectional(tmp_path):
    """Regression test: neighbors() is bidirectional (for context rendering),
    but traverse() must only follow src->dst, or 'walk every part-of ancestor'
    would wander backward into descendants too."""
    g = _graph(tmp_path)
    g.link("concepts/a", "concepts/b", "part-of")
    g.link("concepts/b", "concepts/c", "part-of")
    g.link("concepts/z", "concepts/a", "part-of")  # z -> a, should NOT appear walking from a

    reached = g.traverse("concepts/a", "part-of", depth=2)
    assert reached == ["concepts/b", "concepts/c"]
    assert "concepts/z" not in reached


def test_confidence_is_clamped_to_unit_interval(tmp_path):
    g = _graph(tmp_path)
    edge = g.link("a", "b", "is-a", confidence=5.0)
    assert edge.confidence == 1.0
    edge2 = g.link("a", "c", "is-a", confidence=-1.0)
    assert edge2.confidence == 0.0


def test_render_context_dedupes_and_formats(tmp_path):
    g = _graph(tmp_path)
    g.link("concepts/a", "concepts/b", "is-a", confidence=0.8)
    ctx = g.render_context(["concepts/a", "concepts/b"])
    assert ctx.count("concepts/a --is-a--> concepts/b") == 1
