"""GET /api/second-brain/graph — read-only bridge into second_brain's graph.db.

(Requires runtime deps; runs in CI / a real env. Couldn't execute this in the
sandbox this was written in -- no fastapi/pydantic installed there -- so this is
verified by direct code review + second_brain's own test suite (which passed),
not by an actual test run. Please run this for real before trusting it.)
"""

import second_brain
from app.routers.second_brain import get_graph
from second_brain.graph import GraphStore
from second_brain.store import WikiStore


def test_empty_workspace_returns_empty_shape_without_creating_a_db_file(tmp_path, monkeypatch):
    monkeypatch.setattr(second_brain, "WORKSPACE_DIR", tmp_path)
    result = get_graph()
    assert result == {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}
    # a GET must not have the side effect of creating graph.db via GraphStore's
    # own create-on-connect behavior
    assert not (tmp_path / "wiki" / "graph.db").exists()


def test_populated_graph_returns_node_and_edge_records_matching_graph_html(tmp_path, monkeypatch):
    monkeypatch.setattr(second_brain, "WORKSPACE_DIR", tmp_path)

    store = WikiStore(tmp_path)
    store.write_page("concepts/a.md", title="Alpha", tldr="Alpha thing.", content="Alpha body.")
    store.write_page("concepts/b.md", title="Beta", tldr="Beta thing.", content="Beta body.")
    graph = GraphStore(store.wiki_dir / "graph.db")
    graph.upsert_node("concepts/a", "Alpha")
    graph.upsert_node("concepts/b", "Beta")
    graph.link("concepts/a", "concepts/b", "depends-on", confidence=0.8, rationale="a needs b")
    graph.close()

    result = get_graph()
    assert result["node_count"] == 2
    assert result["edge_count"] == 1
    assert {n["id"] for n in result["nodes"]} == {"concepts/a", "concepts/b"}
    alpha = next(n for n in result["nodes"] if n["id"] == "concepts/a")
    assert alpha["label"] == "Alpha"
    assert alpha["title"] == "Alpha thing."
    edge = result["edges"][0]
    assert edge["from"] == "concepts/a" and edge["to"] == "concepts/b"
    assert edge["relation"] == "depends-on"
    assert edge["dashes"] is False  # confidence 0.8 >= the 0.6 threshold
