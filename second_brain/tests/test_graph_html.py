import json
import re
from pathlib import Path

from second_brain.graph import GraphStore
from second_brain.graph_html import render_graph_html, write_graph_html
from second_brain.store import WikiStore


def _seeded(tmp_path: Path) -> tuple[WikiStore, GraphStore]:
    store = WikiStore(tmp_path)
    store.write_page("concepts/a.md", title="Alpha", tldr="Alpha thing.", content="Alpha body.")
    store.write_page("concepts/b.md", title="Beta", tldr="Beta thing.", content="Beta body.")
    graph = GraphStore(store.wiki_dir / "graph.db")
    graph.upsert_node("concepts/a", "Alpha")
    graph.upsert_node("concepts/b", "Beta")
    graph.link("concepts/a", "concepts/b", "depends-on", confidence=0.8, rationale="a needs b")
    return store, graph


def _extract_inline_json(html: str, var_name: str) -> object:
    m = re.search(rf"const {var_name} = (.*?);\n", html)
    assert m, f"couldn't find {var_name} in generated HTML"
    # json.loads natively unescapes the JSON-spec-legal "\/" back to "/" -- the
    # same way JS's own literal-array parsing would, no manual reversal needed.
    return json.loads(m.group(1))


def test_render_includes_verified_cdn_integrity_hash(tmp_path):
    store, graph = _seeded(tmp_path)
    html = render_graph_html(graph, store)
    assert "unpkg.com/vis-network" in html
    assert 'integrity="sha384-' in html


def test_render_embeds_correct_node_and_edge_data(tmp_path):
    store, graph = _seeded(tmp_path)
    html = render_graph_html(graph, store)

    nodes = _extract_inline_json(html, "NODES")
    edges = _extract_inline_json(html, "EDGES")

    ids = {n["id"] for n in nodes}
    assert ids == {"concepts/a", "concepts/b"}
    labels = {n["id"]: n["label"] for n in nodes}
    assert labels["concepts/a"] == "Alpha"
    assert next(n for n in nodes if n["id"] == "concepts/a")["title"] == "Alpha thing."

    assert len(edges) == 1
    assert edges[0]["from"] == "concepts/a"
    assert edges[0]["to"] == "concepts/b"
    assert edges[0]["relation"] == "depends-on"
    assert edges[0]["dashes"] is False  # confidence 0.8 >= 0.6 threshold


def test_orphan_nodes_are_flagged(tmp_path):
    store, graph = _seeded(tmp_path)
    graph.upsert_node("concepts/lonely", "Lonely")
    store.write_page("concepts/lonely.md", title="Lonely", tldr="No links.", content="x")

    html = render_graph_html(graph, store)
    nodes = _extract_inline_json(html, "NODES")
    lonely = next(n for n in nodes if n["id"] == "concepts/lonely")
    assert lonely["orphan"] is True
    connected = next(n for n in nodes if n["id"] == "concepts/a")
    assert connected["orphan"] is False


def test_low_confidence_edge_is_dashed(tmp_path):
    store, graph = _seeded(tmp_path)
    graph.link("concepts/b", "concepts/a", "contradicts", confidence=0.3)
    html = render_graph_html(graph, store)
    edges = _extract_inline_json(html, "EDGES")
    weak = next(e for e in edges if e["relation"] == "contradicts")
    assert weak["dashes"] is True


def test_script_injection_in_label_cannot_close_the_script_tag(tmp_path):
    store, graph = _seeded(tmp_path)
    graph.upsert_node("concepts/evil", "</script><script>alert(1)</script>")
    html = render_graph_html(graph, store)
    assert "</script><script>alert(1)" not in html
    # still parses back to the original (unescaped) string once JS would un-escape it
    nodes = _extract_inline_json(html, "NODES")
    evil = next(n for n in nodes if n["id"] == "concepts/evil")
    assert evil["label"] == "</script><script>alert(1)</script>"


def test_write_graph_html_creates_file(tmp_path):
    store, graph = _seeded(tmp_path)
    out = write_graph_html(graph, store, tmp_path / "out" / "graph.html")
    assert out.exists()
    assert "vis-network" in out.read_text(encoding="utf-8")


def test_empty_graph_renders_without_error(tmp_path):
    store = WikiStore(tmp_path)
    graph = GraphStore(store.wiki_dir / "graph.db")
    html = render_graph_html(graph, store)
    nodes = _extract_inline_json(html, "NODES")
    assert nodes == []
