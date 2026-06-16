"""PageIndex navigation — pure-logic units. (Requires runtime deps for import.)

Covers the parts of the reasoning-based retrieval path that don't need an LLM:
id coercion/validation, tree flattening/rendering, and page→passage splitting.
"""

import pytest

from app.services import tree_search as ts


def test_coerce_ids_handles_list_csv_and_junk():
    assert ts._coerce_ids(["a", " b ", "", None]) == ["a", "b"]   # null/empty dropped
    assert ts._coerce_ids("a, b ,c") == ["a", "b", "c"]            # CSV string
    assert ts._coerce_ids("") == []
    assert ts._coerce_ids(42) == []
    assert ts._coerce_ids(None) == []


def test_dedup_preserves_order():
    assert ts._dedup(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_flatten_indexes_every_node():
    tree = [{"node_id": "a", "children": [
        {"node_id": "a1", "children": []},
        {"node_id": "a2", "children": [{"node_id": "a2x", "children": []}]},
    ]}]
    flat = ts._flatten(tree, {})
    assert set(flat) == {"a", "a1", "a2", "a2x"}


def test_render_level_marks_expandable_nodes():
    nodes = [
        {"node_id": "p", "title": "Parent", "start_physical_index": 1, "end_physical_index": 9,
         "children": [{"node_id": "c", "title": "Child"}]},
        {"node_id": "leaf", "title": "Leaf", "start_physical_index": 10, "end_physical_index": 10, "children": []},
    ]
    rendered = ts._render_level(nodes)
    assert "[p] Parent (pp. 1-9)" in rendered
    assert "can be expanded" in rendered            # parent advertises its subsections
    assert "[leaf] Leaf (pp. 10-10)" in rendered


@pytest.mark.asyncio
async def test_search_tree_empty_tree_is_safe():
    out = await ts.search_tree("anything", tree=[], valid_ids=set(), domain_context="")
    assert out["node_ids"] == [] and out["confidence"] == 0.0


@pytest.mark.asyncio
async def test_single_shot_validates_and_repairs(monkeypatch):
    """Hallucinated ids are dropped; a repair call recovers a valid one."""
    tree = [{"node_id": "real_1", "title": "Fees", "summary": "fee schedule",
             "start_physical_index": 3, "end_physical_index": 4, "children": []}]
    calls = {"n": 0}

    async def fake_chat_json(prompt, *, system=None, temperature=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"thinking": "t", "node_ids": ["hallucinated_id"], "confidence": 0.9}, 10
        return {"node_ids": ["real_1"]}, 5      # repair re-prompt returns a valid id

    monkeypatch.setattr(ts, "chat_json", fake_chat_json)
    out = await ts.search_tree("what are the fees?", tree=tree, valid_ids={"real_1"}, domain_context="fees doc")
    assert out["node_ids"] == ["real_1"]
    assert calls["n"] == 2                       # original + one repair
