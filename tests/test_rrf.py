"""RRF fusion — rank-based consensus across ranked lists. (Requires runtime deps.)"""

from app.services.multi_doc import _rrf


def test_rrf_rewards_consensus():
    # 'b' is high in both lists → should top the fusion.
    fused = _rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused[0] in ("a", "b")
    assert set(fused[:2]) == {"a", "b"}


def test_rrf_includes_all_items():
    fused = _rrf([["a", "b"], ["c"]], k=60)
    assert set(fused) == {"a", "b", "c"}


def test_rrf_single_list_preserves_order():
    assert _rrf([["x", "y", "z"]], k=60) == ["x", "y", "z"]
