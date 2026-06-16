"""MMR diversity — greedy relevance/redundancy selection. (Embeddings monkeypatched → runs offline.)"""

from app.services import diversity

# Deterministic 3-d vectors. Query aligns with both the 'a' axis and the 'b' axis,
# so 'a' and 'b' are equally relevant; 'a2' is a near-duplicate of 'a'; 'c' is off-axis.
_VECS = {"a": [1.0, 0.0, 0.0], "a2": [0.99, 0.01, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}


async def test_mmr_prefers_diversity_over_near_duplicate(monkeypatch):
    async def fake_query(q, **kw):
        return [1.0, 1.0, 0.0]

    async def fake_texts(texts, **kw):
        return [_VECS[t] for t in texts]

    monkeypatch.setattr(diversity, "embed_query", fake_query)
    monkeypatch.setattr(diversity, "embed_texts", fake_texts)

    candidates = [{"text": "a"}, {"text": "a2"}, {"text": "b"}, {"text": "c"}]
    picked = await diversity.mmr_select("q", candidates, k=2, lambda_=0.6)
    chosen = {c["text"] for c in picked}

    # Diversity behavior: take the orthogonal-but-relevant 'b', keep only ONE of the
    # near-duplicate a-cluster, and never the irrelevant 'c'.
    assert "b" in chosen
    assert "c" not in chosen
    assert len(chosen & {"a", "a2"}) == 1


async def test_mmr_noop_when_pool_fits(monkeypatch):
    # When candidates <= k there's nothing to diversify; embeddings shouldn't even be needed.
    candidates = [{"text": "a"}, {"text": "b"}]
    picked = await diversity.mmr_select("q", candidates, k=5)
    assert picked == candidates
