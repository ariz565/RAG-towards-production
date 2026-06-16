# Tier 1 — Naive RAG

The baseline every other tier improves on.

## `naive_rag.py`
**Working:** `query → embed → top-k dense similarity → concatenate chunks → one LLM call → answer`.
One retrieval, one generation. No fusion, no rerank, no verification.

**Use cases:** FAQ bots and semantic document Q&A where the answer lives in one or two
chunks that are close to the question. Cheapest/fastest tier (~1 LLM call, <1s) — and right
for the ~60–70% of production queries that are genuinely simple.

**Limitations (the demo shows two on purpose):**
1. **Exact terms / codes** like `SEC-17` get diluted by an embedding (geometry, not symbols)
   → sparse/BM25 handles these (Tier 2 `hybrid_rrf`).
2. **Vocabulary mismatch** — a paraphrase the embedding doesn't bridge simply misses.

These two failure modes motivate the entire Advanced tier.

```bash
python -m advanced_rag.tier1_naive.naive_rag
```
