# Tier 2 — Advanced retrieval

Everything here improves **retrieval quality** before the LLM ever runs. This is where
most production systems live (~85–90% precision) and it's the highest ROI per unit effort.

| Module | Working | Use case |
|---|---|---|
| [`hybrid_rrf`](hybrid_rrf.py) | Run BM25 (exact terms) + dense (meaning) and fuse ranks with **RRF** `Σ 1/(rank+k)`. Rank-based → no score normalization, robust. | The default retriever. Sparse catches codes/IDs (`SEC-17`, `order 991`); dense catches paraphrases (`time off`~`leave`). |
| [`reranking`](reranking.py) | Two-stage: retrieve **wide** (top-20, recall) → **cross-encoder** scores each (query, passage) jointly → keep **narrow** (top-3, precision). | When the right passage is in the pool but not the top-3 the LLM reads. Biggest precision win after hybrid. |
| [`query_transforms`](query_transforms.py) | Reshape the query: **rewrite/normalize** (conservative), **HyDE** (embed a hypothetical answer), **multi-query** (paraphrase + fuse), **decomposition** (split compare/multi-part), **step-back** (ask the general question first). | Short/ambiguous queries (HyDE), vocab mismatch (multi-query), "compare A and B" (decomposition), reasoning (step-back). |
| [`contextual_retrieval`](contextual_retrieval.py) | Prepend an LLM-generated, chunk-situating context **before embedding** ("This chunk is from SEC-17…"). Index-time only; query path unchanged. | Long structured docs (policies/contracts/manuals) where chunks say "this section / the policy" without naming it. |

```bash
python -m advanced_rag.tier2_advanced.hybrid_rrf
python -m advanced_rag.tier2_advanced.reranking
python -m advanced_rag.tier2_advanced.query_transforms
python -m advanced_rag.tier2_advanced.contextual_retrieval
```

**Key intuitions to take away**
- **RRF over weighted sum:** BM25 and cosine scores live on different scales; fusing by
  *rank* is scale-free and robust.
- **Retrieve wide, rerank narrow:** bi-encoders shortlist fast (lossy); cross-encoders
  re-order accurately (slow) — use each for what it's good at.
- **Transforms are conservative:** aggressive rewriting can *poison* recall; keep meaning
  intact and measure on an eval set.
- **Contextual Retrieval is index-time:** one cheap call per chunk (cacheable), big recall
  win, no query-time latency cost.
