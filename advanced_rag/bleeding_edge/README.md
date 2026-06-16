# Bleeding edge — newer research directions

Where 2026 research is pushing RAG beyond the embed→retrieve→stuff loop. These are
directional (one is an explicit conceptual stand-in), but each is runnable and shows the
core idea.

| Module | Working | Use case | Limitation |
|---|---|---|---|
| [`a_rag`](a_rag.py) | Expose retrieval as **tools at different granularities** — `keyword_search`, `semantic_search`, `chunk_read` — and let the agent plan which to call and when to read. | Heterogeneous queries where the right *access pattern* varies; multi-hop search→read→search. | More round-trips; cap tool calls to avoid loops. |
| [`dci`](dci.py) | **Direct Corpus Interaction** — `grep` + `read` over raw text, **no embeddings, no index**. | Exact/symbolic queries (codes, IDs, logs, code) and as a zero-build fallback. | No semantic recall — pair with semantic search for paraphrases. |
| [`latent_rag`](latent_rag.py) | Multi-hop walk in **latent space**: fold retrieved doc vectors into a latent state with cheap vector ops, **decode once** at the end (no per-hop LLM call). | Latency/cost-sensitive multi-hop where an explicit agent's per-step LLM calls dominate. | Less interpretable; needs a purpose-built model — shown as a labeled concept. |

```bash
python -m advanced_rag.bleeding_edge.a_rag
python -m advanced_rag.bleeding_edge.dci
python -m advanced_rag.bleeding_edge.latent_rag
```

**The throughline:** Tier 2–3 still treat retrieval as a black box the *pipeline* drives.
The bleeding edge hands control of *how to access the corpus* to the model — via tools
(A-RAG), via raw-text ops (DCI), or by keeping the whole loop in vectors (LatentRAG). The
A-RAG tool pattern is the most production-ready today and mirrors Vision's PageIndex
navigator (reasoning over structure instead of pure similarity).
