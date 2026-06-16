# Advanced RAG Lab — every 2026 RAG technique as a runnable module

> **The one-sentence shift:** RAG moved *"from vibe-checking to reasoning."* Classic
> vector RAG solves *reasoning* problems with *geometry* tools — embedding similarity is
> great for factual lookup ("What's the GPA cutoff?") but breaks on logic-dependent
> questions (compare clauses across versions, temporal constraints, multi-hop). **Advanced
> RAG is the set of techniques that bolt reasoning, control, and verification onto retrieval.**

This lab teaches each technique as its **own small, runnable module** — no API key, no
heavy deps. Everything uses a shared offline harness ([`_harness.py`](_harness.py)): a toy
corpus, deterministic "embeddings" (with light synonym expansion so dense ≠ sparse), BM25,
a lexical cross-encoder stand-in, retrieval metrics, and a rule-based `FakeLLM`. Swap any
piece for a real embedder / reranker / LLM and the *algorithm is identical*.

```bash
# from agent-backend/admission-guide/
python -m advanced_rag.tier2_advanced.hybrid_rrf      # run any module directly
```

---

## The 4-tier ladder (the dominant 2026 mental model)

| Tier | What it is | Precision | Latency | Cost/query |
|---|---|---|---|---|
| **1 · Naive** | single vector search → stuff → generate | ~70–80% | <1s | ~$0.001 |
| **2 · Advanced** | hybrid search + reranking + query transforms | ~85–90% | 2–3s | ~$0.005 |
| **3 · Agentic** | LLM controls an iterative retrieve→evaluate→retrieve loop | ~90–95% | 5–15s | ~$0.01–0.05 |
| **4 · Adaptive** | a classifier routes each query to the cheapest tier that works | 90%+ avg | varies | optimized |

**Key insight:** agentic is ~10× the cost (4–6 LLM calls vs 1) and **60–70% of production
queries are simple** — so **Adaptive RAG** (route simple→cheap, complex→agentic) is the
current state of the art for real workloads.

---

## Module map

✅ = built & runnable now · 🔜 = next wave

### Tier 1 — Naive  ([tier1_naive/](tier1_naive/))
| Module | Teaches |
|---|---|
| ✅ [`naive_rag`](tier1_naive/naive_rag.py) | single dense retrieval → stuff → generate; and *why* it fails on codes/paraphrases |

### Tier 2 — Advanced retrieval  ([tier2_advanced/](tier2_advanced/))
| Module | Teaches |
|---|---|
| ✅ [`hybrid_rrf`](tier2_advanced/hybrid_rrf.py) | BM25 + dense fused with **Reciprocal Rank Fusion** (scale-free, rank-based) |
| ✅ [`reranking`](tier2_advanced/reranking.py) | **cross-encoder** rerank: retrieve wide → rerank narrow (biggest precision win) |
| ✅ [`query_transforms`](tier2_advanced/query_transforms.py) | rewrite/normalize, **HyDE**, multi-query, **decomposition**, step-back |
| ✅ [`contextual_retrieval`](tier2_advanced/contextual_retrieval.py) | **Anthropic Contextual Retrieval** — situate chunks before embedding |

### Tier 3 — Agentic & reasoning  ([tier3_agentic/](tier3_agentic/))
| Module | Teaches |
|---|---|
| ✅ `self_rag` | reflection gates — `IsRelevant` / `IsSupported` / `IsUseful` — to kill hallucination (legal/medical/finance) |
| ✅ `corrective_rag` | **CRAG** — grade retrieved context; if weak, re-retrieve or fall back (e.g. web) |
| ✅ `adaptive_rag` | the **complexity classifier/router** (simple→cheap, complex→agentic) |
| ✅ `multi_agent_rag` | a supervisor orchestrating specialist retrieval agents for cross-domain queries |
| ✅ `graph_rag` | entity–relationship graph for **multi-hop** reasoning without repeated retrieval rounds |

### Bleeding edge  ([bleeding_edge/](bleeding_edge/))
| Module | Teaches |
|---|---|
| ✅ `a_rag` | **hierarchical retrieval interfaces** — expose keyword/semantic/chunk-read *tools* to the model |
| ✅ `dci` | **Direct Corpus Interaction** — agent searches raw text with grep/file-reads (no vector index) |
| ✅ `latent_rag` | reasoning+retrieval in **latent space** (~90% lower latency vs explicit agentic loops) — conceptual |

### Production wrappers  ([production/](production/))
| Module | Teaches |
|---|---|
| ✅ `evaluation` | retrieval (recall@k / nDCG / MRR) **and** generation (faithfulness/completeness) — measured separately |
| ✅ `grounding` | citation grounding — every claim attributed to a chunk; unsupported claims flagged/refused |
| ✅ `cost_control` | cap agent iterations (~3), stream/async long runs, compress prior-round context |

---

## The concrete techniques (reference)

**Advanced tier:** hybrid (BM25 + dense) with RRF · cross-encoder reranking (retrieve
wide, rerank narrow) · query transforms (decomposition, multi-query, HyDE, step-back,
normalize) · Contextual Retrieval.

**Agentic tier:** Self-RAG (reflection gates) · Corrective RAG (grade → re-retrieve/fallback)
· Adaptive RAG (complexity router) · Graph RAG (entity graph, multi-hop) · Multi-Agent RAG
(supervisor + specialists).

**Bleeding edge:** A-RAG (hierarchical tool interfaces) · Direct Corpus Interaction
(grep/file-reads, no embeddings) · LatentRAG (continuous latent space).

**Non-negotiable production wrappers:** independent evaluation (retrieval vs generation
measured separately) · citation grounding (flag/refuse unsupported claims) · cost & latency
control (cap iterations, stream/async, compress context).

---

## How this maps to the Vision app

This lab is the *teaching* counterpart to the production system in `app/`:
- Tier 2 here ≙ Vision's hybrid + RRF + cross-encoder rerank + query understanding.
- Tier 3 `corrective_rag`/`self_rag` ≙ Vision's `grade_context` + grounding/refusal.
- `graph_rag` ≙ the deep-dive in [`code-n-concepts/99-graph-rag-complete-learning-guide.md`](../../../code-n-concepts/99-graph-rag-complete-learning-guide.md).
- The PageIndex navigator in `app/services/tree_search.py` is a cousin of `a_rag`
  (reasoning over structure instead of pure similarity).

## Suggested learning path
1. Run `tier1_naive.naive_rag` — feel the baseline and its two failure modes.
2. Walk Tier 2 in order: `hybrid_rrf` → `reranking` → `query_transforms` → `contextual_retrieval`.
3. Tier 3 to see *control & reasoning* (`self_rag`, `corrective_rag`, `graph_rag`); then `adaptive_rag` to see *when to use which*.
4. Skim `bleeding_edge/` for where research is heading (tools / grep / latent space).
5. Always end with `production/` — evaluation + grounding + cost control are what make it real, not a demo.
