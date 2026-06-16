# Retrieval Lab

Production retrieval techniques — a **standalone reference** (not wired into the
app). Runs **offline**: pure-Python BM25 + the embeddings lab's HashEmbedder for
the dense path; real models (cross-encoder, LLM transforms) load lazily.

Completes the trio: **chunking → embeddings → retrieval**. You chunk a corpus
(chunking lab), embed it (embeddings lab), then retrieve over it here.

## 1. The three base retrievers

| Retriever | Catches | Misses | Code |
|---|---|---|---|
| **BM25** (sparse/lexical) | exact terms — codes, IDs, names, jargon | paraphrases, synonyms | [bm25.py](bm25.py) |
| **Dense** (bi-encoder) | semantics — paraphrases, synonyms | rare exact tokens | [dense.py](dense.py) |
| **Hybrid** = BM25 + Dense → **RRF** | both | — | [hybrid.py](hybrid.py) |

The whole reason hybrid exists: each base method fails on the queries the other
handles. In the benchmark below, **hybrid beats both** alone.

## 2. RRF — fuse by rank, not score ([fusion.py](fusion.py))

BM25 scores are unbounded positives; cosine is `[-1, 1]`. Averaging raw scores
lets BM25 dominate. **Reciprocal Rank Fusion** uses only positions:

```
score(d) = Σ_lists  weight / (k + rank_in_list)      # k = 60, rank 0-based
```

No normalization, no tuning to start. Cormack et al., SIGIR 2009. Also used to
merge multi-query results (RAG-Fusion).

## 3. Query transforms ([query_transform.py](query_transform.py))

Retrieval is capped by the query. LLM-driven rewrites (injected `llm_fn`; degrade
to the original query offline):

| Transform | Fixes | Output |
|---|---|---|
| **MultiQuery / RAG-Fusion** | recall (one phrasing misses) | N paraphrases → retrieve all → RRF |
| **HyDE** | short-query↔long-doc gap | a hypothetical answer used as the query |
| **Decomposition** | multi-part questions | sub-questions |
| **StepBack** | too-concrete queries | a broader concept question + original |

## 4. Two-stage reranking ([rerank.py](rerank.py))

```
retrieve top ~100 (recall)  →  cross-encoder rerank to top ~10 (precision)  →  LLM
```

A **cross-encoder** reads (query, doc) *together* → +5–15 NDCG@10 (BGE-reranker /
Cohere Rerank). Also included: **ColBERT** late-interaction (MaxSim) and a
dependency-free **lexical** baseline so the lab runs offline.

## 5. Metrics ([base.py](base.py))

You don't *choose* a retriever — you *measure* it.

| Metric | Question | Target (2026) |
|---|---|---|
| **recall@k** | did the relevant docs reach top-k? (RAG ceiling) | ≥0.8 @ k=20 (broad) |
| **precision@k** | how clean is the top-k? | ≥0.7 @ k=5 (narrow) |
| **MRR** | how early is the first hit? | higher = better |
| **nDCG@k** | position-weighted, graded — best correlates with end-to-end RAG | higher = better |

## Quick start

```bash
cd agent-backend/admission-guide

python -m retrieval.cli --strategy hybrid --query "minimum GPA for CS"
python -m retrieval.cli --strategy bm25  --query "TOEFL"
python -m retrieval.cli --strategy hybrid --rerank lexical
python -m retrieval.cli --strategy hybrid --transform multi_query   # needs OPENAI_API_KEY

python -m retrieval.benchmark --k 5
```

Sample benchmark (offline hash embedder — real models lift dense/rerank):

```
strategy        recall@5    ndcg@5      mrr
bm25            1.0         0.938       0.917
dense           1.0         0.794       0.722
hybrid          1.0         1.0         1.0     ← fusion wins
hybrid+rerank   1.0         1.0         1.0
```

## Decision cheat-sheet
- Start with **hybrid (BM25 + dense) + RRF (k=60)**, ~20 candidates each. It's the robust default.
- Jargon/codes corpus → give **BM25 more candidates**; conversational → give **dense** more.
- Add a **cross-encoder rerank** (retrieve 100 → rerank 10) for the biggest precision lift.
- Queries vague/complex → add a **query transform** (multi-query for recall, HyDE for short queries, decomposition for multi-part).
- **Measure recall@k + nDCG on your gold set** before/after each addition — keep only what moves the metric.

## Layout
```
retrieval/
├── base.py            # Retriever ABC, Doc/RetrievedDoc, IR metrics
├── bm25.py            # pure-Python Okapi BM25
├── dense.py           # bi-encoder retrieval (reuses embeddings lab)
├── fusion.py          # Reciprocal Rank Fusion
├── hybrid.py          # BM25 + dense → RRF
├── query_transform.py # HyDE / multi-query / decomposition / step-back
├── rerank.py          # cross-encoder / ColBERT / lexical
├── pipeline.py        # transform → retrieve → fuse → rerank
├── factory.py         # flag selectors
├── sample_data.py     # tiny labeled corpus (QRELS)
├── cli.py             # flag-driven demo
└── benchmark.py       # recall@k / nDCG / MRR across strategies
```

## References
- RRF — Cormack et al., SIGIR 2009; Azure AI Search hybrid ranking docs.
- Hybrid search 2026 (Digital Applied, Superlinked VectorHub); BM25+dense+RRF.
- Query transforms — LangChain "Query Transformations"; HyDE (Gao et al.); Step-Back (Google).
- Rerankers — Pinecone two-stage retrieval; BGE-reranker-v2-m3; Cohere Rerank 3.
- Metrics — recall@k / MRR / nDCG (Weaviate, Towards Data Science).
