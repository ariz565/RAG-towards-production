# RAG Fundamentals — a runnable study suite

Four standalone, web-researched, **offline-runnable** labs that together cover RAG
end-to-end, basic to advanced — built for the Senior AI Engineer interview bar
("walk me through RAG from scratch; how would you chunk / embed / retrieve /
evaluate?"). Each lab is modular, flag-selectable, benchmarked, and decoupled
from the production app.

| Lab | Covers | Run |
|---|---|---|
| [`chunking/`](chunking/README.md) | fixed · recursive · sentence · markdown · parent-child · semantic · **late** · **contextual** · **proposition** · **agentic** | `python -m chunking.benchmark` |
| [`embeddings/`](embeddings/README.md) | prefixes · pooling · normalize · **Matryoshka** · **int8/binary quant** · dense · **sparse (SPLADE)** · **multi-vector (ColBERT)** | `python -m embeddings.benchmark` |
| [`retrieval/`](retrieval/README.md) | BM25 · dense · **hybrid + RRF** · HyDE/multi-query/decomposition/step-back · **cross-encoder/ColBERT rerank** · recall@k/nDCG/MRR | `python -m retrieval.benchmark` |
| [`evaluation/`](evaluation/README.md) | RAGAS-style **faithfulness · answer-relevancy · context-precision · context-recall** | `python -m evaluation.cli` |

## The end-to-end flow

```
            INDEX TIME                                    QUERY TIME
  ┌──────────────────────────────┐         ┌──────────────────────────────────────┐
  │ documents                    │         │ user question                        │
  │   │ parse (layout-aware)     │         │   │ query transform ── retrieval/    │
  │   ▼                          │         │   ▼   (HyDE / multi-query / …)        │
  │ CHUNK ───────── chunking/    │         │ RETRIEVE ─────────── retrieval/      │
  │   │  (recursive / parent-    │         │   │  BM25 + dense                    │
  │   │   child / semantic / …)  │         │   ▼                                  │
  │   ▼                          │         │ FUSE (RRF) ───────── retrieval/      │
  │ EMBED ───────── embeddings/  │         │   │                                  │
  │   │  (prefix → pool →        │         │   ▼                                  │
  │   │   MRL → normalize →      │         │ RERANK ───────────── retrieval/      │
  │   │   quantize)              │         │   │  (cross-encoder)                 │
  │   ▼                          │         │   ▼                                  │
  │ INDEX (vector store) ────────┼────────▶│ top-k context → LLM → ANSWER         │
  └──────────────────────────────┘         │            │                         │
                                            │            ▼                         │
                                            │ EVALUATE ─────────── evaluation/     │
                                            │  retrieval: recall@k / nDCG / MRR    │
                                            │  generation: faithfulness / ans-rel /│
                                            │              ctx-precision / recall  │
                                            └───────────────┬──────────────────────┘
                                                            ▼
                                            metrics feed back → fix the weakest stage
```

## How the labs connect (it's one system, not four demos)

- `retrieval/dense.py` **embeds with `embeddings/`** (`HashEmbedder` offline; swap a real model in one line).
- `retrieval/rerank.py` **ColBERT path reuses `embeddings/multivector.py`** (MaxSim).
- `chunking/` produces the documents that `retrieval/` indexes and `evaluation/` scores.
- LLM-driven pieces (agentic/contextual chunking, HyDE/decomposition, RAGAS judges) share **`chunking/llm_util.py`** (injectable `llm_fn`, lazy OpenAI).
- Everything runs offline through dependency-free fallbacks (hash embedder, lexical rerank/judge), and upgrades to real models/APIs by injection.

## Decision cheat-sheets (condensed)

- **Chunk:** default **recursive** (512 tok, ~15% overlap); context-loss → **late/contextual**; fact-dense → **proposition**; answers need context → **parent-child**.
- **Embed:** strong dense model + **right prefix** + **L2 normalize**; storage pressure → **int8** then **MRL** (MRL-trained models only); exact-term robustness → add **sparse**.
- **Retrieve:** **hybrid (BM25+dense) + RRF k=60**, then **cross-encoder rerank** (retrieve 100 → rerank 10); vague/complex queries → a **query transform**.
- **Evaluate:** gate on **faithfulness**; diagnose with the map below.

## The evaluation loop (where to look when RAG is bad)

| Symptom | Likely culprit | Fix in lab |
|---|---|---|
| low **context_recall** | retrieval/chunking missed it | `chunking/` (strategy/size), `retrieval/` (hybrid, transforms) |
| low **context_precision** | right docs ranked low | `retrieval/` (add reranker) |
| low **faithfulness** | hallucination / weak context | prompt + grounding; check context_recall first |
| low **answer_relevancy** | answer drifts off-question | prompt / generation |

## The 60-second interview narrative

> "RAG quality is a chain — parse → chunk → embed → retrieve → rerank → generate —
> and it's only as strong as its weakest link, so I instrument the whole chain.
> I chunk with recursive-or-parent-child by default and reach for semantic/late/
> contextual when evals justify the cost. I embed with a strong model using the
> correct query/passage prefixes and L2 normalization, and I lean on Matryoshka +
> int8 to cut storage. I retrieve hybrid (BM25 + dense) fused with RRF because
> each catches what the other misses, then cross-encoder rerank the top 100 to 10.
> I add query transforms only when the query is the bottleneck. And I never guess —
> I measure recall@k / nDCG for retrieval and faithfulness / answer-relevancy /
> context-precision / recall for generation, then fix whichever link the metrics
> say is weakest."

## Where these are wired in production

The main app (`app/`) applies these ideas for real: hybrid retrieval + RRF, a
cross-encoder rerank node, span-level grounding (faithfulness) with refuse-below-
threshold, contextual-retrieval ingestion, and a DeepEval CI gate in `evals/`
enforcing the 2026 thresholds. The labs are the *teaching* decomposition; the app
is the *integrated* system. See [VISION-6-TO-10.md](VISION-6-TO-10.md).
