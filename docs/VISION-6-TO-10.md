# Vision → UniDocs: The 6/10 → 10/10 Product Vision

> Grounded in the latest library docs & best practices as of June 2026.
> This is not "make the chatbot prettier." It's "move up the stack from a commoditized
> doc-RAG demo to a **measurable, trustworthy, multi-tenant document-intelligence platform**."

---

## What "10/10" actually means

A 10/10 here is **not** more features. It's four properties a senior/architect reviewer can verify in 60 seconds:

1. **Measured** — every retrieval and answer is scored against a golden set, gated in CI. No self-rated confidence.
2. **Trustworthy** — span-level citation attribution, refuse-below-threshold, no hallucinated facts.
3. **Durable & multi-tenant** — survives crashes, resumes, isolates documents/tenants, scales past one PDF in global memory.
4. **Observable** — OpenTelemetry GenAI traces + online evals, so you know *why* every answer happened.

The current system scores ~6/10: clean architecture, real multi-strategy retrieval, but **no evals (1/10), single-tenant globals (3/10), and a half-blind grounding check**. The roadmap below closes each gap with a current, named best practice.

---

## Scorecard → target

| Area | Now | Target | The upgrade |
|---|---|---|---|
| Evaluation / quality proof | 1 | 10 | Ragas (offline) + DeepEval (CI gate) + Langfuse online evals |
| Retrieval quality | 8 | 10 | Anthropic Contextual Retrieval + cross-encoder/ColBERT reranking + query transforms |
| Anti-hallucination / grounding | 5.5 | 10 | Full-page + span-level attribution, refuse-below-threshold |
| Production readiness | 3 | 9.5 | Multi-tenant, durable LangGraph 1.2 execution, persistent Qdrant, fix concurrency |
| Observability | 6.5 | 9.5 | OpenTelemetry GenAI semconv + Langfuse tracing/datasets |
| Agentic depth | 7 | 9 | HITL interrupts, query planning, corrective RAG loop |
| Differentiation (2026) | 4 | 9 | Vectorless-vs-hybrid public benchmark + verification spine |

---

## Target architecture (advanced)

```
                          ┌─────────────────────────────────────────────┐
  Request (tenant, doc) ──▶│  FastAPI gateway: auth, tenant ctx, rate-limit│
                          └───────────────┬─────────────────────────────┘
                                          ▼
              ┌──────────── LangGraph 1.2 durable pipeline (per-run checkpoint) ───────────┐
              │  guardrail → query_transform → retrieve → rerank → answer →                 │
              │              grounding(span-level) → [refuse | HITL interrupt | corrective] │
              └───────────────┬─────────────────────────────────────────────┬──────────────┘
                              ▼ (retrieval)                                   ▼ (every span)
        ┌───────────────────────────────────────────┐         ┌──────────────────────────────┐
        │ Qdrant 1.10+ (persistent, named vectors):  │         │ OpenTelemetry GenAI semconv    │
        │  dense + sparse + ColBERT(late-interaction)│         │  → Langfuse (traces, online    │
        │  per-tenant collection / payload filter    │         │     evals, datasets, prompts)  │
        └───────────────────────────────────────────┘         └──────────────────────────────┘
                              ▲                                              ▲
              ┌───────────────┴───────────────┐              ┌──────────────┴───────────────┐
              │ Ingestion: contextual chunking │              │ CI: Ragas/DeepEval golden set │
              │ (50–100 tok context per chunk) │              │ gate — block PR on regression │
              └────────────────────────────────┘              └───────────────────────────────┘
```

---

## Phase plan (6 → 10), each grounded in a current practice

> **Status:** Phase A ✅ · Phase B ✅ (incl. Contextual Retrieval, opt-in) · Phase C ✅ (multi-tenant registry, persistent Qdrant, durable checkpointer, concurrency fix) · Phase D ✅ (OpenTelemetry GenAI spans, no-op when disabled) · Phase E ✅ (strategy benchmark, HITL clarification interrupts + `/api/ask/resume`, corrective-RAG context grading — both opt-in). **All phases A–E complete.**
>
> **Hardening pass (code review):** fixed event-loop-blocking embeddings (now run in executor), unbounded answer context (now char-budgeted), sync→**async** SqliteSaver under `ainvoke`, cross-page chunking + **tiktoken** sizing, Qdrant embedding-dim/collection mismatch guard, `pipeline_steps` **operator.add reducer**, Ollama embeddings wired, LLM **timeouts + retry/backoff**, `cosine_similarity` zero-guard, and a config-mutation lock.
>
> **Document-intelligence pass:** added a **document router** ([doc_router.py](app/services/doc_router.py)) — `/api/ask` with no `doc_id` routes to the best-matching document per tenant via profile-embedding + lexical scoring (multi-policy Q&A) — and **map-reduce summarization** ([summarize.py](app/services/summarize.py) + `/api/summarize`) for whole-document summaries of large PDFs (MAP per chunk → hierarchical REDUCE tree → final).
>
> **Query understanding + cross-doc + versioning + async (built):**
> - **Query understanding** ([query_understanding.py](app/services/query_understanding.py)): classify → normalize → conservative rewrite in front of `/ask` (chit-chat skips retrieval; `summarize` short-circuits to map-reduce; the rewritten query drives routing + retrieval).
> - **(b) Multi-doc fan-out** ([multi_doc.py](app/services/multi_doc.py)): `compare` queries fan out across the top-N routed docs, **RRF-fuse + cross-encoder rerank**, and answer with per-document citations.
> - **(a) Versioning** ([versions.py](app/services/versions.py)): per-tenant version manifest groups document **families**; upload registers `family`/`version`/`effective_date`; `/ask` with `family` (+ optional `as_of`) resolves to the latest/as-of version. `GET /api/versions` lists chains.
> - **(c) Async summarize** ([jobs.py](app/services/jobs.py)): `POST /api/summarize/async` → `job_id`; `GET /api/jobs/{id}` to poll — big PDFs don't block.
>
> **Tenancy & onboarding pass:** added **signup/login** (`/api/auth/*`, salted PBKDF2 + signed bearer tokens, stdlib-only) → each user gets a `tenant_id`; a **pluggable storage layer** ([storage.py](app/services/storage.py): `LocalStorage` per-tenant dirs now, S3-ready interface); a **`/api/upload`** endpoint (auth → store → index → register); and **tenant-namespaced artifacts** end-to-end (index/bm25/profile files + Qdrant collections keyed `<tenant>__<doc>`), with `ask`/`documents`/`tree`/`page` scoped to the caller's tenant. Auth is optional on read/index (default tenant) so the demo doc + CLI keep working.
>
> **Trust & freshness pass:** surfaced **chunk_id** in citations + **`indexed_at`** data-freshness on every answer; added an **input guardrail** (prompt-injection/jailbreak detection → `blocked` node), **indirect-injection hardening** (retrieved context wrapped + treated as untrusted data), **PII redaction** output guardrail, and **incremental embedding updates** (`content_hash` diffing + stable Qdrant point ids → re-embed only changed chunks via `HybridIndex.update` / `cli update-hybrid`).

### Phase A — Evals first (6 → 7). ✅ *The single highest-leverage change.*
The 2026 consensus: **"the difference between teams that ship good RAG and bad RAG is almost entirely whether they measure."**

- **Offline (tuning):** Ragas — `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. Use while tuning chunking/embeddings/reranker.
- **CI gate:** DeepEval as **pytest tests** on a curated golden set; block PRs that drop below threshold.
- **Production thresholds (2026 defaults):** faithfulness ≥ **0.75**, answer relevancy ≥ **0.8**, context precision ≥ **0.7**, context recall ≥ **0.8**.
- **Deliverable:** `evals/golden_set.jsonl` (50–100 admission Q&As with expected pages), `evals/test_rag.py` (DeepEval), GitHub Actions gate.
- Replaces the LLM self-scored "confidence" theater with real, defensible numbers.

### Phase B — Fix the grounding & retrieval truth (7 → 8).
- ✅ **Bug fixed:** grounding now reads the **full retrieved context** (char-budgeted), not a 500-char prefix, and does **claim/span-level verification** — it decomposes the answer into claims and checks each against the sources ([pipeline.py](app/services/pipeline.py) `grounding_check_node`).
- ✅ **Reranking:** new cross-encoder `rerank` node ([reranker.py](app/services/reranker.py)) casts a wide net then keeps the top-k; local `ms-marco-MiniLM` by default, degrades gracefully if unavailable. Configurable (`RERANK_*`).
- ✅ **Refuse-below-threshold:** new `insufficient_grounding` node declines honestly (lists unverifiable claims + pages checked) instead of emitting a confident hallucination, gated by `grounding_accept_threshold` / `grounding_refuse_threshold`.
- ✅ **Contextual Retrieval (Anthropic):** ingestion now generates a short situating context per chunk and prepends it to the **embed/BM25 text** (display text preserved) — [ingestion.py](app/services/ingestion.py) `_contextualize_chunks` + `Chunk.embed_text`. Opt-in via `CONTEXTUAL_RETRIEVAL_ENABLED=true` (adds 1 LLM call/chunk at index time; re-index to take effect).

### Phase C — Durable, multi-tenant backend (8 → 9). *Kills the worst weaknesses.*
- ✅ **Concurrency bug fixed:** no more mutating global `settings.active_model` at request time. Per-request provider/model flow through **task-isolated `contextvars`** ([llm.py](app/services/llm.py) `use_request_model` / `current_model_name`).
- ✅ **Multi-tenant:** module singletons removed; a [CorpusRegistry](app/services/registry.py) holds per-`(tenant_id, doc_id)` `DocumentBundle`s, the active one selected per request via an **active-document contextvar** (kept out of state so the checkpointer stays serializable). Endpoints/CLI are multi-document; new `GET /api/documents`.
- ✅ **Persistent Qdrant:** a **shared client** with **per-document collections** ([hybrid_retrieval.py](app/services/hybrid_retrieval.py) `get_qdrant_client`); persisted collections are reused on boot — **no more re-embedding every chunk** (`_build_vectors_from_dicts` skips when populated). `QDRANT_IN_MEMORY=false` by default.
- ✅ **Durable execution (LangGraph checkpointer):** [pipeline.py](app/services/pipeline.py) compiles with a **SqliteSaver** (resume/replay; graceful fallback to in-memory), every run carries a `thread_id`.
- ⏳ **Caching (optional):** embedding / rerank / answer caches for repeated queries — deferred.

### Phase D — Observability & online evals (9 → 9.5).
- ✅ **OpenTelemetry GenAI semantic conventions (`gen_ai.*`)** — [observability.py](app/services/observability.py) emits `gen_ai.chat` spans (system/model/tokens) for every LLM call, a `rag.request` root span (strategy/tenant/doc/confidence/grounded), and per-node spans. **No-op + zero-dependency** unless `OTEL_ENABLED=true`.
- ✅ **Vendor-neutral export:** OTLP exporter ships spans to **Langfuse / Datadog / Honeycomb** (`OTEL_EXPORTER_OTLP_ENDPOINT`) or console for local dev — no vendor lock-in.
- ⏳ **Online evals (optional):** running faithfulness/relevancy on sampled production traffic (via Langfuse) — deferred; the CI gate + traces already cover the core.

### Phase E — Agentic depth + the differentiator (9.5 → 10).
- ✅ **THE standout — strategy benchmark:** [benchmark.py](evals/benchmark.py) runs the golden set through PageIndex / Hybrid / BM25 / Vector and writes a reproducible `results/benchmark.md` + JSON (page recall/precision, grounded rate, OOS accuracy, latency, tokens). A citable artifact, not a tutorial clone.
- ✅ **HITL clarification interrupts:** a `clarify` node detects ambiguous questions and `interrupt()`s the graph; the response carries the clarifying question + `thread_id`; **POST `/api/ask/resume`** continues from the checkpoint with the user's answer. Opt-in via `HITL_ENABLED`.
- ✅ **Corrective-RAG:** a `grade_context` node scores retrieved-context relevance *before* answering and re-retrieves (via `query_rewrite`) when it's too weak, bounded by `max_retrieval_attempts`. Opt-in via `CORRECTIVE_RAG_ENABLED`.

---

## Technology choices (latest, named)

| Concern | Choice (June 2026) | Why |
|---|---|---|
| Orchestration | **LangGraph 1.2** | Durability modes, checkpointing, interrupts, replay — the framework the industry settled on. |
| Offline eval | **Ragas** | Reference-free RAG metrics; best for tuning chunking/embeddings. |
| CI eval gate | **DeepEval** (pytest) | Quality gate on golden set; blocks regressions in PRs. |
| Online eval + tracing | **Langfuse** + **OTel GenAI semconv** | Standard, no lock-in, LLM-as-judge on prod traffic. |
| Vector store | **Qdrant 1.10+** | Universal Query, sparse + dense + ColBERT named vectors, persistent. |
| Retrieval quality | **Contextual Retrieval** + **reranker** (Cohere Rerank 3 / bge / ColBERT) | −67% retrieval failures with rerank; biggest quality lever. |
| Hybrid fusion | **RRF** (keep), BM42 only experimentally | RRF is the production-safe default; BM42 still experimental per Qdrant. |

---

## Definition of done (acceptance criteria for 10/10)

- [ ] Golden set of ≥50 Q&As; DeepEval CI gate enforcing the 2026 thresholds; PRs blocked on regression.
- [ ] Faithfulness ≥ 0.75, context precision ≥ 0.7, context recall ≥ 0.8 on the golden set — **published in the README**.
- [ ] Span-level citations; system **refuses** (with reason) below confidence threshold; zero hallucinated facts on the golden set.
- [ ] No global mutable state; concurrent multi-provider requests pass an isolation test.
- [ ] Multi-tenant: ≥3 documents served concurrently with isolation.
- [ ] Durable: kill the process mid-run, restart, resume from checkpoint.
- [ ] OTel traces visible in Langfuse for every request, with token/cost/latency per node.
- [ ] **Public benchmark**: PageIndex vs Hybrid — accuracy/latency/cost table, reproducible.

---

## Why this is the 2026 standout move

A polished React widget keeps this in the "doc-RAG chatbot" bucket the market is commoditizing. **Evals + verification + the vectorless benchmark** move it into the bucket that *doesn't* commoditize: provable quality and trust. That's the exact signal — "this person measures and can be trusted with production" — that separates AI Engineer from Architect.

---

### Sources
- LangGraph durable execution / 1.2 — https://docs.langchain.com/oss/python/langgraph/durable-execution
- Ragas metrics — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- DeepEval vs Ragas / CI gating (2026) — https://qaskills.sh/blog/deepeval-vs-ragas-rag-evaluation-2026
- Anthropic Contextual Retrieval — https://www.anthropic.com/news/contextual-retrieval
- Advanced RAG patterns 2026 (reranking, corrective RAG) — https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo
- Qdrant Universal Query / ColBERT (1.10) — https://qdrant.tech/blog/qdrant-1.10.x/
- Qdrant hybrid + reranking — https://qdrant.tech/documentation/tutorials-search-engineering/reranking-hybrid-search/
- OpenTelemetry GenAI semantic conventions — https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Langfuse + LangGraph tracing — https://langfuse.com/guides/cookbook/example_langgraph_agents
