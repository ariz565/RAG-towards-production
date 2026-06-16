# Vision — Multi-tenant Document Intelligence

Sign up, upload documents (policies, manuals, versioned legal docs, large PDFs),
and ask questions or request summaries. The system understands the query, routes
it to the right document(s), retrieves with a hybrid + reranked pipeline, grounds
the answer, and returns **citations + freshness** — tenant-isolated, guardrailed,
observable, and eval-gated.

## Docs
- 📐 [ARCHITECTURE.md](ARCHITECTURE.md) — full system, flow diagrams, end-to-end.
- 🧠 [RAG_FUNDAMENTALS.md](RAG_FUNDAMENTALS.md) — the standalone study labs (chunking · embeddings · retrieval · evaluation · agents).
- 🗺️ [VISION-6-TO-10.md](VISION-6-TO-10.md) — how it was built, hardening passes, roadmap.
- 🔬 [evals/README.md](evals/README.md) — the eval harness + CI gate.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # set provider keys; CHANGE auth_secret for prod
python -m app.main            # serves http://localhost:8001  (docs at /docs)
```

### End-to-end (signup → upload → ask → summarize)

```bash
BASE=http://localhost:8001

# 1) Sign up → get a bearer token (scoped to your tenant)
TOKEN=$(curl -s -X POST $BASE/api/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"me@acme.com","password":"supersecret"}' | jq -r .access_token)

# 2) Upload a PDF (stored per-tenant + indexed). Optional: version it.
curl -s -X POST $BASE/api/upload -H "Authorization: Bearer $TOKEN" \
  -F "file=@maternity_policy.pdf"
# versioned:
curl -s -X POST $BASE/api/upload -H "Authorization: Bearer $TOKEN" \
  -F "file=@leave_policy_2026.pdf" -F "family=leave_policy" -F "version=2026" -F "effective_date=2026-01-01"

# 3) Ask — routed to the right doc, with citations + freshness
curl -s -X POST $BASE/api/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many days of maternity leave do I get?"}'

# cross-document compare (fan-out + RRF):
curl -s -X POST $BASE/api/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Compare maternity vs paternity leave."}'

# version-aware ("as of" a date):
curl -s -X POST $BASE/api/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the leave entitlement?","family":"leave_policy","as_of":"2025-06-01"}'

# 4) Summarize a big PDF — async (returns job_id), then poll
JOB=$(curl -s -X POST $BASE/api/summarize/async -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"doc_id":"maternity_policy"}' | jq -r .job_id)
curl -s $BASE/api/jobs/$JOB -H "Authorization: Bearer $TOKEN"
```

> Auth is **optional** on read/index endpoints (falls back to the `default`
> tenant) so the bundled demo doc and CLI keep working without a token; it's
> **required** for `/api/upload`.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/signup`, `/api/auth/login` | Get a bearer token (tenant-scoped) |
| POST | `/api/upload` | Upload + index a PDF (auth; optional `family`/`version`) |
| POST | `/api/ask`, `/api/ask/stream` | Ask (router + understanding + pipeline); SSE stream |
| POST | `/api/ask/resume` | Resume a HITL-clarification run (`thread_id` + `answer`) |
| POST | `/api/summarize`, `/api/summarize/async` | Map-reduce summary (sync / job) |
| GET | `/api/jobs/{job_id}` | Poll an async job |
| GET | `/api/versions` | List version chains (`?family=`) |
| POST | `/api/index`, `/api/index/hybrid`, `/api/index/all` | Index a PDF already in storage |
| GET | `/api/documents` | List the tenant's documents |
| GET/POST | `/api/config` | View / switch active model·retrieval·embedding |
| GET | `/api/tree`, `/api/page/{n}` | Document tree / raw page |
| GET | `/api/health` | Health + counts |

## CLI (offline indexing / admin)

```bash
python -m app.cli index          # PageIndex (vectorless tree)
python -m app.cli index-hybrid   # BM25 + Qdrant hybrid
python -m app.cli update-hybrid  # incremental re-index (only changed chunks)
python -m app.cli index-all      # both
python -m app.cli profile        # (re)derive the document's domain profile
python -m app.cli config         # show config
```

## What it does (capabilities)
- **Multi-tenant**: each user's docs are isolated (`data/pdf/<tenant>/`, namespaced indexes + Qdrant collections).
- **Query understanding**: classify → normalize → conservative rewrite before retrieval.
- **Routing**: picks the right document; **fan-out + RRF** for cross-document `compare`.
- **Hybrid retrieval**: BM25 + dense → RRF → cross-encoder rerank → grounded answer with citations + `indexed_at`.
- **Summarization**: map-reduce for large PDFs (sync or async job).
- **Versioning**: families + `effective_date`; latest or `as_of` retrieval.
- **Guardrails**: prompt-injection block, untrusted-context handling, PII redaction, grounding refuse.
- **Durable + observable**: LangGraph checkpointer (resume/HITL), OpenTelemetry GenAI spans.
- **Eval-gated**: golden set + DeepEval CI gate ([evals/](evals/)).

## Configuration
Everything is env-toggleable — see [.env.example](.env.example) (providers, retrieval,
guardrails, router, summarization, auth, storage, observability, checkpointer).

## Tech
FastAPI · LangGraph · Qdrant · BM25 · sentence-transformers / OpenAI / Ollama + cross-encoder ·
PyMuPDF + pdfplumber · tiktoken · OpenTelemetry · DeepEval/Ragas.
