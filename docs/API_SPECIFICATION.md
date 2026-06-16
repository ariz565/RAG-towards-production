# Vision — API & Configuration Specification

Complete contract for building the frontend. Every endpoint, every request/response
field, and every environment setting — annotated with the **UI control** a
configuration panel should render and whether a value is **runtime-mutable** or
**restart-scoped**.

- **Base URL:** `http://{HOST}:{PORT}` — default `http://localhost:8001`
- **Content type:** `application/json` unless noted (upload is `multipart/form-data`,
  streaming is `text/event-stream`).
- **Interactive docs (live):** `GET /docs` (Swagger UI) · `GET /openapi.json` (raw schema).
- **CORS:** allowed origins come from `CORS_ORIGINS` (see config). Add your frontend origin there.

---

## 1. Authentication

Auth is **optional but recommended**. Endpoints accept a **bearer token**; without one,
the caller is *anonymous* and pinned to the `DEFAULT_TENANT`.

- **Header:** `Authorization: Bearer <access_token>`
- **Token:** HMAC-signed, stdlib-only (no external JWT lib), TTL = `AUTH_TOKEN_TTL_SECONDS`.
- **Tenant isolation:** an authenticated caller is **pinned to their own tenant** — they
  cannot read/write another tenant's documents even by passing `tenant_id`. Anonymous
  callers may target a tenant or fall back to the default.

| Endpoint | Auth required? | Behavior |
|---|---|---|
| `/api/upload` | **Yes** | Rejects anonymous (tenant isolation for writes). |
| All other `/api/*` | Optional | Works anonymously against the default tenant; token scopes to your tenant. |

### POST `/api/auth/signup`
Create a user + tenant and return a token.

**Request** (`SignupRequest`):
```json
{ "email": "demo@acme.com", "password": "supersecret" }  // password min 8 chars
```
**Response** (`TokenResponse`):
```json
{ "access_token": "…", "token_type": "bearer", "tenant_id": "t_ab12…", "email": "demo@acme.com" }
```
**Errors:** `400` email already registered / invalid.

### POST `/api/auth/login`
**Request** (`LoginRequest`): `{ "email": "...", "password": "..." }`
**Response:** `TokenResponse` (as above).
**Errors:** `401` invalid email or password.

---

## 2. Documents — upload & indexing

### POST `/api/upload`  *(auth required)*
Upload a PDF into the caller's tenant storage and build the **hybrid** index. Optionally
register it as a version of a document family.

**Request:** `multipart/form-data`
| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file (`.pdf`) | yes | Only `.pdf` accepted (else `400`). |
| `family` | string | no | Document-family key for versioning. |
| `version` | string | no | Version label (e.g. `2026`). Registered only if `family` + `version` both present. |
| `effective_date` | string (ISO date) | no | Effective date for `as_of` resolution. |

**Response** (`UploadResponse`):
```json
{ "success": true, "doc_id": "maternity_policy", "tenant_id": "t_ab12…",
  "indexed": true, "message": "Uploaded + indexed (142 chunks). Registered as leave_policy v2026." }
```
**UI:** drag-drop uploader → show `message`; if `indexed:false`, surface the indexing error.

### POST `/api/index`  ·  POST `/api/index/hybrid`  ·  POST `/api/index/all`
Build an index for a PDF **already present** in tenant storage. `index` = PageIndex tree,
`index/hybrid` = BM25+vector, `index/all` = both.

**Request** (`IndexRequest`): `{ "filename": "admission-guide.pdf" }` (`doc_id` = filename stem)
**Response** (`IndexResponse`):
```json
{ "success": true, "doc_name": "admission-guide", "total_pages": 64, "tree_nodes": 28,
  "message": "All indexes built: 28 tree nodes + 142 hybrid chunks." }
```
**Errors:** `404` PDF not found, `500` indexing failed.

### GET `/api/documents`
List the documents in the caller's tenant.
```json
{ "tenant_id": "t_ab12…", "documents": [
  { "tenant_id": "t_ab12…", "doc_id": "maternity_policy", "total_pages": 12, "hybrid": true, "topics": ["leave","benefits"] }
] }
```

### GET `/api/versions?family={family}`
List registered versions (optionally filtered by family).
```json
{ "tenant_id": "t_ab12…", "versions": [ { "family":"leave_policy","version":"2026","doc_id":"leave_2026","effective_date":"2026-01-01" } ] }
```

---

## 3. Ask (the core Q&A)

### POST `/api/ask`
Full pipeline: query-understanding → routing → retrieval → rerank → grounding → cited answer.
Handles intents automatically (factual / summarize / compare / chitchat) and version resolution.

**Request** (`AskRequest`):
| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string (3–2000) | — | **Required.** The user question. |
| `strategy` | string | server default | `pageindex` \| `hybrid` \| `bm25_only` \| `vector_only`. |
| `model_provider` | string | server default | `openai` \| `azure_openai` \| `openrouter` \| `groq` \| `gemini` \| `ollama`. |
| `tenant_id` | string | default tenant | Ignored for authenticated callers (pinned to their tenant). |
| `doc_id` | string | router picks | Force a specific document; else the router chooses. |
| `family` | string | — | Versioned family → resolves to latest (or `as_of`) version. |
| `as_of` | string (ISO date) | — | Answer from the version effective on/before this date. |

**Response** (`AskResponse`) — key fields for the UI:
```json
{
  "query": "How many days of maternity leave do I get?",
  "answer": "You are entitled to 26 weeks… [page 4]",
  "citations": [ { "page_numbers": [4], "section_title": "Maternity", "chunk_id": "c_12", "node_id": "", "relevance": "states the entitlement" } ],
  "pipeline_steps": [ { "node_name": "hybrid_retrieval", "status": "completed", "thinking": "…", "result": "…", "duration_ms": 31.2, "tokens_used": 0 } ],
  "confidence": 0.82,
  "grounded": true,
  "refused": false,
  "blocked": false,
  "out_of_scope": false,
  "unsupported_claims": [],
  "retrieval_attempts": 1,
  "rewritten_query": null,
  "understood_query": "maternity leave entitlement days",
  "query_intent": "factual",
  "strategy_used": "hybrid",
  "model_used": "gpt-4o-mini",
  "total_tokens": 1240,
  "total_duration_ms": 1830.5,
  "retrieved_page_numbers": [4, 5],
  "tenant_id": "t_ab12…",
  "doc_id": "maternity_policy",
  "indexed_at": "2026-06-01T10:00:00Z",
  "thread_id": "…",
  "interrupted": false,
  "clarification": null
}
```

**UI rendering guide:**
- `answer` + `citations` → main answer card with clickable page chips.
- `pipeline_steps` → an "agent trace" timeline (status, thinking, duration).
- `confidence` / `grounded` / `refused` → trust badge; if `refused`, show the honest decline.
- `blocked` → prompt-injection blocked banner. `out_of_scope` → "outside this document" notice.
- `query_intent` / `understood_query` / `rewritten_query` → "how we interpreted your question."
- `interrupted: true` + `clarification` → render a follow-up question, then call **`/api/ask/resume`**.

### POST `/api/ask/stream`  *(Server-Sent Events)*
Same inputs as `/api/ask`; streams the pipeline live. `Content-Type: text/event-stream`.

Each event: `event: <type>\ndata: <json>\n\n`. Event types (`SSEEvent.event`):
`node_started` · `node_thinking` · `node_completed` · `answer_chunk` · `pipeline_complete` · `error`.

**UI:** consume with `EventSource`/`fetch`-stream; append `answer_chunk` tokens, update the
trace on `node_*`, finalize on `pipeline_complete`.

### POST `/api/ask/resume`  *(Human-in-the-loop)*
Resume an interrupted run after the user answers a clarification.

**Request** (`ResumeRequest`):
```json
{ "thread_id": "…", "answer": "I meant the 2026 policy", "tenant_id": null, "doc_id": null, "strategy": null, "model_provider": null }
```
**Response:** `AskResponse`. *(HITL only triggers when `HITL_ENABLED=true`.)*

---

## 4. Summarization (map-reduce over whole documents)

### POST `/api/summarize`  *(synchronous)*
**Request** (`SummarizeRequest`): `{ "doc_id": "maternity_policy", "style": "concise" }` (both optional; `doc_id` defaults to the tenant's doc, `style` defaults to `concise`).
**Response** (`SummaryResponse`): `{ "doc_id":"…", "summary":"…", "chunks": 142, "reduce_levels": 2 }`

### POST `/api/summarize/async`  →  GET `/api/jobs/{job_id}`
For large PDFs, run summarization as a background job and poll.

**`/api/summarize/async`** → `JobResponse`: `{ "job_id":"…", "kind":"summarize", "status":"pending" }`
**`GET /api/jobs/{job_id}`** → `JobResponse`:
```json
{ "job_id":"…", "kind":"summarize", "status":"done", "result": { "summary":"…", "chunks":142, "reduce_levels":2 }, "error":"" }
```
`status` ∈ `pending | running | done | failed`. **UI:** poll every ~1–2 s until terminal.

---

## 5. Inspection & system

### GET `/api/tree?doc_id={id}`
PageIndex tree for the document (requires a PageIndex build). Returns `TreeResponse`
(`doc_name`, `total_pages`, nested `tree` of `TreeNode`). **UI:** tree explorer.

### GET `/api/page/{page_num}?doc_id={id}`
Raw page text. Returns `PageResponse`: `{ "page_number": 4, "text": "…", "token_count": 380 }`.

### GET `/api/health`
```json
{ "status":"ok", "indexed": true, "doc_name":"…", "total_pages":64, "documents":3,
  "active_model":"ollama", "active_retrieval":"hybrid", "active_embedding":"huggingface", "hybrid_index_loaded": true }
```

### GET `/api/config`  ·  POST `/api/config`  *(runtime switches)*
**GET** returns the live switches + the options the UI should offer:
```json
{ "active_model":"ollama", "active_retrieval":"hybrid", "active_embedding":"huggingface",
  "providers": ["ollama","openai"], "strategies": ["pageindex","hybrid","bm25_only","vector_only"] }
```
**POST** mutates the three active switches at runtime (lock-guarded). Send any subset:
```json
{ "active_model":"openai", "active_retrieval":"hybrid", "active_embedding":"openai" }
```
> Only these three are runtime-mutable. `providers` lists which model backends actually have
> credentials configured — drive the model dropdown from it. All other settings are env/restart-scoped
> (see §6) and should be surfaced as **read-only** in the UI unless you add admin endpoints.

---

## 6. Environment configuration (`.env`)

Legend — **Scope:** `runtime` = changeable via `POST /api/config` without restart · `restart` = edit `.env` and restart.
**UI control** = suggested input widget for a settings panel.

### 6.1 Active switches *(runtime-mutable)*
| Env var | Type | Default | Scope | UI control |
|---|---|---|---|---|
| `ACTIVE_MODEL` | enum | `ollama` | runtime | dropdown: openai, azure_openai, openrouter, groq, gemini, ollama |
| `ACTIVE_RETRIEVAL` | enum | `hybrid` | runtime | dropdown: pageindex, hybrid, bm25_only, vector_only |
| `ACTIVE_EMBEDDING` | enum | `huggingface` | runtime | dropdown: huggingface, openai, ollama |

### 6.2 LLM providers & credentials *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `OPENAI_API_KEY` | secret | `""` | password field |
| `OPENAI_MODEL` | string | `gpt-4o-mini` | text |
| `AZURE_OPENAI_API_KEY` | secret | `""` | password |
| `AZURE_OPENAI_ENDPOINT` | url | `""` | text |
| `AZURE_OPENAI_API_VERSION` | string | `2024-12-01-preview` | text |
| `AZURE_OPENAI_DEPLOYMENT` | string | `gpt-4o-mini` | text |
| `OPENROUTER_API_KEY` | secret | `""` | password |
| `OPENROUTER_BASE_URL` | url | `https://openrouter.ai/api/v1` | text |
| `OPENROUTER_MODEL` | string | `openai/gpt-4o-mini` | text |
| `GROQ_API_KEY` | secret | `""` | password |
| `GROQ_BASE_URL` | url | `https://api.groq.com/openai/v1` | text |
| `GROQ_MODEL` | string | `llama-3.3-70b-versatile` | text |
| `GEMINI_API_KEY` | secret | `""` | password |
| `GEMINI_MODEL` | string | `gemini-2.0-flash` | text |
| `OLLAMA_BASE_URL` | url | `http://localhost:11434` | text |
| `OLLAMA_MODEL` | string | `llama3.1` | text |
| `TEMPERATURE` | float 0–2 | `0` | slider |

### 6.3 Embeddings *(restart — changing dim requires re-index)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `HF_EMBEDDING_MODEL` | string | `all-MiniLM-L6-v2` | text |
| `HF_EMBEDDING_DIM` | int | `384` | number (read-only after index) |
| `OLLAMA_EMBEDDING_DIM` | int | `768` | number (read-only after index) |
| `EMBEDDING_CACHE_ENABLED` | bool | `true` | toggle |
| `EMBEDDING_CACHE_SIZE` | int | `2048` | number |

### 6.4 Qdrant vector store *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `QDRANT_HOST` | string | `localhost` | text |
| `QDRANT_PORT` | int | `6333` | number |
| `QDRANT_COLLECTION` | string | `vision` | text |
| `QDRANT_IN_MEMORY` | bool | `false` | toggle |
| `QDRANT_USE_SERVER` | bool | `false` | toggle |
| `QDRANT_PATH` | path | `data/qdrant` | text |

### 6.5 Chunking *(restart — affects new indexes only)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `CHUNK_SIZE` | int (tokens) | `512` | number |
| `CHUNK_OVERLAP` | int (tokens) | `100` | number |
| *(internal)* `semantic_merge_threshold` | float | `0.8` | slider |

### 6.6 Hybrid retrieval & reranking *(restart)*
| Env var | Type | Default | UI control | Notes |
|---|---|---|---|---|
| `HYBRID_TOP_K` | int | `10` | number | final results after fusion |
| `BM25_TOP_K` | int | `20` | number | sparse candidates pre-fusion |
| `VECTOR_TOP_K` | int | `20` | number | dense candidates pre-fusion |
| `RRF_K` | int | `60` | number | RRF constant (standard 60) |
| `RERANK_ENABLED` | bool | `true` | toggle | cross-encoder rerank |
| `RERANK_MODEL` | string | `cross-encoder/ms-marco-MiniLM-L-6-v2` | text |  |
| `RERANK_CANDIDATES` | int | `30` | number | pool retrieved **and** scored |
| `RERANK_TOP_K` | int | `6` | number | kept after rerank (final context size) |
| `RERANK_MAX_CHARS` | int | `2400` | number | per-candidate text scored |

### 6.7 MMR diversity *(restart, opt-in)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `MMR_ENABLED` | bool | `false` | toggle |
| `MMR_LAMBDA` | float 0–1 | `0.6` | slider (1=relevance, 0=diversity) |

### 6.8 Contextual retrieval *(restart, opt-in — needs re-index)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `CONTEXTUAL_RETRIEVAL_ENABLED` | bool | `false` | toggle |
| `CONTEXTUAL_DOC_SAMPLE_CHARS` | int | `4000` | number |
| `CONTEXTUAL_MAX_CONCURRENCY` | int | `8` | number |

### 6.9 PageIndex (tree retrieval) *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `TOC_CHECK_PAGES` | int | `20` | number |
| `MAX_PAGES_PER_NODE` | int | `10` | number |
| `MAX_TOKENS_PER_NODE` | int | `20000` | number |

### 6.10 Query understanding & routing *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `QUERY_UNDERSTANDING_ENABLED` | bool | `true` | toggle |
| `QUERY_REWRITE_ENABLED` | bool | `true` | toggle |
| `ROUTER_ENABLED` | bool | `true` | toggle |
| `ROUTER_MIN_SCORE` | float 0–1 | `0.15` | slider |
| `MULTI_DOC_TOP_N` | int | `3` | number |

### 6.11 Grounding / verification & answer budget *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `GROUNDING_MAX_CONTEXT_CHARS` | int | `12000` | number |
| `GROUNDING_ACCEPT_THRESHOLD` | float 0–1 | `0.6` | slider |
| `GROUNDING_REFUSE_THRESHOLD` | float 0–1 | `0.5` | slider |
| `ANSWER_MAX_CONTEXT_CHARS` | int | `12000` | number |
| *(internal)* `max_retrieval_attempts` | int | `2` | number |
| *(internal)* `guardrail_threshold` | int | `40` | number |

### 6.12 Summarization (map-reduce) *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `SUMMARIZE_MAX_MAP_CALLS` | int | `200` | number |
| `SUMMARIZE_MAX_CONCURRENCY` | int | `6` | number |
| *(internal)* `summarize_map_chars` | int | `4000` | number |
| *(internal)* `summarize_reduce_char_budget` | int | `8000` | number |
| *(internal)* `summarize_reduce_group_size` | int | `10` | number |

### 6.13 Safety guardrails *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `INJECTION_GUARD_ENABLED` | bool | `true` | toggle |
| `PII_REDACTION_ENABLED` | bool | `true` | toggle |

### 6.14 Agentic depth (HITL / corrective-RAG) *(restart, opt-in)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `HITL_ENABLED` | bool | `false` | toggle |
| `CORRECTIVE_RAG_ENABLED` | bool | `false` | toggle |
| `CORRECTIVE_GRADE_THRESHOLD` | float 0–1 | `0.6` | slider |

### 6.15 Multi-tenancy, storage & auth *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `DEFAULT_TENANT` | string | `default` | text |
| `STORAGE_BACKEND` | enum | `local` | dropdown: local (`s3` reserved) |
| `AUTH_SECRET` | secret | `dev-secret-change-me` | password ⚠️ change in prod |
| `AUTH_TOKEN_TTL_SECONDS` | int | `86400` | number |
| `USERS_DB_PATH` | path | `data/users.json` | text |

### 6.16 Durable execution (checkpointer) *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `CHECKPOINTER` | enum | `sqlite` | dropdown: sqlite, memory |
| `CHECKPOINT_DB_PATH` | path | `data/checkpoints.sqlite` | text |

### 6.17 Observability (OpenTelemetry) *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `OTEL_ENABLED` | bool | `false` | toggle |
| `OTEL_SERVICE_NAME` | string | `vision` | text |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | url | `""` | text |
| `OTEL_CONSOLE_EXPORT` | bool | `false` | toggle |
| `OTEL_CAPTURE_CONTENT` | bool | `false` | toggle ⚠️ PII-sensitive |

### 6.18 LLM resilience (timeouts + retry) *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `LLM_TIMEOUT_SECONDS` | float | `60` | number |
| `LLM_MAX_RETRIES` | int | `2` | number |
| `LLM_RETRY_BASE_DELAY` | float | `0.5` | number |
| `LLM_RETRY_MAX_DELAY` | float | `20` | number |
| `LLM_RETRY_JITTER` | bool | `true` | toggle |
| `LLM_RETRY_RESPECT_RETRY_AFTER` | bool | `true` | toggle |

### 6.19 Evaluation gating *(restart / CI)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `EVAL_JUDGE_PROVIDER` | string | `""` (use active) | text |
| `EVAL_JUDGE_MODEL` | string | `""` | text |
| `EVAL_FAITHFULNESS_THRESHOLD` | float 0–1 | `0.75` | slider |
| `EVAL_ANSWER_RELEVANCY_THRESHOLD` | float 0–1 | `0.80` | slider |
| `EVAL_CONTEXT_PRECISION_THRESHOLD` | float 0–1 | `0.70` | slider |
| `EVAL_CONTEXT_RECALL_THRESHOLD` | float 0–1 | `0.80` | slider |

### 6.20 Domain profile (document-agnostic prompts) *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `AUTO_GENERATE_PROFILE` | bool | `true` | toggle |
| `DOMAIN_DESCRIPTION_OVERRIDE` | string | `""` | textarea |
| `DOMAIN_PERSONA_OVERRIDE` | string | `""` | textarea |

### 6.21 Server & paths *(restart)*
| Env var | Type | Default | UI control |
|---|---|---|---|
| `HOST` | string | `0.0.0.0` | text |
| `PORT` | int | `8001` | number |
| `CORS_ORIGINS` | json array | `["http://localhost:5173","http://localhost:3000","http://localhost:8501"]` | tag input |
| `PDF_DIR` | path | `data/pdf` | text |
| `INDEX_DIR` | path | `data/index` | text |
| `BM25_DIR` | path | `data/bm25` | text |
| `CHUNKS_DIR` | path | `data/chunks` | text |

---

## 7. Frontend build checklist

1. **Auth screen** → `signup`/`login`, store `access_token`, send it on every call.
2. **Document manager** → `upload`, `documents`, `versions`; trigger `index*` if needed.
3. **Chat** → `ask` (or `ask/stream` for live tokens); render answer + citations + trust badges;
   handle `interrupted`/`clarification` → `ask/resume`.
4. **Summarize** → `summarize/async` + poll `jobs/{id}` (with a progress UI).
5. **Inspectors** → `tree`, `page/{n}`, `health`.
6. **Settings panel** → `GET /api/config` to populate dropdowns; `POST /api/config` for the
   three runtime switches; render §6 env tables as read-only system info (or admin-only).
7. **Error handling** → standard FastAPI shape `{ "detail": "..." }`; common codes:
   `400` bad input · `401` auth · `404` not found · `503` no index yet · `500` server.
