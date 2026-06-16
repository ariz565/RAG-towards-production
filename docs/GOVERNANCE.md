# Vision — AI Governance

How Vision satisfies 2026 AI-governance expectations for LLM/RAG systems:
controls that prove model behavior is **evaluated before release, enforced at
runtime, audited after deployment, and traceable to its sources**.

Governance is usually framed as four pillars — **policy & risk tiering, data
lineage, runtime enforcement, and auditability**. This document maps Vision's
features to each, and points to the code.

---

## 1. Policy & risk tiering

A single switch, `GOVERNANCE_RISK_TIER`, sets how strict enforcement is. The
effective policy is resolved in [governance.py](app/services/governance.py)
(`active_policy()`) and applied at runtime.

| Tier | Grounding accept | Refuse-when-unsure | Citations |
|---|---|---|---|
| `standard` (default) | configured (`0.60`) | configured (`0.50`) | encouraged |
| `high` (legal/medical/financial) | `≥ 0.70` | `≥ 0.60` | required |

`high` tier raises the grounding bar in the pipeline's grounding gate
([pipeline.py](app/services/pipeline.py) → `grounding_check_node`), so unverifiable
answers are declined rather than served. `standard` leaves behavior unchanged.

**Inspect:** `GET /api/governance/info` → `policy`.

## 2. Data lineage & source certification

Every uploaded document gets a provenance record
([lineage.py](app/services/lineage.py)), persisted at
`data/index/<tenant>__<doc>_lineage.json`:

- **content fingerprint** — SHA-256 of the original PDF bytes (tamper-evidence);
- **uploader** (hashed) + upload timestamp;
- **vector lineage** — the embedding model + dimension used to index it;
- which indexes exist (PageIndex tree / hybrid) + chunk count;
- a **`certified`** flag a reviewer sets once the source is vetted.

This answers the core audit question — *"where did this answer come from, and is
the source trusted?"* — combined with the per-answer `doc_id`, `indexed_at`, and
page/chunk citations already returned by `/api/ask`.

**Inspect:** `GET /api/governance/lineage/{doc_id}` · **Certify:** `POST /api/governance/lineage/{doc_id}/certify`.

## 3. Runtime enforcement (guardrails)

Checks on inputs and outputs during the request, all in the pipeline:

- **Input guardrail** — prompt-injection / jailbreak detection blocks the request
  ([guardrails.py](app/services/guardrails.py), `safety_check_node`).
- **Scope guardrail** — off-topic questions are redirected, not answered.
- **PII redaction** — sensitive output is redacted (`PII_REDACTION_ENABLED`).
- **Grounding / hallucination control** — answers are decomposed into claims and
  verified against the retrieved sources; below the (tier-aware) refuse threshold
  the system **declines honestly** instead of guessing.
- **Corrective-RAG / HITL** — optional context-grading re-retrieval and
  human-in-the-loop clarification for ambiguous questions.

Prompts for these are centralized and reviewable in [prompts.py](app/prompts.py).

**Inspect:** `GET /api/governance/info` → `controls`.

## 4. Auditability

Every answered query is recorded to an **append-only JSONL audit log**
([audit.py](app/services/audit.py)) at `data/audit/audit-YYYY-MM-DD.jsonl`:

- timestamp, tenant, **hashed** caller, intent, `doc_id`, strategy, model;
- **outcome** — `served` | `refused` | `blocked` | `out_of_scope`;
- grounding confidence, source ids + pages used, unsupported-claim count;
- tokens, latency, and the active risk tier + app version.

PII-safe by design: the caller is stored as a SHA-256 prefix, and query text is
recorded only when `AUDIT_LOG_QUERY_TEXT=true` (otherwise a hash). Writes are
best-effort and never break a request.

**Inspect:** `GET /api/governance/audit?limit=50` (tenant-scoped, newest first).

## 5. Model & version registry

`GET /api/governance/info` → `models` reports exactly what is serving answers:
LLM provider/model + temperature, embedding model + dimension, reranker model,
active retrieval strategy, and app version — also embedded in every audit record
for point-in-time traceability.

## 6. Access control & isolation

- **Auth** — signed bearer tokens (PBKDF2 + HMAC), TTL-bounded
  ([auth.py](app/services/auth.py)).
- **Tenant isolation** — authenticated callers are pinned to their own tenant and
  cannot read/write another tenant's documents; artifacts are namespaced
  `<tenant>__<doc>` (a slice of ACL-aware retrieval).

## 7. Evaluation gating & observability

- **Eval gates** — RAGAS/DeepEval thresholds (faithfulness, answer-relevancy,
  context precision/recall) in `evals/` gate quality before release.
- **Tracing** — OpenTelemetry GenAI spans (`OTEL_ENABLED`) export to
  Langfuse/Datadog/etc. via OTLP for post-deployment monitoring.

---

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `GOVERNANCE_RISK_TIER` | `standard` | `standard` \| `high` enforcement strictness |
| `AUDIT_ENABLED` | `true` | append-only audit log |
| `AUDIT_DIR` | `data/audit` | audit log location |
| `AUDIT_LOG_QUERY_TEXT` | `true` | `false` → store only a query hash (max privacy) |
| `AUDIT_RETENTION_DAYS` | `90` | advisory retention window |
| `LINEAGE_ENABLED` | `true` | per-document source certification |

## Known gaps / roadmap

Honest scope notes (these are deliberate next steps, not hidden):

- **Gateway-level enforcement** — guardrails run in-app today; the enterprise
  direction is a shared LLM gateway so policy is enforced once across providers.
- **Streaming audit** — `/api/ask` is fully audited; `/api/ask/stream` audits are
  not yet emitted at stream end.
- **Retention automation** — `AUDIT_RETENTION_DAYS` is advisory; no purge job yet.
- **Immutable/exportable audit sink** — JSONL is local; production would ship to
  an append-only store (e.g. object storage with object-lock, or a SIEM).
