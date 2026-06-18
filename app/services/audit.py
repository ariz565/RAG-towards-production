"""Append-only audit log — a governance record of every answered query.

Answers the core audit question 2026 governance asks — *"who asked what, what was
retrieved, and what did we serve or refuse?"* — without leaking PII:

- the caller is recorded as a **salted-ish SHA-256 prefix** of their email, never raw;
- the query text is recorded only if ``AUDIT_LOG_QUERY_TEXT=true`` (else a hash);
- records are newline-delimited JSON (JSONL), one file per UTC day, append-only.

Writes are best-effort and lock-guarded: auditing must never break a request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import settings
from app.services.db import create_postgres_pool
from app.services.governance import APP_VERSION

if TYPE_CHECKING:  # avoid import cycles / heavy imports at runtime
    from app.models.schemas import AskResponse

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class AuditStoreBase(ABC):
    async def connect(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    @abstractmethod
    async def record_async(self, record: dict) -> None:
        pass

    @abstractmethod
    async def read_recent_async(self, tenant_id: str | None = None, limit: int = 50) -> list[dict]:
        pass


class JSONAuditStore(AuditStoreBase):
    def _audit_path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return Path(settings.audit_dir) / f"audit-{day}.jsonl"

    async def record_async(self, record: dict) -> None:
        if not settings.audit_enabled:
            return
        await asyncio.to_thread(self._write_record, record)

    def _write_record(self, record: dict) -> None:
        path = self._audit_path()
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def read_recent_async(self, tenant_id: str | None = None, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._read_recent, tenant_id, limit)

    def _read_recent(self, tenant_id: str | None, limit: int) -> list[dict]:
        out: list[dict] = []
        try:
            files = sorted(Path(settings.audit_dir).glob("audit-*.jsonl"), reverse=True)
            for path in files:
                lines = path.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if tenant_id and rec.get("tenant_id") != tenant_id:
                        continue
                    out.append(rec)
                    if len(out) >= limit:
                        return out
        except Exception as e:
            logger.warning(f"Audit read failed: {e}")
        return out


class PostgresAuditStore(AuditStoreBase):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
        self._fallback = JSONAuditStore()

    async def connect(self) -> bool:
        if self.pool:
            return True
        try:
            self.pool = await create_postgres_pool(self.db_url)
            await self._init_schema()
            return True
        except Exception as e:
            logger.warning(f"Postgres audit connection failed, falling back to JSON: {e}")
            self.pool = None
            return False

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def _init_schema(self) -> None:
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    id SERIAL PRIMARY KEY,
                    tenant_id TEXT,
                    record JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )

    async def record_async(self, record: dict) -> None:
        if not settings.audit_enabled:
            return
        if not await self.connect():
            return await self._fallback.record_async(record)
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO audit_records (tenant_id, record) VALUES ($1, $2)",
                    record.get("tenant_id"), record,
                )
        except Exception as e:
            logger.warning(f"Postgres audit insert failed, falling back to JSON: {e}")
            await self._fallback.record_async(record)

    async def read_recent_async(self, tenant_id: str | None = None, limit: int = 50) -> list[dict]:
        if not await self.connect():
            return await self._fallback.read_recent_async(tenant_id, limit)
        try:
            if tenant_id:
                rows = await self.pool.fetch(
                    "SELECT record FROM audit_records WHERE tenant_id = $1 ORDER BY created_at DESC LIMIT $2",
                    tenant_id, limit,
                )
            else:
                rows = await self.pool.fetch(
                    "SELECT record FROM audit_records ORDER BY created_at DESC LIMIT $1",
                    limit,
                )
            return [dict(row["record"]) for row in rows]
        except Exception as e:
            logger.warning(f"Postgres audit read failed, falling back to JSON: {e}")
            return await self._fallback.read_recent_async(tenant_id, limit)


def get_audit_store() -> AuditStoreBase:
    if settings.audit_backend == "postgres" and settings.database_url:
        return PostgresAuditStore(settings.database_url)
    return JSONAuditStore()


audit_store = get_audit_store()


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _audit_path() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Path(settings.audit_dir) / f"audit-{day}.jsonl"


def build_record(
    *,
    tenant_id: str,
    user_email: str | None,
    query: str,
    response: "AskResponse",
    latency_ms: float,
) -> dict:
    """Assemble one PII-safe audit record from a request + its answer."""
    # Outcome: the single most important governance field.
    if response.blocked:
        outcome = "blocked"
    elif response.refused:
        outcome = "refused"
    elif response.out_of_scope:
        outcome = "out_of_scope"
    else:
        outcome = "served"

    # Citations may be Citation models (pydantic) or plain dicts — handle both.
    def _source_id(c) -> str:
        if isinstance(c, dict):
            return c.get("chunk_id") or c.get("node_id") or ""
        return getattr(c, "chunk_id", "") or getattr(c, "node_id", "") or ""

    sources = sorted({_source_id(c) for c in (response.citations or [])} - {""})

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        "tenant_id": tenant_id,
        "user": _hash(user_email) if user_email else "anonymous",
        "doc_id": response.doc_id,
        "intent": response.query_intent,
        "strategy": response.strategy_used,
        "model": response.model_used,
        "outcome": outcome,
        "grounded": response.grounded,
        "confidence": round(response.confidence, 3),
        "n_sources": len(sources),
        "source_ids": sources[:20],
        "retrieved_pages": response.retrieved_page_numbers,
        "unsupported_claims": len(response.unsupported_claims or []),
        "total_tokens": response.total_tokens,
        "latency_ms": round(latency_ms, 1),
        "risk_tier": settings.governance_risk_tier,
    }
    # Query text is opt-in; otherwise only a stable hash (for dedup/analytics).
    if settings.audit_log_query_text:
        record["query"] = query
    else:
        record["query_hash"] = _hash(query)
    return record


def record_ask(
    *,
    tenant_id: str,
    user_email: str | None,
    query: str,
    response: "AskResponse",
    latency_ms: float,
) -> None:
    """Append one audit record. Best-effort: never raises into the request path."""
    if not settings.audit_enabled:
        return
    record = build_record(
        tenant_id=tenant_id, user_email=user_email, query=query,
        response=response, latency_ms=latency_ms,
    )
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(audit_store.record_async(record))
    except RuntimeError:
        # Not in an async event loop; write synchronously.
        try:
            asyncio.run(audit_store.record_async(record))
        except Exception as e:  # pragma: no cover - auditing must not break answering
            logger.warning(f"Audit write failed: {e}")


async def read_recent_async(tenant_id: str | None = None, limit: int = 50) -> list[dict]:
    """Return the most recent audit records (newest first), optionally tenant-scoped."""
    return await audit_store.read_recent_async(tenant_id, limit)


def read_recent(tenant_id: str | None = None, limit: int = 50) -> list[dict]:
    """Return the most recent audit records (newest first), optionally tenant-scoped."""
    out: list[dict] = []
    try:
        files = sorted(Path(settings.audit_dir).glob("audit-*.jsonl"), reverse=True)
        for path in files:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tenant_id and rec.get("tenant_id") != tenant_id:
                    continue
                out.append(rec)
                if len(out) >= limit:
                    return out
    except Exception as e:  # pragma: no cover
        logger.warning(f"Audit read failed: {e}")
    return out
