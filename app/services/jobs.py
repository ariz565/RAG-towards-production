"""Async job store — durable SQLite persistence for long-running tasks.

This store survives process restarts and keeps the same submit/get API.
If SQLite is unavailable, it falls back to an in-memory store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    kind: str
    tenant_id: str = ""
    status: str = "pending"          # pending | running | done | failed
    result: dict | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or settings.jobs_db_path)
        self._initialized = False
        self._use_db = True
        self._fallback: dict[str, Job] = {}
        self._init_lock = asyncio.Lock()

    async def _init_db(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiosqlite.connect(str(self.db_path)) as db:
                    await db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS jobs (
                            job_id TEXT PRIMARY KEY,
                            kind TEXT NOT NULL,
                            tenant_id TEXT NOT NULL DEFAULT '',
                            status TEXT NOT NULL,
                            result TEXT,
                            error TEXT,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL
                        )
                        """
                    )
                    # Backfill for DBs created before tenant scoping existed.
                    try:
                        await db.execute("ALTER TABLE jobs ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''")
                        await db.commit()
                    except aiosqlite.OperationalError:
                        pass  # column already exists
                    await db.commit()
                self._initialized = True
                logger.info("Job store initialized at %s", self.db_path)
            except Exception as exc:
                self._use_db = False
                self._initialized = True
                logger.warning("Job store unavailable, falling back to memory: %s", exc)

    async def create(self, kind: str, tenant_id: str = "") -> Job:
        await self._init_db()
        job = Job(job_id=uuid.uuid4().hex[:16], kind=kind, tenant_id=tenant_id)
        if self._use_db:
            try:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    await db.execute(
                        """
                        INSERT INTO jobs (job_id, kind, tenant_id, status, result, error, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job.job_id,
                            job.kind,
                            job.tenant_id,
                            job.status,
                            None,
                            "",
                            job.created_at,
                            job.updated_at,
                        ),
                    )
                    await db.commit()
            except Exception as exc:
                self._use_db = False
                self._fallback[job.job_id] = job
                logger.warning("Persisting job failed, using fallback: %s", exc)
        else:
            self._fallback[job.job_id] = job
        return job

    async def get(self, job_id: str) -> Job | None:
        await self._init_db()
        if self._use_db:
            try:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    row = await db.execute_fetchone(
                        "SELECT kind, tenant_id, status, result, error, created_at, updated_at "
                        "FROM jobs WHERE job_id = ?",
                        (job_id,),
                    )
                    if row:
                        kind, tenant_id, status, result, error, created_at, updated_at = row
                        return Job(
                            job_id=job_id,
                            kind=kind,
                            tenant_id=tenant_id or "",
                            status=status,
                            result=json.loads(result) if result else None,
                            error=error or "",
                            created_at=created_at,
                            updated_at=updated_at,
                        )
            except Exception as exc:
                logger.warning("Fetching job failed, using fallback: %s", exc)
                return self._fallback.get(job_id)
        return self._fallback.get(job_id)

    async def update(self, job_id: str, status: str, result: dict | None = None, error: str = "") -> None:
        await self._init_db()
        updated_at = time.time()
        if self._use_db:
            try:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    await db.execute(
                        """
                        UPDATE jobs
                        SET status = ?, result = ?, error = ?, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (
                            status,
                            json.dumps(result) if result is not None else None,
                            error,
                            updated_at,
                            job_id,
                        ),
                    )
                    await db.commit()
                return
            except Exception as exc:
                self._use_db = False
                logger.warning("Updating job failed, using fallback: %s", exc)
        job = self._fallback.get(job_id)
        if job:
            job.status = status
            job.result = result
            job.error = error
            job.updated_at = updated_at

    async def submit(
        self, kind: str, coro_factory: Callable[[], Awaitable[dict]], tenant_id: str = ""
    ) -> Job:
        job = await self.create(kind, tenant_id=tenant_id)

        async def _run() -> None:
            await self.update(job.job_id, "running")
            try:
                result = await coro_factory()
                await self.update(job.job_id, "done", result=result)
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                await self.update(job.job_id, "failed", error=error_msg)
                logger.exception("Job %s failed", job.job_id)

        asyncio.create_task(_run())
        return job


job_store = JobStore()
