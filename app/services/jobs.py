"""Async job store — for long-running work (e.g. summarizing a 500-page PDF).

Big summaries shouldn't block the request: submit → get a `job_id` → poll. This
is a minimal in-memory store (single process). For production, back it with
Redis/DB + a worker (Celery/RQ/Arq) so jobs survive restarts and scale across
workers — the interface here stays the same.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "pending"          # pending | running | done | failed
    result: dict | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id, "kind": self.kind, "status": self.status,
            "result": self.result, "error": self.error,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create(self, kind: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:16], kind=kind)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def submit(self, kind: str, coro_factory: Callable[[], Awaitable[dict]]) -> Job:
        """Create a job and run `coro_factory()` in the background."""
        job = self.create(kind)

        async def _run():
            job.status = "running"
            job.updated_at = time.time()
            try:
                job.result = await coro_factory()
                job.status = "done"
            except Exception as e:  # surface error to the poller
                job.error = f"{type(e).__name__}: {e}"
                job.status = "failed"
            job.updated_at = time.time()

        asyncio.create_task(_run())
        return job


job_store = JobStore()
