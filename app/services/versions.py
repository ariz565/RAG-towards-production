"""Document versioning — version chains for evolving docs (legal/policy).

A legal law or HR policy changes over time; you must answer from the **current**
version by default, or **as of** a date. Each version is still indexed as its own
document (its own `doc_id`), and a per-tenant manifest groups versions under a
**family** (the logical document name):

    family "leave_policy" → [
        {version: "2024", doc_id: "leave_policy_2024", effective_date: "2024-01-01"},
        {version: "2026", doc_id: "leave_policy_2026", effective_date: "2026-01-01"},
    ]

This service supports both JSON and Postgres backends, with a fallback to JSON
storage when the database is unavailable.
"""

from __future__ import annotations

import datetime
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.db import create_postgres_pool
from app.services.storage import tenant_slug


@dataclass
class Version:
    family: str
    version: str
    doc_id: str
    effective_date: str = ""   # ISO date (YYYY-MM-DD); blank → use registration time
    indexed_at: str = ""


class VersionStoreBase(ABC):
    async def connect(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    @abstractmethod
    async def register(self, tenant_id: str, family: str, version: str,
                       doc_id: str, effective_date: str = "") -> Version:
        pass

    @abstractmethod
    async def list(self, tenant_id: str, family: str | None = None) -> dict | list:
        pass

    @abstractmethod
    async def resolve(self, tenant_id: str, family: str,
                      as_of: str | None = None) -> str | None:
        pass

    @abstractmethod
    async def version_of(self, tenant_id: str, doc_id: str) -> str | None:
        pass


class JSONVersionStore(VersionStoreBase):
    def _path(self, tenant_id: str) -> Path:
        return settings.index_path / f"{tenant_slug(tenant_id)}__versions.json"

    def _load(self, tenant_id: str) -> dict[str, list[dict]]:
        p = self._path(tenant_id)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _save(self, tenant_id: str, db: dict) -> None:
        p = self._path(tenant_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(db, indent=2), encoding="utf-8")

    async def register(self, tenant_id: str, family: str, version: str,
                       doc_id: str, effective_date: str = "") -> Version:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec = Version(family, version, doc_id, effective_date or now[:10], now)
        db = self._load(tenant_id)
        fam = [v for v in db.get(family, []) if v["version"] != version]
        fam.append(asdict(rec))
        db[family] = fam
        self._save(tenant_id, db)
        return rec

    async def list(self, tenant_id: str, family: str | None = None) -> dict | list:
        db = self._load(tenant_id)
        return db.get(family, []) if family else db

    async def resolve(self, tenant_id: str, family: str,
                      as_of: str | None = None) -> str | None:
        versions = self._load(tenant_id).get(family, [])
        if as_of:
            versions = [v for v in versions if (v.get("effective_date") or "") <= as_of]
        if not versions:
            return None
        best = max(versions, key=lambda v: (v.get("effective_date") or "", v.get("version", "")))
        return best["doc_id"]

    async def version_of(self, tenant_id: str, doc_id: str) -> str | None:
        for fam in self._load(tenant_id).values():
            for v in fam:
                if v["doc_id"] == doc_id:
                    return v["version"]
        return None


class PostgresVersionStore(VersionStoreBase):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
        self._fallback = JSONVersionStore()

    async def connect(self) -> bool:
        if self.pool:
            return True
        try:
            self.pool = await create_postgres_pool(self.db_url)
            await self._init_schema()
            return True
        except Exception as e:
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
                CREATE TABLE IF NOT EXISTS versions (
                    tenant_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    version TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    effective_date TEXT,
                    indexed_at TEXT,
                    PRIMARY KEY (tenant_id, family, version)
                )
                """
            )

    async def _ensure_connected(self) -> bool:
        if self.pool:
            return True
        return await self.connect()

    async def register(self, tenant_id: str, family: str, version: str,
                       doc_id: str, effective_date: str = "") -> Version:
        if not await self._ensure_connected():
            return await self._fallback.register(tenant_id, family, version, doc_id, effective_date)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec = Version(family, version, doc_id, effective_date or now[:10], now)
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO versions (tenant_id, family, version, doc_id, effective_date, indexed_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id, family, version)
                    DO UPDATE SET doc_id = EXCLUDED.doc_id,
                                  effective_date = EXCLUDED.effective_date,
                                  indexed_at = EXCLUDED.indexed_at
                    """,
                    tenant_id, family, version, doc_id, rec.effective_date, rec.indexed_at,
                )
            return rec
        except Exception:
            return await self._fallback.register(tenant_id, family, version, doc_id, effective_date)

    async def list(self, tenant_id: str, family: str | None = None) -> dict | list:
        if not await self._ensure_connected():
            return await self._fallback.list(tenant_id, family)
        try:
            if family:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT family, version, doc_id, effective_date, indexed_at"
                        " FROM versions WHERE tenant_id = $1 AND family = $2",
                        tenant_id, family,
                    )
                return [dict(row) for row in rows]
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT family, version, doc_id, effective_date, indexed_at"
                    " FROM versions WHERE tenant_id = $1",
                    tenant_id,
                )
            out: dict[str, list[dict]] = {}
            for row in rows:
                out.setdefault(row["family"], []).append(dict(row))
            return out
        except Exception:
            return await self._fallback.list(tenant_id, family)

    async def resolve(self, tenant_id: str, family: str,
                      as_of: str | None = None) -> str | None:
        if not await self._ensure_connected():
            return await self._fallback.resolve(tenant_id, family, as_of)
        try:
            query = (
                "SELECT doc_id FROM versions WHERE tenant_id = $1 AND family = $2"
                "{} ORDER BY effective_date DESC, version DESC LIMIT 1"
            )
            params: list[Any] = [tenant_id, family]
            if as_of:
                query = query.format(" AND effective_date <= $3")
                params.append(as_of)
            else:
                query = query.format("")
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, *params)
            return row["doc_id"] if row else None
        except Exception:
            return await self._fallback.resolve(tenant_id, family, as_of)

    async def version_of(self, tenant_id: str, doc_id: str) -> str | None:
        if not await self._ensure_connected():
            return await self._fallback.version_of(tenant_id, doc_id)
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT version FROM versions WHERE tenant_id = $1 AND doc_id = $2 LIMIT 1",
                    tenant_id, doc_id,
                )
            return row["version"] if row else None
        except Exception:
            return await self._fallback.version_of(tenant_id, doc_id)


def get_version_store() -> VersionStoreBase:
    if settings.versions_backend == "postgres" and settings.database_url:
        return PostgresVersionStore(settings.database_url)
    return JSONVersionStore()


version_store = get_version_store()
