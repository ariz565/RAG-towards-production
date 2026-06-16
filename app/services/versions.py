"""Document versioning — version chains for evolving docs (legal/policy).

A legal law or HR policy changes over time; you must answer from the **current**
version by default, or **as of** a date. Each version is still indexed as its own
document (its own `doc_id`), and a per-tenant manifest groups versions under a
**family** (the logical document name):

    family "leave_policy" → [
        {version: "2024", doc_id: "leave_policy_2024", effective_date: "2024-01-01"},
        {version: "2026", doc_id: "leave_policy_2026", effective_date: "2026-01-01"},
    ]

`resolve()` returns the latest version's doc_id (or the latest effective on/before
an `as_of` date). JSON-backed; swap for a DB later without touching callers.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings
from app.services.storage import tenant_slug


@dataclass
class Version:
    family: str
    version: str
    doc_id: str
    effective_date: str = ""   # ISO date (YYYY-MM-DD); blank → use registration time
    indexed_at: str = ""


class VersionStore:
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

    def register(self, tenant_id: str, family: str, version: str, doc_id: str,
                 effective_date: str = "") -> Version:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec = Version(family, version, doc_id, effective_date or now[:10], now)
        db = self._load(tenant_id)
        fam = [v for v in db.get(family, []) if v["version"] != version]  # replace same version
        fam.append(asdict(rec))
        db[family] = fam
        self._save(tenant_id, db)
        return rec

    def list(self, tenant_id: str, family: str | None = None) -> dict | list:
        db = self._load(tenant_id)
        return db.get(family, []) if family else db

    def resolve(self, tenant_id: str, family: str, as_of: str | None = None) -> str | None:
        """doc_id of the latest version (or latest effective on/before `as_of`)."""
        versions = self._load(tenant_id).get(family, [])
        if as_of:
            versions = [v for v in versions if (v.get("effective_date") or "") <= as_of]
        if not versions:
            return None
        best = max(versions, key=lambda v: (v.get("effective_date") or "", v.get("version", "")))
        return best["doc_id"]

    def version_of(self, tenant_id: str, doc_id: str) -> str | None:
        for fam in self._load(tenant_id).values():
            for v in fam:
                if v["doc_id"] == doc_id:
                    return v["version"]
        return None


version_store = VersionStore()
