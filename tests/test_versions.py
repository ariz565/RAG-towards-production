"""Versioning — latest vs as-of resolution. (Requires runtime deps.)"""

from app.config import settings


def test_resolve_latest_and_as_of(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", str(tmp_path))
    from app.services.versions import VersionStore

    vs = VersionStore()
    vs.register("tenantA", "leave_policy", "2024", "leave_2024", "2024-01-01")
    vs.register("tenantA", "leave_policy", "2026", "leave_2026", "2026-01-01")

    assert vs.resolve("tenantA", "leave_policy") == "leave_2026"            # latest
    assert vs.resolve("tenantA", "leave_policy", as_of="2025-06-01") == "leave_2024"
    assert vs.resolve("tenantA", "missing") is None
    assert vs.version_of("tenantA", "leave_2024") == "2024"


def test_tenant_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", str(tmp_path))
    from app.services.versions import VersionStore

    vs = VersionStore()
    vs.register("t1", "policy", "1", "t1_policy_1", "2024-01-01")
    assert vs.resolve("t2", "policy") is None   # other tenant can't see it
