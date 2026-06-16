"""AI governance — policy tiers, model registry, lineage hashing, audit records.

(Requires runtime deps for import; logic is pure and deterministic.)
"""

from app.config import settings
from app.models.schemas import AskResponse
from app.services import audit, governance, lineage


def test_sha256_is_stable_and_full_length():
    assert lineage.sha256_of(b"abc") == lineage.sha256_of(b"abc")
    assert len(lineage.sha256_of(b"abc")) == 64


def test_policy_standard_is_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "governance_risk_tier", "standard")
    monkeypatch.setattr(settings, "grounding_accept_threshold", 0.6)
    monkeypatch.setattr(settings, "grounding_refuse_threshold", 0.5)
    p = governance.active_policy()
    assert p.risk_tier == "standard"
    assert p.grounding_accept_threshold == 0.6
    assert p.grounding_refuse_threshold == 0.5
    assert p.require_citations is False


def test_policy_high_tier_raises_the_bar(monkeypatch):
    monkeypatch.setattr(settings, "governance_risk_tier", "high")
    monkeypatch.setattr(settings, "grounding_accept_threshold", 0.6)
    monkeypatch.setattr(settings, "grounding_refuse_threshold", 0.5)
    p = governance.active_policy()
    assert p.grounding_accept_threshold >= 0.70
    assert p.grounding_refuse_threshold >= 0.60
    assert p.require_citations is True


def test_model_registry_reports_active_models():
    reg = governance.model_registry()
    assert "llm" in reg and "embedding" in reg and "reranker" in reg
    assert reg["embedding"]["dim"] > 0


def test_audit_record_outcome_sources_and_pii(monkeypatch):
    monkeypatch.setattr(settings, "audit_log_query_text", False)
    resp = AskResponse(
        query="q", answer="a", refused=True, confidence=0.4, doc_id="d1",
        citations=[{"page_numbers": [2], "chunk_id": "c1"}],
        retrieved_page_numbers=[2],
    )
    rec = audit.build_record(
        tenant_id="t1", user_email="me@example.com", query="secret query",
        response=resp, latency_ms=12.3,
    )
    assert rec["outcome"] == "refused"
    assert rec["doc_id"] == "d1"
    assert rec["n_sources"] == 1 and rec["source_ids"] == ["c1"]
    # PII-safe: caller hashed, raw query suppressed when AUDIT_LOG_QUERY_TEXT=false.
    assert rec["user"] != "me@example.com" and len(rec["user"]) == 16
    assert "query" not in rec and "query_hash" in rec


def test_audit_outcome_served_when_clean():
    resp = AskResponse(query="q", answer="a", grounded=True, confidence=0.9)
    rec = audit.build_record(tenant_id="t1", user_email=None, query="q", response=resp, latency_ms=1.0)
    assert rec["outcome"] == "served"
    assert rec["user"] == "anonymous"
