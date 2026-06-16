"""AI governance — model/version registry and risk-tier policy.

Centralizes the "what is running and under what rules" view that 2026 governance
frameworks expect: a **model registry** (which LLM / embedding / reranker versions
are serving answers) and an **enforcement policy** derived from the configured
**risk tier**. The audit log and the /api/governance endpoints read from here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.config import EmbeddingProvider, settings

APP_VERSION = "2.0.0"


def _embedding_model_name() -> str:
    if settings.active_embedding == EmbeddingProvider.HUGGINGFACE:
        return settings.hf_embedding_model
    if settings.active_embedding == EmbeddingProvider.OLLAMA:
        return settings.ollama_embedding_model
    return settings.openai_embedding_model


def model_registry() -> dict:
    """The models/versions currently serving answers (for audit + transparency)."""
    return {
        "app_version": APP_VERSION,
        "llm": {
            "provider": settings.active_model.value,
            "model": settings.active_model_name,
            "temperature": settings.temperature,
        },
        "embedding": {
            "provider": settings.active_embedding.value,
            "model": _embedding_model_name(),
            "dim": settings.embedding_dim,
        },
        "reranker": {
            "enabled": settings.rerank_enabled,
            "model": settings.rerank_model if settings.rerank_enabled else None,
        },
        "retrieval_strategy": settings.active_retrieval.value,
    }


@dataclass
class Policy:
    """Effective enforcement policy for the active risk tier."""

    risk_tier: str
    grounding_accept_threshold: float
    grounding_refuse_threshold: float
    require_citations: bool
    injection_guard: bool
    pii_redaction: bool

    def to_dict(self) -> dict:
        return asdict(self)


def active_policy() -> Policy:
    """Resolve the runtime policy from the configured risk tier.

    ``standard`` uses the configured grounding thresholds unchanged. ``high``
    raises the bar (stricter acceptance + refuse-when-unsure) and expects
    citations — appropriate for legal/medical/financial documents.
    """
    tier = (settings.governance_risk_tier or "standard").lower()
    accept = settings.grounding_accept_threshold
    refuse = settings.grounding_refuse_threshold
    require_citations = False

    if tier == "high":
        accept = max(accept, 0.70)
        refuse = max(refuse, 0.60)
        require_citations = True

    return Policy(
        risk_tier=tier,
        grounding_accept_threshold=accept,
        grounding_refuse_threshold=refuse,
        require_citations=require_citations,
        injection_guard=settings.injection_guard_enabled,
        pii_redaction=settings.pii_redaction_enabled,
    )


def controls_summary() -> dict:
    """Which governance controls are active (for /api/governance/info)."""
    return {
        "input_guardrail": settings.injection_guard_enabled,
        "pii_redaction": settings.pii_redaction_enabled,
        "grounding_verification": True,
        "corrective_rag": settings.corrective_rag_enabled,
        "human_in_the_loop": settings.hitl_enabled,
        "audit_log": settings.audit_enabled,
        "document_lineage": settings.lineage_enabled,
        "tenant_isolation": True,
        "observability_tracing": settings.otel_enabled,
        "eval_gating": True,
    }
