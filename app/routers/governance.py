"""Governance router — transparency + auditability endpoints.

Exposes the "what's running, under what policy, and what happened" surface that
2026 AI governance expects: a model/version registry, the active risk-tier policy
and control inventory, per-document provenance (source certification/lineage), and
a tail of the audit trail.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.deps import Principal, get_principal, get_principal_optional
from app.services import audit, governance, lineage
from app.services.registry import namespace

router = APIRouter(prefix="/api/governance", tags=["governance"])


@router.get("/info")
async def governance_info() -> dict:
    """Model/version registry + active policy + which controls are enabled."""
    return {
        "models": governance.model_registry(),
        "policy": governance.active_policy().to_dict(),
        "controls": governance.controls_summary(),
    }


@router.get("/lineage/{doc_id}")
async def get_lineage(
    doc_id: str, principal: Principal = Depends(get_principal_optional)
) -> dict:
    """Provenance / source-certification record for one of the caller's documents."""
    tenant = principal.tenant_id
    record = lineage.load(namespace(tenant, doc_id))
    if record is None:
        raise HTTPException(404, f"No lineage record for '{doc_id}'.")
    return record


@router.post("/lineage/{doc_id}/certify")
async def certify_lineage(
    doc_id: str, certified: bool = True, principal: Principal = Depends(get_principal)
) -> dict:
    """Mark a document's source as reviewed/certified (or revoke). Auth required."""
    record = lineage.certify(namespace(principal.tenant_id, doc_id), certified)
    if record is None:
        raise HTTPException(404, f"No lineage record for '{doc_id}'.")
    return record


@router.get("/audit")
async def get_audit(
    limit: int = 50, principal: Principal = Depends(get_principal_optional)
) -> dict:
    """Most-recent audit records for the caller's tenant (newest first)."""
    limit = max(1, min(limit, 500))
    records = await audit.audit_store.read_recent_async(principal.tenant_id, limit)
    return {"tenant_id": principal.tenant_id, "records": records}
