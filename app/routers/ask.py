from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import RetrievalStrategy, settings
from app.deps import Principal, get_principal, get_principal_optional
from app.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    JobResponse,
    PageResponse,
    ResumeRequest,
    SummarizeRequest,
    SummaryResponse,
    TreeResponse,
    UploadResponse,
)
from app.services.audit import record_ask
from app.services.doc_router import doc_router
from app.services.governance import active_policy
from app.services.jobs import job_store
from app.services.lineage import record_upload
from app.services.multi_doc import multi_doc_ask
from app.services.pipeline import ask, ask_streaming, resume
from app.services.query_understanding import analyze
from app.services.registry import DocumentBundle, collection_for, namespace, registry
from app.services.storage import get_storage
from app.services.summarize import summarize_document
from app.services.versions import version_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["vision"])

# Serializes runtime config mutations (POST /api/config) across worker threads.
_config_lock = threading.Lock()


# ── Helpers ──────────────────────────────────────────────────────────

def _tenant_for(principal: Principal, requested: str | None) -> str:
    """Authenticated callers are pinned to their own tenant (isolation);
    anonymous/legacy callers may target a tenant or the default."""
    if not principal.anonymous:
        return principal.tenant_id
    return requested or settings.default_tenant


async def _route_doc(tenant_id: str, query: str, requested: str | None) -> str | None:
    """Resolve the target doc: explicit > router pick > None (caller falls back)."""
    if requested:
        return requested
    if not settings.router_enabled:
        return None
    routes = await doc_router.route(tenant_id, query, top_n=1)
    if routes and routes[0].score >= settings.router_min_score:
        return routes[0].doc_id
    return None


def _resolve_or_404(tenant_id: str | None, doc_id: str | None) -> DocumentBundle:
    bundle = registry.resolve(tenant_id, doc_id)
    if bundle is None:
        raise HTTPException(503, "No document indexed for this tenant. Upload/index one first.")
    return bundle


def _check_strategy_ready(bundle: DocumentBundle, strategy: str) -> None:
    if strategy == RetrievalStrategy.PAGEINDEX.value:
        if not bundle.document_index.is_loaded:
            raise HTTPException(503, f"PageIndex not built for '{bundle.doc_id}'.")
    elif not bundle.hybrid_index.is_loaded:
        raise HTTPException(503, f"Hybrid index not built for '{bundle.doc_id}'.")


async def _build_pageindex(tenant_id: str, doc_id: str) -> dict:
    from app.services.indexer import DocumentIndex

    pdf = get_storage().pdf_path(tenant_id, doc_id)
    if not pdf.exists():
        raise HTTPException(404, f"PDF not found for '{doc_id}'. Upload it first.")
    document_index = DocumentIndex()
    return await document_index.build_index(
        pdf_path=pdf,
        index_dir=settings.index_path,
        doc_stem=namespace(tenant_id, doc_id),
        max_pages_per_node=settings.max_pages_per_node,
        max_tokens_per_node=settings.max_tokens_per_node,
        toc_check_pages=settings.toc_check_pages,
    )


async def _build_hybrid(tenant_id: str, doc_id: str) -> dict:
    from app.services.hybrid_retrieval import HybridIndex
    from app.services.ingestion import ingest_pdf

    pdf = get_storage().pdf_path(tenant_id, doc_id)
    if not pdf.exists():
        raise HTTPException(404, f"PDF not found for '{doc_id}'. Upload it first.")
    stem = namespace(tenant_id, doc_id)
    chunks = await ingest_pdf(pdf)
    hybrid_index = HybridIndex(collection_name=collection_for(stem))
    stats = await hybrid_index.build(chunks)
    hybrid_index.save(stem)
    return stats


# ── Upload (auth required) ───────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    family: str | None = Form(default=None),          # versioning (optional)
    version: str | None = Form(default=None),
    effective_date: str | None = Form(default=None),
    principal: Principal = Depends(get_principal),
) -> UploadResponse:
    """Upload a PDF into the caller's tenant storage and index it (hybrid).

    Pass `family` + `version` (+ optional `effective_date`) to register this as a
    version of a document family for version-aware retrieval.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are supported")
    data = await file.read()
    try:
        doc_id = get_storage().save_pdf(principal.tenant_id, file.filename, data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        stats = await _build_hybrid(principal.tenant_id, doc_id)
        await registry.load_bundle(principal.tenant_id, doc_id,
                                   stem=namespace(principal.tenant_id, doc_id))
        # Source certification / provenance for governance + audit.
        record_upload(
            tenant_id=principal.tenant_id, doc_id=doc_id,
            stem=namespace(principal.tenant_id, doc_id),
            source_filename=file.filename or f"{doc_id}.pdf", pdf_bytes=data,
            uploaded_by=principal.email or None,
            chunk_count=stats.get("total_chunks", 0), has_hybrid=True,
        )
        if family and version:
            await version_store.register(principal.tenant_id, family, version, doc_id, effective_date or "")
        msg = f"Uploaded + indexed ({stats.get('total_chunks', 0)} chunks)."
        if family and version:
            msg += f" Registered as {family} v{version}."
        return UploadResponse(success=True, doc_id=doc_id, tenant_id=principal.tenant_id,
                              indexed=True, message=msg)
    except Exception as e:
        logger.exception("Upload indexing failed")
        return UploadResponse(success=True, doc_id=doc_id, tenant_id=principal.tenant_id,
                              indexed=False, message=f"Uploaded, but indexing failed: {e}")


# ── Versions ─────────────────────────────────────────────────────────

@router.get("/versions")
async def list_versions(
    family: str | None = None, principal: Principal = Depends(get_principal_optional)
):
    tenant = _tenant_for(principal, None)
    return {"tenant_id": tenant, "versions": await version_store.list(tenant, family)}


# ── Async summarize (job → poll) ─────────────────────────────────────

@router.post("/summarize/async", response_model=JobResponse)
async def summarize_async(
    request: SummarizeRequest, principal: Principal = Depends(get_principal_optional)
) -> JobResponse:
    tenant = _tenant_for(principal, None)
    doc_id = request.doc_id
    if not doc_id:
        default = registry.resolve(tenant, None)
        doc_id = default.doc_id if default else None
    if not doc_id:
        raise HTTPException(503, "No document to summarize for this tenant.")
    _resolve_or_404(tenant, doc_id)
    style = request.style or "concise"
    job = await job_store.submit(
        "summarize", lambda: summarize_document(tenant, doc_id, style=style)
    )
    return JobResponse(job_id=job.job_id, kind=job.kind, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"job '{job_id}' not found")
    return JobResponse(**job.to_dict())


# ── Summarize (map-reduce over a whole document) ─────────────────────

@router.post("/summarize", response_model=SummaryResponse)
async def summarize_endpoint(
    request: SummarizeRequest, principal: Principal = Depends(get_principal_optional)
) -> SummaryResponse:
    tenant = _tenant_for(principal, None)
    doc_id = request.doc_id
    if not doc_id:
        default = registry.resolve(tenant, None)
        doc_id = default.doc_id if default else None
    if not doc_id:
        raise HTTPException(503, "No document to summarize for this tenant.")
    _resolve_or_404(tenant, doc_id)
    result = await summarize_document(tenant, doc_id, style=request.style or "concise")
    return SummaryResponse(doc_id=doc_id, **result)


# ── Ask (standard) ───────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest, principal: Principal = Depends(get_principal_optional)
) -> AskResponse:
    tenant = _tenant_for(principal, request.tenant_id)
    start = time.time()

    def _audited(resp: AskResponse) -> AskResponse:
        """Write a governance audit record for this answer, then return it."""
        record_ask(
            tenant_id=tenant,
            user_email=None if principal.anonymous else principal.email,
            query=request.query, response=resp, latency_ms=(time.time() - start) * 1000,
        )
        return resp

    analysis = await analyze(request.query)

    # Chit-chat → answer directly, skip retrieval.
    if analysis.intent == "chitchat":
        return _audited(AskResponse(
            query=request.query, tenant_id=tenant, query_intent="chitchat",
            answer="Hi! Ask me a question about your documents and I'll answer with citations.",
        ))

    search_query = analysis.rewritten or request.query

    # Versioned family → resolve to the latest (or as-of) version's doc_id.
    requested_doc = request.doc_id
    if request.family:
        resolved = await version_store.resolve(tenant, request.family, request.as_of)
        if not resolved:
            raise HTTPException(404, f"No version found for family '{request.family}'.")
        requested_doc = resolved

    # Compare / cross-document → fan out across the top-N routed documents + RRF.
    if analysis.intent == "compare" and not requested_doc:
        routes = await doc_router.route(tenant, search_query, top_n=settings.multi_doc_top_n)
        doc_ids = [r.doc_id for r in routes if r.score >= settings.router_min_score]
        if len(doc_ids) >= 2:
            result = await multi_doc_ask(tenant, search_query, doc_ids)
            result["query"] = request.query
            result["query_intent"] = "compare"
            result["understood_query"] = search_query
            return _audited(AskResponse(**result))

    doc_id = await _route_doc(tenant, search_query, requested_doc)

    # Summarize intent → map-reduce the whole document.
    if analysis.intent == "summarize" and doc_id:
        _resolve_or_404(tenant, doc_id)
        summ = await summarize_document(tenant, doc_id, style="concise")
        return _audited(AskResponse(
            query=request.query, tenant_id=tenant, doc_id=doc_id, answer=summ["summary"],
            query_intent="summarize", understood_query=search_query,
        ))

    bundle = _resolve_or_404(tenant, doc_id)
    _check_strategy_ready(bundle, request.strategy or settings.active_retrieval.value)
    result = await ask(
        search_query, strategy=request.strategy, model_provider=request.model_provider,
        tenant_id=tenant, doc_id=doc_id, thread_id=request.thread_id
    )
    result["query"] = request.query                 # echo the user's original
    result["query_intent"] = analysis.intent
    result["understood_query"] = search_query
    return _audited(AskResponse(**result))


# ── Ask (streaming SSE) ──────────────────────────────────────────────

@router.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest, principal: Principal = Depends(get_principal_optional)
) -> StreamingResponse:
    tenant = _tenant_for(principal, request.tenant_id)
    analysis = await analyze(request.query)
    search_query = analysis.rewritten or request.query
    requested_doc = request.doc_id

    if request.family:
        resolved = await version_store.resolve(tenant, request.family, request.as_of)
        if not resolved:
            raise HTTPException(404, f"No version found for family '{request.family}'.")
        requested_doc = resolved

    doc_id = await _route_doc(tenant, search_query, requested_doc)
    bundle = _resolve_or_404(tenant, doc_id)
    _check_strategy_ready(bundle, request.strategy or settings.active_retrieval.value)

    async def event_generator():
        async for event in ask_streaming(
            search_query, strategy=request.strategy, model_provider=request.model_provider,
            tenant_id=tenant, doc_id=doc_id, thread_id=request.thread_id
        ):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── Resume (HITL) ────────────────────────────────────────────────────

@router.post("/ask/resume", response_model=AskResponse)
async def resume_question(
    request: ResumeRequest, principal: Principal = Depends(get_principal_optional)
) -> AskResponse:
    tenant = _tenant_for(principal, request.tenant_id)
    result = await resume(
        request.thread_id, request.answer, tenant_id=tenant, doc_id=request.doc_id,
        strategy=request.strategy, model_provider=request.model_provider,
    )
    return AskResponse(**result)


# ── Index (PageIndex / hybrid / both) ────────────────────────────────

@router.post("/index", response_model=IndexResponse)
async def index_document(
    request: IndexRequest, principal: Principal = Depends(get_principal_optional)
) -> IndexResponse:
    tenant = _tenant_for(principal, None)
    doc_id = Path(request.filename).stem
    try:
        result = await _build_pageindex(tenant, doc_id)
        await registry.load_bundle(tenant, doc_id, stem=namespace(tenant, doc_id))
        return IndexResponse(success=True, doc_name=result["doc_name"],
                             total_pages=result["total_pages"], tree_nodes=result["tree_nodes"],
                             message="Document indexed (PageIndex tree).")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(500, f"Indexing failed: {e}")


@router.post("/index/hybrid", response_model=IndexResponse)
async def index_document_hybrid(
    request: IndexRequest, principal: Principal = Depends(get_principal_optional)
) -> IndexResponse:
    tenant = _tenant_for(principal, None)
    doc_id = Path(request.filename).stem
    try:
        stats = await _build_hybrid(tenant, doc_id)
        await registry.load_bundle(tenant, doc_id, stem=namespace(tenant, doc_id))
        return IndexResponse(success=True, doc_name=doc_id, total_pages=stats.get("total_chunks", 0),
                             tree_nodes=0, message=f"Hybrid index built ({stats.get('total_chunks', 0)} chunks).")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Hybrid indexing failed")
        raise HTTPException(500, f"Hybrid indexing failed: {e}")


@router.post("/index/all", response_model=IndexResponse)
async def index_document_all(
    request: IndexRequest, principal: Principal = Depends(get_principal_optional)
) -> IndexResponse:
    tenant = _tenant_for(principal, None)
    doc_id = Path(request.filename).stem
    try:
        tree_result = await _build_pageindex(tenant, doc_id)
        hybrid_stats = await _build_hybrid(tenant, doc_id)
        await registry.load_bundle(tenant, doc_id, stem=namespace(tenant, doc_id))
        return IndexResponse(
            success=True, doc_name=tree_result.get("doc_name") or doc_id,
            total_pages=tree_result.get("total_pages", 0), tree_nodes=tree_result.get("tree_nodes", 0),
            message=f"All indexes built: {tree_result.get('tree_nodes', 0)} tree nodes + "
                    f"{hybrid_stats.get('total_chunks', 0)} hybrid chunks.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Full indexing failed")
        raise HTTPException(500, f"Indexing failed: {e}")


# ── Documents (tenant-scoped) ────────────────────────────────────────

@router.get("/documents")
async def list_documents(principal: Principal = Depends(get_principal_optional)):
    tenant = _tenant_for(principal, None)
    docs = [d for d in registry.list() if d["tenant_id"] == tenant]
    return {"tenant_id": tenant, "documents": docs}


# ── Config ───────────────────────────────────────────────────────────

def _is_admin(principal: Principal) -> bool:
    return principal.tenant_id == settings.default_tenant


@router.get("/config")
async def get_config(principal: Principal = Depends(get_principal_optional)):
    from app.services.llm import get_available_providers

    policy = active_policy()
    providers = [p for p in get_available_providers() if p["configured"] or p.get("local")]

    return {
        "active_model": settings.active_model.value,
        "active_retrieval": settings.active_retrieval.value,
        "active_embedding": settings.active_embedding.value,
        "providers": providers,
        "strategies": [s.value for s in RetrievalStrategy],
        "governance_tier": policy.risk_tier,
        "tenant_id": None if principal.anonymous else principal.tenant_id,
    }


@router.post("/config")
async def update_config(updates: dict, principal: Principal = Depends(get_principal)):
    from app.config import EmbeddingProvider, ModelProvider

    policy = active_policy()
    if policy.risk_tier != "high" and not _is_admin(principal):
        raise HTTPException(
            403,
            "Config mutations require admin privileges or high governance tier.",
        )

    with _config_lock:  # admin-only; lock-guarded global mutation
        if "active_model" in updates:
            mp = {p.value: p for p in ModelProvider}
            if updates["active_model"] in mp:
                settings.active_model = mp[updates["active_model"]]
        if "active_retrieval" in updates:
            rs = {s.value: s for s in RetrievalStrategy}
            if updates["active_retrieval"] in rs:
                settings.active_retrieval = rs[updates["active_retrieval"]]
        if "active_embedding" in updates:
            ep = {e.value: e for e in EmbeddingProvider}
            if updates["active_embedding"] in ep:
                settings.active_embedding = ep[updates["active_embedding"]]
    return {
        "active_model": settings.active_model.value,
        "active_retrieval": settings.active_retrieval.value,
        "active_embedding": settings.active_embedding.value,
    }


# ── Tree / Page (tenant-scoped) ──────────────────────────────────────

@router.get("/tree", response_model=TreeResponse)
async def get_tree(
    doc_id: str | None = None, principal: Principal = Depends(get_principal_optional)
) -> TreeResponse:
    bundle = _resolve_or_404(_tenant_for(principal, None), doc_id)
    if not bundle.document_index.is_loaded:
        raise HTTPException(503, f"No PageIndex for '{bundle.doc_id}'.")
    return TreeResponse(**bundle.document_index.get_tree_response())


@router.get("/page/{page_num}", response_model=PageResponse)
async def get_page(
    page_num: int, doc_id: str | None = None, principal: Principal = Depends(get_principal_optional)
) -> PageResponse:
    bundle = _resolve_or_404(_tenant_for(principal, None), doc_id)
    page = bundle.document_index.get_page(page_num)
    if not page:
        raise HTTPException(404, f"Page {page_num} not found in '{bundle.doc_id}'.")
    return PageResponse(**page)


# ── Health ───────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    docs = registry.list()
    default = registry.resolve(None, None)
    return HealthResponse(
        status="ok", indexed=bool(docs),
        doc_name=(default.document_index.doc_name or default.doc_id) if default else None,
        total_pages=default.document_index.total_pages if default else 0,
        documents=len(docs),
        active_model=settings.active_model.value,
        active_retrieval=settings.active_retrieval.value,
        active_embedding=settings.active_embedding.value,
        hybrid_index_loaded=bool(default and default.hybrid_index.is_loaded),
    )
