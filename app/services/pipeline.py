"""Agentic RAG Pipeline — LangGraph orchestration for admission guide Q&A.

This is the heart of the system. Supports two retrieval strategies:
1. PageIndex (tree-based, vectorless) — LLM reasons through doc hierarchy
2. Hybrid (BM25 + Vector / Qdrant) — traditional RAG with RRF fusion

Pipeline: Guardrail → [Tree Search | Hybrid Retrieval] → Context Assembly → Answer Generation → Grounding Check
                                                                                                      ↓ (retry)
                                                                                                 Query Rewrite → ...

Each node is a pure async function. No classes, no closures, no magic.
State flows through PipelineState. All LLM calls go through services/llm.py.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import prompts
from app.config import RetrievalStrategy, settings
from app.models.schemas import Citation, PipelineStep
from app.services.governance import active_policy
from app.models.state import PipelineState
from app.services.domain_profile import default_profile
from app.services.guardrails import detect_prompt_injection, redact_pii
from app.services.llm import chat, chat_json, clear_request_model, use_request_model
from app.services.observability import span
from app.services.registry import clear_document, current_bundle, registry, use_document
from app.services.reranker import reranker
from app.services.tree_search import rewrite_query, search_tree

logger = logging.getLogger(__name__)


def _active_profile():
    """The active document's domain profile, or a permissive default."""
    bundle = current_bundle()
    return bundle.profile if bundle else default_profile()


def _domain_context() -> str:
    """Short domain description (+ topics) injected into navigation/rewrite prompts.

    Keeps tree search document-agnostic: the navigator is primed by the actual
    document's profile instead of a hardcoded subject.
    """
    profile = _active_profile()
    parts = [(profile.description or "").strip()]
    if profile.topics:
        parts.append("Topics: " + ", ".join(profile.topics[:8]))
    return " ".join(p for p in parts if p).strip()


def _split_pages_into_passages(pages: list[dict]) -> list[dict]:
    """Split retrieved full pages into focused, paragraph-aligned passages.

    A cross-encoder reranks far better over passages than over whole pages (which
    it would truncate). Each passage keeps its page number and section so citations
    and grounding stay page-accurate.
    """
    size = settings.pageindex_passage_chars
    out: list[dict] = []
    for p in pages:
        text = (p.get("text") or "").strip()
        page_num = p.get("page_number")
        node_id = p.get("node_id", "")
        section = p.get("section_title", "")
        if len(text) <= size:
            out.append({**p, "chunk_id": p.get("chunk_id") or f"{node_id}_p{page_num}"})
            continue

        # Pack paragraphs into ~size-char windows; emit each as a passage.
        idx = 0
        buf = ""
        paras = [para for para in text.split("\n\n") if para.strip()]
        for para in paras:
            if buf and len(buf) + len(para) + 2 > size:
                out.append({
                    "page_number": page_num, "text": buf.strip(), "section_title": section,
                    "node_id": node_id, "chunk_id": f"{node_id}_p{page_num}_{idx}",
                })
                idx += 1
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf.strip():
            out.append({
                "page_number": page_num, "text": buf.strip(), "section_title": section,
                "node_id": node_id, "chunk_id": f"{node_id}_p{page_num}_{idx}",
            })
    return out


def _traced(name: str, fn):
    """Wrap a pipeline node so each execution emits an OTel span."""

    @functools.wraps(fn)
    async def wrapper(state):
        with span(f"node.{name}", {"agent.node": name}):
            return await fn(state)

    return wrapper


# ═══════════════════════════════════════════════════════════════════════
# NODE 0: SAFETY CHECK (input guardrail)
# Block prompt-injection / jailbreak attempts before doing any work.
# ═══════════════════════════════════════════════════════════════════════

async def safety_check_node(state: PipelineState) -> dict:
    start = time.time()
    query = state["original_query"]

    if not settings.injection_guard_enabled:
        return {"safety_blocked": False, "pipeline_steps": [{
            "node_name": "safety_check", "status": "completed",
            "thinking": "Injection guard disabled.", "result": "skipped",
            "duration_ms": (time.time() - start) * 1000, "tokens_used": 0,
        }]}

    injected, reason = detect_prompt_injection(query)
    return {
        "safety_blocked": injected,
        "guardrail_reasoning": reason if injected else state.get("guardrail_reasoning", ""),
        "pipeline_steps": [{
            "node_name": "safety_check", "status": "completed",
            "thinking": reason or "No injection patterns detected.",
            "result": "BLOCKED (prompt injection)" if injected else "clean",
            "duration_ms": (time.time() - start) * 1000, "tokens_used": 0,
        }],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 0.5: QUERY UNDERSTANDING
# Rewrite query contextually based on conversation history and summarize memory
# ═══════════════════════════════════════════════════════════════════════

async def query_understanding_node(state: PipelineState) -> dict:
    start = time.time()
    query = state["original_query"]
    messages = state.get("messages", [])
    summary = state.get("conversation_summary", "")

    # Exclude the very last message since it's the current query
    history_msgs = messages[:-1] if len(messages) > 0 else []

    if not history_msgs and not summary:
        return {
            "pipeline_steps": [{
                "node_name": "query_understanding", "status": "completed",
                "thinking": "No history to consider.", "result": "skipped",
                "duration_ms": (time.time() - start) * 1000, "tokens_used": 0,
            }]
        }

    total_tokens = 0
    new_summary = summary
    
    # Context Summarization: if history gets long (e.g. > 4 messages = 2 turns)
    if len(history_msgs) >= 4:
        # Compact everything except the most recent 2 turns
        msgs_to_summarize = history_msgs[:-4]
        if msgs_to_summarize:
            new_text = "\n".join([f"{m.type}: {m.content}" for m in msgs_to_summarize])
            prompt = prompts.summarize_conversation(summary, new_text)
            resp = await chat(prompt, temperature=0)
            new_summary = resp.content.strip()
            total_tokens += resp.tokens_used
            # Note: We keep `messages` full in the state for auditing, 
            # but only pass summary + latest to the rewrite LLM to save tokens.
            
    # Prepare context for rewrite
    recent_msgs = history_msgs[-4:]
    history_text = new_summary + "\n" if new_summary else ""
    history_text += "\n".join([f"{m.type}: {m.content}" for m in recent_msgs])

    prompt = prompts.query_understanding(history_text, query)
    result, tokens = await chat_json(prompt, temperature=0)
    total_tokens += tokens
    
    rewritten = result.get("rewritten", query)
    reasoning = result.get("reasoning", "")

    return {
        "conversation_summary": new_summary,
        "rewritten_query": rewritten if rewritten != query else None,
        "pipeline_steps": [{
            "node_name": "query_understanding", "status": "completed",
            "thinking": reasoning or "Rewrote based on history.", 
            "result": f"Rewritten: {rewritten}",
            "duration_ms": (time.time() - start) * 1000, "tokens_used": total_tokens,
        }]
    }

# ═══════════════════════════════════════════════════════════════════════
# NODE 1: GUARDRAIL
# Is this question about admissions / the university guide?
# ═══════════════════════════════════════════════════════════════════════

async def guardrail_node(state: PipelineState) -> dict:
    """Score how relevant the query is to the admission guide.

    Returns a 0-100 score. Below threshold → polite redirect.
    This prevents the LLM from hallucinating answers to off-topic questions.
    """
    start = time.time()
    query = state["original_query"]

    scope_context = _active_profile().guardrail_context()

    prompt = prompts.scope_guardrail(scope_context, query)

    result, tokens = await chat_json(prompt, temperature=0)
    score = result.get("score", 50)
    reasoning = result.get("reasoning", "")
    duration = (time.time() - start) * 1000

    step = {
        "node_name": "guardrail",
        "status": "completed",
        "thinking": reasoning,
        "result": f"Score: {score}/100",
        "duration_ms": duration,
        "tokens_used": tokens,
    }

    if score < settings.guardrail_threshold:
        return {
            "guardrail_score": score,
            "guardrail_reasoning": reasoning,
            "routing_decision": "out_of_scope",
            "pipeline_steps": [step],
        }

    return {
        "guardrail_score": score,
        "guardrail_reasoning": reasoning,
        "routing_decision": "continue",
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 2A: TREE SEARCH (PageIndex strategy)
# Navigate the PageIndex tree to find relevant sections
# ═══════════════════════════════════════════════════════════════════════

async def tree_search_node(state: PipelineState) -> dict:
    """Use LLM reasoning to navigate the document tree.

    The LLM sees the full tree structure and reasons about which
    sections likely contain the answer. This replaces vector similarity
    with actual understanding.
    """
    start = time.time()
    query = state.get("rewritten_query") or state["original_query"]
    attempts = state.get("retrieval_attempts", 0) + 1

    bundle = current_bundle()
    if bundle:
        di = bundle.document_index
        result = await search_tree(
            query,
            tree=di.tree_nodes(),
            valid_ids=di.node_ids(),
            domain_context=_domain_context(),
        )
    else:
        result = {"thinking": "No document indexed.", "node_ids": [], "confidence": 0.0, "tokens_used": 0}

    duration = (time.time() - start) * 1000

    step = {
        "node_name": "tree_search",
        "status": "completed",
        "thinking": result["thinking"],
        "result": f"Selected {len(result['node_ids'])} section(s) (confidence: {result['confidence']:.0%})",
        "duration_ms": duration,
        "tokens_used": result.get("tokens_used", 0),
    }

    return {
        "target_node_ids": result["node_ids"],
        "search_thinking": result["thinking"],
        "retrieval_attempts": attempts,
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 2B: HYBRID RETRIEVAL (BM25 + Vector strategy)
# Search using BM25 + Qdrant vector with RRF fusion
# ═══════════════════════════════════════════════════════════════════════

async def hybrid_retrieval_node(state: PipelineState) -> dict:
    """Retrieve relevant chunks using BM25 + Vector hybrid search.

    Uses RRF (Reciprocal Rank Fusion) to combine sparse and dense results.
    """
    start = time.time()
    query = state.get("rewritten_query") or state["original_query"]
    attempts = state.get("retrieval_attempts", 0) + 1
    strategy_str = state.get("retrieval_strategy", settings.active_retrieval.value)

    # Map string back to enum
    strategy_map = {s.value: s for s in RetrievalStrategy}
    strategy = strategy_map.get(strategy_str, settings.active_retrieval)

    # Cast a wide net: when reranking is on, retrieve the full candidate pool
    # (rerank_candidates) so the cross-encoder has real choices to promote —
    # not just the top handful that fused retrieval already favored.
    pool_k = settings.rerank_candidates if settings.rerank_enabled else settings.retrieval_top_k

    bundle = current_bundle()
    results = await bundle.hybrid_index.search(query, strategy=strategy, top_k=pool_k) if bundle else []

    duration = (time.time() - start) * 1000

    # Keep one candidate per chunk (not per page): distinct chunks on the same
    # page are different evidence, and collapsing them silently drops candidates
    # before the reranker ever sees them.
    retrieved_pages = []
    seen_chunks = set()
    for r in results:
        if r.chunk_id in seen_chunks:
            continue
        seen_chunks.add(r.chunk_id)
        retrieved_pages.append({
            "page_number": r.page_numbers[0] if r.page_numbers else 0,
            "page_numbers": r.page_numbers,
            "text": r.text,
            "section_title": r.section_title,
            "node_id": r.chunk_id,
            "chunk_id": r.chunk_id,
            "score": r.score,
            "source": r.source,
        })

    step = {
        "node_name": "hybrid_retrieval",
        "status": "completed",
        "thinking": f"Searched with {strategy.value} strategy",
        "result": f"Found {len(results)} chunks → {len(retrieved_pages)} candidates (strategy: {strategy.value})",
        "duration_ms": duration,
        "tokens_used": 0,
    }

    return {
        "hybrid_results": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "page_numbers": r.page_numbers,
                "section_title": r.section_title,
                "source": r.source,
            }
            for r in results
        ],
        "retrieved_pages": retrieved_pages,
        "retrieval_attempts": attempts,
        "search_thinking": f"Hybrid search ({strategy.value}): {len(results)} results",
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 3: PAGE RETRIEVAL (PageIndex path only)
# Extract full page text for the identified tree nodes
# ═══════════════════════════════════════════════════════════════════════

async def page_retrieval_node(state: PipelineState) -> dict:
    """Retrieve full page texts for the tree search results.

    Unlike vector RAG which returns arbitrary chunks, we retrieve
    complete pages — preserving tables, formatting, and context.
    """
    start = time.time()
    node_ids = state.get("target_node_ids", [])

    bundle = current_bundle()
    pages = bundle.document_index.get_pages_for_nodes(
        node_ids, max_pages=settings.pageindex_max_pages
    ) if bundle else []

    # Split full pages into focused passages so the reranker scores real evidence,
    # not truncated pages. (Skipped when reranking is off — full pages go through.)
    candidates = _split_pages_into_passages(pages) if (settings.rerank_enabled and pages) else pages

    duration = (time.time() - start) * 1000

    step = {
        "node_name": "page_retrieval",
        "status": "completed",
        "thinking": f"Retrieving pages for nodes: {', '.join(node_ids) or '(none)'}",
        "result": f"Retrieved {len(pages)} page(s) → {len(candidates)} passage(s)",
        "duration_ms": duration,
        "tokens_used": 0,
    }

    return {
        "retrieved_pages": candidates,
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 3.5: RERANK (both strategies)
# Cross-encoder rerank of retrieved candidates → keep the most relevant few
# ═══════════════════════════════════════════════════════════════════════

async def rerank_node(state: PipelineState) -> dict:
    """Rerank retrieved pages/chunks with a cross-encoder and keep the top-k.

    Cast-a-wide-net retrieval surfaces candidates; the reranker scores each
    against the query for precision. Degrades to retrieval order if the
    reranker model is unavailable or disabled.
    """
    start = time.time()
    pages = state.get("retrieved_pages", [])
    query = state.get("rewritten_query") or state["original_query"]

    if not pages:
        return {
            "pipeline_steps": [{
                "node_name": "rerank",
                "status": "completed",
                "thinking": "No candidates to rerank.",
                "result": "Skipped — no retrieved content.",
                "duration_ms": (time.time() - start) * 1000,
                "tokens_used": 0,
            }],
        }

    before = len(pages)
    final_k = settings.rerank_top_k

    # Stage 1 — cross-encoder rerank for precision. If MMR will run next, keep a
    # wider pool so diversification has candidates to choose among.
    stage1_k = settings.rerank_candidates if settings.mmr_enabled else final_k
    reranked = await reranker.rerank(query, pages, top_k=stage1_k)

    # Stage 2 — MMR diversity (opt-in): trade a little relevance for less
    # redundancy so the LLM sees varied passages, not N paraphrases.
    diversified = False
    if settings.mmr_enabled and len(reranked) > final_k:
        from app.services.diversity import mmr_select

        reranked = await mmr_select(query, reranked, k=final_k)
        diversified = True

    duration = (time.time() - start) * 1000

    step = {
        "node_name": "rerank",
        "status": "completed",
        "thinking": f"Cross-encoder reranking (active: {reranker.is_active})"
                    + (f" → MMR diversity (λ={settings.mmr_lambda})" if diversified else ""),
        "result": f"Reranked {before} → kept {len(reranked)} most relevant"
                  + (" (diversified)" if diversified else ""),
        "duration_ms": duration,
        "tokens_used": 0,
    }

    return {
        "retrieved_pages": reranked,
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 4: ANSWER GENERATION
# Synthesize a cited answer from the retrieved pages/chunks
# ═══════════════════════════════════════════════════════════════════════

async def answer_generation_node(state: PipelineState) -> dict:
    """Generate an answer from retrieved content with citations.

    Works with both page-based (PageIndex) and chunk-based (Hybrid) content.
    """
    start = time.time()
    query = state["original_query"]
    pages = state.get("retrieved_pages", [])
    strategy = state.get("retrieval_strategy", "pageindex")

    if not pages:
        return {
            "answer": "I couldn't find relevant information in the document for this question. "
                      "Could you try rephrasing your question?",
            "citations": [],
            "confidence": 0.0,
            "pipeline_steps": [{
                "node_name": "answer_generation",
                "status": "completed",
                "thinking": "No pages retrieved.",
                "result": "No answer generated — no source material.",
                "duration_ms": (time.time() - start) * 1000,
                "tokens_used": 0,
            }],
        }

    # Build context within a char budget so we never overflow the model window.
    budget = settings.answer_max_context_chars
    context_parts = []
    used = 0
    for page in pages:
        source_label = f"Page {page['page_number']}"
        if page.get("section_title"):
            source_label += f" ({page['section_title']})"
        if page.get("score"):
            source_label += f" [score: {page['score']:.3f}]"
        header = f"--- {source_label} ---\n"
        remaining = budget - used - len(header)
        if remaining <= 0:
            break
        body = (page.get("text") or "")[:remaining]
        context_parts.append(f"{header}{body}\n")
        used += len(header) + len(body) + 1
    context = "\n".join(context_parts)

    persona = _active_profile().answer_persona
    prompt = prompts.answer_generation(persona, query, context)

    response = await chat(prompt, temperature=0)
    content = response.content
    duration = (time.time() - start) * 1000
    tokens = response.tokens_used

    # Parse answer and citations
    answer, citations = _parse_answer_with_citations(content, pages)

    step = {
        "node_name": "answer_generation",
        "status": "completed",
        "thinking": f"Synthesized answer from {len(pages)} pages ({strategy})",
        "result": f"Generated {len(answer)} char answer with {len(citations)} citations",
        "duration_ms": duration,
        "tokens_used": tokens,
    }

    return {
        "answer": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 5: GROUNDING CHECK
# Verify every claim is supported by the source material
# ═══════════════════════════════════════════════════════════════════════

async def grounding_check_node(state: PipelineState) -> dict:
    """Verify the answer is grounded in the retrieved content.

    This is the anti-hallucination layer. The LLM checks whether
    every factual claim in the answer is supported by the pages.
    """
    start = time.time()
    answer = state.get("answer", "")
    pages = state.get("retrieved_pages", [])
    attempts = state.get("retrieval_attempts", 0)

    if not answer or not pages:
        return {
            "confidence": 0.0,
            "grounded": False,
            "unsupported_claims": [],
            "routing_decision": "complete",
            "pipeline_steps": [{
                "node_name": "grounding_check",
                "status": "completed",
                "thinking": "No answer to verify.",
                "result": "Skipped — no content.",
                "duration_ms": (time.time() - start) * 1000,
                "tokens_used": 0,
            }],
        }

    # Build source text from the FULL retrieved context (within a char budget),
    # not a 500-char prefix — claims supported later in a page must still verify.
    budget = settings.grounding_max_context_chars
    parts: list[str] = []
    used = 0
    for p in pages:
        label = f"[Page {p.get('page_number')}]"
        remaining = budget - used
        if remaining <= 0:
            break
        snippet = (p.get("text") or "")[:remaining]
        parts.append(f"{label}: {snippet}")
        used += len(snippet) + len(label) + 4
    source_text = "\n\n".join(parts)

    # Span-level verification: decompose the answer into claims and check each.
    prompt = prompts.grounding_check(answer, source_text)

    result, tokens = await chat_json(prompt, temperature=0)
    duration = (time.time() - start) * 1000

    claims = result.get("claims", []) if isinstance(result, dict) else []
    if claims:
        supported = [c for c in claims if c.get("supported")]
        confidence = len(supported) / len(claims)
        unsupported = [str(c.get("claim", "")).strip() for c in claims if not c.get("supported")]
        unsupported = [c for c in unsupported if c]
    else:
        # Parse failure or no decomposition — fall back to a coarse model score.
        confidence = float(result.get("confidence", 0.5)) if isinstance(result, dict) else 0.5
        unsupported = result.get("unsupported_claims", []) if isinstance(result, dict) else []

    # Risk-tier policy sets the bar: "standard" uses configured thresholds;
    # "high" raises acceptance and refuses-when-unsure (legal/medical/financial).
    policy = active_policy()
    grounded = confidence >= policy.grounding_accept_threshold

    if grounded:
        routing = "complete"
    elif attempts < settings.max_retrieval_attempts:
        routing = "retry"
    elif confidence < policy.grounding_refuse_threshold:
        routing = "refuse"          # honest decline beats a confident hallucination
    else:
        routing = "complete"        # accept with low confidence (caveated)

    routing_label = {
        "complete": " — accepted" if grounded else " — accepted (low confidence)",
        "retry": f" — retrying (attempt {attempts}/{settings.max_retrieval_attempts})",
        "refuse": " — insufficient grounding, declining",
    }[routing]

    step = {
        "node_name": "grounding_check",
        "status": "completed",
        "thinking": result.get("reasoning", "") if isinstance(result, dict) else "",
        "result": f"Confidence: {confidence:.0%} ({len(claims)} claims, "
                  f"{len(unsupported)} unsupported){routing_label}",
        "duration_ms": duration,
        "tokens_used": tokens,
    }

    return {
        "confidence": confidence,
        "grounded": grounded,
        "unsupported_claims": unsupported,
        "routing_decision": routing,
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 6: QUERY REWRITE (conditional, only on retry)
# ═══════════════════════════════════════════════════════════════════════

async def query_rewrite_node(state: PipelineState) -> dict:
    """Rewrite the query for a second retrieval attempt."""
    start = time.time()
    original = state["original_query"]
    previous_thinking = state.get("search_thinking", "")

    result = await rewrite_query(original, previous_thinking, domain_context=_domain_context())
    duration = (time.time() - start) * 1000

    step = {
        "node_name": "query_rewrite",
        "status": "completed",
        "thinking": result["reasoning"],
        "result": f"Rewritten: {result['rewritten_query']}",
        "duration_ms": duration,
        "tokens_used": result.get("tokens_used", 0),
    }

    return {
        "rewritten_query": result["rewritten_query"],
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 7: OUT OF SCOPE (terminal)
# ═══════════════════════════════════════════════════════════════════════

async def out_of_scope_node(state: PipelineState) -> dict:
    """Generate a polite redirect for off-topic questions."""
    query = state["original_query"]
    reasoning = state.get("guardrail_reasoning", "")

    topics = _active_profile().topics
    if topics:
        topic_lines = "\n".join(f"- **{t}**" for t in topics[:6])
        help_block = f"I can help with topics from this document, such as:\n{topic_lines}\n\n"
    else:
        help_block = "I can help with questions about the content of this document.\n\n"

    answer = (
        f"I'm designed to answer questions grounded in this specific document. "
        f"Your question about \"{query}\" appears to be outside its scope.\n\n"
        f"{help_block}"
        f"Would you like to ask about any of these?"
    )

    step = {
        "node_name": "out_of_scope",
        "status": "completed",
        "thinking": reasoning,
        "result": "Redirected — question outside the document's scope.",
        "duration_ms": 0,
        "tokens_used": 0,
    }

    return {
        "answer": answer,
        "citations": [],
        "confidence": 1.0,  # Confident it's out of scope
        "messages": [AIMessage(content=answer)],
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 8: INSUFFICIENT GROUNDING (terminal)
# Honest decline when the answer can't be verified — no confident hallucination
# ═══════════════════════════════════════════════════════════════════════

async def insufficient_grounding_node(state: PipelineState) -> dict:
    """Replace an unverifiable answer with an honest, grounded decline.

    The verification spine's payoff: rather than emit a confident answer the
    sources don't support, we state the grounding gap and what couldn't be
    confirmed, and point the user back to the document.
    """
    confidence = state.get("confidence", 0.0)
    unsupported = state.get("unsupported_claims", [])
    pages = state.get("retrieved_pages", [])
    page_nums = sorted({p["page_number"] for p in pages if p.get("page_number") is not None})

    msg = (
        f"I couldn't confidently verify an answer to this question from the document "
        f"(grounding confidence {confidence:.0%}), so I'd rather not guess.\n\n"
    )
    if unsupported:
        bullets = "\n".join(f"- {c}" for c in unsupported[:5])
        msg += f"These points could not be confirmed in the retrieved pages:\n{bullets}\n\n"
    if page_nums:
        page_list = ", ".join(str(n) for n in page_nums)
        msg += f"I reviewed page(s) {page_list}. "
    msg += "Try rephrasing the question, or check the source document directly for these specifics."

    step = {
        "node_name": "insufficient_grounding",
        "status": "completed",
        "thinking": f"{len(unsupported)} unsupported claim(s); confidence {confidence:.0%} "
                    f"below refuse threshold {settings.grounding_refuse_threshold:.0%}.",
        "result": "Declined — insufficient grounding.",
        "duration_ms": 0,
        "tokens_used": 0,
    }

    return {
        "answer": msg,
        "grounded": False,
        "messages": [AIMessage(content=msg)],
        "routing_decision": "refused",
        "pipeline_steps": [step],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 9: CLARIFY (HITL, opt-in) — pause on ambiguous questions
# ═══════════════════════════════════════════════════════════════════════

async def clarify_node(state: PipelineState) -> dict:
    """If the question is ambiguous, pause (interrupt) and ask the user.

    Opt-in via settings.hitl_enabled. Uses LangGraph interrupt(): the run pauses
    and surfaces a clarifying question; POST /api/ask/resume continues it with the
    user's answer. No-op pass-through when disabled or already clarified.
    """
    if not settings.hitl_enabled or state.get("clarified"):
        return {"clarified": True}

    start = time.time()
    query = state["original_query"]
    profile = _active_profile()
    topics = ", ".join(profile.topics) if profile.topics else "the document's subject"

    prompt = prompts.clarify(topics, query)

    result, tokens = await chat_json(prompt, temperature=0)
    ambiguous = bool(result.get("ambiguous")) if isinstance(result, dict) else False
    clarifying_q = (result.get("question") or "").strip() if isinstance(result, dict) else ""
    duration = (time.time() - start) * 1000

    if not ambiguous or not clarifying_q:
        return {
            "clarified": True,
            "pipeline_steps": [{
                "node_name": "clarify", "status": "completed",
                "thinking": "Question is specific enough.", "result": "No clarification needed.",
                "duration_ms": duration, "tokens_used": tokens,
            }],
        }

    # Pause for human input — resumes via Command(resume=<answer>).
    answer = interrupt({"type": "clarification", "question": clarifying_q, "original_query": query})

    return {
        "rewritten_query": f"{query}\n\n[User clarification: {answer}]",
        "clarified": True,
        "clarification": str(answer),
        "pipeline_steps": [{
            "node_name": "clarify", "status": "completed",
            "thinking": f"Asked: {clarifying_q}", "result": f"Clarified: {answer}",
            "duration_ms": duration, "tokens_used": tokens,
        }],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 10: GRADE CONTEXT (Corrective-RAG, opt-in) — re-retrieve if weak
# ═══════════════════════════════════════════════════════════════════════

async def grade_context_node(state: PipelineState) -> dict:
    """Grade retrieved context relevance; if too weak, route to query rewrite.

    Opt-in via settings.corrective_rag_enabled. No-op pass-through (→ answer)
    when disabled.
    """
    if not settings.corrective_rag_enabled:
        return {"routing_decision": "answer"}

    start = time.time()
    query = state.get("rewritten_query") or state["original_query"]
    pages = state.get("retrieved_pages", [])
    attempts = state.get("retrieval_attempts", 0)

    if not pages:
        routing = "rewrite" if attempts < settings.max_retrieval_attempts else "answer"
        return {
            "routing_decision": routing,
            "context_grade": 0.0,
            "pipeline_steps": [{
                "node_name": "grade_context", "status": "completed",
                "thinking": "No context retrieved.", "result": f"grade 0% → {routing}",
                "duration_ms": (time.time() - start) * 1000, "tokens_used": 0,
            }],
        }

    snippet = "\n".join(
        f"[Page {p.get('page_number')}] {(p.get('text') or '')[:300]}" for p in pages[:6]
    )
    prompt = prompts.grade_context(query, snippet)

    result, tokens = await chat_json(prompt, temperature=0)
    score = float(result.get("score", 0.5)) if isinstance(result, dict) else 0.5
    sufficient = score >= settings.corrective_grade_threshold
    routing = "answer" if (sufficient or attempts >= settings.max_retrieval_attempts) else "rewrite"

    return {
        "context_grade": score,
        "routing_decision": routing,
        "pipeline_steps": [{
            "node_name": "grade_context", "status": "completed",
            "thinking": result.get("reasoning", "") if isinstance(result, dict) else "",
            "result": f"Context grade {score:.0%} → {routing}",
            "duration_ms": (time.time() - start) * 1000, "tokens_used": tokens,
        }],
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 11: BLOCKED (terminal) — safe refusal for injection attempts
# ═══════════════════════════════════════════════════════════════════════

async def blocked_node(state: PipelineState) -> dict:
    answer = (
        "I can't help with that request. I answer questions grounded in this "
        "document and can't follow instructions that try to change my behavior."
    )
    return {
        "answer": answer,
        "citations": [],
        "confidence": 1.0,
        "grounded": False,
        "routing_decision": "blocked",
        "messages": [AIMessage(content=answer)],
        "pipeline_steps": [{
            "node_name": "blocked", "status": "completed",
            "thinking": state.get("guardrail_reasoning", ""),
            "result": "Request blocked by input guardrail.",
            "duration_ms": 0, "tokens_used": 0,
        }],
    }


# ═══════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def route_after_safety(state: PipelineState) -> str:
    """Injection detected → blocked; otherwise continue to the scope guardrail."""
    return "blocked" if state.get("safety_blocked") else "guardrail"

def route_after_guardrail(state: PipelineState) -> str:
    """In-scope → clarify (HITL gate); otherwise out_of_scope."""
    if state.get("routing_decision") == "out_of_scope":
        return "out_of_scope"
    return "clarify"


def _retrieval_route(state: PipelineState) -> str:
    strategy = state.get("retrieval_strategy", settings.active_retrieval.value)
    if strategy == RetrievalStrategy.PAGEINDEX.value:
        return "tree_search"
    return "hybrid_retrieval"


def route_after_clarify(state: PipelineState) -> str:
    """After clarification, route to the strategy-specific retrieval node."""
    return _retrieval_route(state)


def route_after_tree_search(state: PipelineState) -> str:
    """If navigation found sections → fetch pages; if it found nothing, rewrite and
    re-navigate (while attempts remain) instead of answering from an empty context."""
    if state.get("target_node_ids"):
        return "page_retrieval"
    if state.get("retrieval_attempts", 0) < settings.max_retrieval_attempts:
        return "query_rewrite"
    return "page_retrieval"  # exhausted retries → let the empty-context path decline honestly


def route_after_grade(state: PipelineState) -> str:
    """Corrective-RAG: re-retrieve if context graded weak, else answer."""
    if state.get("routing_decision") == "rewrite":
        return "query_rewrite"
    return "answer_generation"


def route_after_grounding(state: PipelineState) -> str:
    """Route based on grounding check results."""
    decision = state.get("routing_decision")
    if decision == "retry":
        return "query_rewrite"
    if decision == "refuse":
        return "insufficient_grounding"
    return END


def route_after_rewrite(state: PipelineState) -> str:
    """After query rewrite, go back to the right retrieval strategy."""
    strategy = state.get("retrieval_strategy", settings.active_retrieval.value)
    if strategy == RetrievalStrategy.PAGEINDEX.value:
        return "tree_search"
    return "hybrid_retrieval"


# ═══════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_pipeline(checkpointer) -> StateGraph:
    """Build and compile the LangGraph pipeline.

    Flow:
        START → guardrail ─┬─→ clarify ─(HITL pause?)→ [tree_search → page_retrieval | hybrid_retrieval]
                           │                                   → rerank → grade_context ─┬─→ answer_generation
                           │                                       (corrective-RAG)      └─(weak)→ query_rewrite ↺
                           └─→ out_of_scope → END
        answer_generation → grounding_check ─┬─→ END
                                             ├─(retry)→ query_rewrite ↺
                                             └─(refuse)→ insufficient_grounding → END
    """
    workflow = StateGraph(PipelineState)

    # Add all nodes
    workflow.add_node("query_understanding", _traced("query_understanding", query_understanding_node))
    workflow.add_node("safety_check", _traced("safety_check", safety_check_node))
    workflow.add_node("blocked", _traced("blocked", blocked_node))
    workflow.add_node("guardrail", _traced("guardrail", guardrail_node))
    workflow.add_node("clarify", _traced("clarify", clarify_node))
    workflow.add_node("grade_context", _traced("grade_context", grade_context_node))
    workflow.add_node("tree_search", _traced("tree_search", tree_search_node))
    workflow.add_node("hybrid_retrieval", _traced("hybrid_retrieval", hybrid_retrieval_node))
    workflow.add_node("page_retrieval", _traced("page_retrieval", page_retrieval_node))
    workflow.add_node("rerank", _traced("rerank", rerank_node))
    workflow.add_node("answer_generation", _traced("answer_generation", answer_generation_node))
    workflow.add_node("grounding_check", _traced("grounding_check", grounding_check_node))
    workflow.add_node("query_rewrite", _traced("query_rewrite", query_rewrite_node))
    workflow.add_node("out_of_scope", _traced("out_of_scope", out_of_scope_node))
    workflow.add_node("insufficient_grounding", _traced("insufficient_grounding", insufficient_grounding_node))

    # Entry: query understanding first, then safety check
    workflow.add_edge(START, "query_understanding")
    workflow.add_edge("query_understanding", "safety_check")
    workflow.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {"blocked": "blocked", "guardrail": "guardrail"},
    )
    workflow.add_edge("blocked", END)

    # Guardrail → clarify (HITL gate) or out_of_scope
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"clarify": "clarify", "out_of_scope": "out_of_scope"},
    )

    # Clarify → strategy-specific retrieval (after optional HITL pause)
    workflow.add_conditional_edges(
        "clarify",
        route_after_clarify,
        {"tree_search": "tree_search", "hybrid_retrieval": "hybrid_retrieval"},
    )

    workflow.add_edge("out_of_scope", END)

    # PageIndex path: tree_search → (page_retrieval | re-navigate) → rerank
    workflow.add_conditional_edges(
        "tree_search",
        route_after_tree_search,
        {"page_retrieval": "page_retrieval", "query_rewrite": "query_rewrite"},
    )
    workflow.add_edge("page_retrieval", "rerank")

    # Hybrid path: hybrid_retrieval → rerank
    workflow.add_edge("hybrid_retrieval", "rerank")

    # Rerank → corrective-RAG context grading → answer | re-retrieve
    workflow.add_edge("rerank", "grade_context")
    workflow.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {"query_rewrite": "query_rewrite", "answer_generation": "answer_generation"},
    )

    # Common tail: answer → grounding → end | retry | refuse
    workflow.add_edge("answer_generation", "grounding_check")

    workflow.add_conditional_edges(
        "grounding_check",
        route_after_grounding,
        {
            "query_rewrite": "query_rewrite",
            "insufficient_grounding": "insufficient_grounding",
            END: END,
        },
    )

    workflow.add_edge("insufficient_grounding", END)

    # Query rewrite routes back to the appropriate retrieval strategy
    workflow.add_conditional_edges(
        "query_rewrite",
        route_after_rewrite,
        {"tree_search": "tree_search", "hybrid_retrieval": "hybrid_retrieval"},
    )

    return workflow.compile(checkpointer=checkpointer)


async def _build_checkpointer():
    """Durable execution (Phase C): persist run state for resume/replay.

    Uses the ASYNC sqlite saver (the graph runs via ainvoke/astream, so a sync
    saver would block or be rejected). Falls back to in-memory if aiosqlite / the
    sqlite checkpoint package isn't available.
    """
    mode = (settings.checkpointer or "memory").lower()
    if mode == "sqlite":
        try:
            from pathlib import Path as _Path

            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            _Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(settings.checkpoint_db_path)
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            logger.info(f"Checkpointer: async sqlite ({settings.checkpoint_db_path})")
            return saver
        except Exception as e:
            logger.warning(f"AsyncSqliteSaver unavailable ({e}); using in-memory checkpointer.")

    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Checkpointer: in-memory")
    return MemorySaver()


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

# Compiled lazily on first use so the async checkpointer can be initialized
# inside the running event loop.
_compiled_pipeline = None
_compile_lock = asyncio.Lock()


async def get_pipeline():
    """Return the compiled pipeline, building it (with its checkpointer) once."""
    global _compiled_pipeline
    if _compiled_pipeline is None:
        async with _compile_lock:
            if _compiled_pipeline is None:
                checkpointer = await _build_checkpointer()
                _compiled_pipeline = build_pipeline(checkpointer)
    return _compiled_pipeline


def _format_result(result, *, query, active_strategy, model_used, bundle, thread_id, total_duration):
    """Shape a finished run into the public response dict."""
    total_tokens = sum(s.get("tokens_used", 0) for s in result.get("pipeline_steps", []))
    retrieved_pages = result.get("retrieved_pages", [])
    retrieval_context = [
        (f"[Page {p.get('page_number')}] " if p.get("page_number") else "") + p.get("text", "")
        for p in retrieved_pages
    ]
    retrieved_page_numbers = sorted({
        p["page_number"] for p in retrieved_pages if p.get("page_number") is not None
    })
    routing = result.get("routing_decision", "")
    answer = result.get("answer", "")
    if settings.pii_redaction_enabled:
        answer, _pii_found = redact_pii(answer)
    return {
        "query": query,
        "answer": answer,
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", 0.0),
        "pipeline_steps": result.get("pipeline_steps", []),
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "rewritten_query": result.get("rewritten_query"),
        "conversation_summary": result.get("conversation_summary"),
        "strategy_used": active_strategy,
        "model_used": model_used,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration,
        "retrieval_context": retrieval_context,
        "retrieved_page_numbers": retrieved_page_numbers,
        "out_of_scope": routing == "out_of_scope",
        "blocked": routing == "blocked",
        "grounded": result.get("grounded", False),
        "unsupported_claims": result.get("unsupported_claims", []),
        "refused": routing == "refused",
        "tenant_id": bundle.tenant_id if bundle else None,
        "doc_id": bundle.doc_id if bundle else None,
        "thread_id": thread_id,
        "indexed_at": getattr(bundle, "indexed_at", None) if bundle else None,
        "interrupted": False,
        "clarification": None,
    }


def _interrupt_response(result, **kwargs):
    """If the run paused (HITL), return an 'interrupted' response; else None."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    intr = interrupts[0]
    value = getattr(intr, "value", intr)
    payload = _format_result(result, **kwargs)
    payload["interrupted"] = True
    payload["clarification"] = value if isinstance(value, dict) else {"question": str(value)}
    return payload


async def ask(
    query: str,
    *,
    strategy: str | None = None,
    model_provider: str | None = None,
    tenant_id: str | None = None,
    doc_id: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """Execute the full pipeline for a question.

    Args:
        query: The user's question.
        strategy: Override retrieval strategy (pageindex/hybrid/bm25_only/vector_only).
        model_provider: Override model provider (openai/azure_openai/...).
        tenant_id / doc_id: Select which document bundle to answer from.
        thread_id: Durable-execution thread id (for resume/replay); auto if omitted.

    Returns:
        Complete response dict with answer, citations, pipeline_steps, etc.
    """
    start = time.time()

    # Resolve strategy and model
    active_strategy = strategy or settings.active_retrieval.value
    active_model = model_provider or settings.active_model.value

    # Resolve provider override WITHOUT mutating global settings (concurrency-safe).
    from app.config import ModelProvider as MP
    provider_override = {p.value: p for p in MP}.get(model_provider) if model_provider else None
    model_used = (
        settings.model_name_for(provider_override) if provider_override else settings.active_model_name
    )

    # Resolve the document bundle (multi-tenant) and a durable thread id.
    bundle = registry.resolve(tenant_id, doc_id)
    thread_id = thread_id or f"req-{uuid.uuid4().hex[:12]}"

    initial_state: PipelineState = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "rewritten_query": None,
        "clarified": False,
        "clarification": "",
        "context_grade": 0.0,
        "retrieval_strategy": active_strategy,
        "model_provider": active_model,
        "guardrail_score": 0,
        "guardrail_reasoning": "",
        "safety_blocked": False,
        "target_node_ids": [],
        "search_thinking": "",
        "hybrid_results": [],
        "retrieved_pages": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "unsupported_claims": [],
        "grounded": False,
        "retrieval_attempts": 0,
        "routing_decision": "",
        "pipeline_steps": [],
    }

    pipeline = await get_pipeline()
    provider_token = use_request_model(provider_override, None)
    doc_token = use_document(bundle)
    with span("rag.request", {
        "rag.strategy": active_strategy,
        "rag.tenant_id": bundle.tenant_id if bundle else None,
        "rag.doc_id": bundle.doc_id if bundle else None,
        "rag.thread_id": thread_id,
    }) as sp:
        try:
            result = await pipeline.ainvoke(
                initial_state, config={"configurable": {"thread_id": thread_id}}
            )
        finally:
            clear_request_model(provider_token)
            clear_document(doc_token)
        sp.set_attribute("rag.confidence", float(result.get("confidence", 0.0)))
        sp.set_attribute("rag.grounded", bool(result.get("grounded", False)))
        sp.set_attribute("rag.out_of_scope", result.get("routing_decision") == "out_of_scope")
    total_duration = (time.time() - start) * 1000
    shaping = dict(
        query=query, active_strategy=active_strategy, model_used=model_used,
        bundle=bundle, thread_id=thread_id, total_duration=total_duration,
    )
    interrupted = _interrupt_response(result, **shaping)
    return interrupted if interrupted is not None else _format_result(result, **shaping)


async def resume(
    thread_id: str,
    answer: str,
    *,
    tenant_id: str | None = None,
    doc_id: str | None = None,
    strategy: str | None = None,
    model_provider: str | None = None,
) -> dict:
    """Resume an interrupted (HITL) run with the user's clarification answer."""
    start = time.time()
    active_strategy = strategy or settings.active_retrieval.value

    from app.config import ModelProvider as MP
    provider_override = {p.value: p for p in MP}.get(model_provider) if model_provider else None
    model_used = (
        settings.model_name_for(provider_override) if provider_override else settings.active_model_name
    )
    bundle = registry.resolve(tenant_id, doc_id)
    pipeline = await get_pipeline()

    provider_token = use_request_model(provider_override, None)
    doc_token = use_document(bundle)
    with span("rag.resume", {
        "rag.thread_id": thread_id,
        "rag.doc_id": bundle.doc_id if bundle else None,
    }):
        try:
            result = await pipeline.ainvoke(
                Command(resume=answer), config={"configurable": {"thread_id": thread_id}}
            )
        finally:
            clear_request_model(provider_token)
            clear_document(doc_token)

    total_duration = (time.time() - start) * 1000
    query = result.get("original_query", "")
    shaping = dict(
        query=query, active_strategy=active_strategy, model_used=model_used,
        bundle=bundle, thread_id=thread_id, total_duration=total_duration,
    )
    interrupted = _interrupt_response(result, **shaping)
    return interrupted if interrupted is not None else _format_result(result, **shaping)


async def ask_streaming(
    query: str,
    *,
    strategy: str | None = None,
    model_provider: str | None = None,
    tenant_id: str | None = None,
    doc_id: str | None = None,
    thread_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Execute the pipeline with streaming events for SSE.

    Yields SSE-compatible event dicts as each node starts/completes.
    """
    start = time.time()

    active_strategy = strategy or settings.active_retrieval.value
    active_model = model_provider or settings.active_model.value

    # Resolve provider override WITHOUT mutating global settings (concurrency-safe).
    from app.config import ModelProvider as MP
    provider_override = {p.value: p for p in MP}.get(model_provider) if model_provider else None
    model_used = (
        settings.model_name_for(provider_override) if provider_override else settings.active_model_name
    )

    initial_state: PipelineState = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "rewritten_query": None,
        "clarified": False,
        "clarification": "",
        "context_grade": 0.0,
        "retrieval_strategy": active_strategy,
        "model_provider": active_model,
        "guardrail_score": 0,
        "guardrail_reasoning": "",
        "safety_blocked": False,
        "target_node_ids": [],
        "search_thinking": "",
        "hybrid_results": [],
        "retrieved_pages": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "unsupported_claims": [],
        "grounded": False,
        "retrieval_attempts": 0,
        "routing_decision": "",
        "pipeline_steps": [],
    }

    # Resolve the document bundle (multi-tenant) and a durable thread id.
    bundle = registry.resolve(tenant_id, doc_id)
    thread_id = thread_id or f"req-{uuid.uuid4().hex[:12]}"
    pipeline = await get_pipeline()

    # Stream node-by-node execution (provider + document are task-local, not global)
    provider_token = use_request_model(provider_override, None)
    doc_token = use_document(bundle)
    request_span = span("rag.request", {
        "rag.strategy": active_strategy,
        "rag.tenant_id": bundle.tenant_id if bundle else None,
        "rag.doc_id": bundle.doc_id if bundle else None,
        "rag.thread_id": thread_id,
        "rag.stream": True,
    })
    request_span.__enter__()
    interrupted = False
    try:
        async for event in pipeline.astream(
            initial_state,
            stream_mode="updates",
            config={"configurable": {"thread_id": thread_id}},
        ):
            # Each event is {node_name: state_update}
            for node_name, update in event.items():
                # LangGraph represents a HITL pause as {"__interrupt__": (Interrupt,...)},
                # not {node_name: state_dict} — handle it before treating `update` as a dict.
                if node_name == "__interrupt__":
                    interrupted = True
                    intr = update[0] if isinstance(update, (list, tuple)) and update else update
                    value = getattr(intr, "value", intr)
                    clarification = value if isinstance(value, dict) else {"question": str(value)}
                    yield {
                        "event": "interrupted",
                        "data": {
                            "interrupted": True,
                            "clarification": clarification,
                            "thread_id": thread_id,
                        },
                    }
                    continue

                # Emit node completion event with the new step data
                # With the operator.add reducer, each update carries only this
                # node's newly-appended step(s).
                for step in update.get("pipeline_steps", []):
                    yield {
                        "event": "node_completed",
                        "data": step,
                    }

                # If this update contains an answer, stream it
                if "answer" in update and update["answer"]:
                    streamed_answer = update["answer"]
                    if settings.pii_redaction_enabled:
                        streamed_answer, _ = redact_pii(streamed_answer)
                    yield {
                        "event": "answer",
                        "data": {
                            "answer": streamed_answer,
                            "citations": update.get("citations", []),
                            "confidence": update.get("confidence", 0.0),
                            "grounded": update.get("grounded", False),
                            "strategy_used": active_strategy,
                            "model_used": model_used,
                        },
                    }

        if not interrupted:
            total_duration = (time.time() - start) * 1000
            yield {
                "event": "pipeline_complete",
                "data": {
                    "total_duration_ms": total_duration,
                    "query": query,
                    "strategy_used": active_strategy,
                    "model_used": model_used,
                    "thread_id": thread_id,
                },
            }
    finally:
        clear_request_model(provider_token)
        clear_document(doc_token)
        request_span.__exit__(None, None, None)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _parse_answer_with_citations(
    content: str,
    pages: list[dict],
) -> tuple[str, list[dict]]:
    """Parse the LLM output into answer text and structured citations."""
    citations = []

    # Try to extract JSON citations block
    if "```citations" in content:
        parts = content.split("```citations")
        answer = parts[0].strip()
        try:
            json_str = parts[1].split("```")[0].strip()
            raw_citations = json.loads(json_str)
            for c in raw_citations:
                citations.append({
                    "page_numbers": c.get("page_numbers", []),
                    "section_title": c.get("section_title", ""),
                    "node_id": "",
                    "chunk_id": "",
                    "relevance": c.get("relevance", ""),
                })
        except (json.JSONDecodeError, IndexError):
            pass
    else:
        answer = content

    # If no structured citations were parsed, extract from page references in answer
    if not citations and pages:
        seen_pages = set()
        for page in pages:
            pn = page["page_number"]
            if pn not in seen_pages:
                seen_pages.add(pn)
                citations.append({
                    "page_numbers": [pn],
                    "section_title": page.get("section_title", ""),
                    "node_id": page.get("node_id", ""),
                    "chunk_id": page.get("chunk_id") or page.get("node_id", ""),
                    "relevance": "Source page for this answer.",
                })

    return answer, citations
