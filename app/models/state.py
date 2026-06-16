"""LangGraph state — what flows between pipeline nodes.
Minimal. Every field must be read by at least one node.
Supports both PageIndex and Hybrid retrieval strategies.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PipelineState(TypedDict):
    """State shared across all nodes in the agentic pipeline.

    Follows LangGraph best practices:
    - messages uses add_messages reducer (append, not replace)
    - Raw data stored, formatted on demand
    - Each field owned by specific nodes
    """

    # ── Conversation ────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Query Tracking ──────────────────────────────────────
    original_query: str
    rewritten_query: str | None

    # ── HITL clarification (Phase E) ────────────────────────
    clarified: bool          # True once ambiguity has been resolved/skipped
    clarification: str       # the user's clarification answer (if any)

    # ── Corrective-RAG (Phase E) ────────────────────────────
    context_grade: float     # 0-1 relevance of retrieved context to the query

    # ── Strategy Selection ──────────────────────────────────
    retrieval_strategy: str  # "pageindex" | "hybrid" | "bm25_only" | "vector_only"
    model_provider: str      # "openai" | "gemini" | "ollama"

    # ── Guardrail ───────────────────────────────────────────
    guardrail_score: int  # 0-100, how relevant is this to admissions
    guardrail_reasoning: str
    safety_blocked: bool   # input guardrail: prompt-injection/jailbreak detected

    # ── Tree Search (PageIndex strategy) ────────────────────
    target_node_ids: list[str]  # PageIndex node IDs identified by LLM
    search_thinking: str  # LLM's reasoning about which sections to check

    # ── Hybrid Retrieval (BM25 + Vector strategy) ───────────
    hybrid_results: list[dict]  # [{chunk_id, text, score, page_numbers, ...}]

    # ── Page Retrieval ──────────────────────────────────────
    retrieved_pages: list[dict]  # [{page_number, text, section_title}]

    # ── Answer Generation ───────────────────────────────────
    answer: str
    citations: list[dict]  # [{page_numbers, section_title, node_id, relevance}]
    confidence: float  # 0-1, grounding confidence

    # ── Verification (Phase B) ──────────────────────────────
    unsupported_claims: list[str]  # claims the grounding check could not verify
    grounded: bool                 # True if confidence ≥ accept threshold

    # ── Pipeline Control ────────────────────────────────────
    retrieval_attempts: int
    routing_decision: str  # "continue" | "out_of_scope" | "retry" | "refuse" | "complete"

    # ── Observability ───────────────────────────────────────
    # operator.add reducer → each node returns only its OWN step(s); LangGraph
    # concatenates them (correct under fan-out, and clean for streaming).
    pipeline_steps: Annotated[list[dict], operator.add]
