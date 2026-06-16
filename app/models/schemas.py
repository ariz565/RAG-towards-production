"""Pydantic schemas — the contract between frontend and backend.

Every field earns its place. Every type tells a story.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────


class AskRequest(BaseModel):
    """A question about the admission guide."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to ask about the admission guide.",
        examples=["What is the GPA requirement for Computer Science?"],
    )
    strategy: str | None = Field(
        default=None,
        description="Retrieval strategy override: pageindex | hybrid | bm25_only | vector_only. "
                    "If None, uses server default.",
    )
    model_provider: str | None = Field(
        default=None,
        description="Model provider override: openai | azure_openai | openrouter | groq | gemini | ollama. "
                    "If None, uses server default.",
    )
    tenant_id: str | None = Field(
        default=None,
        description="Tenant to scope the document lookup. If None, uses the default tenant.",
    )
    doc_id: str | None = Field(
        default=None,
        description="Document to answer from. If None, the router picks the best match.",
    )
    family: str | None = Field(
        default=None,
        description="Versioned document family; resolves to the latest (or as_of) version.",
    )
    as_of: str | None = Field(
        default=None,
        description="ISO date — answer from the version effective on/before this date.",
    )

# Resume an interrupted (HITL) run with the user's clarification answer.
class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread id from the interrupted response.")
    answer: str = Field(..., min_length=1, description="The user's clarification answer.")
    tenant_id: str | None = Field(default=None, description="Tenant of the original request.")
    doc_id: str | None = Field(default=None, description="Document of the original request.")
    strategy: str | None = Field(default=None, description="Retrieval strategy override.")
    model_provider: str | None = Field(default=None, description="Model provider override.")


# ── Auth ─────────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str = Field(..., description="User email.")
    password: str = Field(..., min_length=8, description="Password (min 8 chars).")


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    email: str = ""


class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    tenant_id: str
    indexed: bool = False
    message: str = ""


class SummarizeRequest(BaseModel):
    doc_id: str | None = Field(default=None, description="Document to summarize (defaults to the tenant's doc).")
    style: str | None = Field(default="concise", description="Summary style hint, e.g. 'concise' | 'detailed'.")


class SummaryResponse(BaseModel):
    doc_id: str
    summary: str
    chunks: int = 0
    reduce_levels: int = Field(default=0, description="Map-reduce tree depth used.")


class JobResponse(BaseModel):
    job_id: str
    kind: str = ""
    status: str = "pending"           # pending | running | done | failed
    result: dict | None = None
    error: str = ""


# Request to index a PDF document
class IndexRequest(BaseModel):

    filename: str = Field(
        ...,
        description="Name of the PDF file in the data/pdf directory.",
        examples=["admission-guide.pdf"],
    )


# ── Response Models ─────────────────────────────────────────────────

# A reference to a specific location in the document.
class Citation(BaseModel):

    page_numbers: list[int] = Field(
        ...,
        description="Page numbers where the information was found.",
    )
    section_title: str = Field(
        default="",
        description="Title of the section containing the information.",
    )
    node_id: str = Field(
        default="",
        description="PageIndex node ID for tree navigation.",
    )
    chunk_id: str = Field(
        default="",
        description="Chunk identifier this citation came from (hybrid retrieval).",
    )
    relevance: str = Field(
        default="",
        description="Why this section is relevant to the query.",
    )

# A single step in the agentic pipeline — for frontend visualization.
class PipelineStep(BaseModel):

    node_name: str = Field(..., description="Pipeline node identifier.")
    status: str = Field(
        default="pending",
        description="Current status: pending | thinking | completed | error",
    )
    thinking: str = Field(default="", description="The LLM's reasoning at this step.")
    result: str = Field(default="", description="Output of this step.")
    duration_ms: float = Field(default=0, description="Time taken in milliseconds.")
    tokens_used: int = Field(default=0, description="Tokens consumed at this step.")

# The complete response to a question.
class AskResponse(BaseModel):

    query: str = Field(..., description="The original question.")
    answer: str = Field(..., description="The synthesized answer with citations.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Page references supporting the answer.",
    )
    pipeline_steps: list[PipelineStep] = Field(
        default_factory=list,
        description="Execution trace of the agentic pipeline.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) based on grounding check.",
    )
    retrieval_attempts: int = Field(
        default=1,
        description="Number of tree search attempts made.",
    )
    rewritten_query: str | None = Field(
        default=None,
        description="Rewritten query if retried.",
    )
    strategy_used: str = Field(
        default="pageindex",
        description="Retrieval strategy that was used.",
    )
    model_used: str = Field(
        default="",
        description="LLM model that was used.",
    )
    total_tokens: int = Field(default=0, description="Total tokens consumed.")
    total_duration_ms: float = Field(default=0, description="Total pipeline time in ms.")
    retrieved_page_numbers: list[int] = Field(
        default_factory=list,
        description="Distinct page numbers used as grounding for the answer.",
    )
    out_of_scope: bool = Field(
        default=False,
        description="True if the question was routed out of scope by the guardrail.",
    )
    blocked: bool = Field(
        default=False,
        description="True if the input guardrail blocked the request (prompt injection).",
    )
    grounded: bool = Field(
        default=False,
        description="True if grounding confidence met the accept threshold.",
    )
    refused: bool = Field(
        default=False,
        description="True if the answer was declined due to insufficient grounding.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims the grounding check could not verify against the sources.",
    )
    tenant_id: str | None = Field(default=None, description="Tenant that served the answer.")
    doc_id: str | None = Field(default=None, description="Document that served the answer.")
    thread_id: str | None = Field(default=None, description="Durable-execution thread id (resume/replay).")
    indexed_at: str | None = Field(
        default=None,
        description="When the answering document was indexed (data freshness, ISO-8601).",
    )
    query_intent: str = Field(default="", description="Classified intent: factual|summarize|compare|chitchat.")
    understood_query: str | None = Field(
        default=None, description="Normalized/rewritten query actually used for retrieval.",
    )
    interrupted: bool = Field(
        default=False,
        description="True if the run paused for human clarification (resume via /api/ask/resume).",
    )
    clarification: dict | None = Field(
        default=None,
        description="Clarification request when interrupted: {type, question, original_query}.",
    )

# A node in the document tree for the tree explorer UI.
class TreeNode(BaseModel):

    node_id: str = Field(default="", description="Unique node identifier.")
    title: str = Field(..., description="Section title.")
    summary: str = Field(default="", description="Section summary.")
    page_start: int | None = Field(default=None, description="Starting page number.")
    page_end: int | None = Field(default=None, description="Ending page number.")
    children: list[TreeNode] = Field(
        default_factory=list,
        description="Child nodes in the hierarchy.",
    )


# The full document tree structure for the tree explorer UI.
class TreeResponse(BaseModel):

    doc_name: str = Field(..., description="Document filename.")
    total_pages: int = Field(default=0, description="Total page count.")
    tree: list[TreeNode] = Field(
        default_factory=list,
        description="Root-level tree nodes.",
    )

# Raw text content of a single page.
class PageResponse(BaseModel):
    page_number: int = Field(..., description="1-based page number.")
    text: str = Field(..., description="Extracted text content.")
    token_count: int = Field(default=0, description="Token count of the page text.")

# Response after indexing a document.
class IndexResponse(BaseModel):
    success: bool
    doc_name: str
    total_pages: int
    tree_nodes: int = Field(default=0, description="Number of nodes in the tree.")
    message: str = Field(default="")

# Service health check.
class HealthResponse(BaseModel):

    status: str = "ok"
    indexed: bool = False
    doc_name: str | None = None
    total_pages: int = 0
    documents: int = 0
    active_model: str = ""
    active_retrieval: str = ""
    active_embedding: str = ""
    hybrid_index_loaded: bool = False


# ── SSE Event Models ────────────────────────────────────────────────

# Server-Sent Event payload for streaming pipeline execution.
class SSEEvent(BaseModel):

    event: str = Field(
        ...,
        description="Event type: node_started | node_thinking | node_completed | answer_chunk | pipeline_complete | error",
    )
    data: dict = Field(default_factory=dict, description="Event payload.")
