"""Configuration — single source of truth for all settings.

Centralized model registry + retrieval strategy switch.
Supports OpenAI, Azure OpenAI, OpenRouter, Groq, Google Gemini, Ollama (local),
and HuggingFace embeddings.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — the switches
# ═══════════════════════════════════════════════════════════════════════


class ModelProvider(str, Enum):
    """Which LLM backend to use."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    NVIDIA = "nvidia"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class RetrievalStrategy(str, Enum):
    """Which retrieval approach to use."""
    PAGEINDEX = "pageindex"       # Tree-based, vectorless (existing)
    HYBRID = "hybrid"             # BM25 + Vector (Qdrant) with RRF fusion
    BM25_ONLY = "bm25_only"      # Sparse retrieval only
    VECTOR_ONLY = "vector_only"  # Dense retrieval only


class EmbeddingProvider(str, Enum):
    """Which embedding model to use."""
    HUGGINGFACE = "huggingface"   # Free, local (all-MiniLM-L6-v2)
    OPENAI = "openai"             # text-embedding-3-small
    OLLAMA = "ollama"             # Local (nomic-embed-text)


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    Designed for experimentation: switch models and retrieval strategies
    via environment variables or the Streamlit UI at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ═══════════════════════════════════════════════════════════════
    # ACTIVE SWITCHES — change these to experiment
    # ═══════════════════════════════════════════════════════════════
    active_model: ModelProvider = ModelProvider.OLLAMA
    active_retrieval: RetrievalStrategy = RetrievalStrategy.HYBRID
    active_embedding: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE

    # ═══════════════════════════════════════════════════════════════
    # MODEL REGISTRY — credentials + model names for each provider
    # ═══════════════════════════════════════════════════════════════
    # Bring your keys → set the provider → everything just works.
    # Each block: credentials + model + sampling overrides.
    # Sampling falls back to Global Sampling Defaults unless overridden.

    # ── OpenAI  (langchain-openai, openai SDK) ────────────────────
    openai_api_key: str = ""
    openai_organization: str = ""           # optional; enterprise accounts
    openai_model: str = "gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-small"
    # Sampling overrides (None → falls back to global defaults)
    openai_max_tokens: int | None = None
    openai_top_p: float | None = None
    openai_presence_penalty: float = 0.0   # -2.0..2.0; penalises repeated topics
    openai_frequency_penalty: float = 0.0  # -2.0..2.0; penalises repeated tokens
    openai_reasoning_effort: str = "medium"  # o-series | GPT-5 series: low | medium | high | xhigh

    # ── Azure OpenAI ────────────────────────────────────────
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_deployment: str = "gpt-4o-mini"
    # Sampling: Azure shares openai_* overrides above.

    # ── OpenRouter (OpenAI-compatible API) ─────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    # Sampling overrides (None → falls back to global defaults)
    openrouter_max_tokens: int | None = None
    openrouter_top_p: float | None = None

    # ── Groq (OpenAI-compatible API) ───────────────────────
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    # Sampling overrides (None → falls back to global defaults)
    groq_max_tokens: int | None = None
    groq_top_p: float | None = None

    # ── NVIDIA NIM (langchain-nvidia-ai-endpoints) ──────────────────
    nvidia_api_key: str = ""
    nvidia_model: str = "deepseek-ai/deepseek-v4-pro"
    nvidia_max_tokens: int = 16384
    nvidia_top_p: float = 0.95
    nvidia_top_k: int | None = None
    nvidia_thinking: bool = False           # thinking toggle (extra_body)

    # ── Google Gemini ──────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # Sampling overrides (None → falls back to global defaults)
    gemini_max_tokens: int | None = None
    gemini_top_p: float | None = None
    gemini_top_k: int | None = None        # Gemini natively supports top_k

    # ── Anthropic Claude ────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"
    # Sampling overrides (None → falls back to global defaults)
    anthropic_max_tokens: int | None = None
    anthropic_top_p: float | None = None
    anthropic_top_k: int | None = None     # Anthropic natively supports top_k
    anthropic_thinking: bool = False       # forces temperature=1 when True
    anthropic_thinking_budget: int = 8000  # budget_tokens; ignored when thinking=False

    # ── AWS Bedrock (ChatBedrockConverse, IAM credentials via boto3) ──
    bedrock_region: str = "us-east-1"
    bedrock_model: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    # Sampling overrides (None → falls back to global defaults)
    bedrock_max_tokens: int | None = None
    bedrock_top_p: float | None = None
    bedrock_top_k: int | None = None

    # ── Ollama  (langchain-ollama, local server) ──────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"
    # Sampling overrides (None → falls back to global defaults)
    ollama_max_tokens: int | None = None   # maps to num_predict
    ollama_top_p: float | None = None
    ollama_top_k: int | None = None
    ollama_repeat_penalty: float | None = None  # default 1.1 in Ollama
    ollama_num_ctx: int | None = None      # context window; None = model default

    # ── HuggingFace Embeddings (local / free) ──────────────
    hf_embedding_model: str = "all-MiniLM-L6-v2"
    hf_embedding_dim: int = 384

    # ── Ollama Embeddings (local / free) ───────────────────
    ollama_embedding_dim: int = 768   # nomic-embed-text

    # ── Global Sampling Defaults ──────────────────────────────────
    # Applied to all providers unless a provider-specific override is set.
    temperature: float = 0.0                 # generation randomness
    llm_max_tokens: int = 4096               # output token budget
    llm_top_p: float = 1.0                   # nucleus sampling threshold
    llm_top_k: int | None = None             # top-k; None = provider default
    llm_seed: int | None = None              # None = non-deterministic
    llm_stop_sequences: str = ""             # comma-separated stop strings

    # ── LLM resilience (timeouts + retry policy) ────────────
    # Retries only transient errors (429/5xx/timeouts) with exponential backoff
    # + full jitter, honoring Retry-After. Deterministic 4xx fail fast.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_retry_base_delay: float = 0.5
    llm_retry_max_delay: float = 20.0
    llm_retry_jitter: bool = True
    llm_retry_respect_retry_after: bool = True

    # ═══════════════════════════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════════════════════════
    pdf_dir: str = "data/pdf"
    index_dir: str = "data/index"
    bm25_dir: str = "data/bm25"
    chunks_dir: str = "data/chunks"

    # ═══════════════════════════════════════════════════════════════
    # SERVER
    # ═══════════════════════════════════════════════════════════════
    host: str = "0.0.0.0"
    port: int = 8001
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit
    ]

    # ═══════════════════════════════════════════════════════════════
    # QDRANT (Vector DB)
    # in_memory=False → persistent local storage at qdrant_path (no Docker,
    # survives restarts). use_server=True → connect to qdrant_host:qdrant_port.
    # Collections are per-document: f"{qdrant_collection}_{doc_id}".
    # ═══════════════════════════════════════════════════════════════
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "vision"     # collection-name prefix
    qdrant_in_memory: bool = False        # False = persist to disk and skip re-embed on boot
    qdrant_use_server: bool = False       # True = connect to host/port server
    qdrant_path: str = "data/qdrant"      # local persistent storage path

    # ═══════════════════════════════════════════════════════════════
    # MULTI-TENANCY (Phase C)
    # ═══════════════════════════════════════════════════════════════
    default_tenant: str = "default"

    # ═══════════════════════════════════════════════════════════════
    # STORAGE — where uploaded source PDFs live (pluggable)
    # "local" → data/pdf/<tenant>/ ; "s3" reserved for later.
    # ═══════════════════════════════════════════════════════════════
    storage_backend: str = "local"

    # ═══════════════════════════════════════════════════════════════
    # DATABASE (Postgres for production) — optional migration path
    # If set, enables dual-mode storage (JSON fallback if DB unavailable).
    # ═══════════════════════════════════════════════════════════════
    database_url: str = ""  # e.g. postgresql://user:pass@localhost/dbname
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 5
    pg_pool_max_inactive_connection_lifetime_seconds: float = 300.0
    pg_command_timeout_seconds: int = 10
    users_backend: str = "json"  # "json" | "postgres"
    versions_backend: str = "json"  # "json" | "postgres"
    audit_backend: str = "jsonl"  # "jsonl" | "postgres"

    # ═══════════════════════════════════════════════════════════════
    # AUTH — signup/login + signed bearer tokens
    # Stdlib only (no JWT/bcrypt dep). SET auth_secret VIA ENV IN PROD.
    # ═══════════════════════════════════════════════════════════════
    auth_secret: str = "dev-secret-change-me"
    auth_token_ttl_seconds: int = 86400
    users_db_path: str = "data/users.json"

    # ═══════════════════════════════════════════════════════════════
    # DURABLE EXECUTION — LangGraph checkpointer (Phase C)
    # "sqlite" persists run state (resume-after-crash); "memory" is ephemeral.
    # Falls back to memory if the sqlite checkpoint package isn't installed.
    # ═══════════════════════════════════════════════════════════════
    checkpointer: str = "sqlite"
    checkpoint_db_path: str = "data/checkpoints.sqlite"

    # ═══════════════════════════════════════════════════════════════
    # JOB STORE (persistence for async jobs like summarization)
    # SQLite-backed for durability across restarts.
    # ═══════════════════════════════════════════════════════════════
    jobs_db_path: str = "data/jobs.db"

    # ═══════════════════════════════════════════════════════════════
    # OBSERVABILITY (Phase D) — OpenTelemetry GenAI semantic conventions
    # Disabled → all spans are no-ops (zero overhead, no dependency needed).
    # Point otel_exporter_otlp_endpoint at Langfuse/Datadog/etc. (OTLP).
    # ═══════════════════════════════════════════════════════════════
    otel_enabled: bool = False
    otel_service_name: str = "vision"
    otel_exporter_otlp_endpoint: str = ""   # e.g. https://cloud.langfuse.com/api/public/otel
    otel_console_export: bool = False        # print spans to stdout for local dev
    otel_capture_content: bool = False       # include prompt/response text on spans (PII-sensitive)

    # ═══════════════════════════════════════════════════════════════
    # CHUNKING
    # ═══════════════════════════════════════════════════════════════
    chunk_size: int = 512          # tokens per chunk
    chunk_overlap: int = 100       # overlap between chunks
    semantic_merge_threshold: float = 0.8  # cosine sim to merge adjacent chunks

    # ═══════════════════════════════════════════════════════════════
    # HYBRID RETRIEVAL
    # ═══════════════════════════════════════════════════════════════
    retrieval_top_k: int = 10      # Final results after fusion
    bm25_top_k: int = 20           # BM25 candidates before fusion
    vector_top_k: int = 20         # Vector candidates before fusion
    rrf_k: int = 60                # RRF constant (standard = 60)

    # ═══════════════════════════════════════════════════════════════
    # PAGEINDEX (tree-based, vectorless retrieval)
    # ═══════════════════════════════════════════════════════════════
    toc_check_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20000

    # ── PageIndex navigation (production reasoning-based retrieval) ──
    # Small trees are navigated in one LLM call; trees whose compact rendering
    # exceeds the char budget are navigated hierarchically (level-by-level beam
    # descent) so we never overflow the context window on large documents.
    pageindex_tree_char_budget: int = 12000   # single-shot above this → hierarchical
    pageindex_max_depth: int = 4              # max levels to descend when navigating
    pageindex_max_select_nodes: int = 6       # cap on selected nodes
    pageindex_max_pages: int = 12             # cap total pages pulled (prevents parent-node blowup)
    pageindex_passage_chars: int = 1200       # split retrieved pages into passages this size for reranking
    pageindex_repair_navigation: bool = True  # re-prompt once if the model returns no valid node ids
    pageindex_preserve_reading_order: bool = True  # extract page text in reading order (multi-column safe)

    # ═══════════════════════════════════════════════════════════════
    # PIPELINE
    # ═══════════════════════════════════════════════════════════════
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 40

    # ── Query understanding (classify → normalize → rewrite) ──
    # Rewrite is separately gated: aggressive rewriting can hurt recall, so keep
    # it conservative and measure it on your eval set.
    query_understanding_enabled: bool = True
    query_rewrite_enabled: bool = True

    # ── Query expansion (multi-query, RAG-Fusion) ──
    # Generates paraphrases of the search query, retrieves for each, and fuses
    # with RRF — widens recall when the original phrasing misses relevant chunks.
    query_expansion_enabled: bool = True
    query_expansion_variants: int = 2

    # ── Web search fallback (Tavily) ──
    # When grounding_check would otherwise refuse (retries exhausted, confidence
    # still below the refuse threshold), try one web search before declining.
    # Opt-in: requires an API key, and a web-sourced answer is never presented
    # as grounded in the document.
    web_search_fallback_enabled: bool = False
    tavily_api_key: str = ""

    # ── Semantic answer cache ──
    # Reuse a past answer when a new query is close enough in *meaning* (not
    # just exact text) to one already answered for the same tenant+document.
    # Opt-in: a wrong cache hit silently serves a stale/wrong answer, which is
    # worse than a slower correct one, so this stays off until explicitly enabled.
    semantic_cache_enabled: bool = False
    semantic_cache_threshold: float = 0.95   # cosine similarity floor to treat as "same question"

    # ── Automatic reindex on source-document change ──
    # Periodically checks each loaded document's source PDF against its
    # recorded content hash (lineage.py) and incrementally re-indexes
    # (HybridIndex.update — same method app/cli.py's update-hybrid uses) if it
    # changed. On by default: unlike the cache/fallback features above, this
    # only closes a correctness gap (a stale index silently going unnoticed
    # until someone remembers to run the CLI), it doesn't introduce new risk.
    auto_reindex_enabled: bool = True
    auto_reindex_interval_seconds: int = 300

    # ── Document routing (multi-doc Q&A within a tenant) ────
    # When /api/ask has no doc_id, route the query to the best-matching document
    # by scoring against each doc's domain profile. Below min_score → no route.
    router_enabled: bool = True
    router_min_score: float = 0.15
    multi_doc_top_n: int = 3        # docs to fan out across for 'compare' queries

    # ── Summarization (map-reduce over large PDFs) ──────────
    summarize_map_chars: int = 4000          # per-chunk text fed to the map step
    summarize_reduce_char_budget: int = 8000  # collapse until combined fits this
    summarize_reduce_group_size: int = 10     # summaries combined per reduce node
    summarize_max_concurrency: int = 6
    summarize_max_map_calls: int = 200        # cap map LLM calls (merge chunks if more)

    # ── Safety guardrails ───────────────────────────────────
    # Input: block prompt-injection/jailbreak queries. Output: redact PII.
    injection_guard_enabled: bool = True
    pii_redaction_enabled: bool = True

    # ── Agentic depth (Phase E) — opt-in, off by default ────
    # HITL: pause on ambiguous questions and ask the user to clarify (interrupt
    # + /api/resume). Corrective-RAG: grade retrieved context before answering
    # and re-retrieve if it's too weak.
    hitl_enabled: bool = False
    corrective_rag_enabled: bool = False
    corrective_grade_threshold: float = 0.6

    # ═══════════════════════════════════════════════════════════════
    # GROUNDING / VERIFICATION (Phase B)
    # The fact-checker reads the FULL retrieved context (not a 500-char
    # prefix) and verifies every claim. Below the refuse threshold after
    # retries, the system declines honestly instead of hallucinating.
    # ═══════════════════════════════════════════════════════════════
    grounding_max_context_chars: int = 12000   # full-context budget for the fact-checker
    grounding_accept_threshold: float = 0.6    # ≥ this → accept the answer
    grounding_refuse_threshold: float = 0.5    # < this after retries → refuse honestly

    # Answer-generation context budget (prevents context-window overflow / cost spikes)
    answer_max_context_chars: int = 12000

    # ═══════════════════════════════════════════════════════════════
    # RERANKING (Phase B) — cross-encoder over retrieved candidates
    # Cast a wide net at retrieval, then rerank to the most relevant few.
    # Local cross-encoder (no API). Degrades gracefully if unavailable.
    # ═══════════════════════════════════════════════════════════════
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30   # candidate pool retrieved AND scored by the reranker
    rerank_top_k: int = 6         # items kept after reranking
    rerank_max_chars: int = 2400  # per-candidate text length scored by the cross-encoder

    # ═══════════════════════════════════════════════════════════════
    # MMR DIVERSITY — Maximal Marginal Relevance over the final candidates.
    # Trades a little relevance for less redundancy so the LLM sees varied
    # passages instead of N near-duplicates. Opt-in (embeds candidates).
    # ═══════════════════════════════════════════════════════════════
    mmr_enabled: bool = False
    mmr_lambda: float = 0.6       # 1.0 = pure relevance, 0.0 = pure diversity

    # ═══════════════════════════════════════════════════════════════
    # QUERY-EMBEDDING CACHE — the same query is embedded multiple times per
    # request (routing + vector search + retries). A small LRU avoids the
    # redundant calls. Keyed by (provider, model, text).
    # ═══════════════════════════════════════════════════════════════
    embedding_cache_enabled: bool = True
    embedding_cache_size: int = 2048

    # ═══════════════════════════════════════════════════════════════
    # CONTEXTUAL RETRIEVAL (Phase B) — Anthropic's technique
    # Prepend a short LLM-generated context to each chunk BEFORE embedding
    # so chunks are situated in the whole document. Opt-in: it adds one LLM
    # call per chunk at index time and requires re-indexing to take effect.
    # ═══════════════════════════════════════════════════════════════
    contextual_retrieval_enabled: bool = False
    contextual_doc_sample_chars: int = 4000   # document context shown to the contextizer
    contextual_max_concurrency: int = 8       # parallel chunk-context LLM calls

    # ═══════════════════════════════════════════════════════════════
    # DOMAIN PROFILE — makes the pipeline document-agnostic
    # The guardrail / answer / redirect prompts adapt to whatever
    # document is indexed instead of hardcoding "admission guide".
    # ═══════════════════════════════════════════════════════════════
    auto_generate_profile: bool = True       # derive a profile at index/startup time
    domain_description_override: str = ""     # optional manual override of the doc description
    domain_persona_override: str = ""         # optional manual override of the answerer persona

    # ═══════════════════════════════════════════════════════════════
    # AI GOVERNANCE — audit trail, document lineage, risk tiering
    # 2026 governance pillars: policy/risk tiering, data lineage, runtime
    # enforcement, and auditability ("where did this answer come from?").
    # ═══════════════════════════════════════════════════════════════
    governance_risk_tier: str = "standard"   # "standard" | "high" (high → stricter grounding + citations)
    audit_enabled: bool = True               # append-only record of every answered query
    audit_dir: str = "data/audit"
    audit_log_query_text: bool = True        # False → store only a hash of the query (max privacy)
    audit_retention_days: int = 90           # advisory; surfaced in GOVERNANCE.md / cleanup jobs
    lineage_enabled: bool = True             # source-certification + provenance per indexed document

    # ═══════════════════════════════════════════════════════════════
    # EVALUATION — golden-set gating (Ragas / DeepEval)
    # 2026 production defaults. The judge defaults to the active model
    # but a hosted model (e.g. gpt-4o-mini) gives far more reliable scores.
    # ═══════════════════════════════════════════════════════════════
    eval_judge_provider: str = ""             # "" => use active_model; else openai/azure_openai/...
    eval_judge_model: str = ""                # optional model-name override for the judge
    eval_faithfulness_threshold: float = 0.75
    eval_answer_relevancy_threshold: float = 0.80
    eval_context_precision_threshold: float = 0.70
    eval_context_recall_threshold: float = 0.80

    # ═══════════════════════════════════════════════════════════════
    # COMPUTED PROPERTIES
    # ═══════════════════════════════════════════════════════════════

    @property
    def pdf_path(self) -> Path:
        return Path(self.pdf_dir)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)

    @property
    def bm25_path(self) -> Path:
        return Path(self.bm25_dir)

    @property
    def chunks_path(self) -> Path:
        return Path(self.chunks_dir)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for the active provider."""
        if self.active_embedding == EmbeddingProvider.HUGGINGFACE:
            return self.hf_embedding_dim
        if self.active_embedding == EmbeddingProvider.OLLAMA:
            return self.ollama_embedding_dim
        return 1536  # OpenAI text-embedding-3-small

    def model_name_for(self, provider: ModelProvider) -> str:
        """Resolve the configured model name for any provider."""
        return {
            ModelProvider.OPENAI: self.openai_model,
            ModelProvider.AZURE_OPENAI: self.azure_openai_deployment,
            ModelProvider.OPENROUTER: self.openrouter_model,
            ModelProvider.ANTHROPIC: self.anthropic_model,
            ModelProvider.GROQ: self.groq_model,
            ModelProvider.NVIDIA: self.nvidia_model,
            ModelProvider.GEMINI: self.gemini_model,
            ModelProvider.BEDROCK: self.bedrock_model,
            ModelProvider.OLLAMA: self.ollama_model,
        }.get(provider, self.ollama_model)

    @property
    def active_model_name(self) -> str:
        """Get the active model name string."""
        return self.model_name_for(self.active_model)


settings = Settings()
