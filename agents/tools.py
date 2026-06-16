"""Tools the agent can call — each carries metadata + safety flags.

`calc`/`search` are read-only + idempotent (safe to auto-run, retry freely).
`delete_doc` is DESTRUCTIVE — flagged `dangerous`/`requires_approval` so the
SecureToolRunner gates it behind human approval. The corpus-search tool reuses
the retrieval lab (agentic RAG).
"""

from __future__ import annotations

from agents.base import Tool


def _calc(expr: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not expr or set(expr) - allowed:
        raise ValueError("only arithmetic expressions are allowed")
    return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - sandboxed char-set


def make_calculator() -> Tool:
    return Tool(
        name="calc",
        description="Evaluate an arithmetic expression, e.g. '3 * (4 + 1)'.",
        fn=_calc,
        parameters={"input": {"type": "string", "required": True, "max_length": 200}},
        read_only=True, idempotent=True, rate_limit_per_min=120,
    )


def make_corpus_search(top_k: int = 3) -> Tool:
    """Hybrid retrieval over the retrieval lab's sample corpus (offline)."""
    from retrieval.factory import get_retriever
    from retrieval.sample_data import CORPUS

    retriever = get_retriever("hybrid")
    retriever.index(CORPUS)

    def _search(query: str) -> str:
        hits = retriever.search(query, k=top_k)
        return " | ".join(f"[{h.id}] {h.text}" for h in hits) if hits else "no results"

    return Tool(
        name="search",
        description="Search the knowledge base for relevant passages.",
        fn=_search,
        parameters={"input": {"type": "string", "required": True, "max_length": 500}},
        read_only=True, idempotent=True,
    )


def make_delete_doc() -> Tool:
    """DESTRUCTIVE example — gated behind approval by the runner."""
    def _delete(doc_id: str) -> str:
        return f"(simulated) deleted document '{doc_id.strip()}'"

    return Tool(
        name="delete_doc",
        description="Delete a document by id. DESTRUCTIVE and irreversible.",
        fn=_delete,
        parameters={"input": {"type": "string", "required": True, "max_length": 64}},
        read_only=False, idempotent=False, dangerous=True, requires_approval=True,
    )


def default_toolset() -> list[Tool]:
    return [make_calculator(), make_corpus_search()]


def risky_toolset() -> list[Tool]:
    return [make_calculator(), make_corpus_search(), make_delete_doc()]
