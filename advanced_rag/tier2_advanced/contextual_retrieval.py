"""Tier 2 — Contextual Retrieval (Anthropic): situate each chunk before embedding.

PROBLEM
    Chunking splits a document, and a chunk read in isolation often loses what it's
    *about*. "Exceptions must be approved by the security team in writing." — which
    policy? An embedding of that lone sentence won't match "SEC-17 exceptions".

WORKING
    Before embedding, prepend a short, LLM-generated context that situates the chunk
    in its document ("This chunk is from Security Policy SEC-17, about remote access
    rules:"). Embed (and BM25-index) the *contextualized* chunk. Retrieval quality
    jumps because the chunk now carries its own context. (Anthropic reports large
    retrieval-failure reductions; pair it with BM25 + reranking.)

COST
    One small LLM call per chunk at index time (cacheable via prompt caching), and a
    re-index to take effect. Pure index-time technique — query path is unchanged.

USE CASES
    Long structured docs (policies, contracts, manuals) where chunks reference "this
    section / the policy / the above" without naming it.
"""

from __future__ import annotations

from advanced_rag._harness import FakeLLM, banner, cosine, embed

# A document split into chunks. Chunk 2 is self-referential and ambiguous alone.
DOC_TITLE = "Security Policy SEC-17"
CHUNKS = [
    "Policy SEC-17 requires multi-factor authentication for all remote access.",
    "Exceptions must be approved by the security team in writing.",   # ← ambiguous in isolation
]


def contextualize(chunk: str, llm: FakeLLM) -> str:
    """Real systems ask an LLM for the situating context; here it's deterministic."""
    llm._bump()
    return f"This chunk is from '{DOC_TITLE}', about remote-access security rules. {chunk}"


def main() -> None:
    banner("TIER 2 — Contextual Retrieval (situate chunks before embedding)")
    llm = FakeLLM()
    query = "what are the exceptions to SEC-17"
    qv = embed(query)

    print(f"\nQuery: {query!r}\n")
    print("Chunk 2 is the answer but is self-referential ('Exceptions must be approved...').\n")
    print(f"{'chunk':>7} | {'raw cosine':>11} | {'contextualized cosine':>22}")
    print("-" * 48)
    for i, chunk in enumerate(CHUNKS):
        raw = cosine(qv, embed(chunk))
        ctx = cosine(qv, embed(contextualize(chunk, llm)))
        print(f"{i:>7} | {raw:>11.3f} | {ctx:>22.3f}")

    raw2 = cosine(qv, embed(CHUNKS[1]))
    ctx2 = cosine(qv, embed(contextualize(CHUNKS[1], llm)))
    print(f"\nThe answer chunk (#1) goes from {raw2:.3f} -> {ctx2:.3f} similarity once "
          f"its document context ('SEC-17') is prepended — now it's retrievable.")
    print(f"(index-time llm_calls={llm.calls}; query path unchanged)")


if __name__ == "__main__":
    main()
