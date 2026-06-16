"""Production — Cost & latency control for agentic RAG.

Agentic loops are powerful but can run away: too many retrieve→reason rounds, and a
context that grows every round until it overflows the window (and the bill). Three
controls keep them production-safe:

  1. ITERATION CAP   — hard limit on agent rounds (≈3). Stop even if "not done".
  2. CONTEXT COMPRESSION — don't accumulate every chunk across rounds; keep only the
     query-relevant sentences within a char budget, so context stays bounded.
  3. STREAM / ASYNC  — for long runs, stream tokens or run as a background job so the
     request never blocks (shown conceptually; see the Vision app's async summarize).

USE CASES
    Any Tier-3 agentic deployment. Complex queries can otherwise cost 20–40× a simple
    one; caps + compression bound the worst case.

LIMITATIONS
    A too-tight cap can under-answer genuinely hard queries; compression can drop a
    needed detail. Tune against your eval set (see evaluation.py).
"""

from __future__ import annotations

import re

from advanced_rag._harness import FakeLLM, banner, dense_search, doc, tokenize

MAX_ITERS = 3
CONTEXT_BUDGET = 400   # chars


def compress(contexts: list[dict], question: str, budget: int = CONTEXT_BUDGET) -> str:
    """Keep only the most query-relevant sentences, within a char budget."""
    q = set(tokenize(question))
    scored: list[tuple[float, str]] = []
    for c in contexts:
        for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
            overlap = len(q & set(tokenize(sent)))
            if overlap:
                scored.append((overlap, f"{sent.strip()} [{c['id']}]"))
    scored.sort(key=lambda x: x[0], reverse=True)
    out, used = [], 0
    for _, sent in scored:
        if used + len(sent) > budget:
            break
        out.append(sent)
        used += len(sent)
    return " ".join(out)


def agentic_loop(question: str, llm: FakeLLM, *, cap: bool, compress_ctx: bool) -> dict:
    subs = llm.decompose(question) or [question]
    accumulated: list[dict] = []
    rounds = 0
    for sub in subs:
        if cap and rounds >= MAX_ITERS:
            break
        rounds += 1
        accumulated += [doc(d) for d, _ in dense_search(sub, k=2)]

    raw_chars = sum(len(c["text"]) for c in accumulated)
    if compress_ctx:
        context = compress(accumulated, question)
        ctx_chars = len(context)
    else:
        context = "\n".join(c["text"] for c in accumulated)
        ctx_chars = raw_chars

    return {"rounds": rounds, "subs": len(subs), "raw_chars": raw_chars,
            "ctx_chars": ctx_chars, "llm_calls": llm.calls}


def main() -> None:
    banner("PRODUCTION — Cost control (iteration cap + context compression)")
    q = ("compare annual leave and parental leave and remote work and holidays and "
         "expense reimbursement and security policy")  # decomposes into many sub-questions

    naive = agentic_loop(q, FakeLLM(), cap=False, compress_ctx=False)
    capped = agentic_loop(q, FakeLLM(), cap=True, compress_ctx=True)

    print(f"\nQ (intentionally sprawling): {q}\n")
    print(f"  {'config':<26} {'rounds':>7} {'ctx_chars':>10} {'llm_calls':>10}")
    print(f"  {'uncapped + accumulate':<26} {naive['rounds']:>7} {naive['ctx_chars']:>10} {naive['llm_calls']:>10}")
    print(f"  {'capped(3) + compressed':<26} {capped['rounds']:>7} {capped['ctx_chars']:>10} {capped['llm_calls']:>10}")
    saved = (1 - capped["ctx_chars"] / (naive["ctx_chars"] or 1)) * 100
    print(f"\n  context shrunk ~{saved:.0f}%, rounds bounded to {MAX_ITERS}.")
    print("\nTakeaway: the cap bounds latency/cost on sprawling queries, and compression keeps "
          "the context window (and bill) flat instead of growing every round. Stream/async long "
          "runs so the request never blocks.")


if __name__ == "__main__":
    main()
