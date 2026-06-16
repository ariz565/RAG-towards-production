"""Bleeding edge — A-RAG: hierarchical retrieval *interfaces* exposed as tools.

WORKING
    Instead of a fixed retrieve→generate pipeline, give the model **tools** at
    different granularities and let it navigate:
      • keyword_search(q)  — exact/lexical (BM25): codes, IDs, rare terms.
      • semantic_search(q) — dense: meaning/paraphrase.
      • chunk_read(id)     — fetch the full passage once a candidate is found.
    The agent plans which tool to call, inspects results, and decides whether to read
    or search again — so retrieval *granularity* is chosen by reasoning, not fixed up
    front. (Here the planner is a deterministic stand-in for the model.)

USE CASES
    Heterogeneous questions where the right *access pattern* varies — a code lookup
    wants keyword search; a "what's the gist" wants semantic; both then want a read.
    Multi-hop QA where the agent searches, reads, and searches again.

LIMITATIONS
    More round-trips; the agent can loop or over-search. Cap tool calls (done here).
"""

from __future__ import annotations

import re

from advanced_rag._harness import FakeLLM, banner, bm25_search, dense_search, doc

_CODE_RE = re.compile(r"[A-Z]{2,}-?\d+|\border\s+\d+", re.I)
MAX_TOOL_CALLS = 4


def keyword_search(q: str, k: int = 3):
    return [d for d, _ in bm25_search(q, k=k)]


def semantic_search(q: str, k: int = 3):
    return [d for d, _ in dense_search(q, k=k)]


def chunk_read(doc_id: str) -> dict:
    return doc(doc_id)


def a_rag(question: str, llm: FakeLLM) -> dict:
    trace: list[str] = []
    calls = 0

    # The agent's plan: code-like queries → keyword first; else semantic first.
    primary, secondary = (keyword_search, semantic_search) if _CODE_RE.search(question) \
        else (semantic_search, keyword_search)
    primary_name = "keyword_search" if primary is keyword_search else "semantic_search"

    hits = primary(question); calls += 1
    trace.append(f"{primary_name}({question!r}) -> {hits}")

    # Inspect: if the primary tool returned too few candidates, try the other.
    if len(hits) < 2 and calls < MAX_TOOL_CALLS:
        more = secondary(question); calls += 1
        sec_name = "keyword_search" if secondary is keyword_search else "semantic_search"
        trace.append(f"{sec_name}({question!r}) -> {more}")
        hits = list(dict.fromkeys(hits + more))

    # Read the top candidate(s) at full granularity.
    to_read = hits[:2]
    contexts = []
    for doc_id in to_read:
        if calls >= MAX_TOOL_CALLS:
            break
        contexts.append(chunk_read(doc_id)); calls += 1
        trace.append(f"chunk_read({doc_id!r})")

    answer = llm.answer(question, contexts)
    return {"trace": trace, "tool_calls": calls, "answer": answer}


def main() -> None:
    banner("BLEEDING EDGE — A-RAG (agent picks keyword / semantic / chunk_read tools)")
    for q in [
        "what are the exceptions to SEC-17",      # code → keyword_search first
        "how much time off do I get",             # paraphrase → semantic_search first
    ]:
        llm = FakeLLM()
        r = a_rag(q, llm)
        print(f"\nQ: {q}")
        for step in r["trace"]:
            print(f"  · {step}")
        print(f"  tool_calls : {r['tool_calls']}")
        print(f"  answer     : {r['answer']}")
    print("\nTakeaway: the agent chooses the retrieval *granularity* — exact vs semantic vs "
          "full read — by reasoning about the query, instead of a one-size pipeline.")


if __name__ == "__main__":
    main()
