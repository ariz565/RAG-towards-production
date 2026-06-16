"""Bleeding edge — Direct Corpus Interaction (DCI): search raw text, no vectors.

WORKING
    Drop the embedding model and vector index entirely. Give the agent general-purpose
    tools over the raw corpus — ``grep`` (regex/substring search) and ``read`` (open a
    file/passage) — and let it navigate like an engineer would. For exact, symbolic, or
    structured queries this is often *more* precise than similarity search and needs
    **zero index build**.

USE CASES
    Codebases, logs, configs, legal text with exact identifiers — anywhere the query is
    a literal string ("SEC-17", "order 991", an error code) and an embedding only adds
    noise and infra. Also a strong fallback when no index exists yet.

LIMITATIONS
    No semantic recall — paraphrases that share no tokens won't match (pair DCI with
    semantic search for those). Regex can over/under-match; ranking is heuristic.
"""

from __future__ import annotations

import re

from advanced_rag._harness import CORPUS, FakeLLM, banner

_BY_ID = {d["id"]: d for d in CORPUS}


def grep(pattern: str) -> list[dict]:
    """Return corpus passages whose text matches the pattern (case-insensitive)."""
    rx = re.compile(pattern, re.I)
    hits = []
    for d in CORPUS:
        matches = [m.group(0) for m in rx.finditer(d["text"])]
        if matches:
            hits.append({"id": d["id"], "title": d["title"], "matches": matches})
    return hits


def read(doc_id: str) -> dict:
    return _BY_ID[doc_id]


def dci(question: str, pattern: str, llm: FakeLLM) -> dict:
    trace = [f"grep({pattern!r})"]
    hits = grep(pattern)
    trace[0] += f" -> {[h['id'] for h in hits]}"
    contexts = []
    for h in hits[:2]:
        contexts.append(read(h["id"]))
        trace.append(f"read({h['id']!r})")
    answer = llm.answer(question, contexts) if contexts else "No grep matches; no answer."
    return {"trace": trace, "hits": hits, "answer": answer}


def main() -> None:
    banner("BLEEDING EDGE — Direct Corpus Interaction (grep/read, no embeddings)")
    cases = [
        ("what are the exceptions to SEC-17", r"SEC-17"),
        ("which order was delayed", r"order\s+\d+"),
    ]
    for q, pattern in cases:
        llm = FakeLLM()
        r = dci(q, pattern, llm)
        print(f"\nQ: {q}")
        for step in r["trace"]:
            print(f"  · {step}")
        for h in r["hits"]:
            print(f"    grep hit {h['id']}: matched {h['matches']}")
        print(f"  answer : {r['answer']}")
    print("\nTakeaway: for exact/symbolic queries, grep over raw text is precise and needs "
          "NO embedding model or vector index — zero index cost. (Add semantic search for paraphrases.)")


if __name__ == "__main__":
    main()
