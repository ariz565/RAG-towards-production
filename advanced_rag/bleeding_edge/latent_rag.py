"""Bleeding edge — LatentRAG: reason and retrieve in latent space (conceptual toy).

IDEA
    Explicit agentic RAG decodes to text and calls the LLM on every hop (retrieve →
    read → think → retrieve …) — expensive and slow. LatentRAG keeps the loop in
    **continuous latent space**: maintain a latent "state" vector, retrieve the nearest
    document vector, fold it into the state, and iterate — *without* an LLM call or
    re-embedding per hop. Decode to text only once at the end. Research reports ~90%
    lower inference latency than explicit agentic loops.

THIS TOY
    A faithful *shape* (not a real latent model): the latent state is a vector; each hop
    is a cheap vector op over cached document embeddings; the only LLM call is the final
    decode. We count LLM calls to show the saving vs an explicit per-hop loop.

USE CASES
    Latency/cost-sensitive multi-hop retrieval where an explicit agent's per-step LLM
    calls dominate. Conceptual here — production needs a model trained to operate in
    latent space.

LIMITATIONS
    Less interpretable (no per-step natural-language trace); needs a purpose-built model
    in reality. Shown as a directional concept, clearly labeled.
"""

from __future__ import annotations

import math

from advanced_rag._harness import CORPUS, FakeLLM, banner, cosine, doc, embed

_DOC_VECS = {d["id"]: embed(d["text"] + " " + d["title"]) for d in CORPUS}


def _blend(a: list[float], b: list[float], alpha: float) -> list[float]:
    mixed = [alpha * x + (1 - alpha) * y for x, y in zip(a, b)]
    norm = math.sqrt(sum(v * v for v in mixed)) or 1.0
    return [v / norm for v in mixed]


def latent_rag(question: str, llm: FakeLLM, *, hops: int = 3, alpha: float = 0.6) -> dict:
    state = embed(question)            # 1 embedding; doc vectors are cached/precomputed
    visited: list[str] = []
    trajectory: list[tuple[str, float]] = []

    for _ in range(hops):
        ranked = sorted(
            ((i, cosine(state, v)) for i, v in _DOC_VECS.items() if i not in visited),
            key=lambda x: x[1], reverse=True,
        )
        if not ranked or ranked[0][1] <= 0:
            break
        doc_id, sim = ranked[0]
        visited.append(doc_id)
        trajectory.append((doc_id, round(sim, 3)))
        state = _blend(state, _DOC_VECS[doc_id], alpha)   # fold evidence into latent state — no LLM call

    answer = llm.answer(question, [doc(i) for i in visited])   # decode ONCE
    return {"trajectory": trajectory, "visited": visited, "answer": answer,
            "llm_calls": llm.calls, "explicit_equivalent_calls": hops + 1}


def main() -> None:
    banner("BLEEDING EDGE — LatentRAG (multi-hop in latent space; decode once)")
    q = "how is parental leave related to remote work and holidays"
    llm = FakeLLM()
    r = latent_rag(q, llm, hops=3)
    print(f"\nQ: {q}")
    print("  latent trajectory (doc, similarity-to-state):")
    for doc_id, sim in r["trajectory"]:
        print(f"    hop -> {doc_id}  (sim={sim})")
    print(f"  visited   : {r['visited']}")
    print(f"  answer    : {r['answer']}")
    print(f"  llm_calls : {r['llm_calls']}  vs explicit per-hop agent: "
          f"{r['explicit_equivalent_calls']}  → ~{round((1-1/r['explicit_equivalent_calls'])*100)}% fewer")
    print("\nTakeaway: the multi-hop walk happens with cheap vector ops over cached embeddings; "
          "only the final answer needs the LLM. That's the latency/cost win LatentRAG targets "
          "(this is a conceptual stand-in, not a trained latent model).")


if __name__ == "__main__":
    main()
