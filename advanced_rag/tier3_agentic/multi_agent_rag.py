"""Tier 3 — Multi-Agent RAG: a supervisor orchestrates specialist retrievers.

WORKING
    Instead of one retriever over everything, split the corpus (or tools) by domain and
    give each a *specialist agent*. A supervisor:
      1. routes the query to the relevant specialist(s) (by domain affinity),
      2. fans out — each specialist retrieves + answers within its own scope,
      3. synthesizes the specialist answers into one response.
    Specialists keep context focused and let you mix heterogeneous sources (docs, SQL,
    APIs) behind a common interface.

USE CASES
    Cross-domain questions ("how does remote work interact with the SEC-17 security
    policy?") and heterogeneous backends where one flat index would blur domains.

LIMITATIONS
    Orchestration overhead and more calls; the supervisor's routing can miss a relevant
    specialist. Best when domains are genuinely distinct.
"""

from __future__ import annotations

from dataclasses import dataclass

from advanced_rag._harness import CORPUS, FakeLLM, banner, cosine, embed

# Partition the toy corpus into specialist domains.
DOMAINS: dict[str, set[str]] = {
    "people_ops": {"d1", "d2", "d3", "d8"},     # leave, parental, expense, holidays
    "security": {"d4", "d7"},                    # SEC-17, remote access
    "supply_chain": {"d5", "d6"},                # incidents, vendor SLAs
}
_BY_ID = {d["id"]: d for d in CORPUS}
_DOMAIN_TEXT = {
    name: " ".join(_BY_ID[i]["title"] + " " + _BY_ID[i]["text"] for i in ids)
    for name, ids in DOMAINS.items()
}


@dataclass
class Specialist:
    name: str
    doc_ids: set[str]

    def retrieve(self, question: str, k: int = 2) -> list[dict]:
        qv = embed(question)
        scored = [(i, cosine(qv, embed(_BY_ID[i]["text"]))) for i in self.doc_ids]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [_BY_ID[i] for i, _ in scored[:k]]

    def answer(self, question: str, llm: FakeLLM) -> dict:
        ctx = self.retrieve(question)
        return {"agent": self.name, "used": [c["id"] for c in ctx],
                "answer": llm.answer(question, ctx) if ctx else "(no in-scope evidence)"}


_SPECIALISTS = [Specialist(name, ids) for name, ids in DOMAINS.items()]


def route(question: str, *, min_affinity: float = 0.05) -> list[Specialist]:
    """Supervisor routing: pick specialists whose domain is similar to the query."""
    qv = embed(question)
    scored = [(sp, cosine(qv, embed(_DOMAIN_TEXT[sp.name]))) for sp in _SPECIALISTS]
    chosen = [sp for sp, s in sorted(scored, key=lambda x: x[1], reverse=True) if s >= min_affinity]
    return chosen[:2] or [max(scored, key=lambda x: x[1])[0]]


def multi_agent_rag(question: str, llm: FakeLLM) -> dict:
    specialists = route(question)
    partials = [sp.answer(question, llm) for sp in specialists]
    # Supervisor synthesis: concatenate distinct specialist findings.
    synthesis = " ".join(f"[{p['agent']}] {p['answer']}" for p in partials if p["answer"])
    return {"routed_to": [sp.name for sp in specialists], "partials": partials,
            "answer": synthesis, "llm_calls": llm.calls}


def main() -> None:
    banner("TIER 3 — Multi-Agent RAG (supervisor routes specialists, then synthesizes)")
    for q in [
        "how does remote work relate to the SEC-17 security policy",  # security (+ people_ops)
        "which vendor missed the SLA for order 991",                  # supply_chain
    ]:
        llm = FakeLLM()
        r = multi_agent_rag(q, llm)
        print(f"\nQ: {q}")
        print(f"  routed_to : {r['routed_to']}")
        for p in r["partials"]:
            print(f"    - {p['agent']:<12} used={p['used']}")
        print(f"  synthesis : {r['answer']}")
        print(f"  llm_calls : {r['llm_calls']}")
    print("\nTakeaway: the supervisor sends each query only to the relevant specialists, "
          "keeps their contexts focused, and merges findings — ideal for cross-domain questions.")


if __name__ == "__main__":
    main()
