"""Tier 3 — Graph RAG: retrieve connected meaning, not just similar text.

WORKING
    Extract entities + relationships from documents into a knowledge graph (here we
    hardcode the triples a real extractor would produce). At query time, link the
    query to graph nodes and **traverse relationships** to assemble a connected
    subgraph + the evidence chunks behind each edge. The LLM then answers over a
    *structured reasoning scaffold* (entities, paths) plus grounding text.

    Vector RAG asks "which chunks are similar?". Graph RAG asks "which entities,
    relationships, and paths explain this?" — so multi-hop and relationship questions
    that no single chunk answers become tractable, with explicit provenance.

USE CASES
    Relationship / multi-hop questions (supply chain, finance, healthcare, security),
    global "what are the themes?" sensemaking, and anywhere provenance matters.

LIMITATIONS
    Higher index cost (extraction, canonicalization, communities) and new failure modes
    (bad extraction, wrong merges, stale graph). See the deep dive in
    code-n-concepts/99-graph-rag-complete-learning-guide.md.
"""

from __future__ import annotations

from collections import deque

from advanced_rag._harness import FakeLLM, banner, doc

# A tiny knowledge graph extracted from incident/vendor docs (d5, d6).
# Each edge carries the relationship and the evidence text-unit (doc id).
TRIPLES = [
    ("Plant 17", "experienced", "coolant failure", "d5"),
    ("coolant failure", "caused delay for", "order 991", "d5"),
    ("Acme Logistics", "handled", "order 991", "d5"),
    ("Acme Logistics", "missed SLA for", "order 991", "d6"),
]

# Build an undirected adjacency map: node -> [(neighbor, relation, evidence)].
_ADJ: dict[str, list[tuple[str, str, str]]] = {}
for s, rel, o, ev in TRIPLES:
    _ADJ.setdefault(s, []).append((o, rel, ev))
    _ADJ.setdefault(o, []).append((s, f"(inv) {rel}", ev))


def shortest_path(a: str, b: str) -> list[tuple[str, str, str]] | None:
    """BFS for the shortest relationship path from entity a to entity b."""
    if a not in _ADJ or b not in _ADJ:
        return None
    queue: deque[tuple[str, list]] = deque([(a, [])])
    seen = {a}
    while queue:
        node, path = queue.popleft()
        if node == b:
            return path
        for neighbor, rel, ev in _ADJ[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + [(node, rel, neighbor, ev)]))
    return None


def graph_rag(entity_a: str, entity_b: str, question: str, llm: FakeLLM) -> dict:
    path = shortest_path(entity_a, entity_b)
    if not path:
        return {"path": None, "answer": f"No graph path found between {entity_a} and {entity_b}."}

    provenance = [f"{s} -[{rel}]-> {o}" for (s, rel, o, _ev) in path]
    evidence_ids = sorted({ev for (_s, _rel, _o, ev) in path})
    contexts = [doc(e) for e in evidence_ids]
    answer = llm.answer(question, contexts)
    return {"path": provenance, "evidence": evidence_ids, "answer": answer}


def main() -> None:
    banner("TIER 3 — Graph RAG (entity graph + multi-hop traversal with provenance)")

    print("\nKnowledge graph (extracted triples):")
    for s, rel, o, ev in TRIPLES:
        print(f"  {s} -[{rel}]-> {o}   (evidence {ev})")

    q = "How is the vendor Acme Logistics connected to Plant 17?"
    llm = FakeLLM()
    r = graph_rag("Acme Logistics", "Plant 17", q, llm)
    print(f"\nQ: {q}")
    print("  graph path (provenance):")
    for hop in r["path"]:
        print(f"    {hop}")
    print(f"  evidence chunks : {r['evidence']}")
    print(f"  grounded answer : {r['answer']}")
    print("\nTakeaway: no single chunk says 'Acme Logistics is connected to Plant 17' — the "
          "answer is a *path* (Acme → order 991 → coolant failure → Plant 17). Graph traversal "
          "surfaces it with explicit, citable provenance that vector similarity can't.")


if __name__ == "__main__":
    main()
