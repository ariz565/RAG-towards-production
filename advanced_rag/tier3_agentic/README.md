# Tier 3 — Agentic & reasoning RAG

This is where **reasoning, control, and verification** get bolted onto retrieval. The
LLM stops being a one-shot generator and starts *deciding*: is this relevant? is it
supported? do I need more? which expert should handle this? Cost rises (more LLM calls),
so Tier 3 is for the hard tail — and Adaptive RAG decides *when* to pay for it.

| Module | Working | Use case | Watch out for |
|---|---|---|---|
| [`self_rag`](self_rag.py) | Reflection gates: **IsRelevant** (drop distractors) → generate → **IsSupported** (ground every claim, else refuse) → **IsUseful**. | Hallucination-intolerant: legal, medical, finance. | More calls; gate quality = critic quality. |
| [`corrective_rag`](corrective_rag.py) | Grade retrieval → **CORRECT** (use), **AMBIGUOUS** (refine + re-retrieve), **INCORRECT** (external fallback). | Open-domain / coverage gaps — degrade gracefully, don't hallucinate. | Needs a good evaluator + trustworthy fallback. |
| [`adaptive_rag`](adaptive_rag.py) | Classify complexity → **SIMPLE** one-shot vs **COMPLEX** decompose-and-synthesize. | Mixed workloads — don't pay agentic cost for "what's the leave policy?". | Misroute = under-answer; keep the router conservative. |
| [`multi_agent_rag`](multi_agent_rag.py) | Supervisor routes the query to domain **specialists**, fans out, synthesizes. | Cross-domain questions + heterogeneous backends (docs/SQL/APIs). | Orchestration overhead; routing can miss a specialist. |
| [`graph_rag`](graph_rag.py) | Extract entities/relationships → traverse a **knowledge graph** → answer over paths + evidence. | Multi-hop / relationship / "themes" questions; provenance matters. | Index cost; bad extraction/merges/stale graph. |

```bash
python -m advanced_rag.tier3_agentic.self_rag
python -m advanced_rag.tier3_agentic.corrective_rag
python -m advanced_rag.tier3_agentic.adaptive_rag
python -m advanced_rag.tier3_agentic.multi_agent_rag
python -m advanced_rag.tier3_agentic.graph_rag
```

**How this mirrors the Vision app**
- `self_rag` / `corrective_rag` ≙ Vision's grounding/refusal gate + `grade_context` (corrective re-retrieval).
- `adaptive_rag` ≙ the recommended next upgrade for Vision: route by *complexity*, not just intent.
- `graph_rag` ≙ the deep dive in [`code-n-concepts/99-graph-rag-complete-learning-guide.md`](../../../../code-n-concepts/99-graph-rag-complete-learning-guide.md).

**The big idea:** Tier 2 makes *retrieval* better; Tier 3 makes the *system* decide and
verify. The cost is real (4–6× the calls), which is exactly why `adaptive_rag` exists.
