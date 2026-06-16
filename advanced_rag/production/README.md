# Production — the non-negotiable wrappers

Techniques make a *demo*; these make it a *product*. None are optional in a real
deployment.

| Module | Working | Use case |
|---|---|---|
| [`evaluation`](evaluation.py) | Score **retrieval** (recall@k / nDCG / MRR vs labeled relevants) and **generation** (faithfulness, completeness) *separately* on a golden set. | The CI gate — know which half (retriever vs generator) a change improved or regressed. |
| [`grounding`](grounding.py) | Decompose the answer into claims, verify each against context, **flag** unsupported, and **refuse** when the supported ratio is too low. | Anti-hallucination spine for user-facing answers; pairs with Self-RAG + risk tiers. |
| [`cost_control`](cost_control.py) | **Iteration cap** (~3 rounds) + **context compression** (keep only query-relevant sentences within a budget) + stream/async for long runs. | Bound latency/cost of agentic loops — complex queries can cost 20–40× a simple one. |

```bash
python -m advanced_rag.production.evaluation
python -m advanced_rag.production.grounding
python -m advanced_rag.production.cost_control
```

**Why these are the point**
- **Measure separately or you fly blind.** A low final score doesn't tell you whether to
  fix retrieval or generation. Two dials, two diagnoses.
- **Grounding is what earns trust.** Citations + refuse-when-unsupported is the difference
  between "confidently wrong" and "honestly limited."
- **Cost control is what keeps it shippable.** Without caps + compression, agentic RAG's
  worst-case latency and bill are unbounded.

These mirror the Vision app exactly: `evals/` (RAGAS/DeepEval gates), the grounding/refusal
node, and async jobs + context budgets.
