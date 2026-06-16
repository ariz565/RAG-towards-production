# RAG Evaluation Lab

RAGAS-style **generation-side** metrics — a standalone reference (not wired into
the app). Runs **offline** via lexical/embedding fallbacks; pass an LLM for the
true judged metrics. (Retrieval metrics — recall@k/nDCG/MRR — live in the
`retrieval/` lab.)

> You can't improve what you don't measure. These four metrics + the retrieval
> metrics are the dashboard that tells you *which* part of the pipeline to fix.

## The metrics

| Metric | Question it answers | Needs | Method (LLM) | Offline fallback |
|---|---|---|---|---|
| **faithfulness** | Is every claim grounded in the context? (hallucination) | answer + contexts | decompose answer → NLI each claim vs context | sentence lexical coverage |
| **answer_relevancy** | Does the answer address the question? | question + answer | generate questions from answer → cosine to original | question↔answer cosine |
| **context_precision** | Are relevant contexts ranked high? | question (+gt) + contexts | LLM judges each context → rank-weighted AP | lexical overlap vs reference |
| **context_recall** | Did retrieval surface all needed info? | ground_truth + contexts | decompose gt → attributable to context? | sentence lexical coverage |

Diagnostic map: **low faithfulness** → generation/prompt or weak context; **low
context_recall** → retrieval/chunking miss; **low context_precision** → ranking
problem (add a reranker); **low answer_relevancy** → prompt/answer drift.

## Quick start

```bash
cd agent-backend/admission-guide

python -m evaluation.cli                 # built-in faithful-vs-hallucinated demo (offline)
python -m evaluation.cli --use-llm       # true RAGAS-style (needs OPENAI_API_KEY)
python -m evaluation.cli --question "..." --answer "..." --context "..." --ground-truth "..."
```

```python
from evaluation import Sample, default_suite, evaluate

samples = [Sample(question="...", answer="...", contexts=["..."], ground_truth="...")]
report = evaluate(samples, default_suite())     # inject llm_fn= for judged metrics
print(report["aggregate"])
```

## Production targets (2026)
faithfulness ≥ 0.9 (gate ships on this) · answer_relevancy ≥ 0.8 ·
context_precision ≥ 0.7 · context_recall ≥ 0.8. Run offline checks in CI; sample
production traffic with LLM-judged metrics.

## References
RAGAS (faithfulness / answer relevancy / context precision & recall);
TruLens "RAG triad"; DeepEval. See the main app's `evals/` for a CI gate that
applies these thresholds with DeepEval.
