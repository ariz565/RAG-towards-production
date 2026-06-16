# Evaluation Harness (Phase A: "measure, don't assume")

This turns the project's self-rated `confidence` into **measured, gated quality** —
the single highest-leverage upgrade from the 6→10 vision.

## What's here

| File | Purpose |
|---|---|
| `golden_set.jsonl` | Curated, corpus-grounded Q&As + out-of-scope negatives. **The asset.** |
| `harness.py` | Loads indexes + domain profile and runs the real pipeline over a query. |
| `run_evals.py` | Fast, judge-free pass: page recall/precision, out-of-scope accuracy, latency, tokens. Saves predictions. |
| `judge.py` | DeepEval judge that reuses the project's multi-provider LLM service. |
| `test_rag_quality.py` | **CI gate** — DeepEval LLM-judge metrics vs 2026 thresholds; blocks regressions. |
| `ragas_offline.py` | Optional offline Ragas run for tuning chunking/embeddings/reranking. |

## Metrics & thresholds (2026 production defaults)

Configured in `app/config.py` (override via `.env`):

| Metric | Threshold | Meaning |
|---|---|---|
| Faithfulness | ≥ 0.75 | Every claim grounded in retrieved context (anti-hallucination). |
| Answer Relevancy | ≥ 0.80 | Answer actually addresses the question. |
| Contextual Precision | ≥ 0.70 | Retrieved chunks are relevant. |
| Contextual Recall | ≥ 0.80 | Retrieval found what was needed. |

Plus deterministic gates: out-of-scope questions must be **refused**, in-scope
questions must **not** be wrongly refused, and retrieval must hit expected pages.

## Run it

```bash
cd agent-backend/admission-guide
pip install -r requirements.txt
pip install -r evals/requirements-eval.txt

# 1) Fast deterministic pass (no judge) — works with any configured model
python -m evals.run_evals --strategy hybrid
python -m evals.run_evals --strategy pageindex   # compare strategies (benchmark seed)

# 2) Full LLM-judge CI gate
export RUN_EVALS=1
export EVAL_JUDGE_PROVIDER=openai     # a hosted judge is strongly recommended
pytest evals/ -v
```

A small local model (Ollama) works for the deterministic pass and as a judge, but
hosted judges (`EVAL_JUDGE_PROVIDER=openai|azure_openai`) give far more reliable
scores. Set `EVAL_JUDGE_MODEL` to override the judge model name.

## Domain-agnostic note

The golden set targets the **currently indexed document** (a Business Layer
architecture doc). The pipeline is now document-agnostic: the guardrail, answer,
and redirect prompts adapt to a derived **domain profile**
(`data/index/<doc>_profile.json`, regenerate with `python -m app.cli profile`).
To evaluate a different corpus, index it, regenerate the profile, and replace
`golden_set.jsonl` with Q&As for that document.

## Roadmap

This seed has 15 cases (13 in-scope + 2 negatives). Per the vision's Definition of
Done, grow it to **≥50** before claiming a published accuracy number. The
`run_evals.py` per-strategy output is the seed of the **PageIndex-vs-Hybrid public
benchmark** (Phase E).
