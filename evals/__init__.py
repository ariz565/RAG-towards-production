"""Evaluation harness for Vision/UniDocs.

Phase A of the 6→10 vision: measure, don't assume.

- ``golden_set.jsonl``    — curated, corpus-grounded Q&As (+ out-of-scope negatives)
- ``judge.py``            — DeepEval judge wrapping the project's multi-provider LLM
- ``harness.py``          — load indexes/profile + run the pipeline over a query
- ``run_evals.py``        — run the golden set, compute deterministic metrics, save predictions
- ``test_rag_quality.py`` — DeepEval LLM-judge metrics as a pytest CI gate
- ``ragas_offline.py``    — optional offline Ragas run for tuning
"""
