"""Production — Evaluation: measure retrieval and generation SEPARATELY.

WHY SEPARATELY
    A bad RAG answer has two possible causes: the retriever didn't surface the right
    evidence, or the generator mishandled good evidence. If you only score the final
    answer you can't tell which half to fix. So measure two layers:

      RETRIEVAL  — recall@k, nDCG@k, MRR against a labeled set of relevant docs.
      GENERATION — faithfulness (every answer claim supported by the context) and
                   completeness (did it include the gold key facts?).

USE CASES
    The CI gate for any RAG change: did this PR improve retrieval, generation, both, or
    silently regress one? Run on a golden set on every change.

LIMITATIONS
    Needs a labeled golden set; generation metrics here use a deterministic stand-in
    (a real system uses an LLM judge / RAGAS / DeepEval).
"""

from __future__ import annotations

from advanced_rag._harness import (
    FakeLLM,
    banner,
    bm25_search,
    dense_search,
    doc,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rrf,
    tokenize,
)

# Golden set: query -> relevant doc ids + key facts the answer should contain.
GOLDEN = [
    {"q": "how many days of annual leave", "relevant": {"d1"}, "key_facts": ["20", "leave"]},
    {"q": "exceptions to SEC-17", "relevant": {"d4"}, "key_facts": ["security", "team", "writing"]},
    {"q": "how do I request time off", "relevant": {"d1"}, "key_facts": ["two", "weeks", "advance"]},
    {"q": "meal reimbursement on travel", "relevant": {"d3"}, "key_facts": ["50", "receipts"]},
]


def _dense(q, k=5):
    return [d for d, _ in dense_search(q, k=k)]


def _hybrid(q, k=5):
    bm25 = [d for d, _ in bm25_search(q, k=10)]
    dense = [d for d, _ in dense_search(q, k=10)]
    return [d for d, _ in rrf([bm25, dense])][:k]


def eval_retrieval(retriever, k: int = 3) -> dict:
    recalls, ndcgs, mrrs = [], [], []
    for ex in GOLDEN:
        got = retriever(ex["q"])
        recalls.append(recall_at_k(got, ex["relevant"], k))
        ndcgs.append(ndcg_at_k(got, ex["relevant"], k))
        mrrs.append(mrr(got, ex["relevant"]))
    n = len(GOLDEN)
    return {f"recall@{k}": sum(recalls) / n, f"ndcg@{k}": sum(ndcgs) / n, "mrr": sum(mrrs) / n}


def eval_generation(llm: FakeLLM) -> dict:
    faiths, comps = [], []
    for ex in GOLDEN:
        contexts = [doc(d) for d in _hybrid(ex["q"], k=3)]
        answer = llm.answer(ex["q"], contexts)
        sentences = [s for s in answer.split("] ") if s.strip()]
        supported = sum(1 for s in sentences if llm.supported(s, contexts))
        faiths.append(supported / (len(sentences) or 1))
        ans_tokens = set(tokenize(answer))
        hit = sum(1 for f in ex["key_facts"] if f.lower() in ans_tokens)
        comps.append(hit / (len(ex["key_facts"]) or 1))
    n = len(GOLDEN)
    return {"faithfulness": sum(faiths) / n, "completeness": sum(comps) / n}


def main() -> None:
    banner("PRODUCTION — Evaluation (retrieval vs generation, measured separately)")
    dense_m = eval_retrieval(_dense)
    hybrid_m = eval_retrieval(_hybrid)
    print("\nRETRIEVAL (avg over golden set, k=3):")
    print(f"  {'metric':<10} {'dense':>8} {'hybrid':>8}")
    for key in dense_m:
        print(f"  {key:<10} {dense_m[key]:>8.2f} {hybrid_m[key]:>8.2f}")

    gen = eval_generation(FakeLLM())
    print("\nGENERATION (avg over golden set):")
    for key, val in gen.items():
        print(f"  {key:<14} {val:.2f}")

    print("\nTakeaway: separate scores localize the failure. If hybrid recall is high but "
          "faithfulness is low, fix the prompt/generator — not the retriever. This is the CI gate.")


if __name__ == "__main__":
    main()
