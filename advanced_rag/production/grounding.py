"""Production — Citation grounding: attribute every claim, flag/refuse the unsupported.

WORKING
    After generation, decompose the answer into claims and check each against the
    retrieved context:
      • supported claim   → keep, with its citation;
      • unsupported claim → FLAG it;
    then apply a policy: if the supported ratio is below a threshold, **refuse** rather
    than emit a confident-but-ungrounded answer. This is the anti-hallucination spine —
    the same idea as Vision's grounding gate.

USE CASES
    Anything user-facing where a wrong-but-confident claim is costly. Pairs with Self-RAG
    (Tier 3) and risk tiers (refuse-when-unsure for legal/medical/finance).

LIMITATIONS
    Claim decomposition + support checking cost extra calls; the check is only as good as
    the verifier. Citations must point at real evidence, not be hallucinated too.
"""

from __future__ import annotations

import re

from advanced_rag._harness import FakeLLM, banner, doc, rrf, bm25_search, dense_search

ACCEPT, REFUSE = 0.8, 0.5   # supported-ratio thresholds


def _retrieve(question, k=3):
    bm25 = [d for d, _ in bm25_search(question, k=10)]
    dense = [d for d, _ in dense_search(question, k=10)]
    return [doc(d) for d, _ in rrf([bm25, dense])][:k]


def ground(question: str, draft: str, contexts: list[dict], llm: FakeLLM) -> dict:
    """Verify each claim in `draft` against `contexts`; flag unsupported; apply policy."""
    claims = [c.strip() for c in re.split(r"(?<=[.!?])\s+", draft) if c.strip()]
    checked = [{"claim": c, "supported": llm.supported(c, contexts)} for c in claims]
    supported = [c for c in checked if c["supported"]]
    flagged = [c["claim"] for c in checked if not c["supported"]]
    ratio = len(supported) / (len(checked) or 1)

    if ratio >= ACCEPT:
        decision = "accept"
    elif ratio < REFUSE:
        decision = "refuse"
    else:
        decision = "accept_with_caveat"

    return {"ratio": round(ratio, 2), "decision": decision, "flagged": flagged, "checked": checked}


def main() -> None:
    banner("PRODUCTION — Citation grounding (flag unsupported claims, refuse when ungrounded)")
    llm = FakeLLM()

    # Case 1: a faithful, extractive draft → all claims grounded → accept.
    q1 = "what are the SEC-17 exceptions"
    ctx1 = _retrieve(q1)
    draft1 = llm.answer(q1, ctx1)
    r1 = ground(q1, draft1, ctx1, llm)
    print(f"\nQ: {q1}")
    print(f"  draft   : {draft1}")
    print(f"  grounded ratio={r1['ratio']} -> {r1['decision']}; flagged={r1['flagged']}")

    # Case 2: a draft with a FABRICATED claim injected → must be flagged / refused.
    q2 = "what are the SEC-17 exceptions"
    ctx2 = _retrieve(q2)
    draft2 = ("Exceptions to SEC-17 must be approved by the security team in writing. "
              "SEC-17 also grants unlimited cryptocurrency trading bonuses to all staff. "
              "Furthermore SEC-17 abolishes all passwords company-wide.")  # two hallucinations
    r2 = ground(q2, draft2, ctx2, llm)
    print(f"\nQ: {q2}  (draft contains a fabricated claim)")
    print(f"  draft   : {draft2}")
    print(f"  grounded ratio={r2['ratio']} -> {r2['decision']}")
    print(f"  FLAGGED unsupported: {r2['flagged']}")

    print("\nTakeaway: grounding catches the fabricated claim and drops the supported ratio, "
          "triggering caveat/refusal — a confident hallucination never reaches the user.")


if __name__ == "__main__":
    main()
