"""Shared offline harness for the Advanced RAG lab.

Every module in this lab is **runnable with no API key and no heavy deps** — it uses
this harness's toy corpus, deterministic "embeddings", BM25, a lexical reranker, and
a small rule-based ``FakeLLM``. The point is to *learn the algorithm*, not to ship a
model. Swap any piece for a real embedder / cross-encoder / LLM and the technique is
unchanged.

Design notes:
- ``embed()`` is a bag-of-tokens hash vector **with light synonym expansion**, so dense
  search can match paraphrases that BM25 (raw terms) misses — which is exactly what makes
  hybrid search worth it in the demos.
- ``FakeLLM`` is deterministic and rule-based: it "answers", "grades", "rewrites",
  "decomposes", and "classifies" using token overlap. Plausible enough to show control
  flow (Self-RAG / CRAG / Adaptive) without a real model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# ── Toy corpus ────────────────────────────────────────────────────────
# A small "ACME employee handbook + ops notes". Crafted so that:
# - factual lookup works (naive),
# - exact codes need BM25 (e.g. "SEC-17"),
# - paraphrases need dense (e.g. "time off" ~ "leave"),
# - relationships enable multi-hop / graph demos (Plant 17 → incident → order → vendor).
CORPUS: list[dict] = [
    {"id": "d1", "title": "Leave Policy",
     "text": "Full-time employees accrue 20 days of annual paid leave. Vacation time off must be "
             "requested two weeks in advance. Unused leave rolls over up to 5 days."},
    {"id": "d2", "title": "Parental Leave",
     "text": "New parents are entitled to 16 weeks of paid parental leave. Parental leave can be "
             "taken any time within the first year after birth or adoption."},
    {"id": "d3", "title": "Expense Reimbursement",
     "text": "Employees can claim reimbursement for travel expenses. Submit receipts within 30 days. "
             "Meals are reimbursed up to 50 dollars per day."},
    {"id": "d4", "title": "Security Policy SEC-17",
     "text": "Policy SEC-17 requires multi-factor authentication for all remote access. Exceptions to "
             "SEC-17 must be approved by the security team in writing."},
    {"id": "d5", "title": "Incident Report 2026-04-16",
     "text": "Plant 17 experienced a coolant failure on April 16. The coolant failure caused a delay "
             "for order 991, which was handled by the vendor Acme Logistics."},
    {"id": "d6", "title": "Vendor SLA Notes",
     "text": "Acme Logistics missed the SLA for order 991. Repeated SLA misses by Acme Logistics are "
             "tracked by the supply chain team for the west region."},
    {"id": "d7", "title": "Remote Work",
     "text": "Employees may work remotely up to three days per week. Remote access requires VPN and "
             "complies with the security policy."},
    {"id": "d8", "title": "Holidays",
     "text": "The company observes 11 public holidays per year. Holidays falling on a weekend are "
             "observed on the following Monday."},
]

# Light synonym map: only dense embedding expands with these, so dense can bridge
# vocabulary the user phrases differently from the document.
SYNONYMS: dict[str, list[str]] = {
    "vacation": ["leave", "timeoff", "pto"],
    "timeoff": ["leave", "vacation"],
    "pto": ["leave", "vacation"],
    "reimburse": ["expense", "claim", "refund"],
    "reimbursement": ["expense", "claim"],
    "maternity": ["parental", "parent"],
    "paternity": ["parental", "parent"],
    "mfa": ["authentication", "multifactor"],
    "wfh": ["remote", "work"],
}

_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "is", "are",
         "be", "by", "up", "per", "can", "must", "all", "any", "within", "this", "that"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]


def _hash(token: str) -> int:
    # Stable, deterministic bucket (not Python's salted hash()).
    h = 2166136261
    for ch in token:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h % _DIM


def embed(text: str, *, expand: bool = True) -> list[float]:
    """Deterministic bag-of-tokens hash vector with optional synonym expansion."""
    vec = [0.0] * _DIM
    for tok in tokenize(text):
        vec[_hash(tok)] += 1.0
        if expand:
            for syn in SYNONYMS.get(tok, []):
                vec[_hash(syn)] += 0.7   # weaker weight for expanded terms
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── BM25 (sparse / lexical) ───────────────────────────────────────────
class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d["text"] + " " + d["title"]) for d in docs]
        self.doc_len = [len(c) for c in self.corpus]
        self.avgdl = (sum(self.doc_len) / len(self.corpus)) if self.corpus else 0.0
        self.df: Counter = Counter()
        for c in self.corpus:
            for t in set(c):
                self.df[t] += 1
        self.N = len(self.corpus)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores: list[tuple[str, float]] = []
        for i, doc in enumerate(self.docs):
            tf = Counter(self.corpus[i])
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self._idf(term)
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s += idf * num / den
            if s > 0:
                scores.append((doc["id"], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ── Dense (semantic) ──────────────────────────────────────────────────
_DOC_VECS = {d["id"]: embed(d["text"] + " " + d["title"]) for d in CORPUS}
_BY_ID = {d["id"]: d for d in CORPUS}
_BM25 = BM25(CORPUS)


def dense_search(query: str, k: int = 5) -> list[tuple[str, float]]:
    qv = embed(query)
    scored = [(d["id"], cosine(qv, _DOC_VECS[d["id"]])) for d in CORPUS]
    scored = [(i, s) for i, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def bm25_search(query: str, k: int = 5) -> list[tuple[str, float]]:
    return _BM25.search(query, k)


def doc(doc_id: str) -> dict:
    return _BY_ID[doc_id]


# ── Fusion + rerank ───────────────────────────────────────────────────
def rrf(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked id-lists."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def lexical_rerank(query: str, doc_ids: list[str], k: int = 5) -> list[tuple[str, float]]:
    """Cross-encoder stand-in: score query-doc *together* by overlap + coverage.

    A real cross-encoder reads (query, passage) jointly; this approximates that joint
    relevance with token overlap weighted by how much of the query is covered.
    """
    q = Counter(tokenize(query))
    out: list[tuple[str, float]] = []
    for doc_id in doc_ids:
        d = Counter(tokenize(_BY_ID[doc_id]["text"]))
        overlap = sum(min(q[t], d[t]) for t in q)
        coverage = sum(1 for t in q if t in d) / (len(q) or 1)
        out.append((doc_id, overlap * (0.5 + coverage)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:k]


# ── Retrieval-quality metrics ─────────────────────────────────────────
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum((1.0 / math.log2(i + 2)) for i, r in enumerate(retrieved[:k]) if r in relevant)
    ideal = sum((1.0 / math.log2(i + 2)) for i in range(min(len(relevant), k)))
    return (dcg / ideal) if ideal else 0.0


# ── Deterministic rule-based "LLM" ────────────────────────────────────
@dataclass
class FakeLLM:
    """A tiny, deterministic stand-in for an LLM.

    Methods mirror the calls advanced/agentic RAG makes on a real model, so module
    code reads like production code — only the model is fake. Swap in a real client
    (the Vision app's ``app.services.llm.chat``) and the logic is identical.
    """

    name: str = "fake-llm-v1"
    calls: int = field(default=0)

    def _bump(self) -> None:
        self.calls += 1

    def answer(self, question: str, contexts: list[dict]) -> str:
        """Extractive, cited answer: pick sentences overlapping the question."""
        self._bump()
        q = set(tokenize(question))
        picked: list[str] = []
        for c in contexts:
            for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
                if q & set(tokenize(sent)):
                    picked.append(f"{sent.strip()} [{c['id']}]")
        if not picked:
            return "I don't have enough grounded information to answer that."
        # De-dup while preserving order.
        seen, out = set(), []
        for p in picked:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return " ".join(out[:3])

    def grade_relevance(self, question: str, context: dict) -> float:
        """0-1 relevance of one context to the question (CRAG/Self-RAG gate)."""
        self._bump()
        q = set(tokenize(question))
        d = set(tokenize(context["text"]))
        if not q:
            return 0.0
        return len(q & d) / len(q)

    def supported(self, claim: str, contexts: list[dict]) -> bool:
        """Is the claim grounded in the contexts? (Self-RAG IsSupported gate)."""
        self._bump()
        joined = set(tokenize(" ".join(c["text"] for c in contexts)))
        c = set(tokenize(claim))
        return bool(c) and len(c & joined) / len(c) >= 0.6

    def rewrite(self, question: str) -> str:
        """Conservative normalization/rewrite (expand a known abbreviation, tidy)."""
        self._bump()
        q = question.strip().rstrip("?")
        expand = {"pto": "paid time off (leave)", "wfh": "remote work", "mfa": "multi-factor authentication"}
        toks = [expand.get(t, t) for t in q.split()]
        return " ".join(toks)

    def hyde(self, question: str) -> str:
        """Hypothetical Document Embeddings: draft a plausible answer to embed."""
        self._bump()
        return f"{question.rstrip('?')}. The policy states the relevant details and conditions."

    def multi_query(self, question: str) -> list[str]:
        """Generate paraphrase variants to widen recall."""
        self._bump()
        base = question.rstrip("?")
        return [base, f"What is the rule for {base.lower()}", f"{base} policy details"]

    def decompose(self, question: str) -> list[str]:
        """Split a multi-part / comparison question into sub-questions."""
        self._bump()
        parts = re.split(r"\b(?:and|vs\.?|versus|compare|difference between)\b", question, flags=re.I)
        parts = [p.strip(" ?.,") for p in parts if p.strip(" ?.,")]
        return parts if len(parts) > 1 else [question]

    def step_back(self, question: str) -> str:
        """Ask the more general question first (step-back prompting)."""
        self._bump()
        return f"What are the general rules relevant to: {question.rstrip('?')}?"

    def classify_complexity(self, question: str) -> str:
        """Adaptive-RAG router: 'simple' | 'complex' from cheap signals."""
        self._bump()
        ql = question.lower()
        complex_markers = ("compare", " vs", "versus", "why", "how are", "related",
                           "connected", "multi", "across", "and ")
        if any(m in ql for m in complex_markers) or len(tokenize(question)) > 12:
            return "complex"
        return "simple"


def banner(title: str) -> None:
    # Windows consoles default to cp1252; force UTF-8 so em-dashes etc. print cleanly.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")
