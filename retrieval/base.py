"""Core types + IR metrics for the retrieval lab.

Standalone study module (not wired into the app). Pure-Python; the dense path
reuses the embeddings lab's offline HashEmbedder, so the whole thing runs with
zero external dependencies.

Metrics are the point: you don't *choose* a retriever, you *measure* it.
- recall@k   — did the relevant docs make it into the top-k? (the RAG ceiling)
- precision@k— how clean is the top-k?
- MRR        — how early is the first relevant doc? (user-facing first impression)
- nDCG@k     — position-weighted, graded; correlates best with end-to-end RAG quality
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Doc:
    id: str
    text: str


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float
    rank: int = 0
    source: str = ""


class Retriever(ABC):
    @abstractmethod
    def index(self, docs: list[Doc]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[RetrievedDoc]:
        ...


# ── IR metrics (binary relevance) ────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    return sum(1 for i in top if i in relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved_ids[:k]
    return sum(1 for i in top if i in relevant_ids) / k


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if any(i in relevant_ids for i in retrieved_ids[:k]) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    gains = [1.0 if i in relevant_ids else 0.0 for i in retrieved_ids[:k]]
    dcg = sum(g / math.log2(idx + 2) for idx, g in enumerate(gains))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
