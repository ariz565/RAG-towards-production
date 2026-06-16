"""Factories — pick retriever / transform / reranker by flag."""

from __future__ import annotations

from enum import Enum

from retrieval.base import Retriever
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.query_transform import (
    Decomposition,
    HyDE,
    MultiQuery,
    QueryTransform,
    StepBack,
)
from retrieval.rerank import ColBERTReranker, CrossEncoderReranker, LexicalReranker, Reranker


class RetrieverStrategy(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


_RETRIEVERS: dict[RetrieverStrategy, type[Retriever]] = {
    RetrieverStrategy.BM25: BM25Retriever,
    RetrieverStrategy.DENSE: DenseRetriever,
    RetrieverStrategy.HYBRID: HybridRetriever,
}

_TRANSFORMS = {
    "none": None,
    "multi_query": MultiQuery,
    "hyde": HyDE,
    "decomposition": Decomposition,
    "step_back": StepBack,
}

_RERANKERS = {
    "none": None,
    "lexical": LexicalReranker,
    "cross_encoder": CrossEncoderReranker,
    "colbert": ColBERTReranker,
}


def get_retriever(strategy: str | RetrieverStrategy, **kwargs) -> Retriever:
    return _RETRIEVERS[RetrieverStrategy(strategy)](**kwargs)


def get_transform(name: str, **kwargs) -> QueryTransform | None:
    cls = _TRANSFORMS.get(name)
    if cls is None:
        return None
    return cls(**kwargs)


def get_reranker(name: str, **kwargs) -> Reranker | None:
    cls = _RERANKERS.get(name)
    if cls is None:
        return None
    return cls(**kwargs)
