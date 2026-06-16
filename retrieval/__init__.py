"""Retrieval lab — BM25, dense, hybrid (RRF), query transforms, reranking.

Standalone study package (not wired into the app). Runs offline: pure-Python
BM25 + the embeddings lab's HashEmbedder for the dense path. Heavy models
(cross-encoder, real embedders, LLM transforms) load lazily.

Pipeline shape: transform → retrieve → RRF fuse → rerank → top-k.
"""

from retrieval.base import (
    Doc,
    RetrievedDoc,
    Retriever,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.factory import (
    RetrieverStrategy,
    get_reranker,
    get_retriever,
    get_transform,
)
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid import HybridRetriever
from retrieval.pipeline import RetrievalPipeline

__all__ = [
    "Doc",
    "RetrievedDoc",
    "Retriever",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "RetrievalPipeline",
    "RetrieverStrategy",
    "get_retriever",
    "get_transform",
    "get_reranker",
]
