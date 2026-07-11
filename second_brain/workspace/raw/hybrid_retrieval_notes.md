Hybrid retrieval combines BM25 (sparse/lexical) and a dense bi-encoder, then fuses
the two ranked lists with Reciprocal Rank Fusion (RRF). BM25 catches exact terms,
codes, and jargon that a dense embedder often misses; dense catches paraphrases
and synonyms that BM25 misses. RRF fuses by rank position, not raw score, because
BM25 scores are unbounded while cosine similarity is bounded to [-1, 1] -- naive
score averaging would let BM25 dominate. Reference: Cormack et al., SIGIR 2009.
See retrieval/hybrid.py and retrieval/fusion.py in this repo for a working,
offline implementation of exactly this.
