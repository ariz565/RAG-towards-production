"""Sparse embeddings — learned (SPLADE) and lexical.

Sparse vectors are term→weight maps over a vocabulary (mostly zeros). They give
**exact term matching** + interpretability, and learned sparse (SPLADE) often
**generalizes to new domains better than dense**. The cost: heavier compute and
language dependence. In practice sparse is fused with dense (hybrid + RRF).

- ``lexical_sparse`` — dependency-free log-TF weights (transparent stand-in).
- ``SpladeEmbedder`` — true learned sparse via an MLM head (lazy transformers),
  falling back to lexical if the model isn't available.
"""

from __future__ import annotations

import math
import re
from collections import Counter

SparseVector = dict[str, float]


def lexical_sparse(text: str) -> SparseVector:
    """Log-TF term weights: 1 + log(count). A transparent sparse baseline."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {term: 1.0 + math.log(c) for term, c in Counter(tokens).items()}


def sparse_dot(a: SparseVector, b: SparseVector) -> float:
    """Dot product over the shared vocabulary (iterate the smaller map)."""
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return sum(w * big.get(term, 0.0) for term, w in small.items())


class SpladeEmbedder:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", top_k: int = 256):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
        self._tokenizer = None

    def embed(self, text: str) -> SparseVector:
        try:
            return self._splade(text)
        except Exception:
            return lexical_sparse(text)  # graceful offline fallback

    def _splade(self, text: str) -> SparseVector:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self._model(**inputs).logits          # [1, T, V]
        # SPLADE pooling: max over tokens of log(1 + relu(logits))
        weights = torch.max(torch.log1p(torch.relu(logits)).squeeze(0), dim=0).values
        top = torch.topk(weights, k=min(self.top_k, weights.shape[-1]))
        vocab = self._tokenizer.convert_ids_to_tokens(top.indices.tolist())
        return {tok: float(w) for tok, w in zip(vocab, top.values.tolist()) if w > 0}
