"""Instruction / task prefixes for asymmetric retrieval embeddings.

Many modern models are *instruction-aware*: they expect a task prefix on queries
and documents, and the asymmetry trains queries and documents into systematically
different regions of the space — which improves retrieval discrimination.

Use the preset that matches your model. Using the WRONG prefix (or none, on a
model that expects one) measurably hurts recall — a classic silent bug.
"""

from __future__ import annotations

# kind ∈ {"query", "document"}
PREFIX_PRESETS: dict[str, dict[str, str]] = {
    "none": {"query": "", "document": ""},
    # intfloat/e5-* family
    "e5": {"query": "query: ", "document": "passage: "},
    # nomic-embed-text family
    "nomic": {"query": "search_query: ", "document": "search_document: "},
    # BAAI/bge-* family (instruction on the query side only)
    "bge": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
}


def apply_prefix(text: str, preset: str, kind: str) -> str:
    table = PREFIX_PRESETS.get(preset, PREFIX_PRESETS["none"])
    return f"{table.get(kind, '')}{text}"
