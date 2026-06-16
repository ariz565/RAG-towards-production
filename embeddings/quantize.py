"""Embedding quantization — the storage/latency lever (with MRL).

Two production methods:
- **Scalar int8**: map float32 → int8 per dimension (calibrated min/max). ~4×
  smaller, tiny accuracy loss. The common default.
- **Binary (1-bit)**: sign(x) → {0,1}, packed into bytes. ~32× smaller; compare
  with Hamming distance. Great for a cheap first-stage that you then rerank with
  full-precision vectors (binary → shortlist → rescore).

Combine with Matryoshka truncation for two compression axes at once
(e.g. 3072-d float32 → 256-d int8 ≈ 48× smaller).
"""

from __future__ import annotations

Vector = list[float]


def storage_bytes(dim: int, dtype: str) -> int:
    """Per-vector storage for a given dtype."""
    return {"float32": 4 * dim, "int8": dim, "binary": (dim + 7) // 8}[dtype]


# ── Scalar int8 ──────────────────────────────────────────────────────

def calibrate_int8(vectors: list[Vector]) -> tuple[list[float], list[float]]:
    """Per-dimension (min, max) calibration over a representative corpus."""
    dim = len(vectors[0])
    mins = [min(v[i] for v in vectors) for i in range(dim)]
    maxs = [max(v[i] for v in vectors) for i in range(dim)]
    return mins, maxs


def quantize_int8(v: Vector, mins: list[float], maxs: list[float]) -> list[int]:
    out = []
    for x, lo, hi in zip(v, mins, maxs):
        span = hi - lo
        scaled = 0.0 if span == 0 else (x - lo) / span      # → [0,1]
        out.append(int(round(scaled * 255)) - 128)          # → [-128,127]
    return out


def dequantize_int8(q: list[int], mins: list[float], maxs: list[float]) -> Vector:
    out = []
    for qi, lo, hi in zip(q, mins, maxs):
        span = hi - lo
        out.append(lo + ((qi + 128) / 255) * span)
    return out


# ── Binary (1-bit) ───────────────────────────────────────────────────

def quantize_binary(v: Vector) -> list[int]:
    """Sign quantization → bit per dimension (1 if x > 0 else 0)."""
    return [1 if x > 0 else 0 for x in v]


def pack_bits(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def hamming(a_bits: list[int], b_bits: list[int]) -> int:
    """Hamming distance (smaller = more similar). Used for binary vectors."""
    return sum(1 for x, y in zip(a_bits, b_bits) if x != y)
