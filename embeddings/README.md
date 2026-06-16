# Embeddings Lab

Production-grade, interchangeable embedding techniques — a **standalone
reference** (not wired into the app). Pure-Python core, so it runs **offline**
via a dependency-free hash embedder; real backends (sentence-transformers,
OpenAI, Ollama) load lazily.

> Embeddings are the other half of retrieval quality. The model matters, but so
> do the *knobs*: asymmetric prefixes, pooling, normalization, dimension
> (Matryoshka), and quantization — each a recall/latency/storage trade-off.

## 1. Choosing a model (MTEB 2026 snapshot)

| Model | Why | Notes |
|---|---|---|
| **Gemini Embedding** | Tops English MTEB; multimodal (text/image/video/audio/PDF), native MRL | API, 3072-d, truncatable |
| **Voyage-3-large** | Leads retrieval-focused metrics | API; strong for RAG |
| **OpenAI text-embedding-3-small/large** | Best price/perf; MRL (`dimensions`) | $0.02/M (small) |
| **Cohere Embed v4** | Native text **+ image** in one space | API; multimodal |
| **BGE-M3** | OSS workhorse: dense **+ sparse + multi-vector** in one model, 100+ langs, MIT | self-host default |
| **Qwen3-Embedding-8B** | SOTA OSS accuracy (Q4 quantized) | heavier |
| **E5 / Nomic-v2 / GTE** | Strong small/medium OSS; instruction-aware | use the right prefix! |

**Caveat:** MTEB rank ≠ your RAG performance. Always re-rank candidates on **your**
eval set (recall@k, nDCG) before committing.

## 2. The knobs (this lab implements all of them)

| Knob | What / why | Code |
|---|---|---|
| **Asymmetric prefixes** | Queries vs docs into different regions of the space → better discrimination. `query:`/`passage:` (E5), `search_query:`/`search_document:` (Nomic), instruction (BGE). **Wrong/missing prefix silently hurts recall.** | [prefixes.py](prefixes.py) |
| **Pooling** | Token vectors → one vector. `mean` (safe default), `cls` (BERT), `last` (LLM-based), `max`. | [base.py](base.py) |
| **L2 normalization** | Makes cosine == dot; stabilizes ANN. | [base.py](base.py) |
| **Matryoshka (MRL)** | MRL-trained models front-load info into early dims → truncate 3072→256 for big savings, minimal recall loss. **Only on MRL-trained models.** | [base.py](base.py) `truncate_mrl` |
| **Quantization** | `int8` (≈4× smaller, ~0 loss — the default) and `binary` (≈32× smaller, compare with Hamming, use as a cheap first stage → rescore). | [quantize.py](quantize.py) |
| **Caching** | Embeddings are pure fns of (model, prefix, text) — cache them. | [cache.py](cache.py) |

## 3. Representation families

| Family | Shape | Score | Best for | Trade-off |
|---|---|---|---|---|
| **Dense** ([dense.py](dense.py)) | 1 vector | cosine/dot | general semantic retrieval | misses exact terms |
| **Sparse / SPLADE** ([sparse.py](sparse.py)) | term→weight | sparse dot | exact terms; cross-domain generalization; interpretable | heavier; language-dependent |
| **Multi-vector / ColBERT** ([multivector.py](multivector.py)) | 1 vector **per token** | **MaxSim** | long/nuanced docs; high precision | storage + compute per token |

Production reality: **hybrid** — dense + sparse fused with RRF, optional ColBERT
rerank on a narrowed candidate set.

## Quick start

```bash
cd agent-backend/admission-guide

python -m embeddings.cli                              # offline (hash provider)
python -m embeddings.cli --prefix e5 --dim 128 --quantize int8
python -m embeddings.cli --quantize binary
python -m embeddings.cli --provider sentence_transformers --model BAAI/bge-small-en-v1.5

python -m embeddings.benchmark                        # MRL + quantization vs storage
```

```python
from embeddings import get_embedder, EmbedderConfig, cosine

emb = get_embedder("hash", config=EmbedderConfig(prefix_preset="e5", dimensions=128))
qv = emb.embed_query("minimum GPA for CS")
dv = emb.embed_documents(["The CS program requires a 3.2 GPA."])[0]
print(cosine(qv, dv))
```

## Decision cheat-sheet
- Default → a strong dense model (BGE-M3 self-host / OpenAI-3 / Voyage) + **right prefix** + **L2 normalize**.
- Storage/latency pressure → **int8** first (≈free); add **MRL** truncation if the model is MRL-trained; **binary** only as a first-stage shortlist → rescore.
- Need exact-term / out-of-domain robustness → add **sparse (SPLADE)**, fuse with RRF.
- Long, nuanced docs where precision is critical → **ColBERT** rerank on top-k.
- Always: pick the cheapest representation that **holds recall on your evals**.

## References
- MTEB 2026 leaderboard & model guides (Milvus, Modal, BentoML).
- Matryoshka Representation Learning — HuggingFace blog; Voyage flexible dimensions.
- Embedding quantization (int8/binary) — HuggingFace; Vespa MRL+binary.
- E5 / Nomic / BGE instruction prefixes; SPLADE (Naver); ColBERT/ColBERTv2 (Stanford), Weaviate late-interaction overview.
