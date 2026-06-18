"""Ingestion Service — PDF → Chunks with metadata.

Full pipeline:
1. Parse PDF with PyMuPDF (text) + pdfplumber (tables)
2. Chunk the whole document with a paragraph-aware recursive splitter
   (tiktoken-sized, cross-page, with overlap and page-span tracking)
3. Enrich with metadata (page span, section titles)
4. Optionally prepend per-chunk Contextual Retrieval context before embedding

Usage:
    from app.services.ingestion import ingest_pdf

    chunks = await ingest_pdf("data/pdf/admission-guide.pdf")
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from app import prompts
from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Chunk:
    """A single chunk of text from the PDF."""
    id: str                          # Unique chunk ID
    text: str                        # The actual text content (used for display + answering)
    page_numbers: list[int]          # Pages this chunk spans (1-based)
    section_title: str = ""          # Extracted section heading
    chunk_index: int = 0             # Sequential index in the document
    token_count: int = 0             # Approximate token count
    context: str = ""                # Contextual Retrieval: situating context (Anthropic)
    metadata: dict = field(default_factory=dict)  # Extra metadata (entities, etc.)

    @property
    def embed_text(self) -> str:
        """Text used for embedding/BM25. Prepends contextual description if present."""
        if self.context:
            return f"{self.context}\n\n{self.text}"
        return self.text

    @property
    def content_hash(self) -> str:
        """Stable hash of the chunk text — used for incremental re-indexing diffs."""
        import hashlib
        return hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════════════


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Extract text from each page of the PDF.

    Uses PyMuPDF for reliable text extraction.
    Also tries pdfplumber for table extraction when available.

    Returns:
        List of {page_number: int, text: str, tables: list[str]}
    """
    return list(extract_pages_stream(pdf_path))


def extract_pages_stream(pdf_path: str | Path) -> Generator[dict, None, None]:
    """Stream page extraction from the PDF without buffering the full document."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    pdfplumber_context = None
    pdfplumber_doc = None

    try:
        import pdfplumber
        pdfplumber_doc = pdfplumber.open(str(pdf_path))
    except ImportError:
        logger.info("pdfplumber not available — skipping table extraction")
    except Exception as e:
        logger.warning(f"Table extraction initialization failed: {e}")
        pdfplumber_doc = None

    try:
        page_count = len(doc)
        logger.info(f"Streaming extraction: {page_count} pages from {pdf_path.name}")

        for i in range(page_count):
            page = doc[i]
            text = page.get_text().strip()
            tables = []
            if pdfplumber_doc and i < len(pdfplumber_doc.pages):
                try:
                    raw_tables = pdfplumber_doc.pages[i].extract_tables()
                    if raw_tables:
                        for table in raw_tables:
                            rows = [" | ".join(str(cell).strip() if cell else "" for cell in row) for row in table]
                            tables.append("\n".join(rows))
                except Exception as e:
                    logger.warning(f"Table extraction failed on page {i+1}: {e}")

            yield {
                "page_number": i + 1,
                "text": text,
                "tables": tables,
            }
    finally:
        try:
            doc.close()
        except Exception:
            pass
        if pdfplumber_doc:
            try:
                pdfplumber_doc.close()
            except Exception:
                pass


def _page_blocks(pages: list[dict]) -> list[tuple[int, str]]:
    """Flatten pages (with tables appended) into (page_number, paragraph) blocks."""
    blocks: list[tuple[int, str]] = []
    for page_data in pages:
        page_num = page_data["page_number"]
        text = page_data["text"]
        if page_data.get("tables"):
            text += "\n\n--- Tables ---\n" + "\n\n".join(page_data["tables"])
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                blocks.append((page_num, para))
    return blocks


def _page_blocks_stream(pages_iterable) -> Generator[tuple[int, str], None, None]:
    """Flatten a page stream into paragraph blocks without buffering the full document."""
    for page_data in pages_iterable:
        page_num = page_data["page_number"]
        text = page_data["text"]
        if page_data.get("tables"):
            text += "\n\n--- Tables ---\n" + "\n\n".join(page_data["tables"])
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                yield page_num, para


def chunk_document_stream(
    pages_iterable,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """Yield chunks from a streaming page iterable without buffering the whole document."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    cur_text = ""
    cur_pages: list[int] = []
    cur_tokens = 0

    def flush(next_page: int | None):
        nonlocal cur_text, cur_pages, cur_tokens
        if not cur_text.strip():
            return None
        chunk = {
            "text": cur_text.strip(),
            "page_numbers": sorted(set(cur_pages)),
            "token_count": cur_tokens,
        }
        overlap = _overlap_tail(cur_text, chunk_overlap)
        carry = cur_pages[-1] if cur_pages else next_page
        cur_text = overlap
        cur_pages = [p for p in (carry,) if p is not None]
        cur_tokens = _count_tokens(overlap)
        return chunk

    def add(piece: str, page_num: int, sep: str) -> None:
        nonlocal cur_text, cur_pages, cur_tokens
        cur_text += (sep + piece) if cur_text else piece
        cur_pages.append(page_num)
        cur_tokens += _count_tokens(piece)

    for page_num, para in _page_blocks_stream(pages_iterable):
        para_tokens = _count_tokens(para)
        if para_tokens > chunk_size:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                sent = sent.strip()
                if not sent:
                    continue
                if cur_tokens + _count_tokens(sent) > chunk_size and cur_text:
                    chunk = flush(page_num)
                    if chunk:
                        yield chunk
                add(sent, page_num, " ")
        elif cur_tokens + para_tokens > chunk_size and cur_text:
            chunk = flush(page_num)
            if chunk:
                yield chunk
            add(para, page_num, "\n\n")
        else:
            add(para, page_num, "\n\n")

    final = flush(None)
    if final:
        yield final


def _document_sample(pdf_path: str | Path, char_limit: int) -> str:
    """Return the first `char_limit` chars from the document text in streaming fashion."""
    result = []
    total = 0
    for page_data in extract_pages_stream(pdf_path):
        text = page_data["text"].strip()
        if text:
            remaining = char_limit - total
            snippet = text[:remaining]
            result.append(snippet)
            total += len(snippet)
            if total >= char_limit:
                break
    return "\n\n".join(result)


def _jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    f.write("[")
    first = True
    try:
        while True:
            obj = yield
            if obj is None:
                break
            if not first:
                f.write(",\n")
            else:
                first = False
            json.dump(obj, f, indent=2, ensure_ascii=False)
    finally:
        f.write("\n]")
        f.close()


async def ingest_pdf_streaming(
    pdf_path: str | Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    save_chunks: bool = False,
):
    """Stream PDF ingestion as chunks without buffering the full document."""
    from app.services.llm import chat

    pdf_path = Path(pdf_path)
    logger.info(f"Starting streaming ingestion: {pdf_path.name}")

    if save_chunks:
        chunks_file = settings.chunks_path / f"{pdf_path.stem}_chunks.json"
        writer = _jsonl_writer(chunks_file)
        next(writer)
    else:
        writer = None

    doc_sample = None
    if settings.contextual_retrieval_enabled:
        doc_sample = _document_sample(pdf_path, settings.contextual_doc_sample_chars)

    chunk_index = 0
    for raw in _stream_raw_chunks(pdf_path, chunk_size, chunk_overlap):
        section_title = detect_section_title(raw["text"])
        pages_span = raw["page_numbers"]
        chunk = Chunk(
            id=f"{pdf_path.stem}_c{chunk_index}",
            text=raw["text"],
            page_numbers=pages_span,
            section_title=section_title,
            chunk_index=chunk_index,
            token_count=raw["token_count"],
            metadata={
                "source": pdf_path.name,
                "pages": pages_span,
            },
        )

        if doc_sample and settings.contextual_retrieval_enabled:
            try:
                prompt = prompts.contextualize_chunk(doc_sample, chunk.text)
                resp = await chat(prompt, temperature=0)
                chunk.context = (resp.content or "").strip()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Contextualization failed for {chunk.id}: {e}")

        if writer is not None:
            writer.send(chunk.to_dict())

        yield chunk
        chunk_index += 1

    if writer is not None:
        writer.send(None)


def _stream_raw_chunks(
    pdf_path: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    return chunk_document_stream(
        extract_pages_stream(pdf_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def ingest_pdf(
    pdf_path: str | Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    save_chunks: bool = True,
) -> list[Chunk]:
    """Full ingestion pipeline: PDF → extracted pages → chunks.

    Steps:
    1. Extract all pages (text + tables)
    2. Chunk each page with overlap
    3. Assign IDs and detect section titles
    4. Optionally save chunks to disk

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Override chunk size from settings.
        chunk_overlap: Override chunk overlap from settings.
        save_chunks: Whether to save chunks to data/chunks/.

    Returns:
        List of Chunk objects ready for embedding and indexing.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Starting ingestion: {pdf_path.name}")

    all_chunks: list[Chunk] = []
    async for chunk in ingest_pdf_streaming(
        pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, save_chunks=save_chunks
    ):
        all_chunks.append(chunk)

    logger.info(
        f"Ingestion complete: {pdf_path.name} → {len(all_chunks)} chunks "
        f"(avg {sum(c.token_count for c in all_chunks) // max(len(all_chunks), 1)} tokens/chunk)"
    )
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════
# CONTEXTUAL RETRIEVAL (Anthropic) — situate each chunk in the document
# ═══════════════════════════════════════════════════════

# Common patterns for section headings in university guides
_HEADING_PATTERNS = [
    re.compile(r"^(Chapter|Section|Part)\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+\.\d*\s+[A-Z]"),        # "1.2 Admission Requirements"
    re.compile(r"^[A-Z][A-Z\s]{5,}$"),        # "ALL CAPS HEADING"
    re.compile(r"^[IVX]+\.\s+[A-Z]"),         # Roman numeral headings
]


def detect_section_title(text: str) -> str:
    """Try to detect a section heading from the first few lines of text."""
    lines = text.strip().split("\n")[:5]
    for line in lines:
        line = line.strip()
        if not line or len(line) > 120:
            continue
        for pattern in _HEADING_PATTERNS:
            if pattern.match(line):
                return line
    return ""


# ═══════════════════════════════════════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════════════════════════════════════


_tokenizer = None


def _count_tokens(text: str) -> int:
    """Token count via tiktoken (cl100k_base); falls back to a word heuristic."""
    global _tokenizer
    try:
        if _tokenizer is None:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(_tokenizer.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return the trailing portion of `text` worth ~overlap_tokens (word-based)."""
    words = text.split()
    if not words:
        return ""
    overlap_words = max(1, int(overlap_tokens / 1.3))
    if overlap_words >= len(words):
        return text
    return " ".join(words[-overlap_words:])


def chunk_document(
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Chunk the WHOLE document (cross-page), tracking the page span of each chunk.

    Paragraph-aware recursive splitting: accumulate paragraphs up to the token
    budget; oversized paragraphs are split by sentence. Chunks may span page
    boundaries, and ``page_numbers`` records every page a chunk draws from.

    Returns: list of {text, page_numbers: list[int], token_count}.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks: list[dict] = []
    cur_text = ""
    cur_pages: list[int] = []
    cur_tokens = 0

    def flush(next_page: int | None) -> None:
        nonlocal cur_text, cur_pages, cur_tokens
        if not cur_text.strip():
            return
        chunks.append({
            "text": cur_text.strip(),
            "page_numbers": sorted(set(cur_pages)),
            "token_count": cur_tokens,
        })
        # Start the next chunk with an overlap tail (carry the last page seen).
        overlap = _overlap_tail(cur_text, chunk_overlap)
        carry = cur_pages[-1] if cur_pages else next_page
        cur_text = overlap
        cur_pages = [p for p in (carry,) if p is not None]
        cur_tokens = _count_tokens(overlap)

    def add(piece: str, page_num: int, sep: str) -> None:
        nonlocal cur_text, cur_pages, cur_tokens
        cur_text += (sep + piece) if cur_text else piece
        cur_pages.append(page_num)
        cur_tokens += _count_tokens(piece)

    for page_num, para in _page_blocks(pages):
        para_tokens = _count_tokens(para)
        if para_tokens > chunk_size:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                sent = sent.strip()
                if not sent:
                    continue
                if cur_tokens + _count_tokens(sent) > chunk_size and cur_text:
                    flush(page_num)
                add(sent, page_num, " ")
        elif cur_tokens + para_tokens > chunk_size and cur_text:
            flush(page_num)
            add(para, page_num, "\n\n")
        else:
            add(para, page_num, "\n\n")

    flush(None)
    return chunks


# ═══════════════════════════════════════════════════════════════════════
# MAIN INGESTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════


async def ingest_pdf(
    pdf_path: str | Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    save_chunks: bool = True,
) -> list[Chunk]:
    """Full ingestion pipeline: PDF → extracted pages → chunks.

    Steps:
    1. Extract all pages (text + tables)
    2. Chunk each page with overlap
    3. Assign IDs and detect section titles
    4. Optionally save chunks to disk

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Override chunk size from settings.
        chunk_overlap: Override chunk overlap from settings.
        save_chunks: Whether to save chunks to data/chunks/.

    Returns:
        List of Chunk objects ready for embedding and indexing.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Starting ingestion: {pdf_path.name}")

    # Step 1: Extract pages
    pages = extract_pages(pdf_path)

    # Step 2: Chunk across the whole document (cross-page, page-span tracked)
    all_chunks: list[Chunk] = []
    raw_chunks = chunk_document(pages, chunk_size, chunk_overlap)

    for chunk_index, raw in enumerate(raw_chunks):
        section_title = detect_section_title(raw["text"])
        pages_span = raw["page_numbers"]
        chunk = Chunk(
            id=f"{pdf_path.stem}_c{chunk_index}",
            text=raw["text"],
            page_numbers=pages_span,
            section_title=section_title,
            chunk_index=chunk_index,
            token_count=raw["token_count"],
            metadata={
                "source": pdf_path.name,
                "pages": pages_span,
            },
        )
        all_chunks.append(chunk)

    logger.info(
        f"Ingestion complete: {len(pages)} pages → {len(all_chunks)} chunks "
        f"(avg {sum(c.token_count for c in all_chunks) // max(len(all_chunks), 1)} tokens/chunk)"
    )

    # Step 2.5: Contextual Retrieval — situate each chunk in the document (opt-in)
    if settings.contextual_retrieval_enabled and all_chunks:
        doc_sample = "\n\n".join(p["text"] for p in pages)[: settings.contextual_doc_sample_chars]
        await _contextualize_chunks(all_chunks, doc_sample)
        logger.info("Contextual Retrieval: per-chunk context generated")

    # Step 3: Save chunks to disk
    if save_chunks:
        chunks_dir = settings.chunks_path
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / f"{pdf_path.stem}_chunks.json"

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)

        logger.info(f"Chunks saved to: {chunks_file}")

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════
# CONTEXTUAL RETRIEVAL (Anthropic) — situate each chunk in the document
# ═══════════════════════════════════════════════════════════════════════


async def _contextualize_chunks(chunks: list[Chunk], doc_sample: str) -> None:
    """Generate a short situating context for each chunk and store it in-place.

    Bounded concurrency keeps index time reasonable; failures degrade to no
    context (the chunk still indexes on its own text).
    """
    import asyncio

    from app.services.llm import chat

    semaphore = asyncio.Semaphore(max(1, settings.contextual_max_concurrency))

    async def _one(chunk: Chunk) -> None:
        prompt = prompts.contextualize_chunk(doc_sample, chunk.text)
        async with semaphore:
            try:
                resp = await chat(prompt, temperature=0)
                chunk.context = (resp.content or "").strip()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Contextualization failed for {chunk.id}: {e}")

    await asyncio.gather(*(_one(c) for c in chunks))


def load_chunks(pdf_stem: str) -> list[Chunk] | None:
    """Load previously saved chunks from disk.

    Args:
        pdf_stem: The PDF filename without extension.

    Returns:
        List of Chunks if found, None otherwise.
    """
    chunks_file = settings.chunks_path / f"{pdf_stem}_chunks.json"
    if not chunks_file.exists():
        return None

    with open(chunks_file, encoding="utf-8") as f:
        raw = json.load(f)

    chunks = [Chunk(**item) for item in raw]
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
    return chunks
