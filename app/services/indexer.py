"""PageIndex wrapper — transforms a PDF into a navigable tree.

This service wraps the PageIndex library to:
1. Build a hierarchical tree index from a PDF (one-time indexing)
2. Load and serve the tree for queries
3. Provide page-level text retrieval via PyMuPDF

Patches PageIndex to use centralized LLM service so it works with any provider
(Azure, Groq, OpenAI, Ollama, etc.) not just OpenAI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# Thread pool for blocking PageIndex operations
_executor = ThreadPoolExecutor(max_workers=1)


class DocumentIndex:
    """Manages the lifecycle of a single document's index and page cache.

    Attributes:
        doc_name: Name of the indexed document.
        tree: The hierarchical tree structure (PageIndex output).
        page_texts: Cached page texts keyed by 1-based page number.
        total_pages: Total number of pages in the document.
    """

    def __init__(self) -> None:
        self.doc_name: str = ""
        self.tree: list[dict] = []
        self.page_texts: dict[int, str] = {}
        self.total_pages: int = 0
        self.indexed_at: str = ""             # ISO-8601 build time (data freshness)
        self._flat_nodes: dict[str, dict] = {}  # node_id → node for fast lookup

    @property
    def is_loaded(self) -> bool:
        return bool(self.tree)

    # ── Indexing ────────────────────────────────────────────────────

    async def build_index(
        self,
        pdf_path: str | Path,
        index_dir: str | Path,
        model: str | None = None,
        max_pages_per_node: int = 10,
        max_tokens_per_node: int = 20000,
        toc_check_pages: int = 20,
        doc_stem: str | None = None,
    ) -> dict:
        """Build a PageIndex tree from a PDF and save it.

        Uses the centralized LLM service, so it works with ANY provider
        (Azure, Groq, OpenAI, Ollama, etc.)

        Args:
            pdf_path: Path to the PDF file.
            index_dir: Directory to save the JSON index.
            model: Model to use (defaults to settings.active_model_name).
            max_pages_per_node: Max pages before a node is subdivided.
            max_tokens_per_node: Max tokens before a node is subdivided.
            toc_check_pages: Number of initial pages to scan for TOC.

        Returns:
            The generated tree structure dict.
        """
        from app.config import settings

        pdf_path = Path(pdf_path)
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Building PageIndex tree for: {pdf_path.name}")
        logger.info(f"Using LLM provider: {settings.active_model.value}")

        # Use active model from settings (unless overridden)
        model = model or settings.active_model_name

        # Import PageIndex dynamically (it lives in the research docs)
        pageindex_path = self._find_pageindex_module()
        if pageindex_path:
            sys.path.insert(0, str(pageindex_path))

        # PATCH OPENAI BEFORE IMPORTING PAGEINDEX. This intercepts PageIndex's
        # OpenAI calls globally; we restore the real module in `finally` so the
        # shadow never leaks past the build.
        _real_openai = self._setup_openai_interception()

        try:
            from pageindex import page_index_main, config as pi_config

            opt = pi_config(
                model=model,
                toc_check_page_num=toc_check_pages,
                max_page_num_each_node=max_pages_per_node,
                max_token_num_each_node=max_tokens_per_node,
                if_add_node_id="yes",
                if_add_node_summary="yes",
                if_add_doc_description="yes",
                if_add_node_text="no",
            )

            # Run PageIndex in thread pool to avoid nested event loop error
            # (PageIndex internally calls asyncio.run())
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                page_index_main,
                str(pdf_path),
                opt,
            )

        except ImportError:
            logger.warning(
                "PageIndex library not found in research docs. "
                "Using fallback PyMuPDF-based indexing."
            )
            result = await self._fallback_index(pdf_path)
        except Exception as e:
            logger.error(f"PageIndex failed: {e}. Falling back to PyMuPDF.")
            result = await self._fallback_index(pdf_path)
        finally:
            self._restore_openai(_real_openai)

        # Stamp build time for data-freshness reporting.
        from datetime import datetime, timezone
        result.setdefault("indexed_at", datetime.now(timezone.utc).isoformat())

        # Save index (tenant-namespaced stem when provided)
        index_file = index_dir / f"{doc_stem or pdf_path.stem}_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Index saved to: {index_file}")

        # Load into memory
        self._load_from_result(result)
        self._cache_pages(pdf_path)

        return {
            "doc_name": self.doc_name,
            "total_pages": self.total_pages,
            "tree_nodes": len(self._flat_nodes),
        }

    def load_index(self, index_file: str | Path, pdf_path: str | Path) -> None:
        """Load a previously built index from disk.

        Args:
            index_file: Path to the JSON index file.
            pdf_path: Path to the original PDF (for page text extraction).
        """
        index_file = Path(index_file)
        pdf_path = Path(pdf_path)

        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")

        with open(index_file, encoding="utf-8") as f:
            result = json.load(f)

        self._load_from_result(result)
        self._cache_pages(pdf_path)

        logger.info(
            f"Loaded index: {self.doc_name} "
            f"({self.total_pages} pages, {len(self._flat_nodes)} nodes)"
        )

    def _setup_openai_interception(self) -> None:
        """Global OpenAI client interception — replaces OpenAI module with our wrapper.
        
        This must be called BEFORE PageIndex is imported. We replace the entire openai
        module in sys.modules with a transparent proxy that:
        - Intercepts OpenAI() client construction and ChatCompletion.create()
        - Delegates ALL other attributes (AsyncOpenAI, AsyncAzureOpenAI, etc.) to the
          REAL openai module so our own llm.py keeps working fine.
        """
        try:
            import importlib
            from app.services.llm import chat as centralized_chat
            from app.config import settings
            
            # Load the REAL openai module first (before we shadow it)
            real_openai = importlib.import_module("openai")
            
            # Helper function for interception logic
            def _intercept_openai_call(*args, **kwargs) -> object:
                """Intercept OpenAI API calls and route through our LLM service."""
                import concurrent.futures

                # Attribute-access object to mimic openai SDK response shape
                # PageIndex uses response.choices[0].message.content (not dict keys)
                class _Obj:
                    def __init__(self, **kw):
                        for k, v in kw.items():
                            setattr(self, k, v)

                try:
                    messages = kwargs.get("messages", [])
                    if not messages and args:
                        messages = args[0] if isinstance(args[0], list) else []
                    
                    if not messages:
                        logger.warning("No messages in OpenAI mock call")
                        return _Obj(choices=[_Obj(message=_Obj(role="assistant", content=""), finish_reason="stop")],
                                    usage=_Obj(completion_tokens=0, total_tokens=0))
                    
                    # Preserve BOTH the system instruction and the user content.
                    # PageIndex puts critical structure-extraction rules in the
                    # system role — dropping it degrades tree/summary quality.
                    system_message = ""
                    user_message = ""
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get("role")
                        if role == "system" and not system_message:
                            system_message = msg.get("content", "") or ""
                        elif role == "user":
                            user_message = msg.get("content", "") or user_message
                    if not user_message and messages:
                        last = messages[-1]
                        user_message = str(last.get("content", "") if isinstance(last, dict) else last)

                    # Structure extraction must be deterministic.
                    temp = kwargs.get("temperature", 0)

                    # Spawn a fresh thread (no inherited event loop) so asyncio.run() works
                    def _run_async():
                        return asyncio.run(
                            centralized_chat(
                                user_message,
                                system=system_message or None,
                                temperature=temp,
                            )
                        )
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        response = pool.submit(_run_async).result(timeout=60)
                    
                    logger.debug(
                        f"OpenAI call intercepted (PageIndex) → routed through {settings.active_model_name}"
                    )
                    
                    # Return object with attribute access (.choices[0].message.content)
                    return _Obj(
                        choices=[
                            _Obj(
                                message=_Obj(role="assistant", content=response.content),
                                finish_reason="stop",
                                index=0,
                            )
                        ],
                        usage=_Obj(
                            prompt_tokens=0,
                            completion_tokens=response.tokens_used,
                            total_tokens=response.tokens_used,
                        ),
                        model=response.model,
                    )
                except Exception as e:
                    logger.error(f"OpenAI interception failed: {e}")
                    return _Obj(
                        choices=[_Obj(message=_Obj(role="assistant", content=""), finish_reason="stop")],
                        usage=_Obj(completion_tokens=0, total_tokens=0),
                        error=str(e),
                    )
            
            # Build mock classes for what PageIndex uses
            class CompletionsMock:
                def create(self, *args, **kwargs):
                    return _intercept_openai_call(*args, **kwargs)
            
            class ChatMock:
                def __init__(self):
                    self.completions = CompletionsMock()
            
            class OpenAIClientMock:
                """Intercepts PageIndex's OpenAI(api_key=...) construction."""
                def __init__(self, *args, **kwargs):
                    self.chat = ChatMock()
            
            class ChatCompletionMock:
                """Old-style openai.ChatCompletion.create()."""
                @staticmethod
                def create(*args, **kwargs):
                    return _intercept_openai_call(*args, **kwargs)
            
            # Transparent proxy: intercept OpenAI client creation,
            # but delegate everything else to the real openai module.
            class OpenAIProxy:
                """Transparent proxy over real openai — intercepts only client construction."""
                
                # Intercepted
                OpenAI = OpenAIClientMock
                ChatCompletion = ChatCompletionMock()
                
                def __getattr__(self, name: str):
                    """Delegate any attribute not explicitly intercepted to real openai."""
                    return getattr(real_openai, name)
            
            sys.modules["openai"] = OpenAIProxy()
            logger.info(
                "OpenAI module intercepted for this build → "
                f"PageIndex will use {settings.active_model_name}"
            )
            return real_openai

        except Exception as e:
            logger.warning(f"Failed to set up OpenAI interception: {e}")
            return None

    def _restore_openai(self, real_openai) -> None:
        """Undo the global openai interception installed for the PageIndex build.

        Keeps the shadow scoped to the build so anything constructing the sync
        ``openai.OpenAI()`` client afterwards gets the real SDK, not our mock.
        """
        if real_openai is not None:
            sys.modules["openai"] = real_openai

    # ── Retrieval ───────────────────────────────────────────────────

    def get_tree_for_search(self) -> str:
        """Return a compact string representation of the tree for LLM consumption.

        Strips page text and verbose fields to minimize token usage.
        The LLM uses this to decide which nodes contain the answer.
        """
        if not self.tree:
            return "No document indexed."

        def compact_node(node: dict, depth: int = 0) -> str:
            indent = "  " * depth
            node_id = node.get("node_id", "?")
            title = node.get("title", "Untitled")
            summary = node.get("summary", "")
            start = node.get("start_physical_index") or node.get("physical_index") or "?"
            end = node.get("end_physical_index") or start

            line = f"{indent}[{node_id}] {title} (pp. {start}-{end})"
            if summary:
                # Truncate long summaries for the search prompt
                short = summary[:150] + "..." if len(summary) > 150 else summary
                line += f"\n{indent}    Summary: {short}"

            children = node.get("children", [])
            child_lines = [compact_node(c, depth + 1) for c in children]

            return "\n".join([line] + child_lines)

        return "\n".join(compact_node(node) for node in self.tree)

    def tree_nodes(self) -> list[dict]:
        """The structured tree (root nodes with nested children) for navigation."""
        return self.tree

    def node_ids(self) -> set[str]:
        """Every real node id in the tree (for navigation validation/repair)."""
        return set(self._flat_nodes.keys())

    def get_pages_for_nodes(self, node_ids: list[str], max_pages: int | None = None) -> list[dict]:
        """Retrieve full page texts for the given node IDs, within a page budget.

        Selecting a broad parent node can span the whole document; the budget caps
        total pages so a single safety-net selection can't flood the context window.

        Args:
            node_ids: PageIndex node IDs to retrieve, in priority order.
            max_pages: Cap on total pages returned (defaults to settings.pageindex_max_pages).

        Returns:
            List of {page_number, text, section_title, node_id} dicts.
        """
        from app.config import settings

        if max_pages is None:
            max_pages = settings.pageindex_max_pages

        pages: list[dict] = []
        seen_pages: set[int] = set()

        for nid in node_ids:
            node = self._flat_nodes.get(nid)
            if not node:
                logger.warning(f"Node ID not found (skipped): {nid}")
                continue

            start = node.get("start_physical_index") or node.get("physical_index")
            end = node.get("end_physical_index") or start
            if start is None:
                continue

            for page_num in range(int(start), int(end) + 1):
                if len(pages) >= max_pages:
                    logger.info(f"Page budget reached ({max_pages}); truncating selection.")
                    return pages
                if page_num in seen_pages:
                    continue
                seen_pages.add(page_num)

                text = self.page_texts.get(page_num, "")
                if text:
                    pages.append({
                        "page_number": page_num,
                        "text": text,
                        "section_title": node.get("title", ""),
                        "node_id": nid,
                    })

        return pages

    def sample_text(self, max_chars: int = 6000) -> str:
        """Return a concatenated sample of the document for profiling."""
        if not self.page_texts:
            return ""
        parts: list[str] = []
        total = 0
        for page_num in sorted(self.page_texts):
            text = (self.page_texts.get(page_num) or "").strip()
            if not text:
                continue
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]

    def get_page(self, page_num: int) -> dict | None:
        """Get a single page's text by 1-based page number."""
        text = self.page_texts.get(page_num)
        if text is None:
            return None
        return {
            "page_number": page_num,
            "text": text,
            "token_count": len(text.split()),  # approximate
        }

    def get_tree_response(self) -> dict:
        """Return the tree in a format suitable for the frontend tree explorer."""

        def to_tree_node(node: dict) -> dict:
            return {
                "node_id": node.get("node_id", ""),
                "title": node.get("title", "Untitled"),
                "summary": node.get("summary", ""),
                "page_start": node.get("start_physical_index") or node.get("physical_index"),
                "page_end": node.get("end_physical_index"),
                "children": [to_tree_node(c) for c in node.get("children", [])],
            }

        return {
            "doc_name": self.doc_name,
            "total_pages": self.total_pages,
            "tree": [to_tree_node(n) for n in self.tree],
        }

    # ── Internal ────────────────────────────────────────────────────

    def _load_from_result(self, result: dict) -> None:
        """Parse a PageIndex result dict into our internal structures."""
        self.doc_name = result.get("doc_name", "unknown")
        self.tree = result.get("structure", result.get("tree", []))
        self.indexed_at = result.get("indexed_at", "")
        self._flat_nodes = {}
        self._flatten_tree(self.tree)

    def _flatten_tree(self, nodes: list[dict]) -> None:
        """Build a flat lookup dict from the tree hierarchy."""
        for node in nodes:
            nid = node.get("node_id")
            if nid:
                self._flat_nodes[nid] = node
            for child in node.get("children", []):
                self._flatten_tree([child])

    def _cache_pages(self, pdf_path: Path) -> None:
        """Extract and cache all page texts from the PDF using PyMuPDF."""
        if not pdf_path.exists():
            logger.warning(f"PDF not found for page caching: {pdf_path}")
            return

        from app.config import settings

        doc = pymupdf.open(str(pdf_path))
        self.total_pages = len(doc)
        self.page_texts = {}

        # Reading-order extraction keeps multi-column / complex layouts coherent
        # (PageIndex returns whole pages, so layout fidelity matters for tables).
        sort = settings.pageindex_preserve_reading_order
        for i in range(len(doc)):
            page = doc[i]
            try:
                text = page.get_text(sort=sort) if sort else page.get_text()
            except TypeError:  # older PyMuPDF without the sort kwarg
                text = page.get_text()
            self.page_texts[i + 1] = text  # 1-based page numbers

        doc.close()
        logger.info(f"Cached {self.total_pages} pages from {pdf_path.name}")

    def _find_pageindex_module(self) -> Path | None:
        """Locate the PageIndex module in the research docs."""
        # Walk up from our location to find the docs folder
        candidates = [
            Path(__file__).parent.parent.parent / "docs" / "RAG" / "PageIndex",
            Path("docs") / "RAG" / "PageIndex",
            Path("..") / "docs" / "RAG" / "PageIndex",
        ]
        for candidate in candidates:
            if (candidate / "pageindex").is_dir():
                logger.info(f"Found PageIndex at: {candidate}")
                return candidate
        return None

    async def _fallback_index(self, pdf_path: Path) -> dict:
        """Build a simple chapter-based index when PageIndex isn't available.

        Uses PyMuPDF's table of contents extraction as a fallback.
        Not as powerful as PageIndex's LLM-based tree, but functional.
        """
        doc = pymupdf.open(str(pdf_path))
        toc = doc.get_toc()  # [[level, title, page_number], ...]
        total_pages = len(doc)
        doc.close()

        if not toc:
            # No TOC found — create a flat structure with page ranges
            logger.warning("No TOC found in PDF. Creating flat page-range index.")
            nodes = []
            chunk_size = 20  # Group every 20 pages
            for start in range(1, total_pages + 1, chunk_size):
                end = min(start + chunk_size - 1, total_pages)
                nodes.append({
                    "node_id": f"n_{start}_{end}",
                    "title": f"Pages {start}-{end}",
                    "summary": "",
                    "start_physical_index": start,
                    "end_physical_index": end,
                    "physical_index": start,
                    "children": [],
                })
            return {
                "doc_name": pdf_path.stem,
                "structure": nodes,
            }

        # Build tree from TOC
        nodes = []
        for i, (level, title, page) in enumerate(toc):
            # Determine end page
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else total_pages
            end_page = max(end_page, page)

            nodes.append({
                "node_id": f"n_{i + 1}",
                "title": title,
                "summary": title,  # no LLM summaries in fallback — title gives the navigator signal
                "level": level,
                "start_physical_index": page,
                "end_physical_index": end_page,
                "physical_index": page,
                "children": [],
            })

        # Nest by level
        tree = self._nest_by_level(nodes)

        return {
            "doc_name": pdf_path.stem,
            "structure": tree,
        }

    def _nest_by_level(self, flat_nodes: list[dict]) -> list[dict]:
        """Convert a flat list of nodes with levels into a nested tree."""
        if not flat_nodes:
            return []

        root_nodes = []
        stack: list[dict] = []

        for node in flat_nodes:
            level = node.pop("level", 1)

            # Pop stack to find parent
            while stack and stack[-1][1] >= level:
                stack.pop()

            if stack:
                stack[-1][0]["children"].append(node)
            else:
                root_nodes.append(node)

            stack.append((node, level))

        return root_nodes
