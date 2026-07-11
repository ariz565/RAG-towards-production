"""Second Brain — a from-scratch implementation of Karpathy's LLM-Wiki pattern.

Standalone study module, same spirit as retrieval/, chunking/, embeddings/:
pure-Python, offline-by-default, zero new dependencies. One deliberate exception
to "not wired into the app": `app/routers/second_brain.py` exposes a single
read-only endpoint over this module's graph, for the vision-ui frontend's graph
page. See README.md for the full design writeup.
"""

from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"
