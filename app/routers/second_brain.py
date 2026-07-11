"""Read-only bridge into the second_brain/ standalone module.

second_brain (see second_brain/README.md) is otherwise fully decoupled from this
app -- no shared auth, no tenant concept, its own CLI and file layout. This is
the one deliberate exception: a single endpoint so vision-ui can render the
personal-wiki graph. It reads second_brain's own SQLite/markdown files directly;
there's no service layer in between because second_brain isn't meant to grow
into a second product surface inside this one, just a viewable one.

No auth dependency, matching /api/health elsewhere in this app -- there's no
tenant concept to scope this to, and the underlying data (a personal knowledge
graph the operator builds locally via the CLI) isn't sensitive in the way
per-tenant documents are.

Plain `def`, not `async def`: node_records()/edge_records() do synchronous
sqlite3/file I/O (second_brain has no async story, by design -- see its own
README). A sync route handler is FastAPI's correct way to run blocking code --
it's dispatched to the threadpool automatically, unlike calling blocking I/O
directly from an `async def` handler (which blocks the event loop; see the
known issues in app/services/registry.py's load_bundle() and friends).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/second-brain", tags=["second-brain"])


@router.get("/graph")
def get_graph() -> dict:
    from second_brain import WORKSPACE_DIR
    from second_brain.graph import GraphStore
    from second_brain.graph_html import edge_records, node_records
    from second_brain.store import WikiStore

    graph_db = WORKSPACE_DIR / "wiki" / "graph.db"
    if not graph_db.exists():
        # Don't let a GET request create an empty graph.db as a side effect --
        # GraphStore.__init__ would happily do that (sqlite3.connect creates the
        # file if missing), which is fine for the CLI's own writes but wrong for
        # a read endpoint to trigger incidentally.
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}

    store = WikiStore(WORKSPACE_DIR)
    graph = GraphStore(graph_db)
    try:
        nodes = node_records(graph, store)
        edges = edge_records(graph)
    finally:
        graph.close()

    return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}
