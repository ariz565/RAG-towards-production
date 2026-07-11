// Sample data for the Graph page's local preview -- used ONLY as a fallback
// when GET /api/second-brain/graph is unreachable (backend not running), never
// as a silent substitute for real data. graph/page.tsx shows a visible "sample
// data" banner whenever this is what's on screen, specifically so this never
// gets confused for a live graph the way vision-ui's Dashboard page's
// Math.random() telemetry does elsewhere in this app -- that's the anti-pattern
// this file is deliberately not repeating.
//
// The content itself isn't fabricated: it's the same real, accurate RAG-concept
// dataset (hybrid retrieval / chunking / evaluation clusters) seeded into
// second_brain's actual graph.db for local testing, in the exact record shape
// second_brain/graph_html.py's node_records()/edge_records() produce -- so this
// preview looks exactly like what you'll see once the backend is running.

export const MOCK_SECOND_BRAIN_GRAPH = {
  node_count: 13,
  edge_count: 11,
  nodes: [
    { id: "concepts/hybrid-retrieval", label: "Hybrid Retrieval", title: "Combines BM25 (sparse) and a dense bi-encoder, fused with Reciprocal Rank Fusion.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 6, orphan: false },
    { id: "concepts/bm25", label: "BM25 (Sparse Retrieval)", title: "Lexical ranking function using term frequency and inverse document frequency.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/dense-retrieval", label: "Dense Retrieval", title: "Bi-encoder embedding similarity search -- catches paraphrases BM25 misses.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/reciprocal-rank-fusion", label: "Reciprocal Rank Fusion (RRF)", title: "Fuses multiple ranked lists by rank position, not raw score.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/reranking", label: "Cross-Encoder Reranking", title: "Second-stage precision pass: re-scores a shortlist by reading query+doc together.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/chunking", label: "Chunking", title: "Splitting a document into retrievable units before embedding.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 3, orphan: false },
    { id: "concepts/fixed-size-chunking", label: "Fixed-Size Chunking", title: "Sliding window over tokens, sized by chunk_size and chunk_overlap.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 2, orphan: false },
    { id: "concepts/recursive-chunking", label: "Recursive Chunking", title: "Structure-aware chunking: headings/paragraphs/sentences before a hard token cut.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/chunk-overlap", label: "Chunk Overlap", title: "Shared tokens between consecutive chunks, preserving context across a boundary.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/rag-evaluation", label: "RAG Evaluation", title: "Measuring retrieval quality with IR metrics before trusting a pipeline change.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 3, orphan: false },
    { id: "concepts/ndcg", label: "nDCG@k", title: "Position-weighted, graded relevance metric -- the one that correlates best with RAG quality.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/recall-at-k", label: "Recall@k", title: "Did the relevant documents make it into the top-k at all? The RAG ceiling metric.", group: "concept", color: { background: "#E7F5FF", border: "#1971C2" }, borderWidth: 1, shapeProperties: { borderDashes: false }, value: 1, orphan: false },
    { id: "concepts/query-rewriting", label: "Query Rewriting", title: "LLM-driven query transforms (HyDE, multi-query, decomposition, step-back).", group: "concept", color: { background: "#E7F5FF", border: "#E03131" }, borderWidth: 3, shapeProperties: { borderDashes: [4, 2] }, value: 1, orphan: true },
  ],
  edges: [
    { id: "e0", from: "concepts/bm25", to: "concepts/hybrid-retrieval", label: "part-of", title: "part-of (confidence 0.95) -- BM25 is one of hybrid retrieval's two component retrievers", color: { color: "#12B886" }, dashes: false, width: 2.9, arrows: "to", relation: "part-of" },
    { id: "e1", from: "concepts/dense-retrieval", to: "concepts/hybrid-retrieval", label: "part-of", title: "part-of (confidence 0.95) -- dense retrieval is hybrid retrieval's other component", color: { color: "#12B886" }, dashes: false, width: 2.9, arrows: "to", relation: "part-of" },
    { id: "e2", from: "concepts/hybrid-retrieval", to: "concepts/reciprocal-rank-fusion", label: "depends-on", title: "depends-on (confidence 0.90) -- hybrid retrieval uses RRF to fuse the BM25 and dense ranked lists", color: { color: "#7048E8" }, dashes: false, width: 2.8, arrows: "to", relation: "depends-on" },
    { id: "e3", from: "concepts/reranking", to: "concepts/hybrid-retrieval", label: "depends-on", title: "depends-on (confidence 0.80) -- reranking operates on hybrid retrieval's candidate shortlist", color: { color: "#7048E8" }, dashes: false, width: 2.6, arrows: "to", relation: "depends-on" },
    { id: "e4", from: "concepts/fixed-size-chunking", to: "concepts/chunking", label: "is-a", title: "is-a (confidence 0.95) -- fixed-size chunking is one chunking strategy", color: { color: "#4C6EF5" }, dashes: false, width: 2.9, arrows: "to", relation: "is-a" },
    { id: "e5", from: "concepts/recursive-chunking", to: "concepts/chunking", label: "is-a", title: "is-a (confidence 0.95) -- recursive chunking is another chunking strategy", color: { color: "#4C6EF5" }, dashes: false, width: 2.9, arrows: "to", relation: "is-a" },
    { id: "e6", from: "concepts/chunk-overlap", to: "concepts/fixed-size-chunking", label: "part-of", title: "part-of (confidence 0.85) -- overlap is a parameter of the fixed-size sliding window", color: { color: "#12B886" }, dashes: false, width: 2.7, arrows: "to", relation: "part-of" },
    { id: "e7", from: "concepts/hybrid-retrieval", to: "concepts/chunking", label: "depends-on", title: "depends-on (confidence 0.85) -- chunk quality caps retrieval quality regardless of retriever sophistication", color: { color: "#7048E8" }, dashes: false, width: 2.7, arrows: "to", relation: "depends-on" },
    { id: "e8", from: "concepts/ndcg", to: "concepts/rag-evaluation", label: "part-of", title: "part-of (confidence 0.90) -- nDCG is one of the standard RAG evaluation metrics", color: { color: "#12B886" }, dashes: false, width: 2.8, arrows: "to", relation: "part-of" },
    { id: "e9", from: "concepts/recall-at-k", to: "concepts/rag-evaluation", label: "part-of", title: "part-of (confidence 0.90) -- recall@k is another standard RAG evaluation metric", color: { color: "#12B886" }, dashes: false, width: 2.8, arrows: "to", relation: "part-of" },
    { id: "e10", from: "concepts/rag-evaluation", to: "concepts/hybrid-retrieval", label: "depends-on", title: "depends-on (confidence 0.75) -- you evaluate a retrieval strategy's actual output, not its design in the abstract", color: { color: "#7048E8" }, dashes: false, width: 2.5, arrows: "to", relation: "depends-on" },
  ],
};
