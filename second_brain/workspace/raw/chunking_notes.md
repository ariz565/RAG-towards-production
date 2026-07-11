Chunking splits a document into retrievable units before embedding. Fixed-size
chunking uses a chunk_size and chunk_overlap in tokens; overlap must be smaller
than chunk_size or the sliding window never advances. Recursive chunking respects
structure (headings, paragraphs, sentences) before falling back to a hard token
cut. Chunk quality caps retrieval quality -- a hybrid retriever fusing BM25 and
dense search (see hybrid_retrieval_notes.md) still only ever sees whatever a
chunker decided a "unit" of the document was. See chunking/fixed.py and
chunking/recursive.py in this repo for offline, dependency-light implementations.
