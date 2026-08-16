"""
retriever.py
------------------------------------------------------------
The piece that turns "here's a corpus of chunks" + "here's a
question" into "here are the K most relevant chunks."

This is deliberately a thin wrapper — its whole job is to hide the
embed-then-search two-step behind one method, `retrieve(query, k)`,
so the RAG chain (qa_chain.py) doesn't need to know embeddings or
vector search exist at all.
------------------------------------------------------------
"""

from typing import List

from ..pipeline.chunker import Chunk
from ..pipeline.embedder import TfidfEmbedder
from ..pipeline.vector_store import NumpyVectorStore, SearchResult


class Retriever:
    def __init__(self, embedder: TfidfEmbedder, store: NumpyVectorStore):
        self.embedder = embedder
        self.store = store

    @classmethod
    def build(cls, chunks: List[Chunk], embedder: TfidfEmbedder | None = None) -> "Retriever":
        """One-shot constructor: fit the embedder on the corpus, embed every
        chunk, and load them into a fresh vector store."""
        embedder = embedder or TfidfEmbedder()
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        vectors = embedder.embed(texts)

        store = NumpyVectorStore()
        store.add(vectors, chunks)
        return cls(embedder, store)

    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        query_vector = self.embedder.embed([query])[0]
        return self.store.search(query_vector, k=k)
