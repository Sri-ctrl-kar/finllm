"""
vector_store.py
------------------------------------------------------------
Stores chunk vectors and answers "which K chunks are most similar
to this query vector?"

This implementation is brute-force: compare the query against
EVERY stored vector, every time. That is O(n) per query, where n =
number of chunks. For a corpus of a few thousand chunks (a handful
of companies' filings — exactly Phase 1's scope) that's a few
milliseconds and totally fine, and it means zero extra dependencies
to get the whole pipeline running today.

WHEN TO UPGRADE TO FAISS: once the corpus grows into the tens/
hundreds of thousands of chunks (many companies, many years of
filings — Phase 2+ territory), brute-force O(n) search starts to
matter. FAISS builds an approximate-nearest-neighbor index (e.g.
HNSW or IVF) that answers the same query in roughly O(log n) by
trading a small amount of recall for a lot of speed. The interface
below (`add`, `search`) is intentionally the same shape FAISS
wrappers use, so swapping the implementation later doesn't require
touching any calling code — just this file.
------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant


class NumpyVectorStore:
    def __init__(self):
        self._vectors: np.ndarray | None = None  # (n_chunks, dim)
        self._chunks: List[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[SearchResult]:
        """
        Assumes query_vector and stored vectors are already L2-normalized
        (TfidfEmbedder/DenseEmbedder both do this) — so cosine similarity
        is just a dot product, which is what makes this fast even as a
        brute-force numpy operation (a single matrix-vector multiply).
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        scores = self._vectors @ query_vector  # (n_chunks,) — dot product with every stored vector at once
        top_k_idx = np.argsort(-scores)[:k]

        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_k_idx]

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str) -> None:
        """Persist to disk so you don't have to re-embed the whole corpus every run."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump({"vectors": self._vectors, "chunks": self._chunks}, f)

    @classmethod
    def load(cls, path: str) -> "NumpyVectorStore":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store._vectors = data["vectors"]
        store._chunks = data["chunks"]
        return store
