"""
embedder.py
------------------------------------------------------------
Turns text into a vector so we can compare chunks by *meaning*
instead of exact keyword match.

TWO IMPLEMENTATIONS, same interface (`fit`, `embed`), so the rest
of the pipeline never needs to know which one is in use:

  1. TfidfEmbedder  — classic bag-of-words weighting. Fully offline,
     no model download, no GPU. This is what actually runs in this
     build. It's a legitimate baseline, not a toy: TF-IDF was the
     standard search technique for decades before neural embeddings.
     Its limitation: it matches on *shared words*, not true meaning
     ("net income fell" and "profit declined" score as unrelated,
     even though they mean roughly the same thing).

  2. DenseEmbedder   — a real neural embedding model (e.g.
     sentence-transformers' all-MiniLM-L6-v2), which DOES capture
     meaning beyond shared words. Needs a one-time model download
     (~90MB) and the `sentence-transformers` package — both require
     internet, which is why this build ships the TF-IDF version as
     the default, with this class ready to swap in once you're
     running with a network connection. Swapping it in later is a
     one-line change (see vector_store.py + README).
------------------------------------------------------------
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams: "net income" is one feature, not two
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        """Learn the vocabulary from the corpus. Must be called once before embed()."""
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns a dense (n_texts, n_features) numpy array, L2-normalized
        so cosine similarity reduces to a plain dot product downstream."""
        if not self._fitted:
            raise RuntimeError("Call fit() on the corpus before embed().")
        sparse = self.vectorizer.transform(texts)
        dense = sparse.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero for empty/all-stopword chunks
        return dense / norms

    @property
    def dim(self) -> int:
        return len(self.vectorizer.vocabulary_)


class DenseEmbedder:
    """
    Real neural embeddings — the upgrade path once you have internet.
    Not used by default in this build (see module docstring).

    Usage once `sentence-transformers` is installed:
        pip install sentence-transformers
        embedder = DenseEmbedder("all-MiniLM-L6-v2")
        embedder.fit([])  # no-op, kept for interface parity with TfidfEmbedder
        vectors = embedder.embed(["some text", "more text"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # imported lazily

            self._model = SentenceTransformer(self.model_name)

    def fit(self, texts: List[str]) -> None:
        # Neural embedding models are pre-trained — nothing to fit on your corpus.
        # Kept as a no-op so DenseEmbedder is a drop-in replacement for TfidfEmbedder.
        self._load()

    def embed(self, texts: List[str]) -> np.ndarray:
        self._load()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
