# Phase 1 Notes — What This Is, and What's Deliberately Simplified

## What's real vs. what's a placeholder right now

| Piece | Status |
|---|---|
| Chunking | Real, final logic — this doesn't change in later phases |
| TF-IDF embedding | **Baseline placeholder.** Works, is tested, but is a keyword-matching technique, not true semantic understanding |
| NumPy brute-force vector search | **Fine as-is** up to tens of thousands of chunks; swap for FAISS beyond that |
| SEC EDGAR fetcher | Real, final logic — untested *live* only because this sandbox has no internet, not because the code is incomplete |
| RAG prompt construction | Real, final logic |
| LLM client | Real — just needs your API key to actually generate text instead of MockClient's placeholder response |

## Why TF-IDF instead of real embeddings, starting out

Dense neural embeddings (sentence-transformers, OpenAI embeddings, etc.)
capture meaning beyond shared words — they'd know "profit declined" and
"net income fell" are related even with zero words in common. TF-IDF
can't do that; it only matches on shared vocabulary.

But TF-IDF needs no model download, no GPU, no API key, and runs
instantly — which is why it's what's actually running in this build, and
why the whole pipeline could be built and tested end-to-end today,
inside a sandbox with no internet access at all. That's a real trade-off,
not a shortcut being hidden from you: know that your retrieval quality
right now is "good enough to prove the pipeline works," not "as good as
this project will eventually be."

## The upgrade path (do this once you have internet + a bit of GPU)

1. `pip install sentence-transformers`
2. In `src/rag/retriever.py`, change:
   ```python
   embedder = embedder or TfidfEmbedder()
   ```
   to:
   ```python
   from ..pipeline.embedder import DenseEmbedder
   embedder = embedder or DenseEmbedder("all-MiniLM-L6-v2")
   ```
3. Everything else — chunker, vector store, retriever interface, RAG
   chain — needs zero changes. This is the entire point of the
   `TfidfEmbedder`/`DenseEmbedder` sharing one interface (`fit`, `embed`).

Similarly, once your corpus grows past a single sample filing:
- Swap `NumpyVectorStore` for a FAISS-backed store once brute-force
  search actually becomes a bottleneck (measure first — don't guess).
- Swap `MockClient` for `AnthropicClient` or `OpenAICompatibleClient`
  once you have an API key, to get real generated answers instead of
  the placeholder string.

## What Phase 2 adds on top of this

- Real filings across multiple companies (via `edgar_fetcher.py`, which
  already works — it just needs to run somewhere with internet)
- LoRA/QLoRA fine-tuning of a small open model on financial Q&A
- An evaluation harness: does the model's stated answer actually match
  what the cited chunk says? (hallucination rate, numeric accuracy)

That evaluation harness is the JPMorgan-relevant artifact — it doesn't
exist yet in Phase 1, which only proves the retrieval pipeline works.
