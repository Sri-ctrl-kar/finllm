"""
test_pipeline.py
------------------------------------------------------------
Runs entirely offline: no network, no API key. Covers chunking,
embedding, vector search, and the end-to-end RAG chain using
MockClient. This is what you'd wire into CI so every commit proves
the pipeline still works, without needing secrets in CI.

Run: python -m pytest tests/ -v
     (or, with zero extra deps: python tests/test_pipeline.py)
------------------------------------------------------------
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.chunker import chunk_text, chunk_sections, clean_text
from src.pipeline.embedder import TfidfEmbedder
from src.pipeline.vector_store import NumpyVectorStore
from src.rag.retriever import Retriever
from src.rag.qa_chain import RAGChain, build_prompt
from src.llm.client import MockClient


def test_clean_text_collapses_whitespace():
    messy = "Hello   world\r\n\r\n\r\n\r\nfoo\tbar"
    cleaned = clean_text(messy)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    print("PASS: clean_text collapses excess whitespace")


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, company="TEST", filing_type="10-K", source_id="t1",
                         chunk_size_words=100, overlap_words=20)

    assert len(chunks) > 1, "expected multiple chunks from 1000 words at chunk_size=100"

    first_words = chunks[0].text.split(" ")
    second_words = chunks[1].text.split(" ")
    assert len(first_words) == 100, f"expected 100 words per chunk, got {len(first_words)}"

    # verify the actual overlap: last 20 words of chunk 0 == first 20 words of chunk 1
    assert first_words[-20:] == second_words[:20], "overlap window doesn't match between consecutive chunks"
    print(f"PASS: chunk_text produced {len(chunks)} chunks with correct size/overlap")


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("some text", "TEST", "10-K", "t1", chunk_size_words=50, overlap_words=50)
        raise AssertionError("expected ValueError when overlap_words >= chunk_size_words")
    except ValueError:
        print("PASS: chunk_text rejects overlap_words >= chunk_size_words")


def test_chunk_sections_tags_section_name():
    sections = {"Item 1A Risk Factors": "risk risk risk " * 50, "Item 7 MD&A": "revenue revenue " * 50}
    chunks = chunk_sections(sections, company="TEST", filing_type="10-K", source_id="t1")
    section_names = {c.section for c in chunks}
    assert section_names == set(sections.keys())
    print("PASS: chunk_sections correctly tags each chunk with its source section")


def test_embedder_normalizes_vectors():
    embedder = TfidfEmbedder()
    texts = ["revenue increased significantly", "net income declined this quarter", "risk factors include competition"]
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    import numpy as np
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"expected unit-norm vectors, got norms {norms}"
    print("PASS: embedder produces L2-normalized vectors")


def test_vector_store_returns_most_similar_first():
    embedder = TfidfEmbedder()
    from src.pipeline.chunker import Chunk

    texts = [
        "Apple reported strong revenue growth this quarter driven by services.",
        "The weather today is sunny with a light breeze.",
        "Total net revenue increased due to strong services performance.",
    ]
    chunks = [Chunk(text=t, company="TEST", filing_type="10-K", source_id="t1", chunk_index=i) for i, t in enumerate(texts)]

    embedder.fit(texts)
    vectors = embedder.embed(texts)
    store = NumpyVectorStore()
    store.add(vectors, chunks)

    query_vec = embedder.embed(["What drove revenue growth?"])[0]
    results = store.search(query_vec, k=2)

    top_texts = [r.chunk.text for r in results]
    assert "weather" not in top_texts[0].lower(), "irrelevant weather chunk should not rank first"
    assert results[0].score >= results[1].score, "results should be sorted by descending score"
    print(f"PASS: vector store ranks relevant chunks above irrelevant ones (top score={results[0].score:.3f})")


def test_end_to_end_rag_chain_with_mock_llm():
    from src.pipeline.chunker import Chunk

    texts = [
        "Total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6%.",
        "We face substantial competition in all markets in which we operate.",
    ]
    chunks = [Chunk(text=t, company="AAPL", filing_type="10-K", source_id="t1", chunk_index=i) for i, t in enumerate(texts)]

    retriever = Retriever.build(chunks)
    chain = RAGChain(retriever, MockClient(), k=2)
    result = chain.ask("What was total net revenue?")

    assert result.answer  # MockClient always returns something non-empty
    assert len(result.sources) == 2
    assert "394.2 billion" in result.prompt, "the relevant chunk's actual number should appear in the built prompt"
    print("PASS: end-to-end RAG chain retrieves relevant chunks and builds a grounded prompt")


def test_build_prompt_includes_citation_and_instructions():
    from src.pipeline.chunker import Chunk
    from src.pipeline.vector_store import SearchResult

    chunk = Chunk(text="Revenue grew 6%.", company="AAPL", filing_type="10-K", source_id="acc-123", chunk_index=0)
    prompt = build_prompt("How much did revenue grow?", [SearchResult(chunk=chunk, score=0.9)])

    assert "acc-123" in prompt, "prompt should include the source citation"
    assert "do not use any outside knowledge" in prompt.lower() or "only the excerpts" in prompt.lower()
    print("PASS: build_prompt grounds the model and embeds citation info")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests...\n")
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)
