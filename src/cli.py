"""
cli.py
------------------------------------------------------------
Command-line entry point tying the whole Phase 1 pipeline together:

    sample filing text -> chunker -> embedder -> vector store
                                                       |
                                                       v
                                                 retriever -> RAG chain -> answer

Run it:
    python -m src.cli                        # uses the bundled sample filing + MockClient
    ANTHROPIC_API_KEY=sk-... python -m src.cli --llm anthropic   # real answers
------------------------------------------------------------
"""

import argparse
import os
import sys
from pathlib import Path

from .pipeline.chunker import chunk_sections
from .rag.retriever import Retriever
from .rag.qa_chain import RAGChain
from .llm.client import MockClient, AnthropicClient, OpenAICompatibleClient

SAMPLE_FILING_PATH = Path(__file__).parent / "data" / "sample_filing.txt"


def load_sample_sections() -> dict:
    """Splits the bundled sample filing into named sections by its 'Item N.' headers,
    mirroring how a real 10-K is structured (this is a simplified stand-in for the
    real EDGAR parsing you'd do with edgar_fetcher.py's output)."""
    raw = SAMPLE_FILING_PATH.read_text()
    sections = {}
    current_title = None
    current_lines = []

    for line in raw.splitlines():
        if line.strip().startswith("Item ") and "." in line[:12]:
            if current_title:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title:
        sections[current_title] = "\n".join(current_lines).strip()

    return sections


def build_llm(name: str):
    if name == "mock":
        return MockClient()
    if name == "anthropic":
        return AnthropicClient()
    if name == "openai":
        return OpenAICompatibleClient()
    raise ValueError(f"Unknown --llm option: {name}")


def main():
    parser = argparse.ArgumentParser(description="FinLLM-RAG Phase 1 CLI")
    parser.add_argument("--llm", default="mock", choices=["mock", "anthropic", "openai"],
                         help="which LLM backend to use for the final answer (default: mock, no API key needed)")
    parser.add_argument("--k", type=int, default=3, help="number of chunks to retrieve per question")
    parser.add_argument("question", nargs="?", help="question to ask; omit for an interactive prompt loop")
    args = parser.parse_args()

    print("Loading sample filing and building index...")
    sections = load_sample_sections()
    chunks = chunk_sections(
        sections,
        company="AAPL",
        filing_type="10-K",
        source_id="sample-fy2025",
        chunk_size_words=120,
        overlap_words=30,
    )
    print(f"  {len(sections)} sections -> {len(chunks)} chunks")

    retriever = Retriever.build(chunks)
    llm = build_llm(args.llm)
    chain = RAGChain(retriever, llm, k=args.k)
    print(f"Ready. (LLM backend: {args.llm})\n")

    if args.question:
        ask_and_print(chain, args.question)
        return

    print("Type a question (or 'quit'):")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit"):
            break
        ask_and_print(chain, question)


def ask_and_print(chain: RAGChain, question: str):
    result = chain.ask(question)
    print(f"\nQ: {question}")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for s in result.sources:
        print(f"  [{s.score:.3f}] {s.chunk.citation}")
    print()


if __name__ == "__main__":
    sys.exit(main())
