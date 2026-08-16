"""
Runs a small evaluation set through the full Phase 1 RAG pipeline
and prints the Phase 2 eval report — the actual artifact to show in
an interview. Runs fully offline with MockClient by default; pass
--llm anthropic (with ANTHROPIC_API_KEY set) for a report against
real generated answers.

Usage: python -m src.eval.run_eval_report [--llm mock|anthropic|openai]
"""

import argparse

from ..cli import load_sample_sections, build_llm
from ..pipeline.chunker import chunk_sections
from ..rag.retriever import Retriever
from ..rag.qa_chain import RAGChain
from .eval_harness import EvalCase, run_eval


EVAL_QUESTIONS = [
    EvalCase("What was total net revenue and how much did it grow?", expected_value="394.2"),
    EvalCase("What was the gross margin for the year?", expected_value="46.2"),
    EvalCase("What was net income?", expected_value="101.8"),
    EvalCase("What were the main risk factors mentioned?"),
    EvalCase("What was diluted earnings per share?", expected_value="6.42"),
    EvalCase("How much cash and marketable securities did the company have?", expected_value="162.1"),
    EvalCase("What was the company's total revenue from the moon colony division?"),  # trap: not in the filing at all
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="mock", choices=["mock", "anthropic", "openai"])
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    sections = load_sample_sections()
    chunks = chunk_sections(
        sections, company="AAPL", filing_type="10-K", source_id="sample-fy2025",
        chunk_size_words=120, overlap_words=30,
    )
    retriever = Retriever.build(chunks)
    llm = build_llm(args.llm)
    chain = RAGChain(retriever, llm, k=args.k)

    report = run_eval(chain, EVAL_QUESTIONS)
    report.print_summary()

    if args.llm == "mock":
        print("\nNOTE: --llm mock produces placeholder answers, so accuracy numbers above")
        print("are meaningless — run with --llm anthropic (and ANTHROPIC_API_KEY set) for")
        print("a real report against real generated answers.")


if __name__ == "__main__":
    main()
