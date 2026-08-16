"""
Generates training_data.jsonl from the sample filing — runs fully
offline, no GPU/API key needed. This is what you'd point
train_lora.py at (once run on a real filing corpus, not just the
one sample document — see README for scaling this up).

Usage: python -m src.finetune.build_dataset
"""

from ..cli import load_sample_sections
from ..pipeline.chunker import chunk_sections
from .dataset_builder import build_templated_examples, save_jsonl


def main():
    sections = load_sample_sections()
    chunks = chunk_sections(
        sections, company="AAPL", filing_type="10-K", source_id="sample-fy2025",
        chunk_size_words=120, overlap_words=30,
    )
    examples = build_templated_examples(chunks)

    print(f"Generated {len(examples)} training examples from {len(chunks)} chunks")
    for ex in examples:
        print(f"  Q: {ex.question}")

    save_jsonl(examples, "training_data.jsonl")
    print("\nSaved to training_data.jsonl")
    print("NOTE: this is templated data from ONE sample filing — a real training run")
    print("needs hundreds/thousands of examples across many real filings (edgar_fetcher.py)")
    print("plus ideally some LLM-assisted examples (build_llm_assisted_examples) for variety.")


if __name__ == "__main__":
    main()
