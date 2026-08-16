"""
dataset_builder.py
------------------------------------------------------------
Turns chunks (from Phase 1's chunker) into a supervised fine-tuning
dataset: (instruction, context, question, answer) examples.

TWO WAYS TO GENERATE THE (question, answer) PAIRS:

  1. Template-based (implemented below, runs offline, zero cost):
     pulls numeric facts out of chunk text with regex (e.g. "revenue
     was $394.2 billion") and turns each into a templated Q&A pair
     ("What was total net revenue?" -> "$394.2 billion"). Cheap,
     deterministic, but limited to numbers-with-a-label patterns —
     it won't generate open-ended reasoning questions.

  2. LLM-assisted (sketched below, needs an API key): ask a strong
     LLM to read a chunk and generate 2-3 realistic analyst questions
     + grounded answers. Much higher quality and variety, but costs
     API calls and needs a human spot-check pass (an LLM generating
     its own training data can introduce its own errors/hallucinations
     into the dataset if unchecked).

Realistic plan: use method 1 to bootstrap a first dataset fast (this
is what's runnable today), use method 2 to expand/diversify it once
you're working with API budget, and always hand-review a sample
before training on it — bad training data teaches the model bad habits.
------------------------------------------------------------
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import List

from ..pipeline.chunker import Chunk


@dataclass
class TrainingExample:
    instruction: str
    context: str
    question: str
    answer: str
    source_citation: str

    def to_prompt_completion(self) -> dict:
        """Format matching what train_lora.py expects: a single text
        field the model learns to continue, structured so the model
        learns 'given context + question, answer only from context.'"""
        prompt = (
            f"{self.instruction}\n\n"
            f"Context:\n{self.context}\n\n"
            f"Question: {self.question}\n"
            f"Answer:"
        )
        return {"prompt": prompt, "completion": f" {self.answer}"}


INSTRUCTION = (
    "You are a financial research assistant. Answer the question using only "
    "the context provided. If the context doesn't contain the answer, say so."
)

# Matches patterns like: "revenue was $394.2 billion", "increased 6%",
# "net income of $101.8 billion". Deliberately conservative (specific
# label words) to avoid generating garbage questions from unrelated numbers.
NUMERIC_FACT_PATTERN = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z '&-]{2,40}?)\s+"
    r"(?:was|were|of|increased|decreased|grew|declined|totaled)\s+"
    r"(?:to\s+|by\s+)?"
    r"(?P<value>\$?[\d,]+(?:\.\d+)?\s?(?:billion|million|thousand|%)?)",
    re.IGNORECASE,
)


# Labels that are grammatically valid matches but semantically useless
# as a question subject (found via testing against the real sample
# filing — "What was an increase?" / "What was which?" are exactly
# the garbage this filters out).
BAD_LABEL_LEADING_WORDS = {
    "a", "an", "the", "this", "that", "which", "it", "its", "our", "their",
    "and", "or", "to", "of", "in", "on", "for",
}


def extract_numeric_facts(text: str) -> List[dict]:
    """Pull (label, value) pairs out of chunk text via regex. Returns
    a list of {"label": ..., "value": ..., "sentence": ...} dicts."""
    facts = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for match in NUMERIC_FACT_PATTERN.finditer(sentence):
            label = match.group("label").strip()
            value = match.group("value").strip()

            if len(label.split()) > 6:
                continue  # likely a bad match spanning too much text
            first_word = label.split()[0].lower() if label.split() else ""
            if first_word in BAD_LABEL_LEADING_WORDS:
                continue  # "an increase", "which" etc. — not a real subject

            facts.append({"label": label, "value": value, "sentence": sentence.strip()})
    return facts


def build_templated_examples(chunks: List[Chunk]) -> List[TrainingExample]:
    """Method 1 from the module docstring: regex-extracted facts -> templated Q&A."""
    examples = []
    for chunk in chunks:
        facts = extract_numeric_facts(chunk.text)
        for fact in facts:
            question = f"What was {fact['label'].lower()}?"
            examples.append(
                TrainingExample(
                    instruction=INSTRUCTION,
                    context=chunk.text,
                    question=question,
                    answer=f"{fact['value']}, per: \"{fact['sentence']}\"",
                    source_citation=chunk.citation,
                )
            )
    return examples


LLM_GENERATION_PROMPT_TEMPLATE = """You are generating training data for a financial \
Q&A assistant. Given the filing excerpt below, write {n} realistic questions a financial \
analyst might ask, each with a precise answer found ONLY in the excerpt. Return JSON: \
a list of {{"question": ..., "answer": ...}} objects. Do not invent facts not in the excerpt.

Excerpt:
{chunk_text}
"""


def build_llm_assisted_examples(chunks: List[Chunk], llm, n_per_chunk: int = 2) -> List[TrainingExample]:
    """
    Method 2 from the module docstring. `llm` is any object with a
    `.generate(prompt) -> str` method (see src/llm/client.py).

    NOTE: this calls the LLM once per chunk — real API cost. Expects
    the LLM to return valid JSON; in practice you'll want to wrap
    this in retry/validation logic before relying on it at scale.
    This is sketched, not hardened — treat it as a starting point.
    """
    examples = []
    for chunk in chunks:
        prompt = LLM_GENERATION_PROMPT_TEMPLATE.format(n=n_per_chunk, chunk_text=chunk.text)
        raw_response = llm.generate(prompt, max_tokens=500)
        try:
            qa_pairs = json.loads(raw_response)
        except json.JSONDecodeError:
            continue  # skip chunks where the LLM didn't return clean JSON

        for pair in qa_pairs:
            examples.append(
                TrainingExample(
                    instruction=INSTRUCTION,
                    context=chunk.text,
                    question=pair["question"],
                    answer=pair["answer"],
                    source_citation=chunk.citation,
                )
            )
    return examples


def save_jsonl(examples: List[TrainingExample], path: str) -> None:
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_prompt_completion()) + "\n")
