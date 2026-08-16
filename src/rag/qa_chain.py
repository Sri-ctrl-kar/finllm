"""
qa_chain.py
------------------------------------------------------------
The actual "ask a question, get a grounded, cited answer" logic.

The core idea, and the whole point of RAG: instead of asking the
LLM "what was Apple's revenue?" and hoping it remembers correctly
(it might not — this is exactly how hallucination happens), we:

  1. Retrieve the chunks of real filing text most relevant to the question
  2. Put those chunks INTO the prompt, verbatim
  3. Instruct the model to answer ONLY from what's in front of it,
     and to say "I don't know" if the retrieved chunks don't contain
     the answer

This doesn't make hallucination impossible — the model can still
misread a number in the context. But it changes the failure mode
from "invented from nowhere" to "misread something that's sitting
right there," which is a MUCH easier thing to catch, and it's the
basis for the evaluation harness you'll build in Phase 2 (comparing
the model's stated answer against what the cited chunk actually says).
------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import List

from ..llm.client import LLMClient
from ..pipeline.vector_store import SearchResult
from .retriever import Retriever


SYSTEM_INSTRUCTIONS = """You are a financial research assistant. Answer the user's question \
using ONLY the excerpts provided below, which are taken directly from SEC filings. \
Do not use any outside knowledge. If the excerpts don't contain enough information to \
answer confidently, say so explicitly instead of guessing. When you state a number or \
fact, mention which excerpt it came from (e.g. "per Excerpt 2")."""


@dataclass
class RAGAnswer:
    answer: str
    sources: List[SearchResult]
    prompt: str  # kept for debugging/inspection — see how the prompt was actually built


def build_prompt(question: str, sources: List[SearchResult]) -> str:
    excerpt_blocks = []
    for i, result in enumerate(sources, start=1):
        excerpt_blocks.append(
            f"--- Excerpt {i} (source: {result.chunk.citation}, relevance score: {result.score:.3f}) ---\n"
            f"{result.chunk.text}\n"
        )
    excerpts_text = "\n".join(excerpt_blocks)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{excerpts_text}\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


class RAGChain:
    def __init__(self, retriever: Retriever, llm: LLMClient, k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def ask(self, question: str) -> RAGAnswer:
        sources = self.retriever.retrieve(question, k=self.k)
        prompt = build_prompt(question, sources)
        answer_text = self.llm.generate(prompt)
        return RAGAnswer(answer=answer_text, sources=sources, prompt=prompt)
