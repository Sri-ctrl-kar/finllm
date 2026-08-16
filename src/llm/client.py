"""
client.py
------------------------------------------------------------
A thin, provider-agnostic wrapper around whichever LLM API you
call to generate the final answer from retrieved context.

Kept separate from qa_chain.py deliberately: the RAG logic (how to
build a prompt from retrieved chunks, how to ask for citations)
shouldn't care whether the underlying call is to Anthropic, OpenAI,
or a local model server — it should just call `llm.generate(prompt)`.
This is also exactly the seam where you'll later swap in YOUR
fine-tuned model (Phase 2) without touching any RAG code.
------------------------------------------------------------
"""

import os
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, max_tokens: int = 500) -> str: ...


class AnthropicClient:
    """Requires: pip install anthropic, and ANTHROPIC_API_KEY set in the environment."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Set ANTHROPIC_API_KEY (env var or constructor arg) to use AnthropicClient.")
        self._client = None

    def _load(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        self._load()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAICompatibleClient:
    """
    Works with OpenAI's API, or any OpenAI-compatible local server
    (e.g. Ollama, vLLM's OpenAI-compatible endpoint) by pointing
    `base_url` at it — this is also your Phase 3 hook, once you're
    serving your own fine-tuned/quantized model via vLLM.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed-for-local-servers")
        self.base_url = base_url
        self._client = None

    def _load(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        self._load()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class MockClient:
    """
    Zero-dependency, zero-API-key stand-in so the RAG chain is
    testable without any network call or API key — used in
    tests/test_pipeline.py. Just echoes back which chunks it 'saw'
    so tests can assert the prompt was built correctly, without
    asserting on real generated text (which would be non-deterministic).
    """

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        return f"[MockClient received a prompt of {len(prompt)} chars — wire up a real LLMClient for real answers]"
