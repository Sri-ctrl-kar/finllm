"""
serve_vllm.py
------------------------------------------------------------
Serves the fine-tuned/quantized model via vLLM's OpenAI-compatible
API — the production-serving layer that makes throughput/latency
gains from quantization actually usable behind a real endpoint,
instead of only measurable in a benchmark script.

REQUIRES A GPU + `pip install vllm`. Not executed anywhere in this
build. The launch command below is what you'd run on your GPU
machine; this file's `example_client_usage()` function shows how
Phase 1's existing `OpenAICompatibleClient` (src/llm/client.py)
plugs straight into it — this is the "one-line swap" moment
referenced in Phase 1's docs, now actually happening.

WHY vLLM SPECIFICALLY: vLLM's key contribution is PagedAttention —
it manages the KV-cache (the per-token memory each request needs
during generation) in fixed-size blocks instead of one contiguous
allocation per request, which lets it pack many concurrent requests
into GPU memory far more efficiently than naive batching. This is
the difference between "serves 1 request well" and "serves dozens
of concurrent requests well" — the latter is what actually matters
for a deployed system, not just a benchmark script.

Launch command (run on your GPU machine, not here):

    pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
        --model ./finetuned-adapter \\
        --quantization bitsandbytes \\
        --max-model-len 4096 \\
        --port 8000

Key flags worth understanding (not just copying):
    --max-num-batched-tokens   caps how many tokens across ALL concurrent
                                requests get processed per forward pass —
                                the main lever for the throughput/latency
                                trade-off under concurrent load.
    --gpu-memory-utilization   how much of the GPU's memory vLLM is allowed
                                to claim for the KV-cache pool up front
                                (default 0.9) — higher means more concurrent
                                requests fit, at the risk of OOM if set too high.
------------------------------------------------------------
"""


def example_client_usage():
    """
    Once the vLLM server above is running, Phase 1's OpenAICompatibleClient
    (src/llm/client.py) talks to it with zero new code — just point
    base_url at the local server instead of OpenAI's:
    """
    from ..llm.client import OpenAICompatibleClient
    from ..rag.qa_chain import RAGChain
    from ..rag.retriever import Retriever

    llm = OpenAICompatibleClient(
        model="finetuned-adapter",       # matches whatever --served-model-name vLLM was given, if set
        api_key="not-needed-for-local-servers",
        base_url="http://localhost:8000/v1",
    )

    # From here it's identical to every other phase: build a retriever
    # over your chunks (Phase 1), wrap it in a RAGChain with this LLM,
    # and Phase 2's eval harness can run against it unchanged — proving
    # the whole point of keeping LLMClient as a swappable interface
    # since Phase 1: retriever = Retriever.build(chunks); chain = RAGChain(retriever, llm)
    return llm


if __name__ == "__main__":
    print(__doc__)
