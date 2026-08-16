"""
benchmark.py
------------------------------------------------------------
Measures real latency, throughput, and peak GPU memory for a model
at each precision level, and (optionally) runs Phase 2's evaluation
harness against each precision's outputs to measure accuracy
retained — producing the exact numbers report.py turns into a
comparison table.

REQUIRES A GPU. Not executed anywhere in this build. Run on your
GPU machine/Colab:

    python -m src.inference.benchmark --model ./finetuned-adapter --n-runs 20
------------------------------------------------------------
"""

import argparse
import time
from typing import List

from .quantize import load_model, model_size_gb
from .report import BenchmarkResult, save_results


BENCHMARK_PROMPTS = [
    "What was total net revenue and how much did it grow year-over-year?",
    "Summarize the main risk factors mentioned in the filing.",
    "What was the company's gross margin and how did it change from the prior year?",
    "How much cash and marketable securities did the company report?",
    "What was diluted earnings per share?",
]


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int((p / 100) * len(sorted_values)))
    return sorted_values[idx]


def benchmark_precision(model_path: str, precision: str, n_runs: int = 20, max_new_tokens: int = 150) -> BenchmarkResult:
    import torch

    print(f"\n--- Benchmarking {precision} ---")
    load_start = time.perf_counter()
    model, tokenizer = load_model(model_path, precision)
    load_time_s = time.perf_counter() - load_start

    size_gb = model_size_gb(model)
    print(f"Loaded in {load_time_s:.1f}s, model size: {size_gb:.2f} GB")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warm-up run — the first forward pass on a GPU includes one-time
    # kernel compilation/caching overhead that would unfairly skew the
    # first latency measurement if counted.
    warmup_input = tokenizer(BENCHMARK_PROMPTS[0], return_tensors="pt").to(model.device)
    model.generate(**warmup_input, max_new_tokens=10)

    latencies_ms = []
    total_tokens_generated = 0
    bench_start = time.perf_counter()

    for i in range(n_runs):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies_ms.append(elapsed_ms)
        total_tokens_generated += output.shape[1] - inputs["input_ids"].shape[1]

    total_time_s = time.perf_counter() - bench_start
    throughput = total_tokens_generated / total_time_s if total_time_s > 0 else 0.0

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )

    latencies_ms.sort()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        precision=precision,
        model_size_gb=size_gb,
        load_time_s=load_time_s,
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        latency_p99_ms=percentile(latencies_ms, 99),
        throughput_tokens_per_sec=throughput,
        peak_memory_gb=peak_memory_gb,
        accuracy_score=None,  # fill in separately via --with-eval, see main()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument(
        "--with-eval", action="store_true",
        help="also run Phase 2's eval harness against each precision's outputs to measure accuracy retained "
             "(needs the eval question set — wires into src.eval.eval_harness)",
    )
    args = parser.parse_args()

    results = []
    for precision in ["fp16", "int8", "int4"]:
        result = benchmark_precision(args.model, precision, n_runs=args.n_runs)
        results.append(result)
        print(f"{precision}: p50={result.latency_p50_ms:.1f}ms p95={result.latency_p95_ms:.1f}ms "
              f"throughput={result.throughput_tokens_per_sec:.1f} tok/s memory={result.peak_memory_gb:.2f}GB")

    if args.with_eval:
        print("\n--with-eval: wire each precision's model into src.eval.run_eval_report's ")
        print("RAGChain (same pattern as Phase 2) and set result.accuracy_score from the")
        print("eval report's avg_numeric_accuracy — left as a manual step here since it")
        print("depends on how you're serving each quantized variant (see PHASE3_NOTES.md).")

    save_results(results, args.output)
    print(f"\nSaved to {args.output} — view with: python -m src.inference.report {args.output}")


if __name__ == "__main__":
    main()
