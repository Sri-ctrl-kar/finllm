"""
report.py
------------------------------------------------------------
Takes raw benchmark numbers (produced by benchmark.py, on a GPU
machine) and turns them into the actual artifact you'd show in an
NVIDIA interview: a comparison table across precision levels, with
computed speedup/memory-savings/accuracy-retained figures — not
just raw numbers, but the RATIOS that make the trade-off legible.

Deliberately separated from benchmark.py (which needs a GPU to
produce real numbers) so this analysis logic can be built and
tested with synthetic data today, and trusted once real numbers
are dropped in — same split as eval_harness.py vs. train_lora.py
in Phase 2.
------------------------------------------------------------
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class BenchmarkResult:
    precision: str              # "fp16", "int8", "int4"
    model_size_gb: float
    load_time_s: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_tokens_per_sec: float
    peak_memory_gb: float
    accuracy_score: Optional[float] = None  # e.g. from Phase 2's eval harness, run against this precision's outputs

    def to_dict(self) -> dict:
        return asdict(self)


def load_results(path: str) -> List[BenchmarkResult]:
    with open(path) as f:
        raw = json.load(f)
    return [BenchmarkResult(**r) for r in raw]


def save_results(results: List[BenchmarkResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


@dataclass
class TradeoffRow:
    precision: str
    latency_p95_ms: float
    speedup_vs_baseline: float       # e.g. 2.3x faster than fp16
    memory_reduction_pct: float      # e.g. 74% less memory than fp16
    accuracy_retained_pct: Optional[float]  # e.g. 97% of fp16's accuracy score


def compute_tradeoffs(results: List[BenchmarkResult], baseline_precision: str = "fp16") -> List[TradeoffRow]:
    """
    Computes each precision's numbers RELATIVE to the baseline (normally
    full fp16 precision) — this is what actually makes a benchmark
    legible. "int4 has 45ms p95 latency" means nothing on its own;
    "int4 is 3.1x faster than fp16 at 97% of its accuracy" is the
    sentence that gets you taken seriously.
    """
    baseline = next((r for r in results if r.precision == baseline_precision), None)
    if baseline is None:
        raise ValueError(f"No result found for baseline precision '{baseline_precision}'")

    rows = []
    for r in results:
        speedup = baseline.latency_p95_ms / r.latency_p95_ms if r.latency_p95_ms > 0 else float("inf")
        memory_reduction = (1 - r.peak_memory_gb / baseline.peak_memory_gb) * 100 if baseline.peak_memory_gb > 0 else 0.0

        accuracy_retained = None
        if r.accuracy_score is not None and baseline.accuracy_score is not None and baseline.accuracy_score > 0:
            accuracy_retained = (r.accuracy_score / baseline.accuracy_score) * 100

        rows.append(
            TradeoffRow(
                precision=r.precision,
                latency_p95_ms=r.latency_p95_ms,
                speedup_vs_baseline=speedup,
                memory_reduction_pct=memory_reduction,
                accuracy_retained_pct=accuracy_retained,
            )
        )
    return rows


def print_report(results: List[BenchmarkResult], baseline_precision: str = "fp16") -> None:
    tradeoffs = compute_tradeoffs(results, baseline_precision)

    print("=" * 78)
    print("Inference Benchmark Report")
    print("=" * 78)
    header = f"{'Precision':<10} {'p95 Latency':<14} {'Speedup':<10} {'Memory Δ':<12} {'Accuracy Retained':<18}"
    print(header)
    print("-" * 78)
    for row in tradeoffs:
        acc_str = f"{row.accuracy_retained_pct:.1f}%" if row.accuracy_retained_pct is not None else "n/a"
        print(
            f"{row.precision:<10} {row.latency_p95_ms:>8.1f}ms    "
            f"{row.speedup_vs_baseline:>6.2f}x   "
            f"{row.memory_reduction_pct:>7.1f}%     "
            f"{acc_str:<18}"
        )
    print("=" * 78)

    # Flag the standout trade-off: best speedup that retains most accuracy
    scored = [r for r in tradeoffs if r.precision != baseline_precision]
    if scored:
        best = max(scored, key=lambda r: r.speedup_vs_baseline * ((r.accuracy_retained_pct or 100) / 100))
        print(f"\nBest trade-off: {best.precision} — {best.speedup_vs_baseline:.2f}x faster, "
              f"{best.memory_reduction_pct:.0f}% less memory"
              + (f", {best.accuracy_retained_pct:.1f}% accuracy retained" if best.accuracy_retained_pct else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print a benchmark comparison report from a saved results file")
    parser.add_argument("results_file", nargs="?", default="benchmark_results.json",
                         help="path to a JSON file saved by benchmark.py's save_results()")
    parser.add_argument("--baseline", default="fp16")
    args = parser.parse_args()

    try:
        results = load_results(args.results_file)
    except FileNotFoundError:
        print(f"No results file found at '{args.results_file}'.")
        print("Run this on a GPU machine first:  python -m src.inference.benchmark --model <path>")
        print("\nShowing an example report with synthetic numbers instead, so you can see the format:\n")
        results = [
            BenchmarkResult("fp16", 14.5, 12.3, 180, 240, 290, 42, 16.8, accuracy_score=0.94),
            BenchmarkResult("int8", 7.3, 9.1, 95, 130, 160, 78, 9.1, accuracy_score=0.93),
            BenchmarkResult("int4", 3.8, 6.4, 58, 79, 98, 118, 5.2, accuracy_score=0.89),
        ]

    print_report(results, baseline_precision=args.baseline)
