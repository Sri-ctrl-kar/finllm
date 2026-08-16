"""
test_inference_report.py
------------------------------------------------------------
Tests the trade-off analysis logic with synthetic benchmark data.
Fully offline — doesn't need a GPU or real model, since it only
tests the MATH on top of benchmark numbers, not the benchmarking
itself (that lives in benchmark.py, run on real hardware).
------------------------------------------------------------
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.report import BenchmarkResult, compute_tradeoffs


def make_result(precision, latency_p95_ms, peak_memory_gb, accuracy_score=None):
    return BenchmarkResult(
        precision=precision, model_size_gb=1.0, load_time_s=1.0,
        latency_p50_ms=latency_p95_ms * 0.7, latency_p95_ms=latency_p95_ms, latency_p99_ms=latency_p95_ms * 1.2,
        throughput_tokens_per_sec=1000 / latency_p95_ms, peak_memory_gb=peak_memory_gb, accuracy_score=accuracy_score,
    )


def test_baseline_has_1x_speedup_and_0_memory_reduction():
    results = [make_result("fp16", 200, 16.0), make_result("int8", 100, 8.0)]
    tradeoffs = compute_tradeoffs(results, baseline_precision="fp16")
    baseline_row = next(r for r in tradeoffs if r.precision == "fp16")
    assert baseline_row.speedup_vs_baseline == 1.0
    assert baseline_row.memory_reduction_pct == 0.0
    print("PASS: baseline precision shows 1.0x speedup and 0% memory reduction relative to itself")


def test_speedup_computed_correctly():
    results = [make_result("fp16", 200, 16.0), make_result("int4", 50, 4.0)]
    tradeoffs = compute_tradeoffs(results, baseline_precision="fp16")
    int4_row = next(r for r in tradeoffs if r.precision == "int4")
    assert int4_row.speedup_vs_baseline == 4.0, f"expected 4.0x speedup (200/50), got {int4_row.speedup_vs_baseline}"
    print("PASS: speedup is computed as baseline_latency / this_precision_latency")


def test_memory_reduction_computed_correctly():
    results = [make_result("fp16", 200, 16.0), make_result("int8", 100, 8.0)]
    tradeoffs = compute_tradeoffs(results, baseline_precision="fp16")
    int8_row = next(r for r in tradeoffs if r.precision == "int8")
    assert int8_row.memory_reduction_pct == 50.0, f"expected 50% reduction, got {int8_row.memory_reduction_pct}"
    print("PASS: memory reduction is computed correctly (halved memory = 50% reduction)")


def test_accuracy_retained_none_when_scores_missing():
    """If accuracy scores weren't provided (e.g. eval harness hasn't been
    run against this precision yet), the report should say so, not silently
    show a fabricated number."""
    results = [make_result("fp16", 200, 16.0, accuracy_score=None), make_result("int4", 50, 4.0, accuracy_score=None)]
    tradeoffs = compute_tradeoffs(results, baseline_precision="fp16")
    assert all(r.accuracy_retained_pct is None for r in tradeoffs)
    print("PASS: accuracy_retained_pct is None (not fabricated) when accuracy scores weren't measured")


def test_accuracy_retained_computed_when_scores_present():
    results = [make_result("fp16", 200, 16.0, accuracy_score=0.94), make_result("int4", 50, 4.0, accuracy_score=0.89)]
    tradeoffs = compute_tradeoffs(results, baseline_precision="fp16")
    int4_row = next(r for r in tradeoffs if r.precision == "int4")
    expected = (0.89 / 0.94) * 100
    assert abs(int4_row.accuracy_retained_pct - expected) < 0.01
    print(f"PASS: accuracy_retained_pct correctly computed ({int4_row.accuracy_retained_pct:.1f}%)")


def test_missing_baseline_raises_clear_error():
    results = [make_result("int8", 100, 8.0)]
    try:
        compute_tradeoffs(results, baseline_precision="fp16")
        raise AssertionError("expected ValueError when baseline precision isn't in results")
    except ValueError as e:
        assert "fp16" in str(e)
        print("PASS: missing baseline precision raises a clear, specific error")


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
