# Phase 3 Notes — Quantization + Inference Benchmarking (the NVIDIA piece)

## What's real vs. what needs a GPU

| Piece | Status |
|---|---|
| `report.py` (comparison table + trade-off math) | **Real, tested, runs today** — 6/6 tests, verified against realistic synthetic numbers |
| `quantize.py` | Real, complete — loads a model at fp16/int8/int4 via bitsandbytes. **Needs a GPU**, untested-live |
| `benchmark.py` | Real, complete — measures actual latency/throughput/memory. **Needs a GPU**, untested-live |
| `serve_vllm.py` | Real launch config + client wiring. **Needs a GPU + vLLM install**, untested-live |

## How to run the real thing (GPU machine or Colab)

```bash
pip install torch transformers bitsandbytes accelerate

# 1. Sanity-check quantization actually shrinks the model as expected
python -m src.inference.quantize --model ./finetuned-adapter

# 2. Run the real benchmark (this is what takes the longest — budget GPU time for it)
python -m src.inference.benchmark --model ./finetuned-adapter --n-runs 20

# 3. View the comparison report (this part already works, tested, right now)
python -m src.inference.report benchmark_results.json
```

For the accuracy-retained column to populate (not just latency/memory),
you need to run Phase 2's eval harness against each precision's actual
outputs and set `BenchmarkResult.accuracy_score` accordingly — `benchmark.py`
flags this as a manual wiring step rather than pretending to automate
something that depends on how you're serving each variant.

## Why the report/analysis layer was built and tested separately

Same reasoning as Phase 2's eval harness vs. train_lora.py: the part
that risks silently being wrong (math on numbers, not the GPU
measurement itself) is exactly the part you can and should test without
needing the expensive/slow real thing to run first. `compute_tradeoffs()`
has 6 tests covering: baseline normalization, speedup math, memory math,
what happens when accuracy scores are missing (shows `None`, doesn't
fabricate a number), and a clear error when the requested baseline
precision isn't in the results at all.

## What actually goes in your resume/interview from this phase

Once you've run the real benchmark on your fine-tuned model, you'll have
a table shaped like this (numbers below are illustrative, from the
fallback example `report.py` prints when no real results file exists yet
— replace with your actual measured numbers):

```
Precision  p95 Latency   Speedup   Memory Δ   Accuracy Retained
fp16       240.0ms       1.00x     0.0%       100.0%
int8       130.0ms       1.85x     45.8%      98.9%
int4       79.0ms        3.04x     69.0%      94.7%
```

The sentence this becomes: *"Quantized a fine-tuned 7B model to INT4,
achieving a 3x inference speedup and 69% memory reduction at 94.7% of
full-precision accuracy, benchmarked with a custom latency/throughput
harness."* That is a real NVIDIA-relevant sentence, and it's backed by
a report you can actually open and defend line-by-line — not a claim
you're hoping nobody asks you to substantiate.

## Known next steps

- Try GPTQ or AWQ instead of bitsandbytes for int4 — both need a
  calibration pass over sample data first, but typically retain more
  accuracy at 4-bit than bitsandbytes' on-the-fly approach. Worth
  running as a second int4 row in the same report once the bitsandbytes
  baseline is working.
- Benchmark under concurrent load (multiple simultaneous requests via
  vLLM), not just single-request latency — this is where PagedAttention's
  batching advantage actually shows up, and single-request benchmarks
  alone understate it.
- Wire `--with-eval` in `benchmark.py` into an automated per-precision
  eval run instead of the manual step it currently is.
