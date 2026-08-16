# FinLLM-RAG — Phase 1: Financial Document RAG Pipeline

Phase 1 of a 2-year capstone: a retrieval-augmented question-answering
system over SEC filings, built to be extended into fine-tuning (Phase 2),
inference optimization/quantization (Phase 3), and a full-stack deployed
app (Phase 4).

**Read [`docs/PHASE1_NOTES.md`](docs/PHASE1_NOTES.md) first** — it explains
exactly what's a real final implementation vs. a deliberate offline-friendly
placeholder, and the exact steps to upgrade each piece.

## Quick start (runs today, zero API keys, zero internet)

```bash
pip install -r requirements.txt   # numpy, scikit-learn, requests, pytest

python -m pytest tests/ -v        # or: python tests/test_pipeline.py
# 8/8 tests should pass

python -m src.cli "What was total net revenue and how much did it grow?"
```

This builds an index from a bundled sample 10-K excerpt and answers using
`MockClient` (no real LLM call — just proves the retrieval pipeline works
and shows you exactly what prompt would be sent).

## Get real generated answers

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python -m src.cli --llm anthropic "What was total net revenue and how much did it grow?"
```

## Fetch a real filing (needs internet)

```bash
python -m src.data.edgar_fetcher
# fetches Apple's latest 10-K from SEC EDGAR and prints a preview
```

Then feed its `text` into `chunk_sections`/`chunk_text` the same way
`cli.py` does with the sample filing.

## How it fits together

```
edgar_fetcher.py  ──▶  chunker.py  ──▶  embedder.py  ──▶  vector_store.py
 (real filing text)     (chunks)         (vectors)          (search index)
                                                                   │
                                                                   ▼
                                                            retriever.py
                                                                   │
                                                                   ▼
                                                 qa_chain.py  ──▶  client.py
                                              (builds grounded        (calls
                                               prompt + citations)     the LLM)
```

## Project structure

```
src/
  data/
    edgar_fetcher.py     Real SEC EDGAR API client (ticker -> CIK -> filing text)
    sample_filing.txt    Bundled offline sample for testing without internet
  pipeline/
    chunker.py             Splits filing text into overlapping, cited chunks
    embedder.py              TfidfEmbedder (default) + DenseEmbedder (upgrade path)
    vector_store.py            NumPy brute-force cosine search
  rag/
    retriever.py           Embedder + vector store -> retrieve(query, k)
    qa_chain.py               Builds grounded prompt, calls LLM, returns cited answer
  llm/
    client.py             AnthropicClient / OpenAICompatibleClient / MockClient
  cli.py                Ties it all together — the runnable entry point
tests/
  test_pipeline.py     8 offline tests covering every module
docs/
  PHASE1_NOTES.md     What's real vs. placeholder, and the exact upgrade steps
```

## Phase 2 — Fine-tuning + Evaluation Harness

**Read [`docs/PHASE2_NOTES.md`](docs/PHASE2_NOTES.md)** — it documents two real bugs
caught by actually testing this code (a garbage-question generator, and a
hallucination detector that couldn't tell hallucinations apart from honest
answers) and exactly how they were fixed and verified.

```bash
python -m src.finetune.build_dataset      # generates training_data.jsonl (offline, tested)
python -m pytest tests/test_eval_harness.py -v   # 11/11 tests, fully offline
python -m src.eval.run_eval_report        # the actual eval report (mock answers by default)
python -m src.eval.run_eval_report --llm anthropic   # real report, needs ANTHROPIC_API_KEY
```

Fine-tuning itself (`src/finetune/train_lora.py`) needs a GPU — run it in
Colab or on a GPU machine, not in a plain CPU environment.

```
src/finetune/
  dataset_builder.py   Chunks -> Q&A training examples (template-based + LLM-assisted)
  build_dataset.py       Runnable: generates training_data.jsonl
  train_lora.py          LoRA/QLoRA fine-tuning script (needs GPU)
src/eval/
  numeric_accuracy.py   Checks: is every number in the answer actually in the source?
  hallucination_checker.py  Checks: refusal detection + source-overlap grounding score
  eval_harness.py          Orchestrates both checks across a test set, prints a report
  run_eval_report.py        Runnable: generates the actual report against Phase 1's pipeline
```

## Phase 3 — Quantization + Inference Benchmarking (NVIDIA)

**Read [`docs/PHASE3_NOTES.md`](docs/PHASE3_NOTES.md)** for what's real vs. GPU-only, and the
exact resume sentence this phase earns you once you've run it.

```bash
python -m pytest tests/test_inference_report.py -v   # 6/6 tests, fully offline
python -m src.inference.report                        # comparison report (synthetic fallback if no real results yet)
```

On a GPU machine / Colab:
```bash
pip install torch transformers bitsandbytes accelerate
python -m src.inference.quantize --model ./finetuned-adapter
python -m src.inference.benchmark --model ./finetuned-adapter --n-runs 20
python -m src.inference.report benchmark_results.json
```

```
src/inference/
  quantize.py     Loads a model at fp16/int8/int4 via bitsandbytes (needs GPU)
  benchmark.py      Measures real latency/throughput/memory per precision (needs GPU)
  report.py           Comparison table + speedup/memory/accuracy trade-off math (tested, offline)
  serve_vllm.py         vLLM serving config + wiring back into Phase 1's LLMClient interface
```

## Phase 4 — Full-Stack App (dashboard + monitoring)

**Read [`docs/PHASE4_NOTES.md`](docs/PHASE4_NOTES.md)** — two services (Python RAG core +
Express app), and exactly what was proven working live vs. what needs `npm install`.

```bash
# Terminal 1, from finllm-rag/
python -m src.service.rag_server

# Terminal 2, from finllm-rag/app/
npm install && npm start
# open http://localhost:3002/dashboard.html
```

```
src/service/
  rag_server.py       Stdlib-only HTTP wrapper around the Phase 1-3 pipeline (no deps needed, tested live)
app/
  src/server.js           Express app entry point
  src/metricsStore.js       Request logging + aggregation (tested directly, real assertions)
  src/routes/ask.js           Proxies to the RAG service, logs every request
  src/routes/metrics.js         Summary/recent endpoints for the dashboard
  public/dashboard.html           Zero-dependency live dashboard (plain HTML/JS)
```

## Phase 5 — Deployment + Technical Report

**Read [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)** — the polished writeup with
every real number from Phases 1-4, ready to link on your resume/LinkedIn as-is.

**Read [`docs/PHASE5_DEPLOYMENT.md`](docs/PHASE5_DEPLOYMENT.md)** for step-by-step Render
deployment (one `render.yaml` blueprint deploys both services), including the one manual
step Render requires (wiring the two services' URLs together) and honest free-tier
caveats (cold-start spin-down) worth knowing before you demo it live.

## Full roadmap

- ✅ Phase 1 — Financial data pipeline + baseline RAG
- ✅ Phase 2 — Fine-tuning + hallucination/numeric-accuracy evaluation harness
- ✅ Phase 3 — Quantization + inference benchmarking
- ✅ Phase 4 — Full-stack app: Express + monitoring dashboard
- ✅ Phase 5 — Deployment + technical report

**Total: 25/25 automated tests passing across all phases**, plus live end-to-end
verification of every network-facing component (see each phase's NOTES.md for exactly
what was run and confirmed working vs. what needs a GPU you'll supply).

## What to say about this in an interview

"I built the retrieval and generation pipeline first with a fully offline
baseline (TF-IDF + brute-force search) so I could test correctness end-to-end
before adding external dependencies — then designed every module around a
shared interface so upgrading to real neural embeddings, FAISS, and a real
LLM is a one-line swap, not a rewrite." That's a real engineering decision,
not a cop-out — and it's demonstrated by the fact that `tests/test_pipeline.py`
passes with zero network access at all.
