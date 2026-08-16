# Phase 2 Notes — Fine-Tuning + Evaluation Harness

## What's real vs. what needs a GPU

| Piece | Status |
|---|---|
| `dataset_builder.py` (template-based) | Real, tested, runs today — `python -m src.finetune.build_dataset` |
| `dataset_builder.py` (LLM-assisted) | Real logic, needs an API key — sketched, not hardened (see its docstring) |
| `eval_harness.py`, `numeric_accuracy.py`, `hallucination_checker.py` | **Real, fully tested — this is the actual JPMorgan-relevant artifact** |
| `train_lora.py` | Real, complete LoRA/QLoRA training script — **needs a GPU to run**, untested-live in this build |

## Two real bugs this caught, worth knowing about (and mentioning in an interview)

**1. Garbage training questions.** The first version of the regex-based
Q&A generator produced questions like "What was an increase?" and "What
was which?" — it was grabbing filler words as the subject of the
question because the regex matched grammatically valid but semantically
empty phrases. Fixed by filtering out labels that start with common
function words (`a`, `an`, `which`, `the`, etc.) — see
`BAD_LABEL_LEADING_WORDS` in `dataset_builder.py`, and the regression
test in `tests/test_eval_harness.py` that would fail if this filter
were removed.

**2. A hallucination detector that couldn't tell hallucinations apart
from honest answers.** The first version of `numeric_accuracy.py`
extracted every number in an answer, including citation self-references
like "per Excerpt 1" — so a fully honest, correctly-grounded answer that
cited its source got flagged as containing an "ungrounded number" (the
`1` from `Excerpt 1`), at the exact same rate as an answer with
genuinely invented numbers. This defeats the entire purpose of the
check. Fixed by stripping citation references (`Excerpt N`, `chunk #N`)
before extracting numbers — verified by re-running the same good-vs-bad
comparison and confirming the good answer now scores 100% while the
hallucinating one scores 0%.

Both bugs were caught by actually running the code against real (if
synthetic) test cases before trusting it — not by inspection. That's
the habit worth carrying into Phase 3 and beyond.

## How to run the real training pipeline (needs a GPU — Colab free tier works)

```bash
pip install torch transformers peft bitsandbytes accelerate datasets

# 1. Generate training data (runs offline, already verified working)
python -m src.finetune.build_dataset

# 2. Fine-tune (needs GPU)
python -m src.finetune.train_lora --data training_data.jsonl --epochs 3

# 3. Evaluate the fine-tuned model's actual behavior
#    (point RAGChain's llm at your fine-tuned model instead of MockClient —
#    load it via PeftModel.from_pretrained(base_model, "./finetuned-adapter")
#    and wrap it in a class with a .generate() method matching LLMClient's interface)
python -m src.eval.run_eval_report --llm anthropic   # or your custom wrapper
```

## Scaling the dataset beyond one sample filing

`build_dataset.py` currently only has the one bundled sample filing to
work with (4 valid Q&A pairs — nowhere near enough to actually
fine-tune on). The real Phase 2 dataset needs:

1. Fetch filings for 20-50 companies across a few sectors using
   `edgar_fetcher.py` (already built in Phase 1, needs internet)
2. Run `build_templated_examples` across all of them — should yield
   hundreds to low thousands of examples
3. Optionally add `build_llm_assisted_examples` on a subset for
   variety beyond simple numeric lookups (open-ended "why" and
   "compare X to Y" style questions the template approach can't generate)
4. **Hand-review a random sample (50-100 examples) before training** —
   the two bugs above prove auto-generated data needs a human check,
   not blind trust

## What Phase 3 adds on top of this

Quantize the fine-tuned model and benchmark inference latency/throughput
at different precision levels (this is the NVIDIA-relevant piece) — see
the main README for the overall roadmap.
