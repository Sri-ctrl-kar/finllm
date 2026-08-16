"""
quantize.py
------------------------------------------------------------
Loads a model at three precisions — fp16 (baseline), int8, and
int4 — via bitsandbytes. This is what benchmark.py measures.

REQUIRES A GPU. Not executed anywhere in this build (no torch/GPU
in this sandbox) — written correctly, run on your GPU machine/Colab.

WHAT QUANTIZATION ACTUALLY DOES: a model's weights are normally
stored as 16-bit floats. Quantization stores them with fewer bits
(8 or 4) using a scale+zero-point per block of weights, so each
weight takes less memory and math on them is faster — at the cost
of some precision, which is why accuracy typically drops a bit as
you go to lower precision (this is exactly what benchmark.py +
Phase 2's eval harness measure together: HOW MUCH accuracy is lost
at each level, not just how much faster it gets).

WHY bitsandbytes SPECIFICALLY (vs. GPTQ/AWQ): bitsandbytes does
quantization "on the fly" at model-load time — no separate
calibration/quantization step needed, which makes it the simplest
starting point. GPTQ and AWQ require a calibration pass over sample
data first, but typically achieve better accuracy retention at 4-bit
than bitsandbytes' default — noted as the natural next experiment
once this baseline comparison is working (see docs/PHASE3_NOTES.md).
------------------------------------------------------------
"""

import argparse


def load_model(model_path: str, precision: str):
    """
    Loads `model_path` (a local fine-tuned checkpoint, or a base
    HuggingFace model id) at the given precision.

    precision: "fp16" | "int8" | "int4"
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")

    elif precision == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")

    else:
        raise ValueError(f"Unknown precision '{precision}' — expected fp16, int8, or int4")

    return model, tokenizer


def model_size_gb(model) -> float:
    """Sums parameter memory footprint — a quick sanity check that
    quantization actually shrank the model the way you'd expect
    (e.g. int4 should be roughly 1/4 the fp16 size)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser(description="Load and sanity-check a model at each quantization level")
    parser.add_argument("--model", required=True, help="path to fine-tuned checkpoint or HF model id")
    args = parser.parse_args()

    for precision in ["fp16", "int8", "int4"]:
        print(f"\nLoading at {precision}...")
        model, tokenizer = load_model(args.model, precision)
        size = model_size_gb(model)
        print(f"  {precision}: {size:.2f} GB")
        del model  # free GPU memory before loading the next precision

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
