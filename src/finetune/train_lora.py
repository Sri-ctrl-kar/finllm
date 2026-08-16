"""
train_lora.py
------------------------------------------------------------
LoRA fine-tuning of a small open base model on the financial Q&A
dataset built by dataset_builder.py.

REQUIRES A GPU. This sandbox has neither a GPU nor `torch`/
`transformers`/`peft` installed, so this script is written
correctly but has NOT been executed anywhere in this build — run
it in Colab (free T4 GPU is enough for a 7-8B model with QLoRA) or
on a GPU machine.

WHY LoRA/QLoRA INSTEAD OF FULL FINE-TUNING: full fine-tuning of an
8B-parameter model updates all 8 billion weights — needs 80GB+ GPU
memory and days of compute, completely out of reach for a student
project. LoRA freezes the original model and injects small trainable
"adapter" matrices into each attention layer (typically <1% of the
original parameter count), so you get most of the benefit of
fine-tuning at a tiny fraction of the memory/compute cost. QLoRA
adds 4-bit quantization of the frozen base weights on top of that,
shrinking memory further — this is what makes fine-tuning an 8B
model feasible on a single free-tier Colab GPU.

Usage (on a GPU machine):
    pip install torch transformers peft bitsandbytes accelerate datasets
    python -m src.finetune.dataset_builder   # first, generate training_data.jsonl
    python -m src.finetune.train_lora --data training_data.jsonl --epochs 3
------------------------------------------------------------
"""

import argparse
import json


def load_jsonl_dataset(path: str):
    """Loads the {"prompt": ..., "completion": ...} lines written by
    dataset_builder.save_jsonl() into a HuggingFace `datasets.Dataset`."""
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example: dict) -> dict:
    """Concatenate prompt+completion into the single text field the
    causal LM trains on, with an explicit EOS so the model learns
    where an answer should actually stop (without this, a fine-tuned
    model tends to ramble past the answer instead of stopping)."""
    return {"text": example["prompt"] + example["completion"] + "</s>"}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune a base model on financial Q&A")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3",
                         help="HuggingFace model id. Mistral-7B and Llama-3-8B both work well; "
                              "start with a smaller model (e.g. microsoft/Phi-3-mini-4k-instruct, "
                              "~3.8B params) if Colab's free GPU runs out of memory.")
    parser.add_argument("--data", required=True, help="path to training_data.jsonl (see dataset_builder.py)")
    parser.add_argument("--output-dir", default="./finetuned-adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank — higher = more capacity, more memory")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # Imported here, not at module level, so this file can still be
    # imported/inspected (e.g. by tests) on machines without these
    # heavy ML dependencies installed.
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"Loading base model: {args.base_model}")

    # 4-bit quantization config for QLoRA — this is what makes an
    # 8B-parameter model fit on a free-tier Colab GPU (~16GB) at all.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA is applied to the attention projection matrices — the layers
    # that matter most for adapting how the model attends to new,
    # domain-specific context, without touching the base model's
    # general language ability.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check: should show <1% of total params as trainable

    print(f"Loading dataset from {args.data}")
    dataset = load_jsonl_dataset(args.data)
    dataset = dataset.map(format_example)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=1024, padding="max_length")

    tokenized = dataset.map(tokenize, batched=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. Load it later with: PeftModel.from_pretrained(base_model, args.output_dir)")


if __name__ == "__main__":
    main()
