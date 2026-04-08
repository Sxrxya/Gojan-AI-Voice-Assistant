"""
Phase A - Step 5: Fine-Tune TinyLlama with QLoRA
==================================================
QLoRA fine-tuning of TinyLlama-1.1B-Chat on multilingual
college QA dataset (English + Tamil + Tanglish).

Run on: Google Colab (T4 GPU required)
Input : data/qa_dataset/train.jsonl + eval.jsonl
Output: models/lora_adapter/
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "qa_dataset", "train.jsonl")
EVAL_PATH = os.path.join(PROJECT_ROOT, "data", "qa_dataset", "eval.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "lora_adapter")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LENGTH = 768  # increased for conversation history

# TinyLlama chat format tokens (built via concat)
EOS = "<" + "/s>"
TAG_SYS = "<" + "|system|>"
TAG_USR = "<" + "|user|>"
TAG_AST = "<" + "|assistant|>"

SYSTEM_PROMPT = (
    "You are an interactive, friendly voice assistant for Gojan School of "
    "Business and Technology (GSBT), an autonomous institution in Chennai. "
    "You have memory of the current conversation.\nRules:\n"
    "- If user speaks Tamil, respond fully in Tamil script\n"
    "- If user speaks Tanglish, respond in natural Tanglish\n"
    "- If user speaks English, respond in English\n"
    "- Always remember previous turns in the conversation\n"
    "- For follow-up questions, use conversation history for context\n"
    "- Be warm and conversational, not robotic\n"
    "- If you don't know something, give the contact: +91 70107 23984 or gsbt@gsbt.edu.in"
)


def format_prompt(sample: dict) -> str:
    """Format a QA pair using the TinyLlama chat template."""
    text = (
        TAG_SYS + "\n"
        + SYSTEM_PROMPT + EOS + "\n"
        + TAG_USR + "\n"
        + sample["input"] + EOS + "\n"
        + TAG_AST + "\n"
        + sample["output"] + EOS
    )
    return text


def load_dataset_jsonl(path):
    """Load a JSONL file into a HuggingFace Dataset."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)


def main():
    print("=" * 55)
    print("  TinyLlama QLoRA Fine-Tuning (Multilingual)")
    print("=" * 55)
    print()

    # Load datasets
    print("Loading datasets...")
    train_ds = load_dataset_jsonl(TRAIN_PATH)
    eval_ds = load_dataset_jsonl(EVAL_PATH)
    print("  Train: {} samples | Eval: {} samples".format(len(train_ds), len(eval_ds)))

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model and tokenizer
    print("Loading base model: {}...".format(BASE_MODEL))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("  Trainable params: {:,} / {:,} ({:.2f}%)".format(
        trainable, total, 100 * trainable / total
    ))

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        save_steps=50,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        fp16=True,
        report_to="none",
    )

    # Format datasets
    train_ds = train_ds.map(lambda s: {"text": format_prompt(s)})
    eval_ds = eval_ds.map(lambda s: {"text": format_prompt(s)})

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Train
    print("\nStarting fine-tuning...")
    trainer.train()

    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 55)
    print("  Fine-tuning complete!")
    print("  Adapter saved to: " + OUTPUT_DIR)
    print("  Next: run 06_export_gguf.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
