"""
Phase A - Step 6: Export Fine-Tuned Model to GGUF
==================================================
Merges the LoRA adapter with the base TinyLlama model
and converts to Q4_K_M GGUF format for CPU inference.

Run on: Google Colab
Input : models/lora_adapter/
Output: models/gguf/gojan_ai_q4.gguf
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LORA_DIR = os.path.join(PROJECT_ROOT, "models", "lora_adapter")
MERGED_DIR = os.path.join(PROJECT_ROOT, "models", "merged")
GGUF_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")
GGUF_PATH = os.path.join(GGUF_DIR, "gojan_ai_q4.gguf")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def step1_merge_adapter():
    """Merge LoRA adapter with base model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[1/3] Loading base model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, LORA_DIR)

    print("       Merging weights...")
    model = model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"       Merged model saved to: {MERGED_DIR}")


def step2_convert_gguf():
    """Convert merged HF model to GGUF Q4_K_M."""
    print("[2/3] Converting to GGUF format...")

    # Clone llama.cpp if not present
    llama_cpp_dir = os.path.join(PROJECT_ROOT, "llama.cpp")
    if not os.path.exists(llama_cpp_dir):
        print("       Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp"],
            cwd=PROJECT_ROOT, check=True
        )
    
    # Install llama.cpp Python requirements
    req_path = os.path.join(llama_cpp_dir, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_path],
            check=True
        )

    # Convert
    os.makedirs(GGUF_DIR, exist_ok=True)
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    subprocess.run([
        sys.executable, convert_script, MERGED_DIR,
        "--outfile", GGUF_PATH,
        "--outtype", "q4_k_m",
    ], check=True)

    print(f"       GGUF model saved to: {GGUF_PATH}")


def step3_verify():
    """Verify the GGUF file size."""
    print("[3/3] Verifying...")
    if not os.path.exists(GGUF_PATH):
        print("       ERROR: GGUF file not found!")
        return

    size_mb = os.path.getsize(GGUF_PATH) / (1024 * 1024)
    print(f"       GGUF size: {size_mb:.1f} MB")
    assert size_mb < 1100, f"Model too large ({size_mb:.0f} MB) for 8GB RAM deployment!"
    print("       Size check: PASSED")

    print()
    print("=" * 55)
    print("  DOWNLOAD THESE FILES TO YOUR LAPTOP:")
    print("  1. models/gguf/gojan_ai_q4.gguf")
    print("  2. vector_db/college.index")
    print("  3. vector_db/documents.pkl")
    print("=" * 55)


def main():
    print("=" * 55)
    print("  GGUF Export Pipeline")
    print("=" * 55)
    print()
    step1_merge_adapter()
    print()
    step2_convert_gguf()
    print()
    step3_verify()


if __name__ == "__main__":
    main()
