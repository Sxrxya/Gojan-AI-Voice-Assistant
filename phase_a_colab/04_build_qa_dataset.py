"""
Phase A - Step 4: Build Multilingual QA Dataset
=================================================
Generates training data from templates, chunks, and multi-turn dialogues
in English, Tamil, and Tanglish.

Run on: Google Colab
Input : data/qa_templates.json, data/qa_multiturn.json,
        data/chunks/all_chunks.json, data/seed_facts.txt
Output: data/qa_dataset/train.jsonl + eval.jsonl
"""

import os
import json
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_PATH = os.path.join(PROJECT_ROOT, "data", "qa_templates.json")
MULTITURN_PATH = os.path.join(PROJECT_ROOT, "data", "qa_multiturn.json")
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "chunks", "all_chunks.json")
QA_DIR = os.path.join(PROJECT_ROOT, "data", "qa_dataset")

TRAIN_SPLIT = 0.9
RANDOM_SEED = 42

SYSTEM_INSTRUCTION = (
    "You are an interactive, friendly voice assistant for Gojan School of "
    "Business and Technology (GSBT), an autonomous institution in Chennai. "
    "You remember the conversation history. "
    "Answer in the same language the user uses - Tamil, Tanglish, or English. "
    "Be warm, helpful, and conversational like a friendly senior student. "
    "If you don't know, suggest contacting +91 70107 23984 or gsbt@gsbt.edu.in."
)

# Keyword-to-question maps for auto-QA from chunks
KEYWORD_QA = {
    "placement": [
        "What are the placement opportunities at Gojan?",
        "Gojan la placement eppadi iruku?",
        "\u0baa\u0bcd\u0bb2\u0bc7\u0b9a\u0bcd\u0bae\u0bc6\u0ba3\u0bcd\u0b9f\u0bcd \u0b8e\u0baa\u0bcd\u0baa\u0b9f\u0bbf?",
    ],
    "department": [
        "Tell me about the departments at Gojan.",
        "Gojan la enna departments iruku?",
    ],
    "faculty": [
        "Who are the faculty at Gojan?",
        "Faculty pathi sollu.",
    ],
    "event": [
        "What events are organized at Gojan?",
        "Events pathi sollu Gojan la.",
    ],
    "lab": [
        "What labs are available at Gojan?",
        "Labs pathi sollu.",
    ],
    "library": [
        "Tell me about the library.",
        "Library nalla iruka?",
    ],
    "hostel": [
        "Does Gojan have hostel?",
        "Hostel pathi sollu.",
    ],
    "sport": [
        "What sports facilities does Gojan have?",
        "Sports facility iruka?",
    ],
    "nss": [
        "Tell me about NSS at Gojan.",
        "NSS activities enna?",
    ],
    "admission": [
        "How to get admission to Gojan?",
        "Admission eppadi?",
    ],
    "research": [
        "Tell me about research at Gojan.",
        "Research nadakkuma?",
    ],
}


def load_json(path):
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_template_qa():
    """Generate QA pairs from trilingual templates."""
    templates = load_json(TEMPLATES_PATH)
    qa = []
    for t in templates:
        qa.append({
            "instruction": SYSTEM_INSTRUCTION,
            "input": t["q"],
            "output": t["a"],
        })
    print(f"  \u2713 Template QA: {len(qa)} pairs")
    return qa


def generate_multiturn_qa():
    """Generate training examples from multi-turn conversations."""
    dialogues = load_json(MULTITURN_PATH)
    qa = []
    for dlg in dialogues:
        turns = dlg["turns"]
        history = ""
        for i in range(0, len(turns) - 1, 2):
            user_turn = turns[i]
            ai_turn = turns[i + 1] if i + 1 < len(turns) else None
            if not ai_turn:
                break
            input_text = history + f"User: {user_turn['text']}" if history else f"User: {user_turn['text']}"
            qa.append({
                "instruction": SYSTEM_INSTRUCTION,
                "input": input_text,
                "output": ai_turn["text"],
            })
            history += f"User: {user_turn['text']}\nAssistant: {ai_turn['text']}\n"
    print(f"  \u2713 Multi-turn QA: {len(qa)} pairs")
    return qa


def generate_chunk_qa():
    """Generate QA pairs from chunks using keyword matching."""
    chunks = load_json(CHUNKS_PATH)
    if not chunks:
        return []
    random.seed(RANDOM_SEED)
    qa = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        for keyword, questions in KEYWORD_QA.items():
            if keyword in text_lower:
                question = random.choice(questions)
                qa.append({
                    "instruction": SYSTEM_INSTRUCTION,
                    "input": question,
                    "output": chunk["text"],
                })
                break
    print(f"  \u2713 Chunk-based QA: {len(qa)} pairs")
    return qa


def main():
    os.makedirs(QA_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    print("\u2550" * 60)
    print("  Multilingual QA Dataset Builder")
    print("\u2550" * 60)
    print()

    all_qa = []
    all_qa.extend(generate_template_qa())
    all_qa.extend(generate_multiturn_qa())
    all_qa.extend(generate_chunk_qa())

    random.shuffle(all_qa)
    split = int(len(all_qa) * TRAIN_SPLIT)
    train_qa, eval_qa = all_qa[:split], all_qa[split:]

    train_path = os.path.join(QA_DIR, "train.jsonl")
    eval_path = os.path.join(QA_DIR, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("\u2550" * 60)
    print(f"  \u2713 Total: {len(all_qa)} | Train: {len(train_qa)} | Eval: {len(eval_qa)}")
    print("\u2550" * 60)


if __name__ == "__main__":
    main()
