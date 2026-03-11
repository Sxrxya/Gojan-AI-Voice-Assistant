"""
Phase A — Step 4: Build QA Dataset for Fine-Tuning
====================================================
Generates question-answer pairs from templates and chunks,
then splits into training and evaluation JSONL files.

Run on: Google Colab
Input : data/chunks/all_chunks.json + data/seed_facts.txt
Output: data/qa_dataset/train.jsonl + data/qa_dataset/eval.jsonl
"""

import os
import json
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "chunks", "all_chunks.json")
SEED_FACTS_PATH = os.path.join(PROJECT_ROOT, "data", "seed_facts.txt")
QA_DIR = os.path.join(PROJECT_ROOT, "data", "qa_dataset")

TRAIN_SPLIT = 0.9
RANDOM_SEED = 42

SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant for Gojan School of Business and Technology. "
    "Answer concisely and accurately."
)

# ---------------------------------------------------------------------------
# Template-based QA pairs (verified facts)
# ---------------------------------------------------------------------------
QUESTION_TEMPLATES = [
    (
        "What is Gojan School of Business and Technology?",
        "Gojan School of Business and Technology is an engineering college established in 2005, "
        "located on an 80-acre campus at Redhills, Chennai. It is affiliated to Anna University "
        "and accredited by NAAC."
    ),
    (
        "Where is Gojan School of Business and Technology located?",
        "Gojan School of Business and Technology is located at 80 Feet Road, Edapalayam, "
        "Redhills, Chennai, Tamil Nadu - 600 052."
    ),
    (
        "What is the TNEA code for Gojan?",
        "The TNEA counselling code for Gojan School of Business and Technology is 1123."
    ),
    (
        "What courses are offered at Gojan?",
        "Gojan offers 8 UG programmes: B.E. Aeronautical Engineering, B.E. CSE, B.E. ECE, "
        "B.E. AI-ML, B.E. Cyber Security, B.E. Medical Electronics, B.E. Mechanical and "
        "Automation Engineering, and B.Tech. IT. The PG programme is MBA."
    ),
    (
        "How do I apply to Gojan?",
        "For engineering courses, apply through TNEA counselling by Anna University using "
        "TNEA code 1123. For MBA, apply through TANCET. You can also apply online at "
        "https://forms.gle/W1pEKX2kHno97YfK9."
    ),
    (
        "What is the contact number for Gojan?",
        "You can contact Gojan School of Business and Technology at +91 7010723984 or "
        "+91 7010723985. You can also reach them at +044-2631 1045 or email "
        "gsbt@gojaneducation.tech."
    ),
    (
        "Is Gojan affiliated to Anna University?",
        "Yes, Gojan School of Business and Technology is affiliated to Anna University, Chennai."
    ),
    (
        "Is Gojan NAAC accredited?",
        "Yes, Gojan School of Business and Technology is accredited by NAAC."
    ),
    (
        "What is the campus size of Gojan?",
        "Gojan School of Business and Technology has an 80-acre campus at Redhills, Chennai."
    ),
    (
        "Does Gojan have hostel facilities?",
        "Yes, Gojan School of Business and Technology provides separate hostel facilities for "
        "boys and girls."
    ),
    (
        "What are the facilities at Gojan?",
        "The campus has a library, modern laboratories, WiFi connectivity, sports facilities, "
        "canteen, hostel accommodation, and transport facilities from various parts of Chennai."
    ),
    (
        "When was Gojan established?",
        "Gojan School of Business and Technology was established in the year 2005."
    ),
    (
        "What is the eligibility for admission to Gojan?",
        "For B.E./B.Tech programmes, students must have completed 12th standard with Physics, "
        "Chemistry, Mathematics and score minimum 45% aggregate (40% for reserved categories). "
        "Admissions are through TNEA counselling."
    ),
    (
        "Does Gojan have a placement cell?",
        "Yes, Gojan has an active placement cell and an Entrepreneurship Development Cell "
        "(ED Cell) that connects students with industry opportunities."
    ),
    (
        "What is the email address of Gojan?",
        "The official email address is gsbt@gojaneducation.tech."
    ),
    (
        "Does Gojan offer MBA?",
        "Yes, Gojan School of Business and Technology offers MBA (Master of Business "
        "Administration) as a postgraduate programme."
    ),
    (
        "What clubs does Gojan have?",
        "Gojan has various student clubs under Gojan Clubs, along with NSS, IEEE Student "
        "Branch, and an Entrepreneurship Development Cell."
    ),
    (
        "Is Gojan approved by AICTE?",
        "Yes, Gojan School of Business and Technology is recognized by AICTE (All India "
        "Council for Technical Education), New Delhi."
    ),
    (
        "What is the vision of Gojan?",
        "The vision of Gojan School of Business and Technology is to create a dynamic, "
        "optimistic community of educated youth by providing quality education at "
        "affordable cost and empowering them with leadership and employable skills."
    ),
    (
        "Does Gojan have AI ML course?",
        "Yes, Gojan offers B.E. Artificial Intelligence and Machine Learning Engineering "
        "as an undergraduate programme."
    ),
    (
        "Does Gojan have Cyber Security course?",
        "Yes, Gojan offers B.E. Cyber Security Engineering as an undergraduate programme."
    ),
    (
        "Does Gojan have transport facility?",
        "Yes, Gojan School of Business and Technology provides transport facilities "
        "connecting various parts of Chennai to the college campus at Redhills."
    ),
]

# ---------------------------------------------------------------------------
# Keyword-to-question mappings for auto-QA from chunks
# ---------------------------------------------------------------------------
KEYWORD_QA_MAP = {
    "placement": [
        "What are the placement opportunities at Gojan?",
        "Tell me about placements at Gojan School of Business and Technology.",
        "Which companies visit Gojan for placements?",
    ],
    "department": [
        "What departments are available at Gojan?",
        "Tell me about the departments at Gojan School of Business and Technology.",
    ],
    "faculty": [
        "Who are the faculty members at Gojan?",
        "Tell me about the faculty at Gojan School of Business and Technology.",
    ],
    "event": [
        "What events are organized at Gojan?",
        "Tell me about events at Gojan School of Business and Technology.",
    ],
    "fest": [
        "What festivals or fests happen at Gojan?",
        "Tell me about the fests at Gojan.",
    ],
    "lab": [
        "What laboratories are available at Gojan?",
        "Tell me about the lab facilities at Gojan.",
    ],
    "library": [
        "Does Gojan have a library?",
        "Tell me about the library at Gojan School of Business and Technology.",
    ],
    "hostel": [
        "Does Gojan have hostel facilities?",
        "Tell me about the hostel at Gojan.",
    ],
    "sports": [
        "What sports facilities does Gojan have?",
        "Tell me about sports at Gojan School of Business and Technology.",
    ],
    "nss": [
        "Does Gojan have NSS activities?",
        "Tell me about NSS at Gojan.",
    ],
    "research": [
        "What research activities happen at Gojan?",
        "Tell me about research at Gojan School of Business and Technology.",
    ],
    "admission": [
        "How can I get admission to Gojan?",
        "What is the admission process at Gojan?",
    ],
}


def generate_template_qa() -> list[dict]:
    """Generate QA pairs from hardcoded templates."""
    qa_pairs = []
    for question, answer in QUESTION_TEMPLATES:
        qa_pairs.append({
            "instruction": SYSTEM_INSTRUCTION,
            "input": question,
            "output": answer,
        })
    return qa_pairs


def generate_chunk_qa(chunks: list[dict]) -> list[dict]:
    """Generate QA pairs from chunks based on keyword matching."""
    qa_pairs = []
    random.seed(RANDOM_SEED)

    for chunk in chunks:
        text_lower = chunk["text"].lower()

        for keyword, questions in KEYWORD_QA_MAP.items():
            if keyword in text_lower:
                # Pick a random question variant
                question = random.choice(questions)
                qa_pairs.append({
                    "instruction": SYSTEM_INSTRUCTION,
                    "input": question,
                    "output": chunk["text"],
                })
                break  # One QA pair per chunk

    return qa_pairs


def main():
    os.makedirs(QA_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    print("═" * 60)
    print("  QA Dataset Builder")
    print("═" * 60)
    print()

    # Template-based QA
    template_qa = generate_template_qa()
    print(f"  ✓ Template QA pairs: {len(template_qa)}")

    # Chunk-based QA
    chunk_qa = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunk_qa = generate_chunk_qa(chunks)
        print(f"  ✓ Chunk-based QA pairs: {len(chunk_qa)}")
    else:
        print(f"  ⚠ Chunks file not found: {CHUNKS_PATH}")
        print("    Run 03_clean_and_chunk.py first.")

    # Combine and shuffle
    all_qa = template_qa + chunk_qa
    random.shuffle(all_qa)

    # Split into train/eval
    split_idx = int(len(all_qa) * TRAIN_SPLIT)
    train_qa = all_qa[:split_idx]
    eval_qa = all_qa[split_idx:]

    # Save as JSONL
    train_path = os.path.join(QA_DIR, "train.jsonl")
    eval_path = os.path.join(QA_DIR, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("═" * 60)
    print(f"  ✓ Total QA pairs: {len(all_qa)} | Train: {len(train_qa)} | Eval: {len(eval_qa)}")
    print(f"  ✓ Saved: {train_path}")
    print(f"  ✓ Saved: {eval_path}")
    print("═" * 60)


if __name__ == "__main__":
    main()
