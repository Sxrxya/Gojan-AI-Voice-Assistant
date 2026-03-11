"""
Phase B - LLM Inference Service (CPU Only)
===========================================
TinyLlama Q4 GGUF via llama-cpp-python.
Multilingual prompt with conversation history support.
"""

import os
import re
from ctransformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gguf", "gojan_ai_q4.gguf")

# TinyLlama chat format tokens — built via concat to stay safe in all tooling
EOS = "<" + "/s>"
TAG_SYS = "<" + "|system|>"
TAG_USR = "<" + "|user|>"
TAG_AST = "<" + "|assistant|>"

# Tags to remove from raw model output
_STRIP_TAGS = [TAG_SYS, TAG_USR, TAG_AST, EOS]

FALLBACK_ANSWER = (
    "I don't have that info right now. "
    "Antha info ennakku theriyala. "
    "Please contact: +91 7010723984 or gsbt@gojaneducation.tech"
)


def load_model():
    """Load the TinyLlama GGUF model for CPU inference."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "GGUF model not found: " + MODEL_PATH + "\n"
            "Download gojan_ai_q4.gguf from Colab and place in models/gguf/"
        )
    print("  Loading LLM: " + os.path.basename(MODEL_PATH) + "...")
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type="llama",
        context_length=1024,
        threads=4
    )
    return llm


def _build_prompt(question, context, detected_language, conversation_history):
    """Build the TinyLlama chat-format prompt."""
    system_msg = (
        "You are an interactive voice assistant for Gojan School of Business "
        "and Technology. Respond ONLY in " + detected_language + ". "
        "Use conversation history for follow-up context. "
        "Answer using the context provided. Be conversational and warm."
    )

    parts = [TAG_SYS, "\n"]
    parts.append(system_msg + "\n")
    parts.append("Context:\n" + context + "\n")
    if conversation_history:
        parts.append("Conversation so far:\n" + conversation_history + "\n")
    parts.append(EOS + "\n")
    parts.append(TAG_USR + "\n")
    parts.append(question + EOS + "\n")
    parts.append(TAG_AST + "\n")
    return "".join(parts)


def post_process_response(text, language):
    """Clean raw model output: strip tags, limit length."""
    if not text:
        return FALLBACK_ANSWER

    cleaned = text.strip()

    # Remove leaked prompt tags
    for tag in _STRIP_TAGS:
        cleaned = cleaned.replace(tag, "")
    cleaned = cleaned.strip()

    # If Tamil requested but response has no Tamil chars, add notice
    if language == "tamil":
        has_tamil = any("\u0B80" <= ch <= "\u0BFF" for ch in cleaned)
        if not has_tamil and len(cleaned) > 5:
            cleaned += " (Tamil response unavailable, please ask in English)"

    # Trim to max 3 sentences for voice output
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentences) > 3:
        cleaned = " ".join(sentences[:3])

    return cleaned if cleaned else FALLBACK_ANSWER


def generate_answer(llm, question, context,
                    detected_language="english",
                    conversation_history=""):
    """Generate an answer using the fine-tuned TinyLlama model."""
    prompt = _build_prompt(question, context, detected_language,
                           conversation_history)

    response = llm(
        prompt,
        max_new_tokens=180,
        temperature=0.2,
        top_p=0.9,
        stop=[EOS, TAG_USR, TAG_SYS],
    )

    raw_text = response.strip()
    return post_process_response(raw_text, detected_language)
