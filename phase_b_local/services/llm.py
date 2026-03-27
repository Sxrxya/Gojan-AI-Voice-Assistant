"""
Phase B - LLM Inference Service (CPU Only) v3.2 (Pure RAG with TinyLlama-Chat)
==========================================================================
Uses TinyLlama-1.1B-Chat-v1.0 Q4_K_M via ctransformers.
Strict FAISS RAG architecture - no hardcoded facts.
Ultra-strict prompt formatting for 1B models.
"""

import os
import re
from ctransformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gguf", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# TinyLlama tokens
TAG_SYS = "<|system|>"
TAG_USR = "<|user|>"
TAG_AST = "<|assistant|>"
EOS = "</s>"

_STRIP_TAGS = [TAG_SYS, TAG_USR, TAG_AST, EOS, "User Question:", "Response:", "Answer:", "Assistant:"]

# ---------------------------------------------------------------------------
# Strict System Prompt
# ---------------------------------------------------------------------------
# 1B models need extremely simple, capitalised instructions
SYSTEM_PROMPT = (
    "You are Campus Sage, the voice assistant for Gojan School of Business and Technology, Chennai.\n"
    "STRICT RULES:\n"
    "1. Answer in EXACTLY 1-2 sentences. NEVER more than 2 sentences.\n"
    "2. Use ONLY facts from the CONTEXT below. Do NOT invent facts.\n"
    "3. If the CONTEXT has no answer, say: "
    "'I don't have that info. Contact Gojan at +91 7010723984.'\n"
    "4. NO bullet points, NO lists, NO numbering, NO markdown.\n"
    "5. Speak naturally like a friendly college senior."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load the TinyLlama-Chat GGUF model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found: " + MODEL_PATH)
    print("  Loading LLM: " + os.path.basename(MODEL_PATH))
    
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type="llama",
        context_length=1024,
        threads=4,
    )
    return llm


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _build_prompt(question, context, detected_language, conversation_history):
    """Build the prompt. Strict boundaries to prevent 1B model confusion."""

    # 1B models struggle with complex language rules, keep it direct.
    lang_map = {
        "english":  "Write the answer in English.",
        "tanglish": "Write the answer using Tamil words written in English alphabet (Tanglish). Example: 'Gojan la courses iruku'.",
        "tamil":    "Write the answer fully in Tamil script.",
    }
    lang_instruction = lang_map.get(detected_language, lang_map["english"])

    parts = []
    
    # System Message
    sys_content = SYSTEM_PROMPT + "\nRule 5: " + lang_instruction
    parts.append(TAG_SYS + "\n" + sys_content + EOS + "\n")
    
    # User Message: Explicit structure
    user_content = "[CONTEXT]:\n"
    if context:
        user_content += context + "\n"
    else:
        user_content += "NONE\n"
        
    user_content += "\n[QUESTION]: " + question
    
    parts.append(TAG_USR + "\n" + user_content + EOS + "\n")
    
    # Assistant Prompt
    parts.append(TAG_AST + "\n")
    
    return "".join(parts)


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------
def clean_response(text, question=""):
    """Clean raw LLM output for voice delivery — hardened against all formatting leaks."""
    if not text:
        return ""

    cleaned = text.strip()

    # 1. Strip all prompt/role tags that leaked
    for tag in _STRIP_TAGS:
        cleaned = cleaned.replace(tag, "")
    cleaned = cleaned.strip()

    # 2. Kill markdown symbols: * # _ ` ~ | >
    cleaned = re.sub(r"[*#_`~|>]", "", cleaned)

    # 3. Kill numbered lists ("1. ", "2) ") and bullet points ("- ", "• ")
    cleaned = re.sub(r"^\s*[\d]+[.)\]]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-•▶►■□]\s*", "", cleaned, flags=re.MULTILINE)

    # 4. Kill bracketed prompt leaks: [CONTEXT], [QUESTION], [ANSWER]
    cleaned = re.sub(r"\[.*?\]:?\s*", "", cleaned)

    # 5. Kill URLs (should not be spoken aloud — keep phone numbers)
    cleaned = re.sub(r"https?://\S+", "", cleaned)

    # 6. Kill repeated punctuation ("...", "!!!")
    cleaned = re.sub(r"([.!?])\1+", r"\1", cleaned)

    # 7. Collapse all whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 8. Fallback if it leaked its own instruction
    if "please contact the college" in cleaned.lower() and "i do not have" not in cleaned.lower():
        return get_fallback()

    # 9. HARD LIMIT: max 2 sentences for voice output
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    cleaned = " ".join(sentences[:2])

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------
def get_fallback(language="english"):
    """Fallback if the LLM completely fails or hallucinates strongly."""
    fallbacks = {
        "english": (
            "I do not have that information. "
            "Please contact Gojan college directly at +91 7010723984."
        ),
        "tanglish": (
            "Ennakku process panna mudiyala. "
            "Gojan college ku directly contact pannunga: +91 7010723984."
        ),
        "tamil": (
            "என்னை தொடர்பு கொள்ள முடியவில்லை. "
            "கோஜன் கல்லூரியை தொடர்பு கொள்ளுங்கள்: +91 7010723984."
        ),
    }
    return fallbacks.get(language, fallbacks["english"])


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_answer(llm, question, context,
                    detected_language="english",
                    conversation_history=""):
    """Pure LLM generative function driven entirely by RAG chunk context."""
    prompt = _build_prompt(question, context, detected_language,
                           conversation_history)

    try:
        response = llm(
            prompt,
            max_new_tokens=80,
            temperature=0.01,  # near zero to force extraction and prevent hallucination
            top_p=0.9,
            repetition_penalty=1.15,
            stop=[EOS, TAG_USR, TAG_SYS, "\n\n", "User:", "Question:"],
        )

        raw_text = response.strip()
        answer = clean_response(raw_text, question)

        # Basic Anti-Hallucination Filter: 
        # 1B models often invent times, dates, or numbers when they don't know the answer.
        # If the answer contains numbers NOT present in the context, it's a hallucination.
        if context:
            ctx_lower = context.lower()
            # Find all numbers in the generated answer
            ans_numbers = re.findall(r'\b\d+\b', answer)
            for num in ans_numbers:
                if num not in ctx_lower:
                    # The LLM invented a number not in the FAISS source text -> Reject
                    return get_fallback(detected_language)

        # Catch hallucinated library timing if model forces it despite zero temp
        hallucination_keywords = ["10 am", "5 pm", "10:30", "weekdays"]
        if not context and any(h in answer.lower() for h in hallucination_keywords):
            return get_fallback(detected_language)

        if len(answer.strip()) < 5:
            return get_fallback(detected_language)

        return answer

    except Exception as e:
        print("      LLM error: " + str(e))
        return get_fallback(detected_language)
