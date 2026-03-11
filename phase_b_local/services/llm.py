"""
Phase B - LLM Inference Service (CPU Only) v3.0 (Pure RAG with Zephyr-7B)
==========================================================================
Uses Zephyr-7B-beta Q4_K_M via ctransformers.
Strict FAISS RAG architecture - no hardcoded facts.
"""

import os
import re
from ctransformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Pointing to the new "Best of Best" Zephyr model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gguf", "zephyr-7b-beta.Q4_K_M.gguf")

# Zephyr tokens
TAG_SYS = "<|system|>"
TAG_USR = "<|user|>"
TAG_AST = "<|assistant|>"
EOS = "</s>"

# Tags to strip completely from output
_STRIP_TAGS = [TAG_SYS, TAG_USR, TAG_AST, EOS]

# ---------------------------------------------------------------------------
# Strict System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Gojan AI, a knowledgeable and friendly voice assistant for Gojan "
    "School of Business and Technology (GSBT), Chennai.\n\n"
    "CRITICAL RULES:\n"
    "1. You MUST answer the user's question using ONLY the provided CONTEXT.\n"
    "2. If the CONTEXT does not contain the answer, say EXACTLY: "
    "'I do not have that information. Please contact the college directly at "
    "+91 7010723984.' DO NOT GUESS.\n"
    "3. Keep your answers brief, maximum 2 to 3 sentences (it will be spoken aloud).\n"
    "4. Reply EXACTLY in the language requested below.\n"
    "5. Be helpful and warm, like a senior student.\n"
    "6. Do not say 'According to the context' - simply state the facts directly."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load the Zephyr-7B GGUF model for CPU inference via ctransformers."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Zephyr-7B model not found: " + MODEL_PATH + "\n"
            "Please wait for the download to complete."
        )
    print("  Loading LLM: " + os.path.basename(MODEL_PATH) + " (Zephyr-7B)...")
    
    # model_type="mistral" works perfectly for Zephyr
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type="mistral",
        context_length=2048,
        threads=4,
    )
    return llm


# ---------------------------------------------------------------------------
# Prompt builder (Zephyr Chat Template)
# ---------------------------------------------------------------------------
def _build_prompt(question, context, detected_language, conversation_history):
    """Build the Zephyr-compatible chat prompt strictly using RAG context."""

    # Language generation instruction
    lang_map = {
        "english":  "Respond in clear simple English.",
        "tanglish": ("Respond in Tanglish - mix Tamil words naturally with "
                     "English script. Example: 'Gojan la courses iruku.'"),
        "tamil":    "Respond fully in Tamil script.",
    }
    lang_instruction = lang_map.get(detected_language, lang_map["english"])

    # Zephyr `<|system|>\n...</s>\n<|user|>\n...</s>\n<|assistant|>` format
    parts = []
    
    # --- System Message ---
    sys_content = SYSTEM_PROMPT + "\n\nLANGUAGE RULE: " + lang_instruction
    parts.append(TAG_SYS + "\n" + sys_content + EOS + "\n")
    
    # --- User Message (injecting Context) ---
    user_content = ""
    if conversation_history:
        user_content += "CONVERSATION HISTORY:\n" + conversation_history + "\n\n"
        
    user_content += "CONTEXT FROM KNOWLEDGE BASE:\n"
    if context:
        user_content += context + "\n\n"
    else:
        user_content += "(No context found)\n\n"
        
    user_content += "USER QUESTION: " + question
    
    parts.append(TAG_USR + "\n" + user_content + EOS + "\n")
    
    # --- Assistant Prompt ---
    parts.append(TAG_AST + "\n")
    
    return "".join(parts)


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------
def clean_response(text, question=""):
    """Clean raw LLM output for voice delivery."""
    if not text or len(text.strip()) < 3:
        return ""

    cleaned = text.strip()

    # Remove strict prompt tags leaked
    for tag in _STRIP_TAGS:
        cleaned = cleaned.replace(tag, "")
    cleaned = cleaned.strip()

    # Remove markdown artifacts
    cleaned = re.sub(r"[*#_`]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Attempt to limit to 3 sentences conceptually for short TTS
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    cleaned = " ".join(sentences[:3])

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------
def get_fallback(language="english"):
    """Generic offline fallback if the LLM completely fails or crashes."""
    fallbacks = {
        "english": (
            "I seem to be having trouble processing that right now. "
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
        # Zephyr hyperparameters optimized for precise RAG reading
        response = llm(
            prompt,
            max_new_tokens=256,
            temperature=0.1,  # extremely low to prevent hallucination
            top_p=0.9,
            repetition_penalty=1.1,
            stop=[EOS, TAG_USR, TAG_SYS],
        )

        raw_text = response.strip()
        answer = clean_response(raw_text, question)

        if len(answer.strip()) < 5:
            return get_fallback(detected_language)

        return answer

    except Exception as e:
        print("      LLM error: " + str(e))
        return get_fallback(detected_language)
