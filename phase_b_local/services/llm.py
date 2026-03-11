"""
Phase B - LLM Inference Service (CPU Only) v2.1
=================================================
TinyLlama Q4 GGUF via ctransformers.
Enhanced with GOJAN_FACTS, topic detection, and smart fallback.
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

_STRIP_TAGS = [TAG_SYS, TAG_USR, TAG_AST, EOS,
               "[INST]", "[/INST]", "System:", "Assistant:", "User:"]

# ---------------------------------------------------------------------------
# Verified Gojan College Facts (source of truth)
# ---------------------------------------------------------------------------
GOJAN_FACTS = {
    "location": "80 Feet Road, Edapalayam, Redhills, Chennai - 600 052",
    "phone": "+91 7010723984 / +91 7010723985",
    "email": "gsbt@gojaneducation.tech",
    "tnea_code": "1123",
    "established": "2005",
    "campus": "80 acres at Redhills, Chennai",
    "affiliation": "Anna University, Chennai",
    "accreditation": "NAAC Accredited, AICTE Recognized",
    "ug_courses": [
        "B.E. Aeronautical Engineering",
        "B.E. Computer Science and Engineering",
        "B.E. Electronics and Communication Engineering",
        "B.E. Artificial Intelligence and Machine Learning",
        "B.E. Cyber Security Engineering",
        "B.E. Medical Electronics Engineering",
        "B.E. Mechanical and Automation Engineering",
        "B.Tech. Information Technology",
    ],
    "pg_courses": ["MBA - Master of Business Administration"],
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Gojan AI, a friendly voice assistant for Gojan School of "
    "Business and Technology, Chennai.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY about Gojan college topics\n"
    "2. Use ONLY the context provided - never make up facts\n"
    "3. If context has no answer say: "
    "'I don't have that info. Contact: +91 7010723984'\n"
    "4. Keep answers SHORT - maximum 2-3 sentences for voice\n"
    "5. Match the language of the user EXACTLY:\n"
    "   - English question -> English answer\n"
    "   - Tanglish question -> Tanglish answer\n"
    "   - Tamil question -> Tamil answer\n"
    "6. Be warm and friendly like a helpful senior student\n"
    "7. Never repeat the question back\n"
    "8. Never say 'Based on the context' - just answer directly"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load the TinyLlama GGUF model for CPU inference."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "GGUF model not found: " + MODEL_PATH + "\n"
            "Download gojan_ai_q4.gguf and place in models/gguf/"
        )
    print("  Loading LLM: " + os.path.basename(MODEL_PATH) + "...")
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type="llama",
        context_length=1024,
        threads=4,
    )
    return llm


# ---------------------------------------------------------------------------
# Topic detection — injects verified facts into context
# ---------------------------------------------------------------------------
def detect_topic(question):
    """Detect topic keywords and return matching verified facts."""
    q = question.lower()
    facts = []

    if any(w in q for w in ["location", "where", "address", "place",
                             "enge", "iruku"]):
        facts.append("Location: " + GOJAN_FACTS["location"])

    if any(w in q for w in ["phone", "contact", "number", "call",
                             "reach", "pesanum"]):
        facts.append("Phone: " + GOJAN_FACTS["phone"])
        facts.append("Email: " + GOJAN_FACTS["email"])

    if any(w in q for w in ["course", "courses", "department", "branch",
                             "programme", "enna"]):
        ug = ", ".join(GOJAN_FACTS["ug_courses"])
        facts.append("UG Courses: " + ug)
        facts.append("PG Course: " + GOJAN_FACTS["pg_courses"][0])

    if any(w in q for w in ["tnea", "code", "admission", "apply",
                             "counselling", "serkkai"]):
        facts.append("TNEA Code: " + GOJAN_FACTS["tnea_code"])

    if any(w in q for w in ["established", "founded", "started", "year",
                             "when", "aarambichu"]):
        facts.append("Established: " + GOJAN_FACTS["established"])
        facts.append("Campus: " + GOJAN_FACTS["campus"])

    if any(w in q for w in ["naac", "aicte", "affiliated", "approved",
                             "accredited"]):
        facts.append("Status: " + GOJAN_FACTS["accreditation"])
        facts.append("Affiliated to: " + GOJAN_FACTS["affiliation"])

    return "\n".join(facts)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _build_prompt(question, context, detected_language, conversation_history):
    """Build TinyLlama chat-format prompt with verified facts."""

    # Language instruction
    lang_map = {
        "english":  "Respond in clear simple English.",
        "tanglish": ("Respond in Tanglish - mix Tamil words naturally with "
                     "English. Example: 'Gojan la nalla courses iruku.'"),
        "tamil":    "Respond fully in Tamil script.",
    }
    lang_instruction = lang_map.get(detected_language, lang_map["english"])

    # Inject verified facts for this topic
    topic_facts = detect_topic(question)

    parts = [TAG_SYS, "\n"]
    parts.append(SYSTEM_PROMPT + "\n\n")
    parts.append("LANGUAGE: " + lang_instruction + "\n\n")

    if topic_facts:
        parts.append("VERIFIED FACTS:\n" + topic_facts + "\n\n")

    if context:
        parts.append("CONTEXT:\n" + context + "\n\n")

    if conversation_history:
        parts.append("CONVERSATION HISTORY:\n" + conversation_history + "\n\n")

    parts.append(EOS + "\n")
    parts.append(TAG_USR + "\n")
    parts.append(question + EOS + "\n")
    parts.append(TAG_AST + "\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------
def clean_response(text, question=""):
    """Clean raw LLM output: strip tags, remove question echo, limit length."""
    if not text or len(text.strip()) < 3:
        return ""

    cleaned = text.strip()

    # Remove prompt tags
    for tag in _STRIP_TAGS:
        cleaned = cleaned.replace(tag, "")
    cleaned = cleaned.strip()

    # Remove lines that just repeat the question
    if question:
        q_words = set(question.lower().split())
        lines = cleaned.split("\n")
        good_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_words = set(line.lower().split())
            overlap = len(q_words & line_words) / max(len(q_words), 1)
            if overlap < 0.7:
                good_lines.append(line)
        cleaned = " ".join(good_lines)

    # Remove markdown artifacts
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"#+", "", cleaned)
    cleaned = re.sub(r"`+", "", cleaned)

    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Limit to 3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    cleaned = " ".join(sentences[:3])

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Fallback answers
# ---------------------------------------------------------------------------
def get_fallback(question, language="english"):
    """Smart fallback based on detected topic or generic contact info."""
    topic_facts = detect_topic(question)

    if topic_facts:
        return topic_facts + "\nFor more info: " + GOJAN_FACTS["phone"]

    fallbacks = {
        "english": (
            "I don't have that specific information right now. "
            "Please contact Gojan college directly at "
            + GOJAN_FACTS["phone"] + " or email "
            + GOJAN_FACTS["email"] + "."
        ),
        "tanglish": (
            "Antha info ennakku theriyala. "
            "Gojan college ku directly contact pannunga: "
            + GOJAN_FACTS["phone"] + "."
        ),
        "tamil": (
            "அந்த தகவல் என்னிடம் இல்லை. "
            "கோஜன் கல்லூரியை தொடர்பு கொள்ளுங்கள்: "
            + GOJAN_FACTS["phone"] + "."
        ),
    }
    return fallbacks.get(language, fallbacks["english"])


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_answer(llm, question, context,
                    detected_language="english",
                    conversation_history=""):
    """Generate an accurate college answer using LLM + RAG + verified facts."""
    prompt = _build_prompt(question, context, detected_language,
                           conversation_history)

    try:
        response = llm(
            prompt,
            max_new_tokens=200,
            temperature=0.15,
            top_p=0.85,
            stop=[EOS, TAG_USR, TAG_SYS],
        )

        raw_text = response.strip()
        answer = clean_response(raw_text, question)

        # If answer too short or empty — use smart fallback
        if len(answer.strip()) < 10:
            answer = get_fallback(question, detected_language)

        return answer

    except Exception as e:
        print("      LLM error: " + str(e))
        return get_fallback(question, detected_language)
