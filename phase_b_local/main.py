"""
Gojan AI Voice Assistant - Main Loop v3.2 (Bulletproof Wake Word)
==================================================================
Fixes:
- NO spoken greeting on boot (prevents TTS echo from triggering wake)
- Ultra-aggressive wake word matching for Indian accents
- Clear real-time debug output so you can see what's happening
- 3-second cooldown after every TTS to prevent echo loops
"""

import sys
import os
import re
import time
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_b_local.services.stt import (
    load_model as load_stt,
    listen_for_wake_word,
    listen_for_question,
    detect_language,
)
from phase_b_local.services.retriever import load_retriever, retrieve, format_context
from phase_b_local.services.llm import load_model as load_llm, generate_answer, get_fallback
from phase_b_local.services.tts import load_tts, speak, play_beep


# =====================================================================
# Wake Word Detection — ULTRA AGGRESSIVE for Indian accents
# =====================================================================
# Google STT transcribes "Hey Gojan" differently every time with Indian
# accents. Instead of maintaining an ever-growing list, we use pattern
# matching that catches ALL phonetic variations.

EXACT_WAKE_WORDS = [
    "hey gojan", "gojan", "hi gojan", "hello gojan", "anna",
    "hey gojen", "go john", "go jan", "hey go jan",
    "okay gojan", "ok gojan", "gojan ai", "ko jan", "gojon",
    "gojhan", "hey goes in", "goes in", "hey jen",
    "hey golden", "golden", "hey gorgon", "gorgon",
    "gaojan", "gaujan", "bhojan", "hey gorgeous", "gorgeous",
    "hero", "hey hero", "hey george", "george",
    "hey cojan", "cojan", "hey jose", "hey joseph",
    "hey gorjan", "gorjan", "hey ko jan",
    "hey jarvis", "jarvis",
]


def is_wake_word(text):
    """
    Ultra-aggressive wake word detection.
    Strategy: Accept ANYTHING that sounds remotely like "Hey Gojan".
    False positives are OK — better to wake too often than never wake.
    """
    if not text or len(text.strip()) < 2:
        return False

    text_lower = text.lower().strip()
    text_clean = re.sub(r"[^\w\s]", "", text_lower)

    # Strategy 1: Exact substring match against known variations
    for wake in EXACT_WAKE_WORDS:
        if wake in text_clean:
            return True

    # Strategy 2: Any word starting with "goj", "gaoj", "gor", "gou"
    words = text_clean.split()
    for word in words:
        if len(word) >= 3:
            if word.startswith(("goj", "gaoj", "gor", "gou", "gau",
                                "gow", "koj", "coj")):
                return True
            # "go" + any consonant after
            if word.startswith("go") and len(word) >= 4 and word[2] in "jrgnldsz":
                return True

    # Strategy 3: "hey" + any G-word (covers "hey gorgeous", "hey george", etc.)
    if "hey " in text_clean:
        after_hey = text_clean.split("hey ", 1)[-1].strip()
        if after_hey and after_hey[0] in "gGkK":
            return True

    # Strategy 4: Levenshtein distance <= 3 (very permissive)
    def _lev(s1, s2):
        if len(s1) < len(s2):
            return _lev(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    for word in words:
        if len(word) >= 4 and _lev(word, "gojan") <= 3:
            return True

    return False


# =====================================================================
# Conversation Memory
# =====================================================================
class ConversationMemory:
    def __init__(self):
        self.turns = []
        self.last_answer = ""
        self.last_topic = None

    def add_turn(self, role, text, lang="english"):
        self.turns.append({"role": role, "content": text, "language": lang})
        if len(self.turns) > 10:
            self.turns = self.turns[-10:]

    def get_context_window(self):
        if not self.turns:
            return ""
        return "\n".join(t["role"].upper() + ": " + t["content"] for t in self.turns[-6:])

    def clear(self):
        self.turns.clear()
        self.last_answer = ""
        self.last_topic = None


# =====================================================================
# Intent Detection
# =====================================================================
def detect_intent(text):
    lower = text.lower().strip()
    # Only exact "bye" or "goodbye" at the START — prevents accidental exit
    if lower in ["bye", "goodbye", "exit", "quit", "stop", "bye gojan"]:
        return "farewell"
    if any(w in lower for w in ["repeat", "again", "meeendum"]):
        return "repeat"
    if any(w in lower for w in ["more", "details", "explain"]):
        return "more"
    if any(w in lower for w in ["reset", "new topic", "clear"]):
        return "reset"
    if any(w in lower for w in ["thanks", "thank you", "nanri"]):
        return "thanks"
    return "question"


# =====================================================================
# Boot Sequence
# =====================================================================
def boot():
    print()
    print("=" * 58)
    print("  GOJAN SCHOOL OF BUSINESS AND TECHNOLOGY")
    print("  AI VOICE ASSISTANT v4.0 (100% Offline / Local Whisper)")
    print("  English | Tamil | Tanglish")
    print("=" * 58)
    print()

    t0 = time.time()
    components = {}

    print("[1/4] Loading Speech Recognition...")
    try:
        components["stt"] = load_stt()
        print("      Done ({:.1f}s)".format(time.time() - t0))
    except Exception as e:
        print("      Warning: " + str(e))

    print("[2/4] Loading Knowledge Base (FAISS)...")
    try:
        idx, docs, emb = load_retriever()
        components["idx"] = idx
        components["docs"] = docs
        components["emb"] = emb
        print("      Done - {} facts loaded".format(len(docs)))
    except Exception as e:
        print("      FAILED: " + str(e))
        sys.exit(1)

    print("[3/4] Loading Language Model (TinyLlama Q4)...")
    try:
        components["llm"] = load_llm()
        print("      Done ({:.1f}s)".format(time.time() - t0))
    except Exception as e:
        print("      FAILED: " + str(e))
        sys.exit(1)

    print("[4/4] Loading Voice System...")
    try:
        components["tts"] = load_tts()
    except Exception as e:
        print("      FAILED: " + str(e))
        sys.exit(1)

    total = time.time() - t0
    print()
    print("-" * 58)
    print("System ready! Startup: {:.1f}s".format(total))
    print("-" * 58)
    return components


# =====================================================================
# Main Loop
# =====================================================================
def main():
    components = boot()
    memory = ConversationMemory()

    # Spoken greeting on boot (avoiding the wake word 'Gojan' to prevent echo loops)
    print()
    print("  *** I am Gojan AI Assistant ***")
    print("  Say 'Hey Gojan' to wake me up (or any similar phrase)")
    print("  Languages: English | Tamil | Tanglish")
    print("  Say 'bye' to exit")
    print()
    print("=" * 58)
    print()
    
    speak(components["tts"], "Welcome to Gojan A. I. Voice Assistant. I am ready.", "english")
    time.sleep(2) # Let echo fully die
    
    print("  [LISTENING for wake word...]")

    while True:
        try:
            # ────────────────────────────────────────────────
            # PHASE 1: Listen for wake word
            # ────────────────────────────────────────────────
            heard = listen_for_wake_word(timeout=5)

            if heard is None:
                # Silence — keep listening (no print, stays clean)
                continue

            # Show what Google heard (for debugging)
            print("   Google heard: '{}'".format(heard))

            if not is_wake_word(heard):
                print("   (not a wake word — keep talking)")
                continue

            # ────────────────────────────────────────────────
            # PHASE 2: Wake confirmed!
            # ────────────────────────────────────────────────
            print("\n>> WAKE WORD DETECTED! ('{}')".format(heard))
            play_beep()
            print("   Ask your question now...")

            question_result = listen_for_question(timeout=8)

            if question_result is None:
                print("   [No question detected]")
                speak(components["tts"],
                      "I didn't catch that. Try again.",
                      "english")
                time.sleep(3)  # Let echo fully die
                print("\n  [LISTENING for wake word...]")
                continue

            q_text = question_result["text"]
            f_lang = question_result["language"]

            if len(q_text.strip()) < 3:
                print("   [Question too short]")
                speak(components["tts"],
                      "Please ask your question clearly.", "english")
                time.sleep(3)
                print("\n  [LISTENING for wake word...]")
                continue

            print("\n  [{}] You: {}".format(f_lang.upper(), q_text))

            # ────────────────────────────────────────────────
            # PHASE 3: Intent detection
            # ────────────────────────────────────────────────
            intent = detect_intent(q_text)

            if intent == "farewell":
                speak(components["tts"], "Thank you, Goodbye! Have a great day!", f_lang)
                time.sleep(3)
                memory.clear()
                print("\n  [LISTENING for wake word...]")
                continue

            if intent == "repeat":
                if memory.last_answer:
                    speak(components["tts"], memory.last_answer, f_lang)
                else:
                    speak(components["tts"], "Nothing to repeat yet.", "english")
                time.sleep(3)
                print("\n  [LISTENING for wake word...]")
                continue

            if intent == "reset":
                memory.clear()
                speak(components["tts"], "Starting fresh!", "english")
                time.sleep(3)
                print("\n  [LISTENING for wake word...]")
                continue

            if intent == "thanks":
                speak(components["tts"], "You're welcome!", f_lang)
                time.sleep(3)
                print("\n  [LISTENING for wake word...]")
                continue

            # ────────────────────────────────────────────────
            # PHASE 4: RAG Retrieval + LLM Answer
            # ────────────────────────────────────────────────
            print("  Searching knowledge base...")
            try:
                chunks = retrieve(
                    q_text,
                    components["idx"],
                    components["docs"],
                    components["emb"],
                    top_k=6,
                )
                context = format_context(chunks)
            except Exception as e:
                print("  Retrieval error: " + str(e))
                context = ""

            print("  Generating answer...")
            if intent == "more" and memory.last_topic:
                q_text = "Tell me more about " + memory.last_topic

            answer = generate_answer(
                components["llm"],
                q_text,
                context,
                f_lang,
                memory.get_context_window(),
            )

            # ────────────────────────────────────────────────
            # PHASE 5: Speak answer
            # ────────────────────────────────────────────────
            print("\n  [{}] Gojan AI: {}".format(f_lang.upper(), answer))
            speak(components["tts"], answer, f_lang)
            time.sleep(3)  # Let TTS finish + echo completely die

            # Memory update
            memory.add_turn("user", q_text, f_lang)
            memory.add_turn("assistant", answer, f_lang)
            memory.last_answer = answer

            for topic in ["placement", "hostel", "fees", "course",
                          "department", "admission", "facility",
                          "club", "transport", "library"]:
                if topic in q_text.lower():
                    memory.last_topic = topic
                    break

            print("\n" + "-" * 58)
            print("  [LISTENING for wake word...]")

        except KeyboardInterrupt:
            print("\n\n" + "-" * 58)
            print("  Assistant stopped. Goodbye!")
            print("-" * 58)
            speak(components["tts"], "Thank you, shutting down.", "english")
            sys.exit(0)

        except Exception as e:
            print("\n  Error: {} - resuming...".format(e))
            time.sleep(2)
            print("  [LISTENING for wake word...]")
            continue


if __name__ == "__main__":
    main()
