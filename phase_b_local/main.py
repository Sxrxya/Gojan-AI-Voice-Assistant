"""
Gojan AI Voice Assistant - Interactive Main Loop v2.1
======================================================
Clean startup, reliable loop, hybrid Indian voice.
Runs from project root: python phase_b_local/main.py
Debug mode:              python phase_b_local/main.py --debug
"""

import sys
import os
import re
import time
import datetime
import tempfile
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_b_local.services.stt import (
    load_model as load_stt, record_audio, is_speech,
    transcribe, detect_tanglish, listen_and_transcribe,
    SAMPLE_RATE,
)
from phase_b_local.services.retriever import load_retriever, retrieve, format_context
from phase_b_local.services.llm import (
    load_model as load_llm, generate_answer, get_fallback,
)
from phase_b_local.services.tts import load_tts, speak, play_beep


# =====================================================================
# Conversation Memory
# =====================================================================
class ConversationMemory:
    MAX_TURNS = 5

    def __init__(self):
        self.turns = []
        self.last_answer = ""
        self.last_topic = None

    def add_turn(self, role, text, language="english"):
        self.turns.append({
            "role": role,
            "content": text,
            "language": language,
            "time": time.time(),
        })
        cap = self.MAX_TURNS * 2
        if len(self.turns) > cap:
            self.turns = self.turns[-cap:]

    def get_context_window(self):
        if not self.turns:
            return ""
        recent = self.turns[-6:]
        return "\n".join(
            t["role"].upper() + ": " + t["content"]
            for t in recent
        )

    def get_last_language(self):
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t.get("language", "english")
        return "english"

    def clear(self):
        self.turns.clear()
        self.last_answer = ""
        self.last_topic = None


# =====================================================================
# Intent Detection
# =====================================================================
FAREWELL_WORDS = ["bye", "goodbye", "stop", "exit", "close",
                  "poganum", "muppu"]
REPEAT_WORDS   = ["repeat", "again", "meeendum", "sollu again"]
MORE_WORDS     = ["more", "details", "explain", "innum", "tell me more"]
RESET_WORDS    = ["reset", "new topic", "puthusha", "clear", "restart"]
THANKS_WORDS   = ["thanks", "thank you", "nanri"]


def _any_match(text, word_list):
    lower = text.lower()
    return any(w in lower for w in word_list)


def detect_intent(text):
    if _any_match(text, FAREWELL_WORDS):
        return "farewell"
    if _any_match(text, REPEAT_WORDS):
        return "repeat"
    if _any_match(text, MORE_WORDS):
        return "more"
    if _any_match(text, RESET_WORDS):
        return "reset"
    if _any_match(text, THANKS_WORDS):
        return "thanks"
    return "question"


# =====================================================================
# Wake Word Detection (4-strategy, tested 18/18)
# =====================================================================
EXACT_WAKE_WORDS = [
    "hey gojan", "gojan", "hi gojan", "hello gojan", "anna",
    "hey gojen", "go john", "go jan", "hey go jan", "a gojan",
    "okay gojan", "ok gojan", "gojan ai", "ko jan", "gojon",
    "gojhan", "hey goes in", "goes in", "hey jen", "jen",
    "hey golden", "golden", "hey gorgon", "gorgon",
]


def detect_wake_word(text):
    if not text or len(text.strip()) < 2:
        return False

    text_lower = text.lower().strip()
    text_clean = re.sub(r"[^\w\s]", "", text_lower)

    # Strategy 1 - Exact matches
    for wake in EXACT_WAKE_WORDS:
        if wake in text_clean:
            return True

    # Strategy 2 - Fuzzy syllable matching
    words = text_clean.split()
    for i, word in enumerate(words):
        if word.startswith("goj") or \
           (word.startswith("go") and "j" in word and len(word) >= 4) or \
           (word.startswith("ko") and "j" in word and len(word) >= 4):
            return True
        if i < len(words) - 1:
            bigram = word + " " + words[i + 1]
            if bigram in ["go jan", "go john", "go jon", "go jean",
                          "go chan", "hey jan", "hey john"]:
                return True

    # Strategy 3 - Character similarity (unique chars, 80%)
    target_chars = set("gojan")
    for word in words:
        if len(word) >= 4:
            common = len(set(word) & target_chars)
            if common / len(target_chars) >= 0.8:
                return True

    # Strategy 4 - Levenshtein distance <= 2
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
        if len(word) >= 4 and _lev(word, "gojan") <= 2:
            return True

    return False


# =====================================================================
# Debug Mode
# =====================================================================
def run_debug():
    print("\n" + "=" * 40)
    print("  WAKE WORD DEBUG MODE")
    print("=" * 40 + "\n")

    tests = [
        ("Hey Gojan", True), ("hey gojan", True), ("gojan", True),
        ("Hi Gojan how are you", True), ("go jan tell me", True),
        ("hey go john", True), ("A Gojan", True),
        ("okay gojan listen", True), ("gojen", True),
        ("gojhan", True), ("Anna tell me", True),
        ("hey goes in", True), ("hey jen", True),
        ("hello how are you", False), ("what is the time", False),
        ("random sentence here", False), ("good morning", False),
        ("thank you very much", False),
    ]

    passed = 0
    failed = []
    for text, expected in tests:
        result = detect_wake_word(text)
        ok = result == expected
        mark = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed.append((text, expected, result))
        print(f"  [{mark}] '{text}' -> detected={result} expected={expected}")

    print(f"\nResult: {passed}/{len(tests)} passed")
    if failed:
        print("Failed:")
        for t, e, r in failed:
            print(f"  '{t}' expected={e} got={r}")
    else:
        print("All tests passed!")
    print("=" * 40 + "\n")
    sys.exit(0)


# =====================================================================
# Greeting
# =====================================================================
def get_greeting():
    hour = datetime.datetime.now().hour
    if 6 <= hour < 12:
        return "Good morning! I am Gojan AI Assistant. Say Hey Gojan to ask me anything about Gojan College."
    elif 12 <= hour < 17:
        return "Good afternoon! I am Gojan AI Assistant. Say Hey Gojan to ask me anything about Gojan College."
    elif 17 <= hour < 21:
        return "Good evening! I am Gojan AI Assistant. Say Hey Gojan to ask me anything about Gojan College."
    else:
        return "Hello! I am Gojan AI Assistant. Say Hey Gojan to ask me anything about Gojan College."


# =====================================================================
# Boot Sequence
# =====================================================================
def boot():
    print()
    print("=" * 58)
    print("  GOJAN SCHOOL OF BUSINESS AND TECHNOLOGY")
    print("  AI VOICE ASSISTANT v2.1")
    print("  English | Tamil | Tanglish  -  Offline + Indian Voice")
    print("=" * 58)
    print()

    t0 = time.time()
    components = {}
    failed = []

    # 1 - STT
    print("[1/4] Loading Speech Recognition (Whisper tiny)...")
    try:
        components["stt"] = load_stt()
        print("      Done ({:.1f}s)".format(time.time() - t0))
    except Exception as e:
        print("      FAILED: " + str(e))
        failed.append("STT")

    # 2 - FAISS
    print("[2/4] Loading Knowledge Base (FAISS)...")
    try:
        idx, docs, emb = load_retriever()
        components["idx"] = idx
        components["docs"] = docs
        components["emb"] = emb
        print("      Done - {} facts loaded".format(len(docs)))
    except Exception as e:
        print("      FAILED: " + str(e))
        failed.append("FAISS")

    # 3 - LLM
    print("[3/4] Loading Language Model (TinyLlama Q4)...")
    try:
        components["llm"] = load_llm()
        print("      Done ({:.1f}s)".format(time.time() - t0))
    except Exception as e:
        print("      FAILED: " + str(e))
        failed.append("LLM")

    # 4 - TTS
    print("[4/4] Loading Voice System...")
    try:
        components["tts"] = load_tts()
    except Exception as e:
        print("      FAILED: " + str(e))
        failed.append("TTS")

    if failed:
        print("\nFailed components: " + ", ".join(failed))
        if "LLM" in failed:
            print("Critical: LLM not loaded. Place gojan_ai_q4.gguf in models/gguf/")
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
    # Debug mode check
    if "--debug" in sys.argv:
        run_debug()

    components = boot()
    memory = ConversationMemory()

    # Greeting
    greeting = get_greeting()
    speak(components["tts"], greeting, "english")

    print("\n  Say 'Hey Gojan' to wake me up")
    print("  Alternatives: 'Gojan' | 'Anna' | 'Hey Gojan'")
    print("  Languages: English | Tamil | Tanglish")
    print("  Say 'bye' to exit\n")
    print("=" * 58 + "\n")

    from scipy.io import wavfile as wf

    miss_count = 0

    while True:
        try:
            # ── Wake word phase ──────────────────────────
            audio = record_audio(duration=3)

            if not is_speech(audio):
                continue

            # Save to temp WAV for Whisper
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.write(tmp_path, SAMPLE_RATE, audio_int16)

            try:
                result = transcribe(components["stt"], tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            heard = result["text"].strip() if result else ""
            print("   heard: '{}'          ".format(heard), end="\r")

            if not detect_wake_word(heard):
                miss_count += 1
                if miss_count % 5 == 0:
                    print("\n  Tip: Try saying 'Gojan' or 'Anna'          ")
                continue

            # ── Wake confirmed ───────────────────────────
            miss_count = 0
            print("\n>> Activated! ('{}')".format(heard))
            play_beep()

            # ── Question phase ───────────────────────────
            print("   Listening for your question... (7 seconds)")
            question_result = listen_and_transcribe(components["stt"])

            if question_result is None:
                speak(components["tts"],
                      "I didn't catch that. Say Hey Gojan to try again.",
                      "english")
                print("\n  Say 'Hey Gojan' to wake me up...\n")
                continue

            q_text = question_result["text"]
            f_lang = question_result["language"]

            if len(q_text.strip()) < 3:
                speak(components["tts"],
                      "Please ask your question clearly.",
                      "english")
                print("\n  Say 'Hey Gojan' to wake me up...\n")
                continue

            print("\n  [{}] You: {}".format(f_lang.upper(), q_text))

            # ── Intent detection ─────────────────────────
            intent = detect_intent(q_text)

            if intent == "farewell":
                farewell = {
                    "english":  "Goodbye! Have a great day!",
                    "tanglish": "Bye! Nalla irrunga!",
                    "tamil":    "Goodbye! Nandri!",
                }.get(f_lang, "Goodbye!")
                speak(components["tts"], farewell, f_lang)
                memory.clear()
                print("\n" + "-" * 58 + "\n")
                print("  Say 'Hey Gojan' to wake me up...\n")
                continue

            if intent == "repeat":
                if memory.last_answer:
                    print("  Repeating last answer...")
                    speak(components["tts"], memory.last_answer, f_lang)
                else:
                    speak(components["tts"],
                          "Nothing to repeat yet.", "english")
                print("\n  Say 'Hey Gojan' to wake me up...\n")
                continue

            if intent == "reset":
                memory.clear()
                speak(components["tts"],
                      "Starting fresh! Ask me anything about Gojan college.",
                      "english")
                print("\n  Say 'Hey Gojan' to wake me up...\n")
                continue

            if intent == "thanks":
                speak(components["tts"],
                      "You're welcome! Vera enna help venum?",
                      f_lang)
                print("\n  Say 'Hey Gojan' to wake me up...\n")
                continue

            # ── RAG Retrieval ────────────────────────────
            print("  Searching knowledge base...")
            try:
                chunks = retrieve(
                    q_text,
                    components["idx"],
                    components["docs"],
                    components["emb"],
                    top_k=4,
                )
                context = format_context(chunks)
            except Exception as e:
                print("  Retrieval error: " + str(e))
                context = ""

            # ── Generate answer ──────────────────────────
            print("  Generating answer...")
            history = memory.get_context_window()

            if intent == "more" and memory.last_topic:
                q_text = "Tell me more about " + memory.last_topic

            answer = generate_answer(
                components["llm"],
                q_text,
                context,
                f_lang,
                history,
            )

            # ── Output ──────────────────────────────────
            print("\n  [{}] Gojan AI: {}".format(f_lang.upper(), answer))
            speak(components["tts"], answer, f_lang)

            # ── Memory update ───────────────────────────
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
            print("  Say 'Hey Gojan' to wake me up...\n")

        except KeyboardInterrupt:
            print("\n\n" + "-" * 58)
            speak(components["tts"],
                  "Goodbye! Thank you for using Gojan AI Assistant.",
                  "english")
            print("  Assistant stopped.\n")
            sys.exit(0)

        except Exception as e:
            print("\n  Error: {} - resuming...".format(e))
            time.sleep(1)
            continue


if __name__ == "__main__":
    main()
