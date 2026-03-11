"""
Gojan AI Voice Assistant - Interactive Main Loop
==================================================
Boots all models, listens for wake word, processes multilingual
speech, retrieves context, generates answers, speaks responses.

Usage: python phase_b_local/main.py
Debug: python phase_b_local/main.py --debug
"""

import sys
import os
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
from phase_b_local.services.llm import load_model as load_llm, generate_answer
from phase_b_local.services.tts import load_tts, speak


# ---------------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------------
class ConversationMemory:
    """Stores last N exchanges for multi-turn context."""

    MAX_TURNS = 5

    def __init__(self):
        self.turns = []
        self.last_answer = ""

    def add_turn(self, role, text, language="english"):
        self.turns.append({
            "role": role,
            "content": text,
            "language": language,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        max_entries = self.MAX_TURNS * 2
        if len(self.turns) > max_entries:
            self.turns = self.turns[-max_entries:]

    def get_context_window(self):
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(prefix + ": " + t["content"])
        return "\n".join(lines)

    def get_last_language(self):
        for t in reversed(self.turns):
            return t.get("language", "english")
        return "english"

    def clear(self):
        self.turns.clear()
        self.last_answer = ""


# ---------------------------------------------------------------------------
# Conversation State
# ---------------------------------------------------------------------------
class ConversationState:
    def __init__(self):
        self.current_topic = "general"
        self.follow_up_count = 0
        self.user_language_preference = "auto"


# ---------------------------------------------------------------------------
# Intent word lists
# ---------------------------------------------------------------------------
FAREWELL_WORDS = ["bye", "goodbye", "\u0baa\u0bc8", "\u0baa\u0bcb\u0b95\u0ba3\u0bc1\u0bae\u0bcd", "poganum", "thanks bye"]
REPEAT_WORDS = ["repeat", "again", "\u0bae\u0bc0\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd", "meeendum sollu"]
MORE_WORDS = ["more", "details", "\u0b87\u0ba9\u0bcd\u0ba9\u0bc1\u0bae\u0bcd", "innum sollu", "explain"]
RESET_WORDS = ["reset", "new topic", "\u0baa\u0bc1\u0ba4\u0bc1\u0b9a\u0bbe", "puthusha"]
THANKS_WORDS = ["thanks", "thank you", "\u0ba8\u0ba9\u0bcd\u0bb1\u0bbf", "nanri"]


# ---------------------------------------------------------------------------
# FIX 1 — Robust Wake Word Detection
# ---------------------------------------------------------------------------
def detect_wake_word(text):
    """
    Robust wake word detection that handles Whisper transcription
    variations of 'Hey Gojan', 'Gojan', 'Anna' etc.
    Uses multiple strategies so at least one always matches.
    """
    if not text or len(text.strip()) < 2:
        return False

    text_lower = text.lower().strip()
    text_clean = text_lower.replace(",", "").replace(".", "") \
                           .replace("!", "").replace("?", "")

    # STRATEGY 1 — Exact wake word list (most common transcriptions)
    EXACT_WAKE_WORDS = [
        "hey gojan",
        "gojan",
        "hi gojan",
        "hello gojan",
        "anna",
        "hey gojen",
        "go john",
        "go jan",
        "hey go jan",
        "a gojan",
        "okay gojan",
        "ok gojan",
        "gojan ai",
        "\u0b95\u0bcb\u0b9c\u0ba9\u0bcd",
        "ko jan",
        "gojon",
        "gojhan",
        # Extra Whisper variations observed in debug logs:
        "hey goes in",
        "goes in",
        "hey jen",
        "jen",
        "hey golden",
        "golden",
        "hey gorgon",
        "gorgon",
    ]
    for wake in EXACT_WAKE_WORDS:
        if wake in text_clean:
            return True

    # STRATEGY 2 — Fuzzy syllable matching
    words = text_clean.split()
    for i, word in enumerate(words):
        # Check if any word sounds like "gojan" — require 'goj' or 'go' with 'j'
        if word.startswith("goj") or \
           (word.startswith("go") and "j" in word and len(word) >= 4) or \
           (word.startswith("ko") and "j" in word and len(word) >= 4):
            return True
        if i < len(words) - 1:
            bigram = word + " " + words[i + 1]
            if bigram in ["go jan", "go john", "go jon", "go jean",
                          "go chan", "hey jan", "hey john"]:
                return True

    # STRATEGY 3 — Character similarity score (unique chars only)
    TARGET = "gojan"
    target_chars = set(TARGET)
    for word in words:
        if len(word) >= 4:
            word_chars = set(word)
            common = len(word_chars & target_chars)
            score = common / len(target_chars)
            if score >= 0.8:   # 80% unique character overlap
                return True

    # STRATEGY 4 — Levenshtein distance
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (c1 != c2)
                ))
            prev = curr
        return prev[-1]

    for word in words:
        if len(word) >= 4:
            dist = levenshtein(word, "gojan")
            if dist <= 2:
                return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def play_beep():
    """Play a short 200ms beep using numpy + sounddevice."""
    import sounddevice as sd
    duration = 0.2
    freq = 880
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sd.play(wave, samplerate=SAMPLE_RATE)
    sd.wait()


def get_greeting():
    """Return a greeting based on time of day."""
    hour = datetime.datetime.now().hour
    if 6 <= hour < 12:
        return "Good morning! \u0b95\u0bbe\u0bb2\u0bc8 \u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!"
    elif 12 <= hour < 17:
        return "Good afternoon! \u0bae\u0ba4\u0bbf\u0baf \u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!"
    elif 17 <= hour < 21:
        return "Good evening! \u0bae\u0bbe\u0bb2\u0bc8 \u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!"
    else:
        return "Hello! \u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!"


def _any_match(text, word_list):
    """Check if any word in word_list appears in text (case-insensitive)."""
    lower = text.lower()
    return any(w in lower for w in word_list)


# ---------------------------------------------------------------------------
# FIX 3 — Debug Mode
# ---------------------------------------------------------------------------
def run_debug():
    print("\n\u2550\u2550\u2550\u2550\u2550\u2550 WAKE WORD DEBUG MODE \u2550\u2550\u2550\u2550\u2550\u2550")
    print("Testing wake word detection without mic...\n")

    TEST_TRANSCRIPTIONS = [
        ("Hey Gojan",              True),
        ("hey gojan",              True),
        ("gojan",                  True),
        ("Hi Gojan how are you",   True),
        ("go jan tell me",         True),
        ("hey go john",            True),
        ("A Gojan",                True),
        ("okay gojan listen",      True),
        ("gojen",                  True),
        ("gojhan",                 True),
        ("Anna tell me",           True),
        ("\u0b95\u0bcb\u0b9c\u0ba9\u0bcd",                  True),
        ("hey goes in",            True),
        ("hey jen",                True),
        ("hello how are you",      False),
        ("what is the time",       False),
        ("random sentence here",   False),
        ("good morning",           False),
    ]

    passed = 0
    failed = []
    for text, expected in TEST_TRANSCRIPTIONS:
        result = detect_wake_word(text)
        status = "\u2713" if result == expected else "\u2717"
        if result == expected:
            passed += 1
        else:
            failed.append((text, expected, result))
        print(f"  {status} '{text}' \u2192 detected={result} expected={expected}")

    print(f"\nWake word test: {passed}/{len(TEST_TRANSCRIPTIONS)} passed")
    if failed:
        print("Failed cases:")
        for t, e, r in failed:
            print(f"  Input: '{t}' | Expected: {e} | Got: {r}")
    else:
        print("\u2705 Wake word detection working perfectly")
    print("\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Boot Sequence
# ---------------------------------------------------------------------------
def boot():
    print()
    print("\u2554" + "\u2550" * 54 + "\u2557")
    print("\u2551      GOJAN SCHOOL OF BUSINESS AND TECHNOLOGY       \u2551")
    print("\u2551           AI VOICE ASSISTANT v2.0                   \u2551")
    print("\u2551   Powered by TinyLlama + FAISS + Whisper            \u2551")
    print("\u2551   Languages: English | Tamil | Tanglish             \u2551")
    print("\u255a" + "\u2550" * 54 + "\u255d")
    print()

    t0 = time.time()

    print("[1/4] Loading Speech Recognition (Whisper tiny)...")
    stt_model = load_stt()
    print("      \u2713 Done ({:.1f}s)".format(time.time() - t0))

    print("[2/4] Loading Knowledge Base (FAISS)...")
    index, documents, embed_model = load_retriever()
    print("      \u2713 Done \u2014 {} facts loaded".format(len(documents)))

    print("[3/4] Loading Language Model (TinyLlama Q4)...")
    llm = load_llm()
    print("      \u2713 Done ({:.1f}s)".format(time.time() - t0))

    print("[4/4] Loading Voice Output (pyttsx3)...")
    tts_engine = load_tts()
    print("      \u2713 Done")

    total = time.time() - t0
    print("\n\u2705 All systems ready! Total startup: {:.1f}s".format(total))

    return stt_model, index, documents, embed_model, llm, tts_engine


# ---------------------------------------------------------------------------
# Main Interactive Loop (FIX 2 — Updated Wake Word Loop)
# ---------------------------------------------------------------------------
def main():
    # FIX 3 — Debug mode check
    if "--debug" in sys.argv:
        run_debug()

    try:
        stt_model, index, documents, embed_model, llm, tts_engine = boot()
    except FileNotFoundError as e:
        print("\n\u274c " + str(e))
        sys.exit(1)

    memory = ConversationMemory()
    state = ConversationState()

    greeting = get_greeting()
    intro = greeting + " I am the Gojan College AI Assistant. English, Tamil, or Tanglish la pesalaam. Say Hey Gojan!"
    speak(tts_engine, intro)

    print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...")
    print("   (also works: 'Gojan', 'Anna', 'Hi Gojan', Tamil: '\u0b95\u0bcb\u0b9c\u0ba9\u0bcd')\n")

    while True:
        try:
            # === WAKE WORD DETECTION ===
            from scipy.io import wavfile as wf

            audio = record_audio(duration=3)

            # Skip silence
            if not is_speech(audio):
                continue

            # Transcribe wake word attempt
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.write(tmp_path, SAMPLE_RATE, audio_int16)
            try:
                result = transcribe(stt_model, tmp_path)
            finally:
                os.unlink(tmp_path)

            text = result["text"] if isinstance(result, dict) else result

            # Print what was heard for debugging (overwrites same line)
            print(f"   heard: '{text.strip()}'", end="\r")

            # Check wake word with robust detection
            if not detect_wake_word(text):
                continue

            # === WAKE WORD DETECTED ===
            print(f"\n\u2713 Wake word detected! ({text.strip()})")
            play_beep()

            # Now listen for the actual question
            print("\U0001f3a4 Listening for your question... (7 seconds)")
            question_result = listen_and_transcribe(stt_model)

            if question_result is None:
                speak(tts_engine, "I didn't catch that. Say Hey Gojan to try again.")
                print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...\n")
                continue

            question_text = question_result["text"]
            detected_lang = question_result["language"]

            if len(question_text.strip()) < 3:
                if detected_lang == "tanglish":
                    speak(tts_engine, "Kekala? Meeendum sollunga.")
                else:
                    speak(tts_engine, "Please ask your question.")
                print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...\n")
                continue

            print(f"\n\u2753 [{detected_lang.upper()}] You: {question_text}")

            # === INTENT DETECTION ===
            if _any_match(question_text, FAREWELL_WORDS):
                speak(tts_engine, "Bye! Gojan-la serthukkuven. Vanakkam!")
                memory.clear()
                print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...\n")
                continue

            if _any_match(question_text, REPEAT_WORDS):
                if memory.last_answer:
                    speak(tts_engine, memory.last_answer)
                else:
                    speak(tts_engine, "Nothing to repeat.")
                print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...\n")
                continue

            if _any_match(question_text, RESET_WORDS):
                memory.clear()
                speak(tts_engine, "Ok! Fresh start. Enna kekanum?")
                print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...\n")
                continue

            if _any_match(question_text, THANKS_WORDS):
                speak(tts_engine, "You're welcome! Vera enna help venum?")
                print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...\n")
                continue

            # === PROCESS QUESTION ===
            print("\U0001f50d Searching knowledge base...")

            chunks = retrieve(question_text, index, documents, embed_model, top_k=4)
            context = format_context(chunks)
            history = memory.get_context_window()

            print("\U0001f916 Generating answer...")
            answer = generate_answer(
                llm, question_text, context, detected_lang, history
            )

            # Store in memory
            memory.add_turn("user", question_text, detected_lang)
            memory.add_turn("assistant", answer, detected_lang)
            memory.last_answer = answer

            print(f"\U0001f4ac [{detected_lang.upper()}] Gojan AI: {answer}")
            speak(tts_engine, answer)
            print("\u2500" * 55)
            print("\n\U0001f3a4 Say 'Hey Gojan' to wake me up...\n")

        except KeyboardInterrupt:
            speak(tts_engine, "Goodbye! Thank you for using Gojan AI Assistant.")
            print("\n\U0001f44b Assistant stopped.")
            sys.exit(0)
        except Exception as e:
            print(f"\n\u26a0 Error: {e} \u2014 resuming...")
            continue


if __name__ == "__main__":
    main()
