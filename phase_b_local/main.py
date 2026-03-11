"""
Gojan AI Voice Assistant - Interactive Main Loop
==================================================
Boots all models, listens for wake word, processes multilingual
speech, retrieves context, generates answers, speaks responses.

Usage: cd phase_b_local && python main.py
"""

import sys
import os
import time
import datetime
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
        # Cap at MAX_TURNS exchanges (2 entries per exchange)
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
# Wake Words
# ---------------------------------------------------------------------------
WAKE_WORDS = ["hey gojan", "gojan", "\u0b95\u0bcb\u0b9c\u0ba9\u0bcd", "anna", "hi gojan"]

FAREWELL_WORDS = ["bye", "goodbye", "\u0baa\u0bc8", "\u0baa\u0bcb\u0b95\u0ba3\u0bc1\u0bae\u0bcd", "poganum", "thanks bye"]
REPEAT_WORDS = ["repeat", "again", "\u0bae\u0bc0\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd", "meeendum sollu"]
MORE_WORDS = ["more", "details", "\u0b87\u0ba9\u0bcd\u0ba9\u0bc1\u0bae\u0bcd", "innum sollu", "explain"]
RESET_WORDS = ["reset", "new topic", "\u0baa\u0bc1\u0ba4\u0bc1\u0b9a\u0bbe", "puthusha"]
THANKS_WORDS = ["thanks", "thank you", "\u0ba8\u0ba9\u0bcd\u0bb1\u0bbf", "nanri"]


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
# Main Interactive Loop
# ---------------------------------------------------------------------------
def main():
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

    print("\n" + "\u2550" * 55)
    print(" Say 'Hey Gojan' to start. Press Ctrl+C to exit.")
    print("\u2550" * 55 + "\n")

    while True:
        try:
            # === WAKE WORD DETECTION ===
            print("(waiting for wake word: say 'Hey Gojan')")
            import tempfile
            from scipy.io import wavfile as wf

            wake_audio = record_audio(duration=3)
            if not is_speech(wake_audio):
                continue

            # Transcribe short clip for wake word
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            audio_int16 = (wake_audio * 32767).astype(np.int16)
            wf.write(tmp_path, SAMPLE_RATE, audio_int16)
            try:
                wake_result = transcribe(stt_model, tmp_path)
            finally:
                os.unlink(tmp_path)

            wake_text = wake_result["text"].lower().strip()
            if not _any_match(wake_text, WAKE_WORDS):
                continue

            # === WAKE WORD DETECTED ===
            play_beep()
            print("\U0001f3a4 Listening for your question...")

            result = listen_and_transcribe(stt_model)
            if result is None:
                speak(tts_engine, "Kekala? Please ask your question.")
                continue

            question_text = result["text"]
            detected_lang = result["language"]

            if len(question_text.strip()) < 3:
                if detected_lang == "tanglish":
                    speak(tts_engine, "Kekala? Meeendum sollunga.")
                else:
                    speak(tts_engine, "Please ask your question.")
                continue

            # === INTENT DETECTION ===
            if _any_match(question_text, FAREWELL_WORDS):
                speak(tts_engine, "Bye! Gojan-la serthukkuven. Vanakkam!")
                memory.clear()
                continue

            if _any_match(question_text, REPEAT_WORDS):
                if memory.last_answer:
                    speak(tts_engine, memory.last_answer)
                else:
                    speak(tts_engine, "Nothing to repeat.")
                continue

            if _any_match(question_text, RESET_WORDS):
                memory.clear()
                speak(tts_engine, "Ok! Fresh start. Enna kekanum?")
                continue

            if _any_match(question_text, THANKS_WORDS):
                speak(tts_engine, "You're welcome! Vera enna help venum?")
                continue

            # === PROCESS QUESTION ===
            print("\n\u2753 [{}] {}".format(detected_lang.upper(), question_text))
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

            print("\U0001f4ac [{}] Gojan AI: {}".format(detected_lang.upper(), answer))
            speak(tts_engine, answer)
            print("\u2500" * 55 + "\n")

        except KeyboardInterrupt:
            speak(tts_engine, "Goodbye! Thank you for using Gojan AI Assistant.")
            print("\n\U0001f44b Assistant stopped.")
            sys.exit(0)
        except Exception as e:
            print("\u26a0  Error: {} \u2014 continuing...\n".format(e))
            continue


if __name__ == "__main__":
    main()
