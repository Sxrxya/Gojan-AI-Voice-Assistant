"""
Gojan AI Voice Assistant - Interactive Main Loop v3.0 (Async/Streaming)
=======================================================================
True Google Assistant-like architecture:
- Continuous background PyAudio microphone stream
- OpenWakeWord precise neural wake detection (No STT required)
- WebRTC-style Python RMS Voice Activity Detection (VAD)
- Hard mic mute during TTS to eliminate feedback loop
"""

import sys
import os
import time
import queue
import threading
from enum import Enum, auto

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_b_local.services.stt import (
    MicrophoneStream, SpeechCapture, Transcriber, WakeWordDetector,
    mute_microphone, unmute_microphone, tts_playing, _detect_language
)
from phase_b_local.services.retriever import load_retriever, retrieve, format_context
from phase_b_local.services.llm import load_model as load_llm, generate_answer
from phase_b_local.services.tts import load_tts, speak, play_beep


# =====================================================================
# Conversation Memory & Intents
# =====================================================================
class ConversationMemory:
    MAX_TURNS = 5

    def __init__(self):
        self.turns = []
        self.last_answer = ""
        self.last_topic = None

    def add_turn(self, role, text, language="english"):
        self.turns.append({"role": role, "content": text, "language": language})
        cap = self.MAX_TURNS * 2
        if len(self.turns) > cap:
            self.turns = self.turns[-cap:]

    def get_context_window(self):
        if not self.turns:
            return ""
        return "\n".join(t["role"].upper() + ": " + t["content"] for t in self.turns[-6:])

    def clear(self):
        self.turns.clear()
        self.last_answer = ""
        self.last_topic = None


def detect_intent(text):
    lower = text.lower()
    if any(w in lower for w in ["bye", "goodbye", "stop", "exit", "close"]):
        return "farewell"
    if any(w in lower for w in ["repeat", "again", "meeendum"]):
        return "repeat"
    if any(w in lower for w in ["more", "details", "explain", "innum"]):
        return "more"
    if any(w in lower for w in ["reset", "new topic", "puthusha", "clear"]):
        return "reset"
    if any(w in lower for w in ["thanks", "thank you", "nanri"]):
        return "thanks"
    return "question"


# =====================================================================
# Boot Sequence
# =====================================================================
def boot():
    print("\n" + "=" * 58)
    print("  GOJAN SCHOOL OF BUSINESS AND TECHNOLOGY")
    print("  AI VOICE ASSISTANT v3.0 (Streaming Async)")
    print("=" * 58 + "\n")

    t0 = time.time()
    components = {}

    print("[1/3] Loading Knowledge Base (FAISS)...")
    idx, docs, emb = load_retriever()
    components["idx"] = idx
    components["docs"] = docs
    components["emb"] = emb

    print("[2/3] Loading Language Model (TinyLlama Q4)...")
    components["llm"] = load_llm()

    print("[3/3] Loading Voice System...")
    components["tts"] = load_tts()

    print("-" * 58)
    print(f"System ready! Data/LLM Startup: {time.time() - t0:.1f}s")
    print("-" * 58)
    return components


# =====================================================================
# Async Assistant State Machine
# =====================================================================
class State(Enum):
    IDLE      = auto()
    LISTENING = auto()
    THINKING  = auto()
    SPEAKING  = auto()


class GojanAssistant:
    OWW_FRAME_SAMPLES = 1280  # 80ms chunks for OpenWakeWord

    def __init__(self, components):
        self.components = components
        self.memory = ConversationMemory()
        self.state = State.IDLE
        
        # Audio Pipeline
        print("Initializing Async Audio Pipeline...")
        self.wake = WakeWordDetector()
        self.capture = SpeechCapture()
        self.stt = Transcriber()

        # Background worker thread for LLM processing
        self._work_q = queue.Queue()
        self._worker = threading.Thread(target=self._pipeline_worker, daemon=True)
        self._worker.start()

        self._running = False

    def safe_speak(self, text, lang="english"):
        """Wrapper around TTS to guarantee hardware microphone mute."""
        mute_microphone()
        try:
            speak(self.components["tts"], text, lang)
        finally:
            time.sleep(0.3)  # Drain echo buffer
            unmute_microphone()

    def run(self):
        self._running = True
        
        # Initial Greeting
        self.safe_speak(
            "Good morning! I am Gojan AI Assistant. Say Hey Jarvis to wake me up.", 
            "english"
        )
        print("\n  Say 'Hey Jarvis' to wake me up")
        print("  (Using 'hey jarvis' model as closest match to 'Hey Gojan')")
        print("  Say 'bye' to exit\n" + "=" * 58 + "\n")

        try:
            with MicrophoneStream() as mic:
                self._main_loop(mic)
        except KeyboardInterrupt:
            print("\nShutting down. Goodbye!")

    def _main_loop(self, mic):
        """Continuous stream reading loop. Does not block!"""
        oww_buffer = b""

        while self._running:
            raw_frame = mic.read_frame()

            if tts_playing.is_set():
                oww_buffer = b""
                continue

            if self.state == State.IDLE:
                oww_buffer += raw_frame
                if len(oww_buffer) >= self.OWW_FRAME_SAMPLES * 2:
                    detected = self.wake.process_frame(oww_buffer[:self.OWW_FRAME_SAMPLES * 2])
                    oww_buffer = oww_buffer[self.OWW_FRAME_SAMPLES * 2:]
                    if detected:
                        self._on_wake_detected(mic)

    def _on_wake_detected(self, mic):
        self.state = State.LISTENING
        print("\n>> Activated! Listening for your question...")
        play_beep()

        chunk = self.capture.listen_for_utterance(mic.stream)

        if chunk is None:
            print("  [No speech detected - returning to sleep]")
            self.state = State.IDLE
            return

        self.state = State.THINKING
        print("  [Thinking...]")
        self._work_q.put(chunk)

    def _pipeline_worker(self):
        """Background thread to process STT and RAG without blocking the mic loop."""
        while True:
            chunk = self._work_q.get()
            try:
                self._process(chunk)
            except Exception as e:
                print(f"  [Pipeline Error]: {e}")
                self.safe_speak("Sorry, I encountered an error.", "english")
            finally:
                self.state = State.IDLE
                print("\n  Say 'Hey Jarvis' to wake me up...")

    def _process(self, chunk):
        raw_text = self.stt.transcribe(chunk)
        if not raw_text:
            return

        f_lang = _detect_language(raw_text, "en")
        print(f"\n  [{f_lang.upper()}] You: {raw_text}")

        intent = detect_intent(raw_text)

        if intent == "farewell":
            self.state = State.SPEAKING
            self.safe_speak("Goodbye! Have a great day!", f_lang)
            self._running = False
            return

        if intent == "reset":
            self.memory.clear()
            self.state = State.SPEAKING
            self.safe_speak("Starting fresh! Ask me anything.", "english")
            return

        if intent == "repeat":
            self.state = State.SPEAKING
            self.safe_speak(self.memory.last_answer or "Nothing to repeat.", f_lang)
            return

        if intent == "thanks":
            self.state = State.SPEAKING
            self.safe_speak("You're welcome!", f_lang)
            return

        # RAG Retrieval
        if intent == "more" and self.memory.last_topic:
            raw_text = "Tell me more about " + self.memory.last_topic

        try:
            chunks = retrieve(
                raw_text,
                self.components["idx"],
                self.components["docs"],
                self.components["emb"],
                top_k=4,
            )
            context = format_context(chunks)
        except Exception as e:
            print("  Retrieval error: " + str(e))
            context = ""

        # LLM Generation
        answer = generate_answer(
            self.components["llm"],
            raw_text,
            context,
            f_lang,
            self.memory.get_context_window(),
        )

        print(f"\n  [{f_lang.upper()}] Gojan AI: {answer}")

        # TTS Output
        self.state = State.SPEAKING
        self.safe_speak(answer, f_lang)

        # Update Memory
        self.memory.add_turn("user", raw_text, f_lang)
        self.memory.add_turn("assistant", answer, f_lang)
        self.memory.last_answer = answer

        for topic in ["placement", "hostel", "fees", "course", "department", "admission"]:
            if topic in raw_text.lower():
                self.memory.last_topic = topic
                break


if __name__ == "__main__":
    components = boot()
    assistant = GojanAssistant(components)
    assistant.run()
