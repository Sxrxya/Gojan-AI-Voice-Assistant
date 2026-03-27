"""
stt.py — Speech-to-Text Service (100% Offline Edition)
=====================================================
Uses speech_recognition's Microphone for dynamic capture
(auto-stops when you stop talking).
Routes the raw audio bytes exclusively to local Whisper 'base'.
0% internet usage.
"""

import io
import time
import logging
import speech_recognition as sr
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

TANGLISH_WORDS = [
    "iruku", "illa", "sollu", "pathi", "enna", "evvalavu",
    "nalla", "konjam", "theriyum", "pannanum", "mattum",
    "varuvanga", "pottu", "paaru", "kelu", "seri", "da", "di",
    "aama", "romba", "thaan", "anga", "inga", "appadi",
    "ippo", "eppadi", "eppo", "ennaku", "unnaku", "avalavu",
    "pannu", "panna", "irukum", "iruka", "sollunge", "solren",
    "poganum", "vanakkam", "nanri", "puthusha", "meeendum",
]

_whisper_model = None


def load_model():
    """Load Whisper once and keep it in memory."""
    global _whisper_model
    if _whisper_model is None:
        print("  Loading Whisper 'base' model for STT (100% offline)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def detect_tanglish(text):
    if not text:
        return None
    words = text.lower().split()
    matches = sum(1 for w in words if w in TANGLISH_WORDS)
    return "tanglish" if matches >= 2 else None


def detect_language(text, base_lang="en"):
    tg = detect_tanglish(text)
    if tg:
        return "tanglish"
    if base_lang == "ta":
        return "tamil"
    return "english"


def _transcribe_offline(audio_data: sr.AudioData):
    """Passes sr.AudioData directly to Whisper without disk writing."""
    global _whisper_model
    if not _whisper_model:
        return None

    # Get raw WAV bytes in memory and wrap in BytesIO so Whisper can read it
    wav_bytes = audio_data.get_wav_data()
    audio_file = io.BytesIO(wav_bytes)

    try:
        # Give Whisper a bigger hint of words to expect so it auto-corrects mispronunciations
        # Note: We cannot pass "all scraped data" here because Whisper has a maximum 200-word limit for hints.
        domain_hint = "Gojan School of Business and Technology, Anna University. TNEA code 1123. Courses: CSE, Computer Science, ECE, Electronics, AI, Artificial Intelligence, ML, Machine Learning, IT, Mechanical, Aeronautical. Questions about hostel, fees, admission process, placement records, salary packages, transport, bus routes, library facilities, departments, principal, address, redhills, chennai."
        
        segments, _ = _whisper_model.transcribe(
            audio_file, beam_size=3, temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=domain_hint
        )
        text = " ".join(s.text for s in segments).strip()
    except Exception as e:
        print(f"Whisper inference error: {e}")
        return None

    # Filter known Whisper base model hallucinations
    clean = text.lower().strip().rstrip(".!?,;:")
    if clean in ["thank you", "thanks for watching", "all right", "you",
                 "okay", "bye", "oh", "um", "ah", "hmm", "so"] or len(clean) < 3:
        return None

    return text


def listen_for_wake_word(timeout=5):
    """Listens dynamically from Mic, transcribes offline using local Whisper."""
    r = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=4)
        
        # 100% Offline Transcription
        return _transcribe_offline(audio)
    except sr.WaitTimeoutError:
        return None
    except Exception:
        return None


def listen_for_question(timeout=8):
    """Listens dynamically for question, transcribes offline using local Whisper."""
    r = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=15)

        # 100% Offline Transcription
        text = _transcribe_offline(audio)

        if not text or len(text.strip()) < 3:
            return None

        lang = detect_language(text, "en")
        return {"text": text.strip(), "language": lang}
    except sr.WaitTimeoutError:
        return None
    except Exception:
        return None
