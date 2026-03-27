"""
stt.py — Speech-to-Text Service (Simple & Reliable)
=====================================================
Uses Google Speech Recognition API as primary (perfect accuracy).
Falls back to Whisper base for offline mode.
No complex dependencies — just works.
"""

import io
import os
import re
import tempfile
import logging
import wave

import numpy as np

# Google Speech Recognition (primary)
try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False

# Whisper (offline fallback)
try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Tanglish word list for language detection
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
    """Load Whisper for offline fallback. Returns the model."""
    global _whisper_model
    if _WHISPER_AVAILABLE and _whisper_model is None:
        print("  Loading Whisper 'base' (offline fallback)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    if _SR_AVAILABLE:
        print("  Google Speech API: Ready")
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


def listen_for_wake_word(timeout=4):
    """
    Listen for the wake word using Google Speech Recognition.
    Returns the transcribed text or None if nothing heard.
    Uses dynamic listening — stops when you stop talking.
    """
    if not _SR_AVAILABLE:
        return None

    r = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            # Removed artificial threshold floor to allow listening in quiet rooms
            audio = r.listen(source, timeout=timeout, phrase_time_limit=3)

        text = r.recognize_google(audio, language="en-IN")
        return text.strip()
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        # Internet down — try Whisper fallback
        return _whisper_listen(timeout)
    except Exception:
        return None


def listen_for_question(timeout=8):
    """
    Listen for a full question using Google Speech Recognition.
    Returns dict {text, language} or None.
    Dynamically stops when user finishes speaking.
    """
    if not _SR_AVAILABLE:
        return _whisper_question(timeout)

    r = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            # Removed artificial threshold floor to allow listening in quiet rooms
            audio = r.listen(source, timeout=timeout, phrase_time_limit=15)

        text = r.recognize_google(audio, language="en-IN")
        if not text or len(text.strip()) < 3:
            return None

        lang = detect_language(text, "en")
        return {"text": text.strip(), "language": lang}
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return _whisper_question(timeout)
    except Exception:
        return None


def _whisper_listen(timeout=4):
    """Offline fallback for wake word using Whisper."""
    global _whisper_model
    if not _whisper_model:
        return None

    import sounddevice as sd
    from scipy.io import wavfile

    try:
        audio = sd.rec(int(timeout * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        # Check if there's actual sound
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.008:
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(tmp_path, SAMPLE_RATE, audio_int16)

        try:
            segments, _ = _whisper_model.transcribe(
                tmp_path, beam_size=3, temperature=0.0,
                condition_on_previous_text=False
            )
            text = " ".join(s.text for s in segments).strip()
        finally:
            os.unlink(tmp_path)

        # Filter known hallucinations
        clean = text.lower().strip().rstrip(".!?,;:")
        if clean in ["thank you", "thanks for watching", "all right", "you",
                      "okay", "bye", "oh", "um", "ah", "hmm", "so"] or len(clean) < 3:
            return None

        return text
    except Exception:
        return None


def _whisper_question(timeout=8):
    """Offline fallback for questions using Whisper."""
    text = _whisper_listen(timeout)
    if text and len(text.strip()) >= 3:
        lang = detect_language(text, "en")
        return {"text": text.strip(), "language": lang}
    return None
