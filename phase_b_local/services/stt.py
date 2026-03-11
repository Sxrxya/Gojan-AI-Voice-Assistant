"""
Phase B - Speech-to-Text Service (CPU Only)
=============================================
Whisper tiny + language detection + Tanglish detection.
"""

import os
import re
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WHISPER_MODEL = "tiny"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16000

# Tanglish detection word list
TANGLISH_WORDS = [
    "iruku", "illa", "sollu", "pathi", "enna", "evvalavu",
    "nalla", "konjam", "theriyum", "pannanum", "mattum",
    "varuvanga", "pottu", "paaru", "kelu", "seri", "da", "di",
    "aama", "romba", "thaan", "anga", "inga", "appadi",
    "ippo", "eppadi", "eppo", "ennaku", "unnaku", "avalavu",
    "pannu", "panna", "irukum", "iruka", "sollunge", "solren",
    "poganum", "vanakkam", "nanri", "puthusha", "meeendum",
]

# Module state
_model = None


def load_model():
    """Load Whisper tiny model (CPU, int8)."""
    global _model
    if _model is not None:
        return _model
    print(f"  Loading Whisper '{WHISPER_MODEL}' (CPU, int8)...")
    _model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def record_audio(duration=6, sample_rate=SAMPLE_RATE):
    """Record audio from microphone for given duration."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def is_speech(audio_data, threshold=0.01):
    """Check if audio contains speech above noise threshold."""
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms > threshold


def transcribe(model, audio_path):
    """Transcribe audio file, return text and detected language."""
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(s.text for s in segments).strip()
    lang_code = info.language if info.language else "en"
    return {"text": text, "language": lang_code}


def detect_tanglish(text):
    """Detect if text is Tanglish (Tamil words in English script)."""
    if not text:
        return None
    words = text.lower().split()
    matches = sum(1 for w in words if w in TANGLISH_WORDS)
    if matches >= 2:
        return "tanglish"
    return None


def _detect_language(text, whisper_lang):
    """Map Whisper language code + Tanglish detection to final language."""
    tg = detect_tanglish(text)
    if tg:
        return "tanglish"
    if whisper_lang == "ta":
        return "tamil"
    return "english"


def listen_and_transcribe(model):
    """Record, check speech, transcribe, detect language. Returns dict or None."""
    audio = record_audio(duration=6)

    if not is_speech(audio):
        return None

    # Save to temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(tmp_path, SAMPLE_RATE, audio_int16)
        result = transcribe(model, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not result["text"] or len(result["text"].strip()) < 3:
        return None

    lang = _detect_language(result["text"], result["language"])
    return {"text": result["text"], "language": lang}
