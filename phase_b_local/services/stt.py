"""
stt.py — Production-grade Speech-to-Text & Wake Word Service
=============================================================
Architecture:
  1. Continuous background mic stream (PyAudio)
  2. OpenWakeWord ("hey jarvis" model, closest to "Hey Gojan")
  3. WebRTC-style Python RMS VAD captures live speech frames after wake
  4. Google STT (en-IN) → faster-whisper offline fallback
  5. Hard mic mute while TTS is playing
"""

import io
import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np
import pyaudio

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False
    logging.warning("speech_recognition not installed — Google STT unavailable")

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not installed — offline fallback unavailable")

try:
    from openwakeword.model import Model as OWWModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWW_AVAILABLE = False
    logging.warning("openwakeword not installed — wake word detection unavailable")

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
RATE          = 16000
CHANNELS      = 1
FRAME_MS      = 30
FRAME_SAMPLES = int(RATE * FRAME_MS / 1000)
FRAME_BYTES   = FRAME_SAMPLES * 2

RMS_THRESHOLD = 0.008
SILENCE_FRAMES_END = 40
MAX_UTTERANCE_FRAMES = 600
OWW_THRESHOLD = 0.5
WHISPER_MODEL_SIZE = "base"

# TTS Mute Gate
tts_playing = threading.Event()

def mute_microphone():
    tts_playing.set()

def unmute_microphone():
    tts_playing.clear()


TANGLISH_WORDS = [
    "iruku", "illa", "sollu", "pathi", "enna", "evvalavu",
    "nalla", "konjam", "theriyum", "pannanum", "mattum",
    "varuvanga", "pottu", "paaru", "kelu", "seri", "da", "di",
    "aama", "romba", "thaan", "anga", "inga", "appadi",
    "ippo", "eppadi", "eppo", "ennaku", "unnaku", "avalavu",
    "pannu", "panna", "irukum", "iruka", "sollunge", "solren",
    "poganum", "vanakkam", "nanri", "puthusha", "meeendum",
]

def detect_tanglish(text):
    if not text:
        return None
    words = text.lower().split()
    matches = sum(1 for w in words if w in TANGLISH_WORDS)
    if matches >= 2:
        return "tanglish"
    return None

def _detect_language(text, base_lang):
    tg = detect_tanglish(text)
    if tg:
        return "tanglish"
    if base_lang == "ta":
        return "tamil"
    return "english"

@dataclass
class AudioChunk:
    frames: list = field(default_factory=list)
    sample_rate: int = RATE

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
        return buf.getvalue()

class WakeWordDetector:
    MODEL_NAME = "hey_jarvis"

    def __init__(self):
        if not _OWW_AVAILABLE:
            raise RuntimeError("openwakeword is not installed.")
        logger.info(f"[WakeWord] Loading OWW model: {self.MODEL_NAME}")
        self._model = OWWModel(
            wakeword_models=[self.MODEL_NAME],
            enable_speex_noise_suppression=True,
            inference_framework="onnx"
        )

    def process_frame(self, pcm_bytes: bytes) -> bool:
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
        prediction = self._model.predict(audio_np)
        score = prediction.get(self.MODEL_NAME, 0.0)
        return score >= OWW_THRESHOLD

class SpeechCapture:
    def listen_for_utterance(self, stream: pyaudio.Stream) -> Optional[AudioChunk]:
        frames = []
        silent_frames = 0
        speech_started = False

        for _ in range(MAX_UTTERANCE_FRAMES):
            if tts_playing.is_set():
                return None

            raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            is_speech = self._is_speech(raw)

            if is_speech:
                silent_frames = 0
                speech_started = True
                frames.append(raw)
            elif speech_started:
                silent_frames += 1
                frames.append(raw)
                if silent_frames >= SILENCE_FRAMES_END:
                    break

        if not speech_started or len(frames) < 5:
            return None

        return AudioChunk(frames=frames)

    def _is_speech(self, frame_bytes: bytes) -> bool:
        audio_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        if audio_np.max() > 1.0:
            audio_np = audio_np / 32768.0
        rms = np.sqrt(np.mean(audio_np ** 2))
        return rms > RMS_THRESHOLD

class Transcriber:
    def __init__(self):
        self._recognizer = sr.Recognizer() if _SR_AVAILABLE else None
        self._whisper: Optional["WhisperModel"] = None
        if _WHISPER_AVAILABLE:
            self._whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    def transcribe(self, chunk: AudioChunk) -> str:
        wav_bytes = chunk.to_wav_bytes()

        if self._recognizer:
            text = self._google_stt(wav_bytes)
            if text:
                return text

        if self._whisper:
            text = self._whisper_stt(wav_bytes)
            if text:
                return text

        return ""

    def _google_stt(self, wav_bytes: bytes) -> str:
        try:
            audio_file = sr.AudioData(wav_bytes, RATE, 2)
            text = self._recognizer.recognize_google(audio_file, language="en-IN")
            return text.strip()
        except Exception:
            return ""

    def _whisper_stt(self, wav_bytes: bytes) -> str:
        try:
            audio_buf = io.BytesIO(wav_bytes)
            segments, _ = self._whisper.transcribe(audio_buf, language="en", beam_size=3, vad_filter=True)
            return " ".join(seg.text for seg in segments).strip()
        except:
            return ""

class MicrophoneStream:
    def __init__(self):
        self._pa: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

    def __enter__(self):
        self._pa = pyaudio.PyAudio()
        self.stream = self._pa.open(rate=RATE, channels=CHANNELS, format=pyaudio.paInt16, input=True, frames_per_buffer=FRAME_SAMPLES)
        return self

    def __exit__(self, *_):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self._pa:
            self._pa.terminate()

    def read_frame(self) -> bytes:
        return self.stream.read(FRAME_SAMPLES, exception_on_overflow=False)
