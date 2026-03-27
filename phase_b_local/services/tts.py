"""
Phase B - Text-to-Speech Service (100% Offline)
================================================
Exclusively uses pyttsx3 (Windows SAPI5 voice engine).
Zero internet required. No gTTS dependencies.
"""

import re
import pyttsx3
import numpy as np


# ── Load ─────────────────────────────────────────────────────────────────────

def load_tts():
    """
    Initialize pyttsx3 and select an appropriate Windows voice.
    """
    print("      TTS: pyttsx3 (100% offline Windows voice)")
    engine = pyttsx3.init()
    engine.setProperty("rate", 148)
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")
    for v in voices:
        name = v.name.lower()
        if "zira" in name or "hazel" in name:
            engine.setProperty("voice", v.id)
            break

    return {"mode": "pyttsx3", "engine": engine}


# ── Speak ────────────────────────────────────────────────────────────────────

def speak(engine_info, text, language="english"):
    """
    Speak text using offline pyttsx3.
    Strips markdown and limits length to prevent monotone rambling.
    """
    if not text:
        return
        
    # Clean markdown
    clean = text.strip()
    clean = re.sub(r"[*#_`]", "", clean)

    # Split into sentences and take max 3
    parts = re.split(r"(?<=[.!?])\s+", clean)
    parts = [p.strip() for p in parts if p.strip()]
    clean = " ".join(parts[:3])

    engine = engine_info["engine"]
    
    if language == "tamil":
        engine.say("Tamil answer shown on screen. Please read the terminal.")
    else:
        engine.say(clean)
        
    engine.runAndWait()


# ── Beep ─────────────────────────────────────────────────────────────────────

def play_beep():
    """Play a short 200ms confirmation beep using sounddevice."""
    try:
        import sounddevice as sd
        sr = 44100
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        wave = (np.sin(2 * np.pi * 880 * t) * 0.3).astype(np.float32)
        sd.play(wave, samplerate=sr)
        sd.wait()
    except Exception:
        pass
