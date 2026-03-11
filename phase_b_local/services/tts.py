"""
Phase B - Text-to-Speech Service (CPU Only, Offline)
=====================================================
Uses pyttsx3 with Windows SAPI5 voices.
Zero model download, zero RAM overhead, instant response.
"""

import pyttsx3


def load_tts():
    """Initialize pyttsx3 TTS engine with preferred settings."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 155)
    engine.setProperty("volume", 1.0)

    # Try to use a female voice (Zira on Windows)
    voices = engine.getProperty("voices")
    for voice in voices:
        name = voice.name.lower()
        if "female" in name or "zira" in name:
            engine.setProperty("voice", voice.id)
            break

    return engine


def speak(engine, text):
    """Speak the given text aloud and print it."""
    print(f"\n\U0001f50a Gojan AI: {text}\n")
    engine.say(text)
    engine.runAndWait()
