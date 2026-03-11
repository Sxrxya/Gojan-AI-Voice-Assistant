"""
Phase B - Text-to-Speech Service (Hybrid: pyttsx3 + gTTS)
==========================================================
Primary:  pyttsx3 (offline, zero RAM, instant)
Bonus:    gTTS with tld="co.in" (Indian English accent, needs internet)
Tamil:    gTTS lang="ta" when online, screen-only when offline
Playback: pygame.mixer for MP3 files (gTTS path)
"""

import os
import re
import tempfile
import numpy as np


# ── Internet check ───────────────────────────────────────────────────────────

def _check_internet():
    """Quick check if internet is available (3s timeout)."""
    import socket
    try:
        socket.setdefaulttimeout(3)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
        s.close()
        return True
    except Exception:
        return False


# ── Load ─────────────────────────────────────────────────────────────────────

def load_tts():
    """
    Returns a dict: {mode: "gtts"|"pyttsx3", engine: <object>}
    Tries gTTS first (better voice), falls back to pyttsx3.
    """
    info = {"mode": None, "engine": None}

    # Try gTTS
    try:
        from gtts import gTTS  # noqa: F401
        if _check_internet():
            info["mode"] = "gtts"
            info["engine"] = gTTS
            print("      TTS: Google TTS (Indian voice, online)")
            return info
        else:
            print("      No internet - falling back to pyttsx3")
    except ImportError:
        print("      gTTS not installed - falling back to pyttsx3")

    # Fallback: pyttsx3
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 148)
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")
    for v in voices:
        name = v.name.lower()
        if "zira" in name or "hazel" in name:
            engine.setProperty("voice", v.id)
            break

    info["mode"] = "pyttsx3"
    info["engine"] = engine
    print("      TTS: pyttsx3 (offline backup)")
    return info


# ── Clean text for voice ─────────────────────────────────────────────────────

def _clean_for_voice(text):
    """Strip markdown, limit to 3 sentences."""
    if not text:
        return ""
    clean = text.strip()
    clean = re.sub(r"[*#_`]", "", clean)

    # Split into sentences and take max 3
    parts = re.split(r"(?<=[.!?])\s+", clean)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:3])


# ── Play MP3 via pygame ──────────────────────────────────────────────────────

def _play_mp3(filepath):
    """Play an MP3 file using pygame.mixer. Initialises and quits cleanly."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        pygame.mixer.quit()
    except Exception as e:
        print(f"      Audio playback error: {e}")


# ── Speak ────────────────────────────────────────────────────────────────────

def speak(engine_info, text, language="english"):
    """
    Speak text using the best available engine.

    Parameters
    ----------
    engine_info : dict from load_tts()
    text        : string to speak
    language    : "english" | "tanglish" | "tamil"
    """
    clean = _clean_for_voice(text)
    if not clean:
        return

    mode = engine_info["mode"]

    # ── gTTS path ────────────────────────────────────────────────────────
    if mode == "gtts":
        try:
            from gtts import gTTS

            if language == "tamil":
                lang_code, tld = "ta", "com"
            else:
                # English and Tanglish both use Indian English accent
                lang_code, tld = "en", "co.in"

            tts_obj = gTTS(text=clean, lang=lang_code, tld=tld, slow=False)

            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_path = tmp.name
            tmp.close()

            tts_obj.save(tmp_path)
            _play_mp3(tmp_path)

            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return

        except Exception as e:
            print(f"      gTTS error: {e} - using pyttsx3 fallback")
            # Fall through to pyttsx3

    # ── pyttsx3 path ─────────────────────────────────────────────────────
    try:
        engine = engine_info["engine"]
        if engine is None:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 148)
            engine_info["engine"] = engine

        if language == "tamil":
            engine.say("Tamil answer shown on screen. Please read the terminal.")
        else:
            engine.say(clean)
        engine.runAndWait()

    except Exception as e:
        print(f"      TTS error: {e}")


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
