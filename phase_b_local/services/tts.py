"""
Phase B - Text-to-Speech Service
================================================
Combines gTTS (for online, high-quality voice) and Pygame for playback to prevent hanging on modern Python versions.
"""

import re
import os
import time

def load_tts():
    """
    Initialize TTS system. Using gTTS + Pygame for robust playback.
    """
    import pygame
    pygame.mixer.init()
    print("      TTS: gTTS loaded correctly.")
    return {"mode": "gtts"}

def speak(engine_info, text, language="english"):
    """
    Speak text using gTTS and pygame.
    """
    if not text:
        return
        
    # Clean markdown
    clean = text.strip()
    clean = re.sub(r"[*#_`]", "", clean)

    # Split into sentences and take max 10
    parts = re.split(r"(?<=[.!?])\s+", clean)
    parts = [p.strip() for p in parts if p.strip()]
    clean = " ".join(parts[:10])
    
    filename = f"temp_tts_{int(time.time()*1000)}.mp3"

    if language == "tamil" or language == "tanglish":
        voice = "ta-IN-PallaviNeural"  # Realistic Indian Tamil female voice
    else:
        voice = "en-IN-NeerjaNeural"   # Realistic Indian English female voice

    # We use en-IN-NeerjaNeural for high quality Indian English
    if language == "english":
        voice = "en-IN-NeerjaNeural"

    import pygame
    import subprocess
    
    fallback_to_gtts = False
    try:
        # Use edge-tts via subprocess with a strict 4-second timeout
        # so the assistant doesn't "freeze" if the Microsoft server is slow.
        clean_escaped = clean.replace('"', '\\"')
        cmd = f'edge-tts --voice {voice} --text "{clean_escaped}" --write-media {filename}'
        
        # Execute with a 4-second timeout to prevent "too slow" feeling
        result = subprocess.run(cmd, shell=True, timeout=4.0, capture_output=True)
        
        if result.returncode != 0 or not os.path.exists(filename) or os.path.getsize(filename) == 0:
            fallback_to_gtts = True
        else:
            # Play the generated realistic audio
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            time.sleep(0.1)
            if os.path.exists(filename):
                os.remove(filename)
                
    except subprocess.TimeoutExpired:
        print("      TTS: Edge-TTS network timeout. Falling back to gTTS...")
        fallback_to_gtts = True
    except Exception as e:
        print(f"      TTS Error: {e}")
        fallback_to_gtts = True
        
    # IMMEDIATE FALLBACK: If Edge-TTS was too slow or failed (Network issues)
    if fallback_to_gtts:
        try:
            from gtts import gTTS
            lang_code = 'ta' if language in ["tamil", "tanglish"] else 'en'
            tts = gTTS(text=clean, lang=lang_code)
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            time.sleep(0.1)
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"      Fallback TTS Error: {e}")

def play_beep():
    """Play a short 200ms confirmation beep using sounddevice."""
    try:
        import numpy as np
        import sounddevice as sd
        sr = 44100
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        wave = (np.sin(2 * np.pi * 880 * t) * 0.3).astype(np.float32)
        sd.play(wave, samplerate=sr)
        sd.wait()
    except Exception:
        pass
