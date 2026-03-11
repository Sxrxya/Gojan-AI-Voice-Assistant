"""
Voice Test Script - Test TTS in all 3 languages.
Run: python phase_b_local/test_voice.py
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase_b_local.services.tts import load_tts, speak

print("\n" + "=" * 50)
print("  GOJAN AI - VOICE TEST")
print("=" * 50 + "\n")

print("Loading TTS engine...")
tts = load_tts()
print()

tests = [
    ("english",
     "Hello! I am Gojan AI Assistant. "
     "Gojan School of Business and Technology "
     "is located at Redhills, Chennai."),
    ("tanglish",
     "Gojan la nalla courses iruku. "
     "CSE, ECE, AI-ML ellam available. "
     "Admission ku TNEA code 1123 use pannunga."),
    ("tamil",
     "\u0b95\u0bcb\u0b9c\u0ba9\u0bcd \u0b95\u0bb2\u0bcd\u0bb2\u0bc2\u0bb0\u0bbf "
     "\u0b9a\u0bc6\u0ba9\u0bcd\u0ba9\u0bc8\u0baf\u0bbf\u0bb2\u0bcd "
     "\u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. "
     "\u0ba8\u0bc1\u0bb4\u0bc8\u0bb5\u0bc1 TNEA "
     "\u0bae\u0bc2\u0bb2\u0bae\u0bcd "
     "\u0ba8\u0b9f\u0bc8\u0baa\u0bc6\u0bb1\u0bc1\u0bae\u0bcd."),
]

for lang, text in tests:
    print("-" * 50)
    print("Testing {} voice...".format(lang.upper()))
    print("Text: {}...".format(text[:60]))
    speak(tts, text, lang)
    input("\nPress ENTER to continue...\n")

print("=" * 50)
print("  Voice test complete!")
print("  If all 3 sounded good, run: python phase_b_local/main.py")
print("=" * 50 + "\n")
