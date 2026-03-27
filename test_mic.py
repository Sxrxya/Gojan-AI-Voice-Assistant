import speech_recognition as sr
import time

print("Available Microphones:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"[{i}] {name}")

r = sr.Recognizer()
print("\nTesting default Microphone...")

try:
    with sr.Microphone() as source:
        print("Adjusting for ambient noise for 2 seconds...")
        r.adjust_for_ambient_noise(source, duration=2)
        print(f"Energy Threshold set to: {r.energy_threshold}")
        
        print("\nSAY SOMETHING NOW! (Listening for 5 seconds...)")
        try:
            audio = r.listen(source, timeout=8, phrase_time_limit=4)
            print("\nGot audio! Transcribing using Google STT...")
            text = r.recognize_google(audio, language="en-IN")
            print(f"-> SUCCESS! I heard: '{text}'")
        except sr.WaitTimeoutError:
            print("-> FAILED: Timeout waiting for speech. Is your mic muted? Is the volume high enough?")
        except sr.UnknownValueError:
            print("-> FAILED: Could not understand what you said (UnknownValueError).")
        except Exception as e:
            print(f"-> FAILED with error: {e}")
except Exception as e:
    print(f"FATAL MIC ERROR: Could not open microphone. {e}")
