import pyttsx3
try:
    engine = pyttsx3.init()
    engine.say("Hello, this is a test of the text to speech system.")
    engine.runAndWait()
    print("SUCCESS: Text-to-speech ran without crashing.")
except Exception as e:
    print(f"FAILED: Text-to-speech crashed. Error: {e}")
