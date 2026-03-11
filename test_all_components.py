"""
test_all_components.py — Gojan AI Assistant Full Test Suite
=============================================================
Tests all 7 components across Colab and local environments.
Auto-detects environment and skips inapplicable tests gracefully.

COLAB:  pip install requests beautifulsoup4 sentence-transformers faiss-cpu
LOCAL:  pip install -r requirements_local.txt

Run: python test_all_components.py
"""

import os
import sys
import platform
import time
import traceback

# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------
def detect_environment():
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"

ENV = detect_environment()

# EOS token for TinyLlama (built at runtime to avoid tooling issues)
EOS = "<" + "/s>"
TAG_SYS = "<" + "|system|>"
TAG_USR = "<" + "|user|>"
TAG_AST = "<" + "|assistant|>"

print()
print("\u2554" + "\u2550" * 56 + "\u2557")
print("\u2551       GOJAN AI ASSISTANT \u2014 FULL COMPONENT TEST        \u2551")
print("\u2551  Environment : {:<10}  Platform: {:<10}      \u2551".format(
    ENV.upper(), platform.system()))
print("\u255a" + "\u2550" * 56 + "\u255d")
print()

results = {}


def run_test(name, fn):
    print("\n" + "\u2550" * 55)
    print("  RUNNING: " + name)
    print("\u2550" * 55)
    try:
        fn()
        results[name] = ("PASS", "")
        print("  \u2713 " + name + ": PASS")
    except AssertionError as e:
        results[name] = ("FAIL", str(e))
        print("  \u2717 " + name + ": FAIL \u2014 " + str(e))
    except Exception as e:
        results[name] = ("FAIL", traceback.format_exc())
        print("  \u2717 " + name + ": FAIL \u2014 " + str(e))


def skip_test(name, reason):
    results[name] = ("SKIP", reason)
    print("\n  \u2298 " + name + ": SKIP \u2014 " + reason)


# ═══════════════════════════════════════════════════════════
# TEST 1 — WEB SCRAPER (both Colab + local)
# ═══════════════════════════════════════════════════════════
def test_scraper():
    import requests
    from bs4 import BeautifulSoup

    TEST_URLS = [
        ("homepage",   "https://gojaneducation.tech/"),
        ("about",      "https://gojaneducation.tech/about-us/"),
        ("admissions", "https://gojaneducation.tech/admissions/"),
        ("cse_dept",   "https://gojaneducation.tech/computer-science-engg/"),
        ("placements", "https://gojaneducation.tech/placements-2/"),
    ]

    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GojanTestBot/1.0)"}
    EXPECTED = {
        "homepage":   ["gojan", "chennai", "engineering"],
        "about":      ["established", "2005", "anna"],
        "admissions": ["admission", "tnea", "apply"],
        "cse_dept":   ["computer", "science", "engineering"],
        "placements": ["placement", "company", "recruit"],
    }
    passed = 0

    for name, url in TEST_URLS:
        try:
            time.sleep(1.5)
            r = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["nav", "footer", "script", "style"]):
                tag.decompose()
            text = " ".join(
                t.get_text(strip=True)
                for t in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "td"])
            )
            words = [w for w in text.split() if len(w) > 2]
            assert len(words) > 50, (
                name + " only got " + str(len(words)) + " words")
            text_lower = text.lower()
            for kw in EXPECTED[name]:
                assert kw in text_lower, (
                    "Expected '" + kw + "' missing from " + name)
            print("     \u2713 " + name + ": " + str(len(words)) + " words")
            passed += 1
        except requests.exceptions.ConnectionError:
            raise AssertionError("Internet required for scraper test")

    assert passed == len(TEST_URLS), (
        "Only " + str(passed) + "/" + str(len(TEST_URLS)) + " pages scraped")
    print("\n     \u2192 All " + str(passed) + " pages scraped. Scraper working.")

run_test("TEST 1 \u2014 Web Scraper", test_scraper)


# ═══════════════════════════════════════════════════════════
# TEST 2 — LANGUAGE DETECTION (both, no model needed)
# ═══════════════════════════════════════════════════════════
def test_language_detection():
    TANGLISH_WORDS = [
        "iruku", "illa", "sollu", "pathi", "enna", "evvalavu",
        "nalla", "konjam", "pannanum", "da", "di", "seri",
        "theriyum", "varuvanga", "kelu", "paaru", "mattum",
        "pottu", "poganum", "vanthu", "therinja", "aagum",
        "kittum", "soldra", "kekkura", "irundhu", "serthu",
    ]

    def detect_language(text):
        tamil_chars = [c for c in text if "\u0B80" <= c <= "\u0BFF"]
        if len(tamil_chars) > 2:
            return "tamil"
        words = text.lower().split()
        hits = sum(1 for w in words if w in TANGLISH_WORDS)
        if hits >= 2:
            return "tanglish"
        return "english"

    TEST_CASES = [
        ("Where is Gojan college located?", "english"),
        ("What courses are available at Gojan?", "english"),
        ("Is Gojan NAAC accredited?", "english"),
        ("Gojan college la evvalavu courses iruku?", "tanglish"),
        ("Admission ku enna pannanum sollu", "tanglish"),
        ("Hostel iruka campus la?", "tanglish"),
        ("Fees evvalavu da konjam sollu", "tanglish"),
        ("TNEA code enna iruku?", "tanglish"),
        ("Placement nalla iruka Gojan la?", "tanglish"),
        ("\u0b95\u0bcb\u0b9c\u0ba9\u0bcd \u0b95\u0bb2\u0bcd\u0bb2\u0bc2\u0bb0\u0bbf \u0b8e\u0b99\u0bcd\u0b95\u0bc7 \u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bc1?", "tamil"),
        ("\u0b9a\u0bc7\u0bb0\u0bcd\u0b95\u0bcd\u0b95\u0bc8 \u0b8e\u0baa\u0bcd\u0baa\u0b9f\u0bbf \u0b9a\u0bc6\u0baf\u0bcd\u0bb5\u0ba4\u0bc1?", "tamil"),
        ("\u0b8e\u0ba9\u0bcd\u0ba9 \u0baa\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0bc1\u0b95\u0bb3\u0bcd \u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bc1?", "tamil"),
        ("\u0b95\u0bb2\u0bcd\u0bb2\u0bc2\u0bb0\u0bbf \u0b8e\u0baa\u0bcd\u0baa\u0bcb \u0b86\u0bb0\u0bae\u0bcd\u0baa\u0bbf\u0b9a\u0bcd\u0b9a\u0bbe\u0b99\u0bcd\u0b95?", "tamil"),
        ("\u0bb9\u0bbe\u0bb8\u0bcd\u0b9f\u0bb2\u0bcd \u0bb5\u0b9a\u0ba4\u0bbf \u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bbe?", "tamil"),
    ]

    passed = 0
    failed_cases = []
    for text, expected in TEST_CASES:
        result = detect_language(text)
        if result == expected:
            print("     \u2713 '" + text[:40] + "...' \u2192 " + result)
            passed += 1
        else:
            print("     \u2717 '" + text[:40] + "...' \u2192 got '" + result + "' expected '" + expected + "'")
            failed_cases.append((text, expected, result))

    if failed_cases:
        raise AssertionError(str(len(failed_cases)) + " detection failures")
    print("\n     \u2192 " + str(passed) + "/" + str(len(TEST_CASES)) + " correct.")

run_test("TEST 2 \u2014 Language Detection", test_language_detection)


# ═══════════════════════════════════════════════════════════
# TEST 3 — FAISS VECTOR SEARCH (both Colab + local)
# ═══════════════════════════════════════════════════════════
def test_faiss():
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    print("     Loading sentence-transformers (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    DOCS = [
        "Gojan School of Business and Technology is located in Redhills Chennai.",
        "The TNEA counselling code for Gojan is 1123.",
        "Gojan was established in 2005 on an 80 acre campus.",
        "The college is affiliated to Anna University Chennai.",
        "Gojan offers B.E. Computer Science and Engineering.",
        "Gojan offers B.E. Artificial Intelligence and Machine Learning.",
        "The college email is gsbt@gojaneducation.tech.",
        "Contact number is +91 7010723984.",
        "Hostel facilities are available for boys and girls separately.",
        "The college is accredited by NAAC and recognized by AICTE.",
        "Gojan offers MBA postgraduate program.",
        "Transport facilities available from various parts of Chennai.",
        "The campus has WiFi library laboratories and sports facilities.",
        "Admission for engineering through TNEA counselling code 1123.",
        "Gojan offers B.E. Cyber Security Engineering.",
    ]

    print("     Building FAISS index on 15 test documents...")
    embeddings = model.encode(DOCS)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings, dtype="float32"))

    QUERIES = [
        ("Where is Gojan college?",        ["Redhills", "Chennai"]),
        ("What is the TNEA code?",         ["1123"]),
        ("Does Gojan have hostel?",        ["Hostel", "hostel"]),
        ("Contact number of Gojan",        ["7010723984"]),
        ("Is Gojan NAAC accredited?",      ["NAAC"]),
        ("What courses does Gojan offer?", ["Engineering", "MBA"]),
        ("How to apply for admission?",    ["TNEA", "counselling"]),
        ("Does Gojan have transport?",     ["Transport", "transport"]),
    ]

    passed = 0
    for query, expected_kws in QUERIES:
        q_embed = model.encode([query])
        D, I = index.search(np.array(q_embed, dtype="float32"), k=3)
        top_results = " ".join([DOCS[i] for i in I[0]])
        found = any(kw in top_results for kw in expected_kws)
        if found:
            print("     \u2713 '" + query + "' \u2192 correct")
            passed += 1
        else:
            print("     \u2717 '" + query + "' \u2192 expected " + str(expected_kws))

    assert passed >= 6, (
        "Only " + str(passed) + "/8 FAISS queries correct (need 6+)")
    print("\n     \u2192 " + str(passed) + "/8 queries correct. FAISS working.")

run_test("TEST 3 \u2014 FAISS Vector Search", test_faiss)


# ═══════════════════════════════════════════════════════════
# TEST 4 — CONVERSATION MEMORY (both, no model needed)
# ═══════════════════════════════════════════════════════════
def test_conversation_memory():

    class ConversationMemory:
        def __init__(self, max_turns=5):
            self.max_turns = max_turns
            self.history = []
            self.last_answer = None

        def add_turn(self, role, text, language):
            self.history.append({
                "role": role, "content": text,
                "language": language, "timestamp": time.time(),
            })
            if len(self.history) > self.max_turns:
                self.history.pop(0)

        def get_context_window(self):
            return "\n".join(
                t["role"].upper() + ": " + t["content"]
                for t in self.history
            )

        def get_last_language(self):
            if not self.history:
                return "english"
            return self.history[-1]["language"]

        def clear(self):
            self.history = []
            self.last_answer = None

    m = ConversationMemory(max_turns=5)

    m.add_turn("user",      "Where is Gojan?",        "english")
    m.add_turn("assistant", "Redhills Chennai",       "english")
    m.add_turn("user",      "Hostel iruka?",          "tanglish")
    m.add_turn("assistant", "Yes hostel iruku",       "tanglish")
    m.add_turn("user",      "\u0b95\u0bcb\u0b9c\u0ba9\u0bcd fees?",    "tamil")
    m.add_turn("assistant", "Fees pathi office kelu", "tanglish")

    assert len(m.history) == 5, (
        "Memory should cap at 5, got " + str(len(m.history)))
    print("     \u2713 Memory cap at 5 turns: working")

    assert m.history[0]["content"] == "Redhills Chennai"
    print("     \u2713 Oldest turn auto-dropped: working")

    assert m.get_last_language() == "tanglish"
    print("     \u2713 Last language detection: working")

    ctx = m.get_context_window()
    assert "USER:" in ctx and "ASSISTANT:" in ctx
    print("     \u2713 Context window format: working")

    m.clear()
    assert len(m.history) == 0 and m.last_answer is None
    assert m.get_last_language() == "english"
    print("     \u2713 Memory clear: working")

    m.add_turn("user", "test", "english")
    m.last_answer = "Gojan is in Chennai"
    assert m.last_answer == "Gojan is in Chennai"
    print("     \u2713 last_answer storage: working")

    print("\n     \u2192 All 6 memory assertions passed.")

run_test("TEST 4 \u2014 Conversation Memory", test_conversation_memory)


# ═══════════════════════════════════════════════════════════
# TEST 5 — TINYLLAMA CPU INFERENCE (needs GGUF model)
# ═══════════════════════════════════════════════════════════
GGUF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "gguf", "gojan_ai_q4.gguf"
)


def test_llm_inference():
    if not os.path.exists(GGUF_PATH):
        raise AssertionError(
            "GGUF model not found at " + GGUF_PATH +
            ". Run 06_export_gguf.py on Colab first.")

    from ctransformers import AutoModelForCausalLM
    print("     Loading TinyLlama Q4 GGUF on CPU...")
    t = time.time()
    llm = AutoModelForCausalLM.from_pretrained(
        GGUF_PATH,
        model_type="llama",
        context_length=512,
        threads=4
    )
    print("     Model loaded in {:.1f}s".format(time.time() - t))

    # Build prompts at runtime using token variables
    TEST_PROMPTS = [
        {
            "name": "English question",
            "prompt": (
                TAG_SYS + "You are an assistant for Gojan college. "
                "Answer using context.\nContext: Gojan is located at "
                "Redhills Chennai established 2005." + EOS
                + TAG_USR + "Where is Gojan college?" + EOS
                + TAG_AST
            ),
            "must_contain_any": ["Chennai", "Redhills", "Tamil Nadu", "located"],
            "max_seconds": 30,
        },
        {
            "name": "Tanglish question",
            "prompt": (
                TAG_SYS + "You are an assistant for Gojan college. "
                "Answer in Tanglish.\nContext: Gojan la 8 UG courses iruku. "
                "CSE ECE IT AI-ML available." + EOS
                + TAG_USR + "Gojan la enna courses iruku?" + EOS
                + TAG_AST
            ),
            "must_contain_any": ["CSE", "ECE", "course", "iruku", "Engineering"],
            "max_seconds": 30,
        },
        {
            "name": "Tamil question",
            "prompt": (
                TAG_SYS + "You are an assistant for Gojan college. "
                "Answer in Tamil.\nContext: \u0b95\u0bcb\u0b9c\u0ba9\u0bcd "
                "\u0b95\u0bb2\u0bcd\u0bb2\u0bc2\u0bb0\u0bbf \u0b9a\u0bc6\u0ba9\u0bcd"
                "\u0ba9\u0bc8 \u0bb0\u0bc6\u0b9f\u0bcd\u0bb9\u0bbf\u0bb2\u0bcd\u0bb8\u0bcd"
                "\u0bb2\u0bcd \u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bc1." + EOS
                + TAG_USR + "\u0b95\u0bcb\u0b9c\u0ba9\u0bcd \u0b8e\u0b99\u0bcd\u0b95\u0bc7 "
                "\u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bc1?" + EOS
                + TAG_AST
            ),
            "must_contain_any": ["\u0b9a\u0bc6\u0ba9\u0bcd\u0ba9\u0bc8", "Chennai",
                                 "Redhills", "\u0bb0\u0bc6\u0b9f\u0bcd\u0bb9\u0bbf\u0bb2\u0bcd\u0bb8\u0bcd"],
            "max_seconds": 35,
        },
        {
            "name": "Follow-up with history",
            "prompt": (
                TAG_SYS + "You are an assistant for Gojan college. Use history."
                "\nContext: TNEA code is 1123 for Gojan."
                "\nHistory:\nUSER: Tell me about Gojan"
                "\nASSISTANT: Gojan is in Chennai established 2005." + EOS
                + TAG_USR + "What is the admission code?" + EOS
                + TAG_AST
            ),
            "must_contain_any": ["1123", "TNEA", "code", "admission"],
            "max_seconds": 30,
        },
    ]

    passed = 0
    for test in TEST_PROMPTS:
        t_start = time.time()
        response = llm(
            test["prompt"], max_new_tokens=150, temperature=0.2, top_p=0.9,
            stop=[EOS, TAG_USR, TAG_SYS],
        )
        elapsed = time.time() - t_start
        text = response.strip()

        print("\n     [" + test["name"] + "]")
        print("     Response ({:.1f}s): {}".format(elapsed, text[:120]))

        time_ok = elapsed < test["max_seconds"]
        keyword_ok = any(kw in text for kw in test["must_contain_any"])
        not_empty = len(text.strip()) > 5

        if time_ok and keyword_ok and not_empty:
            print("     \u2713 PASS \u2014 {:.1f}s".format(elapsed))
            passed += 1
        else:
            issues = []
            if not time_ok:
                issues.append("too slow ({:.1f}s)".format(elapsed))
            if not keyword_ok:
                issues.append("missing keywords")
            if not not_empty:
                issues.append("empty response")
            print("     \u2717 FAIL \u2014 " + ", ".join(issues))

    assert passed >= 3, (
        "Only " + str(passed) + "/4 LLM tests passed (need 3+)")
    print("\n     \u2192 " + str(passed) + "/4 inference tests passed.")


if os.path.exists(GGUF_PATH):
    run_test("TEST 5 \u2014 TinyLlama CPU Inference", test_llm_inference)
else:
    skip_test("TEST 5 \u2014 TinyLlama CPU Inference",
              "GGUF not found \u2014 run Colab training first")


# ═══════════════════════════════════════════════════════════
# TEST 6 — WHISPER STT + LANGUAGE DETECTION (local only)
# ═══════════════════════════════════════════════════════════
def test_stt():
    from faster_whisper import WhisperModel
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    import numpy as np
    import tempfile

    TANGLISH_WORDS = [
        "iruku", "illa", "sollu", "enna", "evvalavu",
        "nalla", "konjam", "pannanum", "da", "seri",
        "theriyum", "poganum", "kittum",
    ]

    def detect_final_language(text, whisper_lang):
        tamil_chars = [c for c in text if "\u0B80" <= c <= "\u0BFF"]
        if len(tamil_chars) > 2:
            return "tamil"
        words = text.lower().split()
        hits = sum(1 for w in words if w in TANGLISH_WORDS)
        if hits >= 2:
            return "tanglish"
        return "english"

    print("     Loading Whisper tiny on CPU...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("     \u2713 Whisper loaded")

    TEST_SPEECHES = [
        {"instruction": "Say in ENGLISH: 'Where is Gojan college located?'",
         "expected_lang": "english"},
        {"instruction": "Say in TANGLISH: 'Gojan la evvalavu courses iruku?'",
         "expected_lang": "tanglish"},
        {"instruction": "Say in TAMIL: '\u0b95\u0bcb\u0b9c\u0ba9\u0bcd \u0b95\u0bb2\u0bcd\u0bb2\u0bc2\u0bb0\u0bbf \u0b8e\u0b99\u0bcd\u0b95\u0bc7 \u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bc1'",
         "expected_lang": "tamil"},
    ]

    SAMPLE_RATE = 16000
    passed = 0

    for i, test in enumerate(TEST_SPEECHES):
        print("\n     \u2500\u2500\u2500 Speech Test {}/3 \u2500\u2500\u2500".format(i + 1))
        print("     \U0001f449 " + test["instruction"])
        input("     Press ENTER when ready to speak (5 seconds)...")

        print("     \U0001f3a4 Recording...")
        audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        print("     \u2713 Recording done. Transcribing...")

        audio_int16 = (audio * 32767).astype("int16")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, SAMPLE_RATE, audio_int16)

        segments, info = model.transcribe(tmp.name)
        text = " ".join([s.text for s in segments]).strip()
        whisper_lang = info.language
        final_lang = detect_final_language(text, whisper_lang)

        print("     Transcribed  : '" + text + "'")
        print("     Whisper lang : " + str(whisper_lang))
        print("     Final lang   : " + final_lang)
        os.unlink(tmp.name)

        text_ok = len(text.strip()) > 2
        lang_ok = final_lang == test["expected_lang"]

        if text_ok and lang_ok:
            print("     \u2713 PASS")
            passed += 1
        else:
            issues = []
            if not text_ok:
                issues.append("empty transcription")
            if not lang_ok:
                issues.append("wrong language (got " + final_lang +
                              ", expected " + test["expected_lang"] + ")")
            print("     \u2717 FAIL \u2014 " + ", ".join(issues))

    assert passed >= 2, (
        "Only " + str(passed) + "/3 STT tests passed (need 2+)")
    print("\n     \u2192 " + str(passed) + "/3 speech tests passed.")


if ENV == "local":
    run_test("TEST 6 \u2014 Whisper STT + Language Detection", test_stt)
else:
    skip_test("TEST 6 \u2014 Whisper STT + Language Detection",
              "Colab has no microphone \u2014 run on local laptop")


# ═══════════════════════════════════════════════════════════
# TEST 7 — PYTTSX3 TTS OUTPUT (local only)
# ═══════════════════════════════════════════════════════════
def test_tts():
    import pyttsx3

    print("     Initializing pyttsx3...")
    engine = pyttsx3.init()
    engine.setProperty("rate", 155)
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")
    print("     Available voices: " + str(len(voices)))
    for v in voices:
        print("       - " + v.name)

    for voice in voices:
        if any(x in voice.name.lower() for x in ["female", "zira", "hazel"]):
            engine.setProperty("voice", voice.id)
            print("     \u2713 Set voice to: " + voice.name)
            break

    TEST_SPEECHES = [
        ("ENGLISH",
         "Hello! I am the Gojan College AI Assistant. How can I help you today?"),
        ("TANGLISH",
         "Gojan la nalla courses iruku. CSE, ECE, AI-ML ellam iruku. Enna kekanum?"),
        ("ROMANIZED TAMIL",
         "Nandri. Gojan college pathi yethavadhu kekanum na sollunga."),
    ]

    passed = 0
    for lang, text in TEST_SPEECHES:
        print("\n     Speaking [" + lang + "]:")
        print("     Text: " + text)
        try:
            engine.say(text)
            engine.runAndWait()
            response = input("     Did you hear the speech clearly? (y/n): ")
            if response.lower().strip() == "y":
                print("     \u2713 PASS")
                passed += 1
            else:
                print("     \u2717 FAIL \u2014 user reported audio not heard")
        except Exception as e:
            print("     \u2717 FAIL \u2014 " + str(e))

    print("\n     \u26a0  NOTE: pyttsx3 cannot speak Tamil Unicode script.")
    print("     Tamil script answers are printed to screen only.")

    assert passed >= 2, (
        "Only " + str(passed) + "/3 TTS tests passed (need 2+)")
    print("\n     \u2192 " + str(passed) + "/3 TTS tests passed.")


if ENV == "local":
    run_test("TEST 7 \u2014 pyttsx3 TTS Output", test_tts)
else:
    skip_test("TEST 7 \u2014 pyttsx3 TTS Output",
              "Colab has no speaker \u2014 run on local laptop")


# ═══════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════
print("\n\n" + "\u2550" * 58)
print("  GOJAN AI ASSISTANT \u2014 COMPLETE TEST REPORT")
print("  Environment: " + ENV.upper())
print("\u2550" * 58)

STATUS_ICON = {"PASS": "\u2713", "FAIL": "\u2717", "SKIP": "\u2298"}
STATUS_LABEL = {"PASS": "PASS", "FAIL": "FAIL \u2190 FIX THIS", "SKIP": "SKIP"}

total = len(results)
n_pass = sum(1 for v in results.values() if v[0] == "PASS")
n_fail = sum(1 for v in results.values() if v[0] == "FAIL")
n_skip = sum(1 for v in results.values() if v[0] == "SKIP")

for name, (status, detail) in results.items():
    icon = STATUS_ICON[status]
    label = STATUS_LABEL[status]
    print("  " + icon + "  " + name.ljust(42) + label)
    if status == "FAIL" and detail:
        short = detail.strip().split("\n")[0][:80]
        print("       \u2514\u2500 " + short)

print("\u2500" * 58)
print("  Total: {} | Passed: {} | Failed: {} | Skipped: {}".format(
    total, n_pass, n_fail, n_skip))
print("\u2550" * 58)

if n_fail == 0:
    print("""
  \u2705 ALL TESTS PASSED
  Safe to run the full Gojan AI build prompt.
  Proceed with: phase_a_colab scripts in order 01 \u2192 07
""")
elif n_fail <= 2:
    print("""
  \u26a0  {} TEST(S) FAILED \u2014 Fix before running full build.

  HOW TO FIX:
  - SCRAPER FAIL    \u2192 Check internet connection, try again
  - LLM FAIL        \u2192 Re-export GGUF or increase n_threads
  - STT FAIL        \u2192 Check microphone permissions in Windows
  - TTS FAIL        \u2192 Run: pip install pyttsx3 --upgrade
  - FAISS FAIL      \u2192 Run: pip install faiss-cpu sentence-transformers
  - MEMORY FAIL     \u2192 Logic error in ConversationMemory class
  - LANG FAIL       \u2192 Expand TANGLISH_WORDS list
""".format(n_fail))
else:
    print("""
  \u274c {} TESTS FAILED \u2014 Environment not ready.

  Run this first on Colab:
    pip install -r requirements_colab.txt

  Run this first on local laptop:
    setup_local.bat

  Then re-run: python test_all_components.py
""".format(n_fail))

print("\u2550" * 58 + "\n")
