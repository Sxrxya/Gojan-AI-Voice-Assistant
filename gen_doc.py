#!/usr/bin/env python3
"""Generate the complete GOJAN_AI_SYSTEM_FLOW document."""
import os, textwrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GOJAN_AI_SYSTEM_FLOW.md")
PIPE = "|"
LT = "<"
GT = ">"

def w(fh, text):
    fh.write(textwrap.dedent(text))

with open(OUT, "w", encoding="utf-8") as f:
    # ============================================================
    # TITLE + SECTION 1
    # ============================================================
    w(f, """\
    # GOJAN AI VOICE ASSISTANT - COMPLETE SYSTEM FLOW DOCUMENT

    **Version:** 2.0
    **Date:** March 11, 2026
    **Author:** AI Engineering Team
    **Project:** Gojan School of Business and Technology - Offline AI Voice Assistant

    ---

    ## SECTION 1 - SYSTEM OVERVIEW

    The Gojan AI Voice Assistant is a fully offline, voice-activated artificial
    intelligence system built specifically for **Gojan School of Business and
    Technology (GSBT)**, Chennai. It is designed to answer questions about the
    college - admissions, courses, departments, placements, campus facilities,
    contact details, and more - using natural voice interaction in three
    languages: **English, Tamil, and Tanglish** (Tamil words written in English
    script).

    The system is built for **students, visitors, parents, and staff** who want
    instant information about GSBT without needing internet access. A user
    simply says **"Hey Gojan"** to wake up the assistant, then asks any question
    about the college. The assistant listens, understands the language being
    spoken, searches its knowledge base of real scraped data from the official
    Gojan website, generates a natural language answer using a local AI model,
    and speaks the answer back through the laptop speakers.

    What makes this system special:

    - **Trained on real data** - scraped from 40+ pages of gojaneducation.tech
      plus verified facts from Anna University, AICTE, NAAC, and TNEA
    - **Trilingual** - automatically detects and responds in English, Tamil
      script, or Tanglish
    - **Conversation memory** - remembers the last 5 exchanges for follow-up
      questions
    - **Wake word activated** - responds to "Hey Gojan", "Gojan", "Anna", or
      Tamil wake word
    - **100% offline** - runs entirely on a Windows 11 laptop with 8GB RAM,
      no GPU, no internet required after initial setup
    - **Lightweight** - uses only ~3.8GB of RAM, leaving 4.2GB headroom on
      an 8GB machine

    """)

    # ============================================================
    # SECTION 2 - ARCHITECTURE DIAGRAM
    # ============================================================
    f.write("---\n\n## SECTION 2 - COMPLETE SYSTEM ARCHITECTURE DIAGRAM\n\n```\n")
    diagram_lines = [
        "+-------------------------------------------------------------+",
        "|              GOJAN AI VOICE ASSISTANT                        |",
        "|         Complete System Architecture                         |",
        "+-------------------------------------------------------------+",
        "",
        "+--------------+",
        "|   USER       |",
        "|  (speaks)    |",
        "+------+-------+",
        "       | voice",
        "       v",
        "+--------------------------------------------------------------+",
        "|  LAYER 1 - AUDIO INPUT                                       |",
        "|                                                              |",
        "|  sounddevice records 3 seconds (wake word detection)         |",
        "|  sounddevice records 7 seconds (question capture)            |",
        "|  scipy saves as 16000Hz WAV file                             |",
        "|  is_speech() checks RMS amplitude > 0.005 threshold          |",
        "|  -> if silence: loop back and listen again                   |",
        "+----------------------+---------------------------------------+",
        "                       | WAV file",
        "                       v",
        "+--------------------------------------------------------------+",
        "|  LAYER 2 - SPEECH RECOGNITION (faster-whisper tiny)          |",
        "|                                                              |",
        "|  WAKE WORD MODE:                                             |",
        "|  -> beam_size=5 for accuracy                                 |",
        "|  -> detect_wake_word() checks transcription                  |",
        "|     * Strategy 1: exact known variations list                |",
        "|     * Strategy 2: fuzzy syllable pattern matching            |",
        "|     * Strategy 3: character similarity score (80%)           |",
        "|     * Strategy 4: Levenshtein distance <= 2                  |",
        "|  -> if wake word confirmed: play beep -> go to Question      |",
        "|  -> if not: print what was heard -> loop back                |",
        "|                                                              |",
        "|  QUESTION MODE:                                              |",
        "|  -> free transcription, no hints                             |",
        "|  -> auto-detects language: English / Tamil                   |",
        "|  -> detect_tanglish() checks for Tamil-origin words          |",
        "|  -> returns {text, language}                                 |",
        "+----------------------+---------------------------------------+",
        "                       | {text, language}",
        "                       v",
        "+--------------------------------------------------------------+",
        "|  LAYER 3 - INTENT DETECTION                                  |",
        "|                                                              |",
        "|  Before retrieving - check what user wants:                  |",
        "|                                                              |",
        "|  'bye / poganum'     -> speak goodbye, clear memory          |",
        "|  'repeat / meeendum' -> replay last answer                   |",
        "|  'more / innum sollu'-> retrieve more chunks                 |",
        "|  'reset / puthusha'  -> clear memory, greet fresh            |",
        "|  'thanks / nanri'    -> acknowledge, continue                |",
        "|                                                              |",
        "|  anything else -> proceed to retrieval                       |",
        "+----------------------+---------------------------------------+",
        "                       | question text",
        "                       v",
        "+--------------------------------------------------------------+",
        "|  LAYER 4 - KNOWLEDGE RETRIEVAL (FAISS + Sentence             |",
        "|            Transformers all-MiniLM-L6-v2)                    |",
        "|                                                              |",
        "|  1. Question text -> encode to 384-dimension vector          |",
        "|  2. Search FAISS IndexFlatL2 index                           |",
        "|  3. Return top 4 most similar document chunks                |",
        "|  4. format_context() joins chunks (max 600 words)            |",
        "|                                                              |",
        "|  Knowledge base contains:                                    |",
        "|  -> scraped from gojaneducation.tech (40+ pages)             |",
        "|  -> Anna University, AICTE, NAAC, TNEA facts                |",
        "|  -> 29 hardcoded verified seed facts                         |",
        "|  -> Total: ~150-200 document chunks                          |",
        "+----------------------+---------------------------------------+",
        "                       | context chunks",
        "                       v",
        "+--------------------------------------------------------------+",
        "|  LAYER 5 - CONVERSATION MEMORY                               |",
        "|                                                              |",
        "|  ConversationMemory stores last 5 exchanges:                 |",
        "|  -> {role, content, language, timestamp}                     |",
        "|  -> get_context_window() returns formatted history           |",
        "|  -> enables follow-up questions like:                        |",
        "|     User: 'Tell me about CSE'                                |",
        "|     User: 'What about fees?' <- understands context          |",
        "|  -> get_last_language() keeps language consistent            |",
        "|  -> capped at 5 turns to protect 8GB RAM                     |",
        "+----------------------+---------------------------------------+",
        "                       | conversation history",
        "                       v",
        "+--------------------------------------------------------------+",
        "|  LAYER 6 - LANGUAGE MODEL (TinyLlama 1.1B Q4 GGUF)          |",
        "|                                                              |",
        "|  Loaded with ctransformers:                                  |",
        "|  -> context_length=1024, threads=4 (CPU only)                |",
        "|                                                              |",
        "|  Prompt template (TinyLlama chat format):                    |",
        "|  [system] You are a voice assistant for Gojan college.       |",
        "|  Respond in {detected_language}.                             |",
        "|  Context: {retrieved_chunks}                                 |",
        "|  History: {conversation_history}                             |",
        "|  [user] {question}                                           |",
        "|  [assistant] ...generates answer...                          |",
        "|                                                              |",
        "|  Settings: max_tokens=180, temperature=0.2, top_p=0.9       |",
        "|                                                              |",
        "|  post_process_response():                                    |",
        "|  -> strips prompt tags from output                           |",
        "|  -> trims to 3 sentences max (voice UX)                      |",
        "|  -> adds fallback if context insufficient:                   |",
        "|    'Contact: +91 7010723984 / gsbt@gojaneducation.tech'     |",
        "|                                                              |",
        "|  Why TinyLlama:                                              |",
        "|  -> 1.1B params -> Q4 GGUF = ~800MB RAM                     |",
        "|  -> Fits safely in 8GB RAM alongside all other components    |",
        "+----------------------+---------------------------------------+",
        "                       | answer text",
        "                       v",
        "+--------------------------------------------------------------+",
        "|  LAYER 7 - RESPONSE OUTPUT                                   |",
        "|                                                              |",
        "|  TERMINAL: print answer with language tag                    |",
        "|  [ENGLISH] Gojan AI: 'The college is located at...'         |",
        "|                                                              |",
        "|  VOICE: pyttsx3 speaks the answer                            |",
        "|  -> uses Windows SAPI5 built-in voice engine                 |",
        "|  -> rate=155 words/minute                                    |",
        "|  -> zero RAM overhead, works 100% offline                    |",
        "|  -> Tamil script displayed on screen (not spoken)            |",
        "|  -> Tanglish and English spoken aloud                        |",
        "|                                                              |",
        "|  MEMORY UPDATE:                                              |",
        "|  -> add user turn to ConversationMemory                      |",
        "|  -> add assistant answer to ConversationMemory               |",
        "|  -> store as last_answer for repeat command                  |",
        "+----------------------+---------------------------------------+",
        "                       |",
        "                       v",
        "              loop back to LAYER 1",
        "              waiting for next wake word",
    ]
    for line in diagram_lines:
        f.write(line + "\n")
    f.write("```\n\n")

    # ============================================================
    # SECTION 3
    # ============================================================
    w(f, """\
    ---

    ## SECTION 3 - TRAINING PIPELINE FLOW (Phase A - Google Colab)

    The training pipeline prepares all the data and models that the local
    assistant needs. It was designed to run on Google Colab (free T4 GPU) but
    most steps were executed locally on the user's laptop during development.

    ### STEP 1 - Web Scraping (01_scrape_website.py)

    This script visits all 40+ pages of `https://gojaneducation.tech/` and
    extracts useful text content from each page.

    - **Tags extracted:** h1, h2, h3, h4, p, li, td (headings, paragraphs,
      lists, table cells)
    - **Tags skipped:** nav, footer, script, style (navigation noise, code,
      styling)
    - **Polite scraping:** 1.5 second delay between HTTP requests to avoid
      overloading the server
    - **Output:** Each page saved as a separate `.txt` file in
      `data/raw/website/`
    - **Index:** `index.json` records all URLs scraped and their word counts
    - **Error handling:** Gracefully skips pages that return HTTP errors (404,
      500) and logs them

    ### STEP 2 - External Sources (02_scrape_trusted_sources.py)

    This script gathers additional information from trusted external sources
    to supplement the Gojan website data.

    - **Scraped sources:**
      - Anna University official pages (affiliation data)
      - AICTE approval records
      - TNEA counselling code listings
    - **Hardcoded verified facts** (manually confirmed and embedded in code):
      - TNEA counselling code: **1123**
      - Affiliated to: **Anna University, Chennai**
      - Accreditation: **NAAC accredited**
      - AICTE approved: **Yes**
      - Address: **80 Feet Road, Edapalayam, Redhills, Chennai - 600 052**
      - Phone: **+91 7010723984**
      - Email: **gsbt@gojaneducation.tech**
      - Campus size: **80 acres**
      - Established: **2007**
    - **Output:** Saved to `data/raw/external/`

    ### STEP 3 - Cleaning and Chunking (03_clean_and_chunk.py)

    This script takes all the raw scraped text and transforms it into clean,
    searchable chunks for the vector database.

    - **Cleaning steps:**
      1. Remove repeated navigation text that appears on every page
      2. Remove lines shorter than 8 words (fragments, menu items)
      3. Remove exact duplicate sentences across all pages
      4. Normalize whitespace and encoding
    - **Chunking strategy:**
      - Each chunk is 120-180 words in length
      - 15 word overlap between consecutive chunks ensures no information is
        lost at chunk boundaries
      - Each chunk gets metadata: `source` (filename), `page` (URL), `chunk_id`
    - **Output:** `data/chunks/all_chunks.json` containing all cleaned chunks
    - **Also generates:** `data/seed_facts.txt` with 29 verified facts

    ### STEP 4 - QA Dataset Generation (04_build_qa_dataset.py)

    This script builds the multilingual training dataset that teaches the LLM
    how to answer questions about Gojan college.

    - **Template-based QA pairs:** 22 hardcoded question-answer pairs using
      verified real facts about Gojan (location, TNEA code, courses, etc.)
    - **English QA pairs:** 40 pairs generated from scraped chunks
    - **Tamil QA pairs:** 40 pairs with proper Tamil Unicode script
    - **Tanglish QA pairs:** 40 pairs with natural Tamil-English mixing
      (e.g., "Gojan la enna courses iruku?")
    - **Multi-turn conversations:** 30 dialogues with 3-4 exchanges each,
      teaching the model to handle follow-up questions
    - **Total:** ~170+ training examples
    - **Format:** Alpaca instruction format (instruction, input, output)
    - **Split:** 90% saved to `data/qa_dataset/train.jsonl`, 10% to
      `data/qa_dataset/eval.jsonl`

    ### STEP 5 - QLoRA Fine-tuning (05_finetune_tinyllama.py)

    This script fine-tunes the TinyLlama base model on the Gojan QA dataset
    using parameter-efficient training.

    - **Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` from HuggingFace
    - **Quantization:** 4-bit with BitsAndBytes (fits Colab free T4 GPU with
      15GB VRAM)
    - **LoRA configuration:**
      - Rank (r) = 16
      - Alpha = 32
      - Target modules: q_proj, k_proj, v_proj, o_proj
      - Dropout = 0.05
    - **Training settings:**
      - 3 epochs over the full training set
      - Batch size = 4 (with gradient accumulation for effective batch of 16)
      - Learning rate = 2e-4 with cosine scheduler
      - Warmup steps = 10
    - **System prompt:** Teaches the model to respond in English, Tamil, or
      Tanglish based on the user's language
    - **Training time:** ~30-60 minutes on Colab T4 GPU
    - **Output:** LoRA adapter weights saved to `models/lora_adapter/`
    - **Note:** This step was bypassed during local setup by using the base
      TinyLlama model directly with FAISS-powered RAG

    ### STEP 6 - GGUF Export (06_export_gguf.py)

    This script converts the fine-tuned model into a format optimized for
    CPU inference.

    - **Process:**
      1. Merge LoRA adapter weights back into the base TinyLlama model
      2. Save the merged full-precision model
      3. Convert to GGUF format using llama.cpp conversion tools
      4. Apply Q4_K_M quantization (4-bit mixed precision)
    - **Size reduction:** Full model ~4GB -> Quantized GGUF ~800MB
    - **Final file:** `models/gguf/gojan_ai_q4.gguf`
    - **This file is what runs on the 8GB RAM laptop**
    - **Note:** During local setup, a pre-quantized TinyLlama GGUF was
      downloaded directly from HuggingFace (TheBloke), bypassing this step

    ### STEP 7 - Vector Database Build (07_build_vectordb.py)

    This script creates the searchable knowledge index used for RAG retrieval.

    - **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
    - **Process:**
      1. Load all chunks from `data/chunks/all_chunks.json`
      2. Encode each chunk into a 384-dimension dense vector
      3. Add all 29 seed facts as separate vectors
      4. Build a FAISS `IndexFlatL2` index (exact L2 distance search)
    - **Output files:**
      - `vector_db/college.index` - the FAISS binary index file
      - `vector_db/documents.pkl` - pickled list of original text chunks
    - **These 2 files are copied to the laptop alongside the GGUF model**
    - **Search speed:** Sub-millisecond for queries against ~200 chunks

    """)

    # ============================================================
    # SECTION 4
    # ============================================================
    w(f, """\
    ---

    ## SECTION 4 - STARTUP SEQUENCE (Phase B - Local Laptop)

    When the user runs `python phase_b_local/main.py`, the following boot
    sequence executes in order:

    ### Boot Sequence

    | Step | Action | RAM Used | Time |
    |------|--------|----------|------|
    | 1 | Print Gojan AI ASCII banner | ~0 MB | instant |
    | 2 | Load Whisper tiny model (CPU, int8) | ~200 MB | ~1 second |
    | 3 | Load FAISS index + documents.pkl | ~150 MB | ~0.5 seconds |
    | 4 | Load sentence-transformers (all-MiniLM-L6-v2) | ~400 MB | ~2 seconds |
    | 5 | Load TinyLlama Q4 GGUF via ctransformers | ~800 MB | ~10-20 seconds |
    | 6 | Initialize pyttsx3 TTS engine | ~5 MB | instant |
    | 7 | **Total at startup** | **~1.6 GB** | **~15 seconds** |

    ### After Boot

    1. System speaks a time-of-day greeting:
       - Morning (6-12): "Good morning!"
       - Afternoon (12-17): "Good afternoon!"
       - Evening (17-21): "Good evening!"
       - Night: "Hello!"
    2. Greeting is spoken in English with Tamil translation displayed
    3. System enters the **wake word listening loop**
    4. Terminal displays: "Say 'Hey Gojan' to wake me up..."

    ### Total System RAM

    ```
    Windows 11 OS baseline:  ~2.0 GB
    Python + Gojan AI:       ~1.6 GB
    ----------------------------------
    Total in use:            ~3.6 GB
    Available (out of 8GB):  ~4.4 GB
    Status: SAFE
    ```

    """)

    # ============================================================
    # SECTION 5
    # ============================================================
    w(f, """\
    ---

    ## SECTION 5 - COMPLETE EXAMPLE CONVERSATIONS

    ### EXAMPLE 1 - English Single Question

    ```
    User says   : "Hey Gojan"
    Whisper     : transcribes as "hey goes in" (common variation)
    Wake Word   : detect_wake_word() -> Strategy 1 exact match -> DETECTED
    System      : [beep sound plays]
    Terminal    : "Listening for your question... (7 seconds)"
    User says   : "Where is Gojan college located?"
    Whisper     : "Where is Gojan college located?" (language: en)
    Language    : detect_tanglish() finds 0 Tamil words -> "english"
    Intent      : not bye/repeat/reset -> proceed to retrieval
    FAISS       : top 4 chunks about location retrieved
    LLM prompt  : system=Gojan assistant, context=location chunks, lang=english
    LLM answer  : "Gojan School of Business and Technology is located at
                   80 Feet Road, Edapalayam, Redhills, Chennai - 600 052.
                   The campus spans 80 acres."
    Terminal    : [ENGLISH] Gojan AI: "Gojan School of Business..."
    pyttsx3     : speaks the answer through laptop speakers
    Memory      : stores this exchange (turn 1 of 5)
    ```

    ### EXAMPLE 2 - Tanglish Multi-Turn Conversation

    ```
    User says   : "Gojan"
    Wake Word   : detect_wake_word() -> exact match -> DETECTED
    System      : [beep]
    User says   : "Gojan la enna courses iruku?"
    Whisper     : transcribes as English text
    Language    : detect_tanglish() finds "enna", "iruku" -> "tanglish"
    FAISS       : retrieves course-related chunks
    LLM answer  : "Gojan la 8 UG courses iruku - CSE, ECE, IT, AI-ML,
                   Cyber Security, Aeronautical, Mechanical,
                   Medical Electronics. Plus MBA PG course um iruku."
    pyttsx3     : speaks answer (Tanglish works well with English TTS)

    [Second turn]
    User says   : "Gojan"
    Wake Word   : DETECTED
    User says   : "CSE pathi sollu"
    Language    : detect_tanglish() finds "pathi", "sollu" -> "tanglish"
    Memory      : includes previous exchange about courses
    FAISS       : retrieves CSE-specific chunks
    LLM answer  : "CSE - Computer Science Engineering - Anna University
                   affiliated. Software, AI, networking ellam cover aagum.
                   4 year degree with placement support."
    Memory      : now stores 2 exchanges
    ```

    ### EXAMPLE 3 - Tamil Question

    ```
    User says   : "Anna" (backup wake word)
    Wake Word   : exact match on "anna" -> DETECTED
    System      : [beep]
    User says   : (speaks in Tamil)
    Whisper     : detects language as "ta" (Tamil)
    Language    : _detect_language() maps "ta" -> "tamil"
    FAISS       : retrieves admission-related chunks
    LLM answer  : (generates Tamil response about admissions)
    Terminal    : Tamil Unicode text displayed on screen
    pyttsx3     : speaks Romanized equivalent (SAPI5 limitation)
    ```

    ### EXAMPLE 4 - Unknown Question Fallback

    ```
    User says   : "Hey Gojan"
    Wake Word   : DETECTED
    User says   : "What is the wifi password?"
    FAISS       : searches index - no relevant chunks found
                  (closest chunks have low similarity scores)
    LLM         : receives empty/weak context
    LLM answer  : "I don't have that information right now.
                   Please contact: +91 7010723984
                   or email gsbt@gojaneducation.tech"
    pyttsx3     : speaks the fallback answer
    ```

    ### EXAMPLE 5 - Repeat Command

    ```
    User says   : "Hey Gojan"
    Wake Word   : DETECTED
    User says   : "What is the TNEA code?"
    FAISS       : retrieves TNEA-related chunks
    LLM answer  : "The TNEA counselling code for Gojan is 1123."
    Memory      : stored as last_answer

    [Later]
    User says   : "Hey Gojan"
    Wake Word   : DETECTED
    User says   : "Repeat"
    Intent      : _any_match() finds "repeat" in REPEAT_WORDS -> True
    System      : replays "The TNEA counselling code for Gojan is 1123."
                  WITHOUT calling the LLM again (instant replay)
    ```

    """)

    # ============================================================
    # SECTION 6
    # ============================================================
    w(f, """\
    ---

    ## SECTION 6 - RAM USAGE MAP

    ```
    Component                    RAM Used
    -----------------------------------------
    Windows 11 OS                ~2.0 GB
    Python + overhead            ~0.3 GB
    Whisper tiny model           ~0.2 GB
    Sentence Transformers        ~0.4 GB
    FAISS index (158 chunks)     ~0.1 GB
    TinyLlama Q4 GGUF            ~0.8 GB
    pyttsx3 TTS engine           ~0.0 GB
    -----------------------------------------
    TOTAL IN USE                 ~3.8 GB
    REMAINING FREE               ~4.2 GB
    DEVICE TOTAL                  8.0 GB
    -----------------------------------------
    STATUS: SAFE - 4.2GB headroom available
    ```

    ### RAM Breakdown by Phase

    | Phase | Components Active | Peak RAM |
    |-------|-------------------|----------|
    | Boot (loading models) | All models loading sequentially | ~1.6 GB |
    | Idle (waiting for wake word) | All models in memory | ~1.6 GB |
    | Processing question | Models + FAISS search + LLM inference | ~1.8 GB |
    | Speaking answer | Models + pyttsx3 active | ~1.6 GB |

    The system never exceeds 2.0 GB of Python process RAM, which combined
    with Windows 11 baseline (~2.0 GB) stays well within 8 GB total.

    """)

    # ============================================================
    # SECTION 7
    # ============================================================
    w(f, """\
    ---

    ## SECTION 7 - LANGUAGE HANDLING FLOW

    The system supports three languages with automatic detection:

    ### ENGLISH Detection

    ```
    User speaks English
        -> Whisper returns language="en"
        -> detect_tanglish() scans for Tamil-origin words
        -> finds 0 Tamil words (threshold is 2+)
        -> final language = "english"
        -> LLM prompted: "Respond in English"
        -> LLM generates English answer
        -> pyttsx3 speaks answer aloud (native English TTS)
    ```

    ### TAMIL Detection

    ```
    User speaks Tamil
        -> Whisper detects Tamil Unicode characters (U+0B80 to U+0BFF)
        -> Whisper returns language="ta"
        -> _detect_language() maps "ta" -> "tamil"
        -> LLM prompted: "Respond in Tamil"
        -> LLM generates Tamil response
        -> Tamil answer printed to terminal (screen display)
        -> pyttsx3 speaks Romanized equivalent aloud
        -> Reason: Windows SAPI5 cannot synthesize Tamil Unicode script
    ```

    ### TANGLISH Detection

    ```
    User speaks Tanglish (Tamil words in English alphabet)
        -> Whisper transcribes as English text (Tanglish uses Latin script)
        -> Whisper returns language="en"
        -> detect_tanglish() scans the transcription for Tamil-origin words:
           iruku, illa, sollu, pathi, enna, evvalavu, nalla, konjam,
           theriyum, pannanum, mattum, da, seri, aama, romba, thaan,
           anga, inga, appadi, ippo, eppadi, eppo, ennaku, unnaku,
           pannu, panna, irukum, iruka, sollunge, poganum, vanakkam
        -> if 2+ Tamil words found -> language = "tanglish"
        -> LLM prompted: "Respond in Tanglish"
        -> LLM generates Tanglish response
        -> pyttsx3 speaks answer aloud (Tanglish works well with English TTS)
    ```

    ### Language Memory

    - `get_last_language()` tracks the last language used in conversation
    - If user switches language mid-conversation, the system adapts immediately
    - `ConversationMemory` stores the language for each turn
    - This enables consistent language responses across follow-up questions

    """)

    # ============================================================
    # SECTION 8
    # ============================================================
    w(f, """\
    ---

    ## SECTION 8 - WAKE WORD SYSTEM FLOW

    ### Why Wake Word Detection is Hard

    "Gojan" is a proper noun that does not exist in Whisper's vocabulary.
    The Whisper tiny model (trained on general English/multilingual data)
    has never seen the word "Gojan" before. When a user says "Hey Gojan",
    Whisper tries to match it to the closest known English words, producing
    transcriptions like:

    - "hey goes in" (most common)
    - "hey jen"
    - "go john"
    - "hey golden"
    - "gorgon"
    - "hey george" and many others

    This means simple exact-string matching ("hey gojan") would fail almost
    every time. The solution is a multi-strategy detection system.

    ### How It Is Solved - 4 Strategies (in order)

    #### Strategy 1 - Known Variations List (Fastest)

    A hardcoded list of 25+ known Whisper transcriptions of "Gojan":

    ```
    "hey gojan", "gojan", "hi gojan", "hello gojan", "anna",
    "hey gojen", "go john", "go jan", "hey go jan", "a gojan",
    "okay gojan", "ok gojan", "gojan ai", "ko jan", "gojon",
    "gojhan", "hey goes in", "goes in", "hey jen", "jen",
    "hey golden", "golden", "hey gorgon", "gorgon"
    ```

    This list was built by recording actual Whisper outputs during testing.
    Runs first because it is the fastest check (simple string matching).

    #### Strategy 2 - Fuzzy Syllable Pattern Matching

    Checks if any word in the transcription matches phonetic patterns:

    - Starts with "goj" (any suffix) -> match
    - Starts with "go" AND contains "j" AND length >= 4 -> match
    - Starts with "ko" AND contains "j" AND length >= 4 -> match
    - Two consecutive words form "go jan", "go john", "go jon",
      "go jean", "go chan", "hey jan", or "hey john" -> match

    This catches variations not in the exact list.

    #### Strategy 3 - Character Similarity Score

    For each word with 4+ characters, calculates the percentage of unique
    characters shared with "gojan":

    ```
    Target: g, o, j, a, n (5 unique characters)
    Word "gojen": g, o, j, e, n -> overlap = {g,o,j,n} = 4/5 = 80% -> MATCH
    Word "morning": m, o, r, n, i, g -> overlap = {o,n,g} = 3/5 = 60% -> NO
    ```

    Threshold: 80% unique character overlap required (prevents false positives
    on common words like "good" or "morning").

    #### Strategy 4 - Levenshtein Distance

    Computes the edit distance (minimum character insertions, deletions, or
    substitutions) between each 4+ character word and "gojan":

    ```
    levenshtein("gojan", "gojan") = 0 -> MATCH
    levenshtein("gojon", "gojan") = 1 -> MATCH
    levenshtein("gojhan","gojan") = 1 -> MATCH
    levenshtein("golden","gojan") = 3 -> NO MATCH (exceeds threshold)
    ```

    Threshold: distance <= 2 (allows up to 2 character differences).

    ### Debug Mode

    Run `python phase_b_local/main.py --debug` to test all 4 strategies
    without needing a microphone. The debug mode runs 18 test cases and
    reports pass/fail for each:

    ```
    WAKE WORD DEBUG MODE
    "Hey Gojan"           -> detected=True  expected=True   PASS
    "gojan"               -> detected=True  expected=True   PASS
    "good morning"        -> detected=False expected=False  PASS
    ...
    Wake word test: 18/18 passed
    Wake word detection working perfectly
    ```

    """)

    # ============================================================
    # SECTION 9
    # ============================================================
    w(f, """\
    ---

    ## SECTION 9 - DATA FLOW SUMMARY (one line per stage)

    ```
    Microphone
        | raw audio (numpy float32 array)
        v
    sounddevice record (3s wake / 7s question)
        | WAV file (16000Hz mono int16)
        v
    is_speech() RMS check (threshold > 0.005)
        | confirmed speech
        v
    faster-whisper tiny transcribe
        | {text: "Hey Gojan", language: "en"}
        v
    detect_wake_word() - 4 strategies
        | wake word confirmed -> beep sound
        v
    faster-whisper tiny transcribe question (7 seconds)
        | {text: "Where is Gojan?", language: "en"}
        v
    detect_tanglish() language classification
        | language = "english"
        v
    intent detection (bye/repeat/more/reset/thanks)
        | intent = "question" (proceed)
        v
    sentence-transformers encode question
        | 384-dimension dense vector
        v
    FAISS IndexFlatL2 similarity search
        | top 4 matching document chunks
        v
    ConversationMemory get_context_window()
        | last 5 conversation turns (formatted text)
        v
    TinyLlama Q4 GGUF generate answer (ctransformers)
        | raw answer text
        v
    post_process_response() clean + trim
        | clean answer (max 3 sentences)
        v
    ConversationMemory add_turn()
        | memory updated with user question + AI answer
        v
    pyttsx3 speak answer
        | audio output through laptop speakers
        v
    terminal print answer
        | displayed with [ENGLISH/TAMIL/TANGLISH] language tag
        v
    loop back to microphone
        waiting for next wake word...
    ```

    """)

    # ============================================================
    # SECTION 10
    # ============================================================
    w(f, """\
    ---

    ## SECTION 10 - COMPONENT VERSIONS AND REASONS

    | Component | Version | Why Chosen |
    |-----------|---------|------------|
    | TinyLlama GGUF | 1.1B Q4_K_M | Only LLM that fits 8GB RAM on CPU. 1.1B parameters quantized to 4-bit = ~800MB. Fast inference (~2s per answer). |
    | faster-whisper | tiny model | 75MB RAM footprint, fast on CPU, good accuracy for English. Supports multilingual detection. |
    | sentence-transformers | all-MiniLM-L6-v2 | 384 dimensions, ~400MB RAM. Best balance of speed vs accuracy for semantic search. |
    | faiss-cpu | 1.8.0 | CPU-only vector search, no GPU needed. IndexFlatL2 provides exact search, fast for < 1000 documents. |
    | ctransformers | latest | Pure Python GGUF loader. No C++ compiler required (unlike llama-cpp-python). Supports CPU-only inference. |
    | pyttsx3 | 2.90 | Zero additional RAM. Uses Windows SAPI5 built-in voice engine. 100% offline, no model download needed. |
    | sounddevice | 0.4.6 | Clean microphone recording API for Windows. Supports 16000Hz mono recording. |
    | scipy | 1.11+ | WAV file I/O for saving recorded audio to temporary files for Whisper processing. |
    | numpy | 1.24+ | Audio array manipulation, RMS calculation, beep tone generation via sine wave. |
    | QLoRA (LoRA) | r=16, alpha=32 | Parameter-efficient fine-tuning on Colab free T4 GPU. Only trains ~0.5% of model parameters. |
    | GGUF Q4_K_M | quantization | Best quality/size ratio for CPU inference. Mixed 4-bit precision preserves model quality. |
    | FAISS FlatL2 | index type | Exact L2 distance search. No approximation errors. Fast enough for < 1000 documents. |

    ### Why These Specific Choices

    **TinyLlama over larger models:** Mistral 7B (the original plan) requires
    ~5.5GB RAM for Q4 quantization, leaving only ~0.5GB for other components
    on an 8GB machine. TinyLlama at 1.1B params uses only ~800MB, leaving
    comfortable headroom.

    **ctransformers over llama-cpp-python:** The llama-cpp-python package
    requires a C++ compiler (MSVC or GCC) to build from source on Windows.
    This was not available on the target machine. ctransformers provides
    the same GGUF loading capability as a pure Python package with no
    compilation step.

    **faster-whisper over openai-whisper:** faster-whisper uses CTranslate2
    for optimized CPU inference, running 4x faster than the original OpenAI
    Whisper implementation with lower memory usage.

    **pyttsx3 over Coqui TTS:** Coqui TTS requires downloading neural voice
    models (~200-500MB) and uses significant RAM. pyttsx3 uses the built-in
    Windows SAPI5 engine with zero additional RAM and works completely offline.

    ---

    ## SECTION 11 - FILE STRUCTURE

    ```
    Gojan-AI-Voice-Assistant/
    |
    |-- phase_a_colab/              (Training Pipeline - run on Colab)
    |   |-- 01_scrape_website.py    (Web scraper for gojaneducation.tech)
    |   |-- 02_scrape_trusted_sources.py  (External data + verified facts)
    |   |-- 03_clean_and_chunk.py   (Text cleaning + chunking pipeline)
    |   |-- 04_build_qa_dataset.py  (Multilingual QA dataset builder)
    |   |-- 05_finetune_tinyllama.py (QLoRA fine-tuning script)
    |   |-- 06_export_gguf.py       (GGUF export + quantization)
    |   |-- 07_build_vectordb.py    (FAISS index builder)
    |
    |-- phase_b_local/              (Local Inference - runs on laptop)
    |   |-- main.py                 (Interactive loop + wake word + memory)
    |   |-- services/
    |       |-- __init__.py
    |       |-- stt.py              (Whisper STT + language detection)
    |       |-- retriever.py        (FAISS retriever + embedding)
    |       |-- llm.py              (TinyLlama inference + prompt builder)
    |       |-- tts.py              (pyttsx3 text-to-speech)
    |
    |-- models/
    |   |-- gguf/
    |       |-- gojan_ai_q4.gguf    (TinyLlama Q4 model - ~638MB)
    |
    |-- vector_db/
    |   |-- college.index           (FAISS binary index)
    |   |-- documents.pkl           (Pickled document chunks)
    |
    |-- data/
    |   |-- raw/                    (Scraped raw text files)
    |   |-- chunks/
    |   |   |-- all_chunks.json     (Cleaned + chunked text)
    |   |-- qa_dataset/
    |   |   |-- train.jsonl         (Training QA pairs)
    |   |   |-- eval.jsonl          (Evaluation QA pairs)
    |   |-- qa_templates.json       (76 trilingual QA templates)
    |   |-- qa_multiturn.json       (30 multi-turn conversations)
    |   |-- seed_facts.txt          (29 verified facts)
    |
    |-- test_all_components.py      (Automated test suite)
    |-- run_tests.bat               (Test runner batch script)
    |-- requirements_local.txt      (Python dependencies)
    |-- .gitignore                  (Excludes models/venv/data)
    |-- GOJAN_AI_SYSTEM_FLOW.md     (This document)
    ```

    ---

    **END OF DOCUMENT**

    This document describes the complete technical architecture and data flow
    of the Gojan AI Voice Assistant v2.0. For questions or modifications,
    refer to the source code in the `phase_b_local/` directory.
    """)

print(f"Document written to: {OUT}")
print(f"Size: {os.path.getsize(OUT)} bytes")
with open(OUT, "r", encoding="utf-8") as f:
    lines = f.readlines()
print(f"Lines: {len(lines)}")
