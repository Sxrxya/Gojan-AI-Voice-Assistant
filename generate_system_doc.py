"""Generate the complete GOJAN_AI_SYSTEM_FLOW document."""
import os

doc = r"""# GOJAN AI VOICE ASSISTANT — COMPLETE SYSTEM FLOW DOCUMENT

**Version:** 2.0
**Date:** March 11, 2026
**Author:** AI Engineering Team
**Project:** Gojan School of Business and Technology — Offline AI Voice Assistant

---

## SECTION 1 — SYSTEM OVERVIEW

The Gojan AI Voice Assistant is a fully offline, voice-activated artificial intelligence system built specifically for **Gojan School of Business and Technology (GSBT)**, Chennai. It is designed to answer questions about the college — admissions, courses, departments, placements, campus facilities, contact details, and more — using natural voice interaction in three languages: **English, Tamil, and Tanglish** (Tamil words written in English script).

The system is built for **students, visitors, parents, and staff** who want instant information about GSBT without needing internet access. A user simply says **"Hey Gojan"** to wake up the assistant, then asks any question about the college. The assistant listens, understands the language being spoken, searches its knowledge base of real scraped data from the official Gojan website, generates a natural language answer using a local AI model, and speaks the answer back through the laptop speakers.

What makes this system special:

- **Trained on real data** — scraped from 40+ pages of gojaneducation.tech plus verified facts from Anna University, AICTE, NAAC, and TNEA
- **Trilingual** — automatically detects and responds in English, Tamil script, or Tanglish
- **Conversation memory** — remembers the last 5 exchanges for follow-up questions
- **Wake word activated** — responds to "Hey Gojan", "Gojan", "Anna", or Tamil wake word
- **100% offline** — runs entirely on a Windows 11 laptop with 8GB RAM, no GPU, no internet required after initial setup
- **Lightweight** — uses only ~3.8GB of RAM, leaving 4.2GB headroom on an 8GB machine

---

## SECTION 2 — COMPLETE SYSTEM ARCHITECTURE DIAGRAM

```
+-------------------------------------------------------------+
|              GOJAN AI VOICE ASSISTANT                        |
|         Complete System Architecture                         |
+-------------------------------------------------------------+

+--------------+
|   USER       |
|  (speaks)    |
+------+-------+
       | voice
       v
+--------------------------------------------------------------+
|  LAYER 1 -- AUDIO INPUT                                      |
|                                                              |
|  sounddevice records 3 seconds (wake word detection)         |
|  sounddevice records 7 seconds (question capture)            |
|  scipy saves as 16000Hz WAV file                             |
|  is_speech() checks RMS amplitude > 0.005 threshold          |
|  -> if silence: loop back and listen again                   |
+----------------------+---------------------------------------+
                       | WAV file
                       v
+--------------------------------------------------------------+
|  LAYER 2 -- SPEECH RECOGNITION (faster-whisper tiny)         |
|                                                              |
|  WAKE WORD MODE:                                             |
|  -> beam_size=5 for accuracy                                 |
|  -> detect_wake_word() checks transcription                  |
|     * Strategy 1: exact known variations list                |
|     * Strategy 2: fuzzy syllable pattern matching            |
|     * Strategy 3: character similarity score (80%)           |
|     * Strategy 4: Levenshtein distance <= 2                  |
|  -> if wake word confirmed: play beep -> go to Question      |
|  -> if not: print what was heard -> loop back                |
|                                                              |
|  QUESTION MODE:                                              |
|  -> free transcription, no hints                             |
|  -> auto-detects language: English / Tamil                   |
|  -> detect_tanglish() checks for Tamil-origin words          |
|  -> returns {text, language}                                 |
+----------------------+---------------------------------------+
                       | {text, language}
                       v
+--------------------------------------------------------------+
|  LAYER 3 -- INTENT DETECTION                                 |
|                                                              |
|  Before retrieving -- check what user wants:                 |
|                                                              |
|  "bye / poganum"        -> speak goodbye, clear memory       |
|  "repeat / meeendum"    -> replay last answer                |
|  "more / innum sollu"   -> retrieve more chunks              |
|  "reset / puthusha"     -> clear memory, greet fresh         |
|  "thanks / nanri"       -> acknowledge, continue             |
|                                                              |
|  anything else -> proceed to retrieval                       |
+----------------------+---------------------------------------+
                       | question text
                       v
+--------------------------------------------------------------+
|  LAYER 4 -- KNOWLEDGE RETRIEVAL (FAISS + Sentence            |
|             Transformers all-MiniLM-L6-v2)                   |
|                                                              |
|  1. Question text -> encode to 384-dimension vector          |
|  2. Search FAISS IndexFlatL2 index                           |
|  3. Return top 4 most similar document chunks                |
|  4. format_context() joins chunks (max 600 words)            |
|                                                              |
|  Knowledge base contains:                                    |
|  -> scraped from gojaneducation.tech (40+ pages)             |
|  -> Anna University, AICTE, NAAC, TNEA facts                |
|  -> 29 hardcoded verified seed facts                         |
|  -> Total: ~150-200 document chunks                          |
+----------------------+---------------------------------------+
                       | context chunks
                       v
+--------------------------------------------------------------+
|  LAYER 5 -- CONVERSATION MEMORY                              |
|                                                              |
|  ConversationMemory stores last 5 exchanges:                 |
|  -> {role, content, language, timestamp}                     |
|  -> get_context_window() returns formatted history           |
|  -> enables follow-up questions like:                        |
|     User: "Tell me about CSE"                                |
|     User: "What about fees?" <- understands context          |
|  -> get_last_language() keeps language consistent            |
|  -> capped at 5 turns to protect 8GB RAM                     |
+----------------------+---------------------------------------+
                       | conversation history
                       v
+--------------------------------------------------------------+
|  LAYER 6 -- LANGUAGE MODEL (TinyLlama 1.1B Q4 GGUF)         |
|                                                              |
|  Loaded with ctransformers:                                  |
|  -> context_length=1024, threads=4 (CPU only)                |
|                                                              |
|  Prompt format (TinyLlama chat template):                    |
|  +---------------------------------------------------+      |
|  | <|system|>                                         |      |
|  | You are a voice assistant for Gojan college.       |      |
|  | Respond in {detected_language}.                    |      |
|  | Context: {retrieved_chunks}                        |      |
|  | History: {conversation_history}                    |      |
|  | </s>                                               |      |
|  | 
