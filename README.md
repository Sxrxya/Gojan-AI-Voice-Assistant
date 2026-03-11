# Gojan School of Business and Technology — AI Voice Assistant

A fully offline, CPU-only AI voice assistant that answers questions about
**Gojan School of Business and Technology (GSBT)**, Chennai.

Built with **TinyLlama 1.1B** (QLoRA fine-tuned, Q4 GGUF) + **FAISS RAG** +
**Whisper tiny** (STT) + **pyttsx3** (TTS).

---

## System Architecture

```
  Voice Input (Mic)
       │
       ▼
  Whisper tiny (STT) — CPU
       │
       ▼
  Question Text
       │
       ▼
  FAISS Vector Search — CPU
       │
       ▼
  TinyLlama Q4 GGUF — CPU
       │
       ▼
  Answer Text
       │
       ▼
  pyttsx3 (TTS) — CPU
       │
       ▼
  Voice Output (Speaker)
```

---

## RAM Budget (8 GB Device)

| Component              | RAM     |
|------------------------|---------|
| Windows OS             | ~2.0 GB |
| Whisper tiny           | ~0.2 GB |
| TinyLlama Q4 GGUF     | ~0.8 GB |
| Sentence-Transformers  | ~0.4 GB |
| FAISS index            | ~0.1 GB |
| Python overhead        | ~0.3 GB |
| **TOTAL**              | **~3.8 GB ✓** |

---

## Project Structure

```
gojan-ai-assistant/
├── phase_a_colab/              ← Run on Google Colab (T4 GPU)
│   ├── 01_scrape_website.py
│   ├── 02_scrape_trusted_sources.py
│   ├── 03_clean_and_chunk.py
│   ├── 04_build_qa_dataset.py
│   ├── 05_finetune_tinyllama.py
│   ├── 06_export_gguf.py
│   └── 07_build_vectordb.py
│
├── phase_b_local/              ← Run on laptop (CPU only)
│   ├── services/
│   │   ├── stt.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   └── tts.py
│   └── main.py
│
├── data/
│   ├── raw/website/            ← Scraped pages
│   ├── raw/external/           ← External sources
│   ├── chunks/all_chunks.json
│   ├── qa_dataset/train.jsonl & eval.jsonl
│   └── seed_facts.txt
│
├── models/
│   ├── lora_adapter/           ← QLoRA adapter from Colab
│   └── gguf/gojan_ai_q4.gguf  ← Final GGUF model
│
├── vector_db/
│   ├── college.index
│   └── documents.pkl
│
├── requirements_colab.txt
├── requirements_local.txt
├── setup_local.bat
└── README.md
```

---

## Phase A — Run on Google Colab

1. Upload project to Colab.
2. `pip install -r requirements_colab.txt`
3. `python phase_a_colab/01_scrape_website.py`
4. `python phase_a_colab/02_scrape_trusted_sources.py`
5. `python phase_a_colab/03_clean_and_chunk.py`
6. `python phase_a_colab/04_build_qa_dataset.py`
7. `python phase_a_colab/05_finetune_tinyllama.py`
8. `python phase_a_colab/06_export_gguf.py`
9. `python phase_a_colab/07_build_vectordb.py`
10. **Download these 3 files to your laptop:**
    - `models/gguf/gojan_ai_q4.gguf`
    - `vector_db/college.index`
    - `vector_db/documents.pkl`

---

## Phase B — Run on Laptop (8 GB RAM, No GPU)

1. Run `setup_local.bat` (creates venv + installs deps).
2. Place the 3 downloaded files in their correct folders.
3. Run the assistant:
   ```
   cd phase_b_local
   python main.py
   ```

---

## Updating College Information

1. Edit `data/seed_facts.txt` **OR** re-run `01_scrape_website.py`.
2. Re-run `03_clean_and_chunk.py`.
3. Re-run `07_build_vectordb.py`.
4. Copy new `vector_db/` files to the laptop.

> **Note:** No retraining needed for info updates — only RAG changes.

---

## Trusted Data Sources

- <https://gojaneducation.tech/> (official college website)
- Anna University affiliation facts
- AICTE recognition facts
- NAAC accreditation facts
- TNEA admission process facts

---

## College Quick Facts

| Field           | Value                                                          |
|-----------------|----------------------------------------------------------------|
| Name            | Gojan School of Business and Technology (GSBT)                |
| Address         | 80 Feet Road, Edapalayam, Redhills, Chennai - 600 052        |
| Established     | 2005                                                           |
| Campus          | 80 acres                                                       |
| Affiliation     | Anna University, Chennai                                       |
| Recognition     | AICTE, New Delhi                                               |
| Accreditation   | NAAC                                                           |
| TNEA Code       | 1123                                                           |
| Phone           | +91 7010723984 / 85                                           |
| Email           | gsbt@gojaneducation.tech                                       |
