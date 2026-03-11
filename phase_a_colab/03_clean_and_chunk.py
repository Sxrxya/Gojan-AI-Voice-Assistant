"""
Phase A — Step 3: Clean and Chunk All Scraped Data
====================================================
Cleans raw scraped text, splits into overlapping chunks,
deduplicates, and saves as JSON.

Run on: Google Colab
Input : data/raw/website/*.txt + data/raw/external/*.txt
Output: data/chunks/all_chunks.json + data/seed_facts.txt
"""

import os
import re
import json
import hashlib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEBSITE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "website")
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "external")
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")
SEED_FACTS_PATH = os.path.join(PROJECT_ROOT, "data", "seed_facts.txt")

CHUNK_TARGET_MIN = 120  # words
CHUNK_TARGET_MAX = 180  # words
CHUNK_OVERLAP = 15      # words

# Lines to remove (navigation / boilerplate)
REMOVE_PATTERNS = [
    "Skip to content",
    "EMAIL LOGIN",
    "Facebook",
    "Twitter",
    "Instagram",
    "Youtube",
    "ADMISSIONS OPEN",
    "TNEA Counselling Code",
    "© 2021 GSBT",
    "All rights reserved",
    "Designed By",
    "Subscribe to get",
    "Connect to Gojan",
]

# ---------------------------------------------------------------------------
# Seed Facts (hardcoded verified data)
# ---------------------------------------------------------------------------
SEED_FACTS = [
    "Gojan School of Business and Technology was established in 2005.",
    "The college is located at 80 Feet Road, Edapalayam, Redhills, Chennai - 600 052.",
    "The campus spans 80 acres at Redhills, Chennai.",
    "The college is affiliated to Anna University, Chennai.",
    "The college is recognized by AICTE, New Delhi.",
    "The college is accredited by NAAC.",
    "The TNEA counselling code for admissions is 1123.",
    "Contact phone numbers are +91 7010723984, +91 7010723985.",
    "The official email is gsbt@gojaneducation.tech.",
    "UG programmes offered: B.E. Aeronautical Engineering.",
    "UG programmes offered: B.E. Computer Science and Engineering.",
    "UG programmes offered: B.E. Electronics and Communication Engineering.",
    "UG programmes offered: B.E. Artificial Intelligence and Machine Learning.",
    "UG programmes offered: B.E. Cyber Security Engineering.",
    "UG programmes offered: B.E. Medical Electronics Engineering.",
    "UG programmes offered: B.E. Mechanical and Automation Engineering.",
    "UG programmes offered: B.Tech. Information Technology.",
    "PG programme offered: MBA - Master of Business Administration.",
    "Admissions for engineering are through TNEA counselling by Anna University.",
    "The college has hostel facilities for boys and girls separately.",
    "The college provides transport facilities from various parts of Chennai.",
    "The campus has WiFi, library, laboratories, and sports facilities.",
    "The college has an active placement cell connecting students with top companies.",
    "The college organizes NSS activities, cultural events and technical symposiums.",
    "The college has professional bodies including IEEE Student Branch.",
    "The college has various student clubs under Gojan Clubs.",
    "The college has an Entrepreneurship Development Cell (ED Cell).",
    "NAAC accreditation confirms quality standards at GSBT.",
    "Anna University rank holders have emerged from GSBT.",
]


def clean_text(text: str) -> str:
    """Clean raw scraped text."""
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip boilerplate patterns
        skip = False
        for pattern in REMOVE_PATTERNS:
            if pattern.lower() in line.lower():
                skip = True
                break
        if skip:
            continue

        # Skip lines shorter than 8 words
        if len(line.split()) < 8:
            # Keep lines that look like facts (contain key terms)
            key_terms = ["gojan", "gsbt", "chennai", "anna university",
                         "aicte", "naac", "tnea", "1123", "engineering"]
            has_key = any(t in line.lower() for t in key_terms)
            if not has_key:
                continue

        # Skip pure URLs or email-only lines
        if re.match(r"^https?://\S+$", line):
            continue
        if re.match(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", line):
            continue

        cleaned.append(line)

    # Remove duplicate lines
    seen = set()
    unique = []
    for line in cleaned:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    # Normalize whitespace
    text = "\n".join(unique)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, source: str, page: str) -> list[dict]:
    """Split text into overlapping chunks of target word count."""
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_words = []
    current_sentences = []

    for sentence in sentences:
        words = sentence.split()
        current_words.extend(words)
        current_sentences.append(sentence)

        if len(current_words) >= CHUNK_TARGET_MIN:
            chunk_text_str = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text_str,
                "source": source,
                "page": page,
                "word_count": len(current_words),
            })

            # Overlap: keep last CHUNK_OVERLAP words worth of sentences
            overlap_words = 0
            overlap_start = len(current_sentences)
            for i in range(len(current_sentences) - 1, -1, -1):
                overlap_words += len(current_sentences[i].split())
                if overlap_words >= CHUNK_OVERLAP:
                    overlap_start = i
                    break

            current_sentences = current_sentences[overlap_start:]
            current_words = []
            for s in current_sentences:
                current_words.extend(s.split())

    # Remaining text becomes a chunk if long enough
    if current_words and len(current_words) >= 10:
        chunk_text_str = " ".join(current_sentences)
        chunks.append({
            "text": chunk_text_str,
            "source": source,
            "page": page,
            "word_count": len(current_words),
        })

    return chunks


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove exact duplicate chunks by text hash."""
    seen_hashes = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(chunk)
    return unique


def load_raw_files(directory: str, source: str) -> list[tuple[str, str, str]]:
    """Load all .txt files from a directory. Returns (text, source, page) tuples."""
    results = []
    if not os.path.exists(directory):
        return results

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            page = filename.replace(".txt", "")
            results.append((text, source, page))

    return results


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    print("═" * 60)
    print("  Data Cleaning & Chunking Pipeline")
    print("═" * 60)
    print()

    all_chunks = []
    website_count = 0
    external_count = 0

    # Process website files
    print("  Processing website pages...")
    for text, source, page in load_raw_files(WEBSITE_DIR, "website"):
        cleaned = clean_text(text)
        if cleaned:
            chunks = chunk_text(cleaned, source, page)
            all_chunks.extend(chunks)
            website_count += len(chunks)
            print(f"    ✓ {page}: {len(chunks)} chunks")

    print()

    # Process external files
    print("  Processing external sources...")
    for text, source, page in load_raw_files(EXTERNAL_DIR, "external"):
        cleaned = clean_text(text)
        if cleaned:
            chunks = chunk_text(cleaned, source, page)
            all_chunks.extend(chunks)
            external_count += len(chunks)
            print(f"    ✓ {page}: {len(chunks)} chunks")

    print()

    # Deduplicate
    before = len(all_chunks)
    all_chunks = deduplicate_chunks(all_chunks)
    after = len(all_chunks)
    print(f"  Deduplication: {before} → {after} chunks (removed {before - after})")

    # Assign IDs
    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"chunk_{i + 1:04d}"

    # Save chunks
    chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Write seed_facts.txt
    with open(SEED_FACTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(SEED_FACTS))
    print(f"  ✓ Seed facts written: {len(SEED_FACTS)} facts")

    print()
    print("═" * 60)
    print(f"  ✓ Chunks created: {after} | Sources: website={website_count} external={external_count}")
    print(f"  ✓ Saved to: {chunks_path}")
    print("═" * 60)


if __name__ == "__main__":
    main()
