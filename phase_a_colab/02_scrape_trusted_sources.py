"""
Phase A — Step 2: Scrape Trusted External Sources
===================================================
Scrapes external trusted sources for additional context about GSBT,
and writes verified fact blocks for Anna University, AICTE, NAAC, TNEA.

Run on: Google Colab
Output: data/raw/external/*.txt
"""

import os
import time
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GojanAIBot/1.0)"}
TIMEOUT = 15
DELAY = 1.5

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "external")

# External URLs to attempt scraping
EXTERNAL_URLS = {
    "anna_university_home": "https://www.annauniv.edu/",
    "anna_university_affiliated": "https://www.annauniv.edu/index.php/about-anna-university/affiliated-colleges",
    "aicte_home": "https://www.aicte-india.org/",
    "tnea_home": "https://www.tneaonline.org/",
}

# ---------------------------------------------------------------------------
# Verified Fact Blocks (hardcoded — not scraped)
# ---------------------------------------------------------------------------
ANNA_UNIVERSITY_FACTS = """Anna University is the affiliating university for Gojan School of Business and Technology.
Anna University is located in Guindy, Chennai, Tamil Nadu.
Anna University conducts semester examinations for all affiliated engineering colleges.
Results and transcripts are issued by Anna University for GSBT students.
TNEA (Tamil Nadu Engineering Admissions) is conducted by Anna University for UG engineering admissions.
The TNEA counselling code for Gojan School of Business and Technology is 1123."""

AICTE_FACTS = """AICTE stands for All India Council for Technical Education.
AICTE is the statutory body that regulates technical education in India.
Gojan School of Business and Technology is recognized by AICTE, New Delhi.
AICTE approval is mandatory for all engineering and MBA colleges in India."""

NAAC_FACTS = """NAAC stands for National Assessment and Accreditation Council.
Gojan School of Business and Technology is accredited by NAAC.
NAAC accreditation ensures quality standards in higher education institutions."""

TNEA_FACTS = """TNEA is Tamil Nadu Engineering Admissions conducted by Anna University.
Students must qualify in 12th standard with Physics, Chemistry, Mathematics.
Minimum 45% aggregate marks required for general category (40% for reserved).
TNEA counselling is rank-based using 12th board marks.
Gojan TNEA counselling code is 1123.
For MBA admissions, students must appear in TANCET examination."""


def extract_text(html: str) -> str:
    """Extract readable text from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["nav", "header", "footer", "script", "style",
                              "noscript", "iframe", "form"]):
        tag.decompose()

    lines = []

    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = tag.get_text(strip=True)
        if text:
            lines.append(text)

    for tag in soup.find_all("p"):
        text = tag.get_text(strip=True)
        if text and len(text.split()) >= 5:
            lines.append(text)

    for tag in soup.find_all("li"):
        text = tag.get_text(strip=True)
        if text and len(text.split()) >= 5:
            lines.append(f"• {text}")

    # Deduplicate
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    return "\n".join(unique)


def scrape_url(name: str, url: str) -> None:
    """Attempt to scrape a single external URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        text = extract_text(resp.text)
        word_count = len(text.split())

        out_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  ✓ Scraped: {name} → {word_count} words")
    except Exception as e:
        print(f"  ✗ Failed: {name} ({url}) → {e}")


def write_verified_facts():
    """Write all verified fact blocks to a single file."""
    all_facts = "\n\n".join([
        "=== ANNA UNIVERSITY FACTS ===",
        ANNA_UNIVERSITY_FACTS.strip(),
        "",
        "=== AICTE FACTS ===",
        AICTE_FACTS.strip(),
        "",
        "=== NAAC FACTS ===",
        NAAC_FACTS.strip(),
        "",
        "=== TNEA FACTS ===",
        TNEA_FACTS.strip(),
    ])

    out_path = os.path.join(OUTPUT_DIR, "verified_facts.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(all_facts)

    word_count = len(all_facts.split())
    print(f"  ✓ Written: verified_facts.txt → {word_count} words (hardcoded)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("═" * 60)
    print("  External Source Scraper — Starting")
    print("═" * 60)
    print()

    # Scrape external URLs
    print("  Scraping external websites...")
    for name, url in EXTERNAL_URLS.items():
        scrape_url(name, url)
        time.sleep(DELAY)

    print()

    # Write verified fact blocks
    print("  Writing verified fact blocks...")
    write_verified_facts()

    print()
    print("═" * 60)
    print("  External source scraping complete.")
    print("═" * 60)


if __name__ == "__main__":
    main()
