"""
Phase A — Step 1: Scrape Gojan Education Website
=================================================
Scrapes all specified pages from https://gojaneducation.tech/
and saves cleaned text per page.

Run on: Google Colab
Output: data/raw/website/{slug}.txt + data/raw/website/index.json
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GojanAIBot/1.0)"}
TIMEOUT = 15
DELAY = 1.5  # seconds between requests

URLS = [
    # ── Main Pages ──────────────────────────────────────────────
    "https://gojaneducation.tech/",
    "https://gojaneducation.tech/about-us/",
    "https://gojaneducation.tech/v-m/",
    "https://gojaneducation.tech/chairman-2/",
    "https://gojaneducation.tech/cp-msg/",
    "https://gojaneducation.tech/organizational-structure/",
    "https://gojaneducation.tech/committees/",
    "https://gojaneducation.tech/academic-calendar/",
    "https://gojaneducation.tech/professional-bodies/",
    # ── Departments ─────────────────────────────────────────────
    "https://gojaneducation.tech/departments/",
    "https://gojaneducation.tech/aeronautical-engg/",
    "https://gojaneducation.tech/computer-science-engg/",
    "https://gojaneducation.tech/electronics-comm-engg/",
    "https://gojaneducation.tech/information-technology/",
    "https://gojaneducation.tech/ai-ml/",
    "https://gojaneducation.tech/cyber-security-engg/",
    "https://gojaneducation.tech/mechanical-automation-engg/",
    "https://gojaneducation.tech/medical-electronics-engg/",
    "https://gojaneducation.tech/master-of-business-administration/",
    # ── Admissions & Placements ─────────────────────────────────
    "https://gojaneducation.tech/admissions/",
    "https://gojaneducation.tech/eligibility/",
    "https://gojaneducation.tech/placements-2/",
    # ── Campus Life ─────────────────────────────────────────────
    "https://gojaneducation.tech/edc/",
    "https://gojaneducation.tech/campus-life/",
    "https://gojaneducation.tech/nss/",
    "https://gojaneducation.tech/gojan-clubs/",
    # ── Accreditation & Rankings ────────────────────────────────
    "https://gojaneducation.tech/naac-p1/",
    "https://gojaneducation.tech/nisp/",
    "https://gojaneducation.tech/nirf-3/",
    "https://gojaneducation.tech/gojan-campus-insights/",
    # ── Existing page_id pages ──────────────────────────────────
    "https://gojaneducation.tech/?page_id=1476",   # Hostel Facilities
    "https://gojaneducation.tech/?page_id=1274",   # Sports
    "https://gojaneducation.tech/?page_id=521",    # Placement Records
    "https://gojaneducation.tech/?page_id=524",    # Contact Us
    "https://gojaneducation.tech/?page_id=506",    # Courses
    "https://gojaneducation.tech/?page_id=1978",   # Faculty
    "https://gojaneducation.tech/?page_id=1968",   # Events
    "https://gojaneducation.tech/?page_id=1006",   # Research
    "https://gojaneducation.tech/?page_id=1503",   # Bus Routes / Transport
    "https://gojaneducation.tech/?page_id=1594",   # Skill Enhancement
    "https://gojaneducation.tech/?page_id=1542",   # Anna Univ Rank Holders
    "https://gojaneducation.tech/?page_id=1609",   # Permanently Affiliated Courses
    # ── NEW: Discovered from website navigation ─────────────────
    "https://gojaneducation.tech/?page_id=502",    # About (detailed)
    "https://gojaneducation.tech/?page_id=512",    # Admissions (detailed)
    "https://gojaneducation.tech/?page_id=930",    # Academic Schedule
    "https://gojaneducation.tech/?page_id=1053",   # Downloads
    "https://gojaneducation.tech/?page_id=1162",   # Approval & Affiliation
    "https://gojaneducation.tech/?page_id=1225",   # IEEE Student Branch
    "https://gojaneducation.tech/?page_id=1986",   # Career
    "https://gojaneducation.tech/?page_id=1991",   # Jobs
    "https://gojaneducation.tech/?page_id=2010",   # Graduation Day Report
    "https://gojaneducation.tech/?page_id=2123",   # NAAC (new page)
    "https://gojaneducation.tech/?page_id=2642",   # Gojan Clubs (detailed)
    "https://gojaneducation.tech/stakeholders-feedback/",   # Stakeholder Feedback
    "https://gojaneducation.tech/mandatory-disclosure/",    # Mandatory Disclosure (AICTE)
]

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "website")


def url_to_slug(url: str) -> str:
    """Convert a URL to a filesystem-safe slug."""
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", path)
    else:
        qs = parse_qs(parsed.query)
        if "page_id" in qs:
            slug = f"page_{qs['page_id'][0]}"
        else:
            slug = "homepage"
    return slug


def extract_text(soup: BeautifulSoup) -> str:
    """Extract meaningful text from a parsed page."""
    # Remove unwanted tags
    for tag in soup.find_all(["nav", "header", "footer", "script", "style",
                              "noscript", "iframe", "form"]):
        tag.decompose()

    lines = []

    # Page title
    title_tag = soup.find("h1") or soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        lines.append(title_tag.get_text(strip=True))
        lines.append("")

    # Headings
    for tag in soup.find_all(["h2", "h3", "h4"]):
        text = tag.get_text(strip=True)
        if text:
            lines.append(text)

    # Paragraphs
    for tag in soup.find_all("p"):
        text = tag.get_text(strip=True)
        if text and len(text.split()) >= 5:
            lines.append(text)

    # List items
    for tag in soup.find_all("li"):
        text = tag.get_text(strip=True)
        if text and len(text.split()) >= 5:
            lines.append(f"• {text}")

    # Table cells — convert to "Label: Value" pairs
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]
            if len(cell_texts) == 2:
                lines.append(f"{cell_texts[0]}: {cell_texts[1]}")
            elif len(cell_texts) == 1 and len(cell_texts[0].split()) >= 3:
                lines.append(cell_texts[0])

    # Deduplicate while preserving order
    seen = set()
    unique_lines = []
    for line in lines:
        normalized = line.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_lines.append(normalized)

    return "\n".join(unique_lines)


def scrape_page(url: str) -> dict:
    """Scrape a single page and return metadata + text."""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    text = extract_text(soup)
    slug = url_to_slug(url)
    word_count = len(text.split())

    return {
        "url": url,
        "slug": slug,
        "text": text,
        "word_count": word_count,
        "status": "ok",
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index_data = []
    total_words = 0
    success_count = 0

    print("═" * 60)
    print("  Gojan Website Scraper — Starting")
    print("═" * 60)
    print()

    for i, url in enumerate(URLS, 1):
        try:
            result = scrape_page(url)

            # Save text file
            txt_path = os.path.join(OUTPUT_DIR, f"{result['slug']}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            index_data.append({
                "url": result["url"],
                "slug": result["slug"],
                "word_count": result["word_count"],
                "status": result["status"],
            })

            total_words += result["word_count"]
            success_count += 1
            print(f"  ✓ [{i}/{len(URLS)}] Scraped: {url} → {result['word_count']} words")

        except Exception as e:
            print(f"  ✗ [{i}/{len(URLS)}] Failed: {url} → {e}")
            index_data.append({
                "url": url,
                "slug": url_to_slug(url),
                "word_count": 0,
                "status": f"error: {e}",
            })

        # Polite delay between requests
        if i < len(URLS):
            time.sleep(DELAY)

    # Save index
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print()
    print("═" * 60)
    print(f"  Total: {success_count}/{len(URLS)} pages | {total_words} words")
    print("═" * 60)


if __name__ == "__main__":
    main()
