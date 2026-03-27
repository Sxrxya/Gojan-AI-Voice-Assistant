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

PLACEMENT_FACTS = """Gojan School of Business and Technology has an active Training and Placement Cell.
The placement cell connects students with top companies for campus recruitment.
Top recruiters visiting GSBT campus include TCS, Infosys, Wipro, Cognizant, and HCL Technologies.
Campus recruitment drives are conducted every year from October to March.
Pre-placement training includes aptitude tests, soft skills, group discussions, and mock interviews.
Both on-campus and off-campus placement support is provided to all students.
Placement reports are published annually and available on the college website.
Students are trained in resume building, technical skills, and interview preparation.
Eligible students from all departments are placed in reputed companies.
Sponsorship is provided to Gojan students and alumni for higher education."""

HOSTEL_FACTS = """Gojan School of Business and Technology has separate hostels for boys and girls within the 80-acre campus.
The hostel provides excellent infrastructure and all necessary facilities.
Hostel mess provides nutritious and hygienic food with vegetarian and non-vegetarian options.
The menu has variety catering to students from all over South India.
Hostel wardens are present round the clock giving attention to every detail.
The hostel ambience is described as a home away from home.
Hostel rooms are shared accommodation with bed, table, chair, and cupboard.
24x7 security supervision and CCTV surveillance is available in hostel premises.
Hostel has WiFi connectivity and common room with TV for recreation."""

TRANSPORT_FACTS = """Gojan School of Business and Technology operates 14 college bus routes across Chennai.
Bus Route 03 covers Pulianthope area, Route 04 covers Aminjikarai area.
Bus Route 05 covers Kundrathur, Route 07 covers Perambur area.
Bus Route 09 covers Minjur, Route 11 covers Naapaalayam area.
Bus Route 12 covers Tondiarpet, Route 15 covers Thiruninravur area.
Bus Route 16 covers Maduravoyal, Route 18 covers Chinna Mathur area.
Bus Route 19 covers IOC Power House, Route 20 covers Ennore area.
Bus Route 21 covers Pazhaverkadu, Route 23 covers Elavoor area.
Each bus has a dedicated driver with contact number for communication.
All college buses are available for daily pick-up and drop of students.
Transport facility covers North Chennai, South Chennai, and West Chennai areas."""

CAMPUS_FACTS = """Gojan School of Business and Technology campus spans 80 acres at Redhills, Chennai.
The campus has WiFi connectivity throughout for students and faculty.
The college library has a wide collection of books, journals, and digital resources.
Multiple computer labs with modern systems are available for all departments.
The campus has sports facilities including indoor and outdoor games.
The college has an Entrepreneurship Development Cell (ED Cell) for aspiring entrepreneurs.
The college organizes NSS activities, cultural events, and technical symposiums.
Gojan Clubs provide opportunities for students in arts, culture, and extracurricular activities.
The college has IEEE Student Branch and ISTE Chapter as professional bodies.
The college participates in NIRF (National Institutional Ranking Framework) rankings.
The college conducts annual graduation day ceremony for passing out students."""



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
        "",
        "=== PLACEMENT FACTS ===",
        PLACEMENT_FACTS.strip(),
        "",
        "=== HOSTEL FACTS ===",
        HOSTEL_FACTS.strip(),
        "",
        "=== TRANSPORT FACTS ===",
        TRANSPORT_FACTS.strip(),
        "",
        "=== CAMPUS FACTS ===",
        CAMPUS_FACTS.strip(),
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
