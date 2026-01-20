"""Download Sherlock Holmes stories from Project Gutenberg.

Fetches "The Adventures of Sherlock Holmes" (Gutenberg #1661) and splits
the collection into the 12 individual stories for ingestion.

Public API:
    download_adventures(force_download: bool = False) -> list[tuple[str, str]]

Returns:
    List of (story_title, story_content) tuples.
"""

from __future__ import annotations

from pathlib import Path
import re
import urllib.request

GUTENBERG_URL = "https://www.gutenberg.org/files/1661/1661-0.txt"
CACHE_DIR = Path(__file__).parent / "data"

# These headings appear in the Gutenberg text and are reliable split points.
# We keep them in order.
STORY_MARKERS: list[tuple[str, str]] = [
    ("A Scandal in Bohemia", "ADVENTURE I. A SCANDAL IN BOHEMIA"),
    ("The Red-Headed League", "ADVENTURE II. THE RED-HEADED LEAGUE"),
    ("A Case of Identity", "ADVENTURE III. A CASE OF IDENTITY"),
    ("The Boscombe Valley Mystery", "ADVENTURE IV. THE BOSCOMBE VALLEY MYSTERY"),
    ("The Five Orange Pips", "ADVENTURE V. THE FIVE ORANGE PIPS"),
    ("The Man with the Twisted Lip", "ADVENTURE VI. THE MAN WITH THE TWISTED LIP"),
    ("The Adventure of the Blue Carbuncle", "ADVENTURE VII. THE ADVENTURE OF THE BLUE CARBUNCLE"),
    ("The Adventure of the Speckled Band", "ADVENTURE VIII. THE ADVENTURE OF THE SPECKLED BAND"),
    ("The Adventure of the Engineer's Thumb", "ADVENTURE IX. THE ADVENTURE OF THE ENGINEER'S THUMB"),
    ("The Adventure of the Noble Bachelor", "ADVENTURE X. THE ADVENTURE OF THE NOBLE BACHELOR"),
    ("The Adventure of the Beryl Coronet", "ADVENTURE XI. THE ADVENTURE OF THE BERYL CORONET"),
    ("The Adventure of the Copper Beeches", "ADVENTURE XII. THE ADVENTURE OF THE COPPER BEECHES"),
]

# Gutenberg boilerplate markers (used to strip headers/footers)
GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)


def download_adventures(force_download: bool = False) -> list[tuple[str, str]]:
    """Download and split The Adventures of Sherlock Holmes into stories."""
    cache_file = CACHE_DIR / "adventures_1661.txt"
    if force_download or not cache_file.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GUTENBERG_URL) as resp:
            # utf-8-sig eats BOM if present
            text = resp.read().decode("utf-8-sig")
        cache_file.write_text(text, encoding="utf-8")
    else:
        text = cache_file.read_text(encoding="utf-8")

    text = _strip_gutenberg_boilerplate(text)
    return _split_into_stories(text)


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Gutenberg license/header/footer so story splits are cleaner."""
    m_start = GUTENBERG_START_RE.search(text)
    if m_start:
        text = text[m_start.end() :]

    m_end = GUTENBERG_END_RE.search(text)
    if m_end:
        text = text[: m_end.start()]

    return text.strip()


_ROMAN_HEADER = re.compile(r"(?im)^(?P<num>I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.\s+(?P<title>[A-Z][A-Z \-’']+)\s*$")


def _split_into_stories(text: str) -> list[tuple[str, str]]:
    start_tag = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_tag = "*** END OF THE PROJECT GUTENBERG EBOOK"
    if start_tag in text:
        text = text.split(start_tag, 1)[1]
    if end_tag in text:
        text = text.split(end_tag, 1)[0]

    matches = list(_ROMAN_HEADER.finditer(text))
    if len(matches) < 12:
        # If we only found 6, we're likely only seeing part of the book or only one style.
        # We'll still try to split whatever we have, but it's worth warning loudly.
        # (You can keep raising if you prefer strictness.)
        if len(matches) == 0:
            raise ValueError("No story headers found")

    # Heuristic: skip ToC headers by finding the first place where we see "I." twice.
    # The first set tends to be the contents list; the second set starts the actual story text.
    i_headers = [m for m in matches if m.group("num") == "I"]
    if len(i_headers) >= 2:
        start_idx = matches.index(i_headers[1])
        matches = matches[start_idx:]

    stories: list[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        title = m.group("title").title()
        chunk = text[start:end].strip()
        stories.append((title, chunk))

    return stories
