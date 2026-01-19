# examples/sherlock/characters.py
from __future__ import annotations

"""
Curated list of Sherlock Holmes characters, locations, and story metadata.

This module is intentionally “dumb data”:
- It provides canonical IDs and alias lists for pattern matching.
- Extractors use these lists to emit mentions with canonical_id_hint.
- Resolver uses hints to create canonical entities (or provisional ones).

Canonical ID scheme:
- Characters: holmes:char:<Name>
- Locations:  holmes:loc:<Name>
- Stories:    holmes:story:<Name>
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

# -----------------------------
# Curated entities
# -----------------------------

KNOWN_CHARACTERS: dict[str, dict] = {
    "holmes:char:SherlockHolmes": {
        "name": "Sherlock Holmes",
        "aliases": (
            "Holmes",
            "Sherlock",
            "Mr. Holmes",
            "Mr. Sherlock Holmes",
            "the detective",
            "my friend Holmes",
        ),
        "role": "detective",
    },
    "holmes:char:JohnWatson": {
        "name": "Dr. John Watson",
        "aliases": (
            "Watson",
            "Dr. Watson",
            "John Watson",
            "Doctor Watson",
            "John",
            "my friend Watson",
            "the doctor",
        ),
        "role": "narrator",
    },
    "holmes:char:IreneAdler": {
        "name": "Irene Adler",
        "aliases": ("Irene", "Miss Adler", "Adler", "the woman"),
        "role": "client",
    },
    "holmes:char:InspectorLestrade": {
        "name": "Inspector Lestrade",
        "aliases": ("Lestrade", "G. Lestrade", "Inspector Lestrade"),
        "role": "inspector",
    },
    "holmes:char:MrsHudson": {
        "name": "Mrs. Hudson",
        "aliases": ("Mrs Hudson", "the landlady", "our landlady"),
        "role": "landlady",
    },
    "holmes:char:MycroftHolmes": {
        "name": "Mycroft Holmes",
        "aliases": ("Mycroft", "my brother Mycroft"),
        "role": "government",
    },
    # Add more as you like.
}

KNOWN_LOCATIONS: dict[str, dict] = {
    "holmes:loc:BakerStreet221B": {
        "name": "221B Baker Street",
        "aliases": ("Baker Street", "221B", "our rooms", "their lodgings"),
        "location_type": "residence",
    },
    "holmes:loc:ScotlandYard": {
        "name": "Scotland Yard",
        "aliases": ("the Yard", "New Scotland Yard"),
        "location_type": "institution",
    },
    "holmes:loc:London": {
        "name": "London",
        "aliases": ("the metropolis", "the city"),
        "location_type": "city",
    },
    "holmes:loc:DiogenesClub": {
        "name": "The Diogenes Club",
        "aliases": ("Diogenes Club", "the club"),
        "location_type": "institution",
    },
    "holmes:loc:ReichenbachFalls": {
        "name": "Reichenbach Falls",
        "aliases": ("Reichenbach", "the Falls"),
        "location_type": "landmark",
    },
}

# -----------------------------
# Stories (Adventures set)
# -----------------------------

ADVENTURES_STORIES: list[dict] = [
    {
        "canonical_id": "holmes:story:AScandalInBohemia",
        "title": "A Scandal in Bohemia",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        # Marker in Gutenberg text:
        "marker": "I. A SCANDAL IN BOHEMIA",
    },
    {
        "canonical_id": "holmes:story:TheRedHeadedLeague",
        "title": "The Red-Headed League",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "II. THE RED-HEADED LEAGUE",
    },
    {
        "canonical_id": "holmes:story:ACaseOfIdentity",
        "title": "A Case of Identity",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "III. A CASE OF IDENTITY",
    },
    {
        "canonical_id": "holmes:story:TheBoscombeValleyMystery",
        "title": "The Boscombe Valley Mystery",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "IV. THE BOSCOMBE VALLEY MYSTERY",
    },
    {
        "canonical_id": "holmes:story:TheFiveOrangePips",
        "title": "The Five Orange Pips",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "V. THE FIVE ORANGE PIPS",
    },
    {
        "canonical_id": "holmes:story:TheManWithTheTwistedLip",
        "title": "The Man With The Twisted Lip",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "VI. THE MAN WITH THE TWISTED LIP",
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheBlueCarbuncle",
        "title": "The Adventure Of The Blue Carbuncle",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "VII. THE ADVENTURE OF THE BLUE CARBUNCLE",
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheSpeckledBand",
        "title": "The Adventure Of The Speckled Band",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "VIII. THE ADVENTURE OF THE SPECKLED BAND",
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheEngineersThumb",
        "title": "The Adventure Of The Engineer's Thumb",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "IX. THE ADVENTURE OF THE ENGINEER'S THUMB",
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheNobleBachelor",
        "title": "The Adventure Of The Noble Bachelor",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "X. THE ADVENTURE OF THE NOBLE BACHELOR",
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheBerylCoronet",
        "title": "The Adventure Of The Beryl Coronet",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "XI. THE ADVENTURE OF THE BERYL CORONET",
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheCopperBeeches",
        "title": "The Adventure Of The Copper Beeches",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
        "marker": "XII. THE ADVENTURE OF THE COPPER BEECHES",
    },
]


def story_markers() -> list[str]:
    return [s["marker"] for s in ADVENTURES_STORIES]


def _norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("’", "'")  # Gutenberg curly apostrophe
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(" +", "", s).strip()
    return s

# build once
_STORY_BY_NORM_TITLE = {
    _norm_title(story["title"]): story
    for story in ADVENTURES_STORIES
}

def find_story_by_title(title: str) -> dict | None:
    return _STORY_BY_NORM_TITLE.get(_norm_title(title))


def find_story_by_marker(marker: str) -> dict | None:
    m = marker.strip().lower()
    for s in ADVENTURES_STORIES:
        if s["marker"].strip().lower() == m:
            return s
    return None
