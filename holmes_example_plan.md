# Sherlock Holmes Knowledge Graph Example - Implementation Plan

## Overview

Create a comprehensive, well-documented example that demonstrates how to build a domain-specific knowledge graph using **kgraph**.
The example downloads Sherlock Holmes stories from Project Gutenberg and creates a queryable knowledge graph of **characters**, **locations**, **stories**, and their **relationships**.

**Purpose**
Provide a complete, correct, and idiomatic reference example for users who want to extend or adapt `kgraph` for their own domains.

Key design goals:

* Faithful alignment with the `kgraph` ingestion pipeline
* Clear separation of responsibilities between pipeline passes
* Canonical vs provisional entity lifecycle illustrated explicitly
* Readable, teachable code over cleverness

---

## File Structure

```
examples/sherlock/
├── README.md           # Comprehensive documentation for users
├── __init__.py         # Package init
├── domain.py           # SherlockDomainSchema + entity/relationship classes
├── characters.py       # Curated character list with aliases
├── extractors.py       # Pattern-based entity/relationship extractors
├── download.py         # Fetch stories from Project Gutenberg
├── ingest.py           # Run the full ingestion pipeline
└── query.py            # Demo queries against the knowledge graph
```

---

## 1. `examples/sherlock/README.md`

Documentation covering:
- **Purpose**: Demonstrate kgraph with a real literary corpus
- **Prerequisites**: Python 3.12+, kgraph installed
- **Quick start**: `python -m examples.sherlock.ingest`
- **Architecture overview** with diagram showing data flow
- **How to extend**: Add characters, new relationship types, new stories
- **How to adapt**: Guidelines for creating similar examples for other domains
- **Links** to relevant kgraph documentation (docs/domains.md, docs/pipeline.md)

## 2. `examples/sherlock/domain.py`

**Entity Types (subclass BaseEntity):**

```python
class SherlockCharacter(BaseEntity):
    """A character in the Sherlock Holmes stories."""
    role: str | None = None  # "detective", "client", "villain", "inspector", "landlady"

    def get_entity_type(self) -> str:
        return "character"


class SherlockLocation(BaseEntity):
    """A location mentioned in the stories."""
    location_type: str | None = None  # "residence", "institution", "city", "country"

    def get_entity_type(self) -> str:
        return "location"


class SherlockStory(BaseEntity):
    """A story or novel in the Holmes canon."""
    collection: str | None = None  # "The Adventures of Sherlock Holmes"
    publication_year: int | None = None

    def get_entity_type(self) -> str:
        return "story"
```

**Relationship Types (subclass BaseRelationship):**

```python
class AppearsInRelationship(BaseRelationship):
    """Character appears in a story."""
    def get_edge_type(self) -> str:
        return "appears_in"


class CoOccursWithRelationship(BaseRelationship):
    """Characters co-occur within the same textual context."""
    def get_edge_type(self) -> str:
        return "co_occurs_with"


class LivesAtRelationship(BaseRelationship):
    """Character lives at a location."""
    def get_edge_type(self) -> str:
        return "lives_at"


class AntagonistOfRelationship(BaseRelationship):
    """Character is an antagonist of another character."""
    def get_edge_type(self) -> str:
        return "antagonist_of"


class AllyOfRelationship(BaseRelationship):
    """Character is an ally of another character."""
    def get_edge_type(self) -> str:
        return "ally_of"
```

**Document Type:**

```python
class SherlockDocument(BaseDocument):
    """A Sherlock Holmes story document."""
    story_id: str | None = None
    collection: str | None = None

    def get_document_type(self) -> str:
        return "sherlock_story"

    def get_sections(self) -> list[tuple[str, str]]:
        return [("body", self.content)]
```

**Domain Schema:**

```python
class SherlockDomainSchema(DomainSchema):
    @property
    def name(self) -> str:
        return "sherlock"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        return {
            "character": SherlockCharacter,
            "location": SherlockLocation,
            "story": SherlockStory,
        }

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        return {
            "appears_in": AppearsInRelationship,
            "co_occurs_with": CoOccursWithRelationship,
            "lives_at": LivesAtRelationship,
            "antagonist_of": AntagonistOfRelationship,
            "ally_of": AllyOfRelationship,
        }

    @property
    def document_types(self) -> dict[str, type[BaseDocument]]:
        return {"sherlock_story": SherlockDocument}

    @property
    def promotion_config(self) -> PromotionConfig:
        return PromotionConfig(
            min_usage_count=2,
            min_confidence=0.7,
            require_embedding=False,
        )

    def validate_entity(self, entity: BaseEntity) -> bool:
        return entity.get_entity_type() in self.entity_types

    def validate_relationship(self, rel: BaseRelationship) -> bool:
        return rel.predicate in self.relationship_types
```

**Canonical ID Scheme:**
- Characters: `holmes:char:SherlockHolmes`, `holmes:char:JohnWatson`
- Locations: `holmes:loc:BakerStreet221B`, `holmes:loc:ScotlandYard`
- Stories: `holmes:story:AScandalInBohemia`

## 3. `examples/sherlock/characters.py`

Curated dictionaries of known characters and locations with aliases:

```python
"""Curated list of Sherlock Holmes characters and locations with aliases.

This module provides canonical entity definitions that the extractor uses
to match text mentions to known entities. Each entry includes:
- Canonical ID following the holmes:type:Name pattern
- Primary display name
- List of aliases (alternative spellings, titles, nicknames)
- Role or type classification

To add a new character, add an entry to KNOWN_CHARACTERS following the
existing pattern. Aliases should include all variations that appear in
the original texts.
"""

KNOWN_CHARACTERS: dict[str, dict] = {
    "holmes:char:SherlockHolmes": {
        "name": "Sherlock Holmes",
        "aliases": [
            "Holmes", "Sherlock", "Mr. Holmes", "Mr. Sherlock Holmes",
            "the detective", "my friend Holmes"
        ],
        "role": "detective",
    },
    "holmes:char:JohnWatson": {
        "name": "Dr. John Watson",
        "aliases": [
            "Watson", "Dr. Watson", "John Watson", "Doctor Watson",
            "John", "my friend Watson", "the doctor"
        ],
        "role": "narrator",
    },
    "holmes:char:JamesMoriarty": {
        "name": "Professor James Moriarty",
        "aliases": [
            "Moriarty", "Professor Moriarty", "the Professor",
            "James Moriarty", "the Napoleon of crime"
        ],
        "role": "villain",
    },
    "holmes:char:IreneAdler": {
        "name": "Irene Adler",
        "aliases": ["Irene", "the woman", "Miss Adler", "Adler"],
        "role": "client",
    },
    "holmes:char:InspectorLestrade": {
        "name": "Inspector Lestrade",
        "aliases": ["Lestrade", "Inspector", "G. Lestrade"],
        "role": "inspector",
    },
    "holmes:char:MrsHudson": {
        "name": "Mrs. Hudson",
        "aliases": ["Mrs Hudson", "the landlady", "our landlady"],
        "role": "landlady",
    },
    "holmes:char:MycroftHolmes": {
        "name": "Mycroft Holmes",
        "aliases": ["Mycroft", "my brother Mycroft", "his brother"],
        "role": "government",
    },
    "holmes:char:MaryMorstan": {
        "name": "Mary Morstan",
        "aliases": ["Mary", "Miss Morstan", "Mrs. Watson", "Mary Watson"],
        "role": "client",
    },
    # ... Continue with ~30 more characters including:
    # - Colonel Sebastian Moran
    # - Charles Augustus Milverton
    # - John Clay
    # - Jabez Wilson
    # - Helen Stoner
    # - Dr. Grimesby Roylott
    # - Victor Trevor
    # - Reginald Musgrave
    # - Various clients from the Adventures
}

KNOWN_LOCATIONS: dict[str, dict] = {
    "holmes:loc:BakerStreet221B": {
        "name": "221B Baker Street",
        "aliases": ["Baker Street", "221B", "their lodgings", "our rooms"],
        "location_type": "residence",
    },
    "holmes:loc:ScotlandYard": {
        "name": "Scotland Yard",
        "aliases": ["the Yard", "New Scotland Yard"],
        "location_type": "institution",
    },
    "holmes:loc:London": {
        "name": "London",
        "aliases": ["the city", "the metropolis", "the great city"],
        "location_type": "city",
    },
    "holmes:loc:DiogenesClub": {
        "name": "The Diogenes Club",
        "aliases": ["Diogenes Club", "the club"],
        "location_type": "institution",
    },
    "holmes:loc:ReichenbachFalls": {
        "name": "Reichenbach Falls",
        "aliases": ["the Falls", "Reichenbach"],
        "location_type": "landmark",
    },
    # ... Continue with ~10 more locations
}

ADVENTURES_STORIES: list[dict] = [
    {
        "canonical_id": "holmes:story:AScandalInBohemia",
        "title": "A Scandal in Bohemia",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
    },
    {
        "canonical_id": "holmes:story:TheRedHeadedLeague",
        "title": "The Red-Headed League",
        "collection": "The Adventures of Sherlock Holmes",
        "year": 1891,
    },
    # ... All 12 stories from Adventures
    {
        "canonical_id": "holmes:story:ACaseOfIdentity",
        "title": "A Case of Identity",
        ...
    },
    {
        "canonical_id": "holmes:story:TheBoscombeValleyMystery",
        "title": "The Boscombe Valley Mystery",
        ...
    },
    {
        "canonical_id": "holmes:story:TheFiveOrangePips",
        "title": "The Five Orange Pips",
        ...
    },
    {
        "canonical_id": "holmes:story:TheManWithTheTwistedLip",
        "title": "The Man with the Twisted Lip",
        ...
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheBlueCarbuuncle",
        "title": "The Adventure of the Blue Carbuncle",
        ...
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheSpeckledBand",
        "title": "The Adventure of the Speckled Band",
        ...
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheEngineersThumb",
        "title": "The Adventure of the Engineer's Thumb",
        ...
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheNobleBachelor",
        "title": "The Adventure of the Noble Bachelor",
        ...
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheBerylCoronet",
        "title": "The Adventure of the Beryl Coronet",
        ...
    },
    {
        "canonical_id": "holmes:story:TheAdventureOfTheCopperBeeches",
        "title": "The Adventure of the Copper Beeches",
        ...
    },
]
```

## 4. `examples/sherlock/extractors.py`

### SherlockDocumentParser

```python
class SherlockDocumentParser(DocumentParserInterface):
    """Parse plain text Sherlock Holmes stories into SherlockDocument objects."""

    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> SherlockDocument:
        """Parse raw text content into a structured document.

        if content_type != "text/plain":
            raise ValueError("SherlockDocumentParser only supports text/plain")

        Returns:
            SherlockDocument with extracted title and content
        """
        text = raw_content.decode("utf-8")
        # Extract title from first line or source_uri
        title = self._extract_title(text, source_uri)

        story_meta = self._lookup_story_metadata(title)

        return SherlockDocument(
            document_id=str(uuid.uuid4()),
            title=title,
            content=text,
            content_type=content_type,
            source_uri=source_uri,
            created_at=datetime.now(timezone.utc),
            story_id=story_meta["canonical_id"],
            collection=story_meta["collection"],
            metadata={
                "publication_year": story_meta.get("year"),
            },
        )
```

---

### SherlockEntityExtractor

**Responsibilities:**

* Emit mentions for **characters**, **locations**, and **the story itself**
* Attach `document_id` to all mentions

```python
async def extract(self, document: BaseDocument) -> list[EntityMention]:
    mentions: list[EntityMention] = []

    # Story mention (one per document)
    mentions.append(EntityMention(
        text=document.title or "",
        entity_type="story",
        start_offset=0,
        end_offset=0,
        confidence=1.0,
        context=None,
        metadata={
            "canonical_id_hint": document.story_id,
            "document_id": document.document_id,
        },
    ))

    for pattern, canonical_id, entity_type, confidence in self._patterns:
        for match in pattern.finditer(document.content):
            mentions.append(EntityMention(
                text=match.group(),
                entity_type=entity_type,
                start_offset=match.start(),
                end_offset=match.end(),
                confidence=confidence,
                context=document.content[max(0, match.start()-50):match.end()+50],
                metadata={
                    "canonical_id_hint": canonical_id,
                    "document_id": document.document_id,
                },
            ))
    return mentions
```

---

### SherlockEntityResolver

```python
class SherlockEntityResolver(EntityResolverInterface):
    """Resolve entity mentions to canonical or provisional entities.

    Uses the canonical_id_hint from extraction metadata to match
    known characters. Creates provisional entities for unknown mentions.
    """
    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        """Resolve a single mention to an entity."""
        # Check if we have a canonical hint from extraction
        canonical_id = mention.metadata.get("canonical_id_hint")

        if canonical_id:
            # Check if already in storage
            existing = await existing_storage.get(canonical_id)
            if existing:
                return existing, mention.confidence

            entity = BaseEntity(
                entity_id=canonical_id,
                name=mention.text,
                status=EntityStatus.CANONICAL,
                confidence=mention.confidence,
                usage_count=1,
                source="sherlock:curated",
                created_at=datetime.now(timezone.utc),
            )
            return entity, mention.confidence

        entity = BaseEntity(
            entity_id=f"prov:{uuid.uuid4()}",
            name=mention.text,
            status=EntityStatus.PROVISIONAL,
            confidence=mention.confidence * 0.5,
            usage_count=1,
            source=mention.metadata["document_id"],
            created_at=datetime.now(timezone.utc),
        )
        return entity, entity.confidence
```

---

### SherlockRelationshipExtractor

**Important:**
No entity creation occurs here.

```python
class SherlockRelationshipExtractor(RelationshipExtractorInterface):
    """Extract relationships based on co-occurrence in text.

    Relationship detection strategies:
    1. appears_in: Every character mentioned in a story gets this relationship
    2. co_occurs_with: Characters mentioned within the same paragraph
    3. Confidence based on proximity and frequency of co-occurrence
    """

    async def extract(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]:
        """Extract relationships from document given resolved entities."""
        relationships: list[BaseRelationship] = []

        story = next(e for e in entities if e.get_entity_type() == "story")
        characters = [e for e in entities if e.get_entity_type() == "character"]

        for char in characters:
            relationships.append(AppearsInRelationship(
                subject_id=char.entity_id,
                predicate="appears_in",
                object_id=story.entity_id,
                confidence=0.95,
                source_documents=(document.document_id,),
                created_at=datetime.now(timezone.utc),
            ))

        paragraphs = [p for p in document.content.split("\n\n") if p.strip()]
        evidence: dict[tuple[str, str], int] = {}

        for para in paragraphs:
            present = sorted(
                {c.entity_id for c in characters if c.name in para}
            )
            for a, b in itertools.combinations(present, 2):
                key = (a, b) if a < b else (b, a)
                evidence[key] = evidence.get(key, 0) + 1

        for (a, b), count in evidence.items():
            confidence = min(0.95, 0.6 + 0.1 * count)
            relationships.append(CoOccursWithRelationship(
                subject_id=a,
                predicate="co_occurs_with",
                object_id=b,
                confidence=confidence,
                source_documents=(document.document_id,),
                created_at=datetime.now(timezone.utc),
                metadata={"co_occurrence_count": count},
            ))

        return relationships
```

**SimpleEmbeddingGenerator (implements EmbeddingGeneratorInterface):**
```python
class SimpleEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Simple hash-based embedding generator for demonstration.

    This generates deterministic embeddings based on text hashing.
    For production use, replace with a real embedding model like
    sentence-transformers or OpenAI embeddings. As implemented here,
    it only demonstrates the interface, *not* meaningful similarity.
    """

    @property
    def dimension(self) -> int:
        return 32

    async def generate(self, text: str) -> tuple[float, ...]:
        """Generate a deterministic embedding from text hash."""
        # Use hash to generate reproducible pseudo-random vector
        h = hashlib.sha256(text.lower().encode()).digest()
        values = [b / 255.0 for b in h[:self.dimension]]
        # Normalize
        magnitude = sum(v**2 for v in values) ** 0.5
        return tuple(v / magnitude for v in values)

    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [await self.generate(t) for t in texts]
```

### 5. `examples/sherlock/download.py`

```python
"""Download Sherlock Holmes stories from Project Gutenberg.

This module fetches "The Adventures of Sherlock Holmes" from Project
Gutenberg and splits it into individual stories for processing.

Usage:
    stories = download_adventures()
    for title, content in stories:
        print(f"Downloaded: {title}")
"""

import urllib.request
from pathlib import Path

GUTENBERG_URL = "https://www.gutenberg.org/files/1661/1661-0.txt"
CACHE_DIR = Path(__file__).parent / "data"

# Story titles used to split the collection
STORY_MARKERS = [
    "ADVENTURE I. A SCANDAL IN BOHEMIA",
    "ADVENTURE II. THE RED-HEADED LEAGUE",
    "ADVENTURE III. A CASE OF IDENTITY",
    # ... all 12 stories
]

def download_adventures(force_download: bool = False) -> list[tuple[str, str]]:
    """Download and split The Adventures of Sherlock Holmes.

    Args:
        force_download: If True, re-download even if cached

    Returns:
        List of (story_title, story_content) tuples
    """
    cache_file = CACHE_DIR / "adventures.txt"

    if not cache_file.exists() or force_download:
        CACHE_DIR.mkdir(exist_ok=True)
        print(f"Downloading from {GUTENBERG_URL}...")
        with urllib.request.urlopen(GUTENBERG_URL) as response:
            content = response.read().decode("utf-8-sig")
        cache_file.write_text(content)
    else:
        content = cache_file.read_text()

    return _split_into_stories(content)

def _split_into_stories(content: str) -> list[tuple[str, str]]:
    """Split the full text into individual stories."""
    stories = []
    # Implementation: find each STORY_MARKER and extract text until next marker
    # ...
    return stories
```

### 6. `examples/sherlock/ingest.py`

```python
"""Run the Sherlock Holmes knowledge graph ingestion pipeline.

This script demonstrates the full kgraph workflow:
1. Download stories from Project Gutenberg
2. Set up the Sherlock domain with all pipeline components
3. Ingest each story through the two-pass pipeline
4. Run entity promotion for frequently-mentioned characters
5. Export the knowledge graph to JSON

Usage:
    python -m examples.sherlock.ingest

Output:
    - Console summary of entities and relationships
    - JSON export in examples/sherlock/output/
"""

import asyncio
from pathlib import Path

from kgraph.ingest import IngestionOrchestrator
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)

from .domain import SherlockDomainSchema
from .extractors import (
    SherlockDocumentParser,
    SherlockEntityExtractor,
    SherlockEntityResolver,
    SherlockRelationshipExtractor,
    SimpleEmbeddingGenerator,
)
from .download import download_adventures


async def main():
    """Run the full ingestion pipeline."""
    print("=" * 60)
    print("Sherlock Holmes Knowledge Graph - Ingestion Pipeline")
    print("=" * 60)

    # 1. Download stories
    print("\n[1/5] Downloading stories from Project Gutenberg...")
    stories = download_adventures()
    print(f"      Loaded {len(stories)} stories")

    # 2. Initialize components
    print("\n[2/5] Initializing pipeline components...")
    entity_storage = InMemoryEntityStorage()
    relationship_storage = InMemoryRelationshipStorage()
    document_storage = InMemoryDocumentStorage()

    orchestrator = IngestionOrchestrator(
        domain=SherlockDomainSchema(),
        parser=SherlockDocumentParser(),
        entity_extractor=SherlockEntityExtractor(),
        entity_resolver=SherlockEntityResolver(),
        relationship_extractor=SherlockRelationshipExtractor(),
        embedding_generator=SimpleEmbeddingGenerator(),
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
    )

    # 3. Ingest stories
    print("\n[3/5] Ingesting stories...")
    for title, content in stories:
        result = await orchestrator.ingest_document(
            raw_content=content.encode("utf-8"),
            content_type="text/plain",
            source_uri=title,
        )
        print(f"      {title}: {result.entities_extracted} entities, "
              f"{result.relationships_extracted} relationships")

    # 4. Run promotion
    print("\n[4/5] Running entity promotion...")
    promoted = await orchestrator.run_promotion()
    print(f"      Promoted {len(promoted)} entities to canonical status")

    # 5. Summary and export
    print("\n[5/5] Generating summary...")
    entity_count = await entity_storage.count()
    rel_count = await relationship_storage.count()
    doc_count = await document_storage.count()

    print(f"""
    ╔══════════════════════════════════════╗
    ║       Knowledge Graph Summary        ║
    ╠══════════════════════════════════════╣
    ║  Documents:     {doc_count:>6}       ║
    ║  Entities:      {entity_count:>6}    ║
    ║  Relationships: {rel_count:>6}       ║
    ╚══════════════════════════════════════╝
    """)

    # Export to JSON
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    # ... export code using orchestrator.export_all()

    print(f"\nExported to {output_dir}/")
    print("\nRun 'python -m examples.sherlock.query' to explore the graph.")


if __name__ == "__main__":
    asyncio.run(main())
```

### 7. `examples/sherlock/query.py`

```python
"""Demonstrate queries against the Sherlock Holmes knowledge graph.

This script shows various ways to query the ingested knowledge graph:
- Find characters in a specific story
- Find stories where two characters appear together
- Explore character relationships
- Find the most connected characters

Prerequisites:
    Run 'python -m examples.sherlock.ingest' first to build the graph.

Usage:
    python -m examples.sherlock.query
"""

async def main():
    """Run example queries against the knowledge graph."""
    # Load from saved state or re-ingest
    storage = load_or_create_storage()

    print("=" * 60)
    print("Sherlock Holmes Knowledge Graph - Query Examples")
    print("=" * 60)

    # Query 1: Characters in "A Scandal in Bohemia"
    print("\n📖 Query 1: Characters in 'A Scandal in Bohemia'")
    story_id = "holmes:story:AScandalInBohemia"
    appears_in = await relationship_storage.get_by_object(story_id, "appears_in")
    for rel in appears_in:
        char = await entity_storage.get(rel.subject_id)
        print(f"   - {char.name}")

    # Query 2: Stories where Holmes and Irene Adler both appear
    print("\n📖 Query 2: Stories with both Holmes and Irene Adler")
    holmes_stories = await get_character_stories("holmes:char:SherlockHolmes")
    irene_stories = await get_character_stories("holmes:char:IreneAdler")
    shared = holmes_stories & irene_stories
    for story_id in shared:
        story = await entity_storage.get(story_id)
        print(f"   - {story.name}")

    # Query 3: Who co_occurs_with Moriarty?
    print("\n📖 Query 3: Characters who know Moriarty")
    co_occurs_with_moriarty = await relationship_storage.get_by_object(
        "holmes:char:JamesMoriarty", "co_occurs_with"
    )
    # ...

    # Query 4: Most frequently mentioned characters
    print("\n📖 Query 4: Top 10 most mentioned characters")
    all_chars = await entity_storage.list_all(status="canonical")
    chars_by_usage = sorted(all_chars, key=lambda e: e.usage_count, reverse=True)
    for i, char in enumerate(chars_by_usage[:10], 1):
        print(f"   {i}. {char.name} ({char.usage_count} mentions)")

    # Query 5: Degrees of separation
    print("\n📖 Query 5: Degrees of separation from Holmes to Moriarty")
    # BFS traversal using 'co_occurs_with' relationships
    # ...
```

<!-- I think I probably need more hints with these bits.

> ## 4. `download.py`, `ingest.py`, `query.py`

> Only **minor alignment changes**:

> * Story canonical IDs come from `characters.py`
> * Queries assume `story` entities exist from pass 1
> * No implicit entity creation in query code

> Everything else remains conceptually correct.
-->

---

## Implementation Order

1. **`characters.py`** - Define the character/location/story data first (foundation)
2. **`domain.py`** - Define schema, entity types, relationship types
3. **`extractors.py`** - Implement all pipeline components
4. **`download.py`** - Gutenberg download utility
5. **`ingest.py`** - Main ingestion script
6. **`query.py`** - Query demonstrations
7. **`README.md`** - Documentation (write last, informed by implementation)
8. **`__init__.py`** - Package exports

---

## Verification

After implementation:

```bash
# Ensure kgraph is installed
uv pip install -e ".[dev]"

# Run the ingestion
python -m examples.sherlock.ingest

# Run queries
python -m examples.sherlock.query

# Run tests if any
uv run pytest examples/sherlock/
```

**Expected output:**
- ~40 canonical character entities (from predefined list)
- ~15 location entities
- 12 story entities
- Hundreds of relationships (appears_in, co_occurs_with)

---

## Key Files to Reference

When implementing, refer to these existing files for patterns:

| File | Use For |
|------|---------|
| `/home/wware/kgraph/tests/conftest.py` | Mock implementations of all interfaces |
| `/home/wware/kgraph/kgraph/domain.py` | DomainSchema base class |
| `/home/wware/kgraph/kgraph/entity.py` | BaseEntity, EntityMention, EntityStatus |
| `/home/wware/kgraph/kgraph/relationship.py` | BaseRelationship |
| `/home/wware/kgraph/kgraph/document.py` | BaseDocument |
| `/home/wware/kgraph/kgraph/pipeline/interfaces.py` | All extractor interface definitions |
| `/home/wware/kgraph/kgraph/pipeline/embedding.py` | EmbeddingGeneratorInterface |
| `/home/wware/kgraph/kgraph/ingest.py` | IngestionOrchestrator usage |
| `/home/wware/kgraph/docs/domains.md` | Domain implementation guide |

---

## Documentation Guidelines

- Every module should have a docstring explaining its purpose and usage
- Every class should have a docstring explaining what it represents
- Every public method should have a docstring with Args/Returns
- The README.md should be comprehensive enough for users to understand and extend the example
- Include code examples in docstrings where helpful
- Link to relevant kgraph documentation
