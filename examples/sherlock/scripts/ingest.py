# examples/sherlock/ingest.py
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from kgraph.ingest import IngestionOrchestrator
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)
from kgraph.export import write_bundle

from ..sources.gutenberg import download_adventures
from ..pipeline import (
    SherlockDocumentParser,
    SherlockEntityExtractor,
    SherlockEntityResolver,
    SherlockRelationshipExtractor,
    SimpleEmbeddingGenerator,
)
from ..domain import SherlockDomainSchema

###################
# A domain wants its own doc files for Mkdocs, right?
# We got you covered.

build_orch_blurb = """
# What's so great about build_orchestrator() ?

Great question — this gets to the *why* of having `build_orchestrator()` at all.

Short answer: **it packages all the domain-specific wiring into one stable,
boring, copy-paste-able place** so extenders don’t have to re-learn the
ingestion graph every time.

Let me be concrete.

---

## What `build_orchestrator()` actually gives an extender

### 1. A *single authoritative wiring point*

`IngestionOrchestrator` has a lot of moving parts:

```python
IngestionOrchestrator(
    domain=...,
    parser=...,
    entity_extractor=...,
    entity_resolver=...,
    relationship_extractor=...,
    embedding_generator=...,
    entity_storage=...,
    relationship_storage=...,
    document_storage=...,
)
```

An extender **should not have to remember**:

* which extractor goes in which slot
* which ones are async
* which storages are required vs optional
* which defaults are safe

`build_orchestrator()` freezes all of that into one known-good configuration.

If someone wants to build *their own* Sherlock-like domain, they can literally
start by copying that function.

---

### 2. A stable surface for experimentation

Extenders often want to tweak **one thing**:

* swap out the embedding generator
* replace the relationship extractor
* use a persistent storage instead of in-memory
* add logging or metrics

With `build_orchestrator()` they can do:

```python
orch = build_orchestrator(
    embedding_generator=MyBetterEmbeddings(),
)
```

or:

```python
orch = build_orchestrator(
    relationship_extractor=MySmarterCoOccurrenceExtractor(),
)
```

instead of re-wiring everything manually.

That’s huge for iteration speed.

---

### 3. A pedagogical artifact

For *this repository*, `build_orchestrator()` is also documentation.

It answers, in executable code:

> “What does a complete, correct kgraph ingestion pipeline look like for a real
> domain?”

That’s much clearer than prose.

Someone reading the example learns:

* which interfaces matter
* how domain schema + pipeline pieces fit together
* which parts are optional vs essential

---

### 4. A place to encode “best practice defaults”

You’ve already discovered this implicitly:

* in-memory storage is fine for examples
* promotion config lives on the domain
* parser must run before entity extraction
* relationship extractor must not create entities

`build_orchestrator()` is where those conventions live.

If you later discover a better default (e.g. batching embeddings, better clock
handling), you change it **once**.

---

## What it does *not* give (important)

It is **not**:

* a required abstraction
* a magic factory
* something users must use

It’s a **convenience + example**, not a framework constraint.

Advanced users can ignore it entirely.

---

## Where I’d put it

You have two reasonable options:

### Option A (most pedagogical)

```text
examples/sherlock/scripts/ingest.py
```

Keep it near the CLI entrypoint, clearly labeled “example wiring”.

### Option B (clean API surface)

```text
examples/sherlock/__init__.py
```

Export it so people can do:

```python
from examples.sherlock import build_orchestrator
```

This makes Sherlock feel like a mini-library.

Either is fine; Option A emphasizes “example”, Option B emphasizes “reusable”.

---

## TL;DR

Extenders want `build_orchestrator()` because it:

* shows the **entire pipeline wiring in one place**
* reduces cognitive load when experimenting
* is copy-pasteable into new domains
* encodes best practices you’ve already learned the hard way

If you didn’t provide it, many users would re-invent it badly.

If you want, I can sketch a *final form* `build_orchestrator()` signature
that’s maximally helpful but still minimal.
"""


mkdocs_yml_replacement = """
site_name: Domain-Agnostic Knowledge Graph Server
site_description: Documentation for the domain-agnostic knowledge graph server
repo_url: https://github.com/wware/kgserver
repo_name: wware/kgserver

theme:
  name: material

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
  - tables
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Architecture: architecture.md
  - BuildOrchestrator: build_orch.md
"""

##########################


def build_orchestrator() -> IngestionOrchestrator:
    domain = SherlockDomainSchema()

    return IngestionOrchestrator(
        domain=domain,
        parser=SherlockDocumentParser(),
        entity_extractor=SherlockEntityExtractor(),
        entity_resolver=SherlockEntityResolver(domain=domain),
        relationship_extractor=SherlockRelationshipExtractor(),
        embedding_generator=SimpleEmbeddingGenerator(),
        entity_storage=InMemoryEntityStorage(),
        relationship_storage=InMemoryRelationshipStorage(),
        document_storage=InMemoryDocumentStorage(),
    )


async def main() -> None:
    print("=" * 60)
    print("Sherlock Holmes Knowledge Graph - Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/4] Downloading stories...")
    stories = download_adventures()
    print(f"      Loaded {len(stories)} stories")

    print("\n[2/4] Initializing pipeline...")
    entity_storage = InMemoryEntityStorage()
    relationship_storage = InMemoryRelationshipStorage()
    document_storage = InMemoryDocumentStorage()

    domain_schema = SherlockDomainSchema()

    orchestrator = IngestionOrchestrator(
        domain=domain_schema,
        parser=SherlockDocumentParser(),
        entity_extractor=SherlockEntityExtractor(),
        entity_resolver=SherlockEntityResolver(domain=domain_schema),
        relationship_extractor=SherlockRelationshipExtractor(),
        embedding_generator=SimpleEmbeddingGenerator(),
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
    )

    print("\n[3/4] Ingesting stories...")
    for title, content in stories:
        res = await orchestrator.ingest_document(
            raw_content=content.encode("utf-8"),
            content_type="text/plain",
            source_uri=title,
        )
        print(f"      {title}: {res.entities_extracted} entities, {res.relationships_extracted} relationships")

    print("\n[4/4] Summary")
    doc_count = await document_storage.count()
    ent_count = await entity_storage.count()
    rel_count = await relationship_storage.count()

    print(f"""
    ╔══════════════════════════════════════╗
    ║       Knowledge Graph Summary        ║
    ╠══════════════════════════════════════╣
    ║  Documents:               {doc_count:>10} ║
    ║  Entities:                {ent_count:>10} ║
    ║  Relationships:           {rel_count:>10} ║
    ╚══════════════════════════════════════╝
    """)

    with TemporaryDirectory() as tmpdir:
        # Create documentation assets that will be packaged in the bundle.
        # These files will be listed in documents.jsonl and copied to the bundle's docs/ directory.
        temp_path = Path(tmpdir)
        md_file = temp_path / "build_orch.md"
        md_file.write_text(build_orch_blurb)
        mkdocs_file = temp_path / "mkdocs.yml"
        mkdocs_file.write_text(mkdocs_yml_replacement)

        # Export the bundle with documentation assets.
        # The write_bundle function will create documents.jsonl and copy all files
        # from temp_path to bundle_path/docs/, preserving directory structure.
        bundle_output_path = Path("./sherlock_bundle")
        await write_bundle(
            entity_storage=entity_storage,
            relationship_storage=relationship_storage,
            bundle_path=bundle_output_path,
            domain="sherlock",
            label="sherlock-holmes-stories",
            docs=temp_path,
            description="Knowledge graph bundle of Sherlock Holmes stories",
        )
        # as the context ends, tmpdir is deleted


if __name__ == "__main__":
    asyncio.run(main())
