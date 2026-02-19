# Plan: Split medlit ingest.py by stage

Split `examples/medlit/scripts/ingest.py` into separate source files by pipeline stage. The plan is written so it can be executed mechanically without ambiguity.

**Reference file:** `examples/medlit/scripts/ingest.py` (current state; line numbers below refer to it).

**Package context:** Scripts live under `examples/medlit/scripts/`. Imports from medlit use `..` for `scripts` → `medlit` and `...` for `scripts/stages` → `medlit`. Run as `python -m examples.medlit.scripts.ingest`.

---

## 1. Target layout

After the refactor:

```
examples/medlit/scripts/
  ingest.py                 # CLI + main flow only
  progress.py               # TraceCollector, ProgressTracker, shutdown helpers
  orchestrator_build.py     # build_orchestrator, find_input_files
  stages/
    __init__.py             # Re-exports only
    entities.py             # Entity extraction phase
    promotion.py            # Promotion phase
    relationships.py        # Relationship extraction phase
    export.py               # Summary + bundle export
```

---

## 2. Step-by-step execution

Execute in this order. After each step, run the verification commands in section 4 to ensure nothing is broken.

### Step 2.1 — Create `progress.py`

**Create** `examples/medlit/scripts/progress.py` with:

1. **Imports (at top of file):**
```python
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..pipeline.authority_lookup import CanonicalIdLookup
```

2. **Constant:** Copy from ingest.py lines 86–88 (the comment and `TRACE_BASE_DIR = Path(...)`).

3. **Class `TraceCollector`:** Copy from ingest.py **lines 93–142** (from `@dataclass` through the end of `print_summary`). Use `TRACE_BASE_DIR` from this file.

4. **Class `ProgressTracker`:** Copy from ingest.py **lines 145–175** (from `@dataclass` through the end of `report`).

5. **Function `_handle_keyboard_interrupt`:** Copy from ingest.py **lines 1104–1139** (def through the final `print(..., file=sys.stderr)`). Remove the inline `import traceback` and use the top-level import.

6. **Function `_cleanup_lookup_service`:** Copy from ingest.py **lines 1142–1159** (async def through the except block).

**Update** `ingest.py`: Add import from progress:
```python
from .progress import (
    TraceCollector,
    ProgressTracker,
    _handle_keyboard_interrupt,
    _cleanup_lookup_service,
)
```
Delete from ingest.py: lines 86–88 (TRACE_BASE_DIR), lines 93–175 (TraceCollector, ProgressTracker), lines 1104–1159 (_handle_keyboard_interrupt, _cleanup_lookup_service). Do not delete anything else yet.

---

### Step 2.2 — Create `orchestrator_build.py`

**Create** `examples/medlit/scripts/orchestrator_build.py` with:

1. **Imports (at top of file):**
```python
import fnmatch
import traceback
from pathlib import Path
from typing import Any

from kgraph.ingest import IngestionOrchestrator
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from kgraph.pipeline.caching import (
    CachedEmbeddingGenerator,
    EmbeddingCacheConfig,
    FileBasedEmbeddingsCache,
)
from kgraph.pipeline.interfaces import EntityExtractorInterface
from kgraph.pipeline.streaming import (
    BatchingEntityExtractor,
    WindowedRelationshipExtractor,
)
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)
from kgraph.provenance import ProvenanceAccumulator
from ..domain import MedLitDomainSchema
from ..pipeline.authority_lookup import CanonicalIdLookup
from ..pipeline.embeddings import OllamaMedLitEmbeddingGenerator
from ..pipeline.llm_client import OllamaLLMClient
from ..pipeline.mentions import MedLitEntityExtractor
from ..pipeline.ner_extractor import MedLitNEREntityExtractor
from ..pipeline.config import load_medlit_config
from ..pipeline.pmc_chunker import PMCStreamingChunker
from ..pipeline.parser import JournalArticleParser
from ..pipeline.relationships import MedLitRelationshipExtractor
from ..pipeline.resolve import MedLitEntityResolver
```

2. **Function `build_orchestrator`:** Copy from ingest.py **lines 179–322** (def through `return orchestrator, None, cached_embedding_generator`). Replace the inline `import traceback` (around original line 257) with use of the top-level `traceback`.

3. **Function `find_input_files`:** Copy from ingest.py **lines 570–611** (def through `return files`).

**Update** `ingest.py`: Add:
```python
from .orchestrator_build import build_orchestrator, find_input_files
```
Delete from ingest.py: lines 179–322 (build_orchestrator), lines 570–611 (find_input_files). Remove any imports that are now only used by the deleted code (see section 3 for what ingest.py must still import).

---

### Step 2.3 — Create `stages/` and `stages/entities.py`

**Create** directory `examples/medlit/scripts/stages/` if it does not exist.

**Create** `examples/medlit/scripts/stages/entities.py` with:

1. **Imports:**
```python
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from kgraph.ingest import IngestionOrchestrator
from kgraph.logging import setup_logging

from ..progress import ProgressTracker
from ...stage_models import EntityExtractionStageResult, PaperEntityExtractionResult
```

2. **Logger:** Add `logger = setup_logging()` after the imports.

3. **Function `extract_entities_from_paper`:** Copy from ingest.py **lines 326–368** (async def through `return (file_path.stem, 0, 0)`). Use this module’s `logger`.

4. **Function `extract_entities_phase`:** Copy from ingest.py **lines 614–729** (async def through `return (processed, errors_count, stage_result)`). It calls `extract_entities_from_paper` (same module) and `ProgressTracker` (imported from `..progress`).

**Update** `ingest.py`: Add `from .stages.entities import extract_entities_phase`. Delete from ingest.py lines 326–368 (extract_entities_from_paper) and lines 614–729 (extract_entities_phase).

---

### Step 2.4 — Create `stages/promotion.py`

**Create** `examples/medlit/scripts/stages/promotion.py` with:

1. **Imports:**
```python
from datetime import datetime, timezone
from pathlib import Path

from kgraph.ingest import IngestionOrchestrator
from kgschema.storage import EntityStorageInterface

from ...pipeline.authority_lookup import CanonicalIdLookup
from ...stage_models import PromotionStageResult, PromotedEntityRecord
```

(From `stages/`, `...` is `medlit`, so `...pipeline` and `...stage_models` are correct.)

2. **Function `_initialize_lookup`:** Copy from ingest.py **lines 731–758** (def through `return None` / `return lookup`).

3. **Function `_build_promoted_records`:** Copy from ingest.py **lines 761–783** (def through `return promoted_records`).

4. **Function `run_promotion_phase`:** Copy from ingest.py **lines 786–852** (async def through `return lookup, stage_result`).

**Update** `ingest.py`: Add `from .stages.promotion import run_promotion_phase`. Delete from ingest.py lines 731–758, 761–783, 786–852.

---

### Step 2.5 — Create `stages/relationships.py`

**Create** `examples/medlit/scripts/stages/relationships.py` with:

1. **Imports:**
```python
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from kgraph.ingest import IngestionOrchestrator

from ..progress import ProgressTracker
from ...stage_models import (
    RelationshipExtractionStageResult,
    PaperRelationshipExtractionResult,
)
```

2. **Function `extract_relationships_from_paper`:** Copy from ingest.py **lines 371–411** (async def through `return (file_path.stem, 0, 0)`).

3. **Function `extract_relationships_phase`:** Copy from ingest.py **lines 862–957** (async def through `return (processed_rels, errors_rels, stage_result)`). It calls `extract_relationships_from_paper` (same module) and `ProgressTracker` (imported).

**Update** `ingest.py`: Add `from .stages.relationships import extract_relationships_phase`. Delete from ingest.py lines 371–411 and lines 862–957.

---

### Step 2.6 — Create `stages/export.py`

**Create** `examples/medlit/scripts/stages/export.py` with:

1. **Imports:**
```python
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from kgraph.export import write_bundle
from kgraph.provenance import ProvenanceAccumulator
from kgschema.storage import (
    DocumentStorageInterface,
    EntityStorageInterface,
    RelationshipStorageInterface,
)
```

2. **Function `print_summary`:** Copy from ingest.py **lines 959–1003** (async def through the closing `file=sys.stderr,` and `)`). Uses the top-level `sys` for `file=sys.stderr`.

3. **Function `export_bundle`:** Copy from ingest.py **lines 1006–1101** (async def through the final `file=sys.stderr,` and closing parens). Remove the inline `import shutil`; use top-level import.

**Update** `ingest.py`: Add `from .stages.export import print_summary, export_bundle`. Delete from ingest.py lines 959–1003 and 1006–1101.

---

### Step 2.7 — Create `stages/__init__.py`

**Create** `examples/medlit/scripts/stages/__init__.py` with:

```python
"""Medlit ingestion pipeline stages."""

from .entities import extract_entities_phase
from .export import export_bundle, print_summary
from .promotion import run_promotion_phase
from .relationships import extract_relationships_phase

__all__ = [
    "extract_entities_phase",
    "export_bundle",
    "print_summary",
    "run_promotion_phase",
    "extract_relationships_phase",
]
```

**Update** `ingest.py`: Optionally replace the four `from .stages.xxx import ...` lines with a single:
```python
from .stages import (
    extract_entities_phase,
    export_bundle,
    print_summary,
    run_promotion_phase,
    extract_relationships_phase,
)
```

---

### Step 2.8 — Slim `ingest.py`: remove dead code and fix imports

1. **Ensure ingest.py retains exactly:**
   - Lines 1–22 (docstring).
   - The imports listed in section 3 below (and only those needed).
   - `parse_arguments` (previously lines 413–566).
   - `_output_stage_result` (previously lines 1162–1169).
   - `_initialize_pipeline` (previously lines 1171–1235).
   - `main` (previously lines 1238–1410).
   - `if __name__ == "__main__":` block (lines 1412–1413).

2. **Remove** from ingest.py any remaining code that was moved (if any block was missed in earlier steps).

3. **Imports in ingest.py:** After all moves, ingest.py must have the imports in section 3 and no unused imports. Run a linter or grep for used symbols to drop unused imports.

---

## 3. Final imports for `ingest.py`

After the refactor, `ingest.py` must contain at least these imports (add only what is actually used):

```python
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from kgraph.logging import setup_logging
from logging import DEBUG
from pydantic import BaseModel

from ..pipeline.llm_client import LLMTimeoutError, OllamaLLMClient
from ..stage_models import (
    EntityExtractionStageResult,
    IngestionPipelineResult,
    PromotionStageResult,
    RelationshipExtractionStageResult,
)
from .progress import (
    TraceCollector,
    ProgressTracker,
    _handle_keyboard_interrupt,
    _cleanup_lookup_service,
)
from .orchestrator_build import build_orchestrator, find_input_files
from .stages import (
    extract_entities_phase,
    export_bundle,
    print_summary,
    run_promotion_phase,
    extract_relationships_phase,
)
```

And in the body, `logger = setup_logging()` (used by `parse_arguments` for `args.debug`). Remove from ingest.py any import or symbol that is only used by the moved code (e.g. `write_bundle`, `ProvenanceAccumulator`, `fnmatch`, `EntityStorageInterface`, storage classes, pipeline submodules other than `llm_client`, etc.).

---

## 4. Verification (run after each step and at the end)

From the repository root:

```bash
# 1. Help and entrypoint
uv run python -m examples.medlit.scripts.ingest --help

# 2. Medlit tests
uv run pytest examples/medlit/ -v

# 3. Stop-after (use an existing dir with at least one JSON or XML; e.g. examples/medlit/pmc_xmls)
uv run python -m examples.medlit.scripts.ingest --input-dir examples/medlit/pmc_xmls --limit 1 --stop-after entities
uv run python -m examples.medlit.scripts.ingest --input-dir examples/medlit/pmc_xmls --limit 1 --use-ollama --stop-after promotion 2>/dev/null || true
```

If any command fails, fix the introduced file before proceeding. After the full refactor, a full pipeline run (with `--use-ollama` if available) and bundle export should behave as before.

---

## 5. Summary: line ranges to move (reference)

| Destination | ingest.py line range | Content |
|-------------|----------------------|--------|
| progress.py | 86–88 | TRACE_BASE_DIR |
| progress.py | 93–142 | TraceCollector |
| progress.py | 145–175 | ProgressTracker |
| progress.py | 1104–1139 | _handle_keyboard_interrupt |
| progress.py | 1142–1159 | _cleanup_lookup_service |
| orchestrator_build.py | 179–322 | build_orchestrator |
| orchestrator_build.py | 570–611 | find_input_files |
| stages/entities.py | 326–368 | extract_entities_from_paper |
| stages/entities.py | 614–729 | extract_entities_phase |
| stages/promotion.py | 731–758 | _initialize_lookup |
| stages/promotion.py | 761–783 | _build_promoted_records |
| stages/promotion.py | 786–852 | run_promotion_phase |
| stages/relationships.py | 371–411 | extract_relationships_from_paper |
| stages/relationships.py | 862–957 | extract_relationships_phase |
| stages/export.py | 959–1003 | print_summary |
| stages/export.py | 1006–1101 | export_bundle |

**Keep in ingest.py:** Lines 1–22 (docstring), 24–85 (imports to be trimmed), 89 (logger), 413–566 (parse_arguments), 1162–1169 (_output_stage_result), 1171–1235 (_initialize_pipeline), 1238–1413 (main + if __name__).

---

## 6. Edge cases and notes

- **Logger:** `stages/entities.py` needs its own `logger = setup_logging()` for `extract_entities_from_paper` (logger.error, logger.exception). `ingest.py` keeps `logger = setup_logging()` for `parse_arguments` (logger.setLevel(DEBUG)).
- **Inline imports:** In the original file, `traceback` is imported inside two functions; `shutil` inside one. In the new files, add `traceback` and `shutil` at the top of the respective modules and remove the inline imports when copying.
- **promotion.py and CanonicalIdLookup:** From `stages/promotion.py`, `...` is `medlit`, so use `from ...pipeline.authority_lookup import CanonicalIdLookup`. Do not use `..pipeline` (that would be `scripts.pipeline`, which does not exist).
- **stage_models path:** From `scripts/stages/entities.py`, `...stage_models` means parent of parent of `stages` = `medlit`, so `...stage_models` resolves to `examples.medlit.stage_models`. Correct.
- **export_bundle and cache_file:** The function takes `cache_file: Path | None` and uses it; no change to signature. `ingest.py` passes `cache_file_path = lookup.cache_file if lookup else cache_file`; that remains in main.
