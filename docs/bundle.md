# Bundle format (kgraph → kgserver)

A **bundle** is the finalized, validated artifact produced by domain-specific pipelines (kgraph) and consumed by the domain-neutral server (kgserver).

**Bundles are a strict contract.**
- Producer pipelines must export a bundle that already matches the server schema.
- The server loads bundles as-is and should fail fast if a bundle is invalid.
- Do not rely on the server to rename fields, interpret metadata, or infer structure.

This document specifies the **bundle file layout**, **manifest schema**, and **row formats**.

## Bundle Models Package

The bundle format is defined by Pydantic models in the **kgbundle** package, a lightweight standalone package with minimal dependencies (pydantic only). This package is used by both:

- **kgraph** (producer): exports bundles using `kgbundle.EntityRow`, `kgbundle.RelationshipRow`, `kgbundle.BundleManifestV1`
- **kgserver** (consumer): loads bundles using the same models

Example:
```python
from kgbundle import BundleManifestV1, EntityRow, RelationshipRow

# In kgraph - export
entity = EntityRow(
    entity_id="char:123",
    entity_type="character",
    name="Sherlock Holmes",
    status="canonical",
    usage_count=1,
    created_at="2024-01-15T10:30:00Z",
    source="sherlock:curated"
)

# In kgserver - load
with open("manifest.json") as f:
    manifest = BundleManifestV1.model_validate_json(f.read())
```

---

## What a bundle is (and is not)

A bundle **is**:
- A set of structured files (typically `manifest.json`, `entities.jsonl`, `relationships.jsonl`)
- Optionally, additional artifacts such as documentation files used by MkDocs
- Packaged as either:
  - a directory on disk, or
  - a `.zip` containing a single bundle directory

A bundle is **not**:
- Raw source documents (PDFs, HTML, text corpora)
- Intermediate NLP artifacts (token offsets, NER spans, embedding matrices)
- Alias resolution logic or ontology definition logic
- Anything that requires the server to “understand” producer internals

Those belong to producer pipelines and are out of scope for the server.

---

## Filesystem layout

A bundle is a directory (or a zip containing a directory) with this structure:

```

<bundle_root>/
manifest.json
entities.jsonl
relationships.jsonl
(optional) docs/
(optional) other artifacts...

```

Example (Sherlock demo):

```

sherlock_bundle/
manifest.json
entities.jsonl
relationships.jsonl

```

When zipped, the zip should contain the *bundle directory*, not the files at the zip root:

✅ Good:
```

S.zip
sherlock_bundle/manifest.json
sherlock_bundle/entities.jsonl
sherlock_bundle/relationships.jsonl

```

⚠️ Avoid:
```

S.zip
manifest.json
entities.jsonl
relationships.jsonl

````

(The loader can support both, but standardizing on “one folder in the zip” makes bundles easier to reason about and prevents collisions.)

---

## `manifest.json`

The manifest is the entry point. It declares:
- bundle version
- bundle identity
- domain
- the file paths and formats for entity/relationship data
- optional metadata and optional documentation/artifact declarations

### Manifest schema (v1)

```json
{
  "bundle_version": "v1",
  "bundle_id": "93f67722-203c-487c-8927-b530b7100c2f",
  "domain": "sherlock",
  "label": "sherlock-holmes-stories",
  "created_at": "2026-01-20T20:56:04.272851+00:00",

  "entities": {
    "path": "entities.jsonl",
    "format": "jsonl"
  },

  "relationships": {
    "path": "relationships.jsonl",
    "format": "jsonl"
  },

  "docs": {
    "path": "docs/",
    "mode": "overlay"
  },

  "metadata": {
    "entity_count": 21,
    "relationship_count": 42,
    "description": "Knowledge graph bundle of Sherlock Holmes stories"
  }
}
````

### Required fields

* `bundle_version`: must be `"v1"`
* `bundle_id`: stable ID for this bundle build (UUID recommended)
* `domain`: short name, e.g. `"sherlock"`, `"medlit"`
* `entities`: file reference `{path, format}`
* `relationships`: file reference `{path, format}`

### Optional fields

* `label`: human-friendly name
* `created_at`: ISO 8601 timestamp with timezone
* `metadata`: free-form JSON object; server stores it but does not interpret it
* `docs`: documentation contribution (see below)

### File reference object

A file reference is:

```json
{
  "path": "entities.jsonl",
  "format": "jsonl"
}
```

* `path` is **relative to the bundle root**
* `format` is either `"jsonl"` or `"json"`

---

## Entity file: `entities.jsonl`

Entities are newline-delimited JSON objects (JSONL). Each line is an entity row.

### Required fields

* `entity_id` (string): stable identifier for the entity
* `entity_type` (string): coarse type/class, domain-defined (e.g. `character`, `drug`, `gene`)
* `properties` (object): opaque JSON object; may be empty `{}`

### Recommended fields (part of server schema)

* `name` (string | null)
* `status` (string | null) — e.g. `"canonical"`, `"alias"`, `"deprecated"`
* `confidence` (number | null)
* `usage_count` (integer | null)
* `created_at` (string timestamp | null)
* `source` (string | null)
* `canonical_url` (string | null) — URL to authoritative source page for canonical entities

### Example entity row

```json
{
  "entity_id": "holmes:char:SherlockHolmes",
  "entity_type": "character",
  "name": "Mr. Sherlock Holmes",
  "status": "canonical",
  "confidence": 0.95,
  "usage_count": 731,
  "created_at": "2026-01-20T20:56:03.338446+00:00",
  "source": "sherlock:curated",
  "properties": {}
}
```

### Important contract notes

* **Do not** nest canonical fields under `metadata`.
* The server **does not** flatten or reinterpret producer metadata.
* If you want extra domain-specific fields, put them under `properties`.

✅ Good:

```json
{"entity_id":"x","entity_type":"drug","name":"Aspirin","canonical_url":"https://example.com/drug/1191","properties":{"rxnorm":"1191","canonical_urls":{"rxnorm":"https://example.com/drug/1191"}}}
```

Note: `canonical_url` can be stored as a top-level field (recommended) or in `properties`. The server extracts it from either location.

🚫 Bad:

```json
{"entity_id":"x","entity_type":"drug","metadata":{"name":"Aspirin","rxnorm":"1191"}}
```

---

## Relationship file: `relationships.jsonl`

Relationships are newline-delimited JSON objects (JSONL). Each line is one relationship row.

Relationships are **directed**:

`subject_id --predicate--> object_id`

### Required fields

* `subject_id` (string)
* `predicate` (string)
* `object_id` (string)
* `properties` (object): opaque JSON object; may be empty `{}`

### Recommended fields (part of server schema)

* `confidence` (number | null)
* `source_documents` (array[string] | null)
* `created_at` (string timestamp | null)

### Example relationship row

```json
{
  "subject_id": "holmes:char:SherlockHolmes",
  "object_id": "holmes:story:AScandalInBohemia",
  "predicate": "appears_in",
  "confidence": 0.95,
  "source_documents": ["8480d4da-80da-48c8-ada4-e48aff54d2a6"],
  "created_at": "2026-01-20T20:56:03.339275+00:00",
  "properties": {}
}
```

### Important contract notes

* The canonical field names are `subject_id` and `object_id`.
* **Do not** emit `source_entity_id` / `target_entity_id` in server bundles.
* The server should not be expected to “map” producer field names.

✅ Good:

```json
{"subject_id":"A","predicate":"treats","object_id":"B","properties":{}}
```

🚫 Bad:

```json
{"source_entity_id":"A","predicate":"treats","target_entity_id":"B","metadata":{...}}
```

If your producer pipeline uses alternate names internally, normalize them when exporting.

---

## Documentation artifacts (optional): `docs/`

Bundles may optionally include documentation files that the server will render via MkDocs.

Use this when the domain wants to provide:

* domain overview
* ontology notes
* bundle-specific provenance or caveats
* curated “how to query this dataset” examples

### Recommended layout

Put docs under a `docs/` directory inside the bundle:

```
<bundle_root>/
  docs/
    index.md
    querying.md
    ontology.md
    img/
      diagram.png
```

### Manifest declaration

```json
"docs": {
  "path": "docs/",
  "mode": "overlay"
}
```

`mode` options (recommended semantics):

* `overlay`: bundle docs are added alongside the server’s standard docs
* `replace`: bundle docs replace the server’s docs entirely

**Note:** Images and other static files are fine inside `docs/` (e.g. PNG/SVG). MkDocs will copy them as normal.

### Contract note

Docs have **no effect** on storage, loading, or querying. They are informational only.

---

## Validation & “fail fast”

A producer pipeline should validate:

* `manifest.json` schema
* that referenced files exist and are readable
* that JSONL parses line-by-line
* that required fields exist
* that IDs referenced in relationships exist in entities (recommended)
* that `properties` is always an object (even if empty)

If validation fails, the producer should refuse to emit the bundle.

---

## Versioning & evolution

* `bundle_version` is currently `"v1"`.
* If the schema evolves, it must do so with an explicit version bump (e.g. `"v2"`).
* Producers should not “guess” compatibility. Emit the version you target.

---

## Quick checklist for producers

Before exporting a bundle:

* [ ] All canonical fields are top-level (not in `metadata`)
* [ ] `entities.jsonl` uses `entity_id` and `entity_type`
* [ ] `relationships.jsonl` uses `subject_id`, `predicate`, `object_id`
* [ ] Every row includes `properties: {}`
* [ ] Manifest paths are relative and correct
* [ ] Bundle can be loaded into kgserver without field renaming or inference

---

## Example: Sherlock bundle

See `sherlock_bundle/` in this repo for a minimal working example.

```
$ tree sherlock_bundle/
sherlock_bundle/
├── docs
│   ├── build_orch.md
│   └── mkdocs.yml
├── doc_assets.jsonl
├── entities.jsonl
├── manifest.json
└── relationships.jsonl

2 directories, 6 files
```

where `doc_assets.jsonl` looks like this.

```json
{"path":"docs/mkdocs.yml","content_type":"application/yaml"}
{"path":"docs/build_orch.md","content_type":"text/markdown"}
```

<!--
Maybe let bundles provide a full a docs sub-bundle, instead of doc_assets.jsonl

"docs": {
  "root": "docs/",
  "mkdocs_yml": "docs/mkdocs.yml"
}

Then the presence of a bundle mkdocs config is declared, not discovered. Automatically
grabs whatever files are found in docs/ so you don't need to spell them out.
-->