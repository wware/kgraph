# Flexible server for knowledge graphs

This repository contains a **domain-neutral knowledge graph server**.

Important architectural note:

- The server does **not** ingest raw documents directly.
- Domain-specific ingestion pipelines produce rich JSON artifacts internally.
- A finalized, validated bundle is exported and loaded by the server as-is at startup.

If you are working on ingestion, storage, or bundle loading, read:
[docs/architecture.md — Producer artifacts vs server bundle](http://localhost:8000/mkdocs/architecture/#producer-side-artifacts-single-source-of-truth)


### Try the GraphiQL playground

![This works on a live KGServer instance, but not on GitHub](GraphiQL_screenshot.png)

[**Link**](http://localhost:8000/graphiql/) - also doesn't work on GitHub


**Why GraphQL is ideal for LLMs + Knowledge Graphs:**

**1. Schema Introspection**
LLMs can query the schema itself to discover what entities, relationships, and properties exist. This means they can adapt to your knowledge graph without being pre-programmed:

```graphql
query {
  __schema {
    types {
      name
      fields { name type }
    }
  }
}
```

**2. Natural Graph Traversal**
GraphQL queries mirror how knowledge graphs are structured - entities with properties and relationships:

```graphql
{
  entity(id: "holmes:character:SherlockHolmes") {
    name
    properties
    relationships {
      type
      targetEntity {
        name
        entityType
      }
    }
  }
}
```

**3. Precise Data Fetching**
LLMs can request exactly what they need - no over-fetching massive JSON blobs. This is crucial for token efficiency:

```graphql
# Just get what you need
{ entity(id: "X") { name entityType } }

# vs REST which might return everything
```

**4. Single Endpoint**
LLMs don't need to learn multiple REST endpoints - just one GraphQL endpoint with a queryable schema. This simplifies the tool definition significantly.

**5. Strongly Typed**
The type system helps LLMs construct valid queries. They can understand field types, required vs optional, and relationships before making requests.

**Standard Resources:**

- [**Official Tutorial**](https://graphql.org/learn/) - comprehensive guide from basics to advanced
- [**How to GraphQL**](https://www.howtographql.com/) - free fullstack tutorial with multiple language tracks

For scientific-journal-based knowledge graphs for instance, GraphQL lets an LLM discover entities like papers, claims, evidence, and then traverse relationships like "which papers support this claim" or "what entities are mentioned in this evidence" - all through introspection and flexible querying.


## How to run this thing

### Local testing (SQLite)

```bash
BUNDLE_PATH=/home/wware/S.zip uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### With Docker Compose (PostgreSQL)

Update docker-compose.yml to add bundle volume and environment variable.
This assumes you've packaged a bundle `S.zip` with the info for your
knowledge graph.

```yaml
api:
    environment:
        DATABASE_URL: postgresql://postgres:postgres@postgres:5432/medlit
        BUNDLE_PATH: /bundle/S.zip
    volumes:
        - /home/wware/S.zip:/bundle/S.zip:ro
```

Then run:
```bash
docker-compose up --build
```

### API endpoints:

- Health: `GET /health`
- Entities: `GET /api/v1/entities?limit=10`
- Relationships: `GET /api/v1/relationships?limit=10`
- GraphQL: `POST /graphql`
- GraphiQL UI: `GET /graphiql`
- API Docs: `GET /docs`

## Bundle format

Let's peek inside a working bundle, from our Sherlock Holmes example
in the `wware/kgraph` repository.

* Contains:

  * graph data (`entities.jsonl`, `relationships.jsonl`)
  * optional documentation assets listed in `doc_assets.jsonl` (markdown files, images, etc.)
  * optional `graphql_examples.yml` to customize the GraphiQL example-query dropdown
* Is treated as **read-only** (single source of truth for that subgraph)
* Bundle is **immutable**, can be validated, cannot be altered

```bash
$ unzip -l ~/S.zip
Archive:  /home/wware/S.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
        0  2026-01-20 15:56   sherlock_bundle/
    11051  2026-01-20 15:56   sherlock_bundle/relationships.jsonl
      499  2026-01-20 15:56   sherlock_bundle/manifest.json
     5239  2026-01-20 15:56   sherlock_bundle/entities.jsonl
     1234  2026-01-20 15:56   sherlock_bundle/doc_assets.jsonl
     5678  2026-01-20 15:56   sherlock_bundle/docs/
---------                     -------
    23660                     6 files
```

### What is intentionally *not* in a bundle

Bundles do not include:
- Raw source documents
- NLP artifacts (NER spans, embeddings, token offsets)
- Alias resolution logic
- Ontology definitions

Those belong to producer pipelines.

### `manifest.json`

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
  "doc_assets": {
    "path": "doc_assets.jsonl",
    "format": "jsonl"
  },
  "metadata": {
    "entity_count": 21,
    "relationship_count": 42,
    "description": "Knowledge graph bundle of Sherlock Holmes stories"
  }
}
```

### `entities.jsonl`
```json
{"entity_id":"holmes:story:AScandalInBohemia","entity_type":"story","name":"A Scandal In Bohemia","status":"canonical","confidence":1.0,"usage_count":1,"created_at":"2026-01-20T20:56:03.338354+00:00","source":"sherlock:curated","properties":{}}
{"entity_id":"holmes:char:SherlockHolmes","entity_type":"character","name":"Mr. Sherlock Holmes","status":"canonical","confidence":0.95,"usage_count":731,"created_at":"2026-01-20T20:56:03.338446+00:00","source":"sherlock:curated","properties":{}}
{"entity_id":"holmes:char:MrsHudson","entity_type":"character","name":"The landlady","status":"canonical","confidence":0.95,"usage_count":4,"created_at":"2026-01-20T20:56:03.338590+00:00","source":"sherlock:curated","properties":{}}
{"entity_id":"holmes:loc:BakerStreet221B","entity_type":"location","name":"Baker Street","status":"canonical","confidence":0.95,"usage_count":27,"created_at":"2026-01-20T20:56:03.338620+00:00","source":"sherlock:curated","properties":{}}
{"entity_id":"holmes:char:IreneAdler","entity_type":"character","name":"Irene Adler","status":"canonical","confidence":1.0,"usage_count":53,"created_at":"2026-01-20T20:56:03.338682+00:00","source":"sherlock:curated","properties":{}}
{"entity_id":"holmes:char:JohnWatson","entity_type":"character","name":"Watson","status":"canonical","confidence":0.8,"usage_count":124,"created_at":"2026-01-20T20:56:03.339025+00:00","source":"sherlock:curated","properties":{}}
...lots more...
```

Each entity looks like this.
```json
{
    "entity_id": "holmes:story:AScandalInBohemia",
    "entity_type": "story",
    "name": "A Scandal In Bohemia",
    "status": "canonical",
    "confidence": 1.0,
    "usage_count": 1,
    "created_at": "2026-01-20T20:56:03.338354+00:00",
    "source": "sherlock:curated",
    "properties": {}
}
```

Notes:

- Fields shown at the top level are part of the **server schema**.
- `properties` is an **opaque JSON object**:
  - Its contents are not interpreted by the server
  - It may vary freely by domain
  - It may be empty

### `relationships.jsonl`
```json
{"subject_id":"holmes:char:SherlockHolmes","object_id":"holmes:story:AScandalInBohemia","predicate":"appears_in","confidence":0.95,"source_documents":["8480d4da-80da-48c8-ada4-e48aff54d2a6"],"created_at":"2026-01-20T20:56:03.339275+00:00","properties":{}}
{"subject_id":"holmes:char:MrsHudson","object_id":"holmes:story:AScandalInBohemia","predicate":"appears_in","confidence":0.95,"source_documents":["8480d4da-80da-48c8-ada4-e48aff54d2a6"],"created_at":"2026-01-20T20:56:03.339297+00:00","properties":{}}
{"subject_id":"holmes:char:IreneAdler","object_id":"holmes:story:AScandalInBohemia","predicate":"appears_in","confidence":0.95,"source_documents":["8480d4da-80da-48c8-ada4-e48aff54d2a6"],"created_at":"2026-01-20T20:56:03.339303+00:00","properties":{}}
{"subject_id":"holmes:char:JohnWatson","object_id":"holmes:story:AScandalInBohemia","predicate":"appears_in","confidence":0.95,"source_documents":["8480d4da-80da-48c8-ada4-e48aff54d2a6"],"created_at":"2026-01-20T20:56:03.339306+00:00","properties":{}}
{"subject_id":"holmes:char:IreneAdler","object_id":"holmes:char:JohnWatson","predicate":"co_occurs_with","confidence":0.7,"source_documents":["8480d4da-80da-48c8-ada4-e48aff54d2a6"],"created_at":"2026-01-20T20:56:03.370337+00:00","properties":{"co_occurrence_count":1}}
{"subject_id":"holmes:char:JohnWatson","object_id":"holmes:char:MrsHudson","predicate":"co_occurs_with","confidence":0.9,"source_documents":["8480d4da-80da-48c8-ada4-e48aff54d2a6"],"created_at":"2026-01-20T20:56:03.370389+00:00","properties":{"co_occurrence_count":3}}
{"subject_id":"holmes:char:IreneAdler","object_id":"holmes:char:SherlockHolmes","predicate":"co_occurs_with","confidence":0.95,"source_documents":["8480d4da-80da-48c8-ada4-e48aff54d2a6"],"created_at":"2026-01-20T20:56:03.370400+00:00","properties":{"co_occurrence_count":5}}
...lots more...
```

Each relationship looks like this.
```json
{
    "subject_id": "holmes:char:SherlockHolmes",
    "object_id": "holmes:story:AScandalInBohemia",
    "predicate": "appears_in",
    "confidence": 0.95,
    "source_documents": [
        "8480d4da-80da-48c8-ada4-e48aff54d2a6"
    ],
    "created_at": "2026-01-20T20:56:03.339275+00:00",
    "properties": {}
}
```

Relationships are directed: `subject_id --predicate--> object_id`.

### `doc_assets.jsonl` (optional)

Bundles may include a `doc_assets.jsonl` file that lists static documentation assets (markdown files, images, configuration files, etc.) to be copied to the server's documentation directory.

Each line is a JSON object describing one asset:

```json
{"path": "docs/build_orch.md", "content_type": "text/markdown"}
{"path": "docs/mkdocs.yml", "content_type": "text/yaml"}
{"path": "docs/images/logo.png", "content_type": "image/png"}
```

When a bundle with `doc_assets.jsonl` is loaded:
- All listed files are copied from `bundle/docs/` to `/app/docs/` preserving directory structure
- If `mkdocs.yml` is present, it's moved to `/app/mkdocs.yml` and MkDocs is built automatically
- This enables domain-specific documentation to be bundled with the knowledge graph

### Bundle contract (important)

The bundle format is a **strict, structured, validated contract** between producer pipelines and the server.

- Bundles **must already be normalized** when exported.
- The server **does not rename fields**, infer structure, or reinterpret metadata.
- If a bundle does not match the declared format, **the server should fail fast at startup**.

Producer pipelines are responsible for:
- Flattening fields to their canonical locations
- Choosing stable identifiers
- Ensuring all required fields are present

The server’s responsibility is limited to validation, loading, and querying.

## Documentation Assets

Bundles can include documentation assets via the `doc_assets.jsonl` file. When a bundle is loaded, these assets are automatically copied to `/app/docs/` and MkDocs is built if `mkdocs.yml` is present. This allows domain-specific documentation to be packaged alongside the knowledge graph data.

## Custom GraphQL Examples

Bundles can include a `graphql_examples.yml` file in the bundle root directory.
When present, it **replaces** the built-in example queries shown in the GraphiQL
interface's dropdown menu. This lets each domain ship its own tailored examples
(e.g. domain-specific entity types, relationship predicates, and filter values).

The file uses the same YAML format as the built-in `query/graphql_examples.yml`:

```yaml
Search Genes: |
  query SearchGenes {
    entities(limit: 5, filter: { entityType: "gene" }) {
      items { entityId name entityType }
      total
    }
  }

Gene-Disease Relationships: |
  query GeneDiseaseRels {
    relationships(limit: 5, filter: { predicate: "associated_with" }) {
      items { subjectId predicate objectId }
      total
    }
  }
```

If the bundle does not contain this file, the server's default examples are used.
