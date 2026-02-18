# Provenance info in the bundle

## Prompt:
2/18/2026, 1:14:19 AM

The bundle contract should include provenance information like mentions and document metadata and locations and all that. And it would be great if this stuff appeared somewhere in the graph visualization. For inspiration, look at kgschema/document.py, kgschema/entity.py (the EntityMention model), and kgschema/relationship.py (BaseRelationship.evidence). This is still very green-field work and backward compatibility is not a concern.



## Response:
2/18/2026, 1:14:41 AM

````plaintext
Thought process: Examined provenance models and architected enriched bundle contract design.

The user is asking about adding provenance information to the bundle contract and graph visualization. They want me to look at the document, entity, and relationship models to understand the current provenance structure and think about how to extend it.

Let me look at the key models they mentioned:

1. From kgschema/document.py - BaseDocument has metadata, source_uri, created_at
2. From kgschema/entity.py - EntityMention has text, start_offset, end_offset, context, metadata
3. From kgschema/relationship.py - BaseRelationship has evidence field

Looking at the current bundle contract in kgbundle/models.py:
- EntityRow has: entity_id, entity_type, name, status, confidence, usage_count, created_at, source, canonical_url, properties
- RelationshipRow has: subject_id, object_id, predicate, confidence, source_documents, created_at, properties

The user wants to add richer provenance that could include:
- Document locations/sections where entities were mentioned
- Original text spans that led to entity extraction  
- Evidence text that supports relationships
- Document metadata

This would be useful for:
1. Traceability - users can see exactly where information came from
2. Graph visualization - showing evidence snippets, document sections
3. Quality assessment - seeing confidence scores, extraction context
4. Debugging - understanding why certain entities/relationships were extracted

I should think about how to extend the bundle contract while keeping it performant for loading/querying.
````

You're absolutely right - provenance is crucial for trustworthiness and debugging. Looking at your current models, you've got good foundation pieces but they're not flowing through to the bundle contract.

**Current provenance pieces:**
- `EntityMention`: Has `start_offset`, `end_offset`, `context`, `metadata`
- `BaseRelationship.evidence`: Currently just `Any` - could be rich evidence objects
- `Provenance` model in domain.py: Has `document_id`, `section`, `start_offset`, `end_offset`

**Bundle contract extensions needed:**

```python
# Enhanced EntityRow
class EntityRow(BaseModel):
    # ... existing fields ...
    mentions: List[EntityMentionProvenance] = []
    first_seen_document: Optional[str] = None
    supporting_documents: List[str] = []

class EntityMentionProvenance(BaseModel):
    document_id: str
    section: Optional[str] = None
    start_offset: int
    end_offset: int
    text_span: str
    context: Optional[str] = None  # Surrounding text
    confidence: float
    extraction_method: str  # "llm", "rule", "lookup"

# Enhanced RelationshipRow  
class RelationshipRow(BaseModel):
    # ... existing fields ...
    evidence_spans: List[EvidenceSpan] = []
    supporting_quotes: List[str] = []

class EvidenceSpan(BaseModel):
    document_id: str
    section: Optional[str] = None
    start_offset: int
    end_offset: int
    text_span: str
    confidence: float
    supports_relation: bool = True  # vs contradicts
```

**Graph visualization integration:**
- **Entity tooltips**: Show first few mentions with document links
- **Edge tooltips**: Show evidence quotes that support the relationship  
- **Entity details panel**: Full mention list with "jump to document" links
- **Relationship details**: Evidence strength, conflicting evidence

**Performance considerations:**
- Keep `mentions` as a separate JSONL file (`entity_mentions.jsonl`) to avoid bloating the main entity rows
- Store only top 3-5 mentions per entity in the main bundle
- Use document fragment IDs for deep linking: `PMC123456#section-results-p3`

Would you want this as a V2 bundle format, or modify the current V1 contract? And should provenance be always-included or opt-in via a `--include-provenance` flag during export?



## Prompt:
2/18/2026, 1:17:17 AM

Modify V1. Always include provenance. Include all mentions, not just 3-5, let KGServer figure out how to limit them. Each mentioned entity gets one line in mentions.jsonl.



## Response:
2/18/2026, 1:17:36 AM

````plaintext
Thought process: Architected enhanced bundle schema integrating comprehensive provenance data.

The user wants to modify the V1 bundle format to always include provenance information. They want:

1. Modify the existing V1 bundle contract (not create V2)
2. Always include provenance (no opt-in flag)
3. Include all mentions, not just top 3-5
4. Let KGServer handle limiting them
5. Each mentioned entity gets one line in mentions.jsonl

Looking at the current bundle models in kgbundle/models.py, I need to:

1. Extend EntityRow to include mention/provenance info
2. Extend RelationshipRow to include evidence spans
3. Add a new mentions.jsonl file to the bundle format
4. Update BundleManifestV1 to reference the new file

The user also mentioned looking at:
- kgschema/document.py (has Provenance model)
- kgschema/entity.py (has EntityMention model) 
- kgschema/relationship.py (BaseRelationship.evidence)

Let me design the enhanced bundle contract:

1. EntityRow gets provenance fields
2. RelationshipRow gets evidence fields  
3. New MentionRow model for mentions.jsonl
4. Update BundleManifestV1 to include mentions file
5. The mentions.jsonl will have one row per entity mention occurrence

This needs to be practical for the ingestion pipeline to populate and the KGServer to consume.
````

Perfect - modify V1, always include, full fidelity. Here's the enhanced bundle contract:

**Enhanced kgbundle/models.py:**

```python
class EntityRow(BaseModel):
    # ... existing fields ...
    # NEW: Provenance fields
    first_seen_document: str  # Document where first extracted
    first_seen_section: Optional[str] = None
    total_mentions: int = 0  # Count of mentions across all docs
    supporting_documents: List[str] = []  # All docs mentioning this entity

class RelationshipRow(BaseModel):  
    # ... existing fields ...
    # NEW: Evidence fields
    evidence_count: int = 0
    strongest_evidence_quote: Optional[str] = None
    evidence_confidence_avg: Optional[float] = None

# NEW: Separate mentions file
class MentionRow(BaseModel):
    """Single entity mention occurrence from a document."""
    entity_id: str  # Links back to EntityRow
    document_id: str
    section: Optional[str] = None
    start_offset: int
    end_offset: int
    text_span: str  # The actual text that was extracted
    context: Optional[str] = None  # ~100 chars around the mention
    confidence: float
    extraction_method: str  # "llm", "rule_based", "canonical_lookup"
    created_at: str

class EvidenceRow(BaseModel):
    """Evidence span supporting a relationship.""" 
    relationship_key: str  # f"{subject_id}:{predicate}:{object_id}"
    document_id: str
    section: Optional[str] = None
    start_offset: int
    end_offset: int
    text_span: str  # The evidence quote
    confidence: float
    supports: bool = True  # True=supports, False=contradicts

class BundleManifestV1(BaseModel):
    # ... existing fields ...
    mentions: Optional[BundleFile] = None  # mentions.jsonl
    evidence: Optional[BundleFile] = None  # evidence.jsonl
```

**Bundle structure:**
```
medlit_bundle/
├── manifest.json
├── entities.jsonl       # EntityRow with provenance summaries  
├── relationships.jsonl  # RelationshipRow with evidence summaries
├── mentions.jsonl       # MentionRow - one per entity extraction
├── evidence.jsonl       # EvidenceRow - one per relationship evidence
└── doc_assets.jsonl
```

**KGServer can then:**
- Show entity details with all mentions
- Display relationship evidence quotes
- Provide "jump to document section" links  
- Build provenance-based trust scores

This gives you full audit trail while keeping the main entity/relationship rows reasonably sized. Sound right?
