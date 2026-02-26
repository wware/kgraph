# Relationship tracing works now

The machinery for tracing relationship ingestion (and the decision to
keep or discard a relationship) is now working.

```bash
# Three-pass pipeline (ingest.py removed). Pass 1 extracts entities and relationships per paper.
$ cd /home/wware/kgraph && rm -f /tmp/kgraph-relationship-traces/*.json && uv run python -m examples.medlit.scripts.pass1_extract --input-dir examples/medlit/pmc_xmls --output-dir pass1_bundles --llm-backend ollama --papers "PMC10759991.xml" 2>&1 | tee /tmp/ingest_output.txt
```

Let me check the trace file. Read /tmp/kgraph-relationship-traces/PMC10759991.relationships.trace.json

The trace is working. Here's what it captured. LLM proposed 6 relationships:

  1. gene-expression profiling → treats → breast cancer (REJECTED by domain validation - procedures don't treat)
  2. luminal A → associated_with → breast cancer ✓
  3. luminal B → associated_with → breast cancer ✓
  4. HER2-enriched → associated_with → breast cancer ✓
  5. basal-like → associated_with → breast cancer ✓
  6. breast cancer → associated_with → gene-expression profiling ✓

5 were accepted (1 rejected by domain validation). The trace shows for each:

  • matched_subject: true
  • matched_object: true
  • semantic_ok: true
  • swap_applied: false
  • drop_reason: null

Key insight: The first relationship (gene-expression profiling treats breast cancer) was accepted by the extractor (which allows procedure→disease for "treats" per the prompt) but rejected by domain validation (which only allows drug→disease). The domain caught this, but the prompt could be tightened to prevent the LLM from proposing it in the first place.

## TODO: Prompt Improvements

- [ ] Update prompt to clarify that "treats" requires a **drug**, not any procedure
- [ ] Consider adding "diagnosed_by" or "analyzed_by" predicate for diagnostic procedures
- [ ] Ensure prompt predicate rules match domain constraints exactly

The trace infrastructure is now fully working. You can inspect:

  • raw_llm_output - exact LLM response
  • parsed_json - what was parsed
  • decisions - per-relationship verdict with reasons
  • final_relationships - what was accepted

## ingest_output.txt

```
  Initializing Ollama LLM: llama3.1:8b at http://localhost:11434 (timeout: 1200.0s)...
  ✓ Ollama client created successfully
  Using LLM-based entity extraction...
  ✓ LLM-based extractor created

  Progress: 1/1 (100.0%)
    Elapsed: 3.0 min
    Rate: 0.01 papers/sec
2026-02-04 17:20:05,550 - /home/wware/kgraph/kgraph/ingest.py:657 - INFO - 'Starting entity promotion process'
2026-02-04 17:20:05,550 - /home/wware/kgraph/kgraph/ingest.py:662 - INFO - {'config': PromotionConfig(min_usage_count=1, min_confidence=0.4, require_embedding=False),
 'message': 'Promotion configuration'}
2026-02-04 17:20:05,550 - /home/wware/kgraph/kgraph/ingest.py:673 - INFO - 'Found 6 provisional entities meeting basic thresholds (usage>=1, confidence>=0.4)'
2026-02-04 17:20:05,558 - /home/wware/kgraph/kgraph/ingest.py:693 - INFO - 'Looking up canonical IDs for 6 candidate entities...'
2026-02-04 17:20:10,871 - /home/wware/kgraph/kgraph/ingest.py:718 - INFO - 'Promotion complete: 2 promoted, 0 skipped (policy), 4 skipped (no canonical ID), 0 skipped (storage failure)'
Invalid subject type for predicate 'treats' in domain 'medlit': Got 'procedure', expected one of {'drug'}. Relationship: subject_id='MeSH:D020869' predicate='treats' object_id='MeSH:D001943' confidence=0.9 source_documents=('PMC10759991',) evidence=Evidence(kind='llm_extracted', source_documents=('PMC10759991',), primary=Provenance(document_id='PMC10759991', source_uri='examples/medlit/pmc_xmls/PMC10759991.xml', section='abstract', paragraph=None, start_offset=None, end_offset=None), mentions=(Provenance(document_id='PMC10759991', source_uri='examples/medlit/pmc_xmls/PMC10759991.xml', section='abstract', paragraph=None, start_offset=None, end_offset=None),), notes={'evidence_text': 'Gene-expression profiling has considerably impacted our understanding of breast cancer biology.', 'extraction_method': 'llm'}) created_at=datetime.datetime(2026, 2, 4, 22, 26, 10, 293705, tzinfo=datetime.timezone.utc) last_updated=None metadata={'extraction_method': 'llm'}

  Progress: 1/1 (100.0%)
    Elapsed: 6.0 min
    Rate: 0.00 papers/sec
  Extracting entities with LLM from 1254 chars...
  LLM returned 6 entities
  Extracting relationships with LLM from 6 entities...
  Wrote relationship trace: /tmp/kgraph-relationship-traces/PMC10759991.relationships.trace.json
  LLM extracted 6 relationships
{
  "pipeline_version": "1.0.0",
  "started_at": "2026-02-04T22:17:04.529702Z",
  "completed_at": "2026-02-04T22:26:10.294714Z",
  "stopped_at_stage": "relationships",
  "entity_extraction": {
    "stage": "entities",
    "completed_at": "2026-02-04T22:20:05.506040Z",
    "papers_processed": 1,
    "papers_failed": 0,
    "total_entities_extracted": 6,
    "total_entities_new": 6,
    "total_entities_existing": 0,
    "paper_results": [
      {
        "document_id": "PMC10759991",
        "source_uri": "examples/medlit/pmc_xmls/PMC10759991.xml",
        "extracted_at": "2026-02-04T22:20:05.505977Z",
        "entities_extracted": 6,
        "entities_new": 6,
        "entities_existing": 0,
        "entities": [],
        "errors": []
      }
    ],
    "entity_type_counts": {
      "disease": 5,
      "procedure": 1
    },
    "provisional_count": 6,
    "canonical_count": 0
  },
  "promotion": {
    "stage": "promotion",
    "completed_at": "2026-02-04T22:20:10.880996Z",
    "candidates_evaluated": 6,
    "entities_promoted": 2,
    "entities_skipped_no_canonical_id": 4,
    "entities_skipped_policy": 0,
    "entities_skipped_storage_failure": 0,
    "promoted_entities": [
      {
        "old_entity_id": "MeSH:D001943",
        "new_entity_id": "MeSH:D001943",
        "name": "breast cancer",
        "entity_type": "disease",
        "canonical_source": "mesh",
        "canonical_url": "https://meshb.nlm.nih.gov/record/ui?ui=D001943"
      },
      {
        "old_entity_id": "MeSH:D020869",
        "new_entity_id": "MeSH:D020869",
        "name": "gene-expression profiling",
        "entity_type": "procedure",
        "canonical_source": "mesh",
        "canonical_url": "https://meshb.nlm.nih.gov/record/ui?ui=D020869"
      }
    ],
    "total_canonical_entities": 2,
    "total_provisional_entities": 4
  },
  "relationship_extraction": {
    "stage": "relationships",
    "completed_at": "2026-02-04T22:26:10.294701Z",
    "papers_processed": 1,
    "papers_with_relationships": 1,
    "total_relationships_extracted": 5,
    "paper_results": [
      {
        "document_id": "PMC10759991",
        "source_uri": "examples/medlit/pmc_xmls/PMC10759991.xml",
        "extracted_at": "2026-02-04T22:26:10.294682Z",
        "relationships_extracted": 5,
        "relationships": [],
        "errors": []
      }
    ],
    "predicate_counts": {
      "associated_with": 5
    }
  },
  "total_documents": 1,
  "total_entities": 6,
  "total_relationships": 5
}
```

## PMC10759991.relationships.trace.json

```json
{
  "document_id": "PMC10759991",
  "source_uri": "examples/medlit/pmc_xmls/PMC10759991.xml",
  "llm_model": "llama3.1:8b",
  "created_at": "2026-02-04T22:20:10.881087+00:00",
  "entity_count": 6,
  "prompt": "Extract medical relationships from the text below.\n\nPREDICATE TYPE RULES (check entity types carefully):\n• treats: ONLY drug/procedure → disease/symptom (drug treats disease, NOT disease treats disease)\n• increases_risk: gene/ethnicity → disease (gene increases risk of disease)\n• prevents: drug/procedure → disease\n• targets: ONLY drug/procedure → gene/protein (drug targets protein, NOT gene targets disease)\n• diagnosed_by: disease → procedure/biomarker (disease diagnosed by test, NOT test diagnosed by disease)\n• associated_with: use for general correlations, disease subtypes, gene-disease links, or when other predicates don't fit\n\nEntities in document:\n- luminal A (disease): prov:4703897671b9452c86d007d5309b94a3\n- luminal B (disease): prov:299e32f05a8f4cf1927d109b07ce838c\n- HER2-enriched (disease): prov:16880b0936724524a88ad0d9f14a3cd5\n- basal-like (disease): prov:e9c8eb12983044db9d0662927bbe87ee\n- breast cancer (disease): MeSH:D001943\n- gene-expression profiling (procedure): MeSH:D020869\n\nText:\nBreast cancer is heterogeneous and differs substantially across different tumors (intertumor heterogeneity) and even within an individual tumor (intratumor heterogeneity). Gene-expression profiling has considerably impacted our understanding of breast cancer biology. Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression, showing significant prognostic and predictive value in multiple clinical scenarios. Thanks to the molecular profiling of breast tumors, breast cancer is a paradigm of treatment personalization. Several standardized prognostic gene-expression assays are presently being used in the clinic to guide treatment decisions. Moreover, the development of single-cell-level resolution molecular profiling has allowed us to appreciate that breast cancer is also heterogeneous within a single tumor. There is an evident functional heterogeneity within the neoplastic and tumor microenvironment cells. Finally, emerging insights from these studies suggest a substantial cellular organization of neoplastic and tumor microenvironment cells, thus defining breast cancer ecosystems and highlighting the importance of spatial localizations.\n\nReturn JSON array of relationships (extract ALL valid relationships you find):\n[\n  {\"subject\": \"entity_name\", \"predicate\": \"treats\", \"object\": \"entity_name\", \"confidence\": 0.9, \"evidence\": \"quote from text\"},\n  ...\n]\n\nValid examples:\n{\"subject\": \"paclitaxel\", \"predicate\": \"treats\", \"object\": \"breast cancer\", \"confidence\": 0.9, \"evidence\": \"Paclitaxel therapy for breast cancer\"}\n{\"subject\": \"trastuzumab\", \"predicate\": \"targets\", \"object\": \"HER2\", \"confidence\": 0.95, \"evidence\": \"Trastuzumab targets HER2 protein\"}\n{\"subject\": \"BRCA1\", \"predicate\": \"increases_risk\", \"object\": \"breast cancer\", \"confidence\": 0.9, \"evidence\": \"BRCA1 mutations increase breast cancer risk\"}\n{\"subject\": \"HER2\", \"predicate\": \"associated_with\", \"object\": \"breast cancer\", \"confidence\": 0.85, \"evidence\": \"HER2 overexpression in breast cancer\"}\n{\"subject\": \"luminal A\", \"predicate\": \"associated_with\", \"object\": \"breast cancer\", \"confidence\": 0.85, \"evidence\": \"Luminal A subtype of breast cancer\"}\n\nINVALID (skip these):\n- {\"subject\": \"breast cancer\", \"predicate\": \"treats\", ...} - disease cannot treat\n- {\"subject\": \"HER2\", \"predicate\": \"targets\", \"object\": \"breast cancer\", ...} - gene cannot target disease, use \"associated_with\"\n- {\"subject\": \"therapy\", \"predicate\": \"increases_risk\", ...} - therapy doesn't increase risk\n\nReturn ONLY the JSON array.",
  "raw_llm_output": "[\n  {\n    \"subject\": \"gene-expression profiling\",\n    \"predicate\": \"treats\",\n    \"object\": \"breast cancer\",\n    \"confidence\": 0.9,\n    \"evidence\": \"Gene-expression profiling has considerably impacted our understanding of breast cancer biology.\"\n  },\n  {\n    \"subject\": \"luminal A\",\n    \"predicate\": \"associated_with\",\n    \"object\": \"breast cancer\",\n    \"confidence\": 0.85,\n    \"evidence\": \"Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,\"\n  },\n  {\n    \"subject\": \"luminal B\",\n    \"predicate\": \"associated_with\",\n    \"object\": \"breast cancer\",\n    \"confidence\": 0.85,\n    \"evidence\": \"Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,\"\n  },\n  {\n    \"subject\": \"HER2-enriched\",\n    \"predicate\": \"associated_with\",\n    \"object\": \"breast cancer\",\n    \"confidence\": 0.85,\n    \"evidence\": \"Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,\"\n  },\n  {\n    \"subject\": \"basal-like\",\n    \"predicate\": \"associated_with\",\n    \"object\": \"breast cancer\",\n    \"confidence\": 0.85,\n    \"evidence\": \"Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,\"\n  },\n  {\n    \"subject\": \"breast cancer\",\n    \"predicate\": \"associated_with\",\n    \"object\": \"gene-expression profiling\",\n    \"confidence\": 0.85,\n    \"evidence\": \"Gene-expression profiling has considerably impacted our understanding of breast cancer biology.\"\n  }\n]",
  "parsed_json": [
    {
      "subject": "gene-expression profiling",
      "predicate": "treats",
      "object": "breast cancer",
      "confidence": 0.9,
      "evidence": "Gene-expression profiling has considerably impacted our understanding of breast cancer biology."
    },
    {
      "subject": "luminal A",
      "predicate": "associated_with",
      "object": "breast cancer",
      "confidence": 0.85,
      "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
    },
    {
      "subject": "luminal B",
      "predicate": "associated_with",
      "object": "breast cancer",
      "confidence": 0.85,
      "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
    },
    {
      "subject": "HER2-enriched",
      "predicate": "associated_with",
      "object": "breast cancer",
      "confidence": 0.85,
      "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
    },
    {
      "subject": "basal-like",
      "predicate": "associated_with",
      "object": "breast cancer",
      "confidence": 0.85,
      "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
    },
    {
      "subject": "breast cancer",
      "predicate": "associated_with",
      "object": "gene-expression profiling",
      "confidence": 0.85,
      "evidence": "Gene-expression profiling has considerably impacted our understanding of breast cancer biology."
    }
  ],
  "decisions": [
    {
      "item": {
        "subject": "gene-expression profiling",
        "predicate": "treats",
        "object": "breast cancer",
        "confidence": 0.9,
        "evidence": "Gene-expression profiling has considerably impacted our understanding of breast cancer biology."
      },
      "subject_name": "gene-expression profiling",
      "object_name": "breast cancer",
      "predicate": "treats",
      "confidence": 0.9,
      "matched_subject": true,
      "matched_object": true,
      "semantic_ok": true,
      "swap_applied": false,
      "accepted": true,
      "drop_reason": null,
      "resolved": {
        "subject_id": "MeSH:D020869",
        "subject_type": "procedure",
        "object_id": "MeSH:D001943",
        "object_type": "disease"
      }
    },
    {
      "item": {
        "subject": "luminal A",
        "predicate": "associated_with",
        "object": "breast cancer",
        "confidence": 0.85,
        "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
      },
      "subject_name": "luminal A",
      "object_name": "breast cancer",
      "predicate": "associated_with",
      "confidence": 0.85,
      "matched_subject": true,
      "matched_object": true,
      "semantic_ok": true,
      "swap_applied": false,
      "accepted": true,
      "drop_reason": null,
      "resolved": {
        "subject_id": "prov:4703897671b9452c86d007d5309b94a3",
        "subject_type": "disease",
        "object_id": "MeSH:D001943",
        "object_type": "disease"
      }
    },
    {
      "item": {
        "subject": "luminal B",
        "predicate": "associated_with",
        "object": "breast cancer",
        "confidence": 0.85,
        "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
      },
      "subject_name": "luminal B",
      "object_name": "breast cancer",
      "predicate": "associated_with",
      "confidence": 0.85,
      "matched_subject": true,
      "matched_object": true,
      "semantic_ok": true,
      "swap_applied": false,
      "accepted": true,
      "drop_reason": null,
      "resolved": {
        "subject_id": "prov:299e32f05a8f4cf1927d109b07ce838c",
        "subject_type": "disease",
        "object_id": "MeSH:D001943",
        "object_type": "disease"
      }
    },
    {
      "item": {
        "subject": "HER2-enriched",
        "predicate": "associated_with",
        "object": "breast cancer",
        "confidence": 0.85,
        "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
      },
      "subject_name": "HER2-enriched",
      "object_name": "breast cancer",
      "predicate": "associated_with",
      "confidence": 0.85,
      "matched_subject": true,
      "matched_object": true,
      "semantic_ok": true,
      "swap_applied": false,
      "accepted": true,
      "drop_reason": null,
      "resolved": {
        "subject_id": "prov:16880b0936724524a88ad0d9f14a3cd5",
        "subject_type": "disease",
        "object_id": "MeSH:D001943",
        "object_type": "disease"
      }
    },
    {
      "item": {
        "subject": "basal-like",
        "predicate": "associated_with",
        "object": "breast cancer",
        "confidence": 0.85,
        "evidence": "Four main “intrinsic subtypes” of breast cancer (i.e., luminal A, luminal B, HER2-enriched, and basal-like) have been consistently identified by gene expression,"
      },
      "subject_name": "basal-like",
      "object_name": "breast cancer",
      "predicate": "associated_with",
      "confidence": 0.85,
      "matched_subject": true,
      "matched_object": true,
      "semantic_ok": true,
      "swap_applied": false,
      "accepted": true,
      "drop_reason": null,
      "resolved": {
        "subject_id": "prov:e9c8eb12983044db9d0662927bbe87ee",
        "subject_type": "disease",
        "object_id": "MeSH:D001943",
        "object_type": "disease"
      }
    },
    {
      "item": {
        "subject": "breast cancer",
        "predicate": "associated_with",
        "object": "gene-expression profiling",
        "confidence": 0.85,
        "evidence": "Gene-expression profiling has considerably impacted our understanding of breast cancer biology."
      },
      "subject_name": "breast cancer",
      "object_name": "gene-expression profiling",
      "predicate": "associated_with",
      "confidence": 0.85,
      "matched_subject": true,
      "matched_object": true,
      "semantic_ok": true,
      "swap_applied": false,
      "accepted": true,
      "drop_reason": null,
      "resolved": {
        "subject_id": "MeSH:D001943",
        "subject_type": "disease",
        "object_id": "MeSH:D020869",
        "object_type": "procedure"
      }
    }
  ],
  "final_relationships": [
    {
      "subject_id": "MeSH:D020869",
      "predicate": "treats",
      "object_id": "MeSH:D001943",
      "confidence": 0.9
    },
    {
      "subject_id": "prov:4703897671b9452c86d007d5309b94a3",
      "predicate": "associated_with",
      "object_id": "MeSH:D001943",
      "confidence": 0.85
    },
    {
      "subject_id": "prov:299e32f05a8f4cf1927d109b07ce838c",
      "predicate": "associated_with",
      "object_id": "MeSH:D001943",
      "confidence": 0.85
    },
    {
      "subject_id": "prov:16880b0936724524a88ad0d9f14a3cd5",
      "predicate": "associated_with",
      "object_id": "MeSH:D001943",
      "confidence": 0.85
    },
    {
      "subject_id": "prov:e9c8eb12983044db9d0662927bbe87ee",
      "predicate": "associated_with",
      "object_id": "MeSH:D001943",
      "confidence": 0.85
    },
    {
      "subject_id": "MeSH:D001943",
      "predicate": "associated_with",
      "object_id": "MeSH:D020869",
      "confidence": 0.85
    }
  ],
  "error": null
}
```
