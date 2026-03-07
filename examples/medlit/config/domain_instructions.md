# Domain Instructions for Medlit Extraction

Domain-specific guidance for biomedical knowledge extraction. Domain experts edit this file. The pipeline injects this content into extraction prompts.

## Entity type classification

Classify at the most specific functional role. If an entity is both a hormone and a protein, classify it as Hormone. Enzymes should be typed Enzyme, not Protein.

- **Protein**: structural or signaling proteins that are NOT better classified as Enzyme, Hormone, Receptor, or Antibody.
- **Hormone**: peptide or steroid hormones (e.g. ACTH, cortisol, catecholamines).
- **Enzyme**: proteins with catalytic function (e.g. aldosterone synthase, kinases).
- **Biomarker**: measurable indicators of biological state (e.g. tumor mutational burden, aneuploidy score, hormone levels). Do NOT use for: pathways (→ Pathway), drugs (→ Drug), cell types (→ AnatomicalStructure), biological processes (→ BiologicalProcess), microbial communities (→ BiologicalProcess), or clinical outcomes like survival (→ StudyDesign).
- **Pathway**: biological pathways (e.g. STING pathway).
- **BiologicalProcess**: biological processes or phenomena (e.g. homologous recombination, genomic instability, chromosomal instability, whole-genome doubling).

### Counterexamples

- ACTH and cortisol are hormones; aldosterone synthase is an enzyme.
- Homologous recombination is BiologicalProcess; PARP inhibitors is Drug; STING pathway is Pathway; disease-free survival is StudyDesign; CD8+ T cells is AnatomicalStructure.

## Predicates

Use the predicate list from the config. Common predicates: TREATS, INCREASES_RISK, INDICATES, ASSOCIATED_WITH, SAME_AS, SUBTYPE_OF, etc.

For SAME_AS relationships, use `"resolution": null` and `"note"` in the output.

## Linguistic trust

For each relationship, classify linguistic trust: **asserted** (direct statement), **suggested** (soft language like "may", "could"), **speculative** (hedged, e.g. "might", "possibly"). Use `linguistic_trust: "asserted" | "suggested" | "speculative"` in the JSON.

## Evidence format

Evidence id format: `{paper_id}:{section}:{paragraph_idx}:llm`

## Output structure

Use "class" for entity type (Disease, Gene, Drug, Hormone, Enzyme, Evidence, etc.). Return ONLY valid JSON, no markdown or commentary.
