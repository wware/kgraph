## 🔍 Deduplication Quality Assessment — Knowledge Graph (404 Entities)

After reviewing all 404 entities across the graph, I identified **multiple classes of deduplication failures** ranging from critical duplicates to structural and semantic issues. Here is a full breakdown:

---

### 🔴 CRITICAL: Confirmed Duplicate Entities (Same Concept, Multiple IDs)

These are clear-cut cases where the **same real-world concept has been ingested more than once** under different entity IDs:

| Concept | Entity 1 ID | Entity 2 ID (or 3rd) | Notes |
|---|---|---|---|
| **metyrapone** | `C0025856` (usageCount=0, canonical) | `6876` (usageCount=2, canonical) | Identical drug, two IDs, different sources |
| **ketoconazole** | `C0022625` (usageCount=0, canonical) | `6135` (usageCount=1, canonical) | Identical drug, C-series vs numeric ID |
| **mifepristone** | `C0026272` (usageCount=0, canonical) | `6585` (usageCount=1, canonical) | Same drug |
| **pasireotide** | `C1871526` (usageCount=2, canonical) | `1234567` (usageCount=1, canonical) | Same drug; `1234567` is a suspicious placeholder ID |
| **osilodrostat** | `C3852966` (usageCount=2, canonical) | `2049836` (usageCount=1, canonical) | Same drug |
| **fludrocortisone** | `C0016314` (usageCount=1, canonical) | `4603` (usageCount=1, canonical) | Same drug |
| **fludrocortisone** | `C0016314` / `4603` | `4316` (usageCount=1, canonical) | **Three-way duplicate!** |
| **hydrocortisone** | `5492` | — | No UMLS CUI; likely should merge with `C0020268` (cortisol) or have a UMLS ID |
| **EGFR** (gene) | `C0034802` (usageCount=0) | `HGNC:3236` (usageCount=1) | Same gene, two authoritative IDs |
| **BRCA1** (gene) | `C0376571` (usageCount=1) | `HGNC:1100` (usageCount=1) | Same gene, UMLS vs HGNC namespaces |
| **TP53** (gene) | `C0079419` (usageCount=1) | `HGNC:11998` (usageCount=5) | Same gene, two IDs |
| **genomic instability** | `C0919532` (usageCount=2) | `C0596988` (usageCount=1) | Two different UMLS CUIs assigned to what appears to be the same concept |
| **glutathione peroxidase 4 / GPX4** | `HGNC:4556` (gene, usageCount=0) | `HGNC:4555` (protein, usageCount=1) | Same molecule, split across gene/protein types with different HGNC IDs |
| **homologous recombination deficiency** | `canon-0b4ee459bd3a` (biomarker) | `canon-31f285d80775` (disease) | Same concept, split into two entity types |

---

### 🟠 HIGH: Semantic Duplicates / Near-Synonyms Stored as Separate Entities

These entities represent the **same clinical concept** expressed with slightly different names or scopes:

| Concept Cluster | Entity IDs | Issue |
|---|---|---|
| **Cushing disease** | `C0221406` ("Cushing disease"), `canon-14e9df5f6c45` ("persistent or recurrent Cushing disease") | The provisional one is a sub-state, not a distinct entity |
| **ACTH-dependent Cushing syndrome** | `canon-4f25744b9951` ("ACTH-dependent Cushing syndrome") vs `canon-2193e082baff` ("ACTH-dependent Cushing's syndrome") | Apostrophe difference — should be merged |
| **Pituitary-induced Cushing / Cushing disease** | `canon-390adc708a33` ("pituitary-induced Cushing") lists synonyms including "Cushing disease" and "corticotrophinoma" — overlaps heavily with `C0221406` | Should be merged into `C0221406` |
| **Endogenous Cushing syndrome** | `canon-1f8884d825b8` lists "Cushing syndrome" as a synonym — may overlap with `C0020626` (hypercortisolism) or the Cushing disease entity | Needs disambiguation |
| **pituitary surgery** | `C0195775` ("pituitary surgery") vs `C0087111` ("transsphenoidal pituitary surgery") | The former could be a parent concept, but with 0 usage may be redundant |
| **severe hypokalemia** | `canon-a0ee87e8dec5` ("severe hypokalaemia") vs `C0020621` ("hypokalemia") | UK vs US spelling variant plus severity qualifier — likely the same concept |
| **diabetes** | `C0011847` ("diabetes") vs `C0011854` ("type 1 diabetes") | Parent-child duplication — "diabetes" alone is too vague |
| **Helicobacter pylori** | `canon-136ba3ca8e29` (type: `other`) | Should be typed as `organism` or `pathogen`, not `other`; no UMLS CUI |
| **cortisol biomarkers** | `C0020268` ("cortisol"), `canon-736331ab2e73` ("midnight cortisol"), `canon-f88ec8b79c32` ("night salivary cortisol"), `canon-1f0e6343b3dd` ("morning cortisol levels"), `canon-8316fb4163d9` ("morning plasma ACTH") | Measurement variants of cortisol should be attributes/sub-types, not separate entities |
| **HRD score** | `canon-c3020b2ac938` ("HRD score") and `canon-0b4ee459bd3a` ("homologous recombination deficiency" as biomarker) | Significant overlap |
| **mutational burden** | `canon-72697c0dca11` ("tumor mutational burden") vs `canon-0276570bf1ea` ("mutational burden") | Should be merged |

---

### 🟡 MEDIUM: Structural / Taxonomic Issues

These are not strict duplicates but represent **mis-classifications or structural inconsistencies** that damage graph integrity:

| Issue | Examples |
|---|---|
| **Mixed entity types for the same concept** | `GPX4` appears as gene (`HGNC:4556`) and protein (`HGNC:4555`). `catecholamines` typed as `protein`. `Pseudomonas aeruginosa` and `Escherichia coli` typed as `anatomicalstructure` — should be `organism`. |
| **Numeric placeholder IDs** | `5492`, `4603`, `4316`, `6876`, `6135`, `6585`, `1234567`, `2049836`, etc. have no recognized namespace prefix — these are likely numeric PubChem CIDs or internal IDs used inconsistently alongside UMLS CUIs |
| **Provisional entities with no source docs** | Many `canon-*` IDs have `total_mentions: 0` and `first_seen_document: null` — these are orphaned entities with no evidentiary basis |
| **`PMC_UNKNOWN`, `PMC_extracted`, `PMC_PLACEHOLDER`** | These are placeholder document IDs appearing in `supporting_documents` — provenance is broken for a significant portion of entities |
| **Type misclassification: drugs as procedures** | `canon-b6653513916a` ("combination therapy") is typed as `procedure` rather than a treatment concept |
| **Class too broad to be useful** | `C0243192` ("agonist") and `C0243076` ("antagonist") are typed as `drug` — these are pharmacological roles, not drugs |
| **`overall survival` as a biomarker** | `C4086681` typed as `biomarker` — it's a clinical outcome/endpoint |
| **`Crooke's cells` as `anatomicalstructure`** | Should be `celltype` or `histologicalfeature` |

---

### 🟢 WHAT IS WORKING WELL

- **UMLS CUI-based entities are generally well-formed** — the majority of canonical entities with `C-prefixed` IDs have correct names, appropriate synonyms, and clear `canonicalUrl` pointers.
- **HGNC gene IDs are consistently used** for genes from certain papers (e.g., `HGNC:11998` for TP53), showing good sourcing discipline.
- **Synonym lists are present and useful** for most canonical entities (e.g., `ACTH`, `APS-1`, `TSS`, `ICB`).
- **Entity type diversity is rich** — covering diseases, genes, proteins, drugs, biomarkers, procedures, pathways, biological processes, and more.

---

### 📊 Summary Scorecard

| Dimension | Score | Notes |
|---|---|---|
| **True duplicates (same concept, 2+ IDs)** | ⚠️ Poor | ~14+ confirmed duplicates found |
| **Semantic near-duplicates** | ⚠️ Poor | ~10+ synonym/sub-type clusters that should be merged |
| **Entity type consistency** | 🟡 Fair | Several misclassified entities (organisms as anatomy, outcomes as biomarkers) |
| **ID namespace consistency** | 🟡 Fair | Mix of UMLS CUIs, HGNC IDs, bare numerics, and `canon-*` hashes |
| **Provenance / source docs** | 🔴 Poor | Heavy use of `PMC_UNKNOWN`, `PMC_extracted`, `PMC_PLACEHOLDER` — provenance is unreliable for many entities |
| **Canonical vs. provisional status** | 🟡 Fair | Provisional entities (confidence 0.5) exist with zero usage — should be pruned or resolved |
| **Synonym management** | 🟢 Good | Most canonical entities have useful synonym lists |

---

### 🛠️ Recommended Actions

1. **Merge the confirmed drug duplicates** — especially `metyrapone`, `ketoconazole`, `mifepristone`, `fludrocortisone` (3-way!), `pasireotide`, and `osilodrostat`. Retain the UMLS CUI as the canonical ID and redirect the numeric IDs.
2. **Merge the Cushing syndrome concept cluster** — collapse `canon-4f25744b9951`, `canon-2193e082baff`, and `canon-390adc708a33` into a single canonical entity, likely extending `C0221406`.
3. **Resolve cross-namespace gene duplicates** — decide whether UMLS or HGNC is the preferred namespace for genes and unify `BRCA1`, `EGFR`, `TP53`, `GPX4`.
4. **Fix entity type errors** — reclassify organisms, outcomes, and pharmacological roles.
5. **Repair provenance** — entities with `PMC_UNKNOWN`/`PMC_PLACEHOLDER` source IDs should be traced back to real PMC IDs or flagged for removal.
6. **Prune zero-usage provisional orphans** — entities with `usageCount: 0`, `total_mentions: 0`, and no `first_seen_document` add noise with no signal.
