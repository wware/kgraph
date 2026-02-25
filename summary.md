
============================================================
# examples/__init__.py
============================================================


============================================================
# examples/medlit/__init__.py
============================================================


============================================================
# examples/medlit/bundle_models.py
============================================================
# imports: examples.medlit_schema.base.ExtractionProvenance, pydantic.BaseModel, ConfigDict, Field, field_validator, typing.Any, Literal, Optional
class PaperInfo(BaseModel):  # "Paper metadata in the per-paper bundle."
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    title: str = ''
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    study_type: Optional[str] = None
    eco_type: Optional[str] = None
    @field_validator('year', mode='before')
    @classmethod
    def coerce_year(cls, v)
class ExtractedEntityRow(BaseModel):  # "Minimal entity record in the bundle. JSON key "class" via alias."
    id: str
    entity_class: str = Field(alias='class', description='Entity type, e.g. Disease, Gene, Drug')
    name: str
    synonyms: list[str] = Field(default_factory=list)
    symbol: Optional[str] = None
    brand_names: list[str] = Field(default_factory=list)
    source: Literal['extracted', 'umls', 'hgnc', 'rxnorm', 'loinc', 'uniprot'] = 'extracted'
    canonical_id: Optional[str] = None
    umls_id: Optional[str] = None
    hgnc_id: Optional[str] = None
    rxnorm_id: Optional[str] = None
    loinc_code: Optional[str] = None
    uniprot_id: Optional[str] = None
class EvidenceEntityRow(BaseModel):  # "Evidence entity in the bundle. id format: {paper_id}:{section}:{paragraph_idx}:{method}."
    id: str
    entity_class: Literal['Evidence'] = Field(default='Evidence', alias='class')
    entity_id: Optional[str] = None
    paper_id: str
    text_span_id: Optional[str] = None
    text: Optional[str] = None
    confidence: float = 0.5
    extraction_method: str = 'llm'
    study_type: Optional[str] = None
    eco_type: Optional[str] = None
    source: Literal['extracted'] = 'extracted'
class RelationshipRow(BaseModel):  # "One relationship in the bundle. evidence_ids optional for SAME_AS."
    subject: str
    predicate: str
    object_id: str = Field(alias='object', description='Object entity ID (bundle-local or canonical)')
    evidence_ids: list[str] = Field(default_factory=list)
    source_papers: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    properties: dict[str, Any] = Field(default_factory=dict)
    section: Optional[str] = None
    asserted_by: str = 'llm'
    resolution: Optional[Literal['merged', 'distinct']] = None
    note: Optional[str] = None
class PerPaperBundle(BaseModel):  # "Per-paper bundle: Pass 1 output and Pass 2 input. Immutable after Pass 1."
    paper: PaperInfo
    extraction_provenance: Optional[ExtractionProvenance] = None
    entities: list[ExtractedEntityRow] = Field(default_factory=list)
    evidence_entities: list[EvidenceEntityRow] = Field(default_factory=list)
    relationships: list[RelationshipRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    def to_bundle_dict(self) -> dict  # "Serialize for JSON with alias 'class' used for entity type."
    @classmethod
    def from_bundle_dict(cls, data: dict) -> 'PerPaperBundle'  # "Load from dict/JSON (accepts key 'class' for entity type)."

============================================================
# examples/medlit/documents.py
============================================================
# imports: kgschema.document.BaseDocument, pydantic.Field
class JournalArticle(BaseDocument):  # "A journal article (research paper) as a source document for extraction."
    authors: tuple[str, ...] = Field(default=(), description='List of author names in citation order')
    abstract: str = Field(description='Complete abstract text')
    publication_date: str | None = Field(default=None, description='Publication date in ISO format (YYYY-MM-DD)')
    journal: str | None = Field(default=None, description='Journal name')
    doi: str | None = Field(default=None, description='Digital Object Identifier')
    pmid: str | None = Field(default=None, description='PubMed ID')
    def get_document_type(self) -> str  # "Return domain-specific document type."
    def get_sections(self) -> list[tuple[str, str]]  # "Return document sections as (section_name, content) tuples."
    @property
    def study_type(self) -> str | None  # "Convenience property for accessing study_type from metadata."
    @property
    def sample_size(self) -> int | None  # "Convenience property for accessing sample_size from metadata."
    @property
    def mesh_terms(self) -> list[str]  # "Convenience property for accessing mesh_terms from metadata."

============================================================
# examples/medlit/domain.py
============================================================
# imports: .documents.JournalArticle, .entities.BiomarkerEntity, DiseaseEntity, DrugEntity, EthnicityEntity, GeneEntity, LocationEntity, PathwayEntity, ProcedureEntity, ProteinEntity, SymptomEntity, .pipeline.authority_lookup.CanonicalIdLookup, .promotion.MedLitPromotionPolicy, .relationships.MedicalClaimRelationship, .vocab.ALL_PREDICATES, get_valid_predicates, kgraph.promotion.PromotionPolicy, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, PredicateConstraint, ValidationIssue, kgschema.entity.BaseEntity, PromotionConfig, kgschema.relationship.BaseRelationship, kgschema.storage.EntityStorageInterface
class MedLitDomainSchema(DomainSchema):  # "Domain schema for medical literature extraction."
    _predicate_constraints: dict[str, PredicateConstraint] | None = None
    @property
    def name(self) -> str
    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]
    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]
    @property
    def predicate_constraints(self) -> dict[str, PredicateConstraint]
    @property
    def document_types(self) -> dict[str, type[BaseDocument]]
    @property
    def promotion_config(self) -> PromotionConfig  # "Medical domain promotion configuration."
    def validate_entity(self, entity: BaseEntity) -> list[ValidationIssue]  # "Validate an entity against medical domain rules."
    async def validate_relationship(self, relationship: BaseRelationship, entity_storage: EntityStorageInterface | None=None) -> bool  # "Validate a relationship against medical domain rules."
    def get_valid_predicates(self, subject_type: str, object_type: str) -> list[str]  # "Return predicates valid between two entity types."
    def get_promotion_policy(self, lookup: CanonicalIdLookup | None=None) -> PromotionPolicy  # "Return the promotion policy for medical literature domain."

============================================================
# examples/medlit/entities.py
============================================================
# imports: kgschema.entity.BaseEntity
class DiseaseEntity(BaseEntity):  # "Represents medical conditions, disorders, and syndromes."
    def get_entity_type(self) -> str
class GeneEntity(BaseEntity):  # "Represents genes and their genomic information."
    def get_entity_type(self) -> str
class DrugEntity(BaseEntity):  # "Represents medications and therapeutic substances."
    def get_entity_type(self) -> str
class ProteinEntity(BaseEntity):  # "Represents proteins and their biological functions."
    def get_entity_type(self) -> str
class SymptomEntity(BaseEntity):  # "Represents clinical signs and symptoms."
    def get_entity_type(self) -> str
class ProcedureEntity(BaseEntity):  # "Represents medical tests, diagnostics, treatments."
    def get_entity_type(self) -> str
class BiomarkerEntity(BaseEntity):  # "Represents measurable indicators."
    def get_entity_type(self) -> str
class PathwayEntity(BaseEntity):  # "Represents biological pathways."
    def get_entity_type(self) -> str
class LocationEntity(BaseEntity):  # "Represents geographic locations relevant to epidemiological analysis."
    def get_entity_type(self) -> str
class EthnicityEntity(BaseEntity):  # "Represents ethnic or population groups for epidemiological analysis."
    def get_entity_type(self) -> str

============================================================
# examples/medlit/pipeline/__init__.py
============================================================
# imports: .authority_lookup.CanonicalIdLookup, .embeddings.OllamaMedLitEmbeddingGenerator, .llm_client.LLMClientInterface, OllamaLLMClient, .mentions.MedLitEntityExtractor, .ner_extractor.MedLitNEREntityExtractor, .parser.JournalArticleParser, .relationships.MedLitRelationshipExtractor, .resolve.MedLitEntityResolver
__all__ = ['CanonicalIdLookup', 'JournalArticleParser', 'MedLitEntityExtractor', 'MedLitEntityResolver', 'MedLitNEREntityExtractor', 'MedLitRelationshipExtractor', 'OllamaMedLitEmbeddingGenerator', 'LLMClientInterface', 'OllamaLLMClient']

============================================================
# examples/medlit/pipeline/authority_lookup.py
============================================================
# imports: .canonical_urls.build_canonical_url, httpx, kgraph.canonical_id.CanonicalId, CanonicalIdLookupInterface, JsonFileCanonicalIdCache, kgraph.logging.setup_logging, kgraph.storage.memory._cosine_similarity, os, pathlib.Path, re, typing.Any, Optional
LOOKUP_BLOCKLIST = frozenset({'gene', 'disease', 'drug', 'protein', 'symptom', 'procedure', 'biomarker', 'pathway', 'location', 'ethnicity', 'variant', 'diseases', 'genes', 'drugs', 'proteins', 'symptoms', 'procedures', 'biomarkers', 'pathways', 'variants', 'pathogenic variant', 'loss-of-function variant', 'genetic variant', 'germline variant', 'somatic variant', 'sequence variant', 'gene variant'})
logger = setup_logging()
class CanonicalIdLookup(CanonicalIdLookupInterface):  # "Look up canonical IDs from various medical ontology authorities."
    def __init__(self, umls_api_key: Optional[str]=None, cache_file: Optional[Path]=None, embedding_generator: Any=None, similarity_threshold: float=0.5)  # "Initialize the canonical ID lookup service."
    def _save_cache(self, force: bool=False) -> None  # "Save cache to disk."
    async def lookup(self, term: str, entity_type: str) -> Optional[CanonicalId]  # "Look up canonical ID for a medical term (interface method)."
    async def lookup_canonical_id(self, term: str, entity_type: str) -> Optional[str]  # "Look up canonical ID for a medical term."
    async def _rerank_by_similarity(self, term: str, candidates: list[tuple[str, str]]) -> Optional[str]  # "Pick the candidate whose label is most similar to the search term."
    async def _lookup_umls(self, term: str) -> Optional[str]  # "Look up UMLS CUI for a disease/symptom term."
    def _normalize_mesh_search_terms(self, term: str) -> list[str]  # "Generate normalized search terms for MeSH lookup."
    async def _lookup_mesh(self, term: str) -> Optional[str]  # "Look up MeSH descriptor ID for a disease/symptom term."
    def _extract_mesh_id_from_results(self, data: list, search_terms: str | list[str]) -> Optional[str]  # "Extract MeSH descriptor ID from API results, preferring best matches."
    async def _try_mesh_descriptor_lookup_all(self, term: str) -> list[dict]  # "Try to find MeSH descriptors for a term, returning all results."
    async def _lookup_hgnc(self, term: str) -> Optional[str]  # "Look up HGNC ID for a gene."
    async def _lookup_rxnorm(self, term: str) -> Optional[str]  # "Look up RxNorm ID for a drug."
    async def _lookup_mesh_by_id(self, mesh_id: str) -> Optional[str]  # "Look up MeSH ID by known ID (no search needed)."
    async def _lookup_umls_by_id(self, umls_id: str) -> Optional[str]  # "Look up UMLS CUI by known ID (no search needed)."
    async def _lookup_hgnc_by_id(self, hgnc_id: str) -> Optional[str]  # "Look up HGNC ID by known ID (no search needed)."
    async def _lookup_rxnorm_by_id(self, rxnorm_id: str) -> Optional[str]  # "Look up RxNorm ID by known ID (no search needed)."
    async def _lookup_uniprot_by_id(self, uniprot_id: str) -> Optional[str]  # "Look up UniProt ID by known ID (no search needed)."
    async def _lookup_uniprot(self, term: str) -> Optional[str]  # "Look up UniProt ID for a protein."
    def _dbpedia_label_matches(self, term: str, label: str) -> bool  # "Check if a DBPedia label is a good match for the search term."
    async def _lookup_dbpedia(self, term: str) -> Optional[str]  # "Look up DBPedia URI as fallback for any entity type."
    async def _extract_authoritative_id_from_dbpedia(self, dbpedia_id: str, entity_type: str, original_term: str) -> Optional[str]  # "Extract authoritative ID from DBPedia resource properties."
    def _extract_authoritative_id_from_dbpedia_sync(self, client: 'httpx.Client', dbpedia_id: str, entity_type: str, original_term: str) -> Optional[str]  # "Synchronous version of authoritative ID extraction from DBPedia."
    def _lookup_mesh_by_id_sync(self, mesh_id: str) -> Optional[str]  # "Sync version: Look up MeSH ID by known ID."
    def _lookup_umls_by_id_sync(self, umls_id: str) -> Optional[str]  # "Sync version: Look up UMLS CUI by known ID."
    def _lookup_hgnc_by_id_sync(self, hgnc_id: str) -> Optional[str]  # "Sync version: Look up HGNC ID by known ID."
    def _lookup_rxnorm_by_id_sync(self, rxnorm_id: str) -> Optional[str]  # "Sync version: Look up RxNorm ID by known ID."
    def _lookup_uniprot_by_id_sync(self, uniprot_id: str) -> Optional[str]  # "Sync version: Look up UniProt ID by known ID."
    def lookup_sync(self, term: str, entity_type: str) -> Optional[CanonicalId]  # "Synchronous lookup (interface method)."
    def lookup_canonical_id_sync(self, term: str, entity_type: str) -> Optional[str]  # "Synchronous wrapper for use as Ollama tool."
    def _lookup_umls_sync(self, client: 'httpx.Client', term: str) -> Optional[str]  # "Synchronous UMLS lookup with MeSH fallback."
    def _lookup_mesh_sync(self, client: 'httpx.Client', term: str) -> Optional[str]  # "Synchronous MeSH lookup with term normalization."
    def _lookup_hgnc_sync(self, client: 'httpx.Client', term: str) -> Optional[str]  # "Synchronous HGNC lookup with alias fallback."
    def _lookup_rxnorm_sync(self, client: 'httpx.Client', term: str) -> Optional[str]  # "Synchronous RxNorm lookup."
    def _lookup_uniprot_sync(self, client: 'httpx.Client', term: str) -> Optional[str]  # "Synchronous UniProt lookup."
    def _lookup_dbpedia_sync(self, client: 'httpx.Client', term: str) -> Optional[str]  # "Synchronous DBPedia lookup as fallback with validation."
    async def close(self) -> None  # "Close the HTTP client and save cache."
    async def __aenter__(self)  # "Async context manager entry."
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # "Async context manager exit - saves cache and closes client."

============================================================
# examples/medlit/pipeline/bundle_builder.py
============================================================
# imports: __future__.annotations, datetime.datetime, timezone, examples.medlit.bundle_models.PerPaperBundle, examples.medlit.pipeline.canonical_urls.build_canonical_url, json, kgbundle.BundleFile, BundleManifestV1, DocAssetRow, EntityRow, EvidenceRow, MentionRow, RelationshipRow, pathlib.Path, shutil, typing.Any, uuid
def load_merged_output(merged_dir: Path) -> tuple[list[dict], list[dict], dict, dict]  # "Load merged Pass 2 output and id_map."
def load_pass1_bundles(bundles_dir: Path) -> list[tuple[str, PerPaperBundle]]  # "Load all paper_*.json bundles from bundles_dir. Returns list of (paper_id, bundle)."
def _entity_usage_from_bundles(bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]  # "Compute usage_count, total_mentions, supporting_documents, first_seen_* per merge_key."
def _merged_entity_to_entity_row(ent: dict, usage: dict[str, Any], created_at: str) -> EntityRow  # "Convert merged entity dict to EntityRow."
def _relationship_evidence_stats(merged_rels: list[dict], bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]  # "For each (sub, pred, obj) merge key, compute evidence_count, strongest_evidence_quote, evidence_confidence_avg."
def _merged_rel_to_relationship_row(rel: dict, stats: dict[tuple[str, str, str], dict[str, Any]], created_at: str) -> RelationshipRow  # "Convert merged relationship dict to RelationshipRow."
def _build_evidence_rows(bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]], merged_relationships: list[dict]) -> list[EvidenceRow]  # "Build EvidenceRow list from bundles; relationship_key uses merge keys. Offsets stubbed (0, len(text))."
def _build_mention_rows(bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]], created_at: str) -> list[MentionRow]  # "Build MentionRow list from bundles; entity_id is merge_key. Offsets stubbed (0, len(text_span))."
def run_pass3(merged_dir: Path, bundles_dir: Path, output_dir: Path) -> dict[str, Any]  # "Build kgbundle from merged Pass 2 output and Pass 1 bundles. Writes all bundle files."

============================================================
# examples/medlit/pipeline/canonical_urls.py
============================================================
# imports: typing.Optional
def build_canonical_url(canonical_id: str, entity_type: Optional[str]=None) -> Optional[str]  # "Build a canonical URL for an entity based on its canonical ID."
def build_canonical_urls_from_dict(canonical_ids: dict[str, str], entity_type: Optional[str]=None) -> dict[str, str]  # "Build canonical URLs for all canonical IDs in a dictionary."

============================================================
# examples/medlit/pipeline/config.py
============================================================
# imports: __future__.annotations, os, pathlib.Path, tomllib, typing.Any
DEFAULT_WINDOW_SIZE = 1536
DEFAULT_OVERLAP = 400
def _default_config_paths() -> list[Path]  # "Return paths to check for medlit.toml (first existing wins)."
def load_medlit_config() -> dict[str, Any]  # "Load medlit config from TOML file."

============================================================
# examples/medlit/pipeline/dedup.py
============================================================
# imports: examples.medlit.bundle_models.ExtractedEntityRow, PerPaperBundle, examples.medlit.pipeline.authority_lookup.CanonicalIdLookup, examples.medlit.pipeline.synonym_cache.add_same_as_to_cache, load_synonym_cache, lookup_entity, save_synonym_cache, json, pathlib.Path, typing.Any, Optional, uuid
def _is_authoritative_id(s: str) -> bool  # "Return True if s looks like an authoritative ontology ID, not a synthetic slug."
def _authoritative_id_from_entity(e: ExtractedEntityRow) -> Optional[str]  # "Return the best authoritative ID from bundle entity row, or None."
def _entity_class_to_lookup_type(entity_class: str) -> Optional[str]  # "Map bundle entity_class to CanonicalIdLookup entity_type (lowercase)."
def _canonical_id_slug() -> str  # "Generate a short synthetic merge key for entities without authoritative ID."
def run_pass2(bundle_dir: Path, output_dir: Path, synonym_cache_path: Optional[Path]=None, canonical_id_cache_path: Optional[Path]=None) -> dict[str, Any]  # "Run Pass 2: dedup and promotion. Reads bundles from bundle_dir, writes to output_dir."
def _run_pass2_impl(bundle_dir: Path, output_dir: Path, synonym_cache_path: Path, cache: dict, lookup: Any) -> dict[str, Any]  # "Inner Pass 2 implementation (lookup created and saved by caller)."

============================================================
# examples/medlit/pipeline/embeddings.py
============================================================
# imports: httpx, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, os, typing.Optional, Sequence
class OllamaMedLitEmbeddingGenerator(EmbeddingGeneratorInterface):  # "Real embedding generator using Ollama."
    def __init__(self, model: str='nomic-embed-text', ollama_host: str | None=None, timeout: float=30.0)
    @property
    def dimension(self) -> int  # "Return embedding dimension for the model."
    def _url(self) -> str
    async def generate(self, text: str) -> tuple[float, ...]  # "Generate embedding for a single text using Ollama /api/embed."
    async def _request_batch(self, texts: list[str]) -> list[tuple[float, ...]]  # "One HTTP request for one or more texts. Response order matches input order."
    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]  # "Generate embeddings for multiple texts in one request when possible."

============================================================
# examples/medlit/pipeline/llm_client.py
============================================================
# imports: abc.ABC, abstractmethod, asyncio, json, ollama, typing.Any, Callable, Optional
class LLMTimeoutError(TimeoutError):  # "Raised when an LLM request (e.g. Ollama) exceeds the configured timeout."
    ...
class LLMClientInterface(ABC):  # "Abstract interface for LLM clients."
    @abstractmethod
    async def generate(self, prompt: str, temperature: float=0.1, max_tokens: Optional[int]=None) -> str  # "Generate text completion from a prompt."
    @abstractmethod
    async def generate_json(self, prompt: str, temperature: float=0.1) -> dict[str, Any] | list[Any]  # "Generate structured JSON response from a prompt."
    async def generate_json_with_raw(self, prompt: str, temperature: float=0.1) -> tuple[dict[str, Any] | list[Any], str]  # "Generate structured JSON response AND return the raw model text."
    async def generate_json_with_tools(self, prompt: str, tools: list[Callable], temperature: float=0.1) -> dict[str, Any] | list[Any]  # "Generate JSON with tool calling support."
class OllamaLLMClient(LLMClientInterface):  # "Ollama LLM client implementation."
    def __init__(self, model: str='llama3.1:8b', host: str='http://localhost:11434', timeout: float=300.0)  # "Initialize Ollama client."
    def _parse_json_from_text(self, response_text: str) -> dict[str, Any] | list[Any]  # "Extract and parse JSON from response text."
    async def generate(self, prompt: str, temperature: float=0.1, max_tokens: Optional[int]=None) -> str  # "Generate text using Ollama."
    async def _call_llm_for_json(self, prompt: str, temperature: float=0.1) -> str  # "Call LLM and return raw response text (internal helper)."
    async def generate_json(self, prompt: str, temperature: float=0.1) -> dict[str, Any] | list[Any]  # "Generate structured JSON response from a prompt."
    async def generate_json_with_raw(self, prompt: str, temperature: float=0.1) -> tuple[dict[str, Any] | list[Any], str]  # "Generate structured JSON response AND return the raw model text."
    async def generate_json_with_tools(self, prompt: str, tools: list[Callable], temperature: float=0.1, max_tool_iterations: int=10) -> dict[str, Any] | list[Any]  # "Generate JSON with Ollama tool calling support."

============================================================
# examples/medlit/pipeline/mentions.py
============================================================
# imports: .llm_client.LLMClientInterface, kgraph.pipeline.interfaces.EntityExtractorInterface, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, kgschema.entity.EntityMention, traceback
def _normalize_mention_key(name: str, entity_type: str) -> tuple[str, str]  # "Normalized key for deduping mentions: (alphanumeric lower name, type)."
TYPE_MAPPING: dict[str, str | None] = {'test': 'procedure', 'diagnostic': 'procedure', 'imaging': 'procedure', 'assay': 'biomarker', 'marker': 'biomarker', 'polymorphism': 'gene', 'mutation': 'gene', 'variant': 'gene', 'system': None, 'organization': None}
KNOWN_TYPE_LABELS: frozenset[str] = frozenset({'disease', 'gene', 'drug', 'protein', 'symptom', 'procedure', 'biomarker', 'pathway', 'location', 'ethnicity', 'variant', 'polymorphism', 'mutation', 'test', 'diagnostic', 'imaging', 'assay', 'marker', 'system', 'organization'})
def _is_type_masquerading_as_name(name: str, entity_type: str) -> bool  # "Return True if the name is just the entity type (or a type label), not a real entity name."
class MedLitEntityExtractor(EntityExtractorInterface):  # "Extract entity mentions from journal articles."
    def __init__(self, llm_client: LLMClientInterface | None=None, domain: DomainSchema | None=None)  # "Initialize entity extractor."
    def _normalize_entity_type(self, entity_type_raw: str) -> str | None  # "Normalize LLM entity types to schema types."
    async def extract(self, document: BaseDocument) -> list[EntityMention]  # "Extract entity mentions from a journal article (single chunk)."

============================================================
# examples/medlit/pipeline/ner_extractor.py
============================================================
# imports: .mentions._is_type_masquerading_as_name, asyncio, kgraph.pipeline.interfaces.EntityExtractorInterface, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, kgschema.entity.EntityMention, torch, transformers.pipeline, typing.Any
LABEL_TO_MEDLIT_TYPE: dict[str, str] = {'chemical': 'drug', 'disease': 'disease', 'drug': 'drug', 'gene': 'gene', 'protein': 'protein', 'symptom': 'symptom', 'procedure': 'procedure', 'biomarker': 'biomarker', 'pathway': 'pathway', 'location': 'location', 'ethnicity': 'ethnicity'}
CHUNK_CHARS = 4000
CHUNK_OVERLAP = 200
MIN_TEXT_LEN = 10
def _normalize_entity_group(label: str) -> str  # "Normalize pipeline entity_group to lowercase, strip B-/I- prefix if present."
def _get_document_text(document: BaseDocument) -> str  # "Get text to run NER on: content or abstract."
def _run_ner_sync(pipeline: Any, text: str) -> list[dict]  # "Run NER pipeline on text; returns list of dicts with start, end, entity_group, score."
def _chunk_text(text: str, chunk_size: int=CHUNK_CHARS, overlap: int=CHUNK_OVERLAP) -> list[tuple[int, str]]  # "Split text into overlapping chunks; returns [(start_offset, chunk_text), ...]."
def _merge_and_dedupe(chunk_results: list[tuple[int, list[dict]]]) -> list[dict]  # "Merge NER results from chunks: adjust offsets and dedupe by span (keep higher score)."
class MedLitNEREntityExtractor(EntityExtractorInterface):  # "Extract entity mentions using a local NER model (e.g. BC5CDR)."
    def __init__(self, model_name_or_path: str='tner/roberta-base-bc5cdr', domain: DomainSchema | None=None, device: str | None=None, max_length: int=512, label_to_type: dict[str, str] | None=None, pipeline: Any=None)  # "Initialize the NER extractor."
    async def extract(self, document: BaseDocument) -> list[EntityMention]  # "Extract entity mentions from document text using the NER model."

============================================================
# examples/medlit/pipeline/parser.py
============================================================
# imports: ..documents.JournalArticle, datetime.datetime, timezone, io, json, kgraph.pipeline.interfaces.DocumentParserInterface, pathlib.Path, typing.Any, xml.etree.ElementTree
class JournalArticleParser(DocumentParserInterface):  # "Parse raw journal article content into JournalArticle documents."
    async def parse(self, raw_content: bytes, content_type: str, source_uri: str | None=None) -> JournalArticle  # "Parses raw document content into a structured `JournalArticle`."
    def _parse_xml_to_dict(self, root: Any, source_uri: str | None) -> dict[str, Any]  # "Converts a PMC XML structure into a dictionary."
    def _parse_from_dict(self, data: dict[str, Any], source_uri: str | None) -> JournalArticle  # "Constructs a `JournalArticle` from a dictionary."

============================================================
# examples/medlit/pipeline/pass1_llm.py
============================================================
# imports: abc.ABC, abstractmethod, anthropic, asyncio, examples.medlit.pipeline.llm_client.OllamaLLMClient, json, openai.AsyncOpenAI, os, typing.Any, Optional
class Pass1LLMInterface(ABC):  # "Interface for Pass 1 LLM: generate JSON from system + user message."
    @abstractmethod
    async def generate_json(self, system_prompt: str, user_message: str, temperature: float=0.1, max_tokens: int=16384) -> dict[str, Any]  # "Return a single JSON object (e.g. per-paper bundle)."
def _parse_json_from_text(response_text: str) -> dict[str, Any]  # "Extract and parse a JSON object from response text."
class AnthropicPass1LLM(Pass1LLMInterface):  # "Pass 1 LLM using Anthropic (Claude) API."
    def __init__(self, api_key: Optional[str]=None, model: str='claude-sonnet-4-20250514', timeout: float=300.0)
    async def generate_json(self, system_prompt: str, user_message: str, temperature: float=0.1, max_tokens: int=16384) -> dict[str, Any]
class OpenAIPass1LLM(Pass1LLMInterface):  # "Pass 1 LLM using OpenAI API or OpenAI-compatible endpoint (e.g. Lambda Labs)."
    def __init__(self, api_key: Optional[str]=None, base_url: Optional[str]=None, model: str='gpt-4o', timeout: float=300.0)
    async def generate_json(self, system_prompt: str, user_message: str, temperature: float=0.1, max_tokens: int=16384) -> dict[str, Any]
class OllamaPass1LLM(Pass1LLMInterface):  # "Pass 1 LLM using existing Ollama client (generate_json)."
    def __init__(self, ollama_client: Any)  # "ollama_client must have async generate_json(prompt, temperature) -> dict|list."
    async def generate_json(self, system_prompt: str, user_message: str, temperature: float=0.1, max_tokens: int=16384) -> dict[str, Any]
def get_pass1_llm(backend: str, *, model: Optional[str]=None, base_url: Optional[str]=None, timeout: float=300.0, ollama_client: Optional[Any]=None) -> Pass1LLMInterface  # "Return a Pass 1 LLM for the given backend."

============================================================
# examples/medlit/pipeline/pmc_chunker.py
============================================================
# imports: .pmc_streaming.DEFAULT_OVERLAP, DEFAULT_WINDOW_SIZE, iter_pmc_windows, __future__.annotations, kgraph.pipeline.streaming.ChunkingConfig, DocumentChunk, DocumentChunkerInterface, WindowedDocumentChunker, kgschema.document.BaseDocument, pathlib.Path
def _content_type_is_xml(content_type: str) -> bool  # "Return True if content_type is XML (strip parameters like ; charset=utf-8)."
class PMCStreamingChunker(DocumentChunkerInterface):  # "Chunker for PMC/JATS XML that uses iter_pmc_windows for memory-efficient chunking."
    def __init__(self, window_size: int=DEFAULT_WINDOW_SIZE, overlap: int=DEFAULT_OVERLAP, include_abstract_separately: bool=True, document_chunk_config: ChunkingConfig | None=None)  # "Initialize the PMC streaming chunker."
    async def chunk(self, document: BaseDocument) -> list[DocumentChunk]  # "Split a parsed document into chunks (fallback when no raw bytes)."
    async def chunk_from_raw(self, raw_content: bytes, content_type: str, document_id: str, source_uri: str | None=None) -> list[DocumentChunk]  # "Chunk from raw PMC XML bytes without loading the full document."
def document_id_from_source_uri(source_uri: str | None) -> str  # "Derive a document ID from source_uri (e.g. file stem). Used when parsing is deferred."

============================================================
# examples/medlit/pipeline/pmc_streaming.py
============================================================
# imports: __future__.annotations, io, typing.Iterator, xml.etree.ElementTree
DEFAULT_WINDOW_SIZE = 1536
DEFAULT_OVERLAP = 400
def _local_tag(tag: str) -> str  # "Strip XML namespace from tag for comparison."
def iter_pmc_sections(raw_content: bytes) -> Iterator[tuple[str, str]]  # "Yield (section_id, text) for abstract and each body section."
def iter_overlapping_windows(sections: Iterator[tuple[str, str]], window_size: int=DEFAULT_WINDOW_SIZE, overlap: int=DEFAULT_OVERLAP, *, include_abstract_separately: bool=True) -> Iterator[tuple[int, str]]  # "Turn a stream of (section_id, text) into overlapping windows."
def iter_pmc_windows(raw_content: bytes, window_size: int=DEFAULT_WINDOW_SIZE, overlap: int=DEFAULT_OVERLAP, include_abstract_separately: bool=True) -> Iterator[tuple[int, str]]  # "Yield overlapping text windows from PMC XML for full-paper extraction."

============================================================
# examples/medlit/pipeline/relationships.py
============================================================
# imports: ..domain.MedLitDomainSchema, ..relationships.MedicalClaimRelationship, .llm_client.LLMClientInterface, collections.defaultdict, datetime.datetime, timezone, json, kgraph.pipeline.interfaces.RelationshipExtractorInterface, kgraph.storage.memory._cosine_similarity, kgschema.document.BaseDocument, kgschema.domain.Evidence, Provenance, kgschema.entity.BaseEntity, kgschema.relationship.BaseRelationship, pathlib.Path, traceback, typing.TYPE_CHECKING, Any, Optional, Sequence
DEFAULT_TRACE_DIR = Path('/tmp/kgraph-relationship-traces')
PREDICATE_SPECIFICITY: dict[str, int] = {'indicates': 2, 'associated_with': 1}
def _normalize_entity_name(e: str) -> str
def _build_entity_index(entities: Sequence[BaseEntity]) -> dict[str, list[BaseEntity]]
def _deduplicate_relationships_by_predicate_specificity(relationships: list[BaseRelationship]) -> list[BaseRelationship]  # "For each (subject_id, object_id), keep only the relationship with highest predicate specificity."
def _normalize_evidence_for_match(text: str) -> str  # "Normalize evidence text for substring matching: lowercase, strip, collapse whitespace."
_EVIDENCE_DISEASE_CONTEXT_WORDS = frozenset({'tumor', 'cancer', 'cell', 'cells', 'positive', 'negativity', 'negative', 'staining', 'ihc', 'immunohisto', 'immunoreactivity', 'positivity', 'neoplastic'})
def _evidence_has_disease_context(evidence: str) -> bool  # "Return True if evidence text suggests disease/marker context (IHC, tumor, etc.)."
def _evidence_contains_both_entities(evidence: str, subject_name: str, object_name: str, subject_entity: BaseEntity | None, object_entity: BaseEntity | None) -> tuple[bool, str | None, dict[str, Any]]  # "Check that both subject and object (or synonyms) appear in the evidence text."
async def _evidence_contains_both_entities_semantic(evidence: str, subject_entity: BaseEntity, object_entity: BaseEntity, embedding_generator: Any, threshold: float, evidence_cache: dict[str, tuple[float, ...]], entity_name_cache: dict[str, tuple[float, ...]]) -> tuple[bool, str | None, dict[str, Any]]  # "Check that evidence semantically contains both entities via embedding similarity."
class MedLitRelationshipExtractor(RelationshipExtractorInterface):  # "Extract relationships from journal articles."
    def __init__(self, llm_client: Optional[LLMClientInterface]=None, domain: Optional['MedLitDomainSchema']=None, trace_dir: Optional[Path]=None, embedding_generator: Any=None, evidence_similarity_threshold: float=0.5)  # "Initialize relationship extractor."
    @property
    def trace_dir(self) -> Path  # "Get the trace directory."
    @trace_dir.setter
    def trace_dir(self, value: Path) -> None  # "Set the trace directory."
    def _should_swap_subject_object(self, predicate: str, subject_entity: BaseEntity, object_entity: BaseEntity) -> bool  # "Check if subject and object should be swapped based on type constraints."
    def _validate_predicate_semantics(self, predicate: str, evidence: str) -> bool  # "Validate that predicate semantics match the evidence text."
    async def extract(self, document: BaseDocument, entities: Sequence[BaseEntity]) -> list[BaseRelationship]  # "Extract relationships from a journal article."
    def _build_llm_prompt(self, text_sample: str, entity_list: str) -> str  # "Build the prompt for the LLM."
    async def _extract_with_llm(self, document: BaseDocument, entities: Sequence[BaseEntity]) -> list[BaseRelationship]  # "Extract relationships using LLM."
    async def _process_llm_item(self, item: Any, entity_index: dict[str, list[BaseEntity]], document: BaseDocument) -> tuple[BaseRelationship | None, dict[str, Any]]  # "Process a single item from the LLM response."
    def write_skip_trace(self, document_id: str, reason: str, entity_count: int) -> None  # "Write a minimal trace file when a window is skipped (e.g. fewer than 2 entities)."
    def _write_trace(self, document_id: str, trace: dict[str, Any]) -> None  # "Write trace file for debugging relationship extraction."

============================================================
# examples/medlit/pipeline/resolve.py
============================================================
# imports: .canonical_urls.build_canonical_url, build_canonical_urls_from_dict, datetime.datetime, timezone, kgraph.logging.setup_logging, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, kgraph.pipeline.interfaces.EntityResolverInterface, kgschema.domain.DomainSchema, kgschema.entity.BaseEntity, EntityMention, EntityStatus, kgschema.storage.EntityStorageInterface, pydantic.BaseModel, ConfigDict, typing.Sequence, uuid
class MedLitEntityResolver(BaseModel, EntityResolverInterface):  # "Resolve medical entity mentions to canonical or provisional entities."
    domain: DomainSchema
    embedding_generator: EmbeddingGeneratorInterface | None = None
    similarity_threshold: float = 0.85
    async def resolve(self, mention: EntityMention, existing_storage: EntityStorageInterface) -> tuple[BaseEntity, float]  # "Resolves a single entity mention to a canonical or provisional entity."
    async def resolve_batch(self, mentions: Sequence[EntityMention], existing_storage: EntityStorageInterface) -> list[tuple[BaseEntity, float]]  # "Resolves a sequence of entity mentions."
    def _parse_canonical_id(self, entity_id: str, entity_type: str) -> dict[str, str]  # "Parses a canonical ID string into a structured dictionary."

============================================================
# examples/medlit/pipeline/synonym_cache.py
============================================================
# imports: json, pathlib.Path, typing.Any, Optional
def _normalize(name: str) -> str
def load_synonym_cache(path: Path) -> dict[str, list[dict[str, Any]]]  # "Load synonym cache from JSON file. Returns dict keyed by normalized name."
def save_synonym_cache(path: Path, data: dict[str, list[dict[str, Any]]]) -> None  # "Save synonym cache to JSON file."
def lookup_entity(cache: dict[str, list[dict[str, Any]]], name: str, entity_class: str) -> tuple[Optional[str], Optional[list[dict[str, Any]]]]  # "Look up canonical_id or ambiguities for (name, class)."
def add_same_as_to_cache(cache: dict[str, list[dict[str, Any]]], entity_a: dict[str, Any], entity_b: dict[str, Any], confidence: float, asserted_by: str, resolution: Optional[str], source_papers: list[str]) -> None  # "Append a SAME_AS link to the in-memory cache (indexed by normalized names)."

============================================================
# examples/medlit/promotion.py
============================================================
# imports: .pipeline.authority_lookup.CanonicalIdLookup, kgraph.canonical_id.CanonicalId, CanonicalIdLookupInterface, check_entity_id_format, extract_canonical_id_from_entity, kgraph.logging.setup_logging, kgraph.promotion.PromotionPolicy, kgschema.entity.BaseEntity, EntityStatus, typing.Optional
class MedLitPromotionPolicy(PromotionPolicy):  # "Promotion policy for medical literature domain."
    def __init__(self, config, lookup: Optional[CanonicalIdLookupInterface]=None)  # "Initialize promotion policy."
    def should_promote(self, entity: BaseEntity) -> bool  # "Check if entity meets promotion thresholds."
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]  # "Assign canonical ID for a provisional entity."

============================================================
# examples/medlit/relationships.py
============================================================
# imports: kgschema.relationship.BaseRelationship
class MedicalClaimRelationship(BaseRelationship):  # "Base class for all medical claim relationships."
    def get_edge_type(self) -> str  # "Return edge type category."

============================================================
# examples/medlit/scripts/__init__.py
============================================================


============================================================
# examples/medlit/scripts/ingest.py
============================================================
# imports: ..domain.MedLitDomainSchema, ..pipeline.authority_lookup.CanonicalIdLookup, ..pipeline.config.load_medlit_config, ..pipeline.embeddings.OllamaMedLitEmbeddingGenerator, ..pipeline.llm_client.LLMTimeoutError, OllamaLLMClient, ..pipeline.mentions.MedLitEntityExtractor, ..pipeline.ner_extractor.MedLitNEREntityExtractor, ..pipeline.parser.JournalArticleParser, ..pipeline.pmc_chunker.PMCStreamingChunker, ..pipeline.relationships.MedLitRelationshipExtractor, ..pipeline.resolve.MedLitEntityResolver, ..stage_models.EntityExtractionStageResult, IngestionPipelineResult, PaperEntityExtractionResult, PaperRelationshipExtractionResult, PromotedEntityRecord, PromotionStageResult, RelationshipExtractionStageResult, argparse, asyncio, dataclasses.dataclass, field, datetime.datetime, timezone, fnmatch, kgraph.export.write_bundle, kgraph.ingest.IngestionOrchestrator, kgraph.logging.setup_logging, kgraph.pipeline.caching.CachedEmbeddingGenerator, EmbeddingCacheConfig, FileBasedEmbeddingsCache, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, kgraph.pipeline.interfaces.EntityExtractorInterface, kgraph.pipeline.streaming.BatchingEntityExtractor, WindowedRelationshipExtractor, kgraph.provenance.ProvenanceAccumulator, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.storage.DocumentStorageInterface, EntityStorageInterface, RelationshipStorageInterface, logging, logging.DEBUG, os, pathlib.Path, pydantic.BaseModel, shutil, sys, tempfile.TemporaryDirectory, time, traceback, typing.Any, uuid
TRACE_BASE_DIR = Path('/tmp/kgraph-traces')
logger = setup_logging()
@dataclass
class TraceCollector:  # "Collects paths to trace files written during ingestion."
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_files: list[Path] = field(default_factory=list)
    @property
    def trace_dir(self) -> Path  # "Get the trace directory for this run."
    @property
    def entity_trace_dir(self) -> Path  # "Get the entity trace directory for this run."
    @property
    def promotion_trace_dir(self) -> Path  # "Get the promotion trace directory for this run."
    @property
    def relationship_trace_dir(self) -> Path  # "Get the relationship trace directory for this run."
    def add(self, path: Path) -> None  # "Add a trace file path."
    def collect_from_directory(self, directory: Path, pattern: str='*.trace.json') -> None  # "Collect all trace files matching pattern from a directory."
    def print_summary(self) -> None  # "Print summary of all trace files written."
@dataclass
class ProgressTracker:  # "Track and report progress during long-running operations."
    total: int
    completed: int = 0
    start_time: float = field(default_factory=time.time)
    last_report_time: float = field(default_factory=time.time)
    report_interval: float = 30.0
    def increment(self) -> None  # "Increment completed count and report if interval elapsed."
    def report(self) -> None  # "Print progress report to stderr."
def build_orchestrator(use_ollama: bool=False, ollama_model: str='llama3.1:8b', ollama_host: str='http://localhost:11434', ollama_timeout: float=300.0, cache_file: Path | None=None, relationship_trace_dir: Path | None=None, embeddings_cache_file: Path | None=None, evidence_validation_mode: str='hybrid', evidence_similarity_threshold: float=0.5, entity_extractor: str='llm', ner_model: str='tner/roberta-base-bc5cdr') -> tuple[IngestionOrchestrator, CanonicalIdLookup | None, CachedEmbeddingGenerator | None]  # "Builds and configures the ingestion orchestrator and its components."
async def extract_entities_from_paper(orchestrator: IngestionOrchestrator, file_path: Path, content_type: str) -> tuple[str, int, int]  # "Extracts entities from a single document file."
async def extract_relationships_from_paper(orchestrator: IngestionOrchestrator, file_path: Path, content_type: str) -> tuple[str, int, int]  # "Extracts relationships from a single document file."
def parse_arguments() -> argparse.Namespace  # "Parses and validates command-line arguments for the ingestion script."
def find_input_files(input_dir: Path, limit: int | None, input_papers: str | None=None) -> list[tuple[Path, str]]  # "Finds all processable JSON and XML files in the input directory."
async def extract_entities_phase(orchestrator: IngestionOrchestrator, input_files: list[tuple[Path, str]], max_workers: int=1, progress_interval: float=30.0, quiet: bool=False, trace_all: bool=False) -> tuple[int, int, EntityExtractionStageResult]  # "Coordinates the entity extraction phase for all input files."
def _initialize_lookup(use_ollama: bool, cache_file: Path | None, quiet: bool, embedding_generator: Any=None) -> CanonicalIdLookup | None  # "Initializes the canonical ID lookup service."
def _build_promoted_records(promoted: list) -> list[PromotedEntityRecord]  # "Builds a list of promoted entity records from a list of promoted entities."
async def run_promotion_phase(orchestrator: IngestionOrchestrator, entity_storage: EntityStorageInterface, cache_file: Path | None=None, use_ollama: bool=False, quiet: bool=False, trace_all: bool=False) -> tuple[CanonicalIdLookup | None, PromotionStageResult]  # "Coordinates the entity promotion phase."
async def extract_relationships_phase(orchestrator: IngestionOrchestrator, input_files: list[tuple[Path, str]], max_workers: int=1, progress_interval: float=30.0, quiet: bool=False, trace_all: bool=False) -> tuple[int, int, RelationshipExtractionStageResult]  # "Coordinates the relationship extraction phase for all input files."
async def print_summary(document_storage: DocumentStorageInterface, entity_storage: EntityStorageInterface, relationship_storage: RelationshipStorageInterface, quiet: bool=False) -> None  # "Prints a formatted summary of the knowledge graph's contents."
async def export_bundle(entity_storage: EntityStorageInterface, relationship_storage: RelationshipStorageInterface, output_dir: Path, processed: int, errors: int, cache_file: Path | None=None, provenance_accumulator: ProvenanceAccumulator | None=None) -> None  # "Exports the final knowledge graph to a bundle directory."
def _handle_keyboard_interrupt(lookup: CanonicalIdLookup | None) -> None  # "Handles graceful shutdown on KeyboardInterrupt (Ctrl+C)."
async def _cleanup_lookup_service(lookup: CanonicalIdLookup | None) -> None  # "Closes resources associated with the lookup service."
def _output_stage_result(result: BaseModel, stage_name: str, quiet: bool) -> None  # "Output stage result as JSON to stdout."
def _initialize_pipeline(args: argparse.Namespace) -> tuple  # "Initializes the pipeline and returns necessary components."
async def main() -> None  # "Runs the main ingestion pipeline for medical literature."

============================================================
# examples/medlit/scripts/parse_pmc_xml.py
============================================================
# imports: argparse, json, pathlib.Path, typing.Any, xml.etree.ElementTree
def parse_pmc_xml_to_paper_schema(xml_path: Path) -> dict  # "Parse PMC XML file directly into Paper schema JSON format."
def main()

============================================================
# examples/medlit/scripts/pass1_extract.py
============================================================
# imports: argparse, asyncio, datetime.datetime, timezone, dotenv.load_dotenv, examples.medlit.bundle_models.EvidenceEntityRow, ExtractedEntityRow, PaperInfo, PerPaperBundle, RelationshipRow, examples.medlit.pipeline.parser.JournalArticleParser, examples.medlit.pipeline.pass1_llm.get_pass1_llm, examples.medlit_schema.base.ExecutionInfo, ExtractionPipelineInfo, ExtractionProvenance, ModelInfo, PromptInfo, hashlib, json, os, pathlib.Path, subprocess, sys, time, typing.Optional
REPO_ROOT = Path(__file__).resolve().parents[3]
def _git_info() -> dict  # "Return git_commit, git_commit_short, git_branch, git_dirty, repo_url."
def build_provenance(llm_name: str, llm_version: str, prompt_version: str='v1', prompt_template: str='medlit_extraction_v1', prompt_checksum: Optional[str]=None, duration_seconds: Optional[float]=None) -> ExtractionProvenance  # "Build extraction_provenance for Pass 1 output."
def _default_system_prompt() -> str  # "Minimal system prompt asking for per-paper bundle JSON."
async def _paper_content_from_parser(raw_content: bytes, content_type: str, source_uri: str) -> tuple[str, Optional[PaperInfo]]  # "Extract text and minimal PaperInfo from raw content using existing parser."
def _paper_content_fallback(raw_content: bytes, source_uri: str) -> tuple[str, PaperInfo]  # "Fallback: use raw text and filename for paper id."
async def run_pass1(input_dir: Path, output_dir: Path, llm_backend: str, limit: Optional[int]=None, papers: Optional[list[str]]=None, system_prompt: Optional[str]=None) -> None  # "Run Pass 1: for each paper in input_dir, call LLM and write bundle JSON to output_dir."
def main() -> None

============================================================
# examples/medlit/scripts/pass2_dedup.py
============================================================
# imports: argparse, examples.medlit.pipeline.dedup.run_pass2, pathlib.Path, sys
REPO_ROOT = Path(__file__).resolve().parents[3]
def main() -> None

============================================================
# examples/medlit/scripts/pass3_build_bundle.py
============================================================
# imports: argparse, dotenv.load_dotenv, examples.medlit.pipeline.bundle_builder.run_pass3, pathlib.Path, sys
REPO_ROOT = Path(__file__).resolve().parents[3]
def main() -> int

============================================================
# examples/medlit/stage_models.py
============================================================
# imports: datetime.datetime, enum.Enum, pydantic.BaseModel, ConfigDict, Field, typing.Any
class IngestionStage(str, Enum):  # "Pipeline stages where ingestion can be stopped."
    ...
class ExtractedEntityRecord(BaseModel):  # "Record of a single extracted entity."
    entity_id: str = Field(description='Assigned entity ID (provisional or canonical)')
    name: str = Field(description='Primary entity name')
    entity_type: str = Field(description='Entity type (disease, gene, drug, etc.)')
    status: str = Field(description='Entity status (provisional or canonical)')
    confidence: float = Field(description='Extraction confidence score')
    source: str = Field(description='Source document ID')
    canonical_ids: dict[str, str] = Field(default_factory=dict, description='Canonical IDs from authoritative sources')
    synonyms: tuple[str, ...] = Field(default=(), description='Known synonyms')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Additional metadata')
class PaperEntityExtractionResult(BaseModel):  # "Result of entity extraction from a single paper."
    document_id: str = Field(description='Document ID')
    source_uri: str | None = Field(default=None, description='Source file path or URI')
    extracted_at: datetime = Field(description='Timestamp of extraction')
    entities_extracted: int = Field(description='Total entity mentions found')
    entities_new: int = Field(description='New entities created')
    entities_existing: int = Field(description='Existing entities matched')
    entities: tuple[ExtractedEntityRecord, ...] = Field(default=(), description='Extracted entity records')
    errors: tuple[str, ...] = Field(default=(), description='Errors encountered')
class EntityExtractionStageResult(BaseModel):  # "Complete result of Stage 1: Entity Extraction across all papers."
    stage: str = Field(default='entities', description='Stage identifier')
    completed_at: datetime = Field(description='Timestamp when stage completed')
    papers_processed: int = Field(description='Number of papers processed')
    papers_failed: int = Field(description='Number of papers with errors')
    total_entities_extracted: int = Field(description='Total entity mentions')
    total_entities_new: int = Field(description='Total new entities created')
    total_entities_existing: int = Field(description='Total existing entities matched')
    paper_results: tuple[PaperEntityExtractionResult, ...] = Field(default=(), description='Per-paper extraction results')
    entity_type_counts: dict[str, int] = Field(default_factory=dict, description='Count of entities by type')
    provisional_count: int = Field(default=0, description='Number of provisional entities')
    canonical_count: int = Field(default=0, description='Number of canonical entities')
class PromotedEntityRecord(BaseModel):  # "Record of an entity that was promoted to canonical status."
    old_entity_id: str = Field(description='Previous provisional entity ID')
    new_entity_id: str = Field(description='New canonical entity ID')
    name: str = Field(description='Entity name')
    entity_type: str = Field(description='Entity type')
    canonical_source: str = Field(description='Source of canonical ID (umls, hgnc, etc.)')
    canonical_url: str | None = Field(default=None, description='URL to canonical source')
class PromotionStageResult(BaseModel):  # "Complete result of Stage 2: Entity Promotion."
    stage: str = Field(default='promotion', description='Stage identifier')
    completed_at: datetime = Field(description='Timestamp when stage completed')
    candidates_evaluated: int = Field(description='Entities meeting promotion thresholds')
    entities_promoted: int = Field(description='Entities successfully promoted')
    entities_skipped_no_canonical_id: int = Field(default=0, description='Skipped because no canonical ID found')
    entities_skipped_policy: int = Field(default=0, description='Skipped by promotion policy')
    entities_skipped_storage_failure: int = Field(default=0, description='Skipped due to storage errors')
    promoted_entities: tuple[PromotedEntityRecord, ...] = Field(default=(), description='Records of promoted entities')
    total_canonical_entities: int = Field(description='Total canonical entities after promotion')
    total_provisional_entities: int = Field(description='Remaining provisional entities')
class ExtractedRelationshipRecord(BaseModel):  # "Record of a single extracted relationship."
    subject_id: str = Field(description='Subject entity ID')
    subject_name: str = Field(description='Subject entity name')
    subject_type: str = Field(description='Subject entity type')
    predicate: str = Field(description='Relationship predicate')
    object_id: str = Field(description='Object entity ID')
    object_name: str = Field(description='Object entity name')
    object_type: str = Field(description='Object entity type')
    confidence: float = Field(description='Extraction confidence')
    source_document: str = Field(description='Source document ID')
    evidence_quote: str | None = Field(default=None, description='Supporting evidence text')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Additional metadata')
class PaperRelationshipExtractionResult(BaseModel):  # "Result of relationship extraction from a single paper."
    document_id: str = Field(description='Document ID')
    source_uri: str | None = Field(default=None, description='Source file path or URI')
    extracted_at: datetime = Field(description='Timestamp of extraction')
    relationships_extracted: int = Field(description='Number of relationships found')
    relationships: tuple[ExtractedRelationshipRecord, ...] = Field(default=(), description='Extracted relationship records')
    errors: tuple[str, ...] = Field(default=(), description='Errors encountered')
class RelationshipExtractionStageResult(BaseModel):  # "Complete result of Stage 3: Relationship Extraction."
    stage: str = Field(default='relationships', description='Stage identifier')
    completed_at: datetime = Field(description='Timestamp when stage completed')
    papers_processed: int = Field(description='Number of papers processed')
    papers_with_relationships: int = Field(description='Papers with at least one relationship')
    total_relationships_extracted: int = Field(description='Total relationships extracted')
    paper_results: tuple[PaperRelationshipExtractionResult, ...] = Field(default=(), description='Per-paper extraction results')
    predicate_counts: dict[str, int] = Field(default_factory=dict, description='Count of relationships by predicate')
class IngestionPipelineResult(BaseModel):  # "Complete result of the full ingestion pipeline."
    pipeline_version: str = Field(default='1.0.0', description='Pipeline version')
    started_at: datetime = Field(description='Pipeline start timestamp')
    completed_at: datetime = Field(description='Pipeline completion timestamp')
    stopped_at_stage: str | None = Field(default=None, description='Stage where pipeline was stopped (if --stop-after used)')
    entity_extraction: EntityExtractionStageResult | None = Field(default=None, description='Stage 1 results')
    promotion: PromotionStageResult | None = Field(default=None, description='Stage 2 results')
    relationship_extraction: RelationshipExtractionStageResult | None = Field(default=None, description='Stage 3 results')
    total_documents: int = Field(description='Total documents processed')
    total_entities: int = Field(description='Total entities in graph')
    total_relationships: int = Field(description='Total relationships in graph')

============================================================
# examples/medlit/tests/__init__.py
============================================================


============================================================
# examples/medlit/tests/conftest.py
============================================================
# imports: tests.conftest.entity_storage, relationship_storage, document_storage

============================================================
# examples/medlit/tests/test_authority_lookup.py
============================================================
# imports: examples.medlit.pipeline.authority_lookup.CanonicalIdLookup, pytest
class TestDBPediaLabelMatching:  # "Test the DBPedia label matching logic."
    @pytest.fixture
    def lookup(self)  # "Create a CanonicalIdLookup instance for testing."
    def test_exact_match(self, lookup)  # "Exact match should succeed."
    def test_term_contained_in_label(self, lookup)  # "Term contained in label should succeed."
    def test_label_contained_in_term(self, lookup)  # "Label contained in term should succeed."
    def test_label_starts_with_term(self, lookup)  # "Label starting with term should succeed."
    def test_common_prefix_singular_plural(self, lookup)  # "Common 6-char prefix should succeed (handles singular/plural)."
    def test_html_tags_stripped(self, lookup)  # "HTML bold tags should be stripped from labels."
    def test_case_insensitive(self, lookup)  # "Matching should be case-insensitive."
    def test_garbage_match_insect(self, lookup)  # "Garbage match 'HER2-enriched' → 'Insect' should fail."
    def test_garbage_match_animal(self, lookup)  # "Garbage match 'basal-like' → 'Animal' should fail."
    def test_unrelated_terms(self, lookup)  # "Completely unrelated terms should fail."
    def test_substring_match_allowed(self, lookup)  # "Substring matching is allowed (term in label)."
    def test_no_overlap_fails(self, lookup)  # "Terms with no overlap should fail."
class TestMeSHTermNormalization:  # "Test MeSH term normalization (cancer → neoplasms)."
    @pytest.fixture
    def lookup(self)  # "Create a CanonicalIdLookup instance for testing."
    def test_mesh_id_extraction(self, lookup)  # "Test extracting MeSH ID from API results."
    def test_mesh_id_extraction_word_order(self, lookup)  # "Test MeSH extraction handles word order differences."
    def test_mesh_id_extraction_no_match(self, lookup)  # "Test MeSH extraction returns None for no match."
    def test_mesh_id_extraction_empty_data(self, lookup)  # "Test MeSH extraction handles empty data."
    def test_mesh_id_extraction_prefers_general_over_complication(self, lookup)  # "Test that general terms are preferred over complications."
    def test_mesh_id_extraction_exact_match_priority(self, lookup)  # "Test that exact matches get highest priority."

============================================================
# examples/medlit/tests/test_entity_normalization.py
============================================================
# imports: datetime.datetime, timezone, examples.medlit.documents.JournalArticle, examples.medlit.domain.MedLitDomainSchema, examples.medlit.pipeline.mentions.MedLitEntityExtractor, TYPE_MAPPING, _is_type_masquerading_as_name, pytest
class TestTypeNormalizationWithDomain:  # "Test entity type normalization with domain schema validation."
    @pytest.fixture
    def extractor(self)  # "Create extractor with domain for full validation."
    def test_valid_type_passes_through(self, extractor)  # "Valid entity types should pass through unchanged."
    def test_case_normalization(self, extractor)  # "Types should be normalized to lowercase."
    def test_whitespace_stripped(self, extractor)  # "Whitespace should be stripped from types."
    def test_pipe_separated_takes_first_valid(self, extractor)  # "Pipe-separated types should return first valid type."
    def test_pipe_separated_skips_invalid(self, extractor)  # "Pipe-separated types should skip invalid types."
    def test_pipe_separated_all_invalid_returns_none(self, extractor)  # "Pipe-separated with all invalid types should return None."
    def test_common_mistake_test_to_procedure(self, extractor)  # "'test' should be normalized to 'procedure'."
    def test_common_mistake_diagnostic_to_procedure(self, extractor)  # "'diagnostic' should be normalized to 'procedure'."
    def test_common_mistake_imaging_to_procedure(self, extractor)  # "'imaging' should be normalized to 'procedure'."
    def test_common_mistake_assay_to_biomarker(self, extractor)  # "'assay' should be normalized to 'biomarker'."
    def test_common_mistake_marker_to_biomarker(self, extractor)  # "'marker' should be normalized to 'biomarker'."
    def test_skip_system_type(self, extractor)  # "'system' should be skipped (returns None)."
    def test_skip_organization_type(self, extractor)  # "'organization' should be skipped (returns None)."
    def test_invalid_type_returns_none(self, extractor)  # "Unknown types should return None."
class TestTypeNormalizationWithoutDomain:  # "Test entity type normalization without domain (basic mode)."
    @pytest.fixture
    def extractor(self)  # "Create extractor without domain for basic normalization."
    def test_basic_type_passes_through(self, extractor)  # "Types pass through in basic mode (no validation)."
    def test_basic_pipe_takes_first(self, extractor)  # "Pipe-separated takes first part in basic mode."
    def test_basic_mapping_applied(self, extractor)  # "TYPE_MAPPING is still applied in basic mode."
class TestTypeMappingConstants:  # "Test the TYPE_MAPPING constant has expected entries."
    def test_procedure_mappings_exist(self)  # "Procedure mappings should exist."
    def test_biomarker_mappings_exist(self)  # "Biomarker mappings should exist."
    def test_skip_mappings_exist(self)  # "Skip mappings (None values) should exist."
class TestTypeMasqueradingAsName:  # "Reject entity names that are actually type labels (e.g. LLM returns entity='disease', type='disease')."
    def test_name_equals_type_rejected(self)  # "When name equals type, treat as type masquerading as name."
    def test_known_type_labels_rejected_as_name(self)  # "Known type labels must not be used as entity names."
    def test_real_entity_names_allowed(self)  # "Real entity names should not be rejected."
    def test_empty_name_rejected(self)  # "Empty or whitespace-only name is rejected."
    @pytest.mark.asyncio
    async def test_pre_extracted_type_as_name_dropped(self)  # "Pre-extracted entities with name=type (e.g. name='disease', type='disease') produce no mention."

============================================================
# examples/medlit/tests/test_ner_extractor.py
============================================================
# imports: datetime.datetime, timezone, examples.medlit.documents.JournalArticle, examples.medlit.domain.MedLitDomainSchema, examples.medlit.pipeline.ner_extractor.LABEL_TO_MEDLIT_TYPE, MedLitNEREntityExtractor, _chunk_text, _get_document_text, _merge_and_dedupe, _normalize_entity_group, pytest, transformers
class TestNormalizeEntityGroup:  # "Test label normalization for NER pipeline output."
    def test_lowercase(self)
    def test_strip_b_i_prefix(self)
    def test_empty(self)
class TestLabelMapping:  # "Test default label -> medlit type mapping."
    def test_chemical_maps_to_drug(self)
    def test_disease_maps_to_disease(self)
    def test_known_types_present(self)
class TestChunkText:  # "Test long-document chunking."
    def test_short_text_single_chunk(self)
    def test_long_text_multiple_chunks(self)
    def test_merge_and_dedupe_adjusts_offsets(self)
    def test_merge_dedupes_overlapping_span_keeps_higher_score(self)
class TestGetDocumentText:  # "Test document text extraction."
    def test_uses_content(self)
    def test_falls_back_to_abstract_when_content_empty(self)
class TestMedLitNEREntityExtractorWithMock:  # "Test NER extractor with a mock pipeline (no real model load)."
    @pytest.fixture
    def mock_pipeline(self)  # "Pipeline that returns fixed entities for testing."
    @pytest.fixture
    def extractor_with_mock(self, mock_pipeline)
    @pytest.mark.asyncio
    async def test_extract_returns_mentions_with_correct_types(self, extractor_with_mock)
    @pytest.mark.asyncio
    async def test_extract_empty_text_returns_empty_list(self, extractor_with_mock)
    @pytest.mark.asyncio
    async def test_extract_short_text_returns_empty_list(self, extractor_with_mock)
    @pytest.mark.asyncio
    async def test_type_as_name_filtered_out(self, mock_pipeline)  # "When mock returns entity with word 'disease' (type label), it should be filtered out."
class TestMedLitNEREntityExtractorImportError:  # "Test that NER extractor raises clear ImportError when transformers not installed."
    def test_instantiation_without_pipeline_raises_import_error_when_no_transformers(self)  # "When transformers is not installed, constructing without pipeline= raises ImportError."

============================================================
# examples/medlit/tests/test_pass3_bundle_builder.py
============================================================
# imports: examples.medlit.bundle_models.PerPaperBundle, examples.medlit.pipeline.bundle_builder.load_merged_output, load_pass1_bundles, run_pass3, json, kgbundle.BundleManifestV1, pytest
@pytest.fixture
def minimal_merged_dir(tmp_path)  # "Minimal merged dir: entities.json, relationships.json, id_map.json, synonym_cache.json."
@pytest.fixture
def minimal_bundles_dir(tmp_path)  # "Minimal bundles dir: one paper_*.json with one relationship and matching evidence_entity."
def test_run_pass3_produces_bundle_files(minimal_merged_dir, minimal_bundles_dir, tmp_path)  # "run_pass3 writes entities.jsonl, relationships.jsonl, evidence.jsonl, mentions.jsonl, manifest.json."
def test_entity_row_has_usage_and_status(minimal_merged_dir, minimal_bundles_dir, tmp_path)  # "EntityRow in entities.jsonl has entity_id, status, usage_count/total_mentions from bundle scan."
def test_evidence_row_relationship_key_uses_merge_keys(minimal_merged_dir, minimal_bundles_dir, tmp_path)  # "EvidenceRow relationship_key uses merge keys (from id_map), not local ids."
def test_run_pass3_raises_when_id_map_missing(tmp_path, minimal_bundles_dir)  # "If id_map.json is missing in merged_dir, run_pass3 raises FileNotFoundError."
def test_load_merged_output_requires_id_map(tmp_path)  # "load_merged_output raises FileNotFoundError when id_map.json is missing."
def test_load_pass1_bundles(minimal_bundles_dir)  # "load_pass1_bundles returns list of (paper_id, PerPaperBundle)."

============================================================
# examples/medlit/tests/test_progress_tracker.py
============================================================
# imports: examples.medlit.scripts.ingest.ProgressTracker, io.StringIO, time, unittest.mock.patch
class TestProgressTrackerBasics:  # "Test basic ProgressTracker functionality."
    def test_initial_state(self)  # "Tracker should start with zero completed."
    def test_increment_increases_completed(self)  # "Increment should increase completed count."
    def test_percentage_calculation(self)  # "Report should calculate correct percentage."
    def test_percentage_zero_total(self)  # "Report should handle zero total gracefully."
    def test_report_shows_progress_count(self)  # "Report should show completed/total count."
class TestProgressTrackerTiming:  # "Test ProgressTracker timing-related functionality."
    def test_rate_calculation(self)  # "Report should calculate processing rate."
    def test_elapsed_time_shown(self)  # "Report should show elapsed time."
    def test_estimated_remaining_shown(self)  # "Report should show estimated remaining time when not complete."
class TestProgressTrackerAutoReport:  # "Test automatic reporting based on interval."
    def test_no_auto_report_before_interval(self)  # "Should not auto-report before interval elapses."
    def test_auto_report_after_interval(self)  # "Should auto-report when interval elapses."
class TestProgressTrackerEdgeCases:  # "Test edge cases for ProgressTracker."
    def test_large_total(self)  # "Should handle large totals."
    def test_custom_report_interval(self)  # "Should respect custom report interval."
    def test_completed_equals_total(self)  # "Should handle 100% completion."

============================================================
# examples/medlit/tests/test_promotion_lookup.py
============================================================
# imports: datetime.datetime, timezone, examples.medlit.domain.MedLitDomainSchema, examples.medlit.entities.DiseaseEntity, examples.medlit.pipeline.authority_lookup.CanonicalIdLookup, examples.medlit.pipeline.embeddings.OllamaMedLitEmbeddingGenerator, examples.medlit.pipeline.mentions.MedLitEntityExtractor, examples.medlit.pipeline.parser.JournalArticleParser, examples.medlit.pipeline.relationships.MedLitRelationshipExtractor, examples.medlit.pipeline.resolve.MedLitEntityResolver, kgraph.canonical_id.CanonicalId, kgraph.canonical_id.CanonicalIdLookupInterface, kgraph.ingest.IngestionOrchestrator, kgraph.storage.memory.InMemoryEntityStorage, InMemoryRelationshipStorage, InMemoryDocumentStorage, kgschema.entity.EntityStatus, pytest, unittest.mock.AsyncMock, MagicMock
@pytest.fixture
async def medlit_orchestrator(entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage, document_storage: InMemoryDocumentStorage)  # "Create orchestrator with MedLit domain for promotion testing."
@pytest.fixture
def mock_lookup()  # "Create a mock CanonicalIdLookupInterface for testing."
class TestPromotionLookupIntegration:  # "Test that lookup service is passed through the promotion chain."
    async def test_get_promotion_policy_accepts_lookup_parameter(self, medlit_orchestrator: IngestionOrchestrator, mock_lookup: MagicMock)  # "get_promotion_policy accepts lookup parameter and passes it to policy."
    async def test_get_promotion_policy_works_without_lookup(self, medlit_orchestrator: IngestionOrchestrator)  # "get_promotion_policy works when lookup is None (creates new instance)."
    async def test_run_promotion_passes_lookup_to_policy(self, medlit_orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage, mock_lookup: MagicMock)  # "run_promotion passes lookup parameter through to get_promotion_policy."
    async def test_promotion_uses_provided_lookup_service(self, medlit_orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage, mock_lookup: MagicMock)  # "Promotion uses the provided lookup service when assigning canonical IDs."

============================================================
# examples/medlit/tests/test_two_pass_ingestion.py
============================================================
# imports: examples.medlit.bundle_models.PerPaperBundle, examples.medlit.pipeline.dedup._is_authoritative_id, run_pass2, examples.medlit.pipeline.synonym_cache.load_synonym_cache, json, os, pathlib.Path, pytest, shutil
FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures' / 'bundles'
@pytest.fixture
def fixture_bundle_dir(tmp_path)  # "Copy fixture bundles to a temp dir so Pass 2 can read them."
def test_pass2_merges_same_name_class(fixture_bundle_dir, tmp_path)  # "Entities with same (name, class) across papers get the same canonical_id."
def test_pass2_writes_synonym_cache(fixture_bundle_dir, tmp_path)  # "Pass 2 writes synonym_cache.json."
def test_pass2_does_not_modify_input_bundles(fixture_bundle_dir, tmp_path)  # "Original bundle files are not modified (read-only)."
def test_pass2_writes_id_map(fixture_bundle_dir, tmp_path)  # "Pass 2 writes id_map.json so Pass 3 can resolve (paper_id, local_id) -> merge_key."
def test_pass2_accumulates_relationship_sources(fixture_bundle_dir, tmp_path)  # "Merged relationships aggregate source_papers and evidence_ids."
def test_fixture_bundles_load(fixture_bundle_dir)  # "Fixture bundles are valid PerPaperBundle."
def test_is_authoritative_id()  # "_is_authoritative_id returns True for ontology IDs, False for synthetic slugs."
def test_pass2_output_has_entity_id_and_canonical_id_null_when_synthetic(fixture_bundle_dir, tmp_path)  # "Pass 2 output entities have entity_id (merge key) and canonical_id null when synthetic."
def test_pass2_authoritative_id_from_bundle_preserved(tmp_path)  # "When a bundle entity has umls_id (or other authoritative ID), Pass 2 uses it as entity_id and canonical_id."

============================================================
# examples/medlit/vocab.py
============================================================
predicate_treats = 'treats'
predicate_causes = 'causes'
predicate_increases_risk = 'increases_risk'
predicate_decreases_risk = 'decreases_risk'
predicate_associated_with = 'associated_with'
predicate_interacts_with = 'interacts_with'
predicate_diagnosed_by = 'diagnosed_by'
predicate_side_effect = 'side_effect'
predicate_encodes = 'encodes'
predicate_participates_in = 'participates_in'
predicate_contraindicated_for = 'contraindicated_for'
predicate_prevents = 'prevents'
predicate_manages = 'manages'
predicate_binds_to = 'binds_to'
predicate_inhibits = 'inhibits'
predicate_activates = 'activates'
predicate_upregulates = 'upregulates'
predicate_downregulates = 'downregulates'
predicate_metabolizes = 'metabolizes'
predicate_diagnoses = 'diagnoses'
predicate_indicates = 'indicates'
predicate_precedes = 'precedes'
predicate_co_occurs_with = 'co_occurs_with'
predicate_located_in = 'located_in'
predicate_affects = 'affects'
predicate_supports = 'supports'
predicate_targets = 'targets'
predicate_subtype_of = 'subtype_of'
predicate_cites = 'cites'
predicate_studied_in = 'studied_in'
predicate_authored_by = 'authored_by'
predicate_part_of = 'part_of'
predicate_predicts = 'predicts'
predicate_refutes = 'refutes'
predicate_tested_by = 'tested_by'
predicate_generates = 'generates'
predicate_prevalent_in = 'prevalent_in'
predicate_endemic_to = 'endemic_to'
predicate_originates_from = 'originates_from'
ALL_PREDICATES = {predicate_treats, predicate_causes, predicate_increases_risk, predicate_decreases_risk, predicate_associated_with, predicate_interacts_with, predicate_diagnosed_by, predicate_side_effect, predicate_encodes, predicate_participates_in, predicate_contraindicated_for, predicate_prevents, predicate_manages, predicate_binds_to, predicate_inhibits, predicate_activates, predicate_upregulates, predicate_downregulates, predicate_metabolizes, predicate_diagnoses, predicate_indicates, predicate_precedes, predicate_co_occurs_with, predicate_located_in, predicate_affects, predicate_supports, predicate_targets, predicate_cites, predicate_studied_in, predicate_authored_by, predicate_part_of, predicate_predicts, predicate_refutes, predicate_tested_by, predicate_generates, predicate_prevalent_in, predicate_endemic_to, predicate_originates_from, predicate_subtype_of}
def get_valid_predicates(subject_type: str, object_type: str) -> list[str]  # "Return predicates valid between two entity types."

============================================================
# examples/medlit_schema/__init__.py
============================================================
__version__ = '1.0.0'
__schema_version__ = '1.0.0'

============================================================
# examples/medlit_schema/base.py
============================================================
# imports: enum.Enum, pydantic.BaseModel, Field, typing.Optional, uuid
class ModelInfo(BaseModel):  # "Information about the model used for extraction."
    name: str
    version: str
class ExtractionProvenance(BaseModel):  # "Complete provenance metadata for an extraction."
    extraction_pipeline: Optional['ExtractionPipelineInfo'] = None
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    prompt: Optional['PromptInfo'] = None
    execution: Optional['ExecutionInfo'] = None
    entity_resolution: Optional['EntityResolutionInfo'] = None
    model_info: Optional[ModelInfo] = None
class SectionType(str, Enum):  # "Type of section in a paper."
    ...
class TextSpanRef(BaseModel):  # "A structural locator for text within a parsed document."
    paper_id: str
    section_type: SectionType
    paragraph_idx: int
    sentence_idx: Optional[int] = None
    text_span: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
class ExtractionMethod(str, Enum):  # "Method used for extraction."
    ...
class StudyType(str, Enum):  # "Type of study."
    ...
class PredicateType(str, Enum):  # "All possible predicates (relationship types) in the medical literature knowledge graph."
    ...
class EntityType(str, Enum):  # "All possible entity types in the knowledge graph."
    ...
class ClaimPredicate(BaseModel):  # "Describes the nature of a claim made in a paper."
    predicate_type: PredicateType
    description: str
class Provenance(BaseModel):  # "Information about the origin of a piece of data."
    source_type: str
    source_id: str
    source_version: Optional[str] = None
    notes: Optional[str] = None
class EvidenceType(BaseModel):  # "The type of evidence supporting a relationship, linked to evidence ontologies."
    ontology_id: str
    ontology_label: str
    description: Optional[str] = None
class EntityReference(BaseModel):  # "Reference to an entity in the knowledge graph."
    id: str = Field(..., description='Canonical entity ID')
    name: str = Field(..., description='Entity name as mentioned in paper')
    type: EntityType = Field(..., description='Entity type')
class Polarity(str, Enum):  # "Polarity of evidence relative to a claim."
    ...
PaperId = uuid.UUID
EdgeId = uuid.UUID
class Edge(BaseModel):  # "Base edge in the knowledge graph."
    id: EdgeId
    subject: EntityReference
    object: EntityReference
    provenance: Provenance
class ExtractionEdge(Edge):  # "Edge from automated extraction."
    extractor: ModelInfo
    confidence: float
class ClaimEdge(Edge):  # "Edge representing a claim from a paper."
    predicate: ClaimPredicate
    asserted_by: PaperId
    polarity: Polarity
class EvidenceEdge(Edge):  # "Edge representing evidence for a claim."
    evidence_type: EvidenceType
    strength: float
class ExtractionPipelineInfo(BaseModel):  # "Information about the extraction pipeline version."
    name: str
    version: str
    git_commit: str
    git_commit_short: str
    git_branch: str
    git_dirty: bool
    repo_url: str
class PromptInfo(BaseModel):  # "Information about the prompt used."
    version: str
    template: str
    checksum: Optional[str] = None
class ExecutionInfo(BaseModel):  # "Information about when and where extraction was performed."
    timestamp: str
    hostname: str
    python_version: str
    duration_seconds: Optional[float] = None
class EntityResolutionInfo(BaseModel):  # "Information about entity resolution process."
    canonical_entities_matched: int
    new_entities_created: int
    similarity_threshold: float
    embedding_model: str
class Measurement(BaseModel):  # "Quantitative measurements associated with relationships."
    value: float
    unit: Optional[str] = None
    value_type: str
    p_value: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    study_population: Optional[str] = None
    measurement_context: Optional[str] = None

============================================================
# examples/medlit_schema/document.py
============================================================
# imports: kgschema.document.BaseDocument
class PaperDocument(BaseDocument):
    def get_document_type(self) -> str
    def get_sections(self) -> list[tuple[str, str]]

============================================================
# examples/medlit_schema/domain.py
============================================================
# imports: examples.medlit_schema.document.PaperDocument, examples.medlit_schema.entity.Disease, Gene, Drug, Protein, Mutation, Symptom, Biomarker, Pathway, Procedure, Paper, Author, ClinicalTrial, Institution, Hypothesis, StudyDesign, StatisticalMethod, EvidenceLine, Evidence, examples.medlit_schema.relationship.Treats, Causes, Prevents, IncreasesRisk, SideEffect, AssociatedWith, InteractsWith, ContraindicatedFor, DiagnosedBy, ParticipatesIn, Encodes, BindsTo, Inhibits, Upregulates, Downregulates, AuthoredBy, Cites, StudiedIn, PartOf, SameAs, Indicates, Predicts, TestedBy, Supports, Refutes, Generates, SubtypeOf, kgschema.canonical_id.CanonicalId, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, PredicateConstraint, ValidationIssue, kgschema.entity.BaseEntity, kgschema.promotion.PromotionPolicy, kgschema.relationship.BaseRelationship, typing.Optional
class MedlitPromotionPolicy(PromotionPolicy):
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]
class MedlitDomain(DomainSchema):  # "Domain schema for medical literature."
    @property
    def name(self) -> str
    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]
    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]
    @property
    def predicate_constraints(self) -> dict[str, PredicateConstraint]
    @property
    def document_types(self) -> dict[str, type[BaseDocument]]
    def validate_entity(self, entity: BaseEntity) -> list[ValidationIssue]
    async def validate_relationship(self, relationship: BaseRelationship, entity_storage=None) -> bool
    def get_promotion_policy(self, lookup=None) -> PromotionPolicy

============================================================
# examples/medlit_schema/entity.py
============================================================
# imports: datetime.datetime, examples.medlit_schema.base.ExtractionProvenance, ExtractionMethod, StudyType, kgschema.entity.BaseEntity, EntityStatus, pydantic.BaseModel, field_validator, model_validator, typing.List, Optional, Literal
class BaseMedicalEntity(BaseEntity):  # "Base for all medical entities."
    name: str
    synonyms: tuple[str, ...] = ()
    abbreviations: List[str] = []
    embedding: Optional[tuple[float, ...]] = None
    source: Literal['umls', 'mesh', 'rxnorm', 'hgnc', 'uniprot', 'extracted']
    @model_validator(mode='after')
    def canonical_entities_have_ontology_ids(self)
class Disease(BaseMedicalEntity):  # "Represents medical conditions, disorders, and syndromes."
    umls_id: Optional[str] = None
    mesh_id: Optional[str] = None
    icd10_codes: List[str] = []
    category: Optional[str] = None
    def get_entity_type(self) -> str
class Gene(BaseMedicalEntity):  # "Represents human genes."
    symbol: Optional[str] = None
    hgnc_id: Optional[str] = None
    chromosome: Optional[str] = None
    entrez_id: Optional[str] = None
    def get_entity_type(self) -> str
class Drug(BaseMedicalEntity):  # "Represents pharmaceutical drugs and medications."
    rxnorm_id: Optional[str] = None
    brand_names: List[str] = []
    drug_class: Optional[str] = None
    mechanism: Optional[str] = None
    def get_entity_type(self) -> str
class Protein(BaseMedicalEntity):  # "Represents proteins and protein complexes."
    uniprot_id: Optional[str] = None
    gene_id: Optional[str] = None
    function: Optional[str] = None
    pathways: List[str] = []
    def get_entity_type(self) -> str
class Mutation(BaseMedicalEntity):  # "Represents genetic mutations and variants."
    variant_notation: Optional[str] = None
    consequence: Optional[str] = None
    clinical_significance: Optional[str] = None
    def get_entity_type(self) -> str
class Symptom(BaseMedicalEntity):  # "Represents clinical signs and symptoms."
    severity_scale: Optional[str] = None
    onset_pattern: Optional[str] = None
    def get_entity_type(self) -> str
class Biomarker(BaseMedicalEntity):  # "Represents biological markers used for diagnosis or prognosis."
    loinc_code: Optional[str] = None
    measurement_type: Optional[str] = None
    clinical_use: Optional[str] = None
    def get_entity_type(self) -> str
class Pathway(BaseMedicalEntity):  # "Represents biological pathways."
    kegg_id: Optional[str] = None
    reactome_id: Optional[str] = None
    pathway_type: Optional[str] = None
    def get_entity_type(self) -> str
class Procedure(BaseMedicalEntity):  # "Represents medical tests, diagnostics, treatments."
    type: Optional[str] = None
    invasiveness: Optional[str] = None
    def get_entity_type(self) -> str
class PaperMetadata(BaseModel):  # "Extended metadata about the research paper."
    study_type: Optional[str] = None
    sample_size: Optional[int] = None
    study_population: Optional[str] = None
    primary_outcome: Optional[str] = None
    clinical_phase: Optional[str] = None
    mesh_terms: List[str] = []
class TextSpan(BaseEntity):  # "Represents a specific span of text within a document, acting as an anchor for evidence."
    promotable: bool = False
    status: EntityStatus = EntityStatus.CANONICAL
    paper_id: str
    section: str
    start_offset: int
    end_offset: int
    text_content: Optional[str] = None
    @field_validator('end_offset')
    def end_must_be_greater_than_start(cls, v, info)  # "Validate that end_offset > start_offset."
    def get_entity_type(self) -> str
class Paper(BaseEntity):  # "A research paper with extracted entities, relationships, and full provenance."
    paper_id: str
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = []
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    paper_metadata: PaperMetadata = PaperMetadata()
    extraction_provenance: Optional[ExtractionProvenance] = None
    def get_entity_type(self) -> str
class Author(BaseEntity):  # "Represents a researcher or author of scientific publications."
    orcid: Optional[str] = None
    affiliations: List[str] = []
    h_index: Optional[int] = None
    def get_entity_type(self) -> str
class ClinicalTrial(BaseEntity):  # "Represents a clinical trial registered on ClinicalTrials.gov."
    nct_id: Optional[str] = None
    title: Optional[str] = None
    phase: Optional[str] = None
    trial_status: Optional[str] = None
    intervention: Optional[str] = None
    def get_entity_type(self) -> str
class Institution(BaseEntity):  # "Represents research institutions and affiliations."
    country: Optional[str] = None
    department: Optional[str] = None
    def get_entity_type(self) -> str
class Hypothesis(BaseEntity):  # "Represents a scientific hypothesis tracked across the literature."
    iao_id: Optional[str] = None
    sepio_id: Optional[str] = None
    proposed_by: Optional[str] = None
    proposed_date: Optional[str] = None
    hypothesis_status: Optional[str] = None
    description: Optional[str] = None
    predicts: List[str] = []
    def get_entity_type(self) -> str
class StudyDesign(BaseEntity):  # "Represents a study design or experimental protocol."
    obi_id: Optional[str] = None
    stato_id: Optional[str] = None
    design_type: Optional[str] = None
    description: Optional[str] = None
    evidence_level: Optional[int] = None
    def get_entity_type(self) -> str
class StatisticalMethod(BaseEntity):  # "Represents a statistical method or test used in analysis."
    stato_id: Optional[str] = None
    method_type: Optional[str] = None
    description: Optional[str] = None
    assumptions: List[str] = []
    def get_entity_type(self) -> str
class EvidenceLine(BaseEntity):  # "Represents a line of evidence using SEPIO framework."
    sepio_type: Optional[str] = None
    eco_type: Optional[str] = None
    assertion_id: Optional[str] = None
    supports_ids: List[str] = []
    refutes_ids: List[str] = []
    evidence_items: List[str] = []
    strength: Optional[str] = None
    provenance_info: Optional[str] = None
    def get_entity_type(self) -> str
class Evidence(BaseEntity):  # "Evidence for a relationship, treated as a first-class entity."
    promotable: bool = False
    status: EntityStatus = EntityStatus.CANONICAL
    paper_id: str
    text_span_id: str
    confidence: float
    extraction_method: 'ExtractionMethod'
    study_type: 'StudyType'
    sample_size: Optional[int] = None
    eco_type: Optional[str] = None
    obi_study_design: Optional[str] = None
    stato_methods: List[str] = []
    @field_validator('paper_id', 'text_span_id')
    def ids_must_not_be_empty(cls, v)
    def get_entity_type(self) -> str

============================================================
# examples/medlit_schema/relationship.py
============================================================
# imports: examples.medlit_schema.base.Measurement, kgschema.relationship.BaseRelationship, pydantic.BaseModel, Field, field_validator, typing.Optional, Literal
class EvidenceItem(BaseModel):  # "Lightweight evidence reference for relationships."
    paper_id: str
    study_type: str
    sample_size: Optional[int] = None
    confidence: float = 0.5
class BaseMedicalRelationship(BaseRelationship):  # "Base class for all medical relationships with comprehensive provenance tracking."
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_papers: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    contradicted_by: list[str] = Field(default_factory=list)
    first_reported_date: Optional[str] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    properties: dict = Field(default_factory=dict)
    @field_validator('evidence_ids')
    def evidence_required_for_medical_assertions(cls, v)  # "Medical assertion relationships must include evidence."
class Treats(BaseMedicalRelationship):  # "Represents a therapeutic relationship between a drug and a disease."
    efficacy: Optional[str] = None
    response_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    line_of_therapy: Optional[Literal['first-line', 'second-line', 'third-line', 'maintenance', 'salvage']] = None
    indication: Optional[str] = None
    def get_edge_type(self) -> str
class Causes(BaseMedicalRelationship):  # "Represents a causal relationship between a disease and a symptom."
    frequency: Optional[Literal['always', 'often', 'sometimes', 'rarely']] = None
    onset: Optional[Literal['early', 'late']] = None
    severity: Optional[Literal['mild', 'moderate', 'severe']] = None
    def get_edge_type(self) -> str
class Prevents(BaseMedicalRelationship):  # "Drug prevents disease relationship."
    efficacy: Optional[str] = None
    risk_reduction: Optional[float] = Field(None, ge=0.0, le=1.0)
    def get_edge_type(self) -> str
class IncreasesRisk(BaseMedicalRelationship):  # "Represents genetic risk factors for diseases."
    risk_ratio: Optional[float] = Field(None, gt=0.0)
    penetrance: Optional[float] = Field(None, ge=0.0, le=1.0)
    age_of_onset: Optional[str] = None
    population: Optional[str] = None
    def get_edge_type(self) -> str
class SideEffect(BaseMedicalRelationship):  # "Represents adverse effects of medications."
    frequency: Optional[Literal['common', 'uncommon', 'rare']] = None
    severity: Optional[Literal['mild', 'moderate', 'severe']] = None
    reversible: bool = True
    def get_edge_type(self) -> str
class AssociatedWith(BaseMedicalRelationship):  # "Represents a general association between entities."
    association_type: Optional[Literal['positive', 'negative', 'neutral']] = None
    strength: Optional[Literal['strong', 'moderate', 'weak']] = None
    statistical_significance: Optional[float] = Field(None, ge=0.0, le=1.0)
    def get_edge_type(self) -> str
class InteractsWith(BaseMedicalRelationship):  # "Represents drug-drug interactions."
    interaction_type: Optional[Literal['synergistic', 'antagonistic', 'additive']] = None
    severity: Optional[Literal['major', 'moderate', 'minor']] = None
    mechanism: Optional[str] = None
    clinical_significance: Optional[str] = None
    def get_edge_type(self) -> str
class ContraindicatedFor(BaseMedicalRelationship):  # "Drug -[CONTRAINDICATED_FOR]-> Disease/Condition"
    severity: Optional[Literal['absolute', 'relative']] = None
    reason: Optional[str] = None
    def get_edge_type(self) -> str
class DiagnosedBy(BaseMedicalRelationship):  # "Represents diagnostic tests or biomarkers used to diagnose a disease."
    sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    specificity: Optional[float] = Field(None, ge=0.0, le=1.0)
    standard_of_care: bool = False
    def get_edge_type(self) -> str
class ParticipatesIn(BaseMedicalRelationship):  # "Gene/Protein -[PARTICIPATES_IN]-> Pathway"
    role: Optional[str] = None
    regulatory_effect: Optional[Literal['activates', 'inhibits', 'modulates']] = None
    def get_edge_type(self) -> str
class Encodes(BaseRelationship):
    def get_edge_type(self) -> str
class BindsTo(BaseRelationship):
    def get_edge_type(self) -> str
class Inhibits(BaseRelationship):
    def get_edge_type(self) -> str
class Upregulates(BaseRelationship):
    def get_edge_type(self) -> str
class Downregulates(BaseRelationship):
    def get_edge_type(self) -> str
class SubtypeOf(BaseMedicalRelationship):  # "When one disease is a subtype of another disease"
    def get_edge_type(self) -> str
class ResearchRelationship(BaseRelationship):  # "Base class for research metadata relationships."
    properties: dict = Field(default_factory=dict)
class Cites(ResearchRelationship):  # "Represents a citation from one paper to another."
    context: Optional[Literal['introduction', 'methods', 'results', 'discussion']] = None
    sentiment: Optional[Literal['supports', 'contradicts', 'mentions']] = None
    def get_edge_type(self) -> str
class StudiedIn(ResearchRelationship):  # "Links medical entities to papers that study them."
    role: Optional[Literal['primary_focus', 'secondary_finding', 'mentioned']] = None
    section: Optional[Literal['results', 'methods', 'discussion', 'introduction']] = None
    def get_edge_type(self) -> str
class AuthoredBy(ResearchRelationship):  # "Paper -[AUTHORED_BY]-> Author"
    position: Optional[Literal['first', 'last', 'corresponding', 'middle']] = None
    def get_edge_type(self) -> str
class PartOf(ResearchRelationship):  # "Paper -[PART_OF]-> ClinicalTrial"
    publication_type: Optional[Literal['protocol', 'results', 'analysis']] = None
    def get_edge_type(self) -> str
class SameAs(ResearchRelationship):  # "Provisional identity link between two entities."
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    resolution: Optional[Literal['merged', 'distinct']] = None
    note: Optional[str] = None
    def get_edge_type(self) -> str
class Indicates(BaseMedicalRelationship):  # "Biomarker or test result indicates disease or condition."
    def get_edge_type(self) -> str
class Predicts(BaseMedicalRelationship):  # "Represents a hypothesis predicting an observable outcome."
    prediction_type: Optional[Literal['positive', 'negative', 'conditional']] = None
    conditions: Optional[str] = None
    testable: bool = True
    def get_edge_type(self) -> str
class Refutes(BaseMedicalRelationship):  # "Represents evidence that refutes a hypothesis."
    refutation_strength: Optional[Literal['strong', 'moderate', 'weak']] = None
    alternative_explanation: Optional[str] = None
    limitations: Optional[str] = None
    def get_edge_type(self) -> str
class TestedBy(BaseMedicalRelationship):  # "Represents a hypothesis being tested by a study or clinical trial."
    test_outcome: Optional[Literal['supported', 'refuted', 'inconclusive']] = None
    methodology: Optional[str] = None
    study_design_id: Optional[str] = None
    def get_edge_type(self) -> str
class Supports(BaseMedicalRelationship):  # "Evidence supports a hypothesis or claim."
    support_strength: Optional[Literal['strong', 'moderate', 'weak']] = None
    def get_edge_type(self) -> str
class Generates(BaseMedicalRelationship):  # "Represents a study generating evidence for analysis."
    evidence_type: Optional[str] = None
    eco_type: Optional[str] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    def get_edge_type(self) -> str
RELATIONSHIP_TYPE_MAP = {'TREATS': Treats, 'PREVENTS': Prevents, 'CONTRAINDICATED_FOR': ContraindicatedFor, 'SIDE_EFFECT': SideEffect, 'CAUSES': Causes, 'INCREASES_RISK': IncreasesRisk, 'ASSOCIATED_WITH': AssociatedWith, 'INTERACTS_WITH': InteractsWith, 'DIAGNOSED_BY': DiagnosedBy, 'PARTICIPATES_IN': ParticipatesIn, 'INDICATES': Indicates, 'SUBTYPE_OF': SubtypeOf, 'SAME_AS': SameAs, 'ENCODES': Encodes, 'BINDS_TO': BindsTo, 'INHIBITS': Inhibits, 'UPREGULATES': Upregulates, 'DOWNREGULATES': Downregulates, 'CITES': Cites, 'STUDIED_IN': StudiedIn, 'AUTHORED_BY': AuthoredBy, 'PART_OF': PartOf, 'PREDICTS': Predicts, 'REFUTES': Refutes, 'TESTED_BY': TestedBy, 'SUPPORTS': Supports, 'GENERATES': Generates}
def create_relationship(predicate: str, subject_id: str, object_id: str, **kwargs) -> BaseRelationship  # "Factory function for creating typed relationship instances."

============================================================
# examples/medlit_schema/storage.py
============================================================


============================================================
# examples/sherlock/__init__.py
============================================================


============================================================
# examples/sherlock/data.py
============================================================
# imports: __future__.annotations, re, unicodedata
KNOWN_CHARACTERS: dict[str, dict] = {'holmes:char:SherlockHolmes': {'name': 'Sherlock Holmes', 'aliases': ('Holmes', 'Sherlock', 'Mr. Holmes', 'Mr. Sherlock Holmes', 'the detective', 'my friend Holmes'), 'role': 'detective'}, 'holmes:char:JohnWatson': {'name': 'Dr. John Watson', 'aliases': ('Watson', 'Dr. Watson', 'John Watson', 'Doctor Watson', 'John', 'my friend Watson', 'the doctor'), 'role': 'narrator'}, 'holmes:char:IreneAdler': {'name': 'Irene Adler', 'aliases': ('Irene', 'Miss Adler', 'Adler', 'the woman'), 'role': 'client'}, 'holmes:char:InspectorLestrade': {'name': 'Inspector Lestrade', 'aliases': ('Lestrade', 'G. Lestrade', 'Inspector Lestrade'), 'role': 'inspector'}, 'holmes:char:MrsHudson': {'name': 'Mrs. Hudson', 'aliases': ('Mrs Hudson', 'the landlady', 'our landlady'), 'role': 'landlady'}, 'holmes:char:MycroftHolmes': {'name': 'Mycroft Holmes', 'aliases': ('Mycroft', 'my brother Mycroft'), 'role': 'government'}}
KNOWN_LOCATIONS: dict[str, dict] = {'holmes:loc:BakerStreet221B': {'name': '221B Baker Street', 'aliases': ('Baker Street', '221B', 'our rooms', 'their lodgings'), 'location_type': 'residence'}, 'holmes:loc:ScotlandYard': {'name': 'Scotland Yard', 'aliases': ('the Yard', 'New Scotland Yard'), 'location_type': 'institution'}, 'holmes:loc:London': {'name': 'London', 'aliases': ('the metropolis', 'the city'), 'location_type': 'city'}, 'holmes:loc:DiogenesClub': {'name': 'The Diogenes Club', 'aliases': ('Diogenes Club', 'the club'), 'location_type': 'institution'}, 'holmes:loc:ReichenbachFalls': {'name': 'Reichenbach Falls', 'aliases': ('Reichenbach', 'the Falls'), 'location_type': 'landmark'}}
ADVENTURES_STORIES: list[dict] = [{'canonical_id': 'holmes:story:AScandalInBohemia', 'title': 'A Scandal in Bohemia', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'I. A SCANDAL IN BOHEMIA'}, {'canonical_id': 'holmes:story:TheRedHeadedLeague', 'title': 'The Red-Headed League', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'II. THE RED-HEADED LEAGUE'}, {'canonical_id': 'holmes:story:ACaseOfIdentity', 'title': 'A Case of Identity', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'III. A CASE OF IDENTITY'}, {'canonical_id': 'holmes:story:TheBoscombeValleyMystery', 'title': 'The Boscombe Valley Mystery', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'IV. THE BOSCOMBE VALLEY MYSTERY'}, {'canonical_id': 'holmes:story:TheFiveOrangePips', 'title': 'The Five Orange Pips', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'V. THE FIVE ORANGE PIPS'}, {'canonical_id': 'holmes:story:TheManWithTheTwistedLip', 'title': 'The Man With The Twisted Lip', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'VI. THE MAN WITH THE TWISTED LIP'}, {'canonical_id': 'holmes:story:TheAdventureOfTheBlueCarbuncle', 'title': 'The Adventure Of The Blue Carbuncle', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'VII. THE ADVENTURE OF THE BLUE CARBUNCLE'}, {'canonical_id': 'holmes:story:TheAdventureOfTheSpeckledBand', 'title': 'The Adventure Of The Speckled Band', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'VIII. THE ADVENTURE OF THE SPECKLED BAND'}, {'canonical_id': 'holmes:story:TheAdventureOfTheEngineersThumb', 'title': "The Adventure Of The Engineer's Thumb", 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': "IX. THE ADVENTURE OF THE ENGINEER'S THUMB"}, {'canonical_id': 'holmes:story:TheAdventureOfTheNobleBachelor', 'title': 'The Adventure Of The Noble Bachelor', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'X. THE ADVENTURE OF THE NOBLE BACHELOR'}, {'canonical_id': 'holmes:story:TheAdventureOfTheBerylCoronet', 'title': 'The Adventure Of The Beryl Coronet', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'XI. THE ADVENTURE OF THE BERYL CORONET'}, {'canonical_id': 'holmes:story:TheAdventureOfTheCopperBeeches', 'title': 'The Adventure Of The Copper Beeches', 'collection': 'The Adventures of Sherlock Holmes', 'year': 1891, 'marker': 'XII. THE ADVENTURE OF THE COPPER BEECHES'}]
def story_markers() -> list[str]
def _norm_title(s: str) -> str
_STORY_BY_NORM_TITLE = {_norm_title(story['title']): story for story in ADVENTURES_STORIES}
def find_story_by_title(title: str) -> dict | None
def find_story_by_marker(marker: str) -> dict | None

============================================================
# examples/sherlock/domain.py
============================================================
# imports: .promotion.SherlockPromotionPolicy, __future__.annotations, kgraph.promotion.PromotionPolicy, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, PredicateConstraint, ValidationIssue, kgschema.entity.BaseEntity, PromotionConfig, kgschema.relationship.BaseRelationship, kgschema.storage.EntityStorageInterface, typing.Optional
class SherlockCharacter(BaseEntity):  # "A character in the Sherlock Holmes stories."
    role: Optional[str] = None
    def get_entity_type(self) -> str
class SherlockLocation(BaseEntity):  # "A location mentioned in the stories."
    location_type: Optional[str] = None
    def get_entity_type(self) -> str
class SherlockStory(BaseEntity):  # "A story or novel in the Holmes canon."
    collection: Optional[str] = None
    publication_year: Optional[int] = None
    def get_entity_type(self) -> str
class AppearsInRelationship(BaseRelationship):  # "Character appears in a story."
    def get_edge_type(self) -> str
class CoOccursWithRelationship(BaseRelationship):  # "Two characters co-occur within the same textual context."
    def get_edge_type(self) -> str
class LivesAtRelationship(BaseRelationship):  # "Character lives at a location."
    def get_edge_type(self) -> str
class AntagonistOfRelationship(BaseRelationship):  # "Character is an antagonist of another character."
    def get_edge_type(self) -> str
class AllyOfRelationship(BaseRelationship):  # "Character is an ally of another character."
    def get_edge_type(self) -> str
class SherlockDocument(BaseDocument):  # "A Sherlock Holmes story document."
    story_id: Optional[str] = None
    collection: Optional[str] = None
    def get_document_type(self) -> str
    def get_sections(self) -> list[tuple[str, str]]
class SherlockDomainSchema(DomainSchema):
    @property
    def name(self) -> str
    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]
    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]
    @property
    def predicate_constraints(self) -> dict[str, PredicateConstraint]  # "Define predicate constraints for the Sherlock domain."
    @property
    def document_types(self) -> dict[str, type[BaseDocument]]
    @property
    def promotion_config(self) -> PromotionConfig
    def get_promotion_policy(self, lookup=None) -> PromotionPolicy
    def validate_entity(self, entity: BaseEntity) -> list[ValidationIssue]
    async def validate_relationship(self, relationship: BaseRelationship, entity_storage: EntityStorageInterface | None=None) -> bool

============================================================
# examples/sherlock/pipeline/__init__.py
============================================================
# imports: .embeddings.SimpleEmbeddingGenerator, .mentions.SherlockEntityExtractor, .parser.SherlockDocumentParser, .relationships.SherlockRelationshipExtractor, .resolve.SherlockEntityResolver
__all__ = ['SherlockDocumentParser', 'SherlockEntityExtractor', 'SherlockEntityResolver', 'SherlockRelationshipExtractor', 'SimpleEmbeddingGenerator']

============================================================
# examples/sherlock/pipeline/embeddings.py
============================================================
# imports: __future__.annotations, hashlib, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, typing.Sequence
class SimpleEmbeddingGenerator(EmbeddingGeneratorInterface):  # "Deterministic hash-based embedding generator (demo only)."
    @property
    def dimension(self) -> int
    async def generate(self, text: str) -> tuple[float, ...]
    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]

============================================================
# examples/sherlock/pipeline/mentions.py
============================================================
# imports: ..data.KNOWN_CHARACTERS, KNOWN_LOCATIONS, __future__.annotations, kgraph.pipeline.interfaces.EntityExtractorInterface, kgschema.document.BaseDocument, kgschema.entity.EntityMention, re
class SherlockEntityExtractor(EntityExtractorInterface):  # "Extract character, location, and story mentions using curated alias lists."
    def __init__(self) -> None
    def _build_patterns(self) -> list[tuple[re.Pattern, str, str, float]]
    async def extract(self, document: BaseDocument) -> list[EntityMention]

============================================================
# examples/sherlock/pipeline/parser.py
============================================================
# imports: ..data.find_story_by_title, ..domain.SherlockDocument, __future__.annotations, datetime.datetime, timezone, kgraph.pipeline.interfaces.DocumentParserInterface, uuid
class SherlockDocumentParser(DocumentParserInterface):  # "Parse plain text Sherlock Holmes stories into SherlockDocument objects."
    async def parse(self, raw_content: bytes, content_type: str, source_uri: str | None=None) -> SherlockDocument
    def _extract_title(self, text: str, source_uri: str | None) -> str

============================================================
# examples/sherlock/pipeline/relationships.py
============================================================
# imports: ..domain.AppearsInRelationship, CoOccursWithRelationship, __future__.annotations, datetime.datetime, timezone, itertools, kgraph.pipeline.interfaces.RelationshipExtractorInterface, kgschema.document.BaseDocument, kgschema.entity.BaseEntity, kgschema.relationship.BaseRelationship, re, typing.Sequence
class SherlockRelationshipExtractor(RelationshipExtractorInterface):  # "Extract relationships from resolved entities + document text."
    async def extract(self, document: BaseDocument, entities: Sequence[BaseEntity]) -> list[BaseRelationship]

============================================================
# examples/sherlock/pipeline/resolve.py
============================================================
# imports: datetime.datetime, timezone, kgraph.pipeline.interfaces.EntityResolverInterface, kgschema.domain.DomainSchema, kgschema.entity.BaseEntity, EntityStatus, EntityMention, kgschema.storage.EntityStorageInterface, pydantic.BaseModel, ConfigDict, typing.Sequence, uuid
class SherlockEntityResolver(BaseModel, EntityResolverInterface):  # "Resolve Sherlock entity mentions to canonical or provisional entities."
    domain: DomainSchema
    async def resolve(self, mention: EntityMention, existing_storage: EntityStorageInterface) -> tuple[BaseEntity, float]
    async def resolve_batch(self, mentions: Sequence[EntityMention], existing_storage: EntityStorageInterface) -> list[tuple[BaseEntity, float]]

============================================================
# examples/sherlock/promotion.py
============================================================
# imports: kgraph.canonical_id.CanonicalId, extract_canonical_id_from_entity, kgraph.promotion.PromotionPolicy, kgschema.entity.BaseEntity, typing.Optional
SHERLOCK_CANONICAL_IDS = {'holmes:char:SherlockHolmes': 'http://dbpedia.org/resource/Sherlock_Holmes', 'holmes:char:JohnWatson': 'http://dbpedia.org/resource/Dr._Watson', 'holmes:char:MrsHudson': 'http://dbpedia.org/resource/Mrs._Hudson', 'holmes:char:IreneAdler': 'http://dbpedia.org/resource/Irene_Adler', 'holmes:loc:BakerStreet221B': 'http://dbpedia.org/resource/221B_Baker_Street', 'holmes:story:AScandalInBohemia': 'http://dbpedia.org/resource/A_Scandal_in_Bohemia'}
class SherlockPromotionPolicy(PromotionPolicy):  # "Promotion policy for Sherlock Holmes domain using curated DBPedia mappings."
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]  # "Assign canonical ID for a provisional entity."

============================================================
# examples/sherlock/scripts/__init__.py
============================================================


============================================================
# examples/sherlock/scripts/ingest.py
============================================================
# imports: ..domain.SherlockDomainSchema, ..pipeline.SherlockDocumentParser, SherlockEntityExtractor, SherlockEntityResolver, SherlockRelationshipExtractor, SimpleEmbeddingGenerator, ..sources.gutenberg.download_adventures, __future__.annotations, asyncio, kgraph.export.write_bundle, kgraph.ingest.IngestionOrchestrator, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, pathlib.Path, tempfile.TemporaryDirectory
build_orch_blurb = "\n# What's so great about build_orchestrator() ?\n\nGreat question — this gets to the *why* of having `build_orchestrator()` at all.\n\nShort answer: **it packages all the domain-specific wiring into one stable,\nboring, copy-paste-able place** so extenders don’t have to re-learn the\ningestion graph every time.\n\nLet me be concrete.\n\n---\n\n## What `build_orchestrator()` actually gives an extender\n\n### 1. A *single authoritative wiring point*\n\n`IngestionOrchestrator` has a lot of moving parts:\n\n```python\nIngestionOrchestrator(\n    domain=...,\n    parser=...,\n    entity_extractor=...,\n    entity_resolver=...,\n    relationship_extractor=...,\n    embedding_generator=...,\n    entity_storage=...,\n    relationship_storage=...,\n    document_storage=...,\n)\n```\n\nAn extender **should not have to remember**:\n\n* which extractor goes in which slot\n* which ones are async\n* which storages are required vs optional\n* which defaults are safe\n\n`build_orchestrator()` freezes all of that into one known-good configuration.\n\nIf someone wants to build *their own* Sherlock-like domain, they can literally\nstart by copying that function.\n\n---\n\n### 2. A stable surface for experimentation\n\nExtenders often want to tweak **one thing**:\n\n* swap out the embedding generator\n* replace the relationship extractor\n* use a persistent storage instead of in-memory\n* add logging or metrics\n\nWith `build_orchestrator()` they can do:\n\n```python\norch = build_orchestrator(\n    embedding_generator=MyBetterEmbeddings(),\n)\n```\n\nor:\n\n```python\norch = build_orchestrator(\n    relationship_extractor=MySmarterCoOccurrenceExtractor(),\n)\n```\n\ninstead of re-wiring everything manually.\n\nThat’s huge for iteration speed.\n\n---\n\n### 3. A pedagogical artifact\n\nFor *this repository*, `build_orchestrator()` is also documentation.\n\nIt answers, in executable code:\n\n> “What does a complete, correct kgraph ingestion pipeline look like for a real\n> domain?”\n\nThat’s much clearer than prose.\n\nSomeone reading the example learns:\n\n* which interfaces matter\n* how domain schema + pipeline pieces fit together\n* which parts are optional vs essential\n\n---\n\n### 4. A place to encode “best practice defaults”\n\nYou’ve already discovered this implicitly:\n\n* in-memory storage is fine for examples\n* promotion config lives on the domain\n* parser must run before entity extraction\n* relationship extractor must not create entities\n\n`build_orchestrator()` is where those conventions live.\n\nIf you later discover a better default (e.g. batching embeddings, better clock\nhandling), you change it **once**.\n\n---\n\n## What it does *not* give (important)\n\nIt is **not**:\n\n* a required abstraction\n* a magic factory\n* something users must use\n\nIt’s a **convenience + example**, not a framework constraint.\n\nAdvanced users can ignore it entirely.\n\n---\n\n## Where I’d put it\n\nYou have two reasonable options:\n\n### Option A (most pedagogical)\n\n```text\nexamples/sherlock/scripts/ingest.py\n```\n\nKeep it near the CLI entrypoint, clearly labeled “example wiring”.\n\n### Option B (clean API surface)\n\n```text\nexamples/sherlock/__init__.py\n```\n\nExport it so people can do:\n\n```python\nfrom examples.sherlock import build_orchestrator\n```\n\nThis makes Sherlock feel like a mini-library.\n\nEither is fine; Option A emphasizes “example”, Option B emphasizes “reusable”.\n\n---\n\n## TL;DR\n\nExtenders want `build_orchestrator()` because it:\n\n* shows the **entire pipeline wiring in one place**\n* reduces cognitive load when experimenting\n* is copy-pasteable into new domains\n* encodes best practices you’ve already learned the hard way\n\nIf you didn’t provide it, many users would re-invent it badly.\n\nIf you want, I can sketch a *final form* `build_orchestrator()` signature\nthat’s maximally helpful but still minimal.\n"
mkdocs_yml_replacement = '\nsite_name: Domain-Agnostic Knowledge Graph Server\nsite_description: Documentation for the domain-agnostic knowledge graph server\nrepo_url: https://github.com/wware/kgserver\nrepo_name: wware/kgserver\n\ntheme:\n  name: material\n\nmarkdown_extensions:\n  - pymdownx.highlight:\n      anchor_linenums: true\n  - pymdownx.superfences\n  - pymdownx.tabbed:\n      alternate_style: true\n  - admonition\n  - pymdownx.details\n  - tables\n  - toc:\n      permalink: true\n\nnav:\n  - Home: index.md\n  - Architecture: architecture.md\n  - BuildOrchestrator: build_orch.md\n'
def build_orchestrator() -> IngestionOrchestrator
async def main() -> None

============================================================
# examples/sherlock/scripts/query.py
============================================================
# imports: ..domain.SherlockDomainSchema, ..pipeline.SherlockDocumentParser, SherlockEntityExtractor, SherlockEntityResolver, SherlockRelationshipExtractor, SimpleEmbeddingGenerator, ..sources.gutenberg.download_adventures, __future__.annotations, asyncio, kgraph.ingest.IngestionOrchestrator, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage
async def build_or_load() -> tuple[InMemoryEntityStorage, InMemoryRelationshipStorage]
async def main() -> None

============================================================
# examples/sherlock/sources/__init__.py
============================================================


============================================================
# examples/sherlock/sources/gutenberg.py
============================================================
# imports: __future__.annotations, pathlib.Path, re, urllib.request
GUTENBERG_URL = 'https://www.gutenberg.org/files/1661/1661-0.txt'
CACHE_DIR = Path(__file__).parent / 'data'
STORY_MARKERS: list[tuple[str, str]] = [('A Scandal in Bohemia', 'ADVENTURE I. A SCANDAL IN BOHEMIA'), ('The Red-Headed League', 'ADVENTURE II. THE RED-HEADED LEAGUE'), ('A Case of Identity', 'ADVENTURE III. A CASE OF IDENTITY'), ('The Boscombe Valley Mystery', 'ADVENTURE IV. THE BOSCOMBE VALLEY MYSTERY'), ('The Five Orange Pips', 'ADVENTURE V. THE FIVE ORANGE PIPS'), ('The Man with the Twisted Lip', 'ADVENTURE VI. THE MAN WITH THE TWISTED LIP'), ('The Adventure of the Blue Carbuncle', 'ADVENTURE VII. THE ADVENTURE OF THE BLUE CARBUNCLE'), ('The Adventure of the Speckled Band', 'ADVENTURE VIII. THE ADVENTURE OF THE SPECKLED BAND'), ("The Adventure of the Engineer's Thumb", "ADVENTURE IX. THE ADVENTURE OF THE ENGINEER'S THUMB"), ('The Adventure of the Noble Bachelor', 'ADVENTURE X. THE ADVENTURE OF THE NOBLE BACHELOR'), ('The Adventure of the Beryl Coronet', 'ADVENTURE XI. THE ADVENTURE OF THE BERYL CORONET'), ('The Adventure of the Copper Beeches', 'ADVENTURE XII. THE ADVENTURE OF THE COPPER BEECHES')]
GUTENBERG_START_RE = re.compile('\\*\\*\\*\\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\\*\\*\\*', re.IGNORECASE)
GUTENBERG_END_RE = re.compile('\\*\\*\\*\\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\\*\\*\\*', re.IGNORECASE)
def download_adventures(force_download: bool=False) -> list[tuple[str, str]]  # "Download and split The Adventures of Sherlock Holmes into stories."
def _strip_gutenberg_boilerplate(text: str) -> str  # "Remove Gutenberg license/header/footer so story splits are cleaner."
_ROMAN_HEADER = re.compile("(?im)^(?P<num>I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\\.\\s+(?P<title>[A-Z][A-Z \\-’']+)\\s*$")
def _split_into_stories(text: str) -> list[tuple[str, str]]

============================================================
# kgbundle/kgbundle/__init__.py
============================================================
# imports: .models.BundleFile, BundleManifestV1, DocAssetRow, EntityRow, EvidenceRow, MentionRow, RelationshipRow
__all__ = ['EntityRow', 'RelationshipRow', 'BundleFile', 'DocAssetRow', 'BundleManifestV1', 'MentionRow', 'EvidenceRow']
__version__ = '0.1.0'

============================================================
# kgbundle/kgbundle/models.py
============================================================
# imports: pydantic.BaseModel, Field, typing.Any, Dict, List, Optional
class EntityRow(BaseModel):  # "Entity row format for bundle JSONL files."
    entity_id: str = Field(..., description='Unique entity identifier (namespaced)')
    entity_type: str = Field(..., description='Type of entity (e.g., character, location)')
    name: Optional[str] = Field(None, description='Primary display name')
    status: str = Field(..., description='Entity status (e.g., canonical, provisional)')
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description='Confidence score')
    usage_count: int = Field(..., description='Number of times entity has been mentioned')
    created_at: str = Field(..., description='ISO 8601 creation timestamp')
    source: str = Field(..., description='Source of the entity (e.g., sherlock:curated)')
    canonical_url: Optional[str] = Field(None, description='URL to the authoritative source for this entity')
    properties: Dict[str, Any] = Field(default_factory=dict, description='Additional entity properties')
    first_seen_document: Optional[str] = Field(None, description='Document ID of earliest mention')
    first_seen_section: Optional[str] = Field(None, description='Section of earliest mention')
    total_mentions: int = Field(0, description='Total number of mention rows for this entity')
    supporting_documents: List[str] = Field(default_factory=list, description='Distinct document IDs where this entity is mentioned')
class RelationshipRow(BaseModel):  # "Relationship row format for bundle JSONL files."
    subject_id: str = Field(..., description='Source/subject entity ID')
    object_id: str = Field(..., description='Target/object entity ID')
    predicate: str = Field(..., description='Relationship type (e.g., appears_in)')
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description='Confidence score')
    source_documents: List[str] = Field(..., description='List of document IDs providing evidence')
    created_at: str = Field(..., description='ISO 8601 creation timestamp')
    properties: Dict[str, Any] = Field(default_factory=dict, description='Additional relationship properties (co_occurrence_count, etc.)')
    evidence_count: int = Field(0, description='Number of evidence spans for this relationship')
    strongest_evidence_quote: Optional[str] = Field(None, description='Text span of highest-confidence evidence')
    evidence_confidence_avg: Optional[float] = Field(None, description='Mean confidence of evidence spans')
class MentionRow(BaseModel):  # "One entity mention occurrence (one line in mentions.jsonl)."
    entity_id: str = Field(..., description='Entity ID this mention resolves to')
    document_id: str = Field(..., description='Document where the mention appears')
    section: Optional[str] = Field(None, description='Section within the document')
    start_offset: int = Field(..., description='Start character offset in document')
    end_offset: int = Field(..., description='End character offset in document')
    text_span: str = Field(..., description='Mention text')
    context: Optional[str] = Field(None, description='Surrounding context')
    confidence: float = Field(..., description='Extraction confidence')
    extraction_method: str = Field(..., description='How the mention was extracted (e.g. llm, rule_based, canonical_lookup)')
    created_at: str = Field(..., description='ISO 8601 creation timestamp')
class EvidenceRow(BaseModel):  # "One evidence span supporting a relationship (one line in evidence.jsonl)."
    relationship_key: str = Field(..., description='Composite key: subject_id:predicate:object_id')
    document_id: str = Field(..., description='Document where the evidence appears')
    section: Optional[str] = Field(None, description='Section within the document')
    start_offset: int = Field(..., description='Start character offset')
    end_offset: int = Field(..., description='End character offset')
    text_span: str = Field(..., description='Evidence quote text')
    confidence: float = Field(..., description='Confidence for this evidence')
    supports: bool = Field(True, description='True=supports, False=contradicts')
class BundleFile(BaseModel):  # "Reference to a file within the bundle."
    path: str = Field(..., description='Path to the file relative to the manifest')
    format: str = Field(..., description='File format (e.g., jsonl)')
class DocAssetRow(BaseModel):  # "Documentation asset row format for bundle doc_assets.jsonl files."
    path: str = Field(..., description='Path to the asset file relative to the bundle root')
    content_type: str = Field(..., description='MIME type of the asset (e.g., text/markdown, image/png)')
class BundleManifestV1(BaseModel):  # "Bundle manifest format matching the server contract."
    bundle_version: str = Field('v1', frozen=True, description='Bundle format version')
    bundle_id: str = Field(..., description='Unique bundle identifier (UUID)')
    domain: str = Field(..., description='Knowledge domain (e.g., sherlock, medical)')
    label: Optional[str] = Field(None, description='Human-readable bundle label')
    created_at: str = Field(..., description='ISO 8601 creation timestamp')
    entities: BundleFile = Field(..., description='Entities file information')
    relationships: BundleFile = Field(..., description='Relationships file information')
    doc_assets: Optional[BundleFile] = Field(None, description='Optional doc_assets.jsonl file listing documentation assets')
    mentions: Optional[BundleFile] = Field(None, description='Optional mentions.jsonl (entity provenance)')
    evidence: Optional[BundleFile] = Field(None, description='Optional evidence.jsonl (relationship evidence)')
    metadata: Dict[str, Any] = Field(default_factory=dict, description='Additional bundle metadata (description, counts, etc.)')

============================================================
# kgbundle/tests/__init__.py
============================================================


============================================================
# kgbundle/tests/test_models.py
============================================================
# imports: kgbundle.BundleFile, BundleManifestV1, EntityRow, EvidenceRow, MentionRow, RelationshipRow
class TestEntityRow:  # "Test EntityRow serialization and validation."
    def test_minimal_entity_row(self)
    def test_entity_row_roundtrip_json(self)
    def test_entity_row_provenance_fields(self)
class TestRelationshipRow:  # "Test RelationshipRow serialization."
    def test_minimal_relationship_row(self)
    def test_relationship_row_evidence_fields(self)
class TestMentionRow:  # "Test MentionRow (mentions.jsonl)."
    def test_mention_row_roundtrip(self)
class TestEvidenceRow:  # "Test EvidenceRow (evidence.jsonl)."
    def test_evidence_row_roundtrip(self)
class TestBundleManifestV1:  # "Test BundleManifestV1."
    def test_manifest_required_fields(self)
    def test_manifest_with_mentions_and_evidence(self)

============================================================
# kgraph/__init__.py
============================================================
# imports: kgraph.canonical_id.CanonicalIdCacheInterface, CanonicalIdLookupInterface, JsonFileCanonicalIdCache, check_entity_id_format, extract_canonical_id_from_entity, kgraph.ingest.IngestionOrchestrator, IngestionResult, kgschema.canonical_id.CanonicalId, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, kgschema.entity.BaseEntity, EntityMention, EntityStatus, PromotionConfig, kgschema.promotion.PromotionPolicy, kgschema.relationship.BaseRelationship
__all__ = ['BaseDocument', 'BaseEntity', 'BaseRelationship', 'CanonicalId', 'CanonicalIdCacheInterface', 'CanonicalIdLookupInterface', 'DomainSchema', 'EntityMention', 'EntityStatus', 'IngestionOrchestrator', 'IngestionResult', 'JsonFileCanonicalIdCache', 'PromotionConfig', 'PromotionPolicy', 'check_entity_id_format', 'extract_canonical_id_from_entity']
__version__ = '0.1.0'

============================================================
# kgraph/builders.py
============================================================
# imports: __future__.annotations, kgraph.clock.IngestionClock, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, Evidence, kgschema.entity.BaseEntity, EntityStatus, EntityMention, kgschema.relationship.BaseRelationship, kgschema.storage.EntityStorageInterface, pydantic.BaseModel, model_validator, ConfigDict, Field, typing.Any, Mapping, uuid
def _strip_nonempty(s: str, *, what: str) -> str
def _dedupe_preserve_order(items: tuple[str, ...]) -> tuple[str, ...]
class EntityBuilder(BaseModel):
    domain: DomainSchema
    clock: IngestionClock
    document: BaseDocument
    entity_storage: EntityStorageInterface | None = Field(default=None, repr=False)
    provisional_prefix: str = 'prov:'
    @model_validator(mode='after')
    def _validate(self) -> 'EntityBuilder'
    def _cls_for_type(self, entity_type: str) -> type[BaseEntity]
    def canonical(self, *, entity_type: str, entity_id: str, name: str, confidence: float=1.0, synonyms: tuple[str, ...]=(), canonical_ids: Mapping[str, str] | None=None, embedding: tuple[float, ...] | None=None, source: str='curated', metadata: dict[str, Any] | None=None, evidence: Evidence | None=None) -> BaseEntity
    def provisional_from_mention(self, *, entity_type: str, mention: EntityMention, confidence: float | None=None, entity_id: str | None=None, embedding: tuple[float, ...] | None=None, metadata: dict[str, Any] | None=None, evidence: Evidence | None=None) -> BaseEntity
    def _evidence_from_mention(self, mention: EntityMention) -> Evidence
    def _default_evidence(self, *, kind: str) -> Evidence
class RelationshipBuilder(BaseModel):
    domain: DomainSchema
    clock: IngestionClock
    document: BaseDocument
    entity_storage: EntityStorageInterface | None = Field(default=None, repr=False)
    @model_validator(mode='after')
    def _validate(self) -> 'RelationshipBuilder'
    def _cls_for_predicate(self, predicate: str) -> type[BaseRelationship]
    async def link(self, *, predicate: str, subject_id: str, object_id: str, confidence: float=0.8, source_documents: tuple[str, ...] | None=None, metadata: dict[str, Any] | None=None, evidence: Evidence | None=None) -> BaseRelationship  # "Create a relationship with structured provenance tracking."
    def _default_evidence(self, *, kind: str) -> Evidence

============================================================
# kgraph/canonical_id/__init__.py
============================================================
# imports: .helpers.check_entity_id_format, extract_canonical_id_from_entity, .json_cache.JsonFileCanonicalIdCache, .lookup.CanonicalIdLookupInterface, .models.CanonicalIdCacheInterface, kgschema.canonical_id.CanonicalId
__all__ = ['CanonicalId', 'CanonicalIdCacheInterface', 'CanonicalIdLookupInterface', 'JsonFileCanonicalIdCache', 'check_entity_id_format', 'extract_canonical_id_from_entity']

============================================================
# kgraph/canonical_id/helpers.py
============================================================
# imports: kgschema.canonical_id.CanonicalId, kgschema.entity.BaseEntity, typing.Optional
def extract_canonical_id_from_entity(entity: BaseEntity, priority_sources: Optional[list[str]]=None) -> Optional[CanonicalId]  # "Extract canonical ID from entity's canonical_ids dict."
def check_entity_id_format(entity: BaseEntity, format_patterns: dict[str, tuple[str, ...]]) -> Optional[CanonicalId]  # "Check if entity_id matches any known canonical ID format."

============================================================
# kgraph/canonical_id/json_cache.py
============================================================
# imports: .models.CanonicalId, CanonicalIdCacheInterface, json, kgraph.logging.setup_logging, os, pathlib.Path, typing.Optional
class JsonFileCanonicalIdCache(CanonicalIdCacheInterface):  # "JSON file-based implementation of CanonicalIdCacheInterface."
    def __init__(self, cache_file: Optional[Path]=None)  # "Initialize the JSON file-based cache."
    def load(self, tag: str) -> None  # "Load cache from JSON file."
    def _migrate_old_format(self, old_data: dict[str, str]) -> None  # "Migrate old cache format (dict[str, str]) to new format."
    def save(self, tag: str) -> None  # "Save cache to JSON file."
    def store(self, term: str, entity_type: str, canonical_id: CanonicalId) -> None  # "Store a canonical ID in the cache."
    def fetch(self, term: str, entity_type: str) -> Optional[CanonicalId]  # "Fetch a canonical ID from the cache."
    def mark_known_bad(self, term: str, entity_type: str) -> None  # "Mark a term as "known bad" (failed lookup, don't retry)."
    def is_known_bad(self, term: str, entity_type: str) -> bool  # "Check if a term is marked as "known bad"."
    def get_metrics(self) -> dict[str, int]  # "Get cache performance metrics."

============================================================
# kgraph/canonical_id/lookup.py
============================================================
# imports: .models.CanonicalId, abc.ABC, abstractmethod, typing.Optional
class CanonicalIdLookupInterface(ABC):  # "Abstract interface for looking up canonical IDs."
    @abstractmethod
    async def lookup(self, term: str, entity_type: str) -> Optional[CanonicalId]  # "Look up a canonical ID for a term."
    @abstractmethod
    def lookup_sync(self, term: str, entity_type: str) -> Optional[CanonicalId]  # "Synchronous version of lookup (for use in sync contexts)."

============================================================
# kgraph/canonical_id/models.py
============================================================
# imports: abc.ABC, abstractmethod, kgschema.canonical_id.CanonicalId, typing.Optional
class CanonicalIdCacheInterface(ABC):  # "Abstract interface for caching canonical ID lookups."
    @abstractmethod
    def load(self, tag: str) -> None  # "Load cache from storage."
    @abstractmethod
    def save(self, tag: str) -> None  # "Save cache to storage."
    @abstractmethod
    def store(self, term: str, entity_type: str, canonical_id: CanonicalId) -> None  # "Store a canonical ID in the cache."
    @abstractmethod
    def fetch(self, term: str, entity_type: str) -> Optional[CanonicalId]  # "Fetch a canonical ID from the cache."
    @abstractmethod
    def mark_known_bad(self, term: str, entity_type: str) -> None  # "Mark a term as "known bad" (failed lookup, don't retry)."
    @abstractmethod
    def is_known_bad(self, term: str, entity_type: str) -> bool  # "Check if a term is marked as "known bad"."
    @abstractmethod
    def get_metrics(self) -> dict[str, int]  # "Get cache performance metrics."
    def _normalize_key(self, term: str, entity_type: str) -> str  # "Normalize cache key for consistent lookups."

============================================================
# kgraph/clock.py
============================================================
# imports: __future__.annotations, datetime.datetime, pydantic.BaseModel, field_validator
class IngestionClock(BaseModel):
    now: datetime
    @field_validator('now')
    @classmethod
    def now_must_be_timezone_aware(cls, value: datetime) -> datetime

============================================================
# kgraph/context.py
============================================================
# imports: __future__.annotations, kgraph.builders.EntityBuilder, RelationshipBuilder, kgraph.clock.IngestionClock, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, Evidence, Provenance, pydantic.BaseModel, model_validator, ConfigDict
class IngestionContext(BaseModel):
    domain: DomainSchema
    clock: IngestionClock
    document: BaseDocument
    entities: EntityBuilder
    relationships: RelationshipBuilder
    @model_validator(mode='after')
    def domain_is_consistent(self) -> 'IngestionContext'
    def provenance(self, *, start_offset: int | None=None, end_offset: int | None=None, section: str | None=None) -> Provenance
    def evidence(self, *, kind: str='extracted', primary: Provenance | None=None, mentions: tuple[Provenance, ...]=(), source_documents: tuple[str, ...] | None=None, notes: dict[str, object] | None=None) -> Evidence

============================================================
# kgraph/export.py
============================================================
# imports: datetime.datetime, timezone, kgbundle.BundleFile, BundleManifestV1, DocAssetRow, EntityRow, RelationshipRow, kgraph.provenance.ProvenanceAccumulator, kgschema.storage.EntityStorageInterface, RelationshipStorageInterface, mimetypes, pathlib.Path, shutil, subprocess, typing.Any, Dict, List, Optional, Protocol, uuid
def get_git_hash() -> Optional[str]  # "Gets the current git commit hash in short format."
def _collect_doc_assets(docs_source: Path, bundle_path: Path) -> List[DocAssetRow]  # "Copies documentation assets from a source directory into the bundle."
def _entity_provenance_summary(accumulator: ProvenanceAccumulator) -> Dict[str, Dict[str, Any]]  # "Build per-entity provenance summary from accumulator mentions (first_seen, total_mentions, supporting_documents)."
def _relationship_evidence_summary(accumulator: ProvenanceAccumulator) -> Dict[str, Dict[str, Any]]  # "Build per-relationship evidence summary from accumulator evidence."
class GraphBundleExporter(Protocol):
    async def export_graph_bundle(self, entity_storage: EntityStorageInterface, relationship_storage: RelationshipStorageInterface, bundle_path: Path, domain: str, label: Optional[str]=None, docs: Optional[Path]=None, description: Optional[str]=None, provenance_accumulator: Optional[ProvenanceAccumulator]=None) -> None
class JsonlGraphBundleExporter:
    async def export_graph_bundle(self, entity_storage: EntityStorageInterface, relationship_storage: RelationshipStorageInterface, bundle_path: Path, domain: str, label: Optional[str]=None, docs: Optional[Path]=None, description: Optional[str]=None, provenance_accumulator: Optional[ProvenanceAccumulator]=None) -> None  # "Exports the graph content into a standardized JSONL bundle format."
default_exporter = JsonlGraphBundleExporter()
async def write_bundle(entity_storage: EntityStorageInterface, relationship_storage: RelationshipStorageInterface, bundle_path: Path, domain: str, label: Optional[str]=None, docs: Optional[Path]=None, description: Optional[str]=None, provenance_accumulator: Optional[ProvenanceAccumulator]=None) -> None  # "Writes a knowledge graph bundle to disk using the default exporter."

============================================================
# kgraph/ingest.py
============================================================
# imports: asyncio, datetime.datetime, timezone, examples.medlit.pipeline.canonical_urls.build_canonical_urls_from_dict, json, kgraph.canonical_id.CanonicalId, kgraph.logging.setup_logging, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, kgraph.pipeline.interfaces.DocumentParserInterface, EntityExtractorInterface, EntityResolverInterface, RelationshipExtractorInterface, kgraph.pipeline.streaming.DocumentChunkerInterface, StreamingEntityExtractorInterface, StreamingRelationshipExtractorInterface, kgraph.promotion.PromotionPolicy, kgraph.provenance.ProvenanceAccumulator, kgschema.domain.DomainSchema, kgschema.entity.BaseEntity, EntityStatus, kgschema.relationship.BaseRelationship, kgschema.storage.DocumentStorageInterface, EntityStorageInterface, RelationshipStorageInterface, numpy, pathlib.Path, pydantic.BaseModel, ConfigDict, sklearn.metrics.pairwise.cosine_similarity, typing.Any, Optional, Sequence
def _record_relationship_evidence(accumulator: Optional[ProvenanceAccumulator], rel: BaseRelationship) -> None  # "If accumulator and rel.evidence are set, record evidence rows."
class DocumentResult(BaseModel):  # "Result of processing a single document through the ingestion pipeline."
    document_id: str
    entities_extracted: int
    entities_new: int
    entities_existing: int
    relationships_extracted: int
    errors: tuple[str, ...] = ()
class IngestionResult(BaseModel):  # "Result of batch document ingestion."
    documents_processed: int
    documents_failed: int
    total_entities_extracted: int
    total_relationships_extracted: int
    document_results: tuple[DocumentResult, ...] = ()
    errors: tuple[str, ...] = ()
def _determine_canonical_id_source(canonical_id: str) -> str  # "Determine the canonical_ids dict key from canonical_id format."
class IngestionOrchestrator(BaseModel):  # "Orchestrates two-pass document ingestion for knowledge graph construction."
    domain: DomainSchema
    parser: DocumentParserInterface
    entity_extractor: EntityExtractorInterface
    entity_resolver: EntityResolverInterface
    relationship_extractor: RelationshipExtractorInterface
    embedding_generator: EmbeddingGeneratorInterface
    entity_storage: EntityStorageInterface
    relationship_storage: RelationshipStorageInterface
    document_storage: DocumentStorageInterface
    document_chunker: DocumentChunkerInterface | None = None
    streaming_entity_extractor: StreamingEntityExtractorInterface | None = None
    streaming_relationship_extractor: StreamingRelationshipExtractorInterface | None = None
    provenance_accumulator: Optional[ProvenanceAccumulator] = None
    async def extract_entities_from_document(self, raw_content: bytes, content_type: str, source_uri: str | None=None) -> DocumentResult  # "Runs the first pass of the ingestion pipeline on a single document."
    async def extract_relationships_from_document(self, raw_content: bytes, content_type: str, source_uri: str | None=None, document_id: str | None=None) -> DocumentResult  # "Runs the second pass of the ingestion pipeline on a single document."
    async def ingest_document(self, raw_content: bytes, content_type: str, source_uri: str | None=None) -> DocumentResult  # "Ingests a single document through the complete two-pass pipeline."
    async def ingest_batch(self, documents: Sequence[tuple[bytes, str, str | None]]) -> IngestionResult  # "Ingests a batch of documents using the two-pass pipeline."
    async def _lookup_canonical_ids_batch(self, policy: PromotionPolicy, candidates: list[BaseEntity], logger: Any) -> dict[str, CanonicalId | None]  # "Look up canonical IDs for all candidate entities in batches."
    async def _promote_single_entity(self, entity: BaseEntity, entity_canonical_id_map: dict[str, CanonicalId | None], policy: PromotionPolicy, logger: Any) -> BaseEntity | None | bool  # "Promote a single entity to canonical status."
    async def run_promotion(self, lookup=None) -> list[BaseEntity]  # "Promotes eligible provisional entities to canonical status."
    async def find_merge_candidates(self, similarity_threshold: float=0.95) -> list[tuple[BaseEntity, BaseEntity, float]]  # "Finds potential duplicate entities based on embedding similarity."
    async def merge_entities(self, source_ids: Sequence[str], target_id: str) -> bool  # "Merges one or more source entities into a single target entity."
    def _serialize_entity(self, entity: BaseEntity) -> dict[str, Any]  # "Serialize an entity to a JSON-compatible dictionary."
    def _serialize_relationship(self, rel: BaseRelationship) -> dict[str, Any]  # "Serialize a relationship to a JSON-compatible dictionary."
    async def export_entities(self, output_path: str | Path, include_provisional: bool=False) -> int  # "Exports entities from storage to a JSON file."
    async def export_document(self, document_id: str, output_path: str | Path) -> dict[str, int]  # "Exports data related to a single document to a JSON file."
    async def export_all(self, output_dir: str | Path) -> dict[str, Any]  # "Exports the entire graph into a directory of JSON files."

============================================================
# kgraph/logging.py
============================================================
# imports: inspect, logging, pprint.pformat, pydantic.BaseModel, typing.Any
class PprintLogger:  # "A logger wrapper that adds pprint support to standard logging methods."
    def __init__(self, logger: logging.Logger)
    def _format_message(self, msg: Any, pprint: bool=True) -> str  # "Format a message, optionally using pprint."
    def debug(self, msg: Any, *args, pprint: bool=True, **kwargs) -> None  # "Log a debug message with optional pprint formatting."
    def info(self, msg: Any, *args, pprint: bool=True, **kwargs) -> None  # "Log an info message with optional pprint formatting."
    def warning(self, msg: Any, *args, pprint: bool=True, **kwargs) -> None  # "Log a warning message with optional pprint formatting."
    def error(self, msg: Any, *args, pprint: bool=True, **kwargs) -> None  # "Log an error message with optional pprint formatting."
    def critical(self, msg: Any, *args, pprint: bool=True, **kwargs) -> None  # "Log a critical message with optional pprint formatting."
    def exception(self, msg: Any, *args, pprint: bool=True, **kwargs) -> None  # "Log an exception message with optional pprint formatting."
    def __getattr__(self, name: str) -> Any  # "Delegate any other attributes to the underlying logger."
def setup_logging(level: int=logging.INFO) -> PprintLogger  # "Set up logging and return a PprintLogger instance."

============================================================
# kgraph/pipeline/__init__.py
============================================================
# imports: kgraph.pipeline.caching.CachedEmbeddingGenerator, EmbeddingCacheConfig, EmbeddingsCacheInterface, FileBasedEmbeddingsCache, InMemoryEmbeddingsCache, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, kgraph.pipeline.interfaces.DocumentParserInterface, EntityExtractorInterface, EntityResolverInterface, RelationshipExtractorInterface, kgraph.pipeline.streaming.BatchingEntityExtractor, ChunkingConfig, DocumentChunk, DocumentChunkerInterface, StreamingEntityExtractorInterface, StreamingRelationshipExtractorInterface, WindowedDocumentChunker, WindowedRelationshipExtractor
__all__ = ['DocumentParserInterface', 'EntityExtractorInterface', 'EntityResolverInterface', 'RelationshipExtractorInterface', 'EmbeddingGeneratorInterface', 'DocumentChunkerInterface', 'StreamingEntityExtractorInterface', 'StreamingRelationshipExtractorInterface', 'DocumentChunk', 'ChunkingConfig', 'WindowedDocumentChunker', 'BatchingEntityExtractor', 'WindowedRelationshipExtractor', 'EmbeddingsCacheInterface', 'EmbeddingCacheConfig', 'InMemoryEmbeddingsCache', 'FileBasedEmbeddingsCache', 'CachedEmbeddingGenerator']

============================================================
# kgraph/pipeline/caching.py
============================================================
# imports: .embedding.EmbeddingGeneratorInterface, abc.ABC, abstractmethod, asyncio, collections.OrderedDict, json, logging, os, pathlib.Path, pydantic.BaseModel, Field, typing.Optional, Sequence
class EmbeddingCacheConfig(BaseModel):  # "Configuration for embedding caching strategies."
    max_cache_size: int = Field(10000, gt=0, description='Maximum number of embeddings in memory cache')
    cache_file: Path | None = Field(None, description='Path to persistent cache file')
    auto_save_interval: int = Field(100, ge=0, description='Auto-save every N updates (0 = manual only)')
    normalize_keys: bool = Field(True, description='Normalize cache keys for consistent lookups')
    normalize_collapse_whitespace: bool = Field(False, description='Collapse internal whitespace to single space when normalizing keys')
class EmbeddingsCacheInterface(ABC):  # "Abstract interface for caching text embeddings."
    @abstractmethod
    async def get(self, text: str) -> Optional[tuple[float, ...]]  # "Retrieve a cached embedding for the given text."
    @abstractmethod
    async def get_batch(self, texts: Sequence[str]) -> list[Optional[tuple[float, ...]]]  # "Retrieve multiple cached embeddings."
    @abstractmethod
    async def put(self, text: str, embedding: tuple[float, ...]) -> None  # "Store an embedding in the cache."
    @abstractmethod
    async def put_batch(self, texts: Sequence[str], embeddings: Sequence[tuple[float, ...]]) -> None  # "Store multiple embeddings efficiently."
    @abstractmethod
    async def clear(self) -> None  # "Clear all cached embeddings and reset statistics."
    @abstractmethod
    def get_stats(self) -> dict[str, int]  # "Get cache statistics."
    @abstractmethod
    async def save(self) -> None  # "Persist cache to storage (for persistent implementations)."
    @abstractmethod
    async def load(self) -> None  # "Load cache from storage (for persistent implementations)."
    def _normalize_key(self, text: str) -> str  # "Normalize cache key for consistent lookups."
class InMemoryEmbeddingsCache(EmbeddingsCacheInterface):  # "In-memory LRU cache for embeddings."
    def __init__(self, config: EmbeddingCacheConfig | None=None)  # "Initialize the in-memory cache."
    def _normalize_key(self, text: str) -> str  # "Normalize cache key; optionally collapse internal whitespace."
    async def get(self, text: str) -> Optional[tuple[float, ...]]  # "Retrieve embedding from memory cache."
    async def get_batch(self, texts: Sequence[str]) -> list[Optional[tuple[float, ...]]]  # "Retrieve multiple embeddings from cache."
    async def put(self, text: str, embedding: tuple[float, ...]) -> None  # "Store embedding in memory cache."
    async def put_batch(self, texts: Sequence[str], embeddings: Sequence[tuple[float, ...]]) -> None  # "Store multiple embeddings efficiently."
    async def clear(self) -> None  # "Clear all cached embeddings and reset statistics."
    def get_stats(self) -> dict[str, int]  # "Get cache statistics."
    async def save(self) -> None  # "No-op for in-memory cache (non-persistent)."
    async def load(self) -> None  # "No-op for in-memory cache (non-persistent)."
class FileBasedEmbeddingsCache(EmbeddingsCacheInterface):  # "Persistent file-based embeddings cache using JSON."
    def __init__(self, config: EmbeddingCacheConfig)  # "Initialize the file-based cache."
    def _normalize_key(self, text: str) -> str  # "Normalize cache key; optionally collapse internal whitespace."
    async def get(self, text: str) -> Optional[tuple[float, ...]]  # "Retrieve embedding from cache."
    async def get_batch(self, texts: Sequence[str]) -> list[Optional[tuple[float, ...]]]  # "Retrieve multiple embeddings from cache."
    async def put(self, text: str, embedding: tuple[float, ...]) -> None  # "Store embedding in cache."
    async def put_batch(self, texts: Sequence[str], embeddings: Sequence[tuple[float, ...]]) -> None  # "Store multiple embeddings efficiently."
    async def clear(self) -> None  # "Clear all cached embeddings and reset statistics."
    def get_stats(self) -> dict[str, int]  # "Get cache statistics."
    async def save(self) -> None  # "Persist cache to JSON file."
    async def load(self) -> None  # "Load cache from JSON file."
class CachedEmbeddingGenerator(EmbeddingGeneratorInterface):  # "Wraps an embedding generator with transparent caching."
    def __init__(self, base_generator: EmbeddingGeneratorInterface, cache: EmbeddingsCacheInterface)  # "Initialize the cached generator."
    @property
    def dimension(self) -> int  # "Return embedding dimension from base generator."
    async def generate(self, text: str) -> tuple[float, ...]  # "Generate embedding with caching."
    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]  # "Generate embeddings in batch with caching."
    async def save_cache(self) -> None  # "Persist cache to storage."
    def get_cache_stats(self) -> dict[str, int]  # "Get cache statistics."

============================================================
# kgraph/pipeline/embedding.py
============================================================
# imports: abc.ABC, abstractmethod, typing.Sequence
class EmbeddingGeneratorInterface(ABC):  # "Generate semantic vector embeddings for text."
    @property
    @abstractmethod
    def dimension(self) -> int  # "Return the dimensionality of generated embeddings."
    @abstractmethod
    async def generate(self, text: str) -> tuple[float, ...]  # "Generate an embedding vector for a single text string."
    @abstractmethod
    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]  # "Generate embeddings for multiple texts efficiently."

============================================================
# kgraph/pipeline/interfaces.py
============================================================
# imports: abc.ABC, abstractmethod, kgschema.document.BaseDocument, kgschema.entity.BaseEntity, EntityMention, kgschema.relationship.BaseRelationship, kgschema.storage.EntityStorageInterface, typing.Sequence
class DocumentParserInterface(ABC):  # "Parse raw documents into structured BaseDocument instances."
    @abstractmethod
    async def parse(self, raw_content: bytes, content_type: str, source_uri: str | None=None) -> BaseDocument  # "Parse raw content into a structured document."
class EntityExtractorInterface(ABC):  # "Extract entity mentions from documents (Pass 1 of ingestion)."
    @abstractmethod
    async def extract(self, document: BaseDocument) -> list[EntityMention]  # "Extract entity mentions from a document."
class EntityResolverInterface(ABC):  # "Resolve entity mentions to canonical or provisional entities."
    @abstractmethod
    async def resolve(self, mention: EntityMention, existing_storage: EntityStorageInterface) -> tuple[BaseEntity, float]  # "Resolve a single entity mention to an entity."
    @abstractmethod
    async def resolve_batch(self, mentions: Sequence[EntityMention], existing_storage: EntityStorageInterface) -> list[tuple[BaseEntity, float]]  # "Resolve multiple entity mentions efficiently."
class RelationshipExtractorInterface(ABC):  # "Extract relationships between entities from documents (Pass 2 of ingestion)."
    @abstractmethod
    async def extract(self, document: BaseDocument, entities: Sequence[BaseEntity]) -> list[BaseRelationship]  # "Extract relationships between entities in a document."

============================================================
# kgraph/pipeline/streaming.py
============================================================
# imports: .interfaces.EntityExtractorInterface, RelationshipExtractorInterface, abc.ABC, abstractmethod, datetime.datetime, timezone, kgschema.document.BaseDocument, kgschema.entity.BaseEntity, EntityMention, kgschema.relationship.BaseRelationship, logging, pydantic.BaseModel, Field, typing.AsyncIterator, Sequence
logger = logging.getLogger(__name__)
class DocumentChunk(BaseModel):  # "Represents a chunk/window of a document."
    content: str = Field(..., description='Text content of this chunk')
    start_offset: int = Field(..., ge=0, description='Starting character offset in original document')
    end_offset: int = Field(..., gt=0, description='Ending character offset in original document')
    chunk_index: int = Field(..., ge=0, description='Sequential index of this chunk (0-based)')
    document_id: str = Field(..., description='ID of the parent document')
    metadata: dict[str, str] = Field(default_factory=dict, description='Optional chunk-specific metadata')
class ChunkingConfig(BaseModel):  # "Configuration for document chunking strategies."
    chunk_size: int = Field(2000, gt=0, description='Target size of each chunk in characters')
    overlap: int = Field(200, ge=0, description='Number of characters to overlap between chunks')
    respect_boundaries: bool = Field(True, description='Respect sentence/paragraph boundaries when chunking')
    min_chunk_size: int = Field(500, gt=0, description='Minimum size for a chunk')
class DocumentChunkerInterface(ABC):  # "Interface for splitting documents into processable chunks."
    @abstractmethod
    async def chunk(self, document: BaseDocument) -> list[DocumentChunk]  # "Split a document into chunks."
    async def chunk_from_raw(self, raw_content: bytes, content_type: str, document_id: str, source_uri: str | None=None) -> list[DocumentChunk]  # "Chunk from raw bytes without parsing the full document (optional)."
class WindowedDocumentChunker(DocumentChunkerInterface):  # "Chunks documents into overlapping fixed-size windows."
    def __init__(self, config: ChunkingConfig | None=None)  # "Initialize the windowed chunker."
    async def chunk(self, document: BaseDocument) -> list[DocumentChunk]  # "Split document into overlapping fixed-size chunks."
class StreamingEntityExtractorInterface(ABC):  # "Interface for extracting entities from document chunks in streaming fashion."
    @abstractmethod
    def extract_streaming(self, chunks: Sequence[DocumentChunk]) -> AsyncIterator[list[EntityMention]]  # "Extract entities from document chunks, yielding results as they're processed."
def normalize_mention_key(name: str, entity_type: str) -> tuple[str, str]  # "Normalize mention key for deduplication across windows."
class BatchingEntityExtractor(StreamingEntityExtractorInterface):  # "Wraps an EntityExtractorInterface to provide streaming extraction with batching."
    def __init__(self, base_extractor: EntityExtractorInterface, batch_size: int=5, deduplicate: bool=True)  # "Initialize the batching extractor."
    async def extract_streaming(self, chunks: Sequence[DocumentChunk]) -> AsyncIterator[list[EntityMention]]  # "Extract entities from chunks, yielding results incrementally."
    def get_unique_mentions(self) -> list[EntityMention]  # "Get all unique mentions after deduplication."
class StreamingRelationshipExtractorInterface(ABC):  # "Interface for extracting relationships from document chunks in streaming fashion."
    @abstractmethod
    def extract_windowed(self, chunks: Sequence[DocumentChunk], entities: Sequence[BaseEntity], window_size: int=2000) -> AsyncIterator[list[BaseRelationship]]  # "Extract relationships from windowed chunks."
class WindowedRelationshipExtractor(StreamingRelationshipExtractorInterface):  # "Extracts relationships using sliding windows over document chunks."
    def __init__(self, base_extractor: RelationshipExtractorInterface)  # "Initialize the windowed relationship extractor."
    async def extract_windowed(self, chunks: Sequence[DocumentChunk], entities: Sequence[BaseEntity], window_size: int=2000) -> AsyncIterator[list[BaseRelationship]]  # "Extract relationships within overlapping windows."

============================================================
# kgraph/promotion.py
============================================================
# imports: kgschema.canonical_id.CanonicalId, kgschema.entity.BaseEntity, kgschema.promotion.PromotionPolicy, typing.Optional
class TodoPromotionPolicy(PromotionPolicy):  # "Placeholder promotion policy that raises NotImplementedError."
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]

============================================================
# kgraph/provenance.py
============================================================
# imports: kgbundle.EvidenceRow, MentionRow, typing.Optional
class ProvenanceAccumulator:  # "In-memory collector for entity mentions and relationship evidence."
    def __init__(self) -> None
    def add_mention(self, *, entity_id: str, document_id: str, section: Optional[str], start_offset: int, end_offset: int, text_span: str, context: Optional[str], confidence: float, extraction_method: str, created_at: str) -> None  # "Record one entity mention (one row in mentions.jsonl)."
    def add_evidence(self, *, relationship_key: str, document_id: str, section: Optional[str], start_offset: int, end_offset: int, text_span: str, confidence: float, supports: bool=True) -> None  # "Record one evidence span (one row in evidence.jsonl)."
    @property
    def mentions(self) -> list[MentionRow]
    @property
    def evidence(self) -> list[EvidenceRow]
    def mention_count(self) -> int
    def evidence_count(self) -> int

============================================================
# kgraph/query/__init__.py
============================================================


============================================================
# kgraph/storage/__init__.py
============================================================
# imports: kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.storage.DocumentStorageInterface, EntityStorageInterface, RelationshipStorageInterface
__all__ = ['EntityStorageInterface', 'RelationshipStorageInterface', 'DocumentStorageInterface', 'InMemoryEntityStorage', 'InMemoryRelationshipStorage', 'InMemoryDocumentStorage']

============================================================
# kgraph/storage/memory.py
============================================================
# imports: kgschema.document.BaseDocument, kgschema.entity.BaseEntity, EntityStatus, kgschema.relationship.BaseRelationship, kgschema.storage.DocumentStorageInterface, EntityStorageInterface, RelationshipStorageInterface, math, typing.Sequence
def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float  # "Compute cosine similarity between two embedding vectors."
class InMemoryEntityStorage(EntityStorageInterface):  # "In-memory entity storage using a dictionary keyed by entity_id."
    def __init__(self) -> None  # "Initialize an empty entity storage."
    async def add(self, entity: BaseEntity) -> str  # "Adds a new entity to the storage."
    async def get(self, entity_id: str) -> BaseEntity | None  # "Retrieves an entity by its ID."
    async def get_batch(self, entity_ids: Sequence[str]) -> list[BaseEntity | None]  # "Retrieves a batch of entities by their IDs."
    async def find_by_embedding(self, embedding: Sequence[float], threshold: float=0.8, limit: int=10) -> list[tuple[BaseEntity, float]]  # "Finds entities with embeddings similar to a given vector."
    async def find_by_name(self, name: str, entity_type: str | None=None, limit: int=10) -> list[BaseEntity]  # "Finds entities by a case-insensitive name or synonym match."
    async def find_provisional_for_promotion(self, min_usage: int, min_confidence: float) -> list[BaseEntity]  # "Finds provisional entities that meet promotion criteria."
    async def update(self, entity: BaseEntity) -> bool  # "Updates an existing entity in the storage."
    async def promote(self, entity_id: str, new_entity_id: str, canonical_ids: dict[str, str]) -> BaseEntity | None  # "Promotes a provisional entity to a canonical one."
    async def merge(self, source_ids: Sequence[str], target_id: str) -> bool  # "Merges multiple source entities into a single target entity."
    async def delete(self, entity_id: str) -> bool  # "Deletes an entity from storage by its ID."
    async def count(self) -> int  # "Returns the total number of entities in storage."
    async def list_all(self, status: str | None=None, limit: int=1000, offset: int=0) -> list[BaseEntity]  # "Lists entities from storage, with optional filtering and pagination."
class InMemoryRelationshipStorage(RelationshipStorageInterface):  # "In-memory relationship storage using triple keys."
    def __init__(self) -> None  # "Initialize an empty relationship storage."
    def _make_key(self, rel: BaseRelationship) -> tuple[str, str, str]  # "Create a dictionary key from a relationship's triple."
    async def add(self, relationship: BaseRelationship) -> str  # "Adds a new relationship to the storage."
    async def get_by_subject(self, subject_id: str, predicate: str | None=None) -> list[BaseRelationship]  # "Retrieves all relationships originating from a given subject."
    async def get_by_object(self, object_id: str, predicate: str | None=None) -> list[BaseRelationship]  # "Retrieves all relationships pointing to a given object."
    async def find_by_triple(self, subject_id: str, predicate: str, object_id: str) -> BaseRelationship | None  # "Finds a specific relationship by its full triple."
    async def update_entity_references(self, old_entity_id: str, new_entity_id: str) -> int  # "Updates all relationships that reference an old entity ID."
    async def get_by_document(self, document_id: str) -> list[BaseRelationship]  # "Retrieves all relationships sourced from a specific document."
    async def delete(self, subject_id: str, predicate: str, object_id: str) -> bool  # "Deletes a relationship from storage by its triple."
    async def count(self) -> int  # "Returns the total number of relationships in storage."
    async def list_all(self, limit: int=1000, offset: int=0) -> list[BaseRelationship]  # "Lists all relationships from storage, with optional pagination."
class InMemoryDocumentStorage(DocumentStorageInterface):  # "In-memory document storage using a dictionary keyed by document_id."
    def __init__(self) -> None  # "Initialize an empty document storage."
    async def add(self, document: BaseDocument) -> str  # "Adds a new document to the storage."
    async def get(self, document_id: str) -> BaseDocument | None  # "Retrieves a document by its ID."
    async def find_by_source(self, source_uri: str) -> BaseDocument | None  # "Finds a document by its source URI."
    async def list_ids(self, limit: int=100, offset: int=0) -> list[str]  # "Lists document IDs from storage, with optional pagination."
    async def delete(self, document_id: str) -> bool  # "Deletes a document from storage by its ID."
    async def count(self) -> int  # "Returns the total number of documents in storage."

============================================================
# kgschema/__init__.py
============================================================
# imports: kgschema.canonical_id.CanonicalId, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, kgschema.entity.BaseEntity, EntityMention, EntityStatus, PromotionConfig, kgschema.promotion.PromotionPolicy, kgschema.relationship.BaseRelationship, kgschema.storage.DocumentStorageInterface, EntityStorageInterface, RelationshipStorageInterface
__all__ = ['BaseDocument', 'BaseEntity', 'BaseRelationship', 'CanonicalId', 'DocumentStorageInterface', 'DomainSchema', 'EntityMention', 'EntityStatus', 'EntityStorageInterface', 'PromotionConfig', 'PromotionPolicy', 'RelationshipStorageInterface']
__version__ = '0.1.0'

============================================================
# kgschema/canonical_id.py
============================================================
# imports: pydantic.BaseModel, Field, typing.Optional
class CanonicalId(BaseModel):  # "Represents a canonical identifier from an authoritative source."
    id: str = Field(description="Canonical ID string from authoritative source (e.g., 'UMLS:C12345', 'MeSH:D000570')")
    url: Optional[str] = Field(default=None, description='URL to the authoritative source page for this ID, if available')
    synonyms: tuple[str, ...] = Field(default_factory=tuple, description='Alternative names/terms that map to this canonical ID')
    def __str__(self) -> str  # "String representation of the canonical ID."

============================================================
# kgschema/document.py
============================================================
# imports: abc.ABC, abstractmethod, datetime.datetime, pydantic.BaseModel, Field
class BaseDocument(ABC, BaseModel):  # "Abstract base class for documents in the knowledge graph."
    document_id: str = Field(description='Unique identifier for this document.')
    title: str | None = Field(default=None, description='Document title if available.')
    content: str = Field(description='Full text content of the document.')
    content_type: str = Field(description="MIME type or format indicator (e.g., 'text/plain', 'application/pdf').")
    source_uri: str | None = Field(default=None, description='Original source location (URL, file path, etc.).')
    created_at: datetime = Field(description='When the document was added to the system.')
    metadata: dict = Field(default_factory=dict, description='Domain-specific document metadata.')
    @abstractmethod
    def get_document_type(self) -> str  # "Return domain-specific document type."
    @abstractmethod
    def get_sections(self) -> list[tuple[str, str]]  # "Return document sections as (section_name, content) tuples."

============================================================
# kgschema/domain.py
============================================================
# imports: abc.ABC, abstractmethod, kgschema.document.BaseDocument, kgschema.entity.BaseEntity, PromotionConfig, kgschema.promotion.PromotionPolicy, kgschema.relationship.BaseRelationship, kgschema.storage.EntityStorageInterface, logging, pydantic.BaseModel, Field, model_validator
logger = logging.getLogger(__name__)
class ValidationIssue(BaseModel):  # "A structured validation error with location and diagnostic information."
    field: str = Field(description='The field or location that failed validation')
    message: str = Field(description='Human-readable description of the validation issue')
    value: str | None = Field(default=None, description='The invalid value (as string for display)')
    code: str | None = Field(default=None, description='Machine-readable error code')
class PredicateConstraint(BaseModel):  # "Defines the valid subject and object entity types for a predicate."
    subject_types: set[str] = Field(default_factory=set, description='Set of valid subject entity types')
    object_types: set[str] = Field(default_factory=set, description='Set of valid object entity types')
    @model_validator(mode='after')
    def check_not_empty(self) -> 'PredicateConstraint'
class Provenance(BaseModel):  # "Tracks the precise location of extracted information within a document."
    document_id: str = Field(description='Unique identifier of the source document')
    source_uri: str | None = Field(default=None, description='Optional URI/path to the original document')
    section: str | None = Field(default=None, description="Document section name (e.g., 'abstract', 'methods', 'results')")
    paragraph: int | None = Field(default=None, description='Paragraph number/index within the section (0-based)', ge=0)
    start_offset: int | None = Field(default=None, description='Character offset where the relevant text begins', ge=0)
    end_offset: int | None = Field(default=None, description='Character offset where the relevant text ends', ge=0)
class Evidence(BaseModel):
    kind: str
    source_documents: tuple[str, ...] = Field(min_length=1)
    primary: Provenance | None = None
    mentions: tuple[Provenance, ...] = ()
    notes: dict[str, object] = Field(default_factory=dict)
class DomainSchema(ABC):  # "Abstract schema definition for a knowledge domain."
    @property
    @abstractmethod
    def name(self) -> str  # "Return the unique identifier for this domain."
    @property
    @abstractmethod
    def entity_types(self) -> dict[str, type[BaseEntity]]  # "Return the registry of entity types for this domain."
    @property
    @abstractmethod
    def relationship_types(self) -> dict[str, type[BaseRelationship]]  # "Return the registry of relationship types for this domain."
    @property
    @abstractmethod
    def predicate_constraints(self) -> dict[str, PredicateConstraint]  # "Return a dictionary of predicate constraints for this domain."
    @property
    @abstractmethod
    def document_types(self) -> dict[str, type[BaseDocument]]  # "Return the registry of document types for this domain."
    @property
    def promotion_config(self) -> PromotionConfig  # "Return the configuration for promoting provisional entities."
    @abstractmethod
    def validate_entity(self, entity: BaseEntity) -> list[ValidationIssue]  # "Validate an entity against domain-specific rules."
    async def validate_relationship(self, relationship: BaseRelationship, entity_storage: EntityStorageInterface | None=None) -> bool  # "Validate a relationship against domain-specific rules."
    def get_valid_predicates(self, subject_type: str, object_type: str) -> list[str]  # "Return predicates valid between two entity types."
    @abstractmethod
    def get_promotion_policy(self, lookup=None) -> PromotionPolicy  # "Return the promotion policy for this domain."
    @property
    def evidence_model(self) -> type[Evidence]  # "Return the domain's version of Evidence"
    @property
    def provenance_model(self) -> type[Provenance]  # "Return the domain's version of Provenance"

============================================================
# kgschema/entity.py
============================================================
# imports: abc.ABC, abstractmethod, datetime.datetime, enum.Enum, pydantic.BaseModel, Field, model_validator
class EntityStatus(str, Enum):  # "Lifecycle status of an entity in the knowledge graph."
    ...
class PromotionConfig(BaseModel):  # "Configuration for promoting provisional entities to canonical status."
    min_usage_count: int = Field(default=3, ge=1, description='Minimum number of appearances before promotion is considered.')
    min_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description='Minimum confidence score required for promotion.')
    require_embedding: bool = Field(default=True, description='Whether an embedding must be present for promotion.')
class BaseEntity(ABC, BaseModel):  # "Abstract base class for all domain entities (knowledge graph nodes)."
    promotable: bool = Field(default=True, description='Whether this entity type can be promoted from provisional to canonical.')
    entity_id: str = Field(description='Domain-specific canonical ID or provisional UUID.')
    status: EntityStatus = Field(default=EntityStatus.PROVISIONAL, description='Whether entity is canonical or provisional.')
    name: str = Field(description='Primary name/label for the entity.')
    synonyms: tuple[str, ...] = Field(default=(), description='Alternative names or aliases for this entity.')
    embedding: tuple[float, ...] | None = Field(default=None, description='Semantic vector embedding for similarity operations.')
    canonical_ids: dict[str, str] = Field(default_factory=dict, description="Authoritative identifiers from various sources (e.g., {'dbpedia': 'uri', 'wikidata': 'Q123'}).")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description='Confidence in the entity resolution and its attributes.')
    usage_count: int = Field(default=0, ge=0, description='Number of times this entity has been referenced.')
    created_at: datetime = Field(description='Timestamp when the entity was first created.')
    source: str = Field(description='Origin indicator (e.g., document ID, extraction pipeline).')
    metadata: dict = Field(default_factory=dict, description='Domain-specific metadata.')
    @model_validator(mode='after')
    def _ensure_canonical_if_not_promotable(self) -> 'BaseEntity'
    @abstractmethod
    def get_entity_type(self) -> str  # "Return domain-specific entity type identifier."
class EntityMention(BaseModel):  # "A raw entity mention extracted from document text."
    text: str = Field(description='The exact text span mentioning the entity.')
    entity_type: str = Field(description='Domain-specific type classification.')
    start_offset: int = Field(ge=0, description='Character offset where mention starts in source text.')
    end_offset: int = Field(ge=0, description='Character offset where mention ends in source text.')
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description='Extraction confidence score.')
    context: str | None = Field(default=None, description='Surrounding text for disambiguation.')
    metadata: dict = Field(default_factory=dict, description='Domain-specific extraction metadata.')

============================================================
# kgschema/promotion.py
============================================================
# imports: abc.ABC, abstractmethod, kgschema.canonical_id.CanonicalId, kgschema.entity.BaseEntity, PromotionConfig, EntityStatus, typing.Optional
class PromotionPolicy(ABC):  # "Abstract base for domain-specific entity promotion policies."
    def __init__(self, config: PromotionConfig)
    def should_promote(self, entity: BaseEntity) -> bool  # "Check if entity meets promotion thresholds."
    @abstractmethod
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]  # "Return canonical ID for this entity, or None if not found."

============================================================
# kgschema/relationship.py
============================================================
# imports: __future__.annotations, abc.ABC, abstractmethod, datetime.datetime, pydantic.BaseModel, Field, typing.TYPE_CHECKING, Any
class BaseRelationship(ABC, BaseModel):  # "Abstract base class for relationships (edges) in the knowledge graph."
    subject_id: str = Field(description='Entity ID of the relationship subject.')
    predicate: str = Field(description='Relationship type (domain defines valid predicates).')
    object_id: str = Field(description='Entity ID of the relationship object.')
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description='Confidence score for this relationship.')
    source_documents: tuple[str, ...] = Field(default=(), description='Document IDs where this relationship was found.')
    evidence: Any = Field(default=None, description='Structured evidence including provenance (document, section, paragraph, offsets). Type: Evidence | None')
    created_at: datetime = Field(description='Timestamp when relationship was first created.')
    last_updated: datetime | None = Field(default=None, description='Timestamp of most recent update.')
    metadata: dict = Field(default_factory=dict, description="Domain-specific metadata. NOTE: Prefer using 'evidence' field for provenance tracking.")
    @abstractmethod
    def get_edge_type(self) -> str  # "Return domain-specific edge type category."

============================================================
# kgschema/storage.py
============================================================
# imports: abc.ABC, abstractmethod, kgschema.document.BaseDocument, kgschema.entity.BaseEntity, kgschema.relationship.BaseRelationship, typing.Sequence
class EntityStorageInterface(ABC):  # "Abstract interface for entity storage operations."
    @abstractmethod
    async def add(self, entity: BaseEntity) -> str  # "Store an entity and return its ID."
    @abstractmethod
    async def get(self, entity_id: str) -> BaseEntity | None  # "Retrieve an entity by its unique identifier."
    @abstractmethod
    async def get_batch(self, entity_ids: Sequence[str]) -> list[BaseEntity | None]  # "Retrieve multiple entities by ID in a single operation."
    @abstractmethod
    async def find_by_embedding(self, embedding: Sequence[float], threshold: float=0.8, limit: int=10) -> list[tuple[BaseEntity, float]]  # "Find entities semantically similar to the given embedding vector."
    @abstractmethod
    async def find_by_name(self, name: str, entity_type: str | None=None, limit: int=10) -> list[BaseEntity]  # "Find entities matching the given name or synonym."
    @abstractmethod
    async def find_provisional_for_promotion(self, min_usage: int, min_confidence: float) -> list[BaseEntity]  # "Find provisional entities eligible for promotion to canonical status."
    @abstractmethod
    async def update(self, entity: BaseEntity) -> bool  # "Update an existing entity's data."
    @abstractmethod
    async def promote(self, entity_id: str, new_entity_id: str, canonical_ids: dict[str, str]) -> BaseEntity | None  # "Promote a provisional entity to canonical status."
    @abstractmethod
    async def merge(self, source_ids: Sequence[str], target_id: str) -> bool  # "Merge multiple entities into a single target entity."
    @abstractmethod
    async def delete(self, entity_id: str) -> bool  # "Delete an entity from storage."
    @abstractmethod
    async def count(self) -> int  # "Return the total number of entities in storage."
    @abstractmethod
    async def list_all(self, status: str | None=None, limit: int=1000, offset: int=0) -> list[BaseEntity]  # "List entities with pagination and optional filtering."
class RelationshipStorageInterface(ABC):  # "Abstract interface for relationship (edge) storage operations."
    @abstractmethod
    async def add(self, relationship: BaseRelationship) -> str  # "Store a relationship and return an identifier."
    @abstractmethod
    async def get_by_subject(self, subject_id: str, predicate: str | None=None) -> list[BaseRelationship]  # "Get all relationships where the given entity is the subject."
    @abstractmethod
    async def get_by_object(self, object_id: str, predicate: str | None=None) -> list[BaseRelationship]  # "Get all relationships where the given entity is the object."
    @abstractmethod
    async def find_by_triple(self, subject_id: str, predicate: str, object_id: str) -> BaseRelationship | None  # "Find a specific relationship by its complete triple."
    @abstractmethod
    async def update_entity_references(self, old_entity_id: str, new_entity_id: str) -> int  # "Update all relationships referencing an entity to use a new ID."
    @abstractmethod
    async def get_by_document(self, document_id: str) -> list[BaseRelationship]  # "Get all relationships extracted from a specific document."
    @abstractmethod
    async def delete(self, subject_id: str, predicate: str, object_id: str) -> bool  # "Delete a specific relationship by its triple."
    @abstractmethod
    async def count(self) -> int  # "Return the total number of relationships in storage."
    @abstractmethod
    async def list_all(self, limit: int=1000, offset: int=0) -> list[BaseRelationship]  # "List all relationships with pagination."
class DocumentStorageInterface(ABC):  # "Abstract interface for document storage operations."
    @abstractmethod
    async def add(self, document: BaseDocument) -> str  # "Store a document and return its ID."
    @abstractmethod
    async def get(self, document_id: str) -> BaseDocument | None  # "Retrieve a document by its unique identifier."
    @abstractmethod
    async def find_by_source(self, source_uri: str) -> BaseDocument | None  # "Find a document by its source URI."
    @abstractmethod
    async def list_ids(self, limit: int=100, offset: int=0) -> list[str]  # "List document IDs with pagination."
    @abstractmethod
    async def delete(self, document_id: str) -> bool  # "Delete a document from storage."
    @abstractmethod
    async def count(self) -> int  # "Return the total number of documents in storage."

============================================================
# kgserver/chainlit/app.py
============================================================
# imports: asyncio, chainlit, chainlit.input_widget.Select, contextlib.AsyncExitStack, json, litellm, mcp.ClientSession, mcp.client.sse.sse_client, os, typing.Any, yaml
MCP_SSE_URL = os.environ.get('MCP_SSE_URL', 'http://localhost/mcp/sse')
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'anthropic').lower()
EXAMPLES_FILE = os.environ.get('EXAMPLES_FILE', 'examples.yaml')
SYSTEM_PROMPT = 'You are an expert assistant for a medical literature knowledge graph.\nYou have access to tools that can query a graph database of medical research papers,\nextract entities, find relationships, and surface evidence for clinical questions.\nAlways cite the papers you draw evidence from when possible.'
def get_litellm_model() -> dict[str, Any]  # "Return the model string and any extra kwargs for litellm.completion."
def load_examples() -> dict[str, str]  # "Load examples from YAML: {label: prompt_text}"
EXAMPLES: dict[str, str] = load_examples()
EXAMPLE_PLACEHOLDER = '— select an example —'
MCP_CONNECT_TIMEOUT = float(os.environ.get('MCP_CONNECT_TIMEOUT', '25'))
async def create_mcp_session() -> tuple[ClientSession, AsyncExitStack]
def mcp_tools_to_litellm(tools) -> list[dict]  # "Convert MCP tool descriptors to OpenAI-style tool dicts for litellm."
@cl.on_chat_start
async def on_chat_start()
@cl.on_settings_update
async def on_settings_update(settings: dict)
@cl.on_message
async def on_message(message: cl.Message)
@cl.on_chat_end
async def on_chat_end()
async def run_chat(user_text: str)
async def execute_tool_calls(tool_calls, mcp_session: ClientSession | None, status_msg: cl.Message) -> list[dict]  # "Run each tool call against the MCP server, return tool-result messages."

============================================================
# kgserver/main.py
============================================================
# imports: query.server.app
__all__ = ['app']

============================================================
# kgserver/mcp_main.py
============================================================
# imports: fastapi.FastAPI, mcp_server.mcp_server
sse_app = mcp_server.http_app(path='/sse', transport='sse')
app = FastAPI(title='Knowledge Graph MCP', version='0.1.0')
@app.get('/health')
async def health()  # "Health check for container/orchestration."

============================================================
# kgserver/mcp_server/__init__.py
============================================================
# imports: .server.mcp_server
__all__ = ['mcp_server']

============================================================
# kgserver/mcp_server/server.py
============================================================
# imports: collections.deque, contextlib.contextmanager, closing, fastmcp.FastMCP, query.graphql_schema.Query, query.storage_factory.get_engine, get_storage, strawberry, typing.Optional, Any
_graphql_schema = strawberry.Schema(query=Query)
mcp_server = FastMCP(name='knowledge-graph', instructions='Query interface for knowledge graph with tools for finding entities, relationships, and bundle information. Supports multiple knowledge domains with canonical IDs.')
@contextmanager
def _get_storage()  # "Context manager for getting a storage instance with proper lifecycle management."
@mcp_server.tool()
def get_entity(entity_id: str) -> dict | None  # "Retrieve a specific entity by its ID."
@mcp_server.tool()
def list_entities(limit: int=100, offset: int=0, entity_type: Optional[str]=None, name: Optional[str]=None, name_contains: Optional[str]=None, source: Optional[str]=None, status: Optional[str]=None) -> dict  # "List entities with pagination and optional filtering."
@mcp_server.tool()
def search_entities(query: str, entity_type: Optional[str]=None, limit: int=10) -> list[dict]  # "Search for entities by name (convenience wrapper around list_entities)."
@mcp_server.tool()
def get_relationship(subject_id: str, predicate: str, object_id: str) -> dict | None  # "Retrieve a specific relationship by its triple (subject, predicate, object)."
@mcp_server.tool()
def find_relationships(subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None, limit: int=100, offset: int=0) -> dict  # "Find relationships with pagination and optional filtering."
@mcp_server.tool()
def find_entities_within_hops(start_id: str, max_hops: int=3, entity_type: Optional[str]=None) -> dict  # "Find all entities within N hops of a starting entity using BFS traversal."
@mcp_server.tool()
def get_bundle_info() -> dict | None  # "Get bundle metadata for debugging and provenance."
@mcp_server.tool()
def graphql_query(query: str, variables: Optional[dict[str, Any]]=None) -> dict  # "Run an arbitrary GraphQL query against the knowledge graph."

============================================================
# kgserver/query/__init__.py
============================================================


============================================================
# kgserver/query/bundle_loader.py
============================================================
# imports: json, kgbundle.BundleManifestV1, logging, os, pathlib.Path, query.graphql_examples.load_examples, shutil, sqlmodel.Session, SQLModel, delete, storage.backends.postgres.PostgresStorage, storage.backends.sqlite.SQLiteStorage, storage.interfaces.StorageInterface, storage.models.bundle.Bundle, storage.models.entity.Entity, storage.models.relationship.Relationship, subprocess, sys, tempfile, zipfile
logger = logging.getLogger()
ch = logging.StreamHandler(sys.stdout)
FORMAT = '%(levelname)s:     %(asctime)s - %(pathname)s:%(lineno)d - %(message)s'
formatter = logging.Formatter(FORMAT)
def load_bundle_at_startup(engine, db_url: str) -> None  # "Load a bundle at server startup if BUNDLE_PATH is set."
def _load_from_zip(engine, db_url: str, zip_path: Path) -> None  # "Extract and load a bundle from a ZIP file."
def _load_from_directory(engine, db_url: str, bundle_dir: Path) -> None  # "Load a bundle from a directory."
def _find_manifest(search_dir: Path) -> Path | None  # "Find manifest.json in a directory (possibly in a subdirectory)."
def _get_docs_destination_path(asset_path: str, app_docs: Path) -> Path | None  # "Determine the destination path for a documentation asset."
def _process_single_doc_asset(line: str, bundle_dir: Path, app_docs: Path) -> bool  # "Process a single documentation asset entry."
def _load_doc_assets(bundle_dir: Path, manifest: BundleManifestV1) -> None  # "Load documentation assets from doc_assets.jsonl into docs directory."
def _build_mkdocs_if_present()  # "Build MkDocs documentation if mkdocs.yml exists and site is not already prebuilt."
def _initialize_storage(session: Session, db_url: str) -> StorageInterface  # "Initialize and return the appropriate StorageInterface."
def _handle_force_reload(session: Session, bundle_id: str, storage: StorageInterface) -> bool  # "Handle force reload logic, returning True if bundle should be skipped."
def _do_load(engine, db_url: str, bundle_dir: Path, manifest_path: Path) -> None  # "Actually load the bundle into storage."

============================================================
# kgserver/query/graph_traversal.py
============================================================
# imports: pydantic.BaseModel, Field, storage.interfaces.StorageInterface, typing.Any, Optional
MAX_HOPS = 5
MAX_NODES_LIMIT = 2000
MAX_EDGES_LIMIT = 10000
DEFAULT_MAX_NODES = 500
class GraphNode(BaseModel):  # "D3-compatible node representation."
    id: str = Field(description='Entity ID (used by D3 for linking)')
    label: str = Field(description='Display label (name or ID fallback)')
    entity_type: str = Field(description='Entity type for styling')
    properties: dict[str, Any] = Field(default_factory=dict, description='Full entity data for detail panel')
class GraphEdge(BaseModel):  # "D3-compatible edge representation."
    source: str = Field(description='Subject entity ID')
    target: str = Field(description='Object entity ID')
    label: str = Field(description='Human-readable predicate')
    predicate: str = Field(description='Raw predicate value')
    properties: dict[str, Any] = Field(default_factory=dict, description='Full relationship data for detail panel')
class SubgraphResponse(BaseModel):  # "Response format for graph visualization."
    nodes: list[GraphNode] = Field(description='Nodes in the subgraph')
    edges: list[GraphEdge] = Field(description='Edges in the subgraph')
    center_id: Optional[str] = Field(default=None, description='Focal entity ID (if subgraph mode)')
    hops: int = Field(description='Depth traversed')
    truncated: bool = Field(default=False, description='True if max_nodes limit was reached')
    total_entities: int = Field(description='Total entities in full graph')
    total_relationships: int = Field(description='Total relationships in full graph')
def _entity_to_node(entity) -> GraphNode  # "Convert a storage Entity to a GraphNode."
def _relationship_to_edge(rel) -> GraphEdge  # "Convert a storage Relationship to a GraphEdge."
def extract_subgraph(storage: StorageInterface, center_id: str, hops: int=2, max_nodes: int=DEFAULT_MAX_NODES) -> SubgraphResponse  # "Extract a subgraph centered on a given entity using BFS."
def extract_full_graph(storage: StorageInterface, max_nodes: int=DEFAULT_MAX_NODES) -> SubgraphResponse  # "Extract the entire graph (up to max_nodes)."

============================================================
# kgserver/query/graphql_examples.py
============================================================
# imports: logging, pathlib.Path, yaml
logger = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent
_DEFAULT_PATH = _HERE / 'graphql_examples.yml'
EXAMPLE_QUERIES: dict[str, str] = {}
DEFAULT_QUERY: str = ''
def load_examples(path: Path | None=None) -> None  # "Load (or reload) example queries from a YAML file."
def get_examples() -> dict[str, str]  # "Return the current example queries dict."
def get_default_query() -> str  # "Return the current default query string."

============================================================
# kgserver/query/graphql_schema.py
============================================================
# imports: datetime.datetime, logging, os, strawberry, strawberry.scalars.JSON, strawberry.types.Info, typing.List, Optional
logger = logging.getLogger(__name__)
MAX_LIMIT = int(os.getenv('GRAPHQL_MAX_LIMIT', '100'))
@strawberry.type
class Entity:  # "Generic entity GraphQL type."
    id: strawberry.ID = strawberry.field(name='id')
    entity_id: strawberry.ID = strawberry.field(name='entityId')
    entity_type: str = strawberry.field(name='entityType')
    name: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    usage_count: Optional[int] = strawberry.field(name='usageCount', default=None)
    source: Optional[str] = None
    canonical_url: Optional[str] = strawberry.field(name='canonicalUrl', default=None)
    synonyms: List[str] = strawberry.field(default_factory=list)
    properties: Optional[JSON] = None
@strawberry.type
class Relationship:  # "Generic relationship GraphQL type."
    id: strawberry.ID = strawberry.field(name='id')
    subject_id: strawberry.ID = strawberry.field(name='subjectId')
    predicate: str
    object_id: strawberry.ID = strawberry.field(name='objectId')
    confidence: Optional[float] = None
    source_documents: List[str] = strawberry.field(name='sourceDocuments', default_factory=list)
    properties: Optional[JSON] = None
@strawberry.type
class EntityPage:  # "Paginated result for entities."
    items: List[Entity]
    total: int
    limit: int
    offset: int
@strawberry.type
class RelationshipPage:  # "Paginated result for relationships."
    items: List[Relationship]
    total: int
    limit: int
    offset: int
@strawberry.input
class EntityFilter:  # "Filter criteria for entity queries."
    entity_type: Optional[str] = None
    name: Optional[str] = None
    name_contains: Optional[str] = None
    source: Optional[str] = None
    status: Optional[str] = None
@strawberry.input
class RelationshipFilter:  # "Filter criteria for relationship queries."
    subject_id: Optional[strawberry.ID] = None
    object_id: Optional[strawberry.ID] = None
    predicate: Optional[str] = None
@strawberry.type
class BundleInfo:  # "Bundle metadata for debugging and provenance."
    id: strawberry.ID = strawberry.field(name='id')
    bundle_id: str = strawberry.field(name='bundleId')
    domain: str
    created_at: Optional[datetime] = strawberry.field(name='createdAt', default=None)
    metadata: Optional[JSON] = None
@strawberry.type
class Query:
    @strawberry.field
    def entity(self, info: Info, id: str) -> Optional[Entity]  # "Retrieve a single entity by its ID."
    @strawberry.field
    def entities(self, info: Info, limit: int=100, offset: int=0, filter: Optional[EntityFilter]=None) -> EntityPage  # "List entities with pagination and optional filtering."
    @strawberry.field
    def relationship(self, info: Info, subject_id: strawberry.ID, predicate: str, object_id: strawberry.ID) -> Optional[Relationship]  # "Retrieve a single relationship by its triple."
    @strawberry.field
    def relationships(self, info: Info, limit: int=100, offset: int=0, filter: Optional[RelationshipFilter]=None) -> RelationshipPage  # "Find relationships with pagination and optional filtering."
    @strawberry.field
    def bundle(self, info: Info) -> Optional[BundleInfo]  # "Get bundle metadata for debugging and provenance."

============================================================
# kgserver/query/routers/graph_api.py
============================================================
# imports: ..graph_traversal.GraphNode, GraphEdge, SubgraphResponse, extract_subgraph, extract_full_graph, MAX_HOPS, MAX_NODES_LIMIT, DEFAULT_MAX_NODES, ..storage_factory.get_storage, fastapi.APIRouter, Depends, HTTPException, Query, pydantic.BaseModel, Field, storage.interfaces.StorageInterface, typing.Optional
class SearchResult(BaseModel):  # "A single entity search result."
    entity_id: str = Field(description="The entity's unique ID")
    name: str = Field(description='Display name')
    entity_type: str = Field(description='Entity type (disease, drug, gene, etc.)')
class SearchResponse(BaseModel):  # "Response from entity search."
    results: list[SearchResult] = Field(description='Matching entities')
    total: int = Field(description='Total matches (may exceed returned results)')
    query: str = Field(description='The search query')
router = APIRouter(prefix='/api/v1/graph', tags=['Graph Visualization'])
@router.get('/search', response_model=SearchResponse, summary='Search entities by name', description='\nSearch for entities by partial name match. Returns up to `limit` results\nsorted by relevance (exact matches first, then by usage count).\n\nUse this to find entity IDs for the subgraph endpoint.\n')
async def search_entities(q: str=Query(..., min_length=1, description='Search query (searches entity names, case-insensitive)'), limit: int=Query(default=20, ge=1, le=100, description='Maximum number of results to return'), entity_type: Optional[str]=Query(default=None, description="Filter by entity type (e.g., 'disease', 'drug', 'gene')"), storage: StorageInterface=Depends(get_storage)) -> SearchResponse  # "Search for entities by name."
@router.get('/subgraph', response_model=SubgraphResponse, summary='Get a subgraph for visualization', description='\nRetrieve a subgraph suitable for D3.js force-directed graph visualization.\n\n**Modes:**\n- **Subgraph mode** (default): BFS traversal from `center_id` for `hops` levels\n- **Full graph mode**: Set `include_all=true` to get entire graph (up to `max_nodes`)\n\n**Response format:**\n- `nodes`: Array of nodes with `id`, `label`, `entity_type`, and `properties`\n- `edges`: Array of edges with `source`, `target`, `label`, `predicate`, and `properties`\n- `truncated`: True if the result was limited by `max_nodes`\n')
async def get_subgraph(center_id: Optional[str]=Query(default=None, description='Entity ID to center the subgraph on (required unless include_all=true)'), hops: int=Query(default=2, ge=1, le=MAX_HOPS, description=f'Number of hops from center entity (1-{MAX_HOPS})'), max_nodes: int=Query(default=DEFAULT_MAX_NODES, ge=1, le=MAX_NODES_LIMIT, description=f'Maximum number of nodes to return (1-{MAX_NODES_LIMIT})'), include_all: bool=Query(default=False, description='If true, return entire graph instead of subgraph around center_id'), storage: StorageInterface=Depends(get_storage)) -> SubgraphResponse  # "Retrieve a subgraph for visualization."
@router.get('/node/{entity_id}', response_model=GraphNode, summary='Get details for a single node', description='Retrieve full details for a single entity as a graph node.')
async def get_node_details(entity_id: str, storage: StorageInterface=Depends(get_storage)) -> GraphNode  # "Get full details for a single node."
@router.get('/edge', response_model=GraphEdge, summary='Get details for a single edge', description='Retrieve full details for a single relationship as a graph edge.')
async def get_edge_details(subject_id: str=Query(..., description='Subject entity ID'), predicate: str=Query(..., description='Relationship predicate'), object_id: str=Query(..., description='Object entity ID'), storage: StorageInterface=Depends(get_storage)) -> GraphEdge  # "Get full details for a single edge."
class MentionsResponse(BaseModel):  # "Mentions (provenance) for an entity."
    mentions: list[dict] = Field(description='List of mention records from bundle provenance')
@router.get('/entity/{entity_id}/mentions', response_model=MentionsResponse, summary='Get mentions for an entity', description='Return all mention records (provenance) for the given entity.')
async def get_entity_mentions(entity_id: str, storage: StorageInterface=Depends(get_storage)) -> MentionsResponse  # "Get mention provenance for an entity."
class EvidenceResponse(BaseModel):  # "Evidence for a relationship."
    evidence: list[dict] = Field(description='List of evidence records for the relationship')
@router.get('/edge/evidence', response_model=EvidenceResponse, summary='Get evidence for an edge', description='Return evidence records for the relationship (subject_id, predicate, object_id).')
async def get_edge_evidence(subject_id: str=Query(..., description='Subject entity ID'), predicate: str=Query(..., description='Relationship predicate'), object_id: str=Query(..., description='Object entity ID'), storage: StorageInterface=Depends(get_storage)) -> EvidenceResponse  # "Get evidence for a relationship."

============================================================
# kgserver/query/routers/graphiql_custom.py
============================================================
# imports: ..graphql_examples.get_examples, get_default_query, fastapi.APIRouter, fastapi.responses.HTMLResponse, json
router = APIRouter()
def create_graphiql_html(graphql_endpoint: str='/graphql') -> str  # "Create custom GraphiQL HTML with example queries dropdown."
@router.get('/', response_class=HTMLResponse)
async def graphiql_interface()  # "Serve custom GraphiQL interface with example queries dropdown."

============================================================
# kgserver/query/routers/rest_api.py
============================================================
# imports: ..storage_factory.get_storage, fastapi.APIRouter, Depends, HTTPException, storage.interfaces.StorageInterface, storage.models.entity.Entity, storage.models.relationship.Relationship, typing.List, Optional
router = APIRouter(prefix='/api/v1')
@router.get('/entities/{entity_id}', response_model=Entity, summary='Get a single entity by its canonical ID')
async def get_entity_by_id(entity_id: str, storage: StorageInterface=Depends(get_storage))  # "Retrieve a single medical entity (e.g., Disease, Gene, Drug) by its"
@router.get('/entities', response_model=List[Entity], summary='List all entities')
async def list_entities(limit: int=100, offset: int=0, storage: StorageInterface=Depends(get_storage))  # "List all medical entities in the knowledge graph."
@router.get('/relationships', response_model=List[Relationship], summary='Find relationships between entities')
async def find_relationships(subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None, limit: int=100, storage: StorageInterface=Depends(get_storage))  # "Find relationships based on subject, predicate, or object."

============================================================
# kgserver/query/server.py
============================================================
# imports: .bundle_loader.load_bundle_at_startup, .graphql_schema.Query, .routers.graph_api, .routers.graphiql_custom, .routers.rest_api, .storage_factory.close_storage, get_engine, get_storage, chainlit.utils.mount_chainlit, contextlib.asynccontextmanager, fastapi.FastAPI, Depends, fastapi.staticfiles.StaticFiles, logging, os, pathlib.Path, storage.interfaces.StorageInterface, strawberry, strawberry.fastapi.GraphQLRouter, subprocess
logger = logging.getLogger()
result = subprocess.run(['uv', 'run', 'zensical', 'build'], check=False)
@asynccontextmanager
async def lifespan(app: FastAPI)  # "Application lifespan manager."
app = FastAPI(title='Medical Literature Knowledge Graph API', description='A read-only API for querying the medical literature knowledge graph.', version='0.1.0', lifespan=lifespan)
async def get_context(storage: StorageInterface=Depends(get_storage))
graphql_schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(graphql_schema, graphiql=False, context_getter=get_context)
@app.get('/health')
async def health_check()  # "Health check endpoint to verify that the server is running."
_graph_viz_static = Path(__file__).parent / 'static'
_chainlit_app = os.environ.get('CHAINLIT_APP_PATH') or next((p for p in [Path(__file__).resolve().parent.parent / 'chainlit' / 'app.py'] if p.exists()), None)
_mkdocs_site = Path(__file__).parent.parent / 'site'

============================================================
# kgserver/query/storage_factory.py
============================================================
# imports: os, sqlalchemy.create_engine, sqlmodel.Session, storage.backends.postgres.PostgresStorage, storage.backends.sqlite.SQLiteStorage, storage.interfaces.StorageInterface, typing.Generator
_engine = None
_db_url = None
def get_engine()  # "Returns a singleton instance of the SQLAlchemy engine and db_url."
def get_storage() -> Generator[StorageInterface, None, None]  # "FastAPI dependency that provides a storage instance with a request-scoped session."
def close_storage()  # "Closes the engine connection."

============================================================
# kgserver/storage/__init__.py
============================================================
__all__ = ['interfaces', 'backends', 'models']

============================================================
# kgserver/storage/backends/__init__.py
============================================================
__all__ = ['sqlite', 'postgres']

============================================================
# kgserver/storage/backends/postgres.py
============================================================
# imports: datetime.datetime, json, kgbundle.BundleManifestV1, EntityRow, EvidenceRow, MentionRow, RelationshipRow, pydantic.ValidationError, sqlalchemy.func, text, sqlmodel.Session, select, storage.interfaces.StorageInterface, storage.models.Bundle, BundleEvidence, Entity, Mention, Relationship, typing.Optional, Sequence
class PostgresStorage(StorageInterface):  # "PostgreSQL implementation of the storage interface."
    def __init__(self, session: Session)
    def load_bundle(self, bundle_manifest: BundleManifestV1, bundle_path: str) -> None  # "Load a data bundle into the storage."
    def _debug_print_sample_entities(self, entities_file: str) -> None  # "Print first few entities for debugging."
    def _load_entities(self, entities_file: str) -> None  # "Load entities from JSONL file."
    def _capture_entity_sample(self, entity_data: dict, entity_id: str, status: str) -> dict  # "Capture a sample entity for debugging."
    def _check_canonical_url_in_props(self, entity_data: dict, entity_id: str, status: str) -> tuple[bool, dict]  # "Check if canonical_url exists in properties and return it along with props."
    def _print_entity_loading_summary(self, canonical_url_count: int, canonical_entities: int, total_entities: int, sample_canonical_entity: Optional[dict], sample_entity_raw: Optional[dict], sample_with_url: Optional[dict], sample_without_url: Optional[dict]) -> None  # "Print summary of entity loading with debug information."
    def _print_entity_sample(self, title: str, sample: dict) -> None  # "Print a sample entity structure."
    def _load_relationships(self, relationships_file: str) -> None  # "Load relationships from JSONL file."
    def _normalize_entity(self, data: dict) -> dict  # "Normalize entity data, flattening metadata fields."
    def _normalize_relationship(self, data: dict) -> dict  # "Normalize relationship data, mapping field names."
    def is_bundle_loaded(self, bundle_id: str) -> bool  # "Check if a bundle with the given ID is already loaded."
    def record_bundle(self, bundle_manifest: BundleManifestV1) -> None  # "Record that a bundle has been loaded."
    def get_entity(self, entity_id: str) -> Optional[Entity]  # "Get an entity by its ID."
    def get_entities(self, limit: int=100, offset: int=0, entity_type: Optional[str]=None, name: Optional[str]=None, name_contains: Optional[str]=None, source: Optional[str]=None, status: Optional[str]=None) -> Sequence[Entity]  # "List entities with optional filtering."
    def count_entities(self, entity_type: Optional[str]=None, name: Optional[str]=None, name_contains: Optional[str]=None, source: Optional[str]=None, status: Optional[str]=None) -> int  # "Count entities matching filter criteria."
    def find_relationships(self, subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None, limit: Optional[int]=None, offset: Optional[int]=None) -> Sequence[Relationship]  # "Find relationships matching criteria."
    def count_relationships(self, subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None) -> int  # "Count relationships matching filter criteria."
    def get_relationship(self, subject_id: str, predicate: str, object_id: str) -> Optional[Relationship]  # "Get a relationship by its canonical triple (subject_id, predicate, object_id)."
    def get_relationships(self, limit: int=100, offset: int=0) -> Sequence[Relationship]  # "List all relationships."
    def get_bundle_info(self)  # "Get bundle metadata (latest bundle)."
    def get_mentions_for_entity(self, entity_id: str) -> Sequence[MentionRow]  # "Return all mention rows for the given entity (bundle provenance)."
    def get_evidence_for_relationship(self, subject_id: str, predicate: str, object_id: str) -> Sequence[EvidenceRow]  # "Return all evidence rows for the given relationship triple (bundle provenance)."
    def close(self) -> None  # "Close connections and clean up resources."

============================================================
# kgserver/storage/backends/sqlite.py
============================================================
# imports: datetime.datetime, json, kgbundle.BundleManifestV1, EvidenceRow, MentionRow, sqlalchemy.func, sqlmodel.Session, SQLModel, create_engine, select, storage.interfaces.StorageInterface, storage.models.Bundle, BundleEvidence, Entity, Mention, Relationship, typing.Optional, Sequence
class SQLiteStorage(StorageInterface):  # "SQLite implementation of the storage interface."
    def __init__(self, db_path: str, check_same_thread: bool=True)
    def add_entity(self, entity: Entity) -> None  # "Add a single entity to the storage."
    def add_relationship(self, relationship: Relationship) -> None  # "Add a single relationship to the storage."
    def load_bundle(self, bundle_manifest: BundleManifestV1, bundle_path: str) -> None  # "Load a data bundle into the storage."
    def _normalize_entity(self, data: dict) -> dict  # "Normalize entity data, flattening metadata fields."
    def _normalize_relationship(self, data: dict) -> dict  # "Normalize relationship data, mapping field names."
    def is_bundle_loaded(self, bundle_id: str) -> bool  # "Check if a bundle with the given ID is already loaded."
    def record_bundle(self, bundle_manifest: BundleManifestV1) -> None  # "Record that a bundle has been loaded."
    def get_entity(self, entity_id: str) -> Optional[Entity]  # "Get an entity by its ID."
    def get_entities(self, limit: int=100, offset: int=0, entity_type: Optional[str]=None, name: Optional[str]=None, name_contains: Optional[str]=None, source: Optional[str]=None, status: Optional[str]=None) -> Sequence[Entity]  # "List entities with optional filtering."
    def count_entities(self, entity_type: Optional[str]=None, name: Optional[str]=None, name_contains: Optional[str]=None, source: Optional[str]=None, status: Optional[str]=None) -> int  # "Count entities matching filter criteria."
    def find_relationships(self, subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None, limit: Optional[int]=None, offset: Optional[int]=None) -> Sequence[Relationship]  # "Find relationships matching criteria."
    def count_relationships(self, subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None) -> int  # "Count relationships matching filter criteria."
    def get_relationship(self, subject_id: str, predicate: str, object_id: str) -> Optional[Relationship]  # "Get a relationship by its canonical triple (subject_id, predicate, object_id)."
    def get_relationships(self, limit: int=100, offset: int=0) -> Sequence[Relationship]  # "List all relationships."
    def get_bundle_info(self)  # "Get bundle metadata (latest bundle)."
    def get_mentions_for_entity(self, entity_id: str) -> Sequence[MentionRow]  # "Return all mention rows for the given entity (bundle provenance)."
    def get_evidence_for_relationship(self, subject_id: str, predicate: str, object_id: str) -> Sequence[EvidenceRow]  # "Return all evidence rows for the given relationship triple (bundle provenance)."
    def close(self) -> None  # "Close connections and clean up resources."

============================================================
# kgserver/storage/interfaces.py
============================================================
# imports: .models.entity.Entity, .models.relationship.Relationship, abc.ABC, abstractmethod, kgbundle.BundleManifestV1, kgbundle.EvidenceRow, MentionRow, typing.TYPE_CHECKING, Optional, Sequence
class StorageInterface(ABC):  # "Abstract interface for a knowledge graph storage backend."
    @abstractmethod
    def load_bundle(self, bundle_manifest: BundleManifestV1, bundle_path: str) -> None  # "Load a data bundle into the storage."
    @abstractmethod
    def is_bundle_loaded(self, bundle_id: str) -> bool  # "Check if a bundle with the given ID is already loaded."
    @abstractmethod
    def record_bundle(self, bundle_manifest: BundleManifestV1) -> None  # "Record that a bundle has been loaded."
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]  # "Get an entity by its ID."
    @abstractmethod
    def get_entities(self, limit: int=100, offset: int=0, entity_type: Optional[str]=None, name: Optional[str]=None, name_contains: Optional[str]=None, source: Optional[str]=None, status: Optional[str]=None) -> Sequence[Entity]  # "List entities with optional filtering."
    @abstractmethod
    def count_entities(self, entity_type: Optional[str]=None, name: Optional[str]=None, name_contains: Optional[str]=None, source: Optional[str]=None, status: Optional[str]=None) -> int  # "Count entities matching filter criteria."
    @abstractmethod
    def find_relationships(self, subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None, limit: Optional[int]=None, offset: Optional[int]=None) -> Sequence[Relationship]  # "Find relationships matching criteria."
    @abstractmethod
    def count_relationships(self, subject_id: Optional[str]=None, predicate: Optional[str]=None, object_id: Optional[str]=None) -> int  # "Count relationships matching filter criteria."
    @abstractmethod
    def get_relationship(self, subject_id: str, predicate: str, object_id: str) -> Optional[Relationship]  # "Get a relationship by its canonical triple (subject_id, predicate, object_id)."
    @abstractmethod
    def get_relationships(self, limit: int=100, offset: int=0) -> Sequence[Relationship]  # "List all relationships."
    @abstractmethod
    def get_bundle_info(self)  # "Get bundle metadata (latest bundle)."
    def get_mentions_for_entity(self, entity_id: str) -> Sequence['MentionRow']  # "Return all mention rows for the given entity (bundle provenance)."
    def get_evidence_for_relationship(self, subject_id: str, predicate: str, object_id: str) -> Sequence['EvidenceRow']  # "Return all evidence rows for the given relationship triple (bundle provenance)."
    @abstractmethod
    def close(self) -> None  # "Close connections and clean up resources."

============================================================
# kgserver/storage/models/__init__.py
============================================================
# imports: .bundle.Bundle, .bundle_evidence.BundleEvidence, .entity.Entity, .mention.Mention, .relationship.Relationship
__all__ = ['Bundle', 'BundleEvidence', 'Entity', 'Mention', 'Relationship']

============================================================
# kgserver/storage/models/bundle.py
============================================================
# imports: datetime.datetime, sqlmodel.Field, SQLModel
class Bundle(SQLModel):  # "Represents a loaded data bundle's metadata for idempotent tracking."
    bundle_id: str = Field(primary_key=True)
    domain: str
    created_at: datetime
    bundle_version: str

============================================================
# kgserver/storage/models/bundle_evidence.py
============================================================
# imports: sqlmodel.Field, SQLModel, typing.Optional
class BundleEvidence(SQLModel):  # "One evidence span for a relationship from a loaded bundle (evidence.jsonl)."
    id: Optional[int] = Field(default=None, primary_key=True)
    relationship_key: str = Field(index=True)
    document_id: str = Field(index=True)
    section: Optional[str] = Field(default=None)
    start_offset: int = Field()
    end_offset: int = Field()
    text_span: str = Field()
    confidence: float = Field()
    supports: bool = Field(default=True)

============================================================
# kgserver/storage/models/entity.py
============================================================
# imports: sqlmodel.Field, SQLModel, JSON, Column, typing.Optional, List, Any
class Entity(SQLModel):  # "A generic entity in the knowledge graph."
    entity_id: str = Field(primary_key=True)
    entity_type: str = Field(index=True)
    name: Optional[str] = Field(default=None, index=True)
    status: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    usage_count: Optional[int] = Field(default=None)
    source: Optional[str] = Field(default=None)
    canonical_url: Optional[str] = Field(default=None, description='URL to the authoritative source for this entity')
    synonyms: List[str] = Field(default=[], sa_column=Column(JSON))
    properties: dict[str, Any] = Field(default={}, sa_column=Column(JSON))

============================================================
# kgserver/storage/models/evidence.py
============================================================
# imports: datetime.datetime, sqlalchemy.DateTime, Column, ForeignKey, text, sqlalchemy.dialects.postgresql.JSONB, sqlmodel.Field, SQLModel, typing.Optional, Dict, Any
class Evidence(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    relationship_id: int = Field(sa_column=Column('relationship_id', ForeignKey('relationships.id', ondelete='CASCADE'), nullable=False))
    paper_id: int = Field(sa_column=Column('paper_id', ForeignKey('papers.id', ondelete='CASCADE'), nullable=False))
    evidence_type: str = Field(max_length=50)
    confidence_score: Optional[float] = Field(default=None)
    metadata_: Dict[str, Any] = Field(default_factory=dict, sa_column=Column('metadata_', JSONB, server_default=text("'{}'::jsonb"), nullable=False))
    created_at: datetime = Field(sa_column=Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False))

============================================================
# kgserver/storage/models/mention.py
============================================================
# imports: sqlmodel.Field, SQLModel, typing.Optional
class Mention(SQLModel):  # "One entity mention from a loaded bundle (mentions.jsonl)."
    id: Optional[int] = Field(default=None, primary_key=True)
    entity_id: str = Field(index=True)
    document_id: str = Field(index=True)
    section: Optional[str] = Field(default=None)
    start_offset: int = Field()
    end_offset: int = Field()
    text_span: str = Field()
    context: Optional[str] = Field(default=None)
    confidence: float = Field()
    extraction_method: str = Field()
    created_at: str = Field()

============================================================
# kgserver/storage/models/paper.py
============================================================
# imports: datetime.datetime, sqlalchemy.dialects.postgresql.JSONB, sqlalchemy.text, DateTime, sqlmodel.Field, SQLModel, Column, typing.Optional
class Paper(SQLModel):
    id: str = Field(primary_key=True, description='Canonical paper ID (e.g. PMC ID)')
    title: str = Field(index=True)
    authors: Optional[str] = None
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = Field(default=None, index=True)
    pubmed_id: Optional[str] = Field(default=None, index=True)
    entity_count: int = Field(default=0)
    relationship_count: int = Field(default=0)
    extraction_provenance_json: Optional[dict] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    metadata_json: Optional[dict] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime(timezone=False), nullable=False, server_default=text('CURRENT_TIMESTAMP')))
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime(timezone=False), nullable=False, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP')))

============================================================
# kgserver/storage/models/relationship.py
============================================================
# imports: sqlmodel.Field, SQLModel, JSON, Column, UniqueConstraint, typing.Optional, List, Any, uuid.UUID, uuid4
class Relationship(SQLModel):  # "A generic relationship in the knowledge graph."
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    subject_id: str = Field(index=True)
    predicate: str = Field(index=True)
    object_id: str = Field(index=True)
    confidence: Optional[float] = Field(default=None)
    source_documents: List[str] = Field(default=[], sa_column=Column(JSON))
    properties: dict[str, Any] = Field(default={}, sa_column=Column(JSON))

============================================================
# kgserver/tests/conftest.py
============================================================
# imports: datetime.datetime, json, kgbundle.BundleManifestV1, kgbundle.BundleManifestV1, BundleFile, pytest, query.graphql_schema.Query, storage.backends.sqlite.SQLiteStorage, storage.models.Bundle, Entity, Relationship, strawberry
@pytest.fixture
def in_memory_storage()  # "Create an in-memory SQLite storage for testing."
@pytest.fixture
def sample_entities()  # "Create sample entities for testing."
@pytest.fixture
def sample_relationships()  # "Create sample relationships for testing."
@pytest.fixture
def populated_storage(in_memory_storage, sample_entities, sample_relationships)  # "Create storage with sample data."
@pytest.fixture
def graphql_context(populated_storage)  # "Create GraphQL context with populated storage."
@pytest.fixture
def graphql_schema()  # "Create GraphQL schema for testing."
@pytest.fixture
def sample_bundle()  # "Create a sample bundle for testing."
@pytest.fixture
def storage_with_bundle(populated_storage, sample_bundle)  # "Create storage with bundle metadata."
@pytest.fixture
def bundle_dir_with_provenance(tmp_path)  # "Create a bundle directory with entities, relationships, mentions, and evidence (for provenance API tests)."
@pytest.fixture
def storage_with_provenance_bundle(tmp_path, bundle_dir_with_provenance)  # "SQLite storage with a bundle loaded that includes mentions and evidence."

============================================================
# kgserver/tests/test_bundle_loader.py
============================================================
# imports: datetime.datetime, json, pytest, query.bundle_loader._load_from_directory, _load_from_zip, query.graphql_examples.load_examples, get_examples, get_default_query, sqlalchemy.create_engine, sqlmodel.SQLModel, storage.backends.sqlite.SQLiteStorage, yaml, zipfile
@pytest.fixture
def sample_manifest_data()  # "Create sample manifest data."
@pytest.fixture
def bundle_directory(sample_manifest_data, tmp_path)  # "Create a temporary bundle directory with manifest and data files."
@pytest.fixture
def bundle_zip(bundle_directory, tmp_path)  # "Create a ZIP file from bundle directory."
@pytest.fixture
def test_engine()  # "Create a test SQLAlchemy engine."
class TestLoadFromDirectory:  # "Test _load_from_directory() function."
    def test_load_from_directory_success(self, bundle_directory, test_engine)  # "Test successfully loading bundle from directory."
    def test_load_from_directory_no_manifest(self, tmp_path, test_engine)  # "Test loading from directory without manifest."
class TestLoadFromZip:  # "Test _load_from_zip() function."
    def test_load_from_zip_success(self, bundle_zip, test_engine)  # "Test successfully loading bundle from ZIP."
    def test_load_from_zip_no_manifest(self, tmp_path, test_engine)  # "Test loading ZIP without manifest."
class TestBundleGraphqlExamples:  # "Test that a bundle's graphql_examples.yml replaces the default examples."
    def test_bundle_examples_override(self, bundle_directory, test_engine)  # "When a bundle contains graphql_examples.yml, those examples"
    def test_no_bundle_examples_keeps_defaults(self, bundle_directory, test_engine)  # "When a bundle does NOT contain graphql_examples.yml, the built-in"

============================================================
# kgserver/tests/test_find_entities_within_hops.py
============================================================
# imports: contextlib.contextmanager, mcp_server.server.find_entities_within_hops, pytest, unittest.mock.patch
@pytest.fixture
def mock_storage(populated_storage)  # "Provide populated storage to the MCP tool via _get_storage."
def _call_tool(start_id: str, max_hops: int=3, entity_type=None)  # "Call the underlying MCP tool function."
def test_find_entities_within_hops_structure(mock_storage)  # "Result has start_id, results_by_hop dict, and hop_distance matches key."
def test_find_entities_within_hops_from_entity_1(mock_storage)  # "From test:entity:1, one hop gives entity 2 and 3 (via edges 1->2, 1->3)."
def test_find_entities_within_hops_entity_type_filter(mock_storage)  # "Filter by entity_type returns only matching entities."

============================================================
# kgserver/tests/test_graph_api.py
============================================================
# imports: fastapi.FastAPI, fastapi.testclient.TestClient, pytest, query.graph_traversal.SubgraphResponse, extract_subgraph, extract_full_graph, query.routers.graph_api, query.storage_factory.get_storage, storage.backends.sqlite.SQLiteStorage, storage.models.Entity, Relationship
@pytest.fixture
def app()  # "Create FastAPI app with Graph API router (module-level for use by all API test classes)."
class TestGraphTraversal:  # "Tests for BFS graph traversal logic."
    def test_extract_subgraph_single_hop(self, populated_storage)  # "Test extracting a subgraph with 1 hop from center."
    def test_extract_subgraph_two_hops(self, populated_storage)  # "Test extracting a subgraph with 2 hops."
    def test_extract_subgraph_nonexistent_center(self, populated_storage)  # "Test extracting subgraph with non-existent center returns empty."
    def test_extract_subgraph_respects_max_nodes(self, populated_storage)  # "Test that max_nodes limit is respected."
    def test_extract_full_graph(self, populated_storage)  # "Test extracting the full graph."
    def test_graph_node_structure(self, populated_storage)  # "Test that GraphNode has correct structure."
    def test_graph_edge_structure(self, populated_storage)  # "Test that GraphEdge has correct structure."
class TestGraphAPI:  # "Tests for graph visualization REST API."
    @pytest.fixture
    def file_storage(self, tmp_path, sample_entities, sample_relationships)  # "Create SQLite storage for thread-safe testing."
    @pytest.fixture
    def client(self, app, file_storage)  # "Create test client with storage dependency override."
    def test_get_subgraph_with_center(self, client)  # "Test GET /api/v1/graph/subgraph with center_id."
    def test_get_subgraph_include_all(self, client)  # "Test GET /api/v1/graph/subgraph with include_all=true."
    def test_get_subgraph_missing_center_id(self, client)  # "Test that missing center_id returns 400 when include_all is false."
    def test_get_node_details(self, client)  # "Test GET /api/v1/graph/node/{entity_id}."
    def test_get_node_details_not_found(self, client)  # "Test GET /api/v1/graph/node with non-existent entity."
    def test_get_edge_details(self, client)  # "Test GET /api/v1/graph/edge."
    def test_get_edge_details_not_found(self, client)  # "Test GET /api/v1/graph/edge with non-existent relationship."
    def test_hops_parameter_validation(self, client)  # "Test that hops parameter is validated."
    def test_max_nodes_parameter(self, client)  # "Test max_nodes parameter."
    def test_search_entities(self, client)  # "Test GET /api/v1/graph/search."
    def test_search_entities_no_results(self, client)  # "Test search with no matching entities."
    def test_search_entities_with_type_filter(self, client)  # "Test search with entity_type filter."
    def test_get_entity_mentions_empty_without_provenance(self, client)  # "GET /entity/{id}/mentions returns 200 and empty list when no mentions stored."
    def test_get_edge_evidence_empty_without_provenance(self, client)  # "GET /edge/evidence returns 200 and empty list when no evidence stored."
class TestGraphAPIProvenance:  # "Graph API includes provenance/evidence in node and edge payloads when present."
    @pytest.fixture
    def storage_with_provenance_properties(self, tmp_path)  # "Storage with entities and relationships that have provenance in properties."
    @pytest.fixture
    def client_provenance(self, app, storage_with_provenance_properties)  # "Test client with storage that has provenance in entity/relationship properties."
    def test_node_details_include_provenance(self, client_provenance)  # "GET /node/{id} includes first_seen_document, total_mentions, supporting_documents in properties."
    def test_edge_details_include_evidence_summary(self, client_provenance)  # "GET /edge includes evidence_count, strongest_evidence_quote, evidence_confidence_avg in properties."
    def test_subgraph_node_properties_include_provenance(self, client_provenance)  # "Subgraph nodes include provenance fields in properties when present."
    def test_subgraph_edge_properties_include_evidence(self, client_provenance)  # "Subgraph edges include evidence summary in properties when present."
class TestGraphAPIMentionsEvidenceEndpoints:  # "GET /entity/{id}/mentions and GET /edge/evidence return stored provenance."
    @pytest.fixture
    def client_with_mentions_evidence(self, app, storage_with_provenance_bundle)  # "Test client with storage that has mentions and evidence from a loaded bundle."
    def test_get_entity_mentions_returns_mentions(self, client_with_mentions_evidence)  # "GET /entity/{id}/mentions returns mention rows when bundle had mentions.jsonl."
    def test_get_edge_evidence_returns_evidence(self, client_with_mentions_evidence)  # "GET /edge/evidence returns evidence rows when bundle had evidence.jsonl."

============================================================
# kgserver/tests/test_graphql_schema.py
============================================================
# imports: query.graphql_schema
def execute_query(schema, query: str, context: dict)  # "Helper to execute a GraphQL query."
class TestEntityQueries:  # "Test entity-related GraphQL queries."
    def test_entity_by_id(self, graphql_schema, graphql_context)  # "Test retrieving a single entity by ID."
    def test_entity_not_found(self, graphql_schema, graphql_context)  # "Test querying for non-existent entity."
    def test_entities_pagination(self, graphql_schema, graphql_context)  # "Test entities query with pagination."
    def test_entities_pagination_offset(self, graphql_schema, graphql_context)  # "Test entities query with offset."
    def test_entities_filter_by_type(self, graphql_schema, graphql_context)  # "Test filtering entities by entity type."
    def test_entities_filter_by_name(self, graphql_schema, graphql_context)  # "Test filtering entities by exact name."
    def test_entities_filter_name_contains(self, graphql_schema, graphql_context)  # "Test filtering entities by name containing string."
    def test_entities_filter_by_source(self, graphql_schema, graphql_context)  # "Test filtering entities by source."
    def test_entities_filter_by_status(self, graphql_schema, graphql_context)  # "Test filtering entities by status."
    def test_entities_filter_combined(self, graphql_schema, graphql_context)  # "Test combining multiple filters."
    def test_entities_max_limit_enforcement(self, graphql_schema, graphql_context, monkeypatch)  # "Test that max limit is enforced."
class TestRelationshipQueries:  # "Test relationship-related GraphQL queries."
    def test_relationship_by_triple(self, graphql_schema, graphql_context)  # "Test retrieving a single relationship by triple."
    def test_relationship_not_found(self, graphql_schema, graphql_context)  # "Test querying for non-existent relationship."
    def test_relationships_pagination(self, graphql_schema, graphql_context)  # "Test relationships query with pagination."
    def test_relationships_filter_by_subject(self, graphql_schema, graphql_context)  # "Test filtering relationships by subject ID."
    def test_relationships_filter_by_object(self, graphql_schema, graphql_context)  # "Test filtering relationships by object ID."
    def test_relationships_filter_by_predicate(self, graphql_schema, graphql_context)  # "Test filtering relationships by predicate."
    def test_relationships_filter_combined(self, graphql_schema, graphql_context)  # "Test combining multiple relationship filters."
    def test_relationships_max_limit_enforcement(self, graphql_schema, graphql_context, monkeypatch)  # "Test that max limit is enforced for relationships."
class TestBundleQuery:  # "Test bundle introspection query."
    def test_bundle_query(self, graphql_schema, storage_with_bundle)  # "Test bundle introspection query."
    def test_bundle_query_no_bundle(self, graphql_schema, graphql_context)  # "Test bundle query when no bundle is loaded."
class TestFieldNaming:  # "Test that GraphQL field names use camelCase."
    def test_entity_camelcase_fields(self, graphql_schema, graphql_context)  # "Test that entity fields are camelCase in GraphQL."
    def test_relationship_camelcase_fields(self, graphql_schema, graphql_context)  # "Test that relationship fields are camelCase in GraphQL."
    def test_relationship_id_field(self, graphql_schema, graphql_context)  # "Test that relationship id field is exposed and is a string (UUID)."
class TestPaginationMetadata:  # "Test pagination metadata correctness."
    def test_entities_pagination_metadata(self, graphql_schema, graphql_context)  # "Test that pagination metadata is correct."
    def test_relationships_pagination_metadata(self, graphql_schema, graphql_context)  # "Test that relationship pagination metadata is correct."

============================================================
# kgserver/tests/test_mcp_graphql_tool.py
============================================================
# imports: contextlib.contextmanager, mcp_server.server.graphql_query, pytest, unittest.mock.patch
@pytest.fixture
def mock_storage(populated_storage)  # "Provide populated storage to the MCP tool via _get_storage."
def _call_tool(query: str, variables=None)  # "Call the underlying MCP tool function (FastMCP wraps it in FunctionTool)."
def test_graphql_query_returns_data_and_errors_shape(mock_storage)  # "graphql_query returns dict with 'data' and 'errors' keys."
def test_graphql_query_entity(mock_storage)  # "graphql_query can fetch an entity by id."
def test_graphql_query_entities_paginated(mock_storage)  # "graphql_query can list entities with pagination."
def test_graphql_query_returns_errors_on_invalid_query(mock_storage)  # "graphql_query returns errors in standard shape for invalid GraphQL."

============================================================
# kgserver/tests/test_rest_api.py
============================================================
# imports: fastapi.FastAPI, fastapi.testclient.TestClient, pytest, query.routers.rest_api, query.storage_factory.get_storage, storage.backends.sqlite.SQLiteStorage
@pytest.fixture
def app()  # "Create FastAPI app with REST API router."
@pytest.fixture
def file_storage(tmp_path, sample_entities, sample_relationships)  # "Create SQLite storage for thread-safe testing with FastAPI TestClient."
@pytest.fixture
def client(app, file_storage)  # "Create test client with storage dependency override."
class TestGetEntityById:  # "Test GET /api/v1/entities/{entity_id} endpoint."
    def test_get_existing_entity(self, client)  # "Test retrieving an existing entity."
    def test_get_nonexistent_entity(self, client)  # "Test retrieving a non-existent entity returns 404."
class TestListEntities:  # "Test GET /api/v1/entities endpoint."
    def test_list_entities_default(self, client)  # "Test listing entities with default parameters."
    def test_list_entities_with_limit(self, client)  # "Test listing entities with limit."
    def test_list_entities_with_offset(self, client)  # "Test listing entities with offset."
    def test_list_entities_empty_result(self, client)  # "Test listing entities with offset beyond available."
class TestFindRelationships:  # "Test GET /api/v1/relationships endpoint."
    def test_find_all_relationships(self, client)  # "Test finding all relationships."
    def test_find_relationships_by_subject(self, client)  # "Test filtering relationships by subject_id."
    def test_find_relationships_by_object(self, client)  # "Test filtering relationships by object_id."
    def test_find_relationships_by_predicate(self, client)  # "Test filtering relationships by predicate."
    def test_find_relationships_combined_filters(self, client)  # "Test filtering relationships with multiple filters."
    def test_find_relationships_with_limit(self, client)  # "Test limiting relationship results."
    def test_find_relationships_no_matches(self, client)  # "Test finding relationships with no matches."

============================================================
# kgserver/tests/test_storage_backends.py
============================================================
# imports: datetime.datetime, pytest, sqlmodel.Session, create_engine, SQLModel, sqlmodel.delete, storage.backends.postgres.PostgresStorage, storage.models.Entity, Relationship, Bundle
class TestSQLiteStorage:  # "Direct tests for SQLiteStorage."
    def test_get_entity_existing(self, in_memory_storage, sample_entities)  # "Test retrieving an existing entity."
    def test_get_entity_nonexistent(self, in_memory_storage)  # "Test retrieving non-existent entity."
    def test_get_entities_with_filters(self, in_memory_storage, sample_entities)  # "Test get_entities with various filters."
    def test_count_entities(self, in_memory_storage, sample_entities)  # "Test count_entities."
    def test_find_relationships_with_filters(self, in_memory_storage, sample_entities, sample_relationships)  # "Test find_relationships with filters."
    def test_count_relationships(self, in_memory_storage, sample_entities, sample_relationships)  # "Test count_relationships."
    def test_get_bundle_info(self, in_memory_storage)  # "Test get_bundle_info."
    def test_get_bundle_info_none(self, in_memory_storage)  # "Test get_bundle_info when no bundle exists."
    def test_is_bundle_loaded(self, in_memory_storage)  # "Test is_bundle_loaded."
class TestPostgresStorage:  # "Direct tests for PostgresStorage using mocked database."
    @pytest.fixture
    def postgres_storage(self)  # "Create PostgresStorage using in-memory SQLite (mocks PostgreSQL)."
    def test_postgres_storage_basic(self, postgres_storage, sample_entities)  # "Test basic PostgresStorage operations."
    def test_postgres_storage_filters(self, postgres_storage, sample_entities)  # "Test PostgresStorage with filters."

============================================================
# kgserver/tests/test_storage_factory.py
============================================================
# imports: pytest, query.storage_factory, query.storage_factory.get_engine, get_storage, close_storage, storage.backends.postgres.PostgresStorage, storage.backends.sqlite.SQLiteStorage, unittest.mock.patch, MagicMock
class TestGetEngine:  # "Test get_engine() function."
    def test_get_engine_with_sqlite_url(self, monkeypatch)  # "Test get_engine with SQLite URL."
    def test_get_engine_with_postgres_url(self, monkeypatch)  # "Test get_engine with PostgreSQL URL."
    def test_get_engine_defaults_to_sqlite(self, monkeypatch)  # "Test that get_engine defaults to SQLite when DATABASE_URL not set."
    def test_get_engine_singleton(self, monkeypatch)  # "Test that get_engine returns the same engine instance."
class TestGetStorage:  # "Test get_storage() function."
    def test_get_storage_sqlite(self, monkeypatch)  # "Test get_storage with SQLite."
    def test_get_storage_postgres(self, monkeypatch)  # "Test get_storage with PostgreSQL."
    def test_get_storage_unsupported_scheme(self, monkeypatch)  # "Test get_storage with unsupported database scheme."
    def test_get_storage_sqlite_file_path(self, monkeypatch)  # "Test get_storage with SQLite file path."
class TestCloseStorage:  # "Test close_storage() function."
    def test_close_storage(self, monkeypatch)  # "Test that close_storage disposes the engine."
    def test_close_storage_when_none(self)  # "Test that close_storage handles None engine gracefully."

============================================================
# kgserver/tests/test_storage_provenance.py
============================================================
# imports: datetime.datetime, json, kgbundle.BundleManifestV1, BundleFile, pytest, storage.backends.sqlite.SQLiteStorage
@pytest.fixture
def bundle_dir_with_provenance(tmp_path)  # "Create a bundle directory with entities, relationships, mentions, and evidence."
@pytest.fixture
def storage_with_provenance_bundle(tmp_path, bundle_dir_with_provenance)  # "SQLite storage with a bundle loaded that includes mentions and evidence."
class TestLoadBundleProvenance:  # "Loading a bundle with mentions and evidence populates provenance tables."
    def test_load_bundle_stores_mentions(self, storage_with_provenance_bundle)  # "After load_bundle, get_mentions_for_entity returns mention rows."
    def test_load_bundle_stores_evidence(self, storage_with_provenance_bundle)  # "After load_bundle, get_evidence_for_relationship returns evidence rows."
    def test_get_mentions_for_entity_nonexistent_returns_empty(self, storage_with_provenance_bundle)  # "get_mentions_for_entity for unknown entity returns empty list."
    def test_get_evidence_for_relationship_nonexistent_returns_empty(self, storage_with_provenance_bundle)  # "get_evidence_for_relationship for unknown triple returns empty list."
class TestEntityRelationshipProvenanceProperties:  # "Entity and relationship provenance summary is stored in properties after load."
    def test_entity_properties_contain_provenance(self, storage_with_provenance_bundle)  # "Entity loaded from bundle has first_seen_document, total_mentions, etc. in properties."
    def test_entity_without_provenance_has_empty_or_no_provenance_keys(self, storage_with_provenance_bundle)  # "Entity e2 was written without provenance fields; properties may not have them."
    def test_relationship_properties_contain_evidence_summary(self, storage_with_provenance_bundle)  # "Relationship loaded from bundle has evidence_count, strongest_evidence_quote in properties."
class TestLoadBundleWithoutProvenanceFiles:  # "Bundles without mentions/evidence files load successfully."
    def test_load_bundle_missing_mentions_file_does_not_raise(self, tmp_path)  # "Manifest has mentions but file is missing; load_bundle does not raise."

============================================================
# summarize_codebase.py
============================================================
# imports: argparse, ast, fnmatch, pathlib.Path, sys
def parse_args()
def get_docstring(node: ast.AST, max_len: int) -> str | None
def format_annotation(node) -> str
def format_default(node) -> str
def format_args(args: ast.arguments) -> str
def format_func(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str, args: argparse.Namespace) -> list[str]
def format_class(node: ast.ClassDef, indent: str, args: argparse.Namespace) -> list[str]
def summarize_module(path: Path, rel: Path, args: argparse.Namespace) -> list[str]
def is_excluded(path: Path, root: Path, excludes: list[str]) -> bool
def main()

============================================================
# tests/__init__.py
============================================================


============================================================
# tests/conftest.py
============================================================
# imports: datetime.datetime, timezone, kgraph.canonical_id.CanonicalId, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, kgraph.pipeline.interfaces.DocumentParserInterface, EntityExtractorInterface, EntityResolverInterface, RelationshipExtractorInterface, kgraph.promotion.PromotionPolicy, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.document.BaseDocument, kgschema.domain.DomainSchema, PredicateConstraint, ValidationIssue, kgschema.entity.BaseEntity, EntityMention, EntityStatus, PromotionConfig, kgschema.relationship.BaseRelationship, kgschema.storage.EntityStorageInterface, pytest, re, typing.Optional, Sequence, uuid
class SimpleEntity(BaseEntity):  # "Minimal concrete entity implementation for unit tests."
    entity_type: str = 'test_entity'
    def get_entity_type(self) -> str  # "Return the entity's type identifier."
class SimpleRelationship(BaseRelationship):  # "Minimal concrete relationship implementation for unit tests."
    subject_entity_type: str = 'test_entity'
    object_entity_type: str = 'test_entity'
    def get_edge_type(self) -> str  # "Return the relationship's edge type (same as predicate)."
class SimpleDocument(BaseDocument):  # "Minimal concrete document implementation for unit tests."
    document_type: str = 'test_document'
    def get_document_type(self) -> str  # "Return the document's type identifier."
    def get_sections(self) -> list[tuple[str, str]]  # "Return document sections as (section_name, content) tuples."
class SimplePromotionPolicy(PromotionPolicy):
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]
class SimpleDomainSchema(DomainSchema):  # "Minimal domain schema defining types and validation for the test domain."
    @property
    def name(self) -> str  # "Return the domain name identifier."
    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]  # "Return mapping of entity type names to their classes."
    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]  # "Return mapping of relationship type names to their classes."
    @property
    def predicate_constraints(self) -> dict[str, PredicateConstraint]  # "Define predicate constraints for the test domain."
    @property
    def document_types(self) -> dict[str, type[BaseDocument]]  # "Return mapping of document type names to their classes."
    @property
    def promotion_config(self) -> PromotionConfig  # "Return configuration for promoting provisional entities to canonical."
    def validate_entity(self, entity: BaseEntity) -> list[ValidationIssue]  # "Check if the entity's type is registered in this schema."
    async def validate_relationship(self, relationship: BaseRelationship, entity_storage: EntityStorageInterface | None=None) -> bool  # "Check if the relationship's predicate is registered in this schema."
    def get_promotion_policy(self, lookup=None) -> PromotionPolicy
class MockEmbeddingGenerator(EmbeddingGeneratorInterface):  # "Mock embedding generator producing deterministic hash-based embeddings."
    def __init__(self, dim: int=8) -> None
    @property
    def dimension(self) -> int  # "Return the embedding vector dimension."
    async def generate(self, text: str) -> tuple[float, ...]  # "Generate a deterministic embedding from text using its hash."
    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]  # "Generate embeddings for multiple texts sequentially."
class MockDocumentParser(DocumentParserInterface):  # "Mock document parser that wraps raw bytes in a SimpleDocument."
    async def parse(self, raw_content: bytes, content_type: str, source_uri: str | None=None) -> BaseDocument  # "Parse raw bytes into a SimpleDocument."
class MockEntityExtractor(EntityExtractorInterface):  # "Mock entity extractor using bracket notation for entity detection."
    async def extract(self, document: BaseDocument) -> list[EntityMention]  # "Extract entity mentions from bracketed text in the document."
class MockEntityResolver(EntityResolverInterface):  # "Mock entity resolver that links mentions to entities via name matching."
    async def resolve(self, mention: EntityMention, existing_storage: EntityStorageInterface) -> tuple[BaseEntity, float]  # "Resolve an entity mention to an existing or new entity."
    async def resolve_batch(self, mentions: Sequence[EntityMention], existing_storage: EntityStorageInterface) -> list[tuple[BaseEntity, float]]  # "Resolve multiple mentions sequentially."
class MockRelationshipExtractor(RelationshipExtractorInterface):  # "Mock relationship extractor that chains adjacent entities."
    async def extract(self, document: BaseDocument, entities: Sequence[BaseEntity]) -> list[BaseRelationship]  # "Extract relationships between consecutive entities."
@pytest.fixture
def test_domain() -> SimpleDomainSchema  # "Provide a SimpleDomainSchema instance for domain-aware tests."
@pytest.fixture
def entity_storage() -> InMemoryEntityStorage  # "Provide a fresh in-memory entity storage instance."
@pytest.fixture
def relationship_storage() -> InMemoryRelationshipStorage  # "Provide a fresh in-memory relationship storage instance."
@pytest.fixture
def document_storage() -> InMemoryDocumentStorage  # "Provide a fresh in-memory document storage instance."
@pytest.fixture
def embedding_generator() -> MockEmbeddingGenerator  # "Provide a MockEmbeddingGenerator with default 8-dimensional vectors."
@pytest.fixture
def document_parser() -> MockDocumentParser  # "Provide a MockDocumentParser for converting raw bytes to SimpleDocument."
@pytest.fixture
def entity_extractor() -> MockEntityExtractor  # "Provide a MockEntityExtractor using bracket notation."
@pytest.fixture
def entity_resolver() -> MockEntityResolver  # "Provide a MockEntityResolver for name-based entity matching."
@pytest.fixture
def relationship_extractor() -> MockRelationshipExtractor  # "Provide a MockRelationshipExtractor for linear entity chaining."
def make_test_entity(name: str, status: EntityStatus=EntityStatus.PROVISIONAL, entity_id: str | None=None, usage_count: int=0, confidence: float=1.0, embedding: tuple[float, ...] | None=None, canonical_ids: dict[str, str] | None=None) -> SimpleEntity  # "Factory function to create SimpleEntity instances with sensible defaults."
def make_test_relationship(subject_id: str, object_id: str, predicate: str='related_to', confidence: float=1.0) -> SimpleRelationship  # "Factory function to create SimpleRelationship instances with defaults."

============================================================
# tests/test_caching.py
============================================================
# imports: asyncio, json, kgraph.pipeline.caching.CachedEmbeddingGenerator, EmbeddingCacheConfig, FileBasedEmbeddingsCache, InMemoryEmbeddingsCache, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, pathlib.Path, pytest, tempfile, tests.conftest.MockEmbeddingGenerator, typing.Sequence
class TestEmbeddingCacheConfig:  # "Test EmbeddingCacheConfig model."
    def test_default_config(self)  # "Test default cache configuration."
    def test_custom_config(self)  # "Test custom cache configuration."
    def test_config_immutability(self)  # "Test that config is immutable (frozen=True)."
class TestInMemoryEmbeddingsCache:  # "Test InMemoryEmbeddingsCache implementation."
    async def test_put_and_get(self)  # "Test basic put and get operations."
    async def test_cache_miss(self)  # "Test get with cache miss."
    async def test_cache_stats(self)  # "Test cache statistics tracking."
    async def test_lru_eviction(self)  # "Test LRU eviction when cache is full."
    async def test_lru_access_order(self)  # "Test that accessing items updates LRU order."
    async def test_batch_get(self)  # "Test batch get operation."
    async def test_batch_put(self)  # "Test batch put operation."
    async def test_clear(self)  # "Test clearing the cache."
    async def test_key_normalization(self)  # "Test that keys are normalized when enabled."
    async def test_no_key_normalization(self)  # "Test cache without key normalization."
class TestFileBasedEmbeddingsCache:  # "Test FileBasedEmbeddingsCache implementation."
    async def test_put_and_get(self)  # "Test basic put and get operations."
    async def test_persistence(self)  # "Test that cache persists across instances."
    async def test_auto_save(self)  # "Test automatic saving at intervals."
    async def test_manual_save(self)  # "Test manual save operation."
    async def test_load_nonexistent_file(self)  # "Test loading from nonexistent file."
    async def test_normalize_keys_on_load(self)  # "Keys loaded from file are normalized so get() with different casing hits."
    async def test_lru_eviction_with_persistence(self)  # "Test LRU eviction works with persistent cache."
    async def test_batch_operations(self)  # "Test batch put and get with persistence."
    async def test_concurrent_access(self)  # "Concurrent get/put/save/load do not corrupt cache."
class TestCachedEmbeddingGenerator:  # "Test CachedEmbeddingGenerator wrapper."
    async def test_cache_hit(self)  # "Test that cached values are returned without calling base generator."
    async def test_cache_miss(self)  # "Test that cache misses call base generator."
    async def test_dimension_property(self)  # "Test that dimension is passed through from base generator."
    async def test_batch_generation_with_cache(self)  # "Test batch generation with partial cache hits."
    async def test_save_cache_convenience_method(self)  # "Test convenience method for saving cache."
    async def test_get_cache_stats(self)  # "Test getting cache statistics through wrapper."
    async def test_cached_generator_calls_base_once_per_text(self)  # "Repeated generate() with same text calls base generator only once."
    async def test_cached_generator_batch_returns_correct_order(self)  # "generate_batch with mixed hits/misses returns list in input order."
class TestCachingIntegration:  # "Integration tests for caching components."
    async def test_end_to_end_caching_workflow(self)  # "Test complete caching workflow with persistence."
    async def test_cache_with_many_items(self)  # "Test cache performance with many items."

============================================================
# tests/test_canonical_id.py
============================================================
# imports: json, kgraph.canonical_id.CanonicalId, JsonFileCanonicalIdCache, check_entity_id_format, extract_canonical_id_from_entity, pathlib.Path, pytest, tempfile, tests.conftest.make_test_entity
class TestCanonicalId:  # "Tests for the CanonicalId model."
    def test_canonical_id_creation(self)  # "CanonicalId can be created with id, url, and synonyms."
    def test_canonical_id_minimal(self)  # "CanonicalId can be created with just an id."
    def test_canonical_id_frozen(self)  # "CanonicalId is frozen (immutable)."
    def test_canonical_id_str_representation(self)  # "CanonicalId string representation returns the id."
class TestJsonFileCanonicalIdCache:  # "Tests for JsonFileCanonicalIdCache implementation."
    def test_cache_creation(self)  # "Cache can be created with a file path."
    def test_store_and_fetch(self)  # "Can store and fetch CanonicalId objects."
    def test_fetch_miss_returns_none(self)  # "Fetching non-existent entry returns None."
    def test_mark_known_bad(self)  # "Can mark terms as known bad and check them."
    def test_cache_persistence(self)  # "Cache persists to disk and can be reloaded."
    def test_cache_metrics(self)  # "Cache tracks metrics correctly."
    def test_cache_migration_from_old_format(self)  # "Cache can migrate from old format (dict[str, str])."
    def test_cache_normalizes_keys(self)  # "Cache normalizes keys (case-insensitive, strips whitespace)."
class TestCanonicalHelpers:  # "Tests for canonical ID helper functions."
    def test_extract_canonical_id_from_entity_with_priority(self)  # "extract_canonical_id_from_entity respects priority order."
    def test_extract_canonical_id_from_entity_no_priority(self)  # "extract_canonical_id_from_entity returns first available if no priority."
    def test_extract_canonical_id_from_entity_no_canonical_ids(self)  # "extract_canonical_id_from_entity returns None if no canonical_ids."
    def test_check_entity_id_format_prefix_match(self)  # "check_entity_id_format matches prefix patterns."
    def test_check_entity_id_format_umls_pattern(self)  # "check_entity_id_format matches UMLS pattern (C + digits)."
    def test_check_entity_id_format_numeric_pattern(self)  # "check_entity_id_format handles numeric patterns (HGNC, RxNorm)."
    def test_check_entity_id_format_uniprot_pattern(self)  # "check_entity_id_format matches UniProt pattern (P/Q + alphanumeric)."
    def test_check_entity_id_format_no_match(self)  # "check_entity_id_format returns None if no pattern matches."
    def test_check_entity_id_format_wrong_entity_type(self)  # "check_entity_id_format returns None for wrong entity type."

============================================================
# tests/test_context_and_builders.py
============================================================
# imports: datetime.datetime, timezone, kgraph.builders.EntityBuilder, RelationshipBuilder, kgraph.clock.IngestionClock, kgraph.context.IngestionContext, kgschema.entity.EntityMention, EntityStatus, pytest, tests.conftest.SimpleDomainSchema
def make_doc(doc_id: str='doc-1', content: str='hello world')
def make_ctx(test_domain, *, doc_id='doc-1', content='hello world')
def test_ingestion_clock_requires_timezone()
def test_ingestion_clock_accepts_timezone_aware()
def test_context_accepts_consistent_builders(test_domain)
def test_context_rejects_mismatched_domain(test_domain)
def test_context_rejects_mismatched_clock(test_domain)
def test_context_rejects_mismatched_document(test_domain)
def test_context_provenance_none_offsets_ok(test_domain)
def test_context_provenance_requires_both_offsets(test_domain)
def test_context_provenance_validates_ranges(test_domain)
def test_context_evidence_defaults_to_current_document(test_domain)
def test_entity_builder_canonical_sets_fields(test_domain)
def test_entity_builder_canonical_rejects_unknown_type(test_domain)
def test_entity_builder_canonical_rejects_blank_id(test_domain)
def test_entity_builder_provisional_from_mention_sets_fields(test_domain)
def test_entity_builder_domain_validation_called(test_domain, monkeypatch)
async def test_relationship_builder_link_sets_fields(test_domain)
async def test_relationship_builder_rejects_unknown_predicate(test_domain)
async def test_relationship_builder_dedupes_source_documents(test_domain)
async def test_relationship_builder_rejects_empty_source_documents(test_domain)

============================================================
# tests/test_entities.py
============================================================
# imports: kgraph.storage.memory.InMemoryEntityStorage, kgschema.entity.EntityStatus, pytest, tests.conftest.make_test_entity
class TestEntityCreation:  # "Tests for creating entities with different statuses and attributes."
    def test_create_provisional_entity(self) -> None  # "Provisional entities are created with PROVISIONAL status and empty canonical IDs."
    def test_create_canonical_entity(self) -> None  # "Canonical entities store validated identifiers from multiple authority sources."
    def test_entity_with_synonyms(self) -> None  # "Entities store alternative names as synonyms for improved matching."
    def test_entity_with_embedding(self) -> None  # "Entities store semantic vector embeddings for similarity-based operations."
    def test_entity_immutability(self) -> None  # "Entities are immutable (frozen Pydantic models) to ensure data integrity."
class TestEntityStorage:  # "Tests for InMemoryEntityStorage CRUD operations and query capabilities."
    async def test_add_and_get(self, entity_storage: InMemoryEntityStorage) -> None  # "Storage supports basic add and retrieve by entity ID."
    async def test_get_nonexistent(self, entity_storage: InMemoryEntityStorage) -> None  # "Retrieving a nonexistent entity ID returns None rather than raising an error."
    async def test_get_batch(self, entity_storage: InMemoryEntityStorage) -> None  # "Batch retrieval returns entities in order, with None for missing IDs."
    async def test_find_by_name(self, entity_storage: InMemoryEntityStorage) -> None  # "Name-based search is case-insensitive for robust entity matching."
    async def test_find_by_name_with_synonyms(self, entity_storage: InMemoryEntityStorage) -> None  # "Name-based search includes synonyms, matching alternative entity names."
    async def test_find_by_embedding(self, entity_storage: InMemoryEntityStorage) -> None  # "Embedding search returns entities with cosine similarity above threshold."
    async def test_find_by_embedding_threshold(self, entity_storage: InMemoryEntityStorage) -> None  # "Embedding search excludes results below the similarity threshold (orthogonal vectors)."
    async def test_update_entity(self, entity_storage: InMemoryEntityStorage) -> None  # "Update replaces an existing entity with a modified copy (e.g., incremented usage count)."
    async def test_update_nonexistent(self, entity_storage: InMemoryEntityStorage) -> None  # "Updating a nonexistent entity returns False rather than creating it."
    async def test_delete_entity(self, entity_storage: InMemoryEntityStorage) -> None  # "Delete removes an entity from storage so subsequent get returns None."
    async def test_count(self, entity_storage: InMemoryEntityStorage) -> None  # "Count returns the total number of entities in storage."

============================================================
# tests/test_evidence_semantic.py
============================================================
# imports: datetime.datetime, timezone, examples.medlit.pipeline.relationships._evidence_contains_both_entities_semantic, kgschema.entity.BaseEntity, EntityStatus, pytest
class _MockEntity(BaseEntity):  # "Minimal entity for testing."
    def get_entity_type(self) -> str
@pytest.fixture
def mock_embedding_generator()  # "Mock embedding generator: same text -> same vector; different texts -> orthogonal."
@pytest.fixture
def entity_with_embedding()  # "Entity with pre-set embedding (so we don't need to generate)."
@pytest.fixture
def entity_without_embedding()  # "Entity without embedding (will be generated via cache)."
@pytest.mark.asyncio
async def test_evidence_empty_rejected(mock_embedding_generator, entity_with_embedding, entity_without_embedding)  # "Empty evidence is rejected without calling embedding generator."
@pytest.mark.asyncio
async def test_evidence_semantic_returns_detail_shape(mock_embedding_generator, entity_with_embedding, entity_without_embedding)  # "Semantic helper returns (ok, reason, detail) with expected keys."
@pytest.mark.asyncio
async def test_evidence_embedding_cached(mock_embedding_generator, entity_with_embedding, entity_without_embedding)  # "Evidence string is cached so same evidence does not call generate twice."

============================================================
# tests/test_evidence_traceability.py
============================================================
# imports: datetime.datetime, examples.medlit_schema.base.ExtractionMethod, StudyType, examples.medlit_schema.base.TextSpanRef, SectionType, examples.medlit_schema.entity.Evidence, TextSpan, examples.medlit_schema.entity.Paper, examples.medlit_schema.relationship.Treats, kgschema.entity.EntityStatus, pydantic.ValidationError, pytest
def test_evidence_with_ids_validates()  # "Test that an Evidence entity with paper_id and text_span_id validates."
def test_evidence_without_paper_id_fails()  # "Test that an Evidence entity without a paper_id fails."
def test_evidence_canonical_id_format()  # "Test that an Evidence entity with a canonical ID format validates."
def test_conceptual_navigation()  # "Test the conceptual navigation from Relationship to Paper."
def test_textspan_is_canonical_only()  # "Test that TextSpan entities are always canonical (not promotable)."
def test_textspan_cannot_be_provisional()  # "Test that TextSpan cannot be created with provisional status."
def test_textspan_requires_offsets()  # "Test that TextSpan requires start_offset and end_offset."
def test_textspan_validates_offset_order()  # "Test that end_offset must be greater than start_offset."
def test_textspan_valid_creation()  # "Test that a valid TextSpan can be created."

============================================================
# tests/test_export.py
============================================================
# imports: datetime.datetime, timezone, json, kgraph.export.write_bundle, kgraph.ingest.IngestionOrchestrator, kgraph.provenance.ProvenanceAccumulator, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.entity.EntityStatus, pathlib.Path, pytest, tests.conftest.MockDocumentParser, MockEmbeddingGenerator, MockEntityExtractor, MockEntityResolver, MockRelationshipExtractor, SimpleDocument, SimpleDomainSchema, SimpleEntity, SimpleRelationship, make_test_entity, make_test_relationship
@pytest.fixture
def orchestrator(tmp_path: Path) -> IngestionOrchestrator  # "Create an orchestrator for export tests."
class TestExportEntities:  # "Tests for exporting entities to a JSON file."
    async def test_export_canonical_entities(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Default export includes only canonical entities, excluding provisionals."
    async def test_export_includes_provisional_when_requested(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Export with include_provisional=True includes both canonical and provisional entities."
    async def test_export_entity_fields(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "All entity fields are correctly serialized: ID, name, synonyms, embedding, canonical_ids, etc."
    async def test_export_creates_parent_directories(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Export automatically creates missing parent directories for the output file."
class TestExportDocument:  # "Tests for exporting per-document JSON files (paper_{doc_id}.json)."
    async def test_export_document_relationships(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Document export includes relationships whose source_documents include the document ID."
    async def test_export_document_provisional_entities(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Document export includes provisional entities whose source matches the document ID."
    async def test_export_document_includes_title(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Document export includes the document title metadata when present."
    async def test_export_nonexistent_document(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Export for a nonexistent document ID creates a file with zero relationships/entities."
class TestExportAll:  # "Tests for full export: global entities.json plus per-document files."
    async def test_export_all_creates_files(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Full export creates entities.json and paper_{doc_id}.json for each document."
    async def test_export_all_returns_statistics(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Full export returns statistics: output_dir, canonical_entities, documents_exported, per-document stats."
class TestListAllMethods:  # "Tests for storage list_all methods used by export functionality."
    async def test_entity_list_all(self, orchestrator: IngestionOrchestrator) -> None  # "list_all() returns all entities regardless of status."
    async def test_entity_list_all_with_status_filter(self, orchestrator: IngestionOrchestrator) -> None  # "list_all(status='canonical'/'provisional') filters entities by status."
    async def test_entity_list_all_pagination(self, orchestrator: IngestionOrchestrator) -> None  # "list_all(limit, offset) supports pagination for large result sets."
    async def test_relationship_get_by_document(self, orchestrator: IngestionOrchestrator) -> None  # "get_by_document() returns relationships whose source_documents include the given document ID."
    async def test_relationship_list_all(self, orchestrator: IngestionOrchestrator) -> None  # "list_all() returns all relationships in storage."
class TestExportBundleProvenance:  # "Tests for bundle export with provenance (mentions.jsonl, evidence.jsonl, summary fields)."
    async def test_write_bundle_with_provenance_writes_mentions_and_evidence(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "With provenance_accumulator populated, write_bundle writes mentions.jsonl and evidence.jsonl and sets manifest."
    async def test_write_bundle_entity_rows_get_provenance_summary(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Entity rows in entities.jsonl include first_seen_document, total_mentions, supporting_documents."
    async def test_write_bundle_relationship_rows_get_evidence_summary(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Relationship rows in relationships.jsonl include evidence_count, strongest_evidence_quote, evidence_confidence_avg."
    async def test_write_bundle_without_provenance_no_mentions_or_evidence_files(self, orchestrator: IngestionOrchestrator, tmp_path: Path) -> None  # "Without provenance_accumulator, no mentions.jsonl or evidence.jsonl and manifest has no mentions/evidence."

============================================================
# tests/test_git_hash.py
============================================================
# imports: kgraph.export.get_git_hash, subprocess, unittest.mock.patch, MagicMock
class TestGetGitHash:  # "Test the get_git_hash() function."
    def test_returns_string_in_git_repo(self)  # "Should return a string when in a git repository."
    def test_returns_short_hash_format(self)  # "Hash should be short format (7+ characters, alphanumeric)."
    def test_returns_none_when_git_unavailable(self)  # "Should return None when git command fails."
    def test_returns_none_when_not_in_repo(self)  # "Should return None when not in a git repository."
    def test_returns_none_on_timeout(self)  # "Should return None when git command times out."
    def test_strips_whitespace_from_output(self)  # "Should strip whitespace from git output."
    def test_uses_correct_git_command(self)  # "Should call git rev-parse --short HEAD."

============================================================
# tests/test_ingestion.py
============================================================
# imports: kgraph.ingest.IngestionOrchestrator, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.entity.EntityStatus, pytest, tests.conftest.MockDocumentParser, MockEmbeddingGenerator, MockEntityExtractor, MockEntityResolver, MockRelationshipExtractor, SimpleDomainSchema, make_test_entity
@pytest.fixture
def orchestrator(test_domain: SimpleDomainSchema, entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage, document_storage: InMemoryDocumentStorage, document_parser: MockDocumentParser, entity_extractor: MockEntityExtractor, entity_resolver: MockEntityResolver, relationship_extractor: MockRelationshipExtractor, embedding_generator: MockEmbeddingGenerator) -> IngestionOrchestrator  # "Create an ingestion orchestrator with mock components."
class TestSingleDocumentIngestion:  # "Tests for ingesting a single document through the two-pass pipeline."
    async def test_ingest_extracts_entities(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Pass 1 extracts bracketed entity mentions and stores them in entity storage."
    async def test_ingest_creates_relationships(self, orchestrator: IngestionOrchestrator, relationship_storage: InMemoryRelationshipStorage) -> None  # "Pass 2 extracts relationships (edges) between entities found in the document."
    async def test_ingest_stores_document(self, orchestrator: IngestionOrchestrator, document_storage: InMemoryDocumentStorage) -> None  # "Ingestion stores the parsed document with its content and metadata."
    async def test_ingest_generates_embeddings(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "New entities receive semantic embeddings for similarity-based operations."
    async def test_ingest_increments_usage_count(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Repeated entity mentions across documents increment the usage count."
    async def test_ingest_no_entities(self, orchestrator: IngestionOrchestrator) -> None  # "Ingestion handles documents without extractable entities gracefully (no errors)."
class TestBatchIngestion:  # "Tests for ingesting multiple documents in a single batch operation."
    async def test_batch_ingest_multiple_documents(self, orchestrator: IngestionOrchestrator) -> None  # "Batch ingestion processes multiple documents and aggregates entity counts."
    async def test_batch_reports_per_document_results(self, orchestrator: IngestionOrchestrator) -> None  # "Batch result includes individual extraction stats for each document."
    async def test_batch_continues_on_error(self, orchestrator: IngestionOrchestrator) -> None  # "Batch ingestion is fault-tolerant: failures in one document don't halt the batch."
class TestDomainValidation:  # "Tests for validating entities and relationships against domain schemas."
    async def test_validates_entities(self, orchestrator: IngestionOrchestrator, test_domain: SimpleDomainSchema) -> None  # "Entities with valid types (per domain schema) are accepted without errors."
    async def test_validates_relationships(self, orchestrator: IngestionOrchestrator) -> None  # "Relationships with valid predicates (per domain schema) are accepted."
class TestMergeCandidateDetection:  # "Tests for detecting potential duplicate entities via embedding similarity."
    async def test_find_merge_candidates_with_similar_entities(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Entities with high cosine similarity (e.g., 'USA' and 'United States') are flagged."
    async def test_find_merge_candidates_returns_empty_list(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "No merge candidates are returned when all entities are below the similarity threshold."

============================================================
# tests/test_logging.py
============================================================
# imports: io.StringIO, kgraph.logging.PprintLogger, setup_logging, logging, pydantic.BaseModel, pytest
class TestPprintLogger:  # "Tests for PprintLogger formatting and delegation."
    def test_pprint_formats_dict(self) -> None  # "Test that pprint=True formats dictionaries nicely."
    def test_pprint_false_uses_str(self) -> None  # "Test that pprint=False uses simple string conversion."
    def test_pprint_defaults_to_true(self) -> None  # "Test that pprint parameter defaults to True."
    def test_simple_string_with_pprint(self) -> None  # "Test that simple strings work with pprint=True."
    def test_all_log_levels_support_pprint(self) -> None  # "Test that all log levels (debug, info, warning, error, critical) support pprint."
    def test_exception_logging(self) -> None  # "Test that exception logging works with pprint."
    def test_delegates_to_underlying_logger(self) -> None  # "Test that PprintLogger delegates other methods to underlying logger."
    def test_nested_structures_formatted(self) -> None  # "Test that deeply nested structures are formatted correctly."
    def test_list_formatting(self) -> None  # "Test that lists are formatted nicely with pprint."
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason='Pydantic not available')
    def test_pydantic_model_uses_model_dump_json(self) -> None  # "Test that Pydantic models use model_dump_json() when pprint=True."
    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason='Pydantic not available')
    def test_pydantic_model_with_pprint_false(self) -> None  # "Test that Pydantic models use str() when pprint=False."
class TestSetupLogging:  # "Tests for setup_logging function."
    def test_setup_logging_returns_pprint_logger(self) -> None  # "Test that setup_logging returns a PprintLogger instance."
    def test_setup_logging_configures_handler(self) -> None  # "Test that setup_logging properly configures handlers and formatters."
    def test_setup_logging_uses_caller_name(self) -> None  # "Test that setup_logging uses the calling function's name as logger name."
    def test_setup_logging_does_not_duplicate_handlers(self) -> None  # "Test that setup_logging doesn't add duplicate handlers on multiple calls."

============================================================
# tests/test_medlit_domain.py
============================================================
# imports: examples.medlit_schema.domain.MedlitDomain
def test_medlit_domain_instantiates()  # "Test that MedlitDomain can be instantiated."
def test_medlit_domain_entity_types()  # "Test that all entity types are registered."
def test_medlit_domain_relationship_types()  # "Test that all relationship types are registered."

============================================================
# tests/test_medlit_entities.py
============================================================
# imports: datetime.datetime, examples.medlit_schema.entity.Disease, Gene, Drug, Protein, Procedure, Institution, Evidence, kgschema.entity.EntityStatus, pydantic.ValidationError, pytest
def test_disease_with_umls_id_validates()  # "Test that a Disease entity with a UMLS ID validates."
def test_gene_with_hgnc_id_validates()  # "Test that a Gene entity with an HGNC ID validates."
def test_drug_with_rxnorm_id_validates()  # "Test that a Drug entity with an RxNorm ID validates."
def test_protein_with_uniprot_id_validates()  # "Test that a Protein entity with a UniProt ID validates."
def test_procedure_validates()  # "Test that a Procedure entity validates."
def test_institution_validates()  # "Test that an Institution entity validates."
def test_provisional_entity_validates()  # "Test that a provisional entity (no ontology ID) validates."
def test_canonical_entity_without_ontology_id_fails()  # "Test that a canonical entity without an ontology ID fails."
def test_evidence_cannot_be_provisional()  # "Test that Evidence entities cannot be created with PROVISIONAL status."

============================================================
# tests/test_medlit_relationships.py
============================================================
# imports: datetime.datetime, examples.medlit_schema.relationship.Treats, AuthoredBy, AssociatedWith, PartOf, pydantic.ValidationError, pytest
def test_treats_with_evidence_validates()  # "Test that a Treats relationship with evidence validates."
def test_treats_without_evidence_fails()  # "Test that a Treats relationship without evidence fails."
def test_bibliographic_relationship_without_evidence_validates()  # "Test that a bibliographic relationship without evidence validates."
def test_associated_with_with_evidence_validates()  # "Test that an AssociatedWith relationship with evidence validates."
def test_part_of_without_evidence_validates()  # "Test that a PartOf relationship without evidence validates."

============================================================
# tests/test_paper_model.py
============================================================
# imports: datetime.datetime, examples.medlit_schema.base.ExtractionProvenance, ModelInfo, examples.medlit_schema.entity.Paper, PaperMetadata
def test_paper_with_full_metadata_validates()  # "Test that a Paper with full metadata validates."
def test_papermetadata_with_study_type_validates()  # "Test that PaperMetadata with a study_type validates."
def test_extractionprovenance_serializes_correctly()  # "Test that ExtractionProvenance serializes correctly."

============================================================
# tests/test_pipeline_integration.py
============================================================
# imports: kgraph.ingest.IngestionOrchestrator, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.entity.EntityStatus, pytest, tests.conftest.MockDocumentParser, MockEmbeddingGenerator, MockEntityExtractor, MockEntityResolver, MockRelationshipExtractor, SimpleDomainSchema, make_test_entity
@pytest.fixture
def orchestrator(test_domain: SimpleDomainSchema, entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage, document_storage: InMemoryDocumentStorage, document_parser: MockDocumentParser, entity_extractor: MockEntityExtractor, entity_resolver: MockEntityResolver, relationship_extractor: MockRelationshipExtractor, embedding_generator: MockEmbeddingGenerator) -> IngestionOrchestrator  # "Create an ingestion orchestrator with mock components."
class TestFullPipelineIntegration:  # "End-to-end integration tests for the complete ingestion pipeline."
    async def test_batch_ingestion_creates_provisional_entities(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Batch ingestion creates provisional entities from document mentions."
    async def test_repeated_mentions_increase_usage_count(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Entities mentioned in multiple documents have higher usage counts."
    async def test_promotion_converts_high_usage_provisionals_to_canonical(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Entities meeting usage threshold are promoted to canonical status."
    async def test_merge_candidates_detected_by_embedding_similarity(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Similar canonical entities are identified as merge candidates."
    async def test_full_pipeline_ingestion_promotion_merge_detection(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Complete pipeline: ingest documents, promote entities, detect merge candidates."

============================================================
# tests/test_pmc_chunker.py
============================================================
# imports: examples.medlit.pipeline.pmc_chunker.PMCStreamingChunker, document_id_from_source_uri, pytest
MINIMAL_PMC_XML = b'<?xml version="1.0"?>\n<article xmlns="http://jats.nlm.nih.gov">\n  <front><article-meta>\n    <abstract><p>Abstract text here.</p></abstract>\n  </article-meta></front>\n  <body>\n    <sec id="s1"><title>Intro</title><p>First section content.</p></sec>\n    <sec id="s2"><p>Second section with more text.</p></sec>\n  </body>\n</article>\n'
@pytest.fixture
def chunker() -> PMCStreamingChunker  # "PMC chunker with small window for tests."
@pytest.mark.asyncio
async def test_chunk_from_raw_xml_returns_document_chunks(chunker: PMCStreamingChunker) -> None  # "chunk_from_raw with XML returns list of DocumentChunk."
@pytest.mark.asyncio
async def test_chunk_from_raw_xml_content_type_with_charset(chunker: PMCStreamingChunker) -> None  # "content_type with charset (e.g. application/xml; charset=utf-8) is treated as XML."
@pytest.mark.asyncio
async def test_chunk_from_raw_non_xml_returns_single_chunk(chunker: PMCStreamingChunker) -> None  # "Non-XML content_type returns a single chunk with decoded text."
def test_document_id_from_source_uri() -> None  # "document_id_from_source_uri returns stem of path."

============================================================
# tests/test_pmc_streaming.py
============================================================
# imports: examples.medlit.pipeline.pmc_streaming.iter_overlapping_windows, iter_pmc_sections, iter_pmc_windows, pytest
MINIMAL_PMC = b'<?xml version="1.0"?>\n<article xmlns="http://jats.nlm.nih.gov">\n  <front><article-meta>\n    <abstract><p>Abstract text here.</p></abstract>\n  </article-meta></front>\n  <body>\n    <sec id="s1"><title>Intro</title><p>First section content.</p></sec>\n    <sec id="s2"><p>Second section with more text.</p></sec>\n  </body>\n</article>\n'
def test_iter_pmc_sections_yields_abstract_and_secs()  # "iter_pmc_sections yields abstract first then body secs."
def test_iter_overlapping_windows_abstract_separately()  # "Abstract is yielded as first window when include_abstract_separately True."
def test_iter_overlapping_windows_has_overlap()  # "Consecutive windows overlap by roughly overlap chars."
def test_iter_pmc_windows_returns_iterator()  # "iter_pmc_windows returns an iterator of (index, text)."

============================================================
# tests/test_promotion.py
============================================================
# imports: datetime.datetime, timezone, examples.sherlock.domain.SherlockCharacter, examples.sherlock.domain.SherlockCharacter, SherlockStory, AppearsInRelationship, examples.sherlock.domain.SherlockDomainSchema, examples.sherlock.pipeline.SherlockDocumentParser, SherlockEntityExtractor, SherlockEntityResolver, SherlockRelationshipExtractor, SimpleEmbeddingGenerator, examples.sherlock.promotion.SherlockPromotionPolicy, kgraph.canonical_id.CanonicalId, kgraph.ingest.IngestionOrchestrator, kgraph.pipeline.embedding.EmbeddingGeneratorInterface, kgraph.pipeline.interfaces.DocumentParserInterface, EntityExtractorInterface, EntityResolverInterface, RelationshipExtractorInterface, kgraph.promotion.PromotionPolicy, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.domain.DomainSchema, kgschema.entity.BaseEntity, EntityStatus, PromotionConfig, pytest, tests.conftest.make_test_entity
class SimplePromotionPolicy(PromotionPolicy):  # "Test implementation with hardcoded mappings."
    async def assign_canonical_id(self, entity: BaseEntity) -> CanonicalId | None
class TestPromotionPolicyBase:  # "Test the base PromotionPolicy class behavior."
    def test_should_promote_rejects_already_canonical(self)  # "should_promote returns False for entities already canonical."
    def test_should_promote_requires_min_usage(self)  # "should_promote checks minimum usage count threshold."
    def test_should_promote_requires_min_confidence(self)  # "should_promote checks minimum confidence threshold."
    def test_should_promote_checks_embedding_requirement(self)  # "should_promote respects require_embedding config."
    async def test_assign_canonical_id_returns_mapping(self)  # "assign_canonical_id returns mapped ID or None."
class TestSherlockPromotion:  # "Test Sherlock-specific promotion policy with DBPedia mappings."
    async def test_sherlock_policy_has_dbpedia_mappings(self)  # "SherlockPromotionPolicy contains DBPedia URI mappings."
    def test_sherlock_promotion_config_has_low_thresholds(self)  # "Sherlock domain uses lower thresholds for small corpus."
    def test_get_promotion_policy_accepts_lookup_parameter(self)  # "get_promotion_policy accepts lookup parameter for signature compliance."
@pytest.fixture
async def orchestrator(test_domain: DomainSchema, entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage, document_storage: InMemoryDocumentStorage, document_parser: DocumentParserInterface, entity_extractor: EntityExtractorInterface, entity_resolver: EntityResolverInterface, relationship_extractor: RelationshipExtractorInterface, embedding_generator: EmbeddingGeneratorInterface)  # "Create a generic orchestrator for testing."
@pytest.fixture
async def sherlock_orchestrator(entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage, document_storage: InMemoryDocumentStorage)  # "Create orchestrator with Sherlock domain for promotion testing."
class TestPromotionWorkflow:  # "Test complete promotion workflow with relationship updates."
    async def test_entities_start_as_provisional_with_canonical_hint(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage)  # "Entities created with canonical_id_hint still start as PROVISIONAL."
    async def test_promotion_changes_id_and_status(self, sherlock_orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage)  # "Promotion updates entity_id to DBPedia URI and status to CANONICAL."
    async def test_promotion_updates_relationship_references(self, sherlock_orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage)  # "Promotion updates relationships to point to new canonical ID."
    async def test_entities_without_mapping_remain_provisional(self, sherlock_orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage)  # "Entities without canonical ID mapping stay provisional even if eligible."
    async def test_low_usage_entities_not_promoted(self, sherlock_orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage)  # "Entities below usage threshold aren't promoted even with mapping."
class TestPromotionIntegration:  # "Test promotion in complete ingestion pipeline."
    async def test_full_sherlock_ingestion_and_promotion(self, sherlock_orchestrator: IngestionOrchestrator)

============================================================
# tests/test_promotion_merge.py
============================================================
# imports: kgraph.ingest.IngestionOrchestrator, kgraph.storage.memory.InMemoryDocumentStorage, InMemoryEntityStorage, InMemoryRelationshipStorage, kgschema.entity.EntityStatus, pytest, tests.conftest.MockDocumentParser, MockEmbeddingGenerator, MockEntityExtractor, MockEntityResolver, MockRelationshipExtractor, SimpleDomainSchema, make_test_entity, make_test_relationship
@pytest.fixture
def orchestrator(test_domain: SimpleDomainSchema, entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage, document_storage: InMemoryDocumentStorage, document_parser: MockDocumentParser, entity_extractor: MockEntityExtractor, entity_resolver: MockEntityResolver, relationship_extractor: MockRelationshipExtractor, embedding_generator: MockEmbeddingGenerator) -> IngestionOrchestrator  # "Create an ingestion orchestrator with mock components."
class TestEntityPromotion:  # "Tests for promoting provisional entities to canonical status."
    async def test_promote_updates_status(self, entity_storage: InMemoryEntityStorage) -> None  # "Promotion changes status from PROVISIONAL to CANONICAL and assigns a new entity ID."
    async def test_promote_updates_storage_reference(self, entity_storage: InMemoryEntityStorage) -> None  # "Promotion replaces the old provisional ID with the new canonical ID in storage."
    async def test_promote_nonexistent_returns_none(self, entity_storage: InMemoryEntityStorage) -> None  # "Attempting to promote a nonexistent entity ID returns None."
    async def test_find_provisional_for_promotion(self, entity_storage: InMemoryEntityStorage) -> None  # "Find provisionals meeting min_usage and min_confidence thresholds for promotion."
    async def test_run_promotion(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage) -> None  # "Orchestrator run_promotion() finds and promotes all eligible provisional entities."
class TestEntityMerging:  # "Tests for merging duplicate canonical entities into a single target entity."
    async def test_merge_combines_synonyms(self, entity_storage: InMemoryEntityStorage) -> None  # "Merge adds the source entity's name and synonyms to the target's synonym list."
    async def test_merge_combines_usage_counts(self, entity_storage: InMemoryEntityStorage) -> None  # "Merge sums the usage counts of all source entities into the target."
    async def test_merge_removes_source_entities(self, entity_storage: InMemoryEntityStorage) -> None  # "Merge deletes source entities from storage after consolidation."
    async def test_merge_nonexistent_target_fails(self, entity_storage: InMemoryEntityStorage) -> None  # "Merge fails (returns False) if the target entity ID does not exist."
    async def test_merge_updates_relationships(self, orchestrator: IngestionOrchestrator, entity_storage: InMemoryEntityStorage, relationship_storage: InMemoryRelationshipStorage) -> None  # "Merge rewrites all relationship subject_id/object_id references from source to target."

============================================================
# tests/test_provenance.py
============================================================
# imports: kgraph.provenance.ProvenanceAccumulator, pytest
class TestProvenanceAccumulator:  # "ProvenanceAccumulator records mentions and evidence for bundle export."
    def test_init_empty(self) -> None  # "New accumulator has no mentions or evidence."
    def test_add_mention_appends_and_increments_count(self) -> None  # "add_mention appends a MentionRow and mention_count increases."
    def test_add_evidence_appends_and_increments_count(self) -> None  # "add_evidence appends an EvidenceRow and evidence_count increases."
    def test_mentions_and_evidence_independent(self) -> None  # "Mention and evidence lists are independent; adding one does not affect the other."
    def test_mentions_exposed_for_export(self) -> None  # "Accumulator exposes mentions list for exporter to iterate."

============================================================
# tests/test_relationship_swap.py
============================================================
# imports: datetime.datetime, timezone, examples.medlit.domain.MedLitDomainSchema, examples.medlit.pipeline.relationships.MedLitRelationshipExtractor, examples.medlit.pipeline.relationships._build_entity_index, MedLitRelationshipExtractor, examples.medlit.pipeline.relationships._evidence_contains_both_entities, kgraph.storage.memory.InMemoryEntityStorage, kgschema.domain.DomainSchema, PredicateConstraint, ValidationIssue, kgschema.entity.BaseEntity, EntityStatus, kgschema.relationship.BaseRelationship, pytest, types.SimpleNamespace
class DrugEntity(BaseEntity):  # "Drug entity for testing."
    def get_entity_type(self) -> str
class DiseaseEntity(BaseEntity):  # "Disease entity for testing."
    def get_entity_type(self) -> str
class TreatsRelationship(BaseRelationship):  # "Treats relationship for testing."
    def get_edge_type(self) -> str
class TestDomainSchema(DomainSchema):  # "Test domain schema with predicate constraints."
    @property
    def name(self) -> str
    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]
    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]
    @property
    def predicate_constraints(self) -> dict[str, PredicateConstraint]
    @property
    def document_types(self) -> dict[str, type]
    def validate_entity(self, entity: BaseEntity) -> list[ValidationIssue]  # "Validate entity is of a registered type."
    def get_promotion_policy(self, lookup=None)  # "Not needed for these tests."
@pytest.fixture
def domain()  # "Create test domain schema."
@pytest.fixture
def drug_entity()  # "Create a drug entity."
@pytest.fixture
def disease_entity()  # "Create a disease entity."
@pytest.mark.asyncio
async def test_validate_correct_order(domain, drug_entity, disease_entity)  # "Test that correctly ordered relationship passes validation."
@pytest.mark.asyncio
async def test_validate_reversed_order_detected(domain, drug_entity, disease_entity)  # "Test that reversed relationship is detected and rejected with helpful message."
def test_should_swap_detection()  # "Test the swap detection logic in the medlit extractor."
@pytest.mark.asyncio
async def test_process_llm_item_reversed_treats_swap_accepted()  # "Reversed (disease, treats, drug) is fixed by swap and relationship is accepted."
def test_evidence_contains_both_entities_both_present()  # "Evidence containing both subject and object is accepted."
def test_evidence_contains_both_entities_missing_subject()  # "Evidence missing subject is rejected with evidence_missing_subject."
def test_evidence_contains_both_entities_empty_evidence()  # "Empty evidence is rejected with evidence_empty."
def test_evidence_contains_both_entities_synonym_match()  # "Entity synonym appearing in evidence counts as match."

============================================================
# tests/test_relationships.py
============================================================
# imports: kgraph.storage.memory.InMemoryRelationshipStorage, pytest, tests.conftest.make_test_relationship
class TestRelationshipCreation:  # "Tests for creating relationship (edge) instances."
    def test_create_relationship(self) -> None  # "Relationships have a subject_id, predicate, and object_id forming a directed edge."
    def test_relationship_with_metadata(self) -> None  # "Relationships store domain-specific metadata (e.g., evidence_type, section)."
    def test_relationship_with_source_documents(self) -> None  # "Relationships track which documents they were extracted from via source_documents."
    def test_relationship_immutability(self) -> None  # "Relationships are immutable (frozen Pydantic models) to ensure data integrity."
class TestRelationshipStorage:  # "Tests for InMemoryRelationshipStorage CRUD operations and queries."
    async def test_add_and_find(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "Storage supports add and find_by_triple (exact subject-predicate-object lookup)."
    async def test_find_nonexistent_triple(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "find_by_triple returns None when no matching relationship exists."
    async def test_get_by_subject(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "get_by_subject returns all relationships where the given entity is the subject."
    async def test_get_by_subject_with_predicate(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "get_by_subject with predicate filter returns only matching edge types."
    async def test_get_by_object(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "get_by_object returns all relationships where the given entity is the object."
    async def test_update_entity_references(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "update_entity_references rewrites subject/object IDs when entities are merged."
    async def test_delete_relationship(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "delete removes a relationship by its triple (subject, predicate, object)."
    async def test_delete_nonexistent(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "delete returns False when the specified triple does not exist."
    async def test_count(self, relationship_storage: InMemoryRelationshipStorage) -> None  # "count returns the total number of relationships in storage."

============================================================
# tests/test_streaming.py
============================================================
# imports: datetime.datetime, timezone, kgraph.pipeline.streaming.BatchingEntityExtractor, ChunkingConfig, DocumentChunk, StreamingEntityExtractorInterface, WindowedDocumentChunker, WindowedRelationshipExtractor, kgschema.document.BaseDocument, kgschema.entity.EntityMention, kgschema.relationship.BaseRelationship, pytest, tests.conftest.MockEntityExtractor, MockRelationshipExtractor, SimpleDocument, SimpleEntity, make_test_entity
def make_simple_document(document_id: str, content: str) -> SimpleDocument  # "Helper to create SimpleDocument with required fields."
class TestDocumentChunk:  # "Test DocumentChunk model."
    def test_chunk_creation(self)  # "Test creating a document chunk."
    def test_chunk_immutability(self)  # "Test that chunks are immutable (frozen=True)."
class TestChunkingConfig:  # "Test ChunkingConfig model."
    def test_default_config(self)  # "Test default chunking configuration."
    def test_custom_config(self)  # "Test custom chunking configuration."
    def test_config_immutability(self)  # "Test that config is immutable (frozen=True)."
class TestWindowedDocumentChunker:  # "Test WindowedDocumentChunker implementation."
    async def test_single_chunk_document(self)  # "Test chunking a document that fits in a single chunk."
    async def test_multiple_chunks_no_overlap(self)  # "Test chunking a document into multiple non-overlapping chunks."
    async def test_multiple_chunks_with_overlap(self)  # "Test chunking with overlap between chunks."
    async def test_respect_sentence_boundaries(self)  # "Test chunking that respects sentence boundaries."
    async def test_chunk_metadata_preserved(self)  # "Test that document ID is preserved in chunks."
    async def test_chunk_indices_sequential(self)  # "Test that chunk indices are sequential."
class TestBatchingEntityExtractor:  # "Test BatchingEntityExtractor implementation."
    async def test_extract_from_single_chunk(self)  # "Test extracting entities from a single chunk."
    async def test_extract_from_multiple_chunks(self)  # "Test extracting entities from multiple chunks."
    async def test_offset_adjustment(self)  # "Test that entity offsets are adjusted for chunk position."
    async def test_streaming_iteration(self)  # "Test that results are yielded incrementally."
class TestWindowedRelationshipExtractor:  # "Test WindowedRelationshipExtractor implementation."
    async def test_extract_from_single_window(self)  # "Test extracting relationships from a single window."
    async def test_deduplication_across_windows(self)  # "Test that duplicate relationships are deduplicated across overlapping windows."
    async def test_empty_window(self)  # "Test handling windows with no entities."
    async def test_single_entity_window(self)  # "Test handling windows with only one entity."
class TestIntegrationStreamingPipeline:  # "Integration tests for streaming pipeline."
    async def test_full_streaming_pipeline(self)  # "Test complete streaming pipeline: chunk -> extract entities -> extract relationships."
    async def test_large_document_chunking(self)  # "Test handling very large documents with many chunks."

# Total: 162 files, ~305,225 chars
