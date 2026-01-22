"""Two-pass ingestion orchestrator for the knowledge graph framework.

This module provides the `IngestionOrchestrator` class, which coordinates the
complete document ingestion pipeline. The orchestrator manages the two-pass
process that transforms raw documents into structured knowledge:

**Pass 1 - Entity Extraction:**
    1. Parse raw document bytes into structured `BaseDocument`
    2. Extract entity mentions using the configured `EntityExtractorInterface`
    3. Resolve mentions to canonical or provisional entities
    4. Generate embeddings for new entities
    5. Store entities, updating usage counts for existing ones

**Pass 2 - Relationship Extraction:**
    1. Extract relationships between resolved entities
    2. Validate relationships against the domain schema
    3. Store relationships with source document references

The orchestrator also provides methods for:
    - Batch ingestion of multiple documents
    - Entity promotion (provisional → canonical)
    - Duplicate detection via embedding similarity
    - Entity merging
    - JSON export of entities and relationships

Example usage:
    ```python
    orchestrator = IngestionOrchestrator(
        domain=my_domain_schema,
        parser=my_parser,
        entity_extractor=my_extractor,
        entity_resolver=my_resolver,
        relationship_extractor=my_rel_extractor,
        embedding_generator=my_embedder,
        entity_storage=entity_store,
        relationship_storage=rel_store,
        document_storage=doc_store,
    )

    result = await orchestrator.ingest_document(
        raw_content=document_bytes,
        content_type="text/plain",
    )
    print(f"Extracted {result.entities_extracted} entities")
    ```
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from kgraph.domain import DomainSchema
from kgraph.entity import BaseEntity, EntityStatus
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from kgraph.pipeline.interfaces import (
    DocumentParserInterface,
    EntityExtractorInterface,
    EntityResolverInterface,
    RelationshipExtractorInterface,
)
from kgraph.relationship import BaseRelationship
from kgraph.storage.interfaces import (
    DocumentStorageInterface,
    EntityStorageInterface,
    RelationshipStorageInterface,
)


@dataclass(frozen=True)
class DocumentResult:
    """Result of processing a single document through the ingestion pipeline.

    Contains statistics about the extraction process and any errors encountered.
    Immutable (frozen) to ensure results can be safely shared and stored.

    Attributes:
        document_id: Unique identifier assigned to the parsed document.
        entities_extracted: Total number of entity mentions found in the document.
        entities_new: Number of mentions that created new provisional entities.
        entities_existing: Number of mentions that matched existing entities.
        relationships_extracted: Number of relationships stored from this document.
        errors: Tuple of error messages encountered during processing.
    """

    document_id: str
    entities_extracted: int
    entities_new: int
    entities_existing: int
    relationships_extracted: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class IngestionResult:
    """Result of batch document ingestion.

    Aggregates statistics across multiple documents and provides per-document
    breakdown via the `document_results` field.

    Attributes:
        documents_processed: Total number of documents in the batch.
        documents_failed: Number of documents that had errors during processing.
        total_entities_extracted: Sum of entity mentions across all documents.
        total_relationships_extracted: Sum of relationships across all documents.
        document_results: Per-document results for detailed inspection.
        errors: Top-level errors that prevented document processing.
    """

    documents_processed: int
    documents_failed: int
    total_entities_extracted: int
    total_relationships_extracted: int
    document_results: tuple[DocumentResult, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class IngestionOrchestrator:
    """Orchestrates two-pass document ingestion for knowledge graph construction.

    The orchestrator is the main entry point for document processing. It
    coordinates all pipeline components (parser, extractors, resolver,
    embedding generator) and storage backends to transform raw documents
    into structured knowledge graph data.

    **Two-Pass Architecture:**

    - **Pass 1 (Entity Extraction)**: Extracts entity mentions from documents,
      resolves them to canonical or provisional entities, generates embeddings,
      and updates storage with new entities or incremented usage counts.

    - **Pass 2 (Relationship Extraction)**: Identifies relationships between
      the resolved entities and stores them with source document references.

    **Additional Operations:**

    - `run_promotion()`: Promotes provisional entities to canonical status
      based on usage frequency and confidence thresholds.
    - `find_merge_candidates()`: Detects potential duplicate entities using
      embedding similarity.
    - `merge_entities()`: Combines duplicate entities and updates references.
    - `export_*()`: Exports entities and relationships to JSON files.

    Attributes:
        domain: Schema defining entity types, relationship types, and validation.
        parser: Converts raw bytes to structured documents.
        entity_extractor: Extracts entity mentions from documents.
        entity_resolver: Maps mentions to canonical or provisional entities.
        relationship_extractor: Extracts relationships between entities.
        embedding_generator: Creates semantic vectors for similarity operations.
        entity_storage: Persistence backend for entities.
        relationship_storage: Persistence backend for relationships.
        document_storage: Persistence backend for source documents.
    """

    domain: DomainSchema
    parser: DocumentParserInterface
    entity_extractor: EntityExtractorInterface
    entity_resolver: EntityResolverInterface
    relationship_extractor: RelationshipExtractorInterface
    embedding_generator: EmbeddingGeneratorInterface
    entity_storage: EntityStorageInterface
    relationship_storage: RelationshipStorageInterface
    document_storage: DocumentStorageInterface

    async def ingest_document(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> DocumentResult:
        """Ingest a single document through the two-pass pipeline.

        Args:
            raw_content: Raw document bytes
            content_type: MIME type or format indicator
            source_uri: Optional source location

        Returns:
            DocumentResult with extraction statistics
        """
        errors: list[str] = []

        # Parse document
        try:
            document = await self.parser.parse(raw_content, content_type, source_uri)
        except Exception as e:
            return DocumentResult(
                document_id="",
                entities_extracted=0,
                entities_new=0,
                entities_existing=0,
                relationships_extracted=0,
                errors=(f"Parse error: {e}",),
            )

        # Store document
        await self.document_storage.add(document)

        # Pass 1: Entity extraction
        mentions = await self.entity_extractor.extract(document)

        resolved_entities: list[BaseEntity] = []
        entities_new = 0
        entities_existing = 0

        for mention in mentions:
            try:
                entity, _ = await self.entity_resolver.resolve(mention, self.entity_storage)

                # Validate entity against domain schema
                if not self.domain.validate_entity(entity):
                    errors.append(f"Entity validation failed: {entity.name} ({entity.get_entity_type()})")
                    continue

                # Check if entity already exists
                existing = await self.entity_storage.get(entity.entity_id)
                if existing:
                    # Increment usage count
                    updated = existing.model_copy(update={"usage_count": existing.usage_count + 1})
                    await self.entity_storage.update(updated)
                    resolved_entities.append(updated)
                    entities_existing += 1
                else:
                    # Generate embedding if needed and not present
                    if entity.embedding is None:
                        try:
                            embedding = await self.embedding_generator.generate(entity.name)
                            entity = entity.model_copy(update={"embedding": embedding})
                        except Exception as e:
                            errors.append(f"Embedding generation failed: {e}")

                    await self.entity_storage.add(entity)
                    resolved_entities.append(entity)
                    entities_new += 1

            except Exception as e:
                errors.append(f"Entity resolution error for '{mention.text}': {e}")

        # Pass 2: Relationship extraction
        relationships: list[BaseRelationship] = []
        if resolved_entities:
            try:
                extracted_rels = await self.relationship_extractor.extract(document, resolved_entities)
                for rel in extracted_rels:
                    if self.domain.validate_relationship(rel):
                        await self.relationship_storage.add(rel)
                        relationships.append(rel)
                    else:
                        errors.append(f"Relationship validation failed: {rel.subject_id} -{rel.predicate}-> {rel.object_id}")
            except Exception as e:
                errors.append(f"Relationship extraction error: {e}")

        return DocumentResult(
            document_id=document.document_id,
            entities_extracted=len(mentions),
            entities_new=entities_new,
            entities_existing=entities_existing,
            relationships_extracted=len(relationships),
            errors=tuple(errors),
        )

    async def ingest_batch(
        self,
        documents: Sequence[tuple[bytes, str, str | None]],
    ) -> IngestionResult:
        """Ingest multiple documents.

        Args:
            documents: Sequence of (raw_content, content_type, source_uri) tuples

        Returns:
            IngestionResult with aggregated statistics
        """
        results: list[DocumentResult] = []
        errors: list[str] = []
        documents_failed = 0

        for raw_content, content_type, source_uri in documents:
            try:
                result = await self.ingest_document(raw_content, content_type, source_uri)
                results.append(result)
                if result.errors:
                    documents_failed += 1
            except Exception as e:
                errors.append(f"Document ingestion failed: {e}")
                documents_failed += 1

        total_entities = sum(r.entities_extracted for r in results)
        total_relationships = sum(r.relationships_extracted for r in results)

        return IngestionResult(
            documents_processed=len(documents),
            documents_failed=documents_failed,
            total_entities_extracted=total_entities,
            total_relationships_extracted=total_relationships,
            document_results=tuple(results),
            errors=tuple(errors),
        )

    async def run_promotion(self) -> list[BaseEntity]:
        """Promote eligible provisional entities to canonical status.

        Uses the domain's promotion policy to determine which entities should
        be promoted and what canonical IDs to assign them.

        The promotion process:
            1. Get the domain's promotion policy
            2. Find provisional entities meeting threshold criteria
            3. For each candidate, check if policy can assign a canonical ID
            4. Update entity status and ID in storage
            5. Update all relationships pointing to the old provisional ID

        Returns:
            List of entities that were successfully promoted.
        """
        policy = self.domain.get_promotion_policy()
        config = self.domain.promotion_config

        # Get candidates meeting basic thresholds
        candidates = await self.entity_storage.find_provisional_for_promotion(
            min_usage=config.min_usage_count,
            min_confidence=config.min_confidence,
        )

        promoted: list[BaseEntity] = []
        for entity in candidates:
            # Check if policy says we should promote
            if not policy.should_promote(entity):
                continue

            # Get canonical ID from policy
            canonical_id = policy.assign_canonical_id(entity)
            if canonical_id is None:
                continue  # No canonical ID available yet

            # Update entity in storage with new ID and status
            new_canonical_ids = entity.canonical_ids.copy()
            new_canonical_ids["dbpedia"] = canonical_id
            promoted_entity = await self.entity_storage.promote(entity.entity_id, canonical_id, canonical_ids=new_canonical_ids)

            if promoted_entity:
                # Update all relationships pointing to old ID
                await self.relationship_storage.update_entity_references(entity.entity_id, canonical_id)
                promoted.append(promoted_entity)

        return promoted

    async def find_merge_candidates(
        self,
        similarity_threshold: float = 0.95,
    ) -> list[tuple[BaseEntity, BaseEntity, float]]:
        """Find canonical entities that may be duplicates based on embedding similarity.

        Scans all canonical entities with embeddings and computes pairwise cosine
        similarity. Returns pairs exceeding the threshold as potential duplicates
        for human review or automatic merging.

        Args:
            similarity_threshold: Minimum cosine similarity (0.0 to 1.0) for a
                pair to be considered a merge candidate. Higher values are more
                conservative. Default 0.95 requires very similar embeddings.

        Returns:
            List of (entity1, entity2, similarity_score) tuples for entity pairs
            with similarity >= threshold. Sorted by descending similarity.

        Note:
            This method performs O(n²) pairwise comparisons, which can be slow
            for large entity collections. For production use with >10K entities,
            consider using approximate nearest neighbor algorithms (FAISS, Annoy)
            or a vector database with built-in similarity search.

        Example:
            ```python
            candidates = await orchestrator.find_merge_candidates(threshold=0.98)
            for e1, e2, sim in candidates:
                print(f"{e1.name} ↔ {e2.name}: {sim:.3f}")
            ```
        """
        candidates: list[tuple[BaseEntity, BaseEntity, float]] = []

        # 1. Get all canonical entities with embeddings
        all_canonical_entities = await self.entity_storage.list_all(status=EntityStatus.CANONICAL.value)
        entities_with_embeddings = [e for e in all_canonical_entities if e.embedding is not None]

        if len(entities_with_embeddings) < 2:
            return candidates

        # 2. Extract embeddings into a matrix
        embedding_matrix = np.array([e.embedding for e in entities_with_embeddings])

        # 3. Calculate cosine similarity for all pairs
        # This gives a matrix where similarity_matrix[i, j] is the similarity
        # between entity i and entity j.
        similarity_matrix = cosine_similarity(embedding_matrix)

        # 4. Find pairs above the threshold
        # We only need to check the upper triangle of the matrix (where i < j)
        # to avoid duplicate pairs (a,b) and (b,a) and self-comparisons (a,a).
        num_entities = len(entities_with_embeddings)
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    entity1 = entities_with_embeddings[i]
                    entity2 = entities_with_embeddings[j]
                    candidates.append((entity1, entity2, similarity))

        return candidates

    async def merge_entities(
        self,
        source_ids: Sequence[str],
        target_id: str,
    ) -> bool:
        """Merge multiple entities into a single target entity.

        Combines data from source entities into the target and updates all
        relationship references to point to the target. Source entities are
        deleted after merging.

        The merge operation:
            1. Updates all relationships referencing source entities to use target_id
            2. Combines synonyms from all source entities into the target
            3. Sums usage counts from all entities
            4. Deletes source entities from storage

        Args:
            source_ids: Entity IDs to merge into the target. These entities
                will be deleted after their data is combined.
            target_id: Entity ID that will absorb the source entities. Must
                exist in storage.

        Returns:
            True if the merge succeeded, False if the target entity was not
            found or if the merge operation failed.

        Example:
            ```python
            # After finding merge candidates
            candidates = await orchestrator.find_merge_candidates()
            for e1, e2, sim in candidates:
                # Keep the entity with more usage, merge the other into it
                if e1.usage_count >= e2.usage_count:
                    await orchestrator.merge_entities([e2.entity_id], e1.entity_id)
                else:
                    await orchestrator.merge_entities([e1.entity_id], e2.entity_id)
            ```
        """
        # Update relationship references
        for source_id in source_ids:
            if source_id != target_id:
                await self.relationship_storage.update_entity_references(source_id, target_id)

        # Merge entity data
        return await self.entity_storage.merge(source_ids, target_id)

    def _serialize_entity(self, entity: BaseEntity) -> dict[str, Any]:
        """Serialize an entity to a JSON-compatible dictionary."""
        return {
            "entity_id": entity.entity_id,
            "status": entity.status.value,
            "name": entity.name,
            "synonyms": list(entity.synonyms),
            "entity_type": entity.get_entity_type(),
            "canonical_ids": entity.canonical_ids,
            "embedding": list(entity.embedding) if entity.embedding else None,
            "confidence": entity.confidence,
            "usage_count": entity.usage_count,
            "created_at": entity.created_at.isoformat(),
            "source": entity.source,
            "metadata": entity.metadata,
        }

    def _serialize_relationship(self, rel: BaseRelationship) -> dict[str, Any]:
        """Serialize a relationship to a JSON-compatible dictionary."""
        return {
            "subject_id": rel.subject_id,
            "predicate": rel.predicate,
            "object_id": rel.object_id,
            "edge_type": rel.get_edge_type(),
            "confidence": rel.confidence,
            "source_documents": list(rel.source_documents),
            "created_at": rel.created_at.isoformat(),
            "last_updated": rel.last_updated.isoformat() if rel.last_updated else None,
            "metadata": rel.metadata,
        }

    async def export_entities(
        self,
        output_path: str | Path,
        include_provisional: bool = False,
    ) -> int:
        """Export entities to a JSON file.

        By default exports only canonical entities. Set include_provisional=True
        to include all entities.

        Args:
            output_path: Path to write the JSON file
            include_provisional: Whether to include provisional entities

        Returns:
            Number of entities exported
        """
        output_path = Path(output_path)

        if include_provisional:
            entities = await self.entity_storage.list_all()
        else:
            entities = await self.entity_storage.list_all(status=EntityStatus.CANONICAL.value)

        export_data = {
            "domain": self.domain.name,
            "exported_at": datetime.now().isoformat(),
            "entity_count": len(entities),
            "entities": [self._serialize_entity(e) for e in entities],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return len(entities)

    async def export_document(
        self,
        document_id: str,
        output_path: str | Path,
    ) -> dict[str, int]:
        """Export document-specific data to a JSON file.

        Exports relationships from this document and provisional entities
        that were first extracted from this document.

        Args:
            document_id: ID of the document to export
            output_path: Path to write the JSON file

        Returns:
            Dict with counts: {"relationships": N, "provisional_entities": N}
        """
        output_path = Path(output_path)

        # Get document metadata
        document = await self.document_storage.get(document_id)

        # Get relationships from this document
        relationships = await self.relationship_storage.get_by_document(document_id)

        # Get provisional entities from this document
        # Filter by source field matching document_id
        all_provisional = await self.entity_storage.list_all(status=EntityStatus.PROVISIONAL.value)
        provisional_entities = [e for e in all_provisional if e.source == document_id]

        export_data = {
            "document_id": document_id,
            "document_title": document.title if document else None,
            "exported_at": datetime.now().isoformat(),
            "relationship_count": len(relationships),
            "provisional_entity_count": len(provisional_entities),
            "relationships": [self._serialize_relationship(r) for r in relationships],
            "provisional_entities": [self._serialize_entity(e) for e in provisional_entities],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return {
            "relationships": len(relationships),
            "provisional_entities": len(provisional_entities),
        }

    async def export_all(
        self,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Export all data: entities.json and per-document files.

        Creates:
        - entities.json: All canonical entities
        - paper_{document_id}.json: Per-document relationships and provisional entities

        Args:
            output_dir: Directory to write export files

        Returns:
            Summary dict with export statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export canonical entities
        entities_count = await self.export_entities(
            output_dir / "entities.json",
            include_provisional=False,
        )

        # Export each document
        document_ids = await self.document_storage.list_ids(limit=10000)
        document_stats = {}
        for doc_id in document_ids:
            stats = await self.export_document(
                doc_id,
                output_dir / f"paper_{doc_id}.json",
            )
            document_stats[doc_id] = stats

        return {
            "output_dir": str(output_dir),
            "canonical_entities": entities_count,
            "documents_exported": len(document_ids),
            "document_stats": document_stats,
        }
