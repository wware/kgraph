"""Two-pass ingestion orchestrator for the knowledge graph framework."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from kgraph.document import BaseDocument
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
    """Result of processing a single document."""

    document_id: str
    entities_extracted: int
    entities_new: int
    entities_existing: int
    relationships_extracted: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class IngestionResult:
    """Result of batch document ingestion."""

    documents_processed: int
    documents_failed: int
    total_entities_extracted: int
    total_relationships_extracted: int
    document_results: tuple[DocumentResult, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class IngestionOrchestrator:
    """Orchestrates two-pass document ingestion.

    Pass 1: Extract entities from documents, resolve to canonical IDs,
            update entity collection with new or incremented counts.
    Pass 2: Extract relationships between entities, store per-document.
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
                entity, confidence = await self.entity_resolver.resolve(
                    mention, self.entity_storage
                )

                # Validate entity against domain schema
                if not self.domain.validate_entity(entity):
                    errors.append(
                        f"Entity validation failed: {entity.name} ({entity.get_entity_type()})"
                    )
                    continue

                # Check if entity already exists
                existing = await self.entity_storage.get(entity.entity_id)
                if existing:
                    # Increment usage count
                    updated = existing.model_copy(
                        update={"usage_count": existing.usage_count + 1}
                    )
                    await self.entity_storage.update(updated)
                    resolved_entities.append(updated)
                    entities_existing += 1
                else:
                    # Generate embedding if needed and not present
                    if entity.embedding is None:
                        try:
                            embedding = await self.embedding_generator.generate(
                                entity.name
                            )
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
                extracted_rels = await self.relationship_extractor.extract(
                    document, resolved_entities
                )
                for rel in extracted_rels:
                    if self.domain.validate_relationship(rel):
                        await self.relationship_storage.add(rel)
                        relationships.append(rel)
                    else:
                        errors.append(
                            f"Relationship validation failed: {rel.subject_id} -{rel.predicate}-> {rel.object_id}"
                        )
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
                result = await self.ingest_document(
                    raw_content, content_type, source_uri
                )
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
        """Run entity promotion based on domain's promotion config.

        Finds provisional entities eligible for promotion and promotes them.
        Returns list of promoted entities.
        """
        config = self.domain.promotion_config
        candidates = await self.entity_storage.find_provisional_for_promotion(
            min_usage=config.min_usage_count,
            min_confidence=config.min_confidence,
        )

        promoted: list[BaseEntity] = []
        for entity in candidates:
            # Check embedding requirement
            if config.require_embedding and entity.embedding is None:
                continue

            # Domain-specific promotion logic would generate canonical ID
            # For now, we just mark as canonical with same ID
            # Real implementations would call an external service
            promoted_entity = await self.entity_storage.promote(
                entity.entity_id, entity.entity_id
            )
            if promoted_entity:
                promoted.append(promoted_entity)

        return promoted

    async def find_merge_candidates(
        self,
        similarity_threshold: float = 0.95,
    ) -> list[tuple[BaseEntity, BaseEntity, float]]:
        """Find canonical entities that may be duplicates.

        Returns list of (entity1, entity2, similarity) tuples for entities
        with embeddings above the similarity threshold.
        """
        candidates: list[tuple[BaseEntity, BaseEntity, float]] = []

        # Get all entities (in production, would be paginated/streamed)
        # This is a simple O(n^2) implementation for testing
        all_entities: list[BaseEntity] = []
        entity_count = await self.entity_storage.count()

        # For testing, collect entities via find_by_name with empty string
        # Real implementation would have a list_all method
        # This is a limitation of the current interface

        return candidates

    async def merge_entities(
        self,
        source_ids: Sequence[str],
        target_id: str,
    ) -> bool:
        """Merge multiple entities into one.

        Updates all relationship references and combines entity data.
        """
        # Update relationship references
        for source_id in source_ids:
            if source_id != target_id:
                await self.relationship_storage.update_entity_references(
                    source_id, target_id
                )

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
            "canonical_id_source": entity.get_canonical_id_source(),
            "embedding": list(entity.embedding) if entity.embedding else None,
            "dbpedia_uri": entity.dbpedia_uri,
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
            entities = await self.entity_storage.list_all(
                status=EntityStatus.CANONICAL.value
            )

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
        all_provisional = await self.entity_storage.list_all(
            status=EntityStatus.PROVISIONAL.value
        )
        provisional_entities = [
            e for e in all_provisional
            if e.source == document_id
        ]

        export_data = {
            "document_id": document_id,
            "document_title": document.title if document else None,
            "exported_at": datetime.now().isoformat(),
            "relationship_count": len(relationships),
            "provisional_entity_count": len(provisional_entities),
            "relationships": [self._serialize_relationship(r) for r in relationships],
            "provisional_entities": [
                self._serialize_entity(e) for e in provisional_entities
            ],
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
