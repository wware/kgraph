from __future__ import annotations

from typing import Any, Mapping
import uuid

from pydantic import BaseModel, model_validator, ConfigDict, Field

from kgraph.clock import IngestionClock
from kgraph.document import BaseDocument
from kgraph.domain import DomainSchema, Evidence
from kgraph.entity import BaseEntity, EntityStatus, EntityMention
from kgraph.relationship import BaseRelationship
from kgraph.storage.interfaces import EntityStorageInterface # Added import


def _strip_nonempty(s: str, *, what: str) -> str:
    s2 = s.strip()
    if not s2:
        raise ValueError(f"{what} must be a non-empty string")
    return s2


def _dedupe_preserve_order(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = x.strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)


class EntityBuilder(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    domain: DomainSchema
    clock: IngestionClock
    document: BaseDocument
    entity_storage: EntityStorageInterface | None = Field(default=None, repr=False)

    provisional_prefix: str = "prov:"

    @model_validator(mode="after")
    def _validate(self) -> "EntityBuilder":
        if not self.document.document_id.strip():
            raise ValueError("document.document_id must be non-empty")
        return self

    def _cls_for_type(self, entity_type: str) -> type[BaseEntity]:
        try:
            return self.domain.entity_types[entity_type]
        except KeyError as e:
            raise ValueError(f"Unknown entity_type {entity_type!r} for domain {self.domain.name!r}") from e

    def canonical(
        self,
        *,
        entity_type: str,
        entity_id: str,
        name: str,
        confidence: float = 1.0,
        synonyms: tuple[str, ...] = (),
        canonical_ids: Mapping[str, str] | None = None,
        embedding: tuple[float, ...] | None = None,
        source: str = "curated",
        metadata: dict[str, Any] | None = None,
        evidence: Evidence | None = None,  # if/when BaseEntity adds this field
    ) -> BaseEntity:
        cls = self._cls_for_type(entity_type)

        synonyms = _dedupe_preserve_order(tuple(s.strip() for s in synonyms if s.strip()))
        ent = cls(
            entity_id=_strip_nonempty(entity_id, what="entity_id"),
            status=EntityStatus.CANONICAL,
            name=_strip_nonempty(name, what="name"),
            synonyms=synonyms,
            embedding=embedding,
            canonical_ids=dict(canonical_ids or {}),
            confidence=confidence,
            usage_count=1,
            created_at=self.clock.now,
            source=source,
            metadata={} if metadata is None else metadata,
            # evidence=evidence or self._default_evidence(kind="curated"),  # uncomment when BaseEntity has it
        )

        if not self.domain.validate_entity(ent):
            raise ValueError(f"Domain {self.domain.name!r} rejected entity {ent.entity_id!r}")
        return ent

    def provisional_from_mention(
        self,
        *,
        entity_type: str,
        mention: EntityMention,
        confidence: float | None = None,
        entity_id: str | None = None,
        embedding: tuple[float, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        evidence: Evidence | None = None,  # if/when BaseEntity adds this field
    ) -> BaseEntity:
        cls = self._cls_for_type(entity_type)

        # Default provisional ID: deterministic-ish and traceable
        # pid = entity_id or f"{self.provisional_prefix}{entity_type}:{mention.text}:{mention.start_offset}:{mention.end_offset}:{self.document.document_id}"
        pid = entity_id or f"{self.provisional_prefix}{uuid.uuid4().hex}"

        conf = float(confidence) if confidence is not None else float(getattr(mention, "confidence", 0.5))

        ent = cls(
            entity_id=_strip_nonempty(pid, what="entity_id"),
            status=EntityStatus.PROVISIONAL,
            name=_strip_nonempty(mention.text, what="mention.text"),
            synonyms=(),
            embedding=embedding,
            canonical_ids={},
            confidence=conf,
            usage_count=1,
            created_at=self.clock.now,
            source=self.document.document_id,
            metadata={} if metadata is None else metadata,
            # evidence=evidence or self._evidence_from_mention(mention),  # uncomment when BaseEntity has it
        )

        if not self.domain.validate_entity(ent):
            raise ValueError(f"Domain {self.domain.name!r} rejected provisional entity {ent.entity_id!r}")
        return ent

    # Optional helpers once BaseEntity supports evidence
    def _evidence_from_mention(self, mention: EntityMention) -> Evidence:
        Ev = self.domain.evidence_model
        Prov = self.domain.provenance_model
        primary = Prov(
            document_id=self.document.document_id,
            source_uri=self.document.source_uri,
            section=None,
            start_offset=mention.start_offset,
            end_offset=mention.end_offset,
        )
        return Ev(
            kind="extracted",
            source_documents=(self.document.document_id,),
            primary=primary,
            mentions=(primary,),
            notes={},
        )

    def _default_evidence(self, *, kind: str) -> Evidence:
        Ev = self.domain.evidence_model
        return Ev(
            kind=kind,
            source_documents=(self.document.document_id,),
            primary=None,
            mentions=(),
            notes={},
        )


class RelationshipBuilder(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    domain: DomainSchema
    clock: IngestionClock
    document: BaseDocument
    entity_storage: EntityStorageInterface | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def _validate(self) -> "RelationshipBuilder":
        if not self.document.document_id.strip():
            raise ValueError("document.document_id must be non-empty")
        return self

    def _cls_for_predicate(self, predicate: str) -> type[BaseRelationship]:
        try:
            return self.domain.relationship_types[predicate]
        except KeyError as e:
            raise ValueError(f"Unknown predicate {predicate!r} for domain {self.domain.name!r}") from e

    def link(
        self,
        *,
        predicate: str,
        subject_id: str,
        object_id: str,
        confidence: float = 0.8,
        source_documents: tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        evidence: Evidence | None = None,  # if/when BaseRelationship adds this field
    ) -> BaseRelationship:
        cls = self._cls_for_predicate(predicate)

        docs = source_documents if source_documents is not None else (self.document.document_id,)
        docs = _dedupe_preserve_order(tuple(docs))

        if len(docs) == 0:
            raise ValueError("source_documents must be non-empty")

        rel = cls(
            subject_id=_strip_nonempty(subject_id, what="subject_id"),
            predicate=_strip_nonempty(predicate, what="predicate"),
            object_id=_strip_nonempty(object_id, what="object_id"),
            confidence=confidence,
            source_documents=docs,
            created_at=self.clock.now,
            last_updated=None,
            metadata={} if metadata is None else metadata,
            # evidence=evidence or self._default_evidence(kind="extracted"),  # uncomment when BaseRelationship has it
        )

        if not self.domain.validate_relationship(rel, entity_storage=self.entity_storage):
            raise ValueError(f"Domain {self.domain.name!r} rejected relationship {predicate!r} " f"({subject_id!r} -> {object_id!r})")
        return rel

    def _default_evidence(self, *, kind: str) -> Evidence:
        Ev = self.domain.evidence_model
        return Ev(
            kind=kind,
            source_documents=(self.document.document_id,),
            primary=None,
            mentions=(),
            notes={},
        )
