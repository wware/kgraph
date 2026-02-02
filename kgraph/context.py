from __future__ import annotations

from pydantic import BaseModel, model_validator, ConfigDict

from kgraph.builders import EntityBuilder, RelationshipBuilder
from kgraph.clock import IngestionClock

from kgschema.document import BaseDocument
from kgschema.domain import DomainSchema, Evidence, Provenance


class IngestionContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    domain: DomainSchema
    clock: IngestionClock
    document: BaseDocument

    entities: EntityBuilder
    relationships: RelationshipBuilder

    @model_validator(mode="after")
    def domain_is_consistent(self) -> "IngestionContext":
        if self.domain is not self.entities.domain:
            raise ValueError("Entities domain do not match this context")
        if self.domain is not self.relationships.domain:
            raise ValueError("Relationships domain do not match this context")
        if self.clock is not self.entities.clock:
            raise ValueError("Entities clock do not match this context")
        if self.clock is not self.relationships.clock:
            raise ValueError("Relationships clock do not match this context")
        if self.document is not self.entities.document:
            raise ValueError("Entities document do not match this context")
        if self.document is not self.relationships.document:
            raise ValueError("Relationships document do not match this context")
        return self

    def provenance(
        self,
        *,
        start_offset: int | None = None,
        end_offset: int | None = None,
        section: str | None = None,
    ) -> Provenance:
        if start_offset is None and end_offset is None:
            pass
        else:
            if start_offset is None or end_offset is None:
                raise ValueError("start_offset and end_offset must be provided together")
            if start_offset < 0:
                raise ValueError("start_offset must be >= 0")
            if end_offset < start_offset:
                raise ValueError("end_offset must be >= start_offset")
            if end_offset > len(self.document.content):
                raise ValueError("end_offset out of range for document content")
        model = self.domain.provenance_model
        return model(
            document_id=self.document.document_id,
            source_uri=self.document.source_uri,
            section=section,
            start_offset=start_offset,
            end_offset=end_offset,
        )

    # something similar for evidence???
    def evidence(
        self,
        *,
        kind: str = "extracted",
        primary: Provenance | None = None,
        mentions: tuple[Provenance, ...] = (),
        source_documents: tuple[str, ...] | None = None,
        notes: dict[str, object] | None = None,
    ) -> Evidence:
        # by default, evidence comes from the current document
        docs = source_documents if source_documents is not None else (self.document.document_id,)
        ev = self.domain.evidence_model
        return ev(
            kind=kind,
            source_documents=docs,
            primary=primary,
            mentions=mentions,
            notes={} if notes is None else notes,
        )
