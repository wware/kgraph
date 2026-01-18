"""
Knowledge Graph Framework - Domain-Agnostic Entity and Relationship Extraction.

A flexible framework for building knowledge graphs across multiple domains
(medical, legal, CS papers, etc.) with a two-pass ingestion process.
"""

from kgraph.entity import (
    BaseEntity,
    EntityMention,
    EntityStatus,
    PromotionConfig,
)
from kgraph.relationship import BaseRelationship
from kgraph.document import BaseDocument
from kgraph.domain import DomainSchema
from kgraph.ingest import IngestionOrchestrator, IngestionResult

__all__ = [
    "BaseEntity",
    "EntityMention",
    "EntityStatus",
    "PromotionConfig",
    "BaseRelationship",
    "BaseDocument",
    "DomainSchema",
    "IngestionOrchestrator",
    "IngestionResult",
]

__version__ = "0.1.0"
