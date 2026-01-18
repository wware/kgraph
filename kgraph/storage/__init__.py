"""Storage interfaces and implementations for the knowledge graph."""

from kgraph.storage.interfaces import (
    DocumentStorageInterface,
    EntityStorageInterface,
    RelationshipStorageInterface,
)
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)

__all__ = [
    "EntityStorageInterface",
    "RelationshipStorageInterface",
    "DocumentStorageInterface",
    "InMemoryEntityStorage",
    "InMemoryRelationshipStorage",
    "InMemoryDocumentStorage",
]
