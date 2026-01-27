"""
REST API router for the Medical Literature Knowledge Graph.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from storage.models.entity import Entity
from storage.models.relationship import Relationship
from storage.interfaces import StorageInterface

from ..storage_factory import get_storage

router = APIRouter(prefix="/api/v1")


@router.get(
    "/entities/{entity_id}",
    response_model=Entity,
    summary="Get a single entity by its canonical ID",
)
async def get_entity_by_id(entity_id: str, storage: StorageInterface = Depends(get_storage)):
    """
    Retrieve a single medical entity (e.g., Disease, Gene, Drug) by its
    canonical identifier (e.g., UMLS ID, HGNC ID).
    """
    entity = storage.get_entity(entity_id=entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@router.get(
    "/entities",
    response_model=List[Entity],
    summary="List all entities",
)
async def list_entities(limit: int = 100, offset: int = 0, storage: StorageInterface = Depends(get_storage)):
    """
    List all medical entities in the knowledge graph.

    - **limit**: Maximum number of entities to return.
    - **offset**: Number of entities to skip for pagination.
    """
    return storage.get_entities(limit=limit, offset=offset)


@router.get(
    "/relationships",
    response_model=List[Relationship],
    summary="Find relationships between entities",
)
async def find_relationships(
    subject_id: Optional[str] = None,
    predicate: Optional[str] = None,
    object_id: Optional[str] = None,
    limit: int = 100,
    storage: StorageInterface = Depends(get_storage),
):
    """
    Find relationships based on subject, predicate, or object.

    - **subject_id**: Canonical ID of the subject entity.
    - **predicate**: Type of the relationship (e.g., 'TREATS', 'CAUSES').
    - **object_id**: Canonical ID of the object entity.
    - **limit**: Maximum number of relationships to return.
    """
    relationships = storage.find_relationships(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        limit=limit,
    )
    return relationships
