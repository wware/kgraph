"""Conftest for medlit tests - imports fixtures from main conftest."""

# Import fixtures from main conftest
from tests.conftest import (  # noqa: F401
    entity_storage,
    relationship_storage,
    document_storage,
)
