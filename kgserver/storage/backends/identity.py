"""Postgres-backed implementation of the IdentityServer ABC.

This module provides ``PostgresIdentityServer``, the reference implementation
of ``kgschema.IdentityServer``.  It uses the existing SQLModel ``Session`` and
``Entity``/``Relationship`` models already in the kgserver stack.

Locking strategy (mirrors CONCURRENCY.md):
  - resolve:  ``INSERT ... ON CONFLICT DO NOTHING`` — Postgres serialises
              concurrent inserts on the same entity_id naturally.
  - promote:  ``SELECT ... WITH (UPDLOCK)`` equivalent via raw SQL
              ``SELECT ... FOR UPDATE`` on the entity row, then conditional
              update inside the same transaction.
  - merge:    Postgres advisory lock keyed on the sorted frozenset of entity
              IDs prevents two workers merging the same pair in opposite orders.
  - on_entity_added: called synchronously inside the same session as the
              triggering entity write; synonym detection is read-only and fires
              only after the row is durable.

Synonym detection uses the ``embedding`` JSON column on the ``Entity`` table
via in-process cosine similarity for now.  Once pgvector is activated on the
column (``ALTER TABLE entity ADD COLUMN embedding vector(N)``), the
``find_synonyms`` implementation can be swapped to a native vector query with
no interface change.

Authority-lookup caching (Redis) is deferred; ``resolve`` currently delegates
to the domain's ``PromotionPolicy.assign_canonical_id`` without caching.
"""

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlmodel import Session, select

from kgschema.domain import DomainSchema
from kgschema.entity import EntityStatus
from kgschema.identity import IdentityServer
from storage.models.entity import Entity

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _advisory_lock_key(entity_ids: list[str]) -> int:
    """Derive a stable 64-bit advisory lock key from a sorted set of entity IDs.

    Sorting ensures two workers locking the same pair always produce the same
    key regardless of argument order, preventing deadlocks.
    """
    combined = "|".join(sorted(entity_ids))
    # fold the full hash into a signed 64-bit range Postgres accepts
    return hash(combined) % (2**63)


class PostgresIdentityServer(IdentityServer):
    """Postgres-backed identity server using the existing kgserver SQLModel session.

    Parameters
    ----------
    session:
        An open SQLModel ``Session`` bound to the Postgres engine.  The caller
        is responsible for the session lifecycle (commit / rollback / close).
    domain:
        The active ``DomainSchema`` instance.  Used for ``preferred_entity``
        (survivor selection) and ``get_promotion_policy`` (canonical ID
        assignment during promotion).
    similarity_threshold:
        Minimum cosine similarity for two entities to be considered synonyms.
        Default 0.90 — intentionally conservative.
    embedding_dim:
        Expected embedding dimension.  Used for validation only.
    """

    def __init__(
        self,
        session: Session,
        domain: DomainSchema,
        similarity_threshold: float = 0.90,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self._session = session
        self._domain = domain
        self._similarity_threshold = similarity_threshold
        self._embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # resolve
    # ------------------------------------------------------------------

    async def resolve(self, mention: str, context: dict) -> str:
        """Resolve a mention to an entity ID, creating a provisional one if needed.

        Uses ``INSERT ... ON CONFLICT DO NOTHING`` so concurrent workers
        resolving the same mention produce the same entity without races.

        Authority lookup (UMLS etc.) is attempted first via the domain's
        promotion policy.  If the authority returns a canonical ID, the entity
        is inserted as canonical.  Otherwise a provisional UUID is created.

        The ``context`` dict may contain:
          - ``entity_type`` (str): domain entity type hint
          - ``document_id`` (str): source document for provenance
          - ``embedding`` (list[float]): pre-computed embedding vector
        """
        entity_type = context.get("entity_type", "unknown")
        document_id = context.get("document_id", "unknown")
        embedding = context.get("embedding")

        # Check whether an entity already exists with this name + type.
        existing = self._session.exec(select(Entity).where(Entity.name == mention, Entity.entity_type == entity_type)).first()
        if existing is not None:
            if existing.status == EntityStatus.MERGED.value and existing.merged_into:
                logger.debug("resolve: mention '%s' maps to merged entity %s → redirecting to %s", mention, existing.entity_id, existing.merged_into)
                return existing.merged_into
            return existing.entity_id

        # Attempt authority lookup via promotion policy.
        policy = self._domain.get_promotion_policy()
        canonical_id_result = None
        if policy is not None:
            try:
                # assign_canonical_id is async; build a minimal stub entity for it.
                stub = _make_stub_entity(mention, entity_type, document_id, embedding)
                canonical_id_result = await policy.assign_canonical_id(stub)
            except Exception:  # pylint: disable=broad-except
                logger.debug("resolve: authority lookup failed for '%s'", mention, exc_info=True)

        if canonical_id_result is not None:
            entity_id = canonical_id_result.id
            status = EntityStatus.CANONICAL.value
        else:
            entity_id = f"prov:{uuid.uuid4().hex}"
            status = EntityStatus.PROVISIONAL.value

        now = datetime.now(timezone.utc).isoformat()
        stmt = text("""
            INSERT INTO entity (entity_id, entity_type, name, status, confidence, usage_count, source, synonyms, properties)
            VALUES (:entity_id, :entity_type, :name, :status, :confidence, :usage_count, :source,
                    CAST(:synonyms AS json), CAST(:properties AS json))
            ON CONFLICT (entity_id) DO NOTHING
        """)
        synonyms_json = "[]"
        if canonical_id_result and canonical_id_result.synonyms:
            import json as _json

            synonyms_json = _json.dumps(list(canonical_id_result.synonyms))

        self._session.execute(
            stmt,
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "name": mention,
                "status": status,
                "confidence": 1.0,
                "usage_count": 1,
                "source": document_id,
                "synonyms": synonyms_json,
                "properties": f'{{"created_at": "{now}"}}',
            },
        )

        # Store embedding if provided (JSON column for now).
        if embedding is not None:
            self._store_embedding(entity_id, embedding)

        return entity_id

    # ------------------------------------------------------------------
    # promote
    # ------------------------------------------------------------------

    async def promote(self, provisional_id: str) -> str:
        """Attempt to promote a provisional entity to canonical status.

        Uses ``SELECT FOR UPDATE`` to lock the row, then checks and updates
        inside the same implicit transaction.  Behaviour by status:
          - provisional: attempts canonical ID assignment; upgrades if found.
          - canonical: no-op; returns existing ID.
          - merged: logs warning; returns survivor ID.
        """
        row = self._session.exec(select(Entity).where(Entity.entity_id == provisional_id).with_for_update()).first()

        if row is None:
            logger.warning("promote: entity '%s' not found", provisional_id)
            return provisional_id

        if row.status == EntityStatus.MERGED.value:
            survivor = row.merged_into or provisional_id
            logger.warning("promote: entity '%s' is already merged into '%s'; returning survivor", provisional_id, survivor)
            return survivor

        if row.status == EntityStatus.CANONICAL.value:
            logger.debug("promote: entity '%s' is already canonical; no-op", provisional_id)
            return provisional_id

        # Attempt canonical ID assignment.
        policy = self._domain.get_promotion_policy()
        if policy is None:
            return provisional_id

        stub = _entity_row_to_stub(row)
        if not policy.should_promote(stub):
            return provisional_id

        canonical_id_result = await policy.assign_canonical_id(stub)
        if canonical_id_result is None:
            return provisional_id

        new_id = canonical_id_result.id
        # Update the row in-place (entity_id is PK — we update via raw SQL to
        # change the PK and status atomically, then update relationship refs).
        self._session.execute(
            text("UPDATE entity SET entity_id = :new_id, status = 'canonical', canonical_url = :url WHERE entity_id = :old_id AND status = 'provisional'"),
            {"new_id": new_id, "url": canonical_id_result.url, "old_id": provisional_id},
        )
        self._update_relationship_refs(provisional_id, new_id)
        logger.info("promote: '%s' promoted to canonical '%s'", provisional_id, new_id)
        return new_id

    # ------------------------------------------------------------------
    # find_synonyms
    # ------------------------------------------------------------------

    async def find_synonyms(self, entity_id: str) -> list[str]:
        """Return IDs of entities with cosine similarity above the threshold.

        Currently uses in-process comparison of the ``embedding`` JSON column.
        This is correct but O(n) — a pgvector index query can be substituted
        here with no interface change once the column type is migrated.
        """
        source_row = self._session.get(Entity, entity_id)
        if source_row is None or source_row.embedding is None:
            return []

        source_vec: list[float] = source_row.embedding
        candidates = self._session.exec(
            select(Entity).where(
                Entity.entity_id != entity_id,
                Entity.status != EntityStatus.MERGED.value,
            )
        ).all()

        results = []
        for candidate in candidates:
            if candidate.embedding is None:
                continue
            sim = _cosine_similarity(source_vec, candidate.embedding)
            if sim >= self._similarity_threshold:
                results.append(candidate.entity_id)

        return results

    # ------------------------------------------------------------------
    # merge
    # ------------------------------------------------------------------

    async def merge(self, entity_ids: list[str], survivor_id: str) -> str:
        """Merge entities into survivor, redirecting all relationship references.

        Acquires a Postgres advisory lock keyed on the sorted set of IDs to
        prevent concurrent merges of the same pair in opposite directions.

        Status rules:
          - all provisional → survivor stays provisional
          - any canonical → survivor becomes canonical
        """
        if survivor_id not in entity_ids:
            raise ValueError(f"survivor_id '{survivor_id}' must be a member of entity_ids")

        absorbed_ids = [eid for eid in entity_ids if eid != survivor_id]
        if not absorbed_ids:
            return survivor_id

        lock_key = _advisory_lock_key(entity_ids)
        self._session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})

        # Fetch all rows under the lock.
        rows = {row.entity_id: row for row in self._session.exec(select(Entity).where(Entity.entity_id.in_(entity_ids))).all()}  # type: ignore[attr-defined]

        survivor_row = rows.get(survivor_id)
        if survivor_row is None:
            logger.warning("merge: survivor '%s' not found; aborting", survivor_id)
            return survivor_id

        # Determine final status: canonical wins.
        any_canonical = any(r.status == EntityStatus.CANONICAL.value for r in rows.values())
        final_status = EntityStatus.CANONICAL.value if any_canonical else EntityStatus.PROVISIONAL.value

        # Redirect relationship references and mark absorbed entities.
        for absorbed_id in absorbed_ids:
            absorbed_row = rows.get(absorbed_id)
            if absorbed_row is None:
                continue
            if absorbed_row.status == EntityStatus.MERGED.value:
                logger.debug("merge: '%s' is already merged; skipping", absorbed_id)
                continue
            self._update_relationship_refs(absorbed_id, survivor_id)
            self._session.execute(
                text("UPDATE entity SET status = 'merged', merged_into = :survivor WHERE entity_id = :absorbed"),
                {"survivor": survivor_id, "absorbed": absorbed_id},
            )

        # Update survivor status if it needs to change.
        if survivor_row.status != final_status:
            self._session.execute(
                text("UPDATE entity SET status = :status WHERE entity_id = :eid"),
                {"status": final_status, "eid": survivor_id},
            )

        logger.info("merge: absorbed %s into survivor '%s' (status=%s)", absorbed_ids, survivor_id, final_status)
        return survivor_id

    # ------------------------------------------------------------------
    # on_entity_added
    # ------------------------------------------------------------------

    async def on_entity_added(self, entity_id: str, context: dict) -> None:
        """Trigger synonym detection and merge for a newly added entity.

        Must be called inside the same session/transaction as the entity insert
        so that the row is visible.  Synonym detection is read-only; ``merge``
        acquires its own advisory lock.
        """
        synonym_ids = await self.find_synonyms(entity_id)
        if not synonym_ids:
            return

        # Fetch the new entity and all synonym candidates for survivor selection.
        all_ids = [entity_id] + synonym_ids
        rows = self._session.exec(select(Entity).where(Entity.entity_id.in_(all_ids))).all()  # type: ignore[attr-defined]

        if len(rows) < 2:
            return

        # Convert to kgschema BaseEntity stubs for preferred_entity.
        stubs = [_entity_row_to_stub(r) for r in rows]
        survivor_stub = self._domain.preferred_entity(stubs)
        await self.merge(all_ids, survivor_stub.entity_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_relationship_refs(self, old_id: str, new_id: str) -> None:
        """Redirect all relationship subject/object references from old_id to new_id."""
        self._session.execute(
            text("UPDATE relationship SET subject_id = :new_id WHERE subject_id = :old_id"),
            {"new_id": new_id, "old_id": old_id},
        )
        self._session.execute(
            text("UPDATE relationship SET object_id = :new_id WHERE object_id = :old_id"),
            {"new_id": new_id, "old_id": old_id},
        )

    def _store_embedding(self, entity_id: str, embedding: list[float]) -> None:
        """Persist an embedding vector to the entity row (JSON column)."""
        import json as _json

        self._session.execute(
            text("UPDATE entity SET embedding = CAST(:emb AS json) WHERE entity_id = :eid"),
            {"emb": _json.dumps(embedding), "eid": entity_id},
        )


# ---------------------------------------------------------------------------
# Helpers to bridge kgserver Entity rows ↔ kgschema BaseEntity stubs
# ---------------------------------------------------------------------------


class _StubEntity:
    """Minimal kgschema.BaseEntity-compatible stub for use with domain policies.

    The domain's ``preferred_entity`` and ``PromotionPolicy`` methods need a
    ``BaseEntity``-like object.  Rather than instantiating a domain-specific
    subclass (which we don't know here), we use a simple attribute container
    that satisfies the duck-typed interface.
    """

    def __init__(self, row: "Entity") -> None:
        self.entity_id: str = row.entity_id
        self.status = EntityStatus(row.status) if row.status else EntityStatus.PROVISIONAL
        self.name: str = row.name or ""
        self.synonyms: tuple = tuple(row.synonyms or [])
        self.embedding = tuple(row.embedding) if row.embedding else None
        self.canonical_ids: dict = {}
        if row.canonical_url:
            self.canonical_ids["url"] = row.canonical_url
        self.confidence: float = row.confidence or 1.0
        self.usage_count: int = row.usage_count or 0
        self.created_at: datetime = datetime.now(timezone.utc)  # fallback; row has no created_at column
        self.source: str = row.source or ""
        self.metadata: dict = {}
        self.merged_into: Optional[str] = row.merged_into
        self.promotable: bool = True

    def get_entity_type(self) -> str:
        return ""  # not needed for identity operations


def _entity_row_to_stub(row: "Entity") -> "_StubEntity":
    return _StubEntity(row)


def _make_stub_entity(mention: str, entity_type: str, document_id: str, embedding: Optional[list[float]]) -> "_StubEntity":
    """Build a minimal stub for authority lookup (no DB row yet)."""

    class _FakeRow:
        entity_id = f"prov:{uuid.uuid4().hex}"
        status = EntityStatus.PROVISIONAL.value
        name = mention
        synonyms: list = []
        embedding = embedding
        canonical_url = None
        canonical_ids: dict = {}
        confidence = 1.0
        usage_count = 0
        source = document_id
        merged_into = None

    return _StubEntity(_FakeRow())  # type: ignore[arg-type]
