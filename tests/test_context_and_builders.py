import pytest
from datetime import datetime, timezone

from kgraph.clock import IngestionClock
from kgraph.builders import EntityBuilder, RelationshipBuilder
from kgraph.context import IngestionContext
from kgraph.entity import EntityMention, EntityStatus


def make_doc(doc_id: str = "doc-1", content: str = "hello world"):
    # Use the concrete class from conftest via the domain registry
    # (avoids importing SimpleDocument directly)
    from tests.conftest import SimpleDomainSchema  # ok in tests

    d = SimpleDomainSchema().document_types["test_document"]
    return d(
        document_id=doc_id,
        content=content,
        content_type="text/plain",
        source_uri="unit:test",
        created_at=datetime.now(timezone.utc),
    )


def make_ctx(test_domain, *, doc_id="doc-1", content="hello world"):
    doc = test_domain.document_types["test_document"](
        document_id=doc_id,
        content=content,
        content_type="text/plain",
        source_uri="unit:test",
        created_at=datetime.now(timezone.utc),
    )
    clock = IngestionClock(now=datetime.now(timezone.utc))
    eb = EntityBuilder(domain=test_domain, clock=clock, document=doc)
    rb = RelationshipBuilder(domain=test_domain, clock=clock, document=doc)
    return IngestionContext(domain=test_domain, clock=clock, document=doc, entities=eb, relationships=rb)


# -----------------------
# IngestionClock tests
# -----------------------


def test_ingestion_clock_requires_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        IngestionClock(now=datetime(2026, 1, 1, 12, 0, 0))  # naive


def test_ingestion_clock_accepts_timezone_aware():
    clk = IngestionClock(now=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert clk.now.tzinfo is not None


# -----------------------
# IngestionContext tests
# -----------------------


def test_context_accepts_consistent_builders(test_domain):
    ctx = make_ctx(test_domain)
    assert ctx.domain is test_domain
    assert ctx.entities.domain is test_domain
    assert ctx.relationships.domain is test_domain


def test_context_rejects_mismatched_domain(test_domain):
    ctx = make_ctx(test_domain)
    other_domain = type(test_domain)()

    with pytest.raises(ValueError, match="Entities domain do not match"):
        IngestionContext(
            domain=other_domain,
            clock=ctx.clock,
            document=ctx.document,
            entities=ctx.entities,
            relationships=ctx.relationships,
        )


def test_context_rejects_mismatched_clock(test_domain):
    doc = test_domain.document_types["test_document"](
        document_id="doc-1",
        content="abc",
        content_type="text/plain",
        source_uri=None,
        created_at=datetime.now(timezone.utc),
    )
    clock1 = IngestionClock(now=datetime.now(timezone.utc))
    clock2 = IngestionClock(now=datetime.now(timezone.utc))

    eb = EntityBuilder(domain=test_domain, clock=clock1, document=doc)
    rb = RelationshipBuilder(domain=test_domain, clock=clock1, document=doc)

    with pytest.raises(ValueError, match="Entities clock do not match|Relationships clock do not match"):
        IngestionContext(domain=test_domain, clock=clock2, document=doc, entities=eb, relationships=rb)


def test_context_rejects_mismatched_document(test_domain):
    doc1 = test_domain.document_types["test_document"](
        document_id="doc-1",
        content="abc",
        content_type="text/plain",
        source_uri=None,
        created_at=datetime.now(timezone.utc),
    )
    doc2 = test_domain.document_types["test_document"](
        document_id="doc-2",
        content="abc",
        content_type="text/plain",
        source_uri=None,
        created_at=datetime.now(timezone.utc),
    )
    clock = IngestionClock(now=datetime.now(timezone.utc))
    eb = EntityBuilder(domain=test_domain, clock=clock, document=doc1)
    rb = RelationshipBuilder(domain=test_domain, clock=clock, document=doc1)

    with pytest.raises(ValueError, match="Entities document do not match|Relationships document do not match"):
        IngestionContext(domain=test_domain, clock=clock, document=doc2, entities=eb, relationships=rb)


# -----------------------
# provenance / evidence helpers
# -----------------------


def test_context_provenance_none_offsets_ok(test_domain):
    ctx = make_ctx(test_domain, content="abcdef")
    p = ctx.provenance()
    assert p.document_id == ctx.document.document_id
    assert p.start_offset is None
    assert p.end_offset is None


def test_context_provenance_requires_both_offsets(test_domain):
    ctx = make_ctx(test_domain, content="abcdef")
    with pytest.raises(ValueError, match="provided together"):
        ctx.provenance(start_offset=1)


def test_context_provenance_validates_ranges(test_domain):
    ctx = make_ctx(test_domain, content="abcdef")
    with pytest.raises(ValueError, match=">= 0"):
        ctx.provenance(start_offset=-1, end_offset=2)

    with pytest.raises(ValueError, match=">= start_offset"):
        ctx.provenance(start_offset=3, end_offset=2)

    with pytest.raises(ValueError, match="out of range"):
        ctx.provenance(start_offset=0, end_offset=999)


def test_context_evidence_defaults_to_current_document(test_domain):
    ctx = make_ctx(test_domain)
    ev = ctx.evidence()
    assert ev.source_documents == (ctx.document.document_id,)
    assert ev.kind == "extracted"


# -----------------------
# EntityBuilder tests
# -----------------------


def test_entity_builder_canonical_sets_fields(test_domain):
    ctx = make_ctx(test_domain)
    ent = ctx.entities.canonical(entity_type="test_entity", entity_id="E1", name="Alice")
    assert ent.entity_id == "E1"
    assert ent.status == EntityStatus.CANONICAL
    assert ent.name == "Alice"
    assert ent.usage_count == 1
    assert ent.created_at == ctx.clock.now


def test_entity_builder_canonical_rejects_unknown_type(test_domain):
    ctx = make_ctx(test_domain)
    with pytest.raises(ValueError, match="Unknown entity_type"):
        ctx.entities.canonical(entity_type="nope", entity_id="E1", name="Alice")


def test_entity_builder_canonical_rejects_blank_id(test_domain):
    ctx = make_ctx(test_domain)
    with pytest.raises(ValueError, match="entity_id must be a non-empty"):
        ctx.entities.canonical(entity_type="test_entity", entity_id="   ", name="Alice")


def test_entity_builder_provisional_from_mention_sets_fields(test_domain):
    ctx = make_ctx(test_domain)
    mention = EntityMention(text="Bob", entity_type="test_entity", start_offset=1, end_offset=4, confidence=0.9)
    ent = ctx.entities.provisional_from_mention(entity_type="test_entity", mention=mention)
    assert ent.status == EntityStatus.PROVISIONAL
    assert ent.name == "Bob"
    assert ent.source == ctx.document.document_id
    assert ent.usage_count == 1
    assert ent.entity_id.startswith(ctx.entities.provisional_prefix)


def test_entity_builder_domain_validation_called(test_domain, monkeypatch):
    ctx = make_ctx(test_domain)

    called = {"n": 0}
    orig = test_domain.validate_entity

    def wrapper(e):
        called["n"] += 1
        return orig(e)

    monkeypatch.setattr(test_domain, "validate_entity", wrapper)
    ctx.entities.canonical(entity_type="test_entity", entity_id="E1", name="Alice")
    assert called["n"] == 1


# -----------------------
# RelationshipBuilder tests
# -----------------------


async def test_relationship_builder_link_sets_fields(test_domain):
    ctx = make_ctx(test_domain)
    rel = await ctx.relationships.link(predicate="related_to", subject_id="E1", object_id="E2")
    assert rel.predicate == "related_to"
    assert rel.subject_id == "E1"
    assert rel.object_id == "E2"
    assert rel.source_documents == (ctx.document.document_id,)
    assert rel.created_at == ctx.clock.now


async def test_relationship_builder_rejects_unknown_predicate(test_domain):
    ctx = make_ctx(test_domain)
    with pytest.raises(ValueError, match="Unknown predicate"):
        await ctx.relationships.link(predicate="nope", subject_id="E1", object_id="E2")


async def test_relationship_builder_dedupes_source_documents(test_domain):
    ctx = make_ctx(test_domain)
    rel = await ctx.relationships.link(
        predicate="related_to",
        subject_id="E1",
        object_id="E2",
        source_documents=(ctx.document.document_id, "  ", ctx.document.document_id, "doc-2"),
    )
    assert rel.source_documents == (ctx.document.document_id, "doc-2")


async def test_relationship_builder_rejects_empty_source_documents(test_domain):
    ctx = make_ctx(test_domain)
    with pytest.raises(ValueError, match="source_documents must be non-empty"):
        await ctx.relationships.link(predicate="related_to", subject_id="E1", object_id="E2", source_documents=("   ",))
