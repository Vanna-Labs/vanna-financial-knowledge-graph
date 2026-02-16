"""Tests for Assembler batch storage writing."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from vanna_kg.ingestion.assembly.assembler import Assembler
from vanna_kg.types import EntityType
from vanna_kg.types.chunks import Chunk, Document
from vanna_kg.types.facts import Fact
from vanna_kg.types.results import AssemblyInput, AssemblyResult, CanonicalEntity


class TestAssemblerEntityTypeMapping:
    """Test entity type mapping from labels to enums."""

    def test_map_company(self):
        """Company label maps to COMPANY enum."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Company") == EntityType.COMPANY

    def test_map_person(self):
        """Person label maps to PERSON enum."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Person") == EntityType.PERSON

    def test_map_organization(self):
        """Organization label maps to ORGANIZATION enum."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Organization") == EntityType.ORGANIZATION

    def test_map_location(self):
        """Location label maps to LOCATION enum."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Location") == EntityType.LOCATION

    def test_map_product(self):
        """Product label maps to PRODUCT enum."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Product") == EntityType.PRODUCT

    def test_map_topic(self):
        """Topic label maps to CONCEPT enum."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Topic") == EntityType.CONCEPT

    def test_map_unknown_defaults_to_concept(self):
        """Unknown labels default to CONCEPT."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Unknown") == EntityType.CONCEPT
        assert assembler._map_entity_type("") == EntityType.CONCEPT


class TestAssemblerCanonicalConversion:
    """Test conversion from CanonicalEntity to Entity."""

    def test_canonical_to_entity_basic(self):
        """CanonicalEntity converts to Entity correctly."""
        assembler = Assembler.__new__(Assembler)
        canonical = CanonicalEntity(
            uuid="test-uuid",
            name="Test Corp",
            entity_type="Company",
            summary="A test company.",
            source_indices=[0, 1],
            aliases=["TC", "TestCo"],
        )

        entities = assembler._canonical_to_entity([canonical])

        assert len(entities) == 1
        assert entities[0].uuid == "test-uuid"
        assert entities[0].name == "Test Corp"
        assert entities[0].entity_type == EntityType.COMPANY
        assert entities[0].aliases == ["TC", "TestCo"]
        assert entities[0].summary == "A test company."

    def test_canonical_to_entity_multiple(self):
        """Multiple CanonicalEntities convert correctly."""
        assembler = Assembler.__new__(Assembler)
        canonicals = [
            CanonicalEntity(
                uuid="uuid-1",
                name="Entity 1",
                entity_type="Person",
                summary="First entity.",
                source_indices=[0],
            ),
            CanonicalEntity(
                uuid="uuid-2",
                name="Entity 2",
                entity_type="Location",
                summary="Second entity.",
                source_indices=[1],
            ),
        ]

        entities = assembler._canonical_to_entity(canonicals)

        assert len(entities) == 2
        assert entities[0].entity_type == EntityType.PERSON
        assert entities[1].entity_type == EntityType.LOCATION


class TestAssemblyResult:
    """Test AssemblyResult structure."""

    def test_default_values(self):
        """AssemblyResult defaults to zeros."""
        result = AssemblyResult()
        assert result.document_written is False
        assert result.chunks_written == 0
        assert result.entities_written == 0
        assert result.facts_written == 0
        assert result.topics_written == 0
        assert result.relationships_written == 0

    def test_with_values(self):
        """AssemblyResult stores values correctly."""
        result = AssemblyResult(
            document_written=True,
            chunks_written=10,
            entities_written=5,
            facts_written=20,
            topics_written=3,
            relationships_written=40,
        )
        assert result.document_written is True
        assert result.chunks_written == 10
        assert result.relationships_written == 40


class TestBuildRelationships:
    """Test relationship building from facts."""

    def test_build_relationships_basic(self):
        """Relationships are direct entity-entity edges with chunk_uuid."""
        assembler = Assembler.__new__(Assembler)

        doc_uuid = str(uuid4())
        chunk_uuid = str(uuid4())
        subject_uuid = str(uuid4())
        object_uuid = str(uuid4())
        fact_uuid = str(uuid4())

        doc = Document(uuid=doc_uuid, name="test.pdf", file_type="pdf")
        fact = Fact(
            uuid=fact_uuid,
            content="Apple acquired Beats.",
            subject_uuid=subject_uuid,
            subject_name="Apple",
            object_uuid=object_uuid,
            object_name="Beats",
            object_type="entity",
            relationship_type="ACQUIRED",
            date_context="2014",
            chunk_uuid=chunk_uuid,
        )

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=[],
            facts=[fact],
            topics=[],
        )

        relationships = assembler._build_relationships(input)

        # Now only 1 relationship per fact (direct edge)
        assert len(relationships) == 1

        # Direct edge: Subject -> Object with chunk_uuid
        rel = relationships[0]
        assert rel["from_uuid"] == subject_uuid
        assert rel["from_type"] == "entity"
        assert rel["to_uuid"] == object_uuid
        assert rel["to_type"] == "entity"
        assert rel["rel_type"] == "ACQUIRED"
        assert rel["chunk_uuid"] == chunk_uuid
        assert rel["fact_id"] == fact_uuid
        assert rel["description"] == "Apple acquired Beats."

    def test_build_relationships_skips_facts_without_chunk(self):
        """Facts without chunk_uuid are skipped."""
        assembler = Assembler.__new__(Assembler)

        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        fact = Fact(
            uuid=str(uuid4()),
            content="Orphan fact.",
            subject_uuid=str(uuid4()),
            subject_name="Subject",
            object_uuid=str(uuid4()),
            object_name="Object",
            object_type="entity",
            relationship_type="RELATED_TO",
            date_context="2025",
            chunk_uuid=None,  # No chunk
        )

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=[],
            facts=[fact],
            topics=[],
        )

        relationships = assembler._build_relationships(input)

        assert len(relationships) == 0

    def test_build_relationships_preserves_date_context(self):
        """Date context is preserved in relationships."""
        assembler = Assembler.__new__(Assembler)

        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        fact = Fact(
            uuid=str(uuid4()),
            content="Revenue in Q4.",
            subject_uuid=str(uuid4()),
            subject_name="Company",
            object_uuid=str(uuid4()),
            object_name="Revenue",
            object_type="entity",
            relationship_type="REPORTED",
            chunk_uuid=str(uuid4()),
            date_context="Q4 2025",
        )

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=[],
            facts=[fact],
            topics=[],
        )

        relationships = assembler._build_relationships(input)

        assert len(relationships) == 1
        assert relationships[0]["date_context"] == "Q4 2025"

    def test_build_relationships_includes_id(self):
        """Each relationship should have a unique id field."""
        assembler = Assembler.__new__(Assembler)

        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        fact1 = Fact(
            uuid=str(uuid4()),
            content="Test fact 1.",
            subject_uuid=str(uuid4()),
            subject_name="Subject1",
            object_uuid=str(uuid4()),
            object_name="Object1",
            object_type="entity",
            relationship_type="RELATED_TO",
            chunk_uuid=str(uuid4()),
            date_context="2025",
        )
        fact2 = Fact(
            uuid=str(uuid4()),
            content="Test fact 2.",
            subject_uuid=str(uuid4()),
            subject_name="Subject2",
            object_uuid=str(uuid4()),
            object_name="Object2",
            object_type="entity",
            relationship_type="ACQUIRED",
            chunk_uuid=str(uuid4()),
            date_context="2025",
        )

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=[],
            facts=[fact1, fact2],
            topics=[],
        )

        relationships = assembler._build_relationships(input)

        # 2 facts = 2 relationships
        assert len(relationships) == 2

        # Both relationships should have id field
        assert "id" in relationships[0]
        assert "id" in relationships[1]

        # IDs should be unique
        assert relationships[0]["id"] != relationships[1]["id"]

        # IDs should be valid UUIDs (36 chars with hyphens)
        assert len(relationships[0]["id"]) == 36
        assert len(relationships[1]["id"]) == 36

    def test_build_relationships_topic_object(self):
        """Facts with object_type='topic' create edges to topics."""
        assembler = Assembler.__new__(Assembler)

        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        chunk_uuid = str(uuid4())
        topic_uuid = str(uuid4())

        fact = Fact(
            uuid=str(uuid4()),
            content="Company discussed tariffs.",
            subject_uuid=str(uuid4()),
            subject_name="Company",
            object_uuid=topic_uuid,
            object_name="Tariffs",
            object_type="topic",
            relationship_type="DISCUSSED",
            chunk_uuid=chunk_uuid,
            date_context="2025",
        )

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=[],
            facts=[fact],
            topics=[],
        )

        relationships = assembler._build_relationships(input)

        assert len(relationships) == 1
        assert relationships[0]["to_type"] == "topic"
        assert relationships[0]["to_uuid"] == topic_uuid
        assert relationships[0]["chunk_uuid"] == chunk_uuid


class TestAssemblerIntegration:
    """Integration tests for Assembler.assemble()."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage backend."""
        storage = AsyncMock()
        storage.write_document = AsyncMock()
        storage.write_chunks = AsyncMock()
        storage.write_entities = AsyncMock()
        storage.write_facts = AsyncMock()
        storage.write_topics = AsyncMock()
        storage.write_relationships = AsyncMock()
        return storage

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embedding provider."""
        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=[[0.1] * 3072])
        return embeddings

    @pytest.mark.asyncio
    async def test_assemble_document_only(self, mock_storage, mock_embeddings):
        """Assembling with only document writes document."""
        doc = Document(
            uuid=str(uuid4()),
            name="test.pdf",
            file_type="pdf",
        )
        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=[],
            facts=[],
            topics=[],
        )

        assembler = Assembler(mock_storage, mock_embeddings)
        result = await assembler.assemble(input)

        assert result.document_written is True
        assert result.chunks_written == 0
        assert result.entities_written == 0
        assert result.facts_written == 0
        assert result.relationships_written == 0
        mock_storage.write_document.assert_called_once_with(doc)

    @pytest.mark.asyncio
    async def test_assemble_full_input(self, mock_storage, mock_embeddings):
        """Assembling with all data writes everything in order."""
        doc_uuid = str(uuid4())
        chunk_uuid = str(uuid4())
        entity1_uuid = str(uuid4())
        entity2_uuid = str(uuid4())
        fact_uuid = str(uuid4())

        doc = Document(uuid=doc_uuid, name="test.pdf", file_type="pdf")
        chunk = Chunk(
            uuid=chunk_uuid,
            content="Test content",
            header_path="# Test",
            position=0,
            document_uuid=doc_uuid,
        )
        entity = CanonicalEntity(
            uuid=entity1_uuid,
            name="Test Corp",
            entity_type="Company",
            summary="A test company.",
            source_indices=[0],
        )
        fact = Fact(
            uuid=fact_uuid,
            content="Test Corp was founded in 2020.",
            subject_uuid=entity1_uuid,
            subject_name="Test Corp",
            object_uuid=entity2_uuid,
            object_name="2020",
            object_type="date",
            relationship_type="FOUNDED_IN",
            date_context="2020",
            chunk_uuid=chunk_uuid,
        )

        input = AssemblyInput(
            document=doc,
            chunks=[chunk],
            entities=[entity],
            facts=[fact],
            topics=[],
        )

        # Mock embeddings to return correct number for each call
        mock_embeddings.embed = AsyncMock(
            side_effect=[
                [[0.1] * 3072],  # entity embeddings
                [[0.2] * 3072],  # fact embeddings
            ]
        )

        assembler = Assembler(mock_storage, mock_embeddings)
        result = await assembler.assemble(input)

        assert result.document_written is True
        assert result.chunks_written == 1
        assert result.entities_written == 1
        assert result.facts_written == 1
        assert result.relationships_written == 1  # Subject->Object (direct edge)

        # Verify all write methods were called
        mock_storage.write_document.assert_called_once()
        mock_storage.write_chunks.assert_called_once()
        mock_storage.write_entities.assert_called_once()
        mock_storage.write_facts.assert_called_once()
        mock_storage.write_relationships.assert_called_once()

    @pytest.mark.asyncio
    async def test_assemble_batches_embeddings(self, mock_storage, mock_embeddings):
        """Embeddings are generated in batch, not per item."""
        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        entities = [
            CanonicalEntity(
                uuid=str(uuid4()),
                name=f"Entity {i}",
                entity_type="Company",
                summary=f"Entity {i} description.",
                source_indices=[i],
            )
            for i in range(5)
        ]

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=entities,
            facts=[],
            topics=[],
        )

        # Return 5 embeddings in one batch call
        mock_embeddings.embed = AsyncMock(return_value=[[0.1] * 3072] * 5)

        assembler = Assembler(mock_storage, mock_embeddings)
        await assembler.assemble(input)

        # Should be called once with 5 texts, not 5 times with 1 text
        assert mock_embeddings.embed.call_count == 1
        call_args = mock_embeddings.embed.call_args[0][0]
        assert len(call_args) == 5

    @pytest.mark.asyncio
    async def test_assemble_no_embed_call_for_empty_lists(
        self, mock_storage, mock_embeddings
    ):
        """Empty entity/fact/topic lists don't trigger embed calls."""
        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        chunk = Chunk(
            uuid=str(uuid4()),
            content="Just a chunk.",
            header_path="# Test",
            position=0,
            document_uuid=doc.uuid,
        )

        input = AssemblyInput(
            document=doc,
            chunks=[chunk],
            entities=[],  # Empty
            facts=[],  # Empty
            topics=[],  # Empty
        )

        assembler = Assembler(mock_storage, mock_embeddings)
        await assembler.assemble(input)

        # embed should not be called for empty lists
        mock_embeddings.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_assemble_embedding_count_mismatch_raises(
        self, mock_storage, mock_embeddings
    ):
        """Embedding count mismatch should raise ValueError."""
        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        entities = [
            CanonicalEntity(
                uuid=str(uuid4()),
                name=f"Entity {i}",
                entity_type="Company",
                summary=f"Entity {i}.",
                source_indices=[i],
            )
            for i in range(5)
        ]

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=entities,  # 5 entities
            facts=[],
            topics=[],
        )

        # Return wrong number of embeddings
        mock_embeddings.embed = AsyncMock(return_value=[[0.1] * 3072] * 3)  # Only 3

        assembler = Assembler(mock_storage, mock_embeddings)

        with pytest.raises(ValueError, match="embedding count mismatch"):
            await assembler.assemble(input)

    @pytest.mark.asyncio
    async def test_assemble_storage_failure_logs_progress(
        self, mock_storage, mock_embeddings
    ):
        """Storage failure should log what was written before failure."""
        doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
        chunk = Chunk(
            uuid=str(uuid4()),
            content="Test content",
            header_path="# Test",
            position=0,
            document_uuid=doc.uuid,
        )

        input = AssemblyInput(
            document=doc,
            chunks=[chunk],
            entities=[],
            facts=[],
            topics=[],
        )

        # Document write succeeds, chunk write fails
        mock_storage.write_document = AsyncMock()
        mock_storage.write_chunks = AsyncMock(side_effect=Exception("Storage error"))

        assembler = Assembler(mock_storage, mock_embeddings)

        with pytest.raises(Exception, match="Storage error"):
            await assembler.assemble(input)
