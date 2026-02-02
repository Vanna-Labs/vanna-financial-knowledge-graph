"""Tests for EntityRegistry cross-document entity resolution."""

from unittest.mock import AsyncMock

import pytest

from zomma_kg.ingestion.resolution.entity_registry import (
    CANDIDATE_LIMIT,
    HIGH_SIMILARITY_THRESHOLD,
    SIMILARITY_DISPLAY_THRESHOLD,
    EntityRegistry,
)
from zomma_kg.types import Entity, EntityType
from zomma_kg.types.results import (
    CanonicalEntity,
    EntityRegistryMatch,
    EntityResolutionResult,
)


class TestEntityRegistryConstants:
    """Test registry constants are set correctly."""

    def test_candidate_limit(self):
        assert CANDIDATE_LIMIT == 25

    def test_similarity_thresholds(self):
        assert SIMILARITY_DISPLAY_THRESHOLD == 0.50
        assert HIGH_SIMILARITY_THRESHOLD == 0.90
        assert HIGH_SIMILARITY_THRESHOLD > SIMILARITY_DISPLAY_THRESHOLD


class TestEntityToText:
    """Test entity text generation for embeddings."""

    def test_entity_to_text_format(self):
        """Text format should be 'name: summary'."""
        entity = CanonicalEntity(
            uuid="test-uuid",
            name="Apple Inc.",
            entity_type="Company",
            summary="Technology company that makes iPhones.",
            source_indices=[0],
        )

        # Create registry with mocks to test the method
        registry = EntityRegistry.__new__(EntityRegistry)
        actual = registry._entity_to_text(entity)
        expected = "Apple Inc.: Technology company that makes iPhones."
        assert actual == expected

    def test_entity_to_text_empty_summary(self):
        """Empty summary should still produce valid text."""
        entity = CanonicalEntity(
            uuid="test-uuid",
            name="Unknown Corp",
            entity_type="Company",
            summary="",
            source_indices=[0],
        )

        registry = EntityRegistry.__new__(EntityRegistry)
        actual = registry._entity_to_text(entity)
        assert actual == "Unknown Corp: "


class TestBuildMatchPrompt:
    """Test prompt building for LLM verification."""

    def test_prompt_contains_entity_info(self):
        """Prompt should contain the new entity's information."""
        entity = CanonicalEntity(
            uuid="new-uuid",
            name="Apple",
            entity_type="Company",
            summary="Makes iPhones.",
            source_indices=[0],
        )
        candidates = [
            (
                Entity(
                    uuid="existing-uuid",
                    name="Apple Inc.",
                    summary="Technology company.",
                    entity_type=EntityType.COMPANY,
                ),
                0.85,
            )
        ]

        registry = EntityRegistry.__new__(EntityRegistry)
        prompt = registry._build_match_prompt(entity, candidates)

        assert "Apple" in prompt
        assert "Makes iPhones" in prompt
        assert "Company" in prompt

    def test_prompt_contains_candidates(self):
        """Prompt should list all candidates with similarity scores."""
        entity = CanonicalEntity(
            uuid="new-uuid",
            name="Test",
            entity_type="Company",
            summary="Test entity.",
            source_indices=[0],
        )
        candidates = [
            (
                Entity(
                    uuid="uuid-1",
                    name="Candidate 1",
                    summary="First candidate.",
                    entity_type=EntityType.COMPANY,
                ),
                0.92,
            ),
            (
                Entity(
                    uuid="uuid-2",
                    name="Candidate 2",
                    summary="Second candidate.",
                    entity_type=EntityType.COMPANY,
                ),
                0.75,
            ),
        ]

        registry = EntityRegistry.__new__(EntityRegistry)
        prompt = registry._build_match_prompt(entity, candidates)

        assert "Candidate 1" in prompt
        assert "92%" in prompt
        assert "uuid-1" in prompt
        assert "Candidate 2" in prompt
        assert "75%" in prompt

    def test_prompt_flags_high_similarity(self):
        """High similarity candidates should be flagged."""
        entity = CanonicalEntity(
            uuid="new-uuid",
            name="Test",
            entity_type="Company",
            summary="Test.",
            source_indices=[0],
        )
        candidates = [
            (
                Entity(
                    uuid="uuid-1",
                    name="Very Similar",
                    summary="Almost the same.",
                    entity_type=EntityType.COMPANY,
                ),
                0.95,  # Above HIGH_SIMILARITY_THRESHOLD
            ),
        ]

        registry = EntityRegistry.__new__(EntityRegistry)
        prompt = registry._build_match_prompt(entity, candidates)

        assert "[LIKELY SAME]" in prompt


class TestEntityResolutionResult:
    """Test EntityResolutionResult structure."""

    def test_empty_result(self):
        """Empty result should have empty collections."""
        result = EntityResolutionResult()
        assert result.new_entities == []
        assert result.uuid_remap == {}
        assert result.summary_updates == {}

    def test_result_with_data(self):
        """Result should correctly store all fields."""
        entity = CanonicalEntity(
            uuid="new-uuid",
            name="Test Corp",
            entity_type="Company",
            summary="A test company.",
            source_indices=[0],
        )
        result = EntityResolutionResult(
            new_entities=[entity],
            uuid_remap={"old-1": "canonical-1"},
            summary_updates={"canonical-1": "Updated summary"},
        )
        assert len(result.new_entities) == 1
        assert result.uuid_remap["old-1"] == "canonical-1"
        assert result.summary_updates["canonical-1"] == "Updated summary"


class TestEntityRegistryResolve:
    """Integration tests for EntityRegistry.resolve()."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage backend."""
        storage = AsyncMock()
        storage.search_entities = AsyncMock(return_value=[])
        return storage

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        llm = AsyncMock()
        llm.generate_structured = AsyncMock()
        llm.generate = AsyncMock(return_value="Merged summary.")
        return llm

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embedding provider."""
        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=[[0.1] * 3072])
        return embeddings

    @pytest.mark.asyncio
    async def test_resolve_empty_list(self, mock_storage, mock_llm, mock_embeddings):
        """Empty entity list returns empty result."""
        registry = EntityRegistry(mock_storage, mock_llm, mock_embeddings)
        result = await registry.resolve([])

        assert result.new_entities == []
        assert result.uuid_remap == {}
        assert result.summary_updates == {}

    @pytest.mark.asyncio
    async def test_resolve_no_candidates_found(self, mock_storage, mock_llm, mock_embeddings):
        """Entity with no KB matches is marked as new."""
        mock_storage.search_entities = AsyncMock(return_value=[])

        registry = EntityRegistry(mock_storage, mock_llm, mock_embeddings)
        entity = CanonicalEntity(
            uuid="new-uuid",
            name="Brand New Corp",
            entity_type="Company",
            summary="A completely new company.",
            source_indices=[0],
        )

        result = await registry.resolve([entity])

        assert len(result.new_entities) == 1
        assert result.new_entities[0].uuid == "new-uuid"
        assert result.uuid_remap == {}
        # LLM should not be called when no candidates
        mock_llm.generate_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_with_match(self, mock_storage, mock_llm, mock_embeddings):
        """Entity matching existing KB entity gets remapped."""
        # Setup: KB has an existing entity
        existing = Entity(
            uuid="existing-uuid",
            name="Apple Inc.",
            summary="Tech company.",
            entity_type=EntityType.COMPANY,
        )
        mock_storage.search_entities = AsyncMock(return_value=[(existing, 0.92)])

        # LLM says it's a match
        mock_llm.generate_structured = AsyncMock(
            return_value=EntityRegistryMatch(
                matches_existing=True,
                matched_uuid="existing-uuid",
                canonical_name="Apple Inc.",
                merged_summary="",  # Will be filled by merge
                confidence=0.95,
                reasoning="Same company.",
            )
        )

        registry = EntityRegistry(mock_storage, mock_llm, mock_embeddings)
        entity = CanonicalEntity(
            uuid="new-uuid",
            name="Apple",
            entity_type="Company",
            summary="Makes iPhones.",
            source_indices=[0],
        )

        result = await registry.resolve([entity])

        assert result.new_entities == []  # Not new
        assert result.uuid_remap["new-uuid"] == "existing-uuid"
        assert "existing-uuid" in result.summary_updates

    @pytest.mark.asyncio
    async def test_resolve_llm_says_distinct(self, mock_storage, mock_llm, mock_embeddings):
        """Entity that LLM says is distinct remains new."""
        existing = Entity(
            uuid="aws-uuid",
            name="AWS",
            summary="Cloud computing.",
            entity_type=EntityType.COMPANY,
        )
        mock_storage.search_entities = AsyncMock(return_value=[(existing, 0.75)])

        # LLM says distinct (subsidiary rule)
        mock_llm.generate_structured = AsyncMock(
            return_value=EntityRegistryMatch(
                matches_existing=False,
                matched_uuid=None,
                canonical_name="Amazon",
                merged_summary="E-commerce company.",
                confidence=0.90,
                reasoning="AWS is subsidiary, not same as Amazon.",
            )
        )

        registry = EntityRegistry(mock_storage, mock_llm, mock_embeddings)
        entity = CanonicalEntity(
            uuid="amazon-uuid",
            name="Amazon",
            entity_type="Company",
            summary="E-commerce company.",
            source_indices=[0],
        )

        result = await registry.resolve([entity])

        assert len(result.new_entities) == 1
        assert result.new_entities[0].uuid == "amazon-uuid"
        assert result.uuid_remap == {}

    @pytest.mark.asyncio
    async def test_resolve_uses_precomputed_embeddings(self, mock_storage, mock_llm, mock_embeddings):
        """Pre-computed embeddings should be used instead of generating new ones."""
        mock_storage.search_entities = AsyncMock(return_value=[])

        registry = EntityRegistry(mock_storage, mock_llm, mock_embeddings)
        entity = CanonicalEntity(
            uuid="test-uuid",
            name="Test",
            entity_type="Company",
            summary="Test company.",
            source_indices=[0],
        )

        precomputed = [[0.5] * 3072]
        await registry.resolve([entity], embeddings=precomputed)

        # Embedding provider should not be called
        mock_embeddings.embed.assert_not_called()
        # Storage search should use precomputed embedding
        mock_storage.search_entities.assert_called_once()
        call_args = mock_storage.search_entities.call_args
        assert call_args[0][0] == precomputed[0]

    @pytest.mark.asyncio
    async def test_resolve_multiple_entities(self, mock_storage, mock_llm, mock_embeddings):
        """Multiple entities should be processed independently."""
        mock_storage.search_entities = AsyncMock(return_value=[])
        mock_embeddings.embed = AsyncMock(return_value=[[0.1] * 3072, [0.2] * 3072])

        registry = EntityRegistry(mock_storage, mock_llm, mock_embeddings)
        entities = [
            CanonicalEntity(
                uuid="uuid-1",
                name="Entity 1",
                entity_type="Company",
                summary="First entity.",
                source_indices=[0],
            ),
            CanonicalEntity(
                uuid="uuid-2",
                name="Entity 2",
                entity_type="Company",
                summary="Second entity.",
                source_indices=[1],
            ),
        ]

        result = await registry.resolve(entities)

        assert len(result.new_entities) == 2
        assert mock_storage.search_entities.call_count == 2

    @pytest.mark.asyncio
    async def test_resolve_llm_failure_treats_as_new(self, mock_storage, mock_llm, mock_embeddings):
        """LLM failure should treat entity as new (conservative approach)."""
        existing = Entity(
            uuid="existing-uuid",
            name="Similar Corp",
            summary="Similar company.",
            entity_type=EntityType.COMPANY,
        )
        mock_storage.search_entities = AsyncMock(return_value=[(existing, 0.85)])

        # LLM raises an exception
        mock_llm.generate_structured = AsyncMock(side_effect=Exception("LLM timeout"))

        registry = EntityRegistry(mock_storage, mock_llm, mock_embeddings)
        entity = CanonicalEntity(
            uuid="new-uuid",
            name="Test Corp",
            entity_type="Company",
            summary="Test company.",
            source_indices=[0],
        )

        result = await registry.resolve([entity])

        # Should be treated as new despite having candidates
        assert len(result.new_entities) == 1
        assert result.new_entities[0].uuid == "new-uuid"
        assert result.uuid_remap == {}

    @pytest.mark.asyncio
    async def test_resolve_with_concurrency_limit(self, mock_storage, mock_llm, mock_embeddings):
        """Concurrency should be configurable."""
        mock_storage.search_entities = AsyncMock(return_value=[])
        mock_embeddings.embed = AsyncMock(return_value=[[0.1] * 3072] * 10)

        # Custom concurrency limit
        registry = EntityRegistry(
            mock_storage, mock_llm, mock_embeddings, concurrency=2
        )
        assert registry.concurrency == 2

        entities = [
            CanonicalEntity(
                uuid=f"uuid-{i}",
                name=f"Entity {i}",
                entity_type="Company",
                summary=f"Entity {i} description.",
                source_indices=[i],
            )
            for i in range(10)
        ]

        result = await registry.resolve(entities)

        assert len(result.new_entities) == 10
