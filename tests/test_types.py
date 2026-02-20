"""Tests for result types."""

import pytest
from pydantic import ValidationError

from vanna_kg.types.entities import EnumeratedEntity
from vanna_kg.types.results import (
    AssemblyResult,
    CanonicalEntity,
    EntityDeduplicationOutput,
    EntityRegistryMatch,
    EntityResolutionResult,
    MergeRecord,
)


class TestMergeRecord:
    """Tests for MergeRecord type."""

    def test_merge_record_valid(self):
        """MergeRecord accepts valid data."""
        record = MergeRecord(
            canonical_uuid="uuid-123",
            canonical_name="Apple Inc.",
            merged_indices=[0, 1, 2],
            merged_names=["Apple Inc.", "AAPL", "Apple"],
            original_summaries=["Tech company", "Stock ticker", "Consumer electronics"],
            final_summary="Technology company known for iPhone and Mac computers.",
        )
        assert record.canonical_name == "Apple Inc."
        assert len(record.merged_indices) == 3

    def test_merge_record_requires_uuid(self):
        """MergeRecord requires canonical_uuid."""
        with pytest.raises(ValidationError):
            MergeRecord(
                canonical_name="Apple Inc.",
                merged_indices=[0],
                merged_names=["Apple"],
                original_summaries=["Tech"],
                final_summary="Tech company",
            )


class TestCanonicalEntity:
    """Tests for CanonicalEntity type."""

    def test_canonical_entity_valid(self):
        """CanonicalEntity accepts valid data."""
        entity = CanonicalEntity(
            uuid="uuid-456",
            name="Apple Inc.",
            entity_type="Company",
            summary="Technology company",
            source_indices=[0, 1],
            aliases=["AAPL", "Apple"],
        )
        assert entity.name == "Apple Inc."
        assert entity.aliases == ["AAPL", "Apple"]

    def test_canonical_entity_defaults(self):
        """CanonicalEntity has sensible defaults."""
        entity = CanonicalEntity(
            uuid="uuid-789",
            name="Tim Cook",
            entity_type="Person",
            summary="CEO of Apple",
        )
        assert entity.source_indices == []
        assert entity.aliases == []


class TestEntityDeduplicationOutput:
    """Tests for EntityDeduplicationOutput type."""

    def test_dedup_output_valid(self):
        """EntityDeduplicationOutput accepts valid data."""
        output = EntityDeduplicationOutput(
            canonical_entities=[
                CanonicalEntity(
                    uuid="uuid-1",
                    name="Apple Inc.",
                    entity_type="Company",
                    summary="Tech company",
                    source_indices=[0, 1],
                    aliases=["AAPL"],
                )
            ],
            index_to_canonical={0: 0, 1: 0},
            merge_history=[],
            canonical_entity_embeddings=[[0.1, 0.2, 0.3]],
        )
        assert len(output.canonical_entities) == 1
        assert output.index_to_canonical[0] == 0
        assert output.index_to_canonical[1] == 0
        assert output.canonical_entity_embeddings == [[0.1, 0.2, 0.3]]

    def test_dedup_output_empty(self):
        """EntityDeduplicationOutput accepts empty lists."""
        output = EntityDeduplicationOutput(
            canonical_entities=[],
            index_to_canonical={},
            merge_history=[],
        )
        assert len(output.canonical_entities) == 0
        assert output.canonical_entity_embeddings == []


class TestEntityRegistryMatch:
    """Tests for EntityRegistryMatch type."""

    def test_entity_registry_match_valid(self):
        """EntityRegistryMatch accepts valid match data."""
        match = EntityRegistryMatch(
            matches_existing=True,
            matched_uuid="abc-123",
            canonical_name="Apple Inc.",
            merged_summary="Technology company that makes iPhones.",
            confidence=0.95,
            reasoning="Same company, different name variations.",
        )
        assert match.matches_existing is True
        assert match.matched_uuid == "abc-123"
        assert match.confidence == 0.95

    def test_entity_registry_match_no_match(self):
        """EntityRegistryMatch accepts valid non-match data."""
        match = EntityRegistryMatch(
            matches_existing=False,
            matched_uuid=None,
            canonical_name="New Corp",
            merged_summary="A new corporation.",
            confidence=0.85,
            reasoning="No similar entities found in KB.",
        )
        assert match.matches_existing is False
        assert match.matched_uuid is None

    def test_entity_registry_match_confidence_bounds(self):
        """EntityRegistryMatch enforces confidence bounds."""
        # Valid at boundaries
        match_low = EntityRegistryMatch(
            matches_existing=False,
            matched_uuid=None,
            canonical_name="Test",
            merged_summary="Test",
            confidence=0.0,
            reasoning="Test",
        )
        assert match_low.confidence == 0.0

        match_high = EntityRegistryMatch(
            matches_existing=False,
            matched_uuid=None,
            canonical_name="Test",
            merged_summary="Test",
            confidence=1.0,
            reasoning="Test",
        )
        assert match_high.confidence == 1.0

        # Invalid: out of bounds
        with pytest.raises(ValidationError):
            EntityRegistryMatch(
                matches_existing=False,
                matched_uuid=None,
                canonical_name="Test",
                merged_summary="Test",
                confidence=1.5,
                reasoning="Test",
            )


class TestEntityResolutionResult:
    """Tests for EntityResolutionResult type."""

    def test_entity_resolution_result_empty(self):
        """EntityResolutionResult defaults to empty collections."""
        result = EntityResolutionResult()
        assert result.new_entities == []
        assert result.uuid_remap == {}
        assert result.summary_updates == {}

    def test_entity_resolution_result_with_data(self):
        """EntityResolutionResult stores all fields correctly."""
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


class TestAssemblyResult:
    """Tests for AssemblyResult type."""

    def test_assembly_result_defaults(self):
        """AssemblyResult defaults to zeros."""
        result = AssemblyResult()
        assert result.document_written is False
        assert result.chunks_written == 0
        assert result.entities_written == 0
        assert result.facts_written == 0
        assert result.topics_written == 0
        assert result.relationships_written == 0

    def test_assembly_result_with_values(self):
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


class TestTopicResolutionResult:
    """Test TopicResolutionResult structure."""

    def test_empty_result(self):
        """Empty result should have empty collections."""
        from vanna_kg.types.topics import TopicResolutionResult

        result = TopicResolutionResult()
        assert result.resolved_topics == []
        assert result.uuid_remap == {}
        assert result.new_topics == []

    def test_result_with_data(self):
        """Result should correctly store all fields."""
        from vanna_kg.types.topics import TopicResolution, TopicResolutionResult

        topic = TopicResolution(
            uuid="topic-uuid",
            canonical_label="Inflation",
            is_new=False,
            definition="A general increase in prices.",
        )
        result = TopicResolutionResult(
            resolved_topics=[topic],
            uuid_remap={"CPI": "topic-uuid"},
            new_topics=["Unknown Topic"],
        )
        assert len(result.resolved_topics) == 1
        assert result.uuid_remap["CPI"] == "topic-uuid"
        assert "Unknown Topic" in result.new_topics


class TestTopicMatchTypes:
    """Test topic match decision types."""

    def test_topic_match_decision_with_match(self):
        """TopicMatchDecision should store match decision."""
        from vanna_kg.types.topics import TopicMatchDecision

        decision = TopicMatchDecision(
            topic="M&A",
            selected_number=1,
            reasoning="Exact semantic match to Mergers And Acquisitions.",
        )
        assert decision.topic == "M&A"
        assert decision.selected_number == 1
        assert "semantic match" in decision.reasoning

    def test_topic_match_decision_no_match(self):
        """TopicMatchDecision should allow null for no match."""
        from vanna_kg.types.topics import TopicMatchDecision

        decision = TopicMatchDecision(
            topic="Random Noise",
            selected_number=None,
            reasoning="No candidates match this topic.",
        )
        assert decision.selected_number is None

    def test_batch_topic_match_response(self):
        """BatchTopicMatchResponse should contain list of decisions."""
        from vanna_kg.types.topics import BatchTopicMatchResponse, TopicMatchDecision

        decisions = [
            TopicMatchDecision(topic="M&A", selected_number=1, reasoning="Match."),
            TopicMatchDecision(topic="Unknown", selected_number=None, reasoning="No match."),
        ]
        response = BatchTopicMatchResponse(decisions=decisions)
        assert len(response.decisions) == 2
        assert response.decisions[0].selected_number == 1
        assert response.decisions[1].selected_number is None


def test_enumerated_entity_has_definition():
    """EnumeratedEntity should have both definition and summary fields."""
    entity = EnumeratedEntity(
        name="Federal Reserve",
        entity_type="Organization",
        definition="The central banking system of the United States",
        summary="Reported concern about inflation in October 2025",
    )
    assert entity.definition == "The central banking system of the United States"
    assert entity.summary == "Reported concern about inflation in October 2025"
