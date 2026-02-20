"""Tests for entity deduplication."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from vanna_kg.types.entities import EnumeratedEntity


class TestStripEntityTypeSuffix:
    """Tests for _strip_entity_type_suffix helper."""

    def test_strips_product_suffix(self):
        """Strips (Product) suffix from entity name."""
        from vanna_kg.ingestion.resolution.entity_dedup import _strip_entity_type_suffix

        assert _strip_entity_type_suffix("Beige Book (Product)") == "Beige Book"

    def test_strips_company_suffix(self):
        """Strips (Company) suffix from entity name."""
        from vanna_kg.ingestion.resolution.entity_dedup import _strip_entity_type_suffix

        assert _strip_entity_type_suffix("Apple Inc. (Company)") == "Apple Inc."

    def test_strips_organization_suffix(self):
        """Strips (Organization) suffix from entity name."""
        from vanna_kg.ingestion.resolution.entity_dedup import _strip_entity_type_suffix

        assert _strip_entity_type_suffix("Federal Reserve (Organization)") == "Federal Reserve"

    def test_preserves_name_without_suffix(self):
        """Preserves name that has no type suffix."""
        from vanna_kg.ingestion.resolution.entity_dedup import _strip_entity_type_suffix

        assert _strip_entity_type_suffix("Federal Reserve") == "Federal Reserve"

    def test_preserves_parenthetical_that_is_not_type(self):
        """Preserves parenthetical content that isn't an entity type."""
        from vanna_kg.ingestion.resolution.entity_dedup import _strip_entity_type_suffix

        # Lowercase isn't a type suffix
        assert _strip_entity_type_suffix("Apple (lowercase)") == "Apple (lowercase)"
        # Numbers aren't type suffixes
        assert _strip_entity_type_suffix("District 12 (2024)") == "District 12 (2024)"

    def test_strips_summary_from_full_entity_line(self):
        """Strips summary text that LLM sometimes includes."""
        from vanna_kg.ingestion.resolution.entity_dedup import _strip_entity_type_suffix

        # LLM sometimes returns full entity line from prompt
        full_line = "Federal Reserve System (Organization): Identified as the publisher of the Beige Book"
        assert _strip_entity_type_suffix(full_line) == "Federal Reserve System"

        # Also handles just name with type
        assert _strip_entity_type_suffix("Federal Reserve System (Organization)") == "Federal Reserve System"


class TestEmbeddingTextGeneration:
    """Tests for _embedding_text helper."""

    def test_embedding_text_with_definition(self):
        """Embedding text prefers definition over summary."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple Inc.",
            entity_type="Company",
            definition="American multinational technology company",
            summary="Reported strong Q3 earnings",
        )
        text = _embedding_text(entity)
        assert text == "Apple Inc.: American multinational technology company"

    def test_embedding_text_falls_back_to_summary(self):
        """Embedding text falls back to summary if no definition."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple Inc.",
            entity_type="Company",
            definition="",
            summary="Technology company known for iPhone",
        )
        text = _embedding_text(entity)
        assert text == "Apple Inc.: Technology company known for iPhone"

    def test_embedding_text_without_definition_or_summary(self):
        """Embedding text falls back to name only."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="AAPL",
            entity_type="Company",
            definition="",
            summary="",
        )
        text = _embedding_text(entity)
        assert text == "AAPL"

    def test_embedding_text_whitespace_definition(self):
        """Whitespace-only definition treated as empty, falls back to summary."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple",
            entity_type="Company",
            definition="   ",
            summary="Tech company",
        )
        text = _embedding_text(entity)
        assert text == "Apple: Tech company"


class TestSimilarityMatrix:
    """Tests for _compute_similarity_matrix."""

    def test_similarity_matrix_shape(self):
        """Similarity matrix is n x n."""
        from vanna_kg.ingestion.resolution.entity_dedup import _compute_similarity_matrix

        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        matrix = _compute_similarity_matrix(vectors)
        assert matrix.shape == (3, 3)

    def test_similarity_matrix_diagonal_ones(self):
        """Diagonal elements are 1.0 (self-similarity)."""
        from vanna_kg.ingestion.resolution.entity_dedup import _compute_similarity_matrix

        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        matrix = _compute_similarity_matrix(vectors)
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[1, 1] == pytest.approx(1.0)

    def test_similarity_matrix_orthogonal_zero(self):
        """Orthogonal vectors have zero similarity."""
        from vanna_kg.ingestion.resolution.entity_dedup import _compute_similarity_matrix

        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        matrix = _compute_similarity_matrix(vectors)
        assert matrix[0, 1] == pytest.approx(0.0)
        assert matrix[1, 0] == pytest.approx(0.0)

    def test_similarity_matrix_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        from vanna_kg.ingestion.resolution.entity_dedup import _compute_similarity_matrix

        vectors = [
            [0.6, 0.8],
            [0.6, 0.8],
        ]
        matrix = _compute_similarity_matrix(vectors)
        assert matrix[0, 1] == pytest.approx(1.0)

    def test_similarity_matrix_symmetric(self):
        """Similarity matrix is symmetric."""
        from vanna_kg.ingestion.resolution.entity_dedup import _compute_similarity_matrix

        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        matrix = _compute_similarity_matrix(vectors)
        assert matrix[0, 1] == pytest.approx(matrix[1, 0])
        assert matrix[0, 2] == pytest.approx(matrix[2, 0])
        assert matrix[1, 2] == pytest.approx(matrix[2, 1])


class TestSimilarityOrder:
    """Tests for _similarity_order (greedy BFS traversal)."""

    def test_similarity_order_returns_all_indices(self):
        """All indices are included in the ordering."""
        from vanna_kg.ingestion.resolution.entity_dedup import _similarity_order

        # 4 entities with varying similarities
        similarity = np.array([
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.8],
            [0.2, 0.1, 0.8, 1.0],
        ])
        order = _similarity_order(4, similarity)
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_similarity_order_starts_at_zero(self):
        """Ordering starts at index 0."""
        from vanna_kg.ingestion.resolution.entity_dedup import _similarity_order

        similarity = np.array([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ])
        order = _similarity_order(3, similarity)
        assert order[0] == 0

    def test_similarity_order_groups_similar(self):
        """Similar entities end up adjacent in the ordering."""
        from vanna_kg.ingestion.resolution.entity_dedup import _similarity_order

        # Entities 0,1 are similar; entities 2,3 are similar
        similarity = np.array([
            [1.0, 0.95, 0.1, 0.1],
            [0.95, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.95],
            [0.1, 0.1, 0.95, 1.0],
        ])
        order = _similarity_order(4, similarity)

        # 0 and 1 should be adjacent
        idx_0 = order.index(0)
        idx_1 = order.index(1)
        assert abs(idx_0 - idx_1) == 1

        # 2 and 3 should be adjacent
        idx_2 = order.index(2)
        idx_3 = order.index(3)
        assert abs(idx_2 - idx_3) == 1

    def test_similarity_order_single_element(self):
        """Single element returns [0]."""
        from vanna_kg.ingestion.resolution.entity_dedup import _similarity_order

        similarity = np.array([[1.0]])
        order = _similarity_order(1, similarity)
        assert order == [0]


class TestVerifyCluster:
    """Tests for _verify_cluster LLM verification."""

    @pytest.mark.asyncio
    async def test_verify_cluster_calls_llm(self):
        """verify_cluster calls LLM with correct prompt structure."""
        from vanna_kg.ingestion.resolution.entity_dedup import _verify_cluster
        from vanna_kg.types.results import EntityDedupeResult, EntityGroup

        entities = [
            EnumeratedEntity(name="Apple Inc.", entity_type="Company", summary="Tech company"),
            EnumeratedEntity(name="AAPL", entity_type="Company", summary="Stock ticker"),
        ]

        # Mock LLM that returns entities as separate groups
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(
            return_value=EntityDedupeResult(
                groups=[
                    EntityGroup(
                        reasoning="Same company",
                        entity_type="COMPANY",
                        canonical="Apple Inc.",
                        members=["AAPL"],
                    )
                ]
            )
        )

        result = await _verify_cluster(entities, [0, 1], mock_llm)

        # Should have called LLM
        mock_llm.generate_structured.assert_called_once()

        # Result should have the group
        assert len(result.groups) == 1
        assert result.groups[0].canonical == "Apple Inc."

    @pytest.mark.asyncio
    async def test_verify_cluster_includes_all_entities_in_prompt(self):
        """Prompt includes all entities with indices."""
        from vanna_kg.ingestion.resolution.entity_dedup import _verify_cluster
        from vanna_kg.types.results import EntityDedupeResult

        entities = [
            EnumeratedEntity(name="A", entity_type="Company", summary="First"),
            EnumeratedEntity(name="B", entity_type="Company", summary="Second"),
            EnumeratedEntity(name="C", entity_type="Company", summary="Third"),
        ]

        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(
            return_value=EntityDedupeResult(groups=[])
        )

        await _verify_cluster(entities, [0, 1, 2], mock_llm)

        # Check the prompt includes all entity names
        call_args = mock_llm.generate_structured.call_args
        prompt = call_args[0][0]  # First positional argument
        assert "A" in prompt
        assert "B" in prompt
        assert "C" in prompt


class TestCreateOverlappingBatches:
    """Tests for _create_overlapping_batches."""

    def test_small_cluster_single_batch(self):
        """Cluster smaller than batch size returns single batch."""
        from vanna_kg.ingestion.resolution.entity_dedup import _create_overlapping_batches

        indices = [0, 1, 2, 3, 4]
        batches = _create_overlapping_batches(indices, max_batch_size=15, overlap=5)
        assert len(batches) == 1
        assert batches[0] == [0, 1, 2, 3, 4]

    def test_exact_batch_size_single_batch(self):
        """Cluster exactly at batch size returns single batch."""
        from vanna_kg.ingestion.resolution.entity_dedup import _create_overlapping_batches

        indices = list(range(15))
        batches = _create_overlapping_batches(indices, max_batch_size=15, overlap=5)
        assert len(batches) == 1

    def test_large_cluster_multiple_batches(self):
        """Large cluster splits into overlapping batches."""
        from vanna_kg.ingestion.resolution.entity_dedup import _create_overlapping_batches

        indices = list(range(25))
        batches = _create_overlapping_batches(indices, max_batch_size=15, overlap=5)

        # Should have 2 batches
        assert len(batches) == 2
        # First batch: 0-14
        assert batches[0] == list(range(15))
        # Second batch: 10-24 (overlap of 5)
        assert batches[1] == list(range(10, 25))

    def test_batches_cover_all_indices(self):
        """All indices appear in at least one batch."""
        from vanna_kg.ingestion.resolution.entity_dedup import _create_overlapping_batches

        indices = list(range(40))
        batches = _create_overlapping_batches(indices, max_batch_size=15, overlap=5)

        all_covered = set()
        for batch in batches:
            all_covered.update(batch)
        assert all_covered == set(indices)

    def test_overlap_between_consecutive_batches(self):
        """Consecutive batches have specified overlap."""
        from vanna_kg.ingestion.resolution.entity_dedup import _create_overlapping_batches

        indices = list(range(30))
        batches = _create_overlapping_batches(indices, max_batch_size=15, overlap=5)

        for i in range(len(batches) - 1):
            overlap = set(batches[i]) & set(batches[i + 1])
            assert len(overlap) >= 5


class TestMergeBatchResults:
    """Tests for _merge_batch_results."""

    def test_merge_non_overlapping_groups(self):
        """Groups with no overlapping entities stay separate."""
        from vanna_kg.ingestion.resolution.entity_dedup import _merge_batch_results
        from vanna_kg.types.results import EntityDedupeResult, EntityGroup

        batch_results = [
            (
                [0, 1, 2],  # batch indices
                EntityDedupeResult(groups=[
                    EntityGroup(
                        reasoning="Same",
                        entity_type="COMPANY",
                        canonical="Apple Inc.",
                        members=["AAPL"],
                    )
                ]),
            ),
            (
                [3, 4, 5],
                EntityDedupeResult(groups=[
                    EntityGroup(
                        reasoning="Same",
                        entity_type="COMPANY",
                        canonical="Google",
                        members=["Alphabet"],
                    )
                ]),
            ),
        ]

        # Map batch-local indices to original indices
        # Batch 1: "Apple Inc." is index 0, "AAPL" is index 1
        # Batch 2: "Google" is index 0 (batch-local), "Alphabet" is index 1
        entities = [
            EnumeratedEntity(name="Apple Inc.", entity_type="Company", summary=""),
            EnumeratedEntity(name="AAPL", entity_type="Company", summary=""),
            EnumeratedEntity(name="Tim", entity_type="Person", summary=""),
            EnumeratedEntity(name="Google", entity_type="Company", summary=""),
            EnumeratedEntity(name="Alphabet", entity_type="Company", summary=""),
            EnumeratedEntity(name="Sundar", entity_type="Person", summary=""),
        ]

        merged = _merge_batch_results(batch_results, entities)

        # Should have 2 separate groups
        assert len(merged) == 2

    def test_merge_keeps_entity_in_first_group(self):
        """Overlapping entity stays in first-assigned group."""
        from vanna_kg.ingestion.resolution.entity_dedup import _merge_batch_results
        from vanna_kg.types.results import EntityDedupeResult, EntityGroup

        entities = [
            EnumeratedEntity(name="Apple Inc.", entity_type="Company", summary="Tech"),
            EnumeratedEntity(name="AAPL", entity_type="Company", summary="Ticker"),
            EnumeratedEntity(name="Apple", entity_type="Company", summary="Company"),
        ]

        # Entity "Apple" (index 2) appears in both batches
        batch_results = [
            (
                [0, 1, 2],
                EntityDedupeResult(groups=[
                    EntityGroup(
                        reasoning="Same company",
                        entity_type="COMPANY",
                        canonical="Apple Inc.",
                        members=["AAPL", "Apple"],  # includes index 2
                    )
                ]),
            ),
            (
                [2],  # Only Apple in second batch
                EntityDedupeResult(groups=[]),  # Singleton, no groups
            ),
        ]

        merged = _merge_batch_results(batch_results, entities)

        # Should have 1 group with all 3
        assert len(merged) == 1
        group = merged[0]
        assert "Apple Inc." in [group["canonical"]] + group["members"]
        assert "AAPL" in [group["canonical"]] + group["members"]
        assert "Apple" in [group["canonical"]] + group["members"]


class TestDeduplicateEntities:
    """Integration tests for deduplicate_entities."""

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty entity list returns empty output."""
        from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities

        mock_llm = MagicMock()
        mock_embeddings = MagicMock()

        result = await deduplicate_entities([], mock_llm, mock_embeddings)

        assert len(result.canonical_entities) == 0
        assert result.index_to_canonical == {}
        assert len(result.merge_history) == 0

    @pytest.mark.asyncio
    async def test_single_entity(self):
        """Single entity passes through with UUID assigned."""
        from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities

        entities = [
            EnumeratedEntity(name="Apple Inc.", entity_type="Company", summary="Tech")
        ]

        mock_llm = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.embed = AsyncMock(
            side_effect=[
                [[0.1, 0.2, 0.3]],  # mention embeddings
                [[0.4, 0.5, 0.6]],  # canonical embeddings
            ]
        )

        result = await deduplicate_entities(entities, mock_llm, mock_embeddings)

        assert len(result.canonical_entities) == 1
        assert result.canonical_entities[0].name == "Apple Inc."
        assert result.canonical_entities[0].uuid  # Has a UUID
        assert result.index_to_canonical == {0: 0}
        assert result.canonical_entity_embeddings == [[0.4, 0.5, 0.6]]
        assert mock_embeddings.embed.await_args_list[1].args[0] == ["Apple Inc.: Tech"]

    @pytest.mark.asyncio
    async def test_merges_similar_entities(self):
        """Similar entities get merged based on LLM decision."""
        from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities
        from vanna_kg.types.results import EntityDedupeResult, EntityGroup

        entities = [
            EnumeratedEntity(name="Apple Inc.", entity_type="Company", summary="Tech"),
            EnumeratedEntity(name="AAPL", entity_type="Company", summary="Ticker"),
            EnumeratedEntity(name="Google", entity_type="Company", summary="Search"),
        ]

        # High similarity between Apple entities, low with Google
        mock_embeddings = MagicMock()
        mock_embeddings.embed = AsyncMock(
            side_effect=[
                [
                    [1.0, 0.0, 0.0],  # Apple Inc.
                    [0.95, 0.1, 0.0],  # AAPL (similar to Apple Inc.)
                    [0.0, 0.0, 1.0],  # Google (different)
                ],
                [
                    [0.9, 0.1, 0.0],  # Apple canonical
                    [0.0, 0.1, 0.9],  # Google canonical
                ],
            ]
        )

        # LLM says Apple Inc. and AAPL are the same
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(
            return_value=EntityDedupeResult(groups=[
                EntityGroup(
                    reasoning="Same company",
                    entity_type="COMPANY",
                    canonical="Apple Inc.",
                    members=["AAPL"],
                )
            ])
        )

        result = await deduplicate_entities(
            entities, mock_llm, mock_embeddings, similarity_threshold=0.70
        )

        # Should have 2 canonical entities (Apple Inc. merged, Google separate)
        assert len(result.canonical_entities) == 2
        assert len(result.canonical_entity_embeddings) == 2

        # Find Apple entity
        apple = next(e for e in result.canonical_entities if e.name == "Apple Inc.")
        assert "AAPL" in apple.aliases
        assert set(apple.source_indices) == {0, 1}

        # Index mapping
        assert result.index_to_canonical[0] == result.index_to_canonical[1]  # Merged
        assert result.index_to_canonical[2] != result.index_to_canonical[0]  # Google separate

    @pytest.mark.asyncio
    async def test_respects_similarity_threshold(self):
        """Entities below threshold are not clustered."""
        from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities

        entities = [
            EnumeratedEntity(name="A", entity_type="Company", summary=""),
            EnumeratedEntity(name="B", entity_type="Company", summary=""),
        ]

        # Low similarity (0.5) - below default threshold (0.7)
        mock_embeddings = MagicMock()
        mock_embeddings.embed = AsyncMock(return_value=[
            [1.0, 0.0],
            [0.5, 0.866],  # cos(60°) = 0.5 similarity
        ])

        mock_llm = MagicMock()

        result = await deduplicate_entities(
            entities, mock_llm, mock_embeddings, similarity_threshold=0.70
        )

        # LLM should not be called (no clusters to verify)
        mock_llm.generate_structured.assert_not_called()

        # Both entities are separate
        assert len(result.canonical_entities) == 2

    @pytest.mark.asyncio
    async def test_same_name_entities_clustered_for_llm_verification(self):
        """Entities with same name are clustered even if embeddings differ."""
        from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities
        from vanna_kg.types.results import EntityDedupeResult, EntityGroup

        # Two entities with same name but different definitions
        entities = [
            EnumeratedEntity(
                name="Federal Reserve System",
                entity_type="Organization",
                definition="The central banking system",
                summary="Sets monetary policy",
            ),
            EnumeratedEntity(
                name="Federal Reserve System",
                entity_type="Organization",
                definition="Publisher of the Beige Book",
                summary="Publishes economic reports",
            ),
        ]

        # Low embedding similarity (below threshold)
        mock_embeddings = MagicMock()
        mock_embeddings.embed = AsyncMock(return_value=[
            [1.0, 0.0],
            [0.5, 0.866],  # cos(60°) = 0.5 similarity - below 0.70 threshold
        ])

        # LLM should be called and will merge them
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(
            return_value=EntityDedupeResult(groups=[
                EntityGroup(
                    reasoning="Same organization",
                    entity_type="ORGANIZATION",
                    canonical="Federal Reserve System",
                    members=["Federal Reserve System"],
                )
            ])
        )

        result = await deduplicate_entities(
            entities, mock_llm, mock_embeddings, similarity_threshold=0.70
        )

        # LLM SHOULD be called because same-name entities are clustered
        mock_llm.generate_structured.assert_called_once()

        # Should merge to 1 canonical entity
        assert len(result.canonical_entities) == 1
        assert result.canonical_entities[0].name == "Federal Reserve System"
