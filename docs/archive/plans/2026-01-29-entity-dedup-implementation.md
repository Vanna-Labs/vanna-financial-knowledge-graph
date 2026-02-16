# Entity Deduplication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement in-document entity deduplication (Phase 2a-c) that clusters and merges duplicate entity mentions within a single document.

**Architecture:** Embedding similarity → Union-Find clustering → LLM verification → Canonical entity output. Uses existing providers and Union-Find utility.

**Tech Stack:** Python 3.10+, numpy (similarity), pydantic (types), pytest-asyncio (tests), existing LLM/Embedding providers.

---

## Task 1: Add New Types to results.py

**Files:**
- Modify: `vanna_kg/types/results.py`
- Test: `tests/test_types.py` (new file)

**Step 1.1: Create tests directory and test file**

Create `tests/__init__.py`:
```python
"""VannaKG test suite."""
```

Create `tests/test_types.py`:
```python
"""Tests for result types."""

import pytest
from pydantic import ValidationError

from vanna_kg.types.results import (
    MergeRecord,
    CanonicalEntity,
    EntityDeduplicationOutput,
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
        )
        assert len(output.canonical_entities) == 1
        assert output.index_to_canonical[0] == 0
        assert output.index_to_canonical[1] == 0

    def test_dedup_output_empty(self):
        """EntityDeduplicationOutput accepts empty lists."""
        output = EntityDeduplicationOutput(
            canonical_entities=[],
            index_to_canonical={},
            merge_history=[],
        )
        assert len(output.canonical_entities) == 0
```

**Step 1.2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with import errors (types don't exist yet)

**Step 1.3: Add types to results.py**

Add these classes to the end of `vanna_kg/types/results.py` (before the closing of the file):

```python
# -----------------------------------------------------------------------------
# Entity Deduplication Types
# -----------------------------------------------------------------------------


class MergeRecord(BaseModel):
    """
    Record of entities merged into one canonical entity.

    Used for traceability and debugging of deduplication decisions.
    """

    canonical_uuid: str = Field(..., description="UUID of the surviving entity")
    canonical_name: str = Field(..., description="Name chosen as canonical")
    merged_indices: list[int] = Field(
        ..., description="Original indices of entities that were merged"
    )
    merged_names: list[str] = Field(..., description="Names of merged entities")
    original_summaries: list[str] = Field(..., description="Summaries before merge")
    final_summary: str = Field(..., description="Combined summary after merge")


class CanonicalEntity(BaseModel):
    """
    A deduplicated entity with assigned UUID.

    Represents one real-world entity after in-document deduplication.
    """

    uuid: str = Field(..., description="Assigned UUID for this entity")
    name: str = Field(..., description="Canonical name")
    entity_type: EntityTypeLabel = Field(..., description="Entity type")
    summary: str = Field(..., description="Summary (merged if multiple sources)")
    source_indices: list[int] = Field(
        default_factory=list,
        description="Which original entity indices this represents",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names from merged entities",
    )


class EntityDeduplicationOutput(BaseModel):
    """
    Complete output from in-document deduplication.

    Provides canonical entities, index mapping for fact rewriting,
    and merge history for traceability.
    """

    canonical_entities: list[CanonicalEntity] = Field(
        ..., description="Deduplicated entities with UUIDs"
    )
    index_to_canonical: dict[int, int] = Field(
        ..., description="Map from original entity index to canonical entity index"
    )
    merge_history: list[MergeRecord] = Field(
        default_factory=list, description="Record of all merges performed"
    )
```

**Step 1.4: Add import for EntityTypeLabel at top of results.py**

Modify the import line at top of `vanna_kg/types/results.py`:
```python
from vanna_kg.types.entities import EntityGroup, EnumeratedEntity, EntityTypeLabel
```

**Step 1.5: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: PASS (all 6 tests)

---

## Task 2: Create entity_dedup.py Skeleton with Embedding Helper

**Files:**
- Create: `vanna_kg/ingestion/resolution/entity_dedup.py`
- Create: `tests/test_entity_dedup.py`

**Step 2.1: Create test file for embedding generation**

Create `tests/test_entity_dedup.py`:
```python
"""Tests for entity deduplication."""

import numpy as np
import pytest

from vanna_kg.types.entities import EnumeratedEntity


class TestEmbeddingTextGeneration:
    """Tests for _embedding_text helper."""

    def test_embedding_text_with_summary(self):
        """Embedding text includes name and summary."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple Inc.",
            entity_type="Company",
            summary="Technology company known for iPhone",
        )
        text = _embedding_text(entity)
        assert text == "Apple Inc.: Technology company known for iPhone"

    def test_embedding_text_without_summary(self):
        """Embedding text falls back to name only."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="AAPL",
            entity_type="Company",
            summary="",
        )
        text = _embedding_text(entity)
        assert text == "AAPL"

    def test_embedding_text_whitespace_summary(self):
        """Whitespace-only summary treated as empty."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple",
            entity_type="Company",
            summary="   ",
        )
        text = _embedding_text(entity)
        assert text == "Apple"
```

**Step 2.2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_dedup.py::TestEmbeddingTextGeneration -v`
Expected: FAIL with import error

**Step 2.3: Create entity_dedup.py with embedding helper**

Create `vanna_kg/ingestion/resolution/entity_dedup.py`:
```python
"""
In-Document Entity Deduplication

Phase 2a-c of the VannaKG ingestion pipeline.

Merges entity mentions that refer to the same real-world entity within
a single document using embedding similarity + LLM verification.

Example:
    >>> from vanna_kg.ingestion.resolution import deduplicate_entities
    >>> result = await deduplicate_entities(entities, llm, embeddings)
    >>> print(f"Reduced to {len(result.canonical_entities)} canonical entities")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vanna_kg.types.entities import EnumeratedEntity

if TYPE_CHECKING:
    from vanna_kg.providers.base import EmbeddingProvider, LLMProvider
    from vanna_kg.types.results import EntityDeduplicationOutput


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _embedding_text(entity: EnumeratedEntity) -> str:
    """
    Generate embedding text for an entity.

    Format: "{name}: {summary}" if summary exists, else just "{name}".
    Including summary provides semantic context to distinguish entities
    with similar names (e.g., "Apple Inc." vs "Apple Records").
    """
    summary = entity.summary.strip() if entity.summary else ""
    if summary:
        return f"{entity.name}: {summary}"
    return entity.name
```

**Step 2.4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_dedup.py::TestEmbeddingTextGeneration -v`
Expected: PASS (3 tests)

---

## Task 3: Implement Similarity Matrix Computation

**Files:**
- Modify: `vanna_kg/ingestion/resolution/entity_dedup.py`
- Modify: `tests/test_entity_dedup.py`

**Step 3.1: Add tests for similarity matrix**

Add to `tests/test_entity_dedup.py`:
```python
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
```

**Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_dedup.py::TestSimilarityMatrix -v`
Expected: FAIL with import error

**Step 3.3: Implement similarity matrix computation**

Add to `vanna_kg/ingestion/resolution/entity_dedup.py`:
```python
import numpy as np


def _compute_similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        vectors: List of embedding vectors (n x d)

    Returns:
        n x n similarity matrix where S[i,j] = cosine_sim(v[i], v[j])
    """
    # Convert to numpy array
    E = np.array(vectors, dtype=np.float32)

    # Compute norms
    norms = np.linalg.norm(E, axis=1, keepdims=True)

    # Handle zero-norm vectors (replace with 1 to avoid division by zero)
    norms = np.where(norms == 0, 1, norms)

    # Normalize
    E_norm = E / norms

    # Cosine similarity = dot product of normalized vectors
    similarity = E_norm @ E_norm.T

    return similarity
```

Also add the import at the top of the file (after `from __future__ import annotations`):
```python
import numpy as np
```

**Step 3.4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_dedup.py::TestSimilarityMatrix -v`
Expected: PASS (5 tests)

---

## Task 4: Implement Similarity Ordering (Greedy BFS)

**Files:**
- Modify: `vanna_kg/ingestion/resolution/entity_dedup.py`
- Modify: `tests/test_entity_dedup.py`

**Step 4.1: Add tests for similarity ordering**

Add to `tests/test_entity_dedup.py`:
```python
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
```

**Step 4.2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_dedup.py::TestSimilarityOrder -v`
Expected: FAIL with import error

**Step 4.3: Implement similarity ordering**

Add to `vanna_kg/ingestion/resolution/entity_dedup.py`:
```python
def _similarity_order(n: int, similarity_matrix: np.ndarray) -> list[int]:
    """
    Order entities so similar ones are adjacent (greedy BFS).

    Starts at entity 0, then repeatedly picks the most similar unvisited
    entity to any of the last few visited entities. This ensures that
    when we slice into batches, entities likely to be duplicates end up
    in the same batch (or in the overlap region).

    Args:
        n: Number of entities
        similarity_matrix: n x n cosine similarity matrix

    Returns:
        List of entity indices in similarity-traversal order
    """
    if n <= 1:
        return list(range(n))

    visited = [False] * n
    order: list[int] = []

    # Start at entity 0
    visited[0] = True
    order.append(0)

    while len(order) < n:
        best_next = -1
        best_sim = -1.0

        # Check last 10 visited for efficiency (or all if fewer)
        check_count = min(10, len(order))
        for visited_idx in order[-check_count:]:
            for j in range(n):
                if not visited[j] and similarity_matrix[visited_idx, j] > best_sim:
                    best_sim = similarity_matrix[visited_idx, j]
                    best_next = j

        visited[best_next] = True
        order.append(best_next)

    return order
```

**Step 4.4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_dedup.py::TestSimilarityOrder -v`
Expected: PASS (4 tests)

---

## Task 5: Implement LLM Verification for Small Clusters

**Files:**
- Modify: `vanna_kg/ingestion/resolution/entity_dedup.py`
- Modify: `tests/test_entity_dedup.py`

**Step 5.1: Add tests for LLM verification (with mock)**

Add to `tests/test_entity_dedup.py`:
```python
from unittest.mock import AsyncMock, MagicMock


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
```

**Step 5.2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_dedup.py::TestVerifyCluster -v`
Expected: FAIL with import error

**Step 5.3: Implement LLM verification prompts and function**

Add to `vanna_kg/ingestion/resolution/entity_dedup.py`:
```python
from vanna_kg.types.results import EntityDedupeResult

# -----------------------------------------------------------------------------
# LLM Prompts
# -----------------------------------------------------------------------------

_DEDUP_SYSTEM_PROMPT = """\
You are deduplicating entities for a knowledge graph.

Your task: Given a list of entities, identify which ones refer to the SAME
real-world entity and should be merged.

## MERGE - same real-world entity, different names:
- Ticker and company: "AAPL" = "Apple Inc." = "Apple"
- Abbreviations: "Fed" = "Federal Reserve"
- Person name variants: "Tim Cook" = "Timothy D. Cook" = "Mr. Cook"
- Full and short names: "Amazon.com, Inc." = "Amazon"

## DO NOT MERGE - related but distinct entities:
- Parent and subsidiary: "Alphabet" ≠ "Google" ≠ "YouTube" ≠ "Waymo"
- Person and company: "Tim Cook" ≠ "Apple"
- Product and company: "iPhone" ≠ "Apple"
- Competitors: "Goldman Sachs" ≠ "Morgan Stanley"
- Different people with same name: Context matters!

## Decision Test
Ask yourself: "In a knowledge graph, would these be the same node?"
- Would ALL facts about entity A also apply to entity B?
- If someone asks about A, should they get information about B?

## Output Rules
- Group entities that are THE SAME real-world entity
- Choose the most formal/complete name as canonical
- Merge summaries to combine information from all mentions
- Entities with no duplicates should NOT appear in output (omit singletons)"""

_DEDUP_USER_TEMPLATE = """\
ENTITIES TO ANALYZE:
{entity_list}

Group entities that refer to the SAME real-world entity.
Omit entities that have no duplicates (singletons)."""


async def _verify_cluster(
    entities: list[EnumeratedEntity],
    indices: list[int],
    llm: "LLMProvider",
) -> EntityDedupeResult:
    """
    Use LLM to verify which entities in a cluster are truly the same.

    Args:
        entities: Full list of entities
        indices: Indices of entities in this cluster
        llm: LLM provider for verification

    Returns:
        EntityDedupeResult with groups of entities to merge
    """
    # Build entity list for prompt
    entity_lines = []
    for i, idx in enumerate(indices, 1):
        e = entities[idx]
        summary_part = f": {e.summary}" if e.summary.strip() else ""
        entity_lines.append(f"{i}. {e.name} ({e.entity_type}){summary_part}")

    entity_list = "\n".join(entity_lines)
    prompt = _DEDUP_USER_TEMPLATE.format(entity_list=entity_list)

    return await llm.generate_structured(
        prompt,
        EntityDedupeResult,
        system=_DEDUP_SYSTEM_PROMPT,
    )
```

Also update the TYPE_CHECKING imports:
```python
if TYPE_CHECKING:
    from vanna_kg.providers.base import EmbeddingProvider, LLMProvider
```

**Step 5.4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_dedup.py::TestVerifyCluster -v`
Expected: PASS (2 tests)

---

## Task 6: Implement Large Cluster Batching

**Files:**
- Modify: `vanna_kg/ingestion/resolution/entity_dedup.py`
- Modify: `tests/test_entity_dedup.py`

**Step 6.1: Add tests for batch splitting**

Add to `tests/test_entity_dedup.py`:
```python
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
```

**Step 6.2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_dedup.py::TestCreateOverlappingBatches -v`
Expected: FAIL with import error

**Step 6.3: Implement batch splitting**

Add to `vanna_kg/ingestion/resolution/entity_dedup.py`:
```python
def _create_overlapping_batches(
    indices: list[int],
    max_batch_size: int = 15,
    overlap: int = 5,
) -> list[list[int]]:
    """
    Split indices into overlapping batches for LLM processing.

    Args:
        indices: List of entity indices (already in similarity order)
        max_batch_size: Maximum entities per batch
        overlap: Number of overlapping entities between batches

    Returns:
        List of batches, where each batch is a list of indices
    """
    if len(indices) <= max_batch_size:
        return [indices]

    batches: list[list[int]] = []
    step = max_batch_size - overlap
    start = 0

    while start < len(indices):
        end = min(start + max_batch_size, len(indices))
        batches.append(indices[start:end])

        # Move to next batch start
        start += step

        # If remaining elements are less than batch size, we've already captured them
        if end == len(indices):
            break

    return batches
```

**Step 6.4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_dedup.py::TestCreateOverlappingBatches -v`
Expected: PASS (5 tests)

---

## Task 7: Implement Batch Result Merging

**Files:**
- Modify: `vanna_kg/ingestion/resolution/entity_dedup.py`
- Modify: `tests/test_entity_dedup.py`

**Step 7.1: Add tests for merging batch results**

Add to `tests/test_entity_dedup.py`:
```python
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
```

**Step 7.2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_dedup.py::TestMergeBatchResults -v`
Expected: FAIL with import error

**Step 7.3: Implement batch result merging**

Add to `vanna_kg/ingestion/resolution/entity_dedup.py`:
```python
def _merge_batch_results(
    batch_results: list[tuple[list[int], EntityDedupeResult]],
    entities: list[EnumeratedEntity],
) -> list[dict]:
    """
    Merge results from overlapping batches.

    When an entity appears in multiple batches with different group assignments,
    we keep it in its first-assigned group.

    Args:
        batch_results: List of (batch_indices, EntityDedupeResult) tuples
        entities: Full entity list for name lookups

    Returns:
        List of merged group dicts with canonical, members, entity_type, reasoning
    """
    # Track which original indices are assigned to which group
    index_to_group: dict[int, int] = {}
    groups: list[dict] = []

    for batch_indices, result in batch_results:
        # Create name-to-batch-index mapping for this batch
        name_to_batch_idx: dict[str, int] = {}
        for batch_idx, orig_idx in enumerate(batch_indices):
            name_to_batch_idx[entities[orig_idx].name] = batch_idx

        for group in result.groups:
            # Get all names in this group
            all_names = [group.canonical] + group.members

            # Convert names to original indices
            orig_indices: list[int] = []
            for name in all_names:
                if name in name_to_batch_idx:
                    batch_idx = name_to_batch_idx[name]
                    orig_idx = batch_indices[batch_idx]
                    orig_indices.append(orig_idx)

            # Check if any of these indices are already assigned
            existing_group_idx = None
            for idx in orig_indices:
                if idx in index_to_group:
                    existing_group_idx = index_to_group[idx]
                    break

            if existing_group_idx is not None:
                # Merge into existing group
                existing = groups[existing_group_idx]
                for idx in orig_indices:
                    name = entities[idx].name
                    if name != existing["canonical"] and name not in existing["members"]:
                        existing["members"].append(name)
                    index_to_group[idx] = existing_group_idx
            else:
                # Create new group
                group_idx = len(groups)
                groups.append({
                    "canonical": group.canonical,
                    "members": list(group.members),
                    "entity_type": group.entity_type,
                    "reasoning": group.reasoning,
                })
                for idx in orig_indices:
                    index_to_group[idx] = group_idx

    return groups
```

**Step 7.4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_dedup.py::TestMergeBatchResults -v`
Expected: PASS (2 tests)

---

## Task 8: Implement Main deduplicate_entities Function

**Files:**
- Modify: `vanna_kg/ingestion/resolution/entity_dedup.py`
- Modify: `tests/test_entity_dedup.py`

**Step 8.1: Add integration tests**

Add to `tests/test_entity_dedup.py`:
```python
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
        mock_embeddings.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        result = await deduplicate_entities(entities, mock_llm, mock_embeddings)

        assert len(result.canonical_entities) == 1
        assert result.canonical_entities[0].name == "Apple Inc."
        assert result.canonical_entities[0].uuid  # Has a UUID
        assert result.index_to_canonical == {0: 0}

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
        mock_embeddings.embed = AsyncMock(return_value=[
            [1.0, 0.0, 0.0],  # Apple Inc.
            [0.95, 0.1, 0.0],  # AAPL (similar to Apple Inc.)
            [0.0, 0.0, 1.0],  # Google (different)
        ])

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
```

**Step 8.2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_dedup.py::TestDeduplicateEntities -v`
Expected: FAIL with import error

**Step 8.3: Implement main function**

Add to `vanna_kg/ingestion/resolution/entity_dedup.py`:
```python
import uuid as uuid_module

from vanna_kg.types.results import (
    CanonicalEntity,
    EntityDedupeResult,
    EntityDeduplicationOutput,
    MergeRecord,
)
from vanna_kg.utils.clustering import union_find_components


async def deduplicate_entities(
    entities: list[EnumeratedEntity],
    llm: "LLMProvider",
    embeddings: "EmbeddingProvider",
    *,
    similarity_threshold: float = 0.70,
    max_batch_size: int = 15,
) -> EntityDeduplicationOutput:
    """
    Deduplicate entities extracted from a single document.

    Uses embedding similarity to find candidate clusters, then LLM verification
    to make final merge decisions.

    Args:
        entities: Entities from extraction (Phase 1 output)
        llm: LLM provider for verification decisions
        embeddings: Embedding provider for similarity computation
        similarity_threshold: Minimum cosine similarity for clustering (default 0.70)
        max_batch_size: Max entities per LLM verification call (default 15)

    Returns:
        EntityDeduplicationOutput with canonical entities, index mapping, and merge history
    """
    # Handle empty input
    if not entities:
        return EntityDeduplicationOutput(
            canonical_entities=[],
            index_to_canonical={},
            merge_history=[],
        )

    n = len(entities)

    # Step 1: Generate embeddings
    texts = [_embedding_text(e) for e in entities]
    vectors = await embeddings.embed(texts)

    # Step 2: Compute similarity matrix and find clusters
    similarity_matrix = _compute_similarity_matrix(vectors)

    # Build edges where similarity >= threshold
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                edges.append((i, j))

    # Find connected components via Union-Find
    components = union_find_components(n, edges)

    # Step 3: LLM verification for multi-entity clusters
    # Maps original index -> (canonical_name, merged_summary, group_indices)
    merge_groups: list[tuple[str, str, str, list[int]]] = []  # (name, type, summary, indices)

    for component in components:
        if len(component) == 1:
            # Singleton - no verification needed
            idx = component[0]
            e = entities[idx]
            merge_groups.append((e.name, e.entity_type, e.summary, [idx]))
        elif len(component) <= max_batch_size:
            # Small cluster - single LLM call
            result = await _verify_cluster(entities, component, llm)
            groups = _process_dedup_result(result, component, entities)
            merge_groups.extend(groups)
        else:
            # Large cluster - overlapping batches
            ordered = _similarity_order(len(component), similarity_matrix[np.ix_(component, component)])
            ordered_indices = [component[i] for i in ordered]

            batches = _create_overlapping_batches(ordered_indices, max_batch_size)
            batch_results: list[tuple[list[int], EntityDedupeResult]] = []

            for batch in batches:
                result = await _verify_cluster(entities, batch, llm)
                batch_results.append((batch, result))

            merged = _merge_batch_results(batch_results, entities)
            for group in merged:
                indices = []
                for name in [group["canonical"]] + group["members"]:
                    for idx in ordered_indices:
                        if entities[idx].name == name and idx not in indices:
                            indices.append(idx)
                            break
                # Merge summaries
                summaries = [entities[idx].summary for idx in indices if entities[idx].summary.strip()]
                merged_summary = " ".join(summaries) if summaries else ""
                merge_groups.append((group["canonical"], group["entity_type"], merged_summary, indices))

            # Handle singletons not in any group
            grouped_indices = set()
            for group in merged:
                for name in [group["canonical"]] + group["members"]:
                    for idx in ordered_indices:
                        if entities[idx].name == name:
                            grouped_indices.add(idx)
                            break
            for idx in ordered_indices:
                if idx not in grouped_indices:
                    e = entities[idx]
                    merge_groups.append((e.name, e.entity_type, e.summary, [idx]))

    # Step 4: Build output
    canonical_entities: list[CanonicalEntity] = []
    index_to_canonical: dict[int, int] = {}
    merge_history: list[MergeRecord] = []

    for canonical_idx, (name, entity_type, summary, indices) in enumerate(merge_groups):
        entity_uuid = str(uuid_module.uuid4())

        # Collect aliases (names other than canonical)
        aliases = [entities[idx].name for idx in indices if entities[idx].name != name]

        canonical_entities.append(CanonicalEntity(
            uuid=entity_uuid,
            name=name,
            entity_type=entity_type,
            summary=summary,
            source_indices=indices,
            aliases=aliases,
        ))

        for idx in indices:
            index_to_canonical[idx] = canonical_idx

        # Record merge if multiple entities
        if len(indices) > 1:
            merge_history.append(MergeRecord(
                canonical_uuid=entity_uuid,
                canonical_name=name,
                merged_indices=indices,
                merged_names=[entities[idx].name for idx in indices],
                original_summaries=[entities[idx].summary for idx in indices],
                final_summary=summary,
            ))

    return EntityDeduplicationOutput(
        canonical_entities=canonical_entities,
        index_to_canonical=index_to_canonical,
        merge_history=merge_history,
    )


def _process_dedup_result(
    result: EntityDedupeResult,
    component: list[int],
    entities: list[EnumeratedEntity],
) -> list[tuple[str, str, str, list[int]]]:
    """
    Process LLM dedup result into merge groups.

    Returns list of (canonical_name, entity_type, merged_summary, indices).
    """
    groups: list[tuple[str, str, str, list[int]]] = []

    # Build name -> index mapping for this component
    name_to_idx: dict[str, int] = {entities[idx].name: idx for idx in component}

    grouped_indices: set[int] = set()

    for group in result.groups:
        all_names = [group.canonical] + group.members
        indices = [name_to_idx[name] for name in all_names if name in name_to_idx]

        if indices:
            grouped_indices.update(indices)
            # Merge summaries
            summaries = [entities[idx].summary for idx in indices if entities[idx].summary.strip()]
            merged_summary = " ".join(summaries) if summaries else ""
            groups.append((group.canonical, group.entity_type, merged_summary, indices))

    # Add singletons (entities not in any group)
    for idx in component:
        if idx not in grouped_indices:
            e = entities[idx]
            groups.append((e.name, e.entity_type, e.summary, [idx]))

    return groups
```

**Step 8.4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_dedup.py::TestDeduplicateEntities -v`
Expected: PASS (4 tests)

---

## Task 9: Update Module Exports

**Files:**
- Modify: `vanna_kg/ingestion/resolution/__init__.py`
- Modify: `vanna_kg/types/results.py` (ensure exports)

**Step 9.1: Update resolution __init__.py**

Replace contents of `vanna_kg/ingestion/resolution/__init__.py`:
```python
"""
Entity and Topic Resolution

Three levels of deduplication to ensure clean knowledge graph:

Modules:
    entity_dedup: In-document deduplication (Phase 2a-c)
    entity_registry: Cross-document entity resolution (Phase 2d)
    topic_resolver: Topic ontology resolution (Phase 2e)

In-Document Deduplication (Phase 2a-c):
    1. Generate embeddings for all entities
    2. Build similarity matrix (cosine similarity)
    3. Find connected components via Union-Find
    4. LLM verification of clusters

Cross-Document Resolution (Phase 2d):
    - Vector search for candidates in existing KB
    - LLM verification of matches
    - UUID reuse for matches, summary merging

Topic Resolution (Phase 2e):
    - Vector search against topic ontology
    - LLM verification of semantic matches

Key Principle: Subsidiary Awareness
    AWS != Amazon (subsidiaries are separate entities)

See: docs/pipeline/DEDUPLICATION_SYSTEM.md
"""

from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities

__all__ = ["deduplicate_entities"]
```

**Step 9.2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

---

## Task 10: Run Full Test Suite and Manual Verification

**Files:**
- Create: `test_dedup_manual.py` (temporary, for manual testing)

**Step 10.1: Create manual test script**

Create `test_dedup_manual.py` at project root:
```python
#!/usr/bin/env python3
"""
Manual test for entity deduplication.

Run with: uv run python test_dedup_manual.py
"""

import asyncio

from dotenv import load_dotenv

from vanna_kg.ingestion.resolution import deduplicate_entities
from vanna_kg.providers.llm.openai import OpenAILLMProvider
from vanna_kg.providers.embedding.openai import OpenAIEmbeddingProvider
from vanna_kg.types.entities import EnumeratedEntity

load_dotenv()


async def main():
    print("=" * 70)
    print("ENTITY DEDUPLICATION TEST")
    print("=" * 70)
    print()

    # Test entities with duplicates
    entities = [
        EnumeratedEntity(
            name="Apple Inc.",
            entity_type="Company",
            summary="Technology company known for iPhone and Mac computers",
        ),
        EnumeratedEntity(
            name="AAPL",
            entity_type="Company",
            summary="Stock ticker symbol for Apple",
        ),
        EnumeratedEntity(
            name="Apple",
            entity_type="Company",
            summary="Consumer electronics manufacturer",
        ),
        EnumeratedEntity(
            name="Tim Cook",
            entity_type="Person",
            summary="CEO of Apple since 2011",
        ),
        EnumeratedEntity(
            name="Timothy D. Cook",
            entity_type="Person",
            summary="Chief Executive Officer of Apple Inc.",
        ),
        EnumeratedEntity(
            name="Google",
            entity_type="Company",
            summary="Search engine and technology company",
        ),
        EnumeratedEntity(
            name="Alphabet",
            entity_type="Company",
            summary="Parent company of Google",
        ),
    ]

    print("Input entities:")
    for i, e in enumerate(entities):
        print(f"  {i}: {e.name} ({e.entity_type})")
    print()

    # Create providers
    llm = OpenAILLMProvider()
    embeddings = OpenAIEmbeddingProvider()

    print("Running deduplication...")
    print()

    result = await deduplicate_entities(entities, llm, embeddings)

    print("=" * 70)
    print(f"CANONICAL ENTITIES ({len(result.canonical_entities)})")
    print("=" * 70)
    for e in result.canonical_entities:
        print(f"  {e.name} ({e.entity_type})")
        print(f"    UUID: {e.uuid}")
        print(f"    Source indices: {e.source_indices}")
        if e.aliases:
            print(f"    Aliases: {e.aliases}")
        print()

    print("=" * 70)
    print("INDEX MAPPING")
    print("=" * 70)
    for orig_idx, canon_idx in sorted(result.index_to_canonical.items()):
        orig_name = entities[orig_idx].name
        canon_name = result.canonical_entities[canon_idx].name
        print(f"  {orig_idx} ({orig_name}) -> {canon_idx} ({canon_name})")
    print()

    print("=" * 70)
    print(f"MERGE HISTORY ({len(result.merge_history)} merges)")
    print("=" * 70)
    for record in result.merge_history:
        print(f"  {record.canonical_name}")
        print(f"    Merged: {record.merged_names}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Input entities: {len(entities)}")
    print(f"  Canonical entities: {len(result.canonical_entities)}")
    print(f"  Merges performed: {len(result.merge_history)}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 10.2: Run manual test**

Run: `uv run python test_dedup_manual.py`

Expected output should show:
- Apple Inc., AAPL, Apple merged into one entity
- Tim Cook, Timothy D. Cook merged into one entity
- Google and Alphabet kept SEPARATE (subsidiary awareness)

**Step 10.3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

---

## Summary

| Task | Files | Tests |
|------|-------|-------|
| 1 | types/results.py | 6 type validation tests |
| 2 | entity_dedup.py (skeleton) | 3 embedding text tests |
| 3 | entity_dedup.py | 5 similarity matrix tests |
| 4 | entity_dedup.py | 4 similarity order tests |
| 5 | entity_dedup.py | 2 LLM verification tests |
| 6 | entity_dedup.py | 5 batch splitting tests |
| 7 | entity_dedup.py | 2 batch merging tests |
| 8 | entity_dedup.py | 4 integration tests |
| 9 | __init__.py exports | - |
| 10 | Manual verification | - |

**Total: ~31 tests covering all functionality**
