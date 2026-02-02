# Entity Registry & Assembler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement cross-document entity resolution and batch storage writing to complete the ingestion pipeline.

**Architecture:** Two modules - `EntityRegistry` matches new entities against existing KB using LanceDB vector search + LLM verification, `Assembler` writes resolved data to Parquet/LanceDB in efficient batches.

**Tech Stack:** Python 3.10+, Pydantic, LanceDB, DuckDB, asyncio, pytest

---

## Task 1: Add New Types to results.py

**Files:**
- Modify: `zomma_kg/types/results.py`
- Test: `tests/test_types.py`

**Step 1: Add EntityRegistryMatch type**

Add after line 354 in `zomma_kg/types/results.py`:

```python
# -----------------------------------------------------------------------------
# Entity Registry Types (cross-document resolution)
# -----------------------------------------------------------------------------


class EntityRegistryMatch(BaseModel):
    """
    LLM output for cross-document entity matching.

    Determines if a new entity matches an existing KB entity.
    """

    matches_existing: bool = Field(
        ..., description="True if entity matches an existing KB entity"
    )
    matched_uuid: str | None = Field(
        default=None, description="UUID of matched KB entity (if matches_existing)"
    )
    canonical_name: str = Field(
        ..., description="Best name to use for this entity"
    )
    merged_summary: str = Field(
        ..., description="Combined summary incorporating new information"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in match decision (0.0-1.0)"
    )
    reasoning: str = Field(
        ..., description="Explanation of match decision"
    )


class EntityResolutionResult(BaseModel):
    """
    Result from EntityRegistry.resolve().

    Contains new entities to write, UUID remapping for merged entities,
    and summary updates for existing entities.
    """

    new_entities: list["CanonicalEntity"] = Field(
        default_factory=list, description="Truly new entities not in KB"
    )
    uuid_remap: dict[str, str] = Field(
        default_factory=dict, description="Map: new_uuid -> canonical_uuid for merges"
    )
    summary_updates: dict[str, str] = Field(
        default_factory=dict, description="Map: uuid -> merged_summary for existing entities"
    )
```

**Step 2: Update __all__ in types/__init__.py**

Add to exports in `zomma_kg/types/__init__.py`:

```python
from zomma_kg.types.results import (
    # ... existing exports ...
    EntityRegistryMatch,
    EntityResolutionResult,
)
```

**Step 3: Write test for new types**

Add to `tests/test_types.py`:

```python
def test_entity_registry_match_valid():
    """Test EntityRegistryMatch with valid data."""
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


def test_entity_registry_match_no_match():
    """Test EntityRegistryMatch when entity is new."""
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


def test_entity_resolution_result():
    """Test EntityResolutionResult structure."""
    result = EntityResolutionResult(
        new_entities=[],
        uuid_remap={"old-uuid": "canonical-uuid"},
        summary_updates={"uuid-1": "Updated summary"},
    )
    assert result.uuid_remap["old-uuid"] == "canonical-uuid"
    assert "uuid-1" in result.summary_updates
```

**Step 4: Run tests**

Run: `cd /home/rithv/Programming/Startups/ZommaLabsKGRust && python -m pytest tests/test_types.py -v -k "entity_registry"`

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add zomma_kg/types/results.py zomma_kg/types/__init__.py tests/test_types.py
git commit -m "feat(types): add EntityRegistryMatch and EntityResolutionResult types"
```

---

## Task 2: Create EntityRegistry - Core Structure

**Files:**
- Create: `zomma_kg/ingestion/resolution/entity_registry.py`
- Test: `tests/test_entity_registry.py`

**Step 1: Create basic EntityRegistry class**

Create `zomma_kg/ingestion/resolution/entity_registry.py`:

```python
"""
Entity Registry - Cross-Document Entity Resolution

Matches new entities against existing KB entities using:
1. Vector similarity search (LanceDB)
2. LLM verification (gpt-5-mini)

This is the cross-document deduplication layer. In-document dedup
happens earlier in entity_dedup.py.
"""

from zomma_kg.config import KGConfig
from zomma_kg.providers.base import EmbeddingProvider, LLMProvider
from zomma_kg.storage.base import StorageBackend
from zomma_kg.types import Entity
from zomma_kg.types.results import (
    CanonicalEntity,
    EntityRegistryMatch,
    EntityResolutionResult,
)

# Constants matching original ZommaLabsKG
CANDIDATE_LIMIT = 25  # Top candidates from vector search
SIMILARITY_DISPLAY_THRESHOLD = 0.50  # Show in LLM prompt if above this
HIGH_SIMILARITY_THRESHOLD = 0.90  # Flag as "likely same" in prompt


class EntityRegistry:
    """
    Cross-document entity resolution against existing KB.

    Usage:
        registry = EntityRegistry(storage, llm, embeddings)
        result = await registry.resolve(entities)

        # result.new_entities - truly new entities to write
        # result.uuid_remap - {old_uuid: canonical_uuid} for merged entities
        # result.summary_updates - {uuid: merged_summary} for existing entities
    """

    def __init__(
        self,
        storage: StorageBackend,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        config: KGConfig | None = None,
    ):
        self.storage = storage
        self.llm = llm_provider
        self.embeddings = embedding_provider
        self.config = config or KGConfig()

    async def resolve(
        self,
        entities: list[CanonicalEntity],
        embeddings: list[list[float]] | None = None,
    ) -> EntityResolutionResult:
        """
        Resolve entities against existing KB.

        Args:
            entities: Entities from extraction (after in-doc dedup)
            embeddings: Pre-computed embeddings, or None to generate

        Returns:
            EntityResolutionResult with new_entities, uuid_remap, summary_updates
        """
        if not entities:
            return EntityResolutionResult()

        # Generate embeddings if not provided
        if embeddings is None:
            texts = [self._entity_to_text(e) for e in entities]
            embeddings = await self.embeddings.embed(texts)

        new_entities: list[CanonicalEntity] = []
        uuid_remap: dict[str, str] = {}
        summary_updates: dict[str, str] = {}

        # Process each entity
        for entity, embedding in zip(entities, embeddings):
            result = await self._resolve_single(entity, embedding)

            if result.matches_existing and result.matched_uuid:
                # Entity matches existing - remap UUID and queue summary update
                uuid_remap[entity.uuid] = result.matched_uuid
                summary_updates[result.matched_uuid] = result.merged_summary
            else:
                # Truly new entity
                new_entities.append(entity)

        return EntityResolutionResult(
            new_entities=new_entities,
            uuid_remap=uuid_remap,
            summary_updates=summary_updates,
        )

    def _entity_to_text(self, entity: CanonicalEntity) -> str:
        """Convert entity to embedding text: '{name}: {summary}'"""
        return f"{entity.name}: {entity.summary}"

    async def _resolve_single(
        self,
        entity: CanonicalEntity,
        embedding: list[float],
    ) -> EntityRegistryMatch:
        """Resolve a single entity against the KB."""
        # Search for candidates
        candidates = await self.storage.search_entities(
            embedding, limit=CANDIDATE_LIMIT, threshold=SIMILARITY_DISPLAY_THRESHOLD
        )

        if not candidates:
            # No matches - truly new entity
            return EntityRegistryMatch(
                matches_existing=False,
                matched_uuid=None,
                canonical_name=entity.name,
                merged_summary=entity.summary,
                confidence=1.0,
                reasoning="No similar entities found in knowledge base.",
            )

        # LLM verification
        return await self._llm_verify_match(entity, candidates)

    async def _llm_verify_match(
        self,
        entity: CanonicalEntity,
        candidates: list[tuple[Entity, float]],
    ) -> EntityRegistryMatch:
        """Use LLM to verify if entity matches any candidate."""
        # Build prompt
        prompt = self._build_match_prompt(entity, candidates)

        # Get structured LLM response
        system = (
            "You are an entity resolution expert. Determine if the new entity "
            "matches any existing entity in the knowledge base. Be conservative - "
            "only match if truly the same real-world entity. Subsidiaries are "
            "SEPARATE from parent companies (AWS != Amazon, YouTube != Google)."
        )

        result = await self.llm.generate_structured(
            prompt, EntityRegistryMatch, system=system
        )

        # If matched, merge summaries
        if result.matches_existing and result.matched_uuid:
            matched_entity = next(
                (e for e, _ in candidates if e.uuid == result.matched_uuid), None
            )
            if matched_entity:
                result.merged_summary = await self._merge_summaries(
                    matched_entity.summary, entity.summary
                )

        return result

    def _build_match_prompt(
        self,
        entity: CanonicalEntity,
        candidates: list[tuple[Entity, float]],
    ) -> str:
        """Build the LLM prompt for entity matching."""
        lines = [
            "Determine if this NEW ENTITY matches any EXISTING entity.",
            "",
            "NEW ENTITY:",
            f"  Name: {entity.name}",
            f"  Type: {entity.entity_type}",
            f"  Summary: {entity.summary}",
            "",
            "EXISTING CANDIDATES (sorted by similarity):",
        ]

        for i, (candidate, score) in enumerate(candidates, 1):
            pct = int(score * 100)
            flag = " [LIKELY SAME]" if score >= HIGH_SIMILARITY_THRESHOLD else ""
            lines.append(
                f"  {i}. \"{candidate.name}\" ({pct}% similar){flag}"
            )
            lines.append(f"     UUID: {candidate.uuid}")
            lines.append(f"     Summary: {candidate.summary[:200]}...")
            lines.append("")

        lines.extend([
            "RULES:",
            "- MATCH if same real-world entity (different names OK: 'Apple' = 'AAPL')",
            "- DISTINCT if related but separate (subsidiaries: AWS != Amazon)",
            "- When uncertain, prefer DISTINCT to avoid incorrect merges",
            "",
            "If matching, set matched_uuid to the candidate's UUID.",
        ])

        return "\n".join(lines)

    async def _merge_summaries(self, existing: str, new: str) -> str:
        """Merge two entity summaries using LLM."""
        prompt = f"""Merge these two entity summaries into one coherent summary.
Preserve all [Source: ...] annotations. Remove redundant information.
Keep it concise (2-4 sentences).

EXISTING SUMMARY:
{existing}

NEW INFORMATION:
{new}

Output only the merged summary, nothing else."""

        return await self.llm.generate(prompt, temperature=0.0)
```

**Step 2: Update resolution __init__.py**

Modify `zomma_kg/ingestion/resolution/__init__.py`:

```python
"""
Entity Resolution

Modules:
    entity_dedup: In-document entity deduplication
    entity_registry: Cross-document entity matching against KB
"""

from zomma_kg.ingestion.resolution.entity_dedup import EntityDeduplicator
from zomma_kg.ingestion.resolution.entity_registry import EntityRegistry

__all__ = ["EntityDeduplicator", "EntityRegistry"]
```

**Step 3: Create test file with basic tests**

Create `tests/test_entity_registry.py`:

```python
"""Tests for EntityRegistry cross-document entity resolution."""

import pytest

from zomma_kg.ingestion.resolution.entity_registry import (
    CANDIDATE_LIMIT,
    HIGH_SIMILARITY_THRESHOLD,
    SIMILARITY_DISPLAY_THRESHOLD,
    EntityRegistry,
)
from zomma_kg.types.results import CanonicalEntity, EntityResolutionResult


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

        # Create registry with mocks (we'll test the method directly)
        # For now, just verify the expected format
        expected = "Apple Inc.: Technology company that makes iPhones."
        actual = f"{entity.name}: {entity.summary}"
        assert actual == expected


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
```

**Step 4: Run tests**

Run: `cd /home/rithv/Programming/Startups/ZommaLabsKGRust && python -m pytest tests/test_entity_registry.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add zomma_kg/ingestion/resolution/entity_registry.py zomma_kg/ingestion/resolution/__init__.py tests/test_entity_registry.py
git commit -m "feat(resolution): add EntityRegistry for cross-document entity matching"
```

---

## Task 3: Create Assembler - Core Structure

**Files:**
- Create: `zomma_kg/ingestion/assembly/assembler.py`
- Test: `tests/test_assembler.py`

**Step 1: Create AssemblyInput and AssemblyResult types**

Add to `zomma_kg/types/results.py` after EntityResolutionResult:

```python
class AssemblyInput(BaseModel):
    """
    Input for the Assembler.

    Contains all resolved data ready to be written to storage.
    """

    document: "Document"
    chunks: list["Chunk"] = Field(default_factory=list)
    entities: list["CanonicalEntity"] = Field(default_factory=list)
    facts: list["Fact"] = Field(default_factory=list)
    topics: list["Topic"] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class AssemblyResult(BaseModel):
    """
    Result from Assembler.assemble().

    Contains counts of items written to storage.
    """

    document_written: bool = False
    chunks_written: int = 0
    entities_written: int = 0
    facts_written: int = 0
    topics_written: int = 0
    relationships_written: int = 0
```

Add imports at top of results.py:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zomma_kg.types.chunks import Chunk, Document
    from zomma_kg.types.facts import Fact
    from zomma_kg.types.topics import Topic
```

**Step 2: Create Assembler class**

Create `zomma_kg/ingestion/assembly/assembler.py`:

```python
"""
Knowledge Base Assembler

Final phase that writes resolved data to storage in efficient batches.

Write order (due to foreign key relationships):
    1. Document - No dependencies
    2. Chunks - Reference document UUID
    3. Entities - No dependencies (UUIDs already resolved)
    4. Facts - Reference entity UUIDs and chunk UUID
    5. Topics - No dependencies
    6. Relationships - Reference all of the above
"""

import asyncio

from zomma_kg.providers.base import EmbeddingProvider
from zomma_kg.storage.base import StorageBackend
from zomma_kg.types import Entity, EntityType
from zomma_kg.types.results import AssemblyInput, AssemblyResult, CanonicalEntity


class Assembler:
    """
    Writes resolved extraction output to storage in batches.

    Usage:
        assembler = Assembler(storage, embedding_provider)
        result = await assembler.assemble(input)
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedding_provider: EmbeddingProvider,
    ):
        self.storage = storage
        self.embeddings = embedding_provider

    async def assemble(self, input: AssemblyInput) -> AssemblyResult:
        """
        Write resolved extraction output to storage in batches.

        Args:
            input: AssemblyInput with document, chunks, entities, facts, topics
                   (entities should already be resolved via EntityRegistry,
                    fact UUIDs should already be remapped)

        Returns:
            AssemblyResult with counts of items written
        """
        result = AssemblyResult()

        # Step 1: Generate all embeddings in parallel batches
        entity_embeddings, fact_embeddings, topic_embeddings = await self._generate_embeddings(
            input.entities, input.facts, input.topics
        )

        # Step 2: Write in order (respecting foreign key relationships)

        # 2a. Document first (no dependencies)
        await self.storage.write_document(input.document)
        result.document_written = True

        # 2b. Chunks (reference document)
        if input.chunks:
            await self.storage.write_chunks(input.chunks)
            result.chunks_written = len(input.chunks)

        # 2c. Entities (no dependencies, but need embeddings for LanceDB)
        if input.entities:
            entities = self._canonical_to_entity(input.entities)
            await self.storage.write_entities(entities, entity_embeddings)
            result.entities_written = len(input.entities)

        # 2d. Facts (reference entities and chunks)
        if input.facts:
            await self.storage.write_facts(input.facts, fact_embeddings)
            result.facts_written = len(input.facts)

        # 2e. Topics
        if input.topics:
            await self.storage.write_topics(input.topics, topic_embeddings)
            result.topics_written = len(input.topics)

        # 2f. Relationships (reference all of the above)
        relationships = self._build_relationships(input)
        if relationships:
            await self.storage.write_relationships(relationships)
            result.relationships_written = len(relationships)

        return result

    async def _generate_embeddings(
        self,
        entities: list[CanonicalEntity],
        facts: list,
        topics: list,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Generate all embeddings in parallel batches."""
        # Prepare texts
        entity_texts = [f"{e.name}: {e.summary}" for e in entities] if entities else []
        fact_texts = [f.content for f in facts] if facts else []
        topic_texts = [f"{t.name}: {t.definition or ''}" for t in topics] if topics else []

        # Generate in parallel
        results = await asyncio.gather(
            self.embeddings.embed(entity_texts) if entity_texts else self._empty_embeddings(),
            self.embeddings.embed(fact_texts) if fact_texts else self._empty_embeddings(),
            self.embeddings.embed(topic_texts) if topic_texts else self._empty_embeddings(),
        )

        return results[0], results[1], results[2]

    async def _empty_embeddings(self) -> list[list[float]]:
        """Return empty embeddings list."""
        return []

    def _canonical_to_entity(self, canonicals: list[CanonicalEntity]) -> list[Entity]:
        """Convert CanonicalEntity to Entity for storage."""
        return [
            Entity(
                uuid=c.uuid,
                name=c.name,
                summary=c.summary,
                entity_type=self._map_entity_type(c.entity_type),
                aliases=c.aliases,
            )
            for c in canonicals
        ]

    def _map_entity_type(self, type_label: str) -> EntityType:
        """Map EntityTypeLabel string to EntityType enum."""
        mapping = {
            "Company": EntityType.COMPANY,
            "Person": EntityType.PERSON,
            "Organization": EntityType.ORGANIZATION,
            "Location": EntityType.LOCATION,
            "Product": EntityType.PRODUCT,
            "Topic": EntityType.CONCEPT,
        }
        return mapping.get(type_label, EntityType.CONCEPT)

    def _build_relationships(self, input: AssemblyInput) -> list[dict]:
        """
        Build relationships following chunk-centric fact pattern.

        For each fact:
            Subject Entity -> Chunk (MENTIONED_IN)
            Chunk -> Object Entity (relationship_type from fact)
        """
        relationships = []

        for fact in input.facts:
            if not fact.chunk_uuid:
                continue

            # Subject -> Chunk
            relationships.append({
                "from_uuid": fact.subject_uuid,
                "from_type": "entity",
                "to_uuid": fact.chunk_uuid,
                "to_type": "chunk",
                "rel_type": "MENTIONED_IN",
                "fact_id": fact.uuid,
                "description": "",
                "date_context": fact.date_context or "",
            })

            # Chunk -> Object
            relationships.append({
                "from_uuid": fact.chunk_uuid,
                "from_type": "chunk",
                "to_uuid": fact.object_uuid,
                "to_type": "entity",
                "rel_type": fact.relationship_type,
                "fact_id": fact.uuid,
                "description": fact.content,
                "date_context": fact.date_context or "",
            })

        return relationships
```

**Step 3: Update assembly __init__.py**

Modify `zomma_kg/ingestion/assembly/__init__.py`:

```python
"""
Knowledge Base Assembly

Final phase that writes resolved data to storage.

Modules:
    assembler: Main assembly logic

Three Sub-Phases:
    3a. Operation Collection:
        - Collect all nodes and relationships into buffer
        - Chunk-centric fact pattern (Subject -> Chunk -> Object)

    3b. Batch Embedding Generation:
        - Entity embeddings (name + summary)
        - Topic embeddings
        - Fact embeddings
        - All run in parallel with rate limit handling

    3c. Bulk Write:
        - Write to Parquet files
        - Index vectors in LanceDB
        - Write order: documents -> chunks -> entities -> facts -> topics -> relationships

Idempotency:
    - Safe to re-run (MERGE semantics)
    - fact_id for relationship deduplication

See: docs/pipeline/ASSEMBLY_SYSTEM.md
"""

from zomma_kg.ingestion.assembly.assembler import Assembler

__all__ = ["Assembler"]
```

**Step 4: Create test file**

Create `tests/test_assembler.py`:

```python
"""Tests for Assembler batch storage writing."""

import pytest

from zomma_kg.ingestion.assembly.assembler import Assembler
from zomma_kg.types import EntityType
from zomma_kg.types.results import AssemblyResult, CanonicalEntity


class TestAssemblerEntityTypeMapping:
    """Test entity type mapping from labels to enums."""

    def test_map_company(self):
        """Company label maps to COMPANY enum."""
        assembler = Assembler.__new__(Assembler)  # Create without __init__
        assert assembler._map_entity_type("Company") == EntityType.COMPANY

    def test_map_person(self):
        """Person label maps to PERSON enum."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Person") == EntityType.PERSON

    def test_map_unknown_defaults_to_concept(self):
        """Unknown labels default to CONCEPT."""
        assembler = Assembler.__new__(Assembler)
        assert assembler._map_entity_type("Unknown") == EntityType.CONCEPT


class TestAssemblerCanonicalConversion:
    """Test conversion from CanonicalEntity to Entity."""

    def test_canonical_to_entity(self):
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
```

**Step 5: Run tests**

Run: `cd /home/rithv/Programming/Startups/ZommaLabsKGRust && python -m pytest tests/test_assembler.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add zomma_kg/types/results.py zomma_kg/ingestion/assembly/assembler.py zomma_kg/ingestion/assembly/__init__.py tests/test_assembler.py
git commit -m "feat(assembly): add Assembler for batch storage writing"
```

---

## Task 4: Integration Test - EntityRegistry with Mock Storage

**Files:**
- Modify: `tests/test_entity_registry.py`

**Step 1: Add mock-based integration test**

Add to `tests/test_entity_registry.py`:

```python
from unittest.mock import AsyncMock, MagicMock

from zomma_kg.types import Entity, EntityType
from zomma_kg.types.results import EntityRegistryMatch


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
        mock_storage.search_entities = AsyncMock(
            return_value=[(existing, 0.92)]
        )

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
        mock_storage.search_entities = AsyncMock(
            return_value=[(existing, 0.75)]
        )

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
```

**Step 2: Run tests**

Run: `cd /home/rithv/Programming/Startups/ZommaLabsKGRust && python -m pytest tests/test_entity_registry.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_entity_registry.py
git commit -m "test(registry): add integration tests with mock storage/LLM"
```

---

## Task 5: Integration Test - Assembler with Mock Storage

**Files:**
- Modify: `tests/test_assembler.py`

**Step 1: Add mock-based integration test**

Add to `tests/test_assembler.py`:

```python
from unittest.mock import AsyncMock
from uuid import uuid4

from zomma_kg.types.chunks import Chunk, Document
from zomma_kg.types.facts import Fact
from zomma_kg.types.results import AssemblyInput


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
    async def test_assemble_empty_input(self, mock_storage, mock_embeddings):
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
            chunk_uuid=chunk_uuid,
        )

        input = AssemblyInput(
            document=doc,
            chunks=[chunk],
            entities=[entity],
            facts=[fact],
            topics=[],
        )

        # Mock embeddings to return correct number
        mock_embeddings.embed = AsyncMock(side_effect=[
            [[0.1] * 3072],  # entity embeddings
            [[0.2] * 3072],  # fact embeddings
            [],  # topic embeddings (empty)
        ])

        assembler = Assembler(mock_storage, mock_embeddings)
        result = await assembler.assemble(input)

        assert result.document_written is True
        assert result.chunks_written == 1
        assert result.entities_written == 1
        assert result.facts_written == 1
        assert result.relationships_written == 2  # Subject->Chunk, Chunk->Object

        # Verify write order by checking calls
        mock_storage.write_document.assert_called_once()
        mock_storage.write_chunks.assert_called_once()
        mock_storage.write_entities.assert_called_once()
        mock_storage.write_facts.assert_called_once()
        mock_storage.write_relationships.assert_called_once()

    @pytest.mark.asyncio
    async def test_relationships_built_correctly(self, mock_storage, mock_embeddings):
        """Relationships follow chunk-centric fact pattern."""
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
            chunk_uuid=chunk_uuid,
        )

        input = AssemblyInput(
            document=doc,
            chunks=[],
            entities=[],
            facts=[fact],
            topics=[],
        )

        mock_embeddings.embed = AsyncMock(return_value=[])

        assembler = Assembler(mock_storage, mock_embeddings)
        relationships = assembler._build_relationships(input)

        assert len(relationships) == 2

        # First: Subject -> Chunk
        assert relationships[0]["from_uuid"] == subject_uuid
        assert relationships[0]["to_uuid"] == chunk_uuid
        assert relationships[0]["rel_type"] == "MENTIONED_IN"
        assert relationships[0]["fact_id"] == fact_uuid

        # Second: Chunk -> Object
        assert relationships[1]["from_uuid"] == chunk_uuid
        assert relationships[1]["to_uuid"] == object_uuid
        assert relationships[1]["rel_type"] == "ACQUIRED"
```

**Step 2: Run tests**

Run: `cd /home/rithv/Programming/Startups/ZommaLabsKGRust && python -m pytest tests/test_assembler.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_assembler.py
git commit -m "test(assembler): add integration tests with mock storage"
```

---

## Task 6: Export Types from Package

**Files:**
- Modify: `zomma_kg/types/__init__.py`

**Step 1: Add new exports**

Update `zomma_kg/types/__init__.py` to export new types:

```python
from zomma_kg.types.results import (
    AssemblyInput,
    AssemblyResult,
    EntityRegistryMatch,
    EntityResolutionResult,
    # ... keep existing exports
)
```

**Step 2: Update ingestion __init__.py**

Update `zomma_kg/ingestion/__init__.py`:

```python
"""
Ingestion Pipeline

Transforms documents into knowledge graph data.

Modules:
    chunking: Document → Chunks
    extraction: Chunks → Entities + Facts
    resolution: Entity deduplication (in-doc and cross-doc)
    assembly: Write to storage
"""

from zomma_kg.ingestion.assembly import Assembler
from zomma_kg.ingestion.resolution import EntityDeduplicator, EntityRegistry

__all__ = ["Assembler", "EntityDeduplicator", "EntityRegistry"]
```

**Step 3: Run full test suite**

Run: `cd /home/rithv/Programming/Startups/ZommaLabsKGRust && python -m pytest tests/ -v --ignore=tests/test_dedup_manual.py --ignore=tests/test_extraction_proper_nouns.py`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add zomma_kg/types/__init__.py zomma_kg/ingestion/__init__.py
git commit -m "feat: export EntityRegistry and Assembler from package"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add new types | `types/results.py`, `tests/test_types.py` |
| 2 | EntityRegistry core | `resolution/entity_registry.py`, `tests/test_entity_registry.py` |
| 3 | Assembler core | `assembly/assembler.py`, `tests/test_assembler.py` |
| 4 | EntityRegistry integration tests | `tests/test_entity_registry.py` |
| 5 | Assembler integration tests | `tests/test_assembler.py` |
| 6 | Package exports | `types/__init__.py`, `ingestion/__init__.py` |

After completing all tasks, the ingestion pipeline will be complete:
```
Document → Chunk → Extract → Dedup (in-doc) → Registry (cross-doc) → Assemble → Storage
```
