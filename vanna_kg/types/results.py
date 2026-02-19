"""
Result Types

Types for query, ingestion, and extraction results.

API Result Models:
    - QueryResult: Final answer from query pipeline
    - IngestResult: Summary of document ingestion
    - SearchResult: Result from vector/keyword search

Extraction Result Models (used during ingestion):
    - ExtractionResult: Facts extracted from a chunk
    - CritiqueResult: Reflexion critique of extraction quality
    - ChainOfThoughtResult: Combined entity enumeration + fact extraction
    - EntityDedupeResult: Entity deduplication output

Query Pipeline Models:
    - QuestionType: Classification of question types
    - SubQuery: A focused sub-query for targeted retrieval
    - QueryDecomposition: Structured decomposition of a question
    - EntityHint: Entity hint with contextual definition
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vanna_kg.types.chunks import Chunk

from vanna_kg.types.entities import EntityGroup, EnumeratedEntity, EntityTypeLabel
from vanna_kg.types.facts import ExtractedFact

# -----------------------------------------------------------------------------
# API Result Models
# -----------------------------------------------------------------------------


class QueryResult(BaseModel):
    """
    Result from a knowledge graph query.

    Attributes:
        answer: The synthesized answer
        confidence: Confidence score (0-1)
        sources: Source citations (if include_sources=True)
        sub_answers: Answers to decomposed sub-queries
        question_type: Detected question type (FACTUAL, COMPARISON, etc.)
        timing: Timing information for each phase (milliseconds)
    """

    answer: str
    confidence: float
    sources: list[dict[str, Any]] = []
    sub_answers: list[dict[str, Any]] = []
    question_type: str | None = None
    timing: dict[str, int] = {}

    @property
    def total_time_ms(self) -> int:
        """Total query time in milliseconds."""
        return sum(self.timing.values())


class IngestResult(BaseModel):
    """
    Result from document ingestion.

    Attributes:
        document_id: UUID of the ingested document
        chunks: Number of chunks created
        entities: Number of entities extracted
        facts: Number of facts extracted
        topics: Number of topics identified
        duration_seconds: Total processing time
        errors: Any errors encountered (non-fatal)
    """

    document_id: str
    chunks: int
    entities: int
    facts: int
    topics: int
    duration_seconds: float
    errors: list[str] = []


class SearchResult(BaseModel):
    """
    Result from a search operation.

    Attributes:
        uuid: ID of the matched item
        content: Content or summary
        score: Similarity score
        metadata: Additional metadata
    """

    uuid: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


class ChunkMatch(BaseModel):
    """
    Result from semantic chunk search.

    Attributes:
        chunk: The matched chunk
        score: Similarity score
    """

    chunk: Chunk
    score: float


# -----------------------------------------------------------------------------
# Extraction Result Models (used during ingestion pipeline)
# -----------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    """
    Result of extracting facts from a single chunk.
    """

    facts: list[ExtractedFact] = Field(
        default_factory=list, description="List of extracted facts from the chunk"
    )


class CritiqueResult(BaseModel):
    """
    Result of the reflexion critique step.

    The critique step verifies extraction quality and identifies issues.
    """

    is_approved: bool = Field(
        ..., description="True if extraction is satisfactory, False if issues found"
    )
    critique: str | None = Field(
        default=None,
        description="Specific issues found and corrections needed (if not approved)",
    )
    missed_facts: list[str] = Field(
        default_factory=list,
        description="Facts that should have been extracted but were missed",
    )
    corrections: list[str] = Field(
        default_factory=list,
        description="Specific corrections to entity names, types, or relationships",
    )


class ChainOfThoughtResult(BaseModel):
    """
    Chain-of-thought extraction result: enumerate entities first, then generate relationships.

    This forces the LLM to explicitly list all entities before determining relationships,
    which improves entity coverage compared to single-pass extraction.
    """

    entities: list[EnumeratedEntity] = Field(
        default_factory=list,
        description="Step 1: ALL entities mentioned in the text (companies, people, orgs, etc.)",
    )
    facts: list[ExtractedFact] = Field(
        default_factory=list,
        description="Step 2: Relationships between the enumerated entities",
    )


class EntityDedupeResult(BaseModel):
    """
    LLM output for entity deduplication with reasoning.

    Groups entities that refer to the same real-world entity.
    """

    groups: list[EntityGroup] = Field(
        default_factory=list,
        description=(
            "Entity groups where each group contains names referring to the same entity. "
            "Singleton entities (no aliases) should be omitted."
        ),
    )


# -----------------------------------------------------------------------------
# Query Pipeline Models
# -----------------------------------------------------------------------------


class QuestionType(str, Enum):
    """Classification of question types for routing retrieval and expansion."""

    FACTUAL = "factual"  # Simple fact lookup: "What happened in Boston?"
    COMPARISON = "comparison"  # Compare entities: "How do Boston and NY differ?"
    CAUSAL = "causal"  # Cause/effect: "Why did wages increase?"
    TEMPORAL = "temporal"  # Time-based: "What changed from Oct to Nov?"
    ENUMERATION = "enumeration"  # List items: "Which districts saw growth?"


class EntityHint(BaseModel):
    """
    An entity or topic hint with contextual definition for better matching.

    Used during query decomposition to help entity resolution.
    """

    name: str = Field(
        ..., description="The entity/topic name as mentioned in the question"
    )
    definition: str = Field(
        ...,
        description="Brief contextual definition to aid matching (e.g., 'Fed regional bank')",
    )


class SubQuery(BaseModel):
    """
    A focused sub-query for targeted retrieval.

    Generated during query decomposition to break complex questions into searchable parts.
    """

    query_text: str = Field(..., description="Search query text (e.g., 'inflation Boston')")
    target_info: str = Field(..., description="What this sub-query aims to find")
    entity_hints: list[str] = Field(
        default_factory=list, description="Entity names to resolve"
    )
    topic_hints: list[str] = Field(
        default_factory=list, description="Topic names to resolve"
    )


class QueryDecomposition(BaseModel):
    """
    Structured output from query decomposition phase.

    Uses chain-of-thought: enumerate required info -> generate sub-queries -> classify.
    """

    # Step 1: What information is needed?
    required_info: list[str] = Field(
        ..., description="List of distinct pieces of information needed to answer"
    )

    # Step 2: Sub-queries to find that information
    sub_queries: list[SubQuery] = Field(
        ..., description="Targeted search queries (combinatorial for multi-entity)"
    )

    # Step 3: Hints for direct graph lookup (with definitions for better matching)
    entity_hints: list[EntityHint] = Field(
        default_factory=list,
        description="Entities to resolve, each with a contextual definition",
    )
    topic_hints: list[EntityHint] = Field(
        default_factory=list,
        description="Topics/themes to search for, each with a contextual definition",
    )
    relationship_hints: list[str] = Field(
        default_factory=list,
        description="Relationship phrases with modifiers (e.g., 'reported slight growth')",
    )

    # Step 4: Temporal scope
    temporal_scope: str | None = Field(
        default=None,
        description="Time period if specified (e.g., 'October 2025', 'recent')",
    )

    # Step 5: Question classification
    question_type: QuestionType = Field(
        ..., description="Classification for routing to retrieval/expansion strategy"
    )

    # Confidence and reasoning
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Chain-of-thought explanation")


# -----------------------------------------------------------------------------
# Temporal Extraction
# -----------------------------------------------------------------------------


class DateExtraction(BaseModel):
    """
    Extracted date information from a document.

    Used to determine the document's publication/creation date.
    """

    date_found: bool = Field(
        ..., description="Whether a valid document creation/publication date was found"
    )
    year: int | None = Field(default=None, description="The year of the document (YYYY)")
    month: int | None = Field(default=None, description="The month of the document (1-12)")
    day: int | None = Field(default=None, description="The day of the document (1-31)")
    reasoning: str = Field(
        default="", description="Quote the text that indicates this date"
    )


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

    new_entities: list[CanonicalEntity] = Field(
        default_factory=list, description="Truly new entities not in KB"
    )
    uuid_remap: dict[str, str] = Field(
        default_factory=dict, description="Map: new_uuid -> canonical_uuid for merges"
    )
    summary_updates: dict[str, str] = Field(
        default_factory=dict, description="Map: uuid -> merged_summary for existing entities"
    )


# -----------------------------------------------------------------------------
# Assembly Types (batch storage writing)
# -----------------------------------------------------------------------------


class AssemblyInput(BaseModel):
    """
    Input for the Assembler.

    Contains all resolved data ready to be written to storage.
    """

    document: Any = Field(..., description="Document to write")  # Avoid circular import
    chunks: list[Any] = Field(default_factory=list, description="Chunks to write")
    entities: list[CanonicalEntity] = Field(default_factory=list, description="Resolved entities")
    facts: list[Any] = Field(default_factory=list, description="Facts to write")
    topics: list[Any] = Field(default_factory=list, description="Topics to write")

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
