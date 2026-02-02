"""
Type Definitions

Pydantic models for all data structures.

Storage Models (persisted to Parquet/LanceDB):
    - Entity, EntityType - Entities extracted from documents
    - Fact - Relationships between entities
    - Chunk, Document - Source document content
    - Topic - Themes and concepts

Extraction Models (used during ingestion pipeline):
    - EnumeratedEntity, ExtractedFact - Raw extraction output
    - EntityGroup, EntityResolution - Deduplication results
    - TopicResolution, TopicDefinition - Topic matching
    - ChainOfThoughtResult, CritiqueResult - Extraction validation

Query Models:
    - QueryResult, SearchResult - API response types
    - QueryDecomposition, SubQuery - Query planning
    - QuestionType, EntityHint - Query classification

All types are:
    - Pydantic BaseModel subclasses
    - Fully typed with annotations
    - Serializable to/from JSON
    - Used for both internal processing and API responses
"""

# Storage Models
from zomma_kg.types.chunks import Chunk, ChunkInput, Document, DocumentPayload

# Extraction Models
from zomma_kg.types.entities import (
    Entity,
    EntityGroup,
    EntityMatchDecision,
    EntityResolution,
    EntityType,
    EntityTypeLabel,
    EnumeratedEntity,
)
from zomma_kg.types.facts import ExtractedFact, Fact

# Result Models
from zomma_kg.types.results import (
    AssemblyInput,
    AssemblyResult,
    CanonicalEntity,
    ChainOfThoughtResult,
    CritiqueResult,
    DateExtraction,
    EntityDedupeResult,
    EntityDeduplicationOutput,
    EntityHint,
    EntityRegistryMatch,
    EntityResolutionResult,
    ExtractionResult,
    IngestResult,
    MergeRecord,
    QueryDecomposition,
    QueryResult,
    QuestionType,
    SearchResult,
    SubQuery,
)
from zomma_kg.types.topics import BatchTopicDefinitions, Topic, TopicDefinition, TopicResolution

__all__ = [
    # Storage Models
    "Entity",
    "EntityType",
    "EntityTypeLabel",
    "Fact",
    "Chunk",
    "Document",
    "Topic",
    # Extraction Models
    "EnumeratedEntity",
    "EntityMatchDecision",
    "EntityGroup",
    "EntityResolution",
    "ExtractedFact",
    "TopicResolution",
    "TopicDefinition",
    "BatchTopicDefinitions",
    "DocumentPayload",
    "ChunkInput",
    # Result Models
    "QueryResult",
    "IngestResult",
    "SearchResult",
    "ExtractionResult",
    "CritiqueResult",
    "ChainOfThoughtResult",
    "EntityDedupeResult",
    "QuestionType",
    "EntityHint",
    "SubQuery",
    "QueryDecomposition",
    "DateExtraction",
    # Deduplication Output Types
    "CanonicalEntity",
    "EntityDeduplicationOutput",
    "MergeRecord",
    # Entity Registry Types
    "EntityRegistryMatch",
    "EntityResolutionResult",
    # Assembly Types
    "AssemblyInput",
    "AssemblyResult",
]
