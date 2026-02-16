"""
Entity Types

Entities represent real-world objects extracted from documents.

Storage Models:
    - Entity: Persisted entity with UUID and metadata
    - EntityType: Classification enum

Extraction Models (used during ingestion):
    - EnumeratedEntity: Entity discovered during chain-of-thought extraction
    - EntityMatchDecision: LLM decision on entity matching
    - EntityGroup: Group of names referring to same entity (deduplication)
    - EntityResolution: Result of resolving entity against the graph
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Literal type for entity type labels (used in extraction)
EntityTypeLabel = Literal["Company", "Person", "Organization", "Location", "Product", "Topic"]


class EntityType(str, Enum):
    """
    Entity classification types.

    IMPORTANT: Subsidiaries are SEPARATE entities from parent companies.
    AWS is NOT the same as Amazon. YouTube is NOT the same as Google.
    """

    COMPANY = "company"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    CONCEPT = "concept"  # Topics, themes, metrics


class Entity(BaseModel):
    """
    A persisted entity in the knowledge graph.

    Attributes:
        uuid: Unique identifier
        name: Canonical entity name (clean, no descriptors)
        summary: LLM-generated description
        entity_type: Classification (company, person, etc.)
        aliases: Alternative names for the entity
    """

    uuid: str
    name: str
    summary: str
    entity_type: EntityType
    aliases: list[str] = []
    created_at: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(use_enum_values=True)


# -----------------------------------------------------------------------------
# Extraction Models (used during ingestion pipeline)
# -----------------------------------------------------------------------------


class EnumeratedEntity(BaseModel):
    """
    An entity discovered during the enumeration step of chain-of-thought extraction.

    This is the first pass output before relationship generation.
    """

    name: str = Field(..., description="Entity name as it appears in the text")
    entity_type: EntityTypeLabel = Field(
        ...,
        description="Entity type: Company, Person, Organization, Location, Product, or Topic",
    )
    definition: str = Field(
        default="",
        description="Stable definition: what the entity IS (e.g., 'US central bank'). Should be the same across all mentions.",
    )
    summary: str = Field(
        default="",
        description="Context-specific findings: what was learned about this entity in this chunk",
    )


class EntityMatchDecision(BaseModel):
    """
    LLM decision on whether a new entity matches an existing one.

    Used during entity registry matching to determine if an extracted entity
    is the same as one already in the knowledge graph.
    """

    is_same: bool = Field(
        ..., description="True if new entity is same as an existing candidate"
    )
    match_index: int | None = Field(
        default=None,
        description="Index of the matching candidate (1-based), None if distinct",
    )
    reasoning: str = Field(default="", description="Brief explanation of the decision")


class EntityGroup(BaseModel):
    """
    A group of entity names that refer to the same real-world entity.

    Used during in-document deduplication to cluster aliases.
    """

    reasoning: str = Field(
        ..., description="Why these are the same entity - think step by step"
    )
    entity_type: str = Field(
        ..., description="PERSON|ORGANIZATION|INDEX|CURRENCY|COMMODITY"
    )
    canonical: str = Field(
        ..., description="The most formal/complete name to use as canonical"
    )
    members: list[str] = Field(
        ..., description="Other names/aliases (NOT including canonical)"
    )


class EntityResolution(BaseModel):
    """
    Result of resolving an entity against the knowledge graph.

    Determines whether an extracted entity matches an existing one or is new.
    """

    uuid: str = Field(..., description="UUID of the canonical entity")
    canonical_name: str = Field(..., description="Established name to use")
    is_new: bool = Field(..., description="True if this created a new entity")
    updated_summary: str = Field(
        default="", description="Combined summary with new info"
    )
    source_chunks: list[str] = Field(
        default_factory=list, description="Chunk UUIDs that contributed"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names for this entity"
    )
