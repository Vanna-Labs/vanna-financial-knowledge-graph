"""
Fact Types

Facts represent relationships between entities extracted from documents.

Storage Models:
    - Fact: Persisted fact with UUIDs linking subject and object

Extraction Models (used during ingestion):
    - ExtractedFact: Raw fact extracted from text before resolution
"""


from pydantic import BaseModel, Field

from vanna_kg.types.entities import EntityTypeLabel


class Fact(BaseModel):
    """
    A persisted fact/relationship in the knowledge graph.

    Attributes:
        uuid: Unique identifier (also used as fact_id in relationships)
        content: The fact statement text (self-contained proposition)
        subject_uuid: Subject entity UUID
        subject_name: Subject entity name
        object_uuid: Object entity/topic UUID
        object_name: Object entity/topic name
        object_type: Whether object is "entity" or "topic"
        relationship_type: Normalized relationship type (e.g., "ACQUIRED")
        date_context: Temporal context (REQUIRED for all facts)

    Chunk-Centric Pattern:
        Subject --[REL]--> Chunk --[REL_TARGET]--> Object
        The fact_id links both directions to the same fact.
    """

    uuid: str
    content: str
    subject_uuid: str
    subject_name: str
    object_uuid: str
    object_name: str
    object_type: str  # "entity" or "topic"
    relationship_type: str
    date_context: str
    chunk_uuid: str | None = None  # Source chunk for provenance
    created_at: str | None = None


# -----------------------------------------------------------------------------
# Extraction Models (used during ingestion pipeline)
# -----------------------------------------------------------------------------


class ExtractedFact(BaseModel):
    """
    A single extracted fact from text, before entity resolution.

    Financial analyst perspective: What would someone search for?
    What connections matter for understanding entity interactions?

    This is transformed into a Fact after entity/topic resolution assigns UUIDs.
    """

    fact: str = Field(
        ..., description="The atomic proposition - a complete, self-contained statement"
    )

    # Subject entity (will be resolved to UUID later)
    subject: str = Field(
        ..., description="The primary entity performing the action (company, person, org)"
    )
    subject_type: EntityTypeLabel = Field(
        ...,
        description="Entity type: Company, Person, Organization, Location, Product, or Topic",
    )
    subject_summary: str = Field(
        default="", description="1-2 sentence description of the subject entity"
    )

    # Object entity (will be resolved to UUID later)
    object: str = Field(
        ..., description="The entity being acted upon or related to"
    )
    object_type: EntityTypeLabel = Field(
        ...,
        description="Entity type: Company, Person, Organization, Location, Product, or Topic",
    )
    object_summary: str = Field(
        default="", description="1-2 sentence description of the object entity"
    )

    # Relationship - FREE-FORM (no enum!)
    relationship: str = Field(
        ...,
        description="How subject relates to object (e.g., 'acquired', 'partnered with')",
    )

    # Context - REQUIRED for temporal search
    date_context: str = Field(
        ...,
        description="Temporal context from text (e.g., 'Q3 2024') or 'Document date: YYYY-MM-DD'",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Related concepts (e.g., 'M&A', 'Earnings', 'Labor Market')",
    )
