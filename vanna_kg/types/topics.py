"""
Topic Types

Topics represent themes, concepts, and metrics from documents.

Storage Models:
    - Topic: Persisted topic with UUID and definition

Resolution Models (used during ingestion):
    - TopicResolution: Result of resolving a topic against the ontology
    - TopicDefinition: A topic with its contextual definition (for batch processing)
"""


from pydantic import BaseModel, Field


class Topic(BaseModel):
    """
    A persisted topic/theme in the knowledge graph.

    Topics are resolved against a curated ontology during ingestion.

    Attributes:
        uuid: Unique identifier
        name: Canonical topic name
        definition: Topic definition from ontology
        parent_topic: Parent topic for hierarchy (optional)
    """

    uuid: str
    name: str
    definition: str | None = None
    parent_topic: str | None = None
    created_at: str | None = None


# -----------------------------------------------------------------------------
# Resolution Models (used during ingestion pipeline)
# -----------------------------------------------------------------------------


class TopicResolution(BaseModel):
    """
    Result of resolving a topic against the ontology.

    Determines whether an extracted topic matches an existing one or is new.
    """

    uuid: str = Field(..., description="UUID of the resolved topic node")
    canonical_label: str = Field(
        ..., description="Canonical topic name from ontology"
    )
    is_new: bool = Field(
        ..., description="True if this is a new topic not in ontology"
    )
    definition: str = Field(
        default="", description="Definition of the topic from ontology"
    )

    @property
    def canonical_name(self) -> str:
        """Alias for compatibility with EntityResolution interface in fact assembly."""
        return self.canonical_label


class TopicDefinition(BaseModel):
    """
    A topic with its contextual definition.

    Used for batch topic definition generation.
    """

    topic: str = Field(description="The topic term exactly as provided")
    definition: str = Field(
        description="A one-sentence definition of what this topic means"
    )


class BatchTopicDefinitions(BaseModel):
    """Batch of topic definitions from LLM."""

    definitions: list[TopicDefinition] = Field(
        description="List of topics with their definitions"
    )


class TopicResolutionResult(BaseModel):
    """Result of resolving topics against the ontology."""

    resolved_topics: list[TopicResolution] = Field(
        default_factory=list,
        description="Successfully resolved topics",
    )
    uuid_remap: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from extracted topic name to ontology UUID",
    )
    new_topics: list[str] = Field(
        default_factory=list,
        description="Topics not found in ontology (for review)",
    )


class TopicMatchDecision(BaseModel):
    """LLM decision for a single topic match."""

    topic: str = Field(description="The extracted topic name")
    selected_number: int | None = Field(
        default=None,
        description="Candidate number (1-indexed) or null if no match",
    )
    reasoning: str = Field(description="Brief explanation of decision")


class BatchTopicMatchResponse(BaseModel):
    """Batched LLM response for topic verification."""

    decisions: list[TopicMatchDecision] = Field(
        description="Match decisions for each topic in the batch"
    )
