"""
Query Pipeline Types

New Pydantic models for the V7 query pipeline retrieval and synthesis phases.

Resolution Types:
    - ResolvedEntity: Entity hint resolved to a KB node
    - ResolvedTopic: Topic hint resolved to a KB topic

Retrieval Types:
    - RetrievedChunk: Chunk with provenance and relevance score
    - RetrievedFact: Fact with vector similarity score
    - StructuredContext: Assembled context for synthesis

Synthesis Types:
    - SubAnswer: Answer to a single sub-query
    - PipelineResult: Final pipeline output with timing

LLM Output Schemas:
    - ResolvedNode: Single resolution result
    - EntityResolutionResponse: LLM resolution output
    - SubAnswerSynthesis: Sub-query answer synthesis
    - FinalSynthesis: Final answer synthesis
"""

from typing import Any

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Resolution Types
# -----------------------------------------------------------------------------


class ResolvedEntity(BaseModel):
    """
    An entity hint that has been resolved to an actual KB entity.

    Resolution happens via vector search + LLM verification:
    1. Search KB entities by embedding similarity
    2. LLM verifies which candidates match the hint
    """

    original_hint: str = Field(
        ..., description="The original entity hint from query decomposition"
    )
    resolved_name: str = Field(
        ..., description="The canonical name in the KB"
    )
    resolved_uuid: str = Field(
        ..., description="UUID of the resolved entity in KB"
    )
    summary: str = Field(
        default="", description="Entity summary from KB"
    )
    entity_type: str = Field(
        default="concept", description="Entity type (company, person, etc.)"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Resolution confidence"
    )


class ResolvedTopic(BaseModel):
    """
    A topic hint resolved to an actual KB topic.

    Topics represent themes/concepts like 'M&A', 'economic outlook', etc.
    """

    original_hint: str = Field(
        ..., description="The original topic hint from query decomposition"
    )
    resolved_name: str = Field(
        ..., description="The canonical topic name in KB"
    )
    resolved_uuid: str = Field(
        ..., description="UUID of the resolved topic in KB"
    )
    definition: str = Field(
        default="", description="Topic definition from ontology"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Resolution confidence"
    )


# -----------------------------------------------------------------------------
# Retrieval Types
# -----------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """
    A retrieved chunk with provenance and relevance metadata.

    Chunks are retrieved from multiple sources:
    - Entity chunks (1-hop from resolved entity)
    - Neighbor chunks (2-hop from entity neighbors)
    - Topic chunks (topic-related)
    - Global search (direct vector similarity)
    """

    chunk_id: str = Field(
        ..., description="UUID of the chunk"
    )
    content: str = Field(
        ..., description="Chunk text content"
    )
    header_path: str = Field(
        default="", description="Breadcrumb path (e.g., 'Section 1 > Overview')"
    )
    doc_id: str = Field(
        default="", description="Parent document UUID"
    )
    doc_name: str = Field(
        default="", description="Parent document name"
    )
    document_date: str | None = Field(
        default=None, description="Document date (ISO format)"
    )
    vector_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Vector similarity score"
    )
    source: str = Field(
        default="", description="Retrieval source (e.g., 'entity:Apple', 'topic:M&A', 'global')"
    )


class RetrievedFact(BaseModel):
    """
    A retrieved fact with its relationship and provenance.

    Facts are atomic propositions extracted during ingestion.
    """

    fact_id: str = Field(
        ..., description="UUID of the fact"
    )
    content: str = Field(
        ..., description="Fact statement"
    )
    subject: str = Field(
        ..., description="Subject entity name"
    )
    relationship_type: str = Field(
        ..., description="Normalized relationship type (e.g., 'ACQUIRED')"
    )
    object: str = Field(
        ..., description="Object entity/topic name"
    )
    chunk_id: str | None = Field(
        default=None, description="Source chunk UUID for provenance"
    )
    vector_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Vector similarity score"
    )


class StructuredContext(BaseModel):
    """
    Assembled context for synthesis, organized by source and relevance.

    This is the input to the synthesizer after retrieval and deduplication.
    Context is split into high/low relevance based on vector scores.
    """

    resolved_entities: list[ResolvedEntity] = Field(
        default_factory=list, description="Successfully resolved entities"
    )
    resolved_topics: list[ResolvedTopic] = Field(
        default_factory=list, description="Successfully resolved topics"
    )
    high_relevance_chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="High-relevance chunks (score >= threshold)"
    )
    low_relevance_chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="Lower-relevance supporting chunks"
    )
    topic_chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="Topic-related chunks"
    )
    facts: list[RetrievedFact] = Field(
        default_factory=list, description="Retrieved facts"
    )

    def to_prompt_text(self) -> str:
        """
        Format context as text for LLM prompt.

        Organizes context sections for optimal synthesis.
        """
        sections: list[str] = []

        # Entities section
        if self.resolved_entities:
            entity_lines = [
                f"- {e.resolved_name}: {e.summary}" if e.summary
                else f"- {e.resolved_name} ({e.entity_type})"
                for e in self.resolved_entities
            ]
            sections.append("## Relevant Entities\n" + "\n".join(entity_lines))

        # Facts section
        if self.facts:
            fact_lines = [
                f"- {f.subject} {f.relationship_type} {f.object}: {f.content}"
                for f in self.facts
            ]
            sections.append("## Key Facts\n" + "\n".join(fact_lines))

        # High relevance chunks
        if self.high_relevance_chunks:
            chunk_lines = []
            for c in self.high_relevance_chunks:
                header = f"[{c.header_path}]" if c.header_path else ""
                date = f" ({c.document_date})" if c.document_date else ""
                chunk_lines.append(f"### {c.doc_name}{date} {header}\n{c.content}")
            sections.append("## Primary Evidence\n" + "\n\n".join(chunk_lines))

        # Topic chunks
        if self.topic_chunks:
            chunk_lines = []
            for c in self.topic_chunks:
                header = f"[{c.header_path}]" if c.header_path else ""
                chunk_lines.append(f"### {c.doc_name} {header}\n{c.content}")
            sections.append("## Topic Context\n" + "\n\n".join(chunk_lines))

        # Low relevance chunks (supporting evidence)
        if self.low_relevance_chunks:
            chunk_lines = []
            for c in self.low_relevance_chunks:
                chunk_lines.append(f"- [{c.doc_name}] {c.content[:200]}...")
            sections.append("## Supporting Evidence\n" + "\n".join(chunk_lines))

        return "\n\n".join(sections) if sections else "(No context available)"

    def is_empty(self) -> bool:
        """Check if context has any content."""
        return (
            not self.resolved_entities
            and not self.resolved_topics
            and not self.high_relevance_chunks
            and not self.low_relevance_chunks
            and not self.topic_chunks
            and not self.facts
        )


# -----------------------------------------------------------------------------
# Synthesis Types
# -----------------------------------------------------------------------------


class SubAnswer(BaseModel):
    """
    Answer to a single sub-query from the decomposition phase.

    Each sub-answer contributes to the final synthesized answer.
    """

    sub_query: str = Field(
        ..., description="The sub-query text"
    )
    target_info: str = Field(
        ..., description="What this sub-query aimed to find"
    )
    answer: str = Field(
        ..., description="The synthesized answer"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this answer"
    )
    entities_found: list[str] = Field(
        default_factory=list, description="Entities mentioned in the answer"
    )
    timing: dict[str, int] = Field(
        default_factory=dict, description="Phase timings in milliseconds"
    )


class PipelineResult(BaseModel):
    """
    Final result from the V7 query pipeline.

    Contains the complete answer with sub-answers, confidence, and sources.
    """

    question: str = Field(
        ..., description="The original question"
    )
    answer: str = Field(
        ..., description="The synthesized answer"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall confidence"
    )
    sub_answers: list[SubAnswer] = Field(
        default_factory=list, description="Answers to each sub-query"
    )
    question_type: str = Field(
        default="factual", description="Detected question type"
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Source citations"
    )
    timing: dict[str, int] = Field(
        default_factory=dict, description="Phase timings in milliseconds"
    )

    @property
    def total_time_ms(self) -> int:
        """Total query time in milliseconds."""
        return sum(self.timing.values())


# -----------------------------------------------------------------------------
# LLM Output Schemas
# -----------------------------------------------------------------------------


class ResolvedNode(BaseModel):
    """
    A single resolved node from LLM verification.

    Used during entity/topic resolution to verify which candidates match.
    """

    name: str = Field(
        ..., description="Name of the matched entity/topic from KB"
    )
    reason: str = Field(
        default="", description="Why this candidate matches the hint"
    )


class EntityResolutionResponse(BaseModel):
    """
    LLM output for entity resolution verification.

    Given a hint and candidate entities, the LLM determines matches.
    Uses a "wide-net" approach - prefers false positives over false negatives.
    """

    resolved_entities: list[ResolvedNode] = Field(
        default_factory=list, description="Entities that match the hint"
    )
    no_match: bool = Field(
        default=False, description="True if no candidates match the hint"
    )


class TopicResolutionResponse(BaseModel):
    """
    LLM output for topic resolution verification.
    """

    resolved_topics: list[ResolvedNode] = Field(
        default_factory=list, description="Topics that match the hint"
    )
    no_match: bool = Field(
        default=False, description="True if no candidates match the hint"
    )


class SubAnswerSynthesis(BaseModel):
    """
    LLM output for sub-query answer synthesis.

    Generated from the structured context for each sub-query.
    """

    answer: str = Field(
        ..., description="Synthesized answer to the sub-query"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in the answer"
    )
    entities_mentioned: list[str] = Field(
        default_factory=list, description="Entities referenced in the answer"
    )


class FinalSynthesis(BaseModel):
    """
    LLM output for final answer synthesis.

    Merges sub-answers according to question type (comparison, temporal, etc.).
    """

    answer: str = Field(
        ..., description="Final synthesized answer"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall confidence"
    )
