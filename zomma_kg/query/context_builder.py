"""
Context Builder

Assembles and deduplicates retrieved chunks and facts into StructuredContext.

Responsibilities:
    - Deduplicate chunks by chunk_id (keep highest score)
    - Deduplicate facts by fact_id (keep highest score)
    - Split chunks into high/low relevance by threshold
    - Apply size limits to each category
    - Preserve source attribution for provenance

See: docs/pipeline/QUERYING_SYSTEM.md Section 4 (Assembly)
"""

from __future__ import annotations

from zomma_kg.query.types import (
    ResolvedEntity,
    ResolvedTopic,
    RetrievedChunk,
    RetrievedFact,
    StructuredContext,
)


class ContextBuilder:
    """
    Builds structured context from retrieved chunks and facts.

    Handles deduplication and relevance-based organization.
    """

    def __init__(
        self,
        *,
        high_relevance_threshold: float = 0.45,
        max_high_relevance_chunks: int = 30,
        max_facts: int = 40,
        max_topic_chunks: int = 15,
        max_low_relevance_chunks: int = 20,
    ) -> None:
        """
        Initialize context builder with limits.

        Args:
            high_relevance_threshold: Score threshold for high-relevance classification
            max_high_relevance_chunks: Maximum chunks in high-relevance category
            max_facts: Maximum facts to include
            max_topic_chunks: Maximum topic-related chunks
            max_low_relevance_chunks: Maximum low-relevance supporting chunks
        """
        self.high_relevance_threshold = high_relevance_threshold
        self.max_high_relevance_chunks = max_high_relevance_chunks
        self.max_facts = max_facts
        self.max_topic_chunks = max_topic_chunks
        self.max_low_relevance_chunks = max_low_relevance_chunks

    def build(
        self,
        resolved_entities: list[ResolvedEntity],
        resolved_topics: list[ResolvedTopic],
        entity_chunks: list[RetrievedChunk],
        neighbor_chunks: list[RetrievedChunk],
        topic_chunks: list[RetrievedChunk],
        global_chunks: list[RetrievedChunk],
        facts: list[RetrievedFact],
    ) -> StructuredContext:
        """
        Build structured context from all retrieval sources.

        Deduplicates, splits by relevance, and applies limits.

        Args:
            resolved_entities: Successfully resolved entities
            resolved_topics: Successfully resolved topics
            entity_chunks: Chunks from entity 1-hop
            neighbor_chunks: Chunks from entity 2-hop (neighbors)
            topic_chunks: Topic-related chunks
            global_chunks: Chunks from global vector search
            facts: Retrieved facts

        Returns:
            StructuredContext ready for synthesis
        """
        # Combine all non-topic chunks for deduplication
        all_chunks = entity_chunks + neighbor_chunks + global_chunks
        deduped_chunks = self._dedupe_chunks(all_chunks)

        # Split into high/low relevance
        high_relevance, low_relevance = self._split_by_relevance(deduped_chunks)

        # Apply limits (already sorted by score from dedup)
        high_relevance = high_relevance[: self.max_high_relevance_chunks]
        low_relevance = low_relevance[: self.max_low_relevance_chunks]

        # Dedupe and limit topic chunks separately
        deduped_topics = self._dedupe_chunks(topic_chunks)
        deduped_topics = deduped_topics[: self.max_topic_chunks]

        # Dedupe and limit facts
        deduped_facts = self._dedupe_facts(facts)
        deduped_facts = deduped_facts[: self.max_facts]

        return StructuredContext(
            resolved_entities=resolved_entities,
            resolved_topics=resolved_topics,
            high_relevance_chunks=high_relevance,
            low_relevance_chunks=low_relevance,
            topic_chunks=deduped_topics,
            facts=deduped_facts,
        )

    def _dedupe_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Deduplicate chunks by chunk_id, keeping highest vector_score.

        Returns chunks sorted by score (descending).
        """
        by_id: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            existing = by_id.get(chunk.chunk_id)
            if existing is None or chunk.vector_score > existing.vector_score:
                by_id[chunk.chunk_id] = chunk

        # Sort by score descending
        return sorted(by_id.values(), key=lambda c: c.vector_score, reverse=True)

    def _dedupe_facts(
        self, facts: list[RetrievedFact]
    ) -> list[RetrievedFact]:
        """
        Deduplicate facts by fact_id, keeping highest vector_score.

        Returns facts sorted by score (descending).
        """
        by_id: dict[str, RetrievedFact] = {}
        for fact in facts:
            existing = by_id.get(fact.fact_id)
            if existing is None or fact.vector_score > existing.vector_score:
                by_id[fact.fact_id] = fact

        # Sort by score descending
        return sorted(by_id.values(), key=lambda f: f.vector_score, reverse=True)

    def _split_by_relevance(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """
        Split chunks into high and low relevance based on threshold.

        Args:
            chunks: Deduplicated chunks (should be sorted by score)

        Returns:
            Tuple of (high_relevance, low_relevance) lists
        """
        high: list[RetrievedChunk] = []
        low: list[RetrievedChunk] = []

        for chunk in chunks:
            if chunk.vector_score >= self.high_relevance_threshold:
                high.append(chunk)
            else:
                low.append(chunk)

        return high, low
