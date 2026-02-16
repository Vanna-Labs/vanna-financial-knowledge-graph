"""
Knowledge Base Assembler

Final phase that writes resolved data to storage in efficient batches.

Write order (due to foreign key relationships):
    1. Document - No dependencies
    2. Chunks - Reference document UUID
    3. Entities - No dependencies (UUIDs already resolved)
    4. Facts - Reference entity UUIDs and chunk UUID
    5. Topics - No dependencies
    6. Relationships - Direct entity-entity edges with chunk_uuid for provenance
"""

import asyncio
import logging
from uuid import uuid4

from vanna_kg.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)
from vanna_kg.storage.base import StorageBackend
from vanna_kg.types import Entity, EntityType
from vanna_kg.types.results import AssemblyInput, AssemblyResult, CanonicalEntity


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

        Raises:
            ValueError: If embedding counts don't match entity/fact/topic counts
            Exception: If storage writes fail (partial writes may occur)
        """
        result = AssemblyResult()

        # Step 1: Generate all embeddings in parallel batches
        entity_embeddings, fact_embeddings, topic_embeddings = (
            await self._generate_embeddings(input.entities, input.facts, input.topics)
        )

        # Validate embedding counts match
        self._validate_embedding_counts(
            input.entities, entity_embeddings, "entities"
        )
        self._validate_embedding_counts(
            input.facts, fact_embeddings, "facts"
        )
        self._validate_embedding_counts(
            input.topics, topic_embeddings, "topics"
        )

        # Step 2: Write in order (respecting foreign key relationships)

        try:
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

        except Exception as e:
            logger.error(
                f"Assembly failed after writing: doc={result.document_written}, "
                f"chunks={result.chunks_written}, entities={result.entities_written}, "
                f"facts={result.facts_written}, topics={result.topics_written}. "
                f"Error: {e}"
            )
            raise

        return result

    def _validate_embedding_counts(
        self,
        items: list,
        embeddings: list[list[float]],
        item_type: str,
    ) -> None:
        """Validate that embedding count matches item count."""
        if items and len(embeddings) != len(items):
            raise ValueError(
                f"{item_type} embedding count mismatch: "
                f"got {len(embeddings)} embeddings for {len(items)} {item_type}"
            )

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
        topic_texts = (
            [f"{t.name}: {t.definition or ''}" for t in topics] if topics else []
        )

        # Generate in parallel - only call embed if there are texts
        async def embed_or_empty(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            return await self.embeddings.embed(texts)

        results = await asyncio.gather(
            embed_or_empty(entity_texts),
            embed_or_empty(fact_texts),
            embed_or_empty(topic_texts),
        )

        return results[0], results[1], results[2]

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
        Build relationships as direct entity-entity edges.

        For each fact:
            Subject Entity -> Object Entity/Topic (relationship_type from fact)
            with chunk_uuid as edge property for provenance
        """
        relationships = []

        for fact in input.facts:
            if not fact.chunk_uuid:
                continue

            # Determine object type: "entity" or "topic"
            object_type = "topic" if fact.object_type == "topic" else "entity"

            # Single direct edge: Subject -> Object
            relationships.append(
                {
                    "id": str(uuid4()),
                    "from_uuid": fact.subject_uuid,
                    "from_type": "entity",
                    "to_uuid": fact.object_uuid,
                    "to_type": object_type,
                    "rel_type": fact.relationship_type,
                    "chunk_uuid": fact.chunk_uuid,
                    "fact_id": fact.uuid,
                    "description": fact.content,
                    "date_context": fact.date_context or "",
                }
            )

        return relationships
