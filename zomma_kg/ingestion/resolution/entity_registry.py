"""
Entity Registry - Cross-Document Entity Resolution

Matches new entities against existing KB entities using:
1. Vector similarity search (LanceDB)
2. LLM verification (gpt-5-mini)

This is the cross-document deduplication layer. In-document dedup
happens earlier in entity_dedup.py.
"""

import asyncio
import logging

from zomma_kg.config import KGConfig

logger = logging.getLogger(__name__)
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
        concurrency: int | None = None,
    ):
        self.storage = storage
        self.llm = llm_provider
        self.embeddings = embedding_provider
        self.config = config or KGConfig()
        # Use explicit concurrency, or fall back to config value
        self.concurrency = concurrency or self.config.registry_concurrency

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

        # Process entities in parallel with bounded concurrency
        results = await self._resolve_batch(entities, embeddings)

        new_entities: list[CanonicalEntity] = []
        uuid_remap: dict[str, str] = {}
        summary_updates: dict[str, str] = {}

        for entity, result in zip(entities, results):
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

    async def _resolve_batch(
        self,
        entities: list[CanonicalEntity],
        embeddings: list[list[float]],
    ) -> list[EntityRegistryMatch]:
        """Resolve entities in parallel with bounded concurrency."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded_resolve(
            entity: CanonicalEntity, embedding: list[float]
        ) -> EntityRegistryMatch:
            async with semaphore:
                return await self._resolve_single(entity, embedding)

        tasks = [
            bounded_resolve(entity, embedding)
            for entity, embedding in zip(entities, embeddings)
        ]
        return await asyncio.gather(*tasks)

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

        try:
            result = await self.llm.generate_structured(
                prompt, EntityRegistryMatch, system=system
            )
        except Exception as e:
            # On LLM failure, treat as new entity (conservative approach)
            logger.warning(
                f"LLM verification failed for '{entity.name}': {e}. "
                "Treating as new entity."
            )
            return EntityRegistryMatch(
                matches_existing=False,
                matched_uuid=None,
                canonical_name=entity.name,
                merged_summary=entity.summary,
                confidence=0.0,
                reasoning=f"LLM verification failed: {e}",
            )

        # If matched, merge summaries
        if result.matches_existing and result.matched_uuid:
            matched_entity = next(
                (e for e, _ in candidates if e.uuid == result.matched_uuid), None
            )
            if matched_entity:
                try:
                    result.merged_summary = await self._merge_summaries(
                        matched_entity.summary, entity.summary
                    )
                except Exception as e:
                    # On merge failure, concatenate summaries
                    logger.warning(
                        f"Summary merge failed for '{entity.name}': {e}. "
                        "Using concatenation."
                    )
                    result.merged_summary = (
                        f"{matched_entity.summary} {entity.summary}"
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
            lines.append(f"  {i}. \"{candidate.name}\" ({pct}% similar){flag}")
            lines.append(f"     UUID: {candidate.uuid}")
            summary_preview = candidate.summary[:200] + "..." if len(candidate.summary) > 200 else candidate.summary
            lines.append(f"     Summary: {summary_preview}")
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
