"""
Researcher

Resolves entity/topic hints to KB nodes and retrieves relevant chunks and facts.

Resolution Algorithm:
    1. Check cache for previous resolutions
    2. Vector search for candidates (wide net)
    3. LLM verification of matches
    4. Cache results (even empty)
    5. Deduplicate by resolved name

Retrieval Sources:
    - Entity chunks (1-hop from resolved entity)
    - Entity facts (facts mentioning the entity)
    - Entity neighbors (2-hop via shared chunks)
    - Neighbor chunks (chunks from neighbors)
    - Topic chunks (topic-related chunks)
    - Global search (direct vector similarity)

See: docs/pipeline/QUERYING_SYSTEM.md Section 2 (Resolution) and Section 3 (Retrieval)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from vanna_kg.query.types import (
    EntityResolutionResponse,
    ResolvedEntity,
    ResolvedTopic,
    RetrievedChunk,
    RetrievedFact,
    TopicResolutionResponse,
)
from vanna_kg.storage.lancedb.indices import LanceDBIndices

if TYPE_CHECKING:
    from vanna_kg.config.settings import KGConfig
    from vanna_kg.providers.base import EmbeddingProvider, LLMProvider
    from vanna_kg.storage.base import StorageBackend
    from vanna_kg.types.results import SubQuery

logger = logging.getLogger(__name__)


# System prompt for entity resolution
ENTITY_RESOLUTION_PROMPT = """You are matching query entity hints to knowledge base entities.

WIDE-NET PRINCIPLE: When in doubt, INCLUDE the match. False positives are better than false negatives.

For each hint, examine the candidate entities and determine which ones could reasonably match.
Consider:
- Exact name matches
- Partial name matches (e.g., "Microsoft" matches "Microsoft Corporation")
- Acronym matches (e.g., "IBM" matches "International Business Machines")
- Alternative names or aliases
- Contextual relevance from the definition

IMPORTANT: A hint may match MULTIPLE entities (e.g., "Boston" might match "Federal Reserve Bank of Boston" AND "Boston economy").

Return entities that should be included for retrieval."""


TOPIC_RESOLUTION_PROMPT = """You are matching query topic hints to knowledge base topics.

Each candidate topic has a NAME and a DEFINITION. Use BOTH to determine relevance:
- The definition explains what the topic actually covers
- A topic may be relevant even if its name doesn't exactly match the hint

WIDE-NET PRINCIPLE: When in doubt, INCLUDE the match. We want comprehensive coverage.
- Include direct concept matches
- Include related/overlapping concepts based on definitions
- Include parent/child topic relationships
- It's better to include too many topics than too few

IMPORTANT: Return topic names EXACTLY as they appear in the candidate list."""


class Researcher:
    """
    Resolves entity/topic hints and retrieves relevant context.

    Maintains caches for resolution results to avoid redundant LLM calls
    when multiple sub-queries share entity hints.
    """

    def __init__(
        self,
        storage: "StorageBackend",
        llm: "LLMProvider",
        embeddings: "EmbeddingProvider",
        config: "KGConfig | None" = None,
    ) -> None:
        """
        Initialize researcher.

        Args:
            storage: Storage backend for retrieval
            llm: LLM provider for resolution verification
            embeddings: Embedding provider for vector search
            config: Optional configuration
        """
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self._config = config

        # Configuration
        if config:
            self._entity_threshold = config.query_entity_threshold
            self._topic_threshold = config.query_topic_threshold
            self._max_entity_candidates = config.query_max_entity_candidates
            self._max_topic_candidates = config.query_max_topic_candidates
            self._enable_expansion = config.query_enable_expansion
            self._global_search_limit = config.query_global_search_limit
            self._topic_llm_verification = config.query_topic_llm_verification
        else:
            self._entity_threshold = 0.3
            self._topic_threshold = 0.35
            self._max_entity_candidates = 30
            self._max_topic_candidates = 20
            self._enable_expansion = True
            self._global_search_limit = 50
            self._topic_llm_verification = True

        # Resolution caches (shared across sub-queries)
        self._entity_cache: dict[str, list[ResolvedEntity]] = {}
        self._topic_cache: dict[str, list[ResolvedTopic]] = {}

        # Locks for thread-safe cache access
        self._entity_cache_lock = asyncio.Lock()
        self._topic_cache_lock = asyncio.Lock()

        # Per-hint locks to prevent duplicate resolution work
        self._entity_resolution_locks: dict[str, asyncio.Lock] = {}
        self._topic_resolution_locks: dict[str, asyncio.Lock] = {}

        # Ontology index for two-stage topic resolution
        # Uses group_id='ontology' to search the curated ontology with synonyms
        self._ontology_index: LanceDBIndices | None = None
        self._ontology_init_lock = asyncio.Lock()

    async def _ensure_ontology_index(self) -> LanceDBIndices:
        """Lazily initialize the ontology LanceDB index."""
        if self._ontology_index is not None:
            return self._ontology_index

        async with self._ontology_init_lock:
            # Double-check after acquiring lock
            if self._ontology_index is not None:
                return self._ontology_index

            # Create ontology index with group_id='ontology'
            # Import KGConfig here to avoid circular imports and handle None config
            from vanna_kg.config import KGConfig
            config = self._config if self._config is not None else KGConfig()

            lancedb_path = self.storage.kb_path / "lancedb"
            # Create and initialize in local var first to avoid race condition
            # where other tasks see self._ontology_index as not-None but uninitialized
            index = LanceDBIndices(lancedb_path, config, group_id="ontology")
            await index.initialize()
            self._ontology_index = index  # Only assign after init completes
            logger.debug("Initialized ontology index for topic resolution")
            return self._ontology_index

    async def research(
        self,
        sub_query: "SubQuery",
        *,
        query_embedding: list[float] | None = None,
        enable_expansion: bool = True,
        enable_global_search: bool = True,
    ) -> tuple[
        list[ResolvedEntity],
        list[ResolvedTopic],
        list[RetrievedChunk],
        list[RetrievedChunk],
        list[RetrievedChunk],
        list[RetrievedChunk],
        list[RetrievedFact],
        dict[str, int],
    ]:
        """
        Research a sub-query by resolving hints and retrieving context.

        Args:
            sub_query: The sub-query with entity/topic hints
            query_embedding: Pre-computed embedding for the query
            enable_expansion: Whether to expand to neighbors
            enable_global_search: Whether to do global chunk search

        Returns:
            Tuple of:
                - resolved_entities
                - resolved_topics
                - entity_chunks
                - neighbor_chunks
                - topic_chunks
                - global_chunks
                - facts
                - timing dict
        """
        timing: dict[str, int] = {}

        # Get query embedding if not provided
        if query_embedding is None:
            start = time.perf_counter_ns()
            query_embedding = await self.embeddings.embed_single(sub_query.query_text)
            timing["embed_query"] = (time.perf_counter_ns() - start) // 1_000_000

        # Phase 1: Resolution (entities + topics in parallel)
        start = time.perf_counter_ns()
        resolved_entities, resolved_topics = await asyncio.gather(
            self.resolve_entities(sub_query.entity_hints, query_embedding),
            self.resolve_topics(sub_query.topic_hints, query_embedding),
        )
        timing["resolution"] = (time.perf_counter_ns() - start) // 1_000_000

        # Phase 2: Retrieval (all operations in parallel)
        # Use named tasks for robust result handling
        start = time.perf_counter_ns()

        # Collect results in typed lists
        entity_chunks: list[RetrievedChunk] = []
        facts: list[RetrievedFact] = []
        neighbor_chunks: list[RetrievedChunk] = []
        topic_chunks: list[RetrievedChunk] = []
        global_chunks: list[RetrievedChunk] = []

        # Build named task dictionary for robust result mapping
        named_tasks: dict[str, asyncio.Task[Any]] = {}

        for i, entity in enumerate(resolved_entities):
            # Entity chunks (1-hop)
            named_tasks[f"entity_chunks:{i}"] = asyncio.create_task(
                self._get_entity_chunks(entity.resolved_name, query_embedding)
            )
            # Entity facts
            named_tasks[f"entity_facts:{i}"] = asyncio.create_task(
                self._get_entity_facts(entity.resolved_name, query_embedding)
            )
            # Neighbor expansion (if enabled)
            if enable_expansion and self._enable_expansion:
                named_tasks[f"neighbor_chunks:{i}"] = asyncio.create_task(
                    self._get_neighbor_chunks(entity.resolved_name, query_embedding)
                )

        # Topic chunks for each resolved topic
        for i, topic in enumerate(resolved_topics):
            named_tasks[f"topic_chunks:{i}"] = asyncio.create_task(
                self._get_topic_chunks(topic.resolved_name)
            )

        # Global search
        if enable_global_search:
            named_tasks["global_chunks"] = asyncio.create_task(
                self._global_chunk_search(query_embedding)
            )

        # Await all retrieval tasks
        if named_tasks:
            # Gather results with exception handling
            task_names = list(named_tasks.keys())
            task_list = list(named_tasks.values())
            results = await asyncio.gather(*task_list, return_exceptions=True)

            # Process results by name
            for name, result in zip(task_names, results):
                if isinstance(result, BaseException):
                    logger.warning(f"Retrieval task '{name}' failed: {result}")
                    continue

                # Route result to appropriate list based on task name
                if name.startswith("entity_chunks:"):
                    entity_chunks.extend(result)
                elif name.startswith("entity_facts:"):
                    facts.extend(result)
                elif name.startswith("neighbor_chunks:"):
                    neighbor_chunks.extend(result)
                elif name.startswith("topic_chunks:"):
                    topic_chunks.extend(result)
                elif name == "global_chunks":
                    global_chunks.extend(result)

        timing["retrieval"] = (time.perf_counter_ns() - start) // 1_000_000

        return (
            resolved_entities,
            resolved_topics,
            entity_chunks,
            neighbor_chunks,
            topic_chunks,
            global_chunks,
            facts,
            timing,
        )

    async def resolve_entities(
        self,
        hints: list[str],
        query_embedding: list[float],
    ) -> list[ResolvedEntity]:
        """
        Resolve entity hints to KB entities.

        Uses vector search + LLM verification with caching.
        Thread-safe via per-hint locks to prevent duplicate resolution work.

        Args:
            hints: Entity name hints from decomposition
            query_embedding: Query embedding for similarity

        Returns:
            List of resolved entities
        """
        if not hints:
            return []

        resolved: list[ResolvedEntity] = []
        hints_to_resolve: list[str] = []

        # Check cache first (with lock)
        async with self._entity_cache_lock:
            for hint in hints:
                cache_key = hint.lower().strip()
                if cache_key in self._entity_cache:
                    resolved.extend(self._entity_cache[cache_key])
                else:
                    hints_to_resolve.append(hint)

                    # Create per-hint lock if needed
                    if cache_key not in self._entity_resolution_locks:
                        self._entity_resolution_locks[cache_key] = asyncio.Lock()

        if not hints_to_resolve:
            return self._dedupe_resolved_entities(resolved)

        # Vector search for candidates
        try:
            candidates = await self.storage.search_entities(
                query_embedding,
                limit=self._max_entity_candidates,
                threshold=self._entity_threshold,
            )
        except Exception as e:
            logger.warning(f"Entity vector search failed: {e}")
            # Cache empty results (with lock)
            async with self._entity_cache_lock:
                for hint in hints_to_resolve:
                    self._entity_cache[hint.lower().strip()] = []
            return self._dedupe_resolved_entities(resolved)

        if not candidates:
            async with self._entity_cache_lock:
                for hint in hints_to_resolve:
                    self._entity_cache[hint.lower().strip()] = []
            return self._dedupe_resolved_entities(resolved)

        # Prepare candidates for LLM verification
        candidate_info = "\n".join([
            f"- {entity.name} ({entity.entity_type}): {entity.summary[:100]}..."
            if entity.summary else f"- {entity.name} ({entity.entity_type})"
            for entity, _ in candidates
        ])

        hints_info = "\n".join([f"- {h}" for h in hints_to_resolve])

        prompt = f"""Given these entity hints from a user query:
{hints_info}

And these candidate entities from the knowledge base:
{candidate_info}

For each hint, identify which candidates match. A hint can match multiple entities."""

        try:
            response = await self.llm.generate_structured(
                prompt,
                EntityResolutionResponse,
                system=ENTITY_RESOLUTION_PROMPT,
            )

            # Map resolved names back to candidates
            candidate_map = {entity.name.lower(): (entity, score) for entity, score in candidates}

            for node in response.resolved_entities:
                # Strip type annotations like "(product)" from LLM response
                # The prompt shows "Entity (type): summary" so LLM may include the type
                node_name = node.name.lower()
                if "(" in node_name:
                    node_name = node_name.rsplit("(", 1)[0].strip()
                entity_data = candidate_map.get(node_name)
                if entity_data:
                    entity, score = entity_data
                    resolved_entity = ResolvedEntity(
                        original_hint=node.name,  # Will be corrected below
                        resolved_name=entity.name,
                        resolved_uuid=entity.uuid,
                        summary=entity.summary or "",
                        entity_type=entity.entity_type,
                        confidence=score,
                    )
                    resolved.append(resolved_entity)

            # Cache results for each hint (with lock)
            async with self._entity_cache_lock:
                for hint in hints_to_resolve:
                    hint_key = hint.lower().strip()
                    matching = [
                        ResolvedEntity(
                            original_hint=hint,
                            resolved_name=r.resolved_name,
                            resolved_uuid=r.resolved_uuid,
                            summary=r.summary,
                            entity_type=r.entity_type,
                            confidence=r.confidence,
                        )
                        for r in resolved
                        if hint.lower() in r.resolved_name.lower()
                        or r.resolved_name.lower() in hint.lower()
                    ]
                    self._entity_cache[hint_key] = matching

        except Exception as e:
            logger.warning(f"LLM entity verification failed: {e}, using top candidates")
            # Fallback: use top 3 candidates by score
            fallback_entities: list[ResolvedEntity] = []
            for entity, score in candidates[:3]:
                fallback_entities.append(ResolvedEntity(
                    original_hint=hints_to_resolve[0] if hints_to_resolve else "",
                    resolved_name=entity.name,
                    resolved_uuid=entity.uuid,
                    summary=entity.summary or "",
                    entity_type=entity.entity_type,
                    confidence=score,
                ))
            resolved.extend(fallback_entities)

            # Cache the fallback (make copy for each hint to avoid shared reference)
            async with self._entity_cache_lock:
                for hint in hints_to_resolve:
                    # Create new list with proper original_hint for each
                    hint_entities = [
                        ResolvedEntity(
                            original_hint=hint,
                            resolved_name=e.resolved_name,
                            resolved_uuid=e.resolved_uuid,
                            summary=e.summary,
                            entity_type=e.entity_type,
                            confidence=e.confidence,
                        )
                        for e in fallback_entities
                    ]
                    self._entity_cache[hint.lower().strip()] = hint_entities

        return self._dedupe_resolved_entities(resolved)

    async def resolve_topics(
        self,
        hints: list[str],
        query_embedding: list[float],
    ) -> list[ResolvedTopic]:
        """
        Resolve topic hints to KB topics using two-stage resolution.

        Stage 1: Search ontology (group_id='ontology') for semantic matches.
                 The ontology contains curated topics with synonyms for better matching.

        Stage 2: Look up matched topic names in KB (group_id='default') to get
                 actual Topic objects that exist in the knowledge graph.

        This approach gives us:
        - Semantic synonym expansion from the ontology
        - Only returns topics that actually have data in our KB

        Thread-safe via per-hint locks to prevent duplicate resolution work.

        Args:
            hints: Topic name hints from decomposition
            query_embedding: Query embedding for similarity

        Returns:
            List of resolved topics
        """
        if not hints:
            return []

        resolved: list[ResolvedTopic] = []
        hints_to_resolve: list[str] = []

        # Check cache first (with lock)
        async with self._topic_cache_lock:
            for hint in hints:
                cache_key = hint.lower().strip()
                if cache_key in self._topic_cache:
                    resolved.extend(self._topic_cache[cache_key])
                else:
                    hints_to_resolve.append(hint)

                    # Create per-hint lock if needed
                    if cache_key not in self._topic_resolution_locks:
                        self._topic_resolution_locks[cache_key] = asyncio.Lock()

        if not hints_to_resolve:
            return self._dedupe_resolved_topics(resolved)

        # =====================================================================
        # STAGE 1: Search ontology for semantic matches
        # =====================================================================
        try:
            ontology_index = await self._ensure_ontology_index()
            ontology_results = await ontology_index.search_topics(
                query_embedding,
                limit=self._max_topic_candidates,
                threshold=self._topic_threshold,
            )
        except Exception as e:
            logger.warning(f"Ontology topic search failed: {e}")
            async with self._topic_cache_lock:
                for hint in hints_to_resolve:
                    self._topic_cache[hint.lower().strip()] = []
            return self._dedupe_resolved_topics(resolved)

        if not ontology_results:
            logger.debug("No ontology matches found for topic hints")
            async with self._topic_cache_lock:
                for hint in hints_to_resolve:
                    self._topic_cache[hint.lower().strip()] = []
            return self._dedupe_resolved_topics(resolved)

        # Extract ontology topic names and scores (dedupe by name, keep highest score)
        seen_names: dict[str, tuple[str, str, float]] = {}
        for result in ontology_results:
            name = result[0]["name"]
            definition = result[0].get("definition", "")
            score = result[1]
            name_lower = name.lower()
            if name_lower not in seen_names or score > seen_names[name_lower][2]:
                seen_names[name_lower] = (name, definition, score)

        ontology_candidates = list(seen_names.values())

        logger.debug(
            f"Ontology returned {len(ontology_candidates)} unique candidates: "
            f"{[c[0] for c in ontology_candidates[:5]]}"
        )

        # LLM verification to filter relevant topics (if enabled)
        matched_topic_names: list[tuple[str, str, float]] = []  # (name, definition, score)

        if not self._topic_llm_verification:
            # Skip LLM verification - use all ontology candidates directly
            logger.debug("LLM topic verification disabled, using all ontology candidates")
            matched_topic_names = ontology_candidates
        else:
            # Prepare candidates for LLM verification with full definitions
            candidate_info = "\n".join([
                f"- {name}: {definition}" if definition else f"- {name}"
                for name, definition, _ in ontology_candidates
            ])

            hints_info = "\n".join([f"- {h}" for h in hints_to_resolve])

            prompt = f"""Given these topic hints from a user query:
{hints_info}

And these candidate topics from the knowledge base (with definitions):
{candidate_info}

Which of these topics are relevant to the query hints? Return the exact topic names that match."""

            try:
                response = await self.llm.generate_structured(
                    prompt,
                    TopicResolutionResponse,
                    system=TOPIC_RESOLUTION_PROMPT,
                )

                logger.debug(
                    f"LLM topic verification returned {len(response.resolved_topics)} topics, "
                    f"no_match={response.no_match}: {[t.name for t in response.resolved_topics]}"
                )

                # Build map from ontology results
                ontology_map = {name.lower(): (name, definition, score)
                              for name, definition, score in ontology_candidates}

                for node in response.resolved_topics:
                    topic_data = ontology_map.get(node.name.lower())
                    if topic_data:
                        matched_topic_names.append(topic_data)
                    else:
                        logger.debug(f"LLM returned topic '{node.name}' not in candidates")

                # If LLM says no_match but we have candidates, use fallback
                if response.no_match and not matched_topic_names:
                    logger.debug("LLM said no_match, using top candidates as fallback")
                    matched_topic_names = ontology_candidates[:3]

            except Exception as e:
                logger.warning(f"LLM topic verification failed: {e}, using top candidates")
                matched_topic_names = ontology_candidates[:3]

        if not matched_topic_names:
            logger.debug("No topics matched after verification")
            async with self._topic_cache_lock:
                for hint in hints_to_resolve:
                    self._topic_cache[hint.lower().strip()] = []
            return self._dedupe_resolved_topics(resolved)

        # =====================================================================
        # STAGE 2: Look up matched names in KB (group_id='default')
        # =====================================================================
        topic_names_to_lookup = [name for name, _, _ in matched_topic_names]
        logger.debug(f"Looking up {len(topic_names_to_lookup)} topics in KB: {topic_names_to_lookup}")

        try:
            kb_topics = await self.storage.get_topics_by_names(topic_names_to_lookup)
        except Exception as e:
            logger.warning(f"KB topic lookup failed: {e}")
            async with self._topic_cache_lock:
                for hint in hints_to_resolve:
                    self._topic_cache[hint.lower().strip()] = []
            return self._dedupe_resolved_topics(resolved)

        if not kb_topics:
            logger.debug("No matching topics found in KB")
            async with self._topic_cache_lock:
                for hint in hints_to_resolve:
                    self._topic_cache[hint.lower().strip()] = []
            return self._dedupe_resolved_topics(resolved)

        logger.debug(f"Found {len(kb_topics)} topics in KB: {[t.name for t in kb_topics]}")

        # Build KB topic map for lookup
        kb_topic_map = {t.name.lower(): t for t in kb_topics}

        # Create resolved topics with KB data + ontology scores
        ontology_score_map = {name.lower(): score for name, _, score in matched_topic_names}

        for kb_topic in kb_topics:
            score = ontology_score_map.get(kb_topic.name.lower(), 0.5)
            resolved_topic = ResolvedTopic(
                original_hint=kb_topic.name,
                resolved_name=kb_topic.name,
                resolved_uuid=kb_topic.uuid,
                definition=kb_topic.definition or "",
                confidence=score,
            )
            resolved.append(resolved_topic)

        # Cache results (with lock)
        async with self._topic_cache_lock:
            for hint in hints_to_resolve:
                hint_key = hint.lower().strip()
                matching = [
                    ResolvedTopic(
                        original_hint=hint,
                        resolved_name=r.resolved_name,
                        resolved_uuid=r.resolved_uuid,
                        definition=r.definition,
                        confidence=r.confidence,
                    )
                    for r in resolved
                    if hint.lower() in r.resolved_name.lower()
                    or r.resolved_name.lower() in hint.lower()
                ]
                self._topic_cache[hint_key] = matching

        return self._dedupe_resolved_topics(resolved)

    async def _get_entity_chunks(
        self,
        entity_name: str,
        query_embedding: list[float],
    ) -> list[RetrievedChunk]:
        """Get chunks where entity appears (1-hop)."""
        try:
            chunk_dicts = await self.storage.get_entity_chunks(entity_name, limit=50)
        except Exception as e:
            logger.warning(f"Failed to get entity chunks for {entity_name}: {e}")
            return []

        chunks: list[RetrievedChunk] = []
        for cd in chunk_dicts:
            # Calculate vector score if we have content
            # For now, use a default score since these are graph-based retrievals
            chunks.append(RetrievedChunk(
                chunk_id=cd.get("chunk_id") or cd.get("uuid", ""),
                content=cd.get("content", ""),
                header_path=cd.get("header_path", ""),
                doc_id=cd.get("document_uuid", ""),
                doc_name=cd.get("doc_name") or cd.get("document_name", ""),
                document_date=cd.get("document_date"),
                vector_score=0.5,  # Graph-based retrieval
                source=f"entity:{entity_name}",
            ))

        return chunks

    async def _get_entity_facts(
        self,
        entity_name: str,
        query_embedding: list[float],
    ) -> list[RetrievedFact]:
        """Get facts involving an entity."""
        try:
            facts = await self.storage.get_entity_facts(entity_name, limit=100)
        except Exception as e:
            logger.warning(f"Failed to get entity facts for {entity_name}: {e}")
            return []

        retrieved: list[RetrievedFact] = []
        for fact in facts:
            retrieved.append(RetrievedFact(
                fact_id=fact.uuid,
                content=fact.content,
                subject=fact.subject_name,
                relationship_type=fact.relationship_type,
                object=fact.object_name,
                chunk_id=fact.chunk_uuid,
                vector_score=0.5,  # Graph-based retrieval
            ))

        return retrieved

    async def _get_neighbor_chunks(
        self,
        entity_name: str,
        query_embedding: list[float],
    ) -> list[RetrievedChunk]:
        """Get chunks from entity neighbors (2-hop expansion)."""
        try:
            neighbors = await self.storage.get_entity_neighbors(entity_name, limit=20)
        except Exception as e:
            logger.warning(f"Failed to get neighbors for {entity_name}: {e}")
            return []

        if not neighbors:
            return []

        chunks: list[RetrievedChunk] = []
        for neighbor in neighbors:
            neighbor_name = neighbor.get("name", "")
            if not neighbor_name:
                continue

            try:
                neighbor_chunks = await self.storage.get_entity_chunks(neighbor_name, limit=10)
                for cd in neighbor_chunks:
                    chunks.append(RetrievedChunk(
                        chunk_id=cd.get("chunk_id") or cd.get("uuid", ""),
                        content=cd.get("content", ""),
                        header_path=cd.get("header_path", ""),
                        doc_id=cd.get("document_uuid", ""),
                        doc_name=cd.get("doc_name") or cd.get("document_name", ""),
                        document_date=cd.get("document_date"),
                        vector_score=0.4,  # Slightly lower for 2-hop
                        source=f"neighbor:{neighbor_name}",
                    ))
            except Exception as e:
                logger.warning(f"Failed to get chunks for neighbor {neighbor_name}: {e}")

        return chunks

    async def _get_topic_chunks(
        self,
        topic_name: str,
    ) -> list[RetrievedChunk]:
        """Get chunks for a topic."""
        try:
            chunk_dicts = await self.storage.get_topic_chunks(topic_name, limit=50)
        except Exception as e:
            logger.warning(f"Failed to get topic chunks for {topic_name}: {e}")
            return []

        chunks: list[RetrievedChunk] = []
        for cd in chunk_dicts:
            chunks.append(RetrievedChunk(
                chunk_id=cd.get("chunk_id") or cd.get("uuid", ""),
                content=cd.get("content", ""),
                header_path=cd.get("header_path", ""),
                doc_id=cd.get("document_uuid", ""),
                doc_name=cd.get("doc_name") or cd.get("document_name", ""),
                document_date=cd.get("document_date"),
                vector_score=0.5,
                source=f"topic:{topic_name}",
            ))

        return chunks

    async def _global_chunk_search(
        self,
        query_embedding: list[float],
    ) -> list[RetrievedChunk]:
        """Perform global vector search for chunks."""
        # Note: This requires a chunk vector search method on storage
        # For now, return empty as chunk vector search may not be implemented
        # In a full implementation, storage would have search_chunks()
        logger.debug("Global chunk search not yet implemented in storage")
        return []

    def _dedupe_resolved_entities(
        self, entities: list[ResolvedEntity]
    ) -> list[ResolvedEntity]:
        """Deduplicate resolved entities by resolved_name."""
        by_name: dict[str, ResolvedEntity] = {}
        for entity in entities:
            key = entity.resolved_name.lower()
            if key not in by_name or entity.confidence > by_name[key].confidence:
                by_name[key] = entity
        return list(by_name.values())

    def _dedupe_resolved_topics(
        self, topics: list[ResolvedTopic]
    ) -> list[ResolvedTopic]:
        """Deduplicate resolved topics by resolved_name."""
        by_name: dict[str, ResolvedTopic] = {}
        for topic in topics:
            key = topic.resolved_name.lower()
            if key not in by_name or topic.confidence > by_name[key].confidence:
                by_name[key] = topic
        return list(by_name.values())

    def clear_cache(self) -> None:
        """Clear resolution caches and per-hint locks."""
        self._entity_cache.clear()
        self._topic_cache.clear()
        self._entity_resolution_locks.clear()
        self._topic_resolution_locks.clear()
