"""
GraphRAG Query Pipeline

Orchestrates the complete query pipeline:
    1. Decomposition: Break question into sub-queries
    2. Research: Resolve hints and retrieve context per sub-query
    3. Context Assembly: Deduplicate and organize retrieved data
    4. Synthesis: Generate sub-answers and final answer

Features:
    - Parallel sub-query research with semaphore-bounded concurrency
    - Resolution caching across sub-queries
    - Question-type-aware synthesis
    - Comprehensive timing metrics
    - Graceful error handling with fallbacks

See: docs/pipeline/QUERYING_SYSTEM.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from vanna_kg.query.context_builder import ContextBuilder
from vanna_kg.query.decomposer import QueryDecomposer
from vanna_kg.query.researcher import Researcher
from vanna_kg.query.synthesizer import Synthesizer
from vanna_kg.query.types import (
    PipelineResult,
    SubAnswer,
)
from vanna_kg.types.results import QueryDecomposition, QuestionType, SubQuery
from vanna_kg.utils.cost_telemetry import telemetry_stage

if TYPE_CHECKING:
    from vanna_kg.config.settings import KGConfig
    from vanna_kg.providers.base import EmbeddingProvider, LLMProvider
    from vanna_kg.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    GraphRAG Query Pipeline.

    Orchestrates decomposition, research, context assembly, and synthesis
    for knowledge graph queries.
    """

    def __init__(
        self,
        storage: "StorageBackend",
        llm: "LLMProvider",
        embeddings: "EmbeddingProvider",
        config: "KGConfig | None" = None,
    ) -> None:
        """
        Initialize pipeline with providers and configuration.

        Args:
            storage: Storage backend for retrieval
            llm: LLM provider for decomposition and synthesis
            embeddings: Embedding provider for vector search
            config: Optional configuration
        """
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.config = config

        # Initialize components
        self.decomposer = QueryDecomposer(llm, config)
        self.researcher = Researcher(storage, llm, embeddings, config)
        self.context_builder = ContextBuilder(
            high_relevance_threshold=config.query_high_relevance_threshold if config else 0.45,
            max_high_relevance_chunks=config.query_max_high_relevance_chunks if config else 30,
            max_facts=config.query_max_facts if config else 40,
            max_topic_chunks=config.query_max_topic_chunks if config else 15,
            max_low_relevance_chunks=config.query_max_low_relevance_chunks if config else 20,
        )
        self.synthesizer = Synthesizer(llm, config)

        # Concurrency control
        self._research_semaphore = asyncio.Semaphore(
            config.query_research_concurrency if config else 5
        )

        # Configuration
        self._enable_expansion = config.query_enable_expansion if config else True
        self._enable_global_search = config.query_enable_global_search if config else True

    async def query(
        self,
        question: str,
        *,
        include_sources: bool = True,
        enable_expansion: bool | None = None,
    ) -> PipelineResult:
        """
        Execute the complete query pipeline.

        Args:
            question: User's question
            include_sources: Whether to include source citations
            enable_expansion: Override for neighbor expansion (uses config default if None)

        Returns:
            PipelineResult with answer, confidence, sub-answers, and timing
        """
        timing: dict[str, int] = {}

        # Use config default if not overridden
        expansion = enable_expansion if enable_expansion is not None else self._enable_expansion

        # Phase 1: Decomposition
        decomposition, decomp_time = await self._phase_decomposition(question)
        timing["decomposition"] = decomp_time

        # Phase 2: Research (parallel sub-queries)
        sub_answers, research_time = await self._phase_research(
            decomposition, question, enable_expansion=expansion
        )
        timing["research"] = research_time

        # Phase 3: Synthesis
        final_answer, confidence, synthesis_time = await self._phase_synthesis(
            question, sub_answers, decomposition.question_type
        )
        timing["synthesis"] = synthesis_time

        # Build sources if requested
        sources: list[dict[str, Any]] = []
        if include_sources:
            sources = self._extract_sources(sub_answers)

        return PipelineResult(
            question=question,
            answer=final_answer,
            confidence=confidence,
            sub_answers=sub_answers,
            question_type=decomposition.question_type.value,
            sources=sources,
            timing=timing,
        )

    async def _phase_decomposition(
        self, question: str
    ) -> tuple[QueryDecomposition, int]:
        """
        Phase 1: Decompose question into sub-queries.

        Returns:
            Tuple of (decomposition, time_ms)
        """
        start = time.perf_counter_ns()

        with telemetry_stage("decomposition"):
            decomposition = await self.decomposer.decompose(question)

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
        logger.info(
            f"Decomposition: {len(decomposition.sub_queries)} sub-queries, "
            f"type={decomposition.question_type.value}, {elapsed_ms}ms"
        )

        return decomposition, elapsed_ms

    async def _phase_research(
        self,
        decomposition: QueryDecomposition,
        question: str,
        *,
        enable_expansion: bool = True,
    ) -> tuple[list[SubAnswer], int]:
        """
        Phase 2: Research each sub-query in parallel.

        Uses semaphore-bounded concurrency to limit parallel LLM calls.

        Returns:
            Tuple of (sub_answers, time_ms)
        """
        start = time.perf_counter_ns()

        # Clear researcher cache for fresh query
        self.researcher.clear_cache()

        # Pre-compute query embedding for the main question
        with telemetry_stage("research_resolution"):
            question_embedding = await self.embeddings.embed_single(question)

        # Research each sub-query in parallel (with semaphore)
        async def research_with_semaphore(sub_query: SubQuery) -> SubAnswer:
            async with self._research_semaphore:
                return await self._research_sub_query(
                    sub_query,
                    question_embedding,
                    enable_expansion=enable_expansion,
                )

        tasks = [
            asyncio.create_task(research_with_semaphore(sq))
            for sq in decomposition.sub_queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling exceptions
        sub_answers: list[SubAnswer] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Sub-query research failed: {result}")
                sub_answers.append(SubAnswer(
                    sub_query=decomposition.sub_queries[i].query_text,
                    target_info=decomposition.sub_queries[i].target_info,
                    answer=f"Research failed: {result}",
                    confidence=0.0,
                    entities_found=[],
                ))
            else:
                # result is SubAnswer here after isinstance check
                sub_answers.append(result)

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
        logger.info(f"Research: {len(sub_answers)} sub-answers, {elapsed_ms}ms")

        return sub_answers, elapsed_ms

    async def _research_sub_query(
        self,
        sub_query: SubQuery,
        query_embedding: list[float],
        *,
        enable_expansion: bool = True,
    ) -> SubAnswer:
        """
        Research a single sub-query and synthesize its answer.

        Args:
            sub_query: The sub-query to research
            query_embedding: Pre-computed embedding for the main question
            enable_expansion: Whether to expand to neighbors

        Returns:
            SubAnswer with synthesized response
        """
        # Research: resolve hints and retrieve context
        with telemetry_stage("research_resolution"):
            (
                resolved_entities,
                resolved_topics,
                entity_chunks,
                neighbor_chunks,
                topic_chunks,
                global_chunks,
                facts,
                research_timing,
            ) = await self.researcher.research(
                sub_query,
                query_embedding=query_embedding,
                enable_expansion=enable_expansion,
                enable_global_search=self._enable_global_search,
            )

        # Build structured context
        context = self.context_builder.build(
            resolved_entities=resolved_entities,
            resolved_topics=resolved_topics,
            entity_chunks=entity_chunks,
            neighbor_chunks=neighbor_chunks,
            topic_chunks=topic_chunks,
            global_chunks=global_chunks,
            facts=facts,
        )

        # Synthesize sub-answer
        with telemetry_stage("synthesis_sub"):
            sub_answer = await self.synthesizer.synthesize_sub_answer(sub_query, context)

        # Add timing info
        sub_answer.timing = research_timing

        return sub_answer

    async def _phase_synthesis(
        self,
        question: str,
        sub_answers: list[SubAnswer],
        question_type: QuestionType,
    ) -> tuple[str, float, int]:
        """
        Phase 3: Synthesize final answer from sub-answers.

        Returns:
            Tuple of (answer, confidence, time_ms)
        """
        start = time.perf_counter_ns()

        with telemetry_stage("synthesis_final"):
            answer, confidence = await self.synthesizer.synthesize_final_answer(
                question, sub_answers, question_type
            )

        elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
        logger.info(f"Synthesis: confidence={confidence:.2f}, {elapsed_ms}ms")

        return answer, confidence, elapsed_ms

    def _extract_sources(self, sub_answers: list[SubAnswer]) -> list[dict[str, Any]]:
        """
        Extract source citations from sub-answers.

        Creates a deduplicated list of sources with relevance info.
        """
        sources: list[dict[str, Any]] = []
        seen_entities: set[str] = set()

        for sa in sub_answers:
            for entity in sa.entities_found:
                if entity not in seen_entities:
                    sources.append({
                        "entity": entity,
                        "sub_query": sa.sub_query,
                        "confidence": sa.confidence,
                    })
                    seen_entities.add(entity)

        return sources
