"""
Query Decomposer

Breaks complex questions into focused sub-queries using chain-of-thought reasoning.

Decomposition Steps:
    1. Identify distinct pieces of information needed
    2. Extract entity hints (people, companies, etc.)
    3. Extract topic hints (themes, concepts)
    4. Identify relationship phrases with modifiers
    5. Detect temporal scope
    6. Classify question type (FACTUAL, COMPARISON, etc.)
    7. Generate targeted sub-queries

For comparison questions, generates combinatorial sub-queries (one per entity).

See: docs/pipeline/QUERYING_SYSTEM.md Section 1 (Decomposition)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from zomma_kg.types.results import (
    EntityHint,
    QueryDecomposition,
    QuestionType,
    SubQuery,
)

if TYPE_CHECKING:
    from zomma_kg.config.settings import KGConfig
    from zomma_kg.providers.base import LLMProvider

logger = logging.getLogger(__name__)


# System prompt for query decomposition
DECOMPOSITION_SYSTEM_PROMPT = """You are a query analysis expert for a knowledge graph system.

Your task is to decompose user questions into structured components that can be used for targeted retrieval from a knowledge graph.

IMPORTANT PRINCIPLES:
1. Be comprehensive - extract ALL entities, topics, and relationships mentioned
2. For COMPARISON questions, generate separate sub-queries for EACH entity being compared
3. Preserve the user's intent while making it searchable
4. Include contextual definitions to help with fuzzy matching
5. Keep sub-query texts focused and searchable (not full sentences)

QUESTION TYPES:
- FACTUAL: Simple fact lookup ("What happened in Boston?")
- COMPARISON: Compare entities ("How do Apple and Microsoft differ in revenue?")
- CAUSAL: Cause/effect relationships ("Why did wages increase?")
- TEMPORAL: Time-based queries ("What changed from October to November?")
- ENUMERATION: List items ("Which companies saw growth?")

OUTPUT REQUIREMENTS:
- required_info: List distinct pieces of information needed (each should be atomic)
- sub_queries: Focused search queries with entity/topic hints
- entity_hints: ALL entities with brief contextual definitions
- topic_hints: ALL topics/themes with definitions
- relationship_hints: Key relationship phrases with modifiers (e.g., "slight growth", "acquired")
- temporal_scope: Time period if specified (e.g., "October 2025")
- question_type: Classification for routing
- reasoning: Brief chain-of-thought explanation"""


def _create_decomposition_prompt(question: str) -> str:
    """Create the user prompt for decomposition."""
    return f"""Analyze and decompose this question for knowledge graph retrieval:

QUESTION: {question}

Think step by step:
1. What distinct pieces of information are needed to answer this?
2. What entities (companies, people, organizations, locations, products) are mentioned?
3. What topics or themes (economic trends, M&A, technology, etc.) are relevant?
4. What relationships or actions are implied (acquired, grew, reported, etc.)?
5. Is there a specific time period mentioned?
6. What type of question is this (factual, comparison, causal, temporal, enumeration)?
7. What targeted sub-queries would find this information?

For COMPARISON questions: Create one sub-query PER entity being compared.

Provide your structured analysis."""


class QueryDecomposer:
    """
    Decomposes complex questions into structured sub-queries.

    Uses chain-of-thought LLM reasoning with fallback to keyword extraction.
    """

    def __init__(
        self,
        llm_provider: "LLMProvider",
        config: "KGConfig | None" = None,
    ) -> None:
        """
        Initialize decomposer.

        Args:
            llm_provider: LLM provider for structured generation
            config: Optional configuration (uses defaults if not provided)
        """
        self.llm = llm_provider
        self._max_subqueries = config.query_max_subqueries if config else 5

    async def decompose(
        self,
        question: str,
        *,
        max_subqueries: int | None = None,
    ) -> QueryDecomposition:
        """
        Decompose a question into structured components.

        Uses LLM chain-of-thought with fallback to keyword extraction on error.

        Args:
            question: The user's question
            max_subqueries: Override for maximum sub-queries

        Returns:
            QueryDecomposition with sub-queries, hints, and classification
        """
        max_sq = max_subqueries or self._max_subqueries

        try:
            # Use LLM for structured decomposition
            decomposition = await self.llm.generate_structured(
                _create_decomposition_prompt(question),
                QueryDecomposition,
                system=DECOMPOSITION_SYSTEM_PROMPT,
            )

            # Enforce sub-query limit
            if len(decomposition.sub_queries) > max_sq:
                decomposition.sub_queries = decomposition.sub_queries[:max_sq]

            # Ensure at least one sub-query exists
            if not decomposition.sub_queries:
                decomposition.sub_queries = [
                    SubQuery(
                        query_text=question,
                        target_info="Answer the question directly",
                        entity_hints=[h.name for h in decomposition.entity_hints],
                        topic_hints=[h.name for h in decomposition.topic_hints],
                    )
                ]

            logger.debug(
                f"Decomposed '{question[:50]}...' into {len(decomposition.sub_queries)} sub-queries, "
                f"type={decomposition.question_type}"
            )

            return decomposition

        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}, using fallback")
            return self._fallback_decomposition(question, str(e))

    def _fallback_decomposition(
        self, question: str, error: str
    ) -> QueryDecomposition:
        """
        Fallback decomposition using keyword extraction.

        Extracts capitalized words as entities and uses simple heuristics
        for question type detection.

        Args:
            question: The original question
            error: The error that triggered fallback

        Returns:
            Basic QueryDecomposition based on pattern matching
        """
        # Extract capitalized words (likely entities)
        # Match words that start with uppercase, excluding sentence starts
        words = question.split()
        entities: list[str] = []

        for i, word in enumerate(words):
            # Skip first word and common words
            clean_word = re.sub(r"[^\w\s]", "", word)
            if (
                i > 0
                and clean_word
                and clean_word[0].isupper()
                and clean_word.lower() not in {"the", "a", "an", "is", "are", "was", "were"}
            ):
                entities.append(clean_word)

        # Detect question type from keywords
        question_lower = question.lower()
        if any(kw in question_lower for kw in ["compare", "differ", "versus", " vs ", "between"]):
            question_type = QuestionType.COMPARISON
        elif any(kw in question_lower for kw in ["why", "because", "cause", "reason", "effect"]):
            question_type = QuestionType.CAUSAL
        elif any(kw in question_lower for kw in ["when", "before", "after", "during", "changed"]):
            question_type = QuestionType.TEMPORAL
        elif any(kw in question_lower for kw in ["list", "which", "what are", "enumerate"]):
            question_type = QuestionType.ENUMERATION
        else:
            question_type = QuestionType.FACTUAL

        # Create entity hints with empty definitions (will rely on vector search)
        entity_hints = [
            EntityHint(name=e, definition="(from question)")
            for e in entities
        ]

        # Create a single sub-query with all entities
        sub_queries = [
            SubQuery(
                query_text=question,
                target_info="Answer the question",
                entity_hints=entities,
                topic_hints=[],
            )
        ]

        return QueryDecomposition(
            required_info=["Answer the user's question"],
            sub_queries=sub_queries,
            entity_hints=entity_hints,
            topic_hints=[],
            relationship_hints=[],
            temporal_scope=None,
            question_type=question_type,
            confidence=0.5,  # Lower confidence for fallback
            reasoning=f"Fallback decomposition due to: {error}",
        )
