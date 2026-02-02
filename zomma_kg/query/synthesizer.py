"""
Synthesizer

Generates sub-answers from context and merges into final answer.

Synthesis Strategy:
    1. For each sub-query, generate an answer from its context
    2. Merge sub-answers according to question type
    3. Apply question-type-specific formatting

Question Type Formatting:
    - FACTUAL: Direct answer with evidence citations
    - COMPARISON: Side-by-side structure with parallel format
    - ENUMERATION: Bullet points, grouped logically
    - CAUSAL: Cause-effect with connective phrases
    - TEMPORAL: Chronological order with dates

See: docs/pipeline/QUERYING_SYSTEM.md Section 5 (Synthesis)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from zomma_kg.query.types import (
    FinalSynthesis,
    StructuredContext,
    SubAnswer,
    SubAnswerSynthesis,
)
from zomma_kg.types.results import QuestionType, SubQuery

if TYPE_CHECKING:
    from zomma_kg.config.settings import KGConfig
    from zomma_kg.providers.base import LLMProvider

logger = logging.getLogger(__name__)


# System prompt for sub-answer synthesis
SUB_ANSWER_SYSTEM_PROMPT = """You are synthesizing an answer from knowledge graph context.

PRINCIPLES:
1. Answer ONLY from the provided context - do not use external knowledge
2. Be specific and cite evidence from the context
3. If the context is insufficient, say so explicitly
4. Use clear, concise language
5. Include relevant details like dates, numbers, and entity names

When the context is empty or irrelevant, respond with:
"Insufficient information available in the knowledge base to answer this query."

OUTPUT:
- answer: Your synthesized answer
- confidence: 0.0-1.0 based on context quality/relevance
- entities_mentioned: List of entity names referenced in your answer"""


# System prompt for final synthesis
FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing a final answer from multiple sub-answers.

PRINCIPLES:
1. Combine sub-answers coherently without redundancy
2. Follow the formatting guidelines for the question type
3. Maintain source attribution where relevant
4. Provide a complete, standalone answer
5. Adjust confidence based on sub-answer quality

OUTPUT:
- answer: The final synthesized answer
- confidence: Overall confidence (0.0-1.0)"""


class Synthesizer:
    """
    Synthesizes answers from structured context.

    Handles both sub-query answers and final answer merging.
    """

    def __init__(
        self,
        llm_provider: "LLMProvider",
        config: "KGConfig | None" = None,
    ) -> None:
        """
        Initialize synthesizer.

        Args:
            llm_provider: LLM provider for generation
            config: Optional configuration
        """
        self.llm = llm_provider
        self._config: "KGConfig | None" = config

    async def synthesize_sub_answer(
        self,
        sub_query: SubQuery,
        context: StructuredContext,
    ) -> SubAnswer:
        """
        Synthesize an answer for a single sub-query.

        Args:
            sub_query: The sub-query to answer
            context: Structured context from retrieval

        Returns:
            SubAnswer with synthesized response
        """
        # Handle empty context
        if context.is_empty():
            return SubAnswer(
                sub_query=sub_query.query_text,
                target_info=sub_query.target_info,
                answer="Insufficient information available in the knowledge base to answer this query.",
                confidence=0.0,
                entities_found=[],
            )

        # Format context for prompt
        context_text = context.to_prompt_text()

        prompt = f"""Based on the following context from a knowledge graph, answer this query:

QUERY: {sub_query.query_text}
TARGET: {sub_query.target_info}

CONTEXT:
{context_text}

Synthesize a clear, evidence-based answer."""

        try:
            response = await self.llm.generate_structured(
                prompt,
                SubAnswerSynthesis,
                system=SUB_ANSWER_SYSTEM_PROMPT,
            )

            return SubAnswer(
                sub_query=sub_query.query_text,
                target_info=sub_query.target_info,
                answer=response.answer,
                confidence=response.confidence,
                entities_found=response.entities_mentioned,
            )

        except Exception as e:
            logger.warning(f"Sub-answer synthesis failed: {e}")
            # Fallback: extract key information from context
            fallback_answer = self._create_fallback_sub_answer(context)
            return SubAnswer(
                sub_query=sub_query.query_text,
                target_info=sub_query.target_info,
                answer=fallback_answer,
                confidence=0.3,
                entities_found=[e.resolved_name for e in context.resolved_entities],
            )

    async def synthesize_final_answer(
        self,
        question: str,
        sub_answers: list[SubAnswer],
        question_type: QuestionType,
    ) -> tuple[str, float]:
        """
        Synthesize final answer from sub-answers.

        Args:
            question: Original question
            sub_answers: Answers to each sub-query
            question_type: Question classification for formatting

        Returns:
            Tuple of (final_answer, confidence)
        """
        # Handle no valid sub-answers
        valid_answers = [sa for sa in sub_answers if sa.confidence > 0.0]
        if not valid_answers:
            return (
                "Unable to find sufficient information in the knowledge base to answer this question.",
                0.0,
            )

        # Handle single sub-answer
        if len(valid_answers) == 1:
            return valid_answers[0].answer, valid_answers[0].confidence

        # Get question-type-specific instructions
        type_instructions = self._get_question_type_instructions(question_type)

        # Format sub-answers for prompt
        sub_answer_text = "\n\n".join([
            f"### Sub-query: {sa.sub_query}\n**Target:** {sa.target_info}\n**Answer:** {sa.answer}\n**Confidence:** {sa.confidence:.2f}"
            for sa in valid_answers
        ])

        prompt = f"""Synthesize a final answer from these sub-answers:

ORIGINAL QUESTION: {question}

SUB-ANSWERS:
{sub_answer_text}

FORMATTING INSTRUCTIONS:
{type_instructions}

Combine these sub-answers into a coherent, complete answer."""

        try:
            response = await self.llm.generate_structured(
                prompt,
                FinalSynthesis,
                system=FINAL_SYNTHESIS_SYSTEM_PROMPT,
            )

            return response.answer, response.confidence

        except Exception as e:
            logger.warning(f"Final synthesis failed: {e}")
            return self._create_fallback_answer(valid_answers)

    def _get_question_type_instructions(self, question_type: QuestionType) -> str:
        """Get formatting instructions for question type."""
        instructions = {
            QuestionType.FACTUAL: """
FACTUAL QUESTION FORMAT:
- Lead with the direct answer
- Follow with supporting evidence
- Cite specific sources, dates, or entities
- Be concise but complete""",

            QuestionType.COMPARISON: """
COMPARISON QUESTION FORMAT:
- Use a side-by-side or parallel structure
- Address each entity being compared
- Highlight similarities AND differences
- Use consistent categories for comparison
- Consider using a structured format (e.g., "Entity A: ... Entity B: ...")""",

            QuestionType.ENUMERATION: """
ENUMERATION QUESTION FORMAT:
- Use bullet points or numbered lists
- Group items logically (by category, region, etc.)
- Include brief descriptions for each item
- Order by importance or relevance if applicable""",

            QuestionType.CAUSAL: """
CAUSAL QUESTION FORMAT:
- Clearly identify causes and effects
- Use connective phrases (because, therefore, as a result, led to)
- Explain the mechanism or reasoning
- Acknowledge uncertainty where appropriate""",

            QuestionType.TEMPORAL: """
TEMPORAL QUESTION FORMAT:
- Present information chronologically
- Include specific dates or time periods
- Highlight changes over time
- Use temporal connectives (before, after, during, subsequently)""",
        }

        return instructions.get(question_type, instructions[QuestionType.FACTUAL])

    def _create_fallback_sub_answer(self, context: StructuredContext) -> str:
        """Create a fallback answer from context when LLM fails."""
        parts: list[str] = []

        # Extract key facts
        if context.facts:
            fact_summary = "; ".join([
                f"{f.subject} {f.relationship_type.lower()} {f.object}"
                for f in context.facts[:3]
            ])
            parts.append(f"Key facts: {fact_summary}")

        # Extract from high-relevance chunks
        if context.high_relevance_chunks:
            # Take first chunk excerpt
            chunk = context.high_relevance_chunks[0]
            excerpt = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            parts.append(f"From {chunk.doc_name}: {excerpt}")

        if parts:
            return " ".join(parts)
        return "Context available but synthesis failed."

    def _create_fallback_answer(
        self, sub_answers: list[SubAnswer]
    ) -> tuple[str, float]:
        """Create a fallback final answer by concatenating sub-answers."""
        combined = "\n\n".join([
            f"**{sa.target_info}:** {sa.answer}"
            for sa in sub_answers
        ])

        # Average confidence
        avg_confidence = sum(sa.confidence for sa in sub_answers) / len(sub_answers)

        return combined, avg_confidence * 0.8  # Reduce confidence for fallback
