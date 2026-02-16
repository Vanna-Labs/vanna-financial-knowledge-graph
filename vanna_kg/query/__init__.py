"""
GraphRAG Query Pipeline

Query pipeline with question decomposition and multi-hop retrieval
over a knowledge graph.

Modules:
    pipeline: GraphRAG pipeline orchestrator
    decomposer: Question decomposition
    researcher: Per-subquery research
    context_builder: Context assembly and deduplication
    synthesizer: Answer synthesis
    types: Query-specific Pydantic models

Pipeline Phases:
    1. Decomposition: Break question into sub-queries with entity/topic hints
    2. Resolution: Wide-net entity/topic resolution (one hint -> many nodes)
    3. Retrieval: Entity chunks, topic chunks, 1-hop neighbors, facts, global search
    4. Assembly: Dedupe, filter by relevance, order context
    5. Synthesis: Question-type-aware answer generation

Question Types:
    - FACTUAL: Direct answer + evidence
    - COMPARISON: Side-by-side structure
    - ENUMERATION: Bulleted/numbered lists
    - TEMPORAL: Chronological organization
    - CAUSAL: Cause-effect relationships

Example:
    >>> from vanna_kg.query import GraphRAGPipeline
    >>> pipeline = GraphRAGPipeline(storage, llm, embeddings, config)
    >>> result = await pipeline.query("What companies did Apple acquire in 2024?")
    >>> print(result.answer)
    >>> print(f"Confidence: {result.confidence:.2f}")

See: docs/pipeline/QUERYING_SYSTEM.md
"""

from vanna_kg.query.context_builder import ContextBuilder
from vanna_kg.query.decomposer import QueryDecomposer
from vanna_kg.query.pipeline import GraphRAGPipeline
from vanna_kg.query.researcher import Researcher
from vanna_kg.query.synthesizer import Synthesizer
from vanna_kg.query.types import (
    EntityResolutionResponse,
    FinalSynthesis,
    PipelineResult,
    ResolvedEntity,
    ResolvedNode,
    ResolvedTopic,
    RetrievedChunk,
    RetrievedFact,
    StructuredContext,
    SubAnswer,
    SubAnswerSynthesis,
    TopicResolutionResponse,
)

__all__ = [
    # Main pipeline
    "GraphRAGPipeline",
    # Components
    "QueryDecomposer",
    "Researcher",
    "ContextBuilder",
    "Synthesizer",
    # Result types
    "PipelineResult",
    "SubAnswer",
    "StructuredContext",
    # Resolution types
    "ResolvedEntity",
    "ResolvedTopic",
    # Retrieval types
    "RetrievedChunk",
    "RetrievedFact",
    # LLM output schemas
    "ResolvedNode",
    "EntityResolutionResponse",
    "TopicResolutionResponse",
    "SubAnswerSynthesis",
    "FinalSynthesis",
]
