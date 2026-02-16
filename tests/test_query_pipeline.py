"""
Tests for GraphRAG Query Pipeline

Unit tests for each component and integration tests for the full pipeline.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vanna_kg.config.settings import KGConfig
from vanna_kg.query.context_builder import ContextBuilder
from vanna_kg.query.decomposer import QueryDecomposer
from vanna_kg.query.pipeline import GraphRAGPipeline
from vanna_kg.query.researcher import Researcher
from vanna_kg.query.synthesizer import Synthesizer
from vanna_kg.query.types import (
    PipelineResult,
    ResolvedEntity,
    RetrievedChunk,
    RetrievedFact,
    StructuredContext,
    SubAnswer,
)
from vanna_kg.types.results import (
    EntityHint,
    QueryDecomposition,
    QuestionType,
    SubQuery,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Generated text")
    llm.generate_structured = AsyncMock()
    llm.model_name = "test-model"
    return llm


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create a mock embedding provider."""
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[[0.1] * 3072])
    embeddings.embed_single = AsyncMock(return_value=[0.1] * 3072)
    embeddings.dimensions = 3072
    return embeddings


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage backend."""
    storage = MagicMock()
    storage.search_entities = AsyncMock(return_value=[])
    storage.search_topics = AsyncMock(return_value=[])
    storage.search_facts = AsyncMock(return_value=[])
    storage.get_entity_chunks = AsyncMock(return_value=[])
    storage.get_entity_facts = AsyncMock(return_value=[])
    storage.get_entity_neighbors = AsyncMock(return_value=[])
    storage.get_topic_chunks = AsyncMock(return_value=[])
    storage.get_topics_by_names = AsyncMock(return_value=[])
    # Avoid MagicMock path artifacts in logs when Researcher builds LanceDB paths.
    storage.kb_path = Path("./test_kb")
    storage.group_id = "test"
    return storage


@pytest.fixture
def config() -> KGConfig:
    """Create a test configuration."""
    return KGConfig(
        query_entity_threshold=0.3,
        query_topic_threshold=0.35,
        query_high_relevance_threshold=0.45,
        query_max_subqueries=5,
        query_max_entity_candidates=30,
        query_max_topic_candidates=20,
        query_enable_expansion=True,
        query_enable_global_search=True,
        query_max_high_relevance_chunks=30,
        query_max_facts=40,
        query_max_topic_chunks=15,
        query_max_low_relevance_chunks=20,
        query_global_search_limit=50,
        query_research_concurrency=5,
    )


@pytest.fixture
def sample_decomposition() -> QueryDecomposition:
    """Create a sample decomposition result."""
    return QueryDecomposition(
        required_info=["Apple's acquisitions in 2024"],
        sub_queries=[
            SubQuery(
                query_text="Apple acquisitions 2024",
                target_info="List of companies Apple acquired in 2024",
                entity_hints=["Apple"],
                topic_hints=["acquisitions", "M&A"],
            )
        ],
        entity_hints=[
            EntityHint(name="Apple", definition="Technology company")
        ],
        topic_hints=[
            EntityHint(name="M&A", definition="Mergers and acquisitions")
        ],
        relationship_hints=["acquired"],
        temporal_scope="2024",
        question_type=QuestionType.ENUMERATION,
        confidence=0.9,
        reasoning="Question asks for a list of acquisitions",
    )


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    """Create sample retrieved chunks."""
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            content="Apple acquired DarwinAI in 2024 for AI capabilities.",
            header_path="Acquisitions > 2024",
            doc_id="doc-1",
            doc_name="Apple News 2024",
            document_date="2024-03-01",
            vector_score=0.85,
            source="entity:Apple",
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            content="The acquisition strengthens Apple's AI team.",
            header_path="Acquisitions > Analysis",
            doc_id="doc-1",
            doc_name="Apple News 2024",
            document_date="2024-03-01",
            vector_score=0.72,
            source="entity:Apple",
        ),
        RetrievedChunk(
            chunk_id="chunk-3",
            content="Background on AI in tech industry.",
            header_path="Industry",
            doc_id="doc-2",
            doc_name="Tech Overview",
            vector_score=0.35,
            source="global",
        ),
    ]


@pytest.fixture
def sample_facts() -> list[RetrievedFact]:
    """Create sample retrieved facts."""
    return [
        RetrievedFact(
            fact_id="fact-1",
            content="Apple acquired DarwinAI in March 2024",
            subject="Apple",
            relationship_type="ACQUIRED",
            object="DarwinAI",
            chunk_id="chunk-1",
            vector_score=0.9,
        ),
    ]


# -----------------------------------------------------------------------------
# ContextBuilder Tests
# -----------------------------------------------------------------------------


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_build_empty_context(self) -> None:
        """Test building context with no data."""
        builder = ContextBuilder()
        context = builder.build(
            resolved_entities=[],
            resolved_topics=[],
            entity_chunks=[],
            neighbor_chunks=[],
            topic_chunks=[],
            global_chunks=[],
            facts=[],
        )

        assert context.is_empty()
        assert len(context.high_relevance_chunks) == 0
        assert len(context.facts) == 0

    def test_build_with_chunks(self, sample_chunks: list[RetrievedChunk]) -> None:
        """Test building context with chunks."""
        builder = ContextBuilder(high_relevance_threshold=0.45)
        context = builder.build(
            resolved_entities=[],
            resolved_topics=[],
            entity_chunks=sample_chunks[:2],
            neighbor_chunks=[],
            topic_chunks=[],
            global_chunks=[sample_chunks[2]],
            facts=[],
        )

        assert not context.is_empty()
        # First two chunks have score >= 0.45, third has 0.35
        assert len(context.high_relevance_chunks) == 2
        assert len(context.low_relevance_chunks) == 1

    def test_dedupe_chunks_by_id(self) -> None:
        """Test that duplicate chunks are deduplicated by ID."""
        builder = ContextBuilder()

        # Same chunk_id, different scores
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                content="Content A",
                vector_score=0.5,
                source="entity:X",
            ),
            RetrievedChunk(
                chunk_id="chunk-1",
                content="Content A",
                vector_score=0.8,  # Higher score
                source="global",
            ),
        ]

        deduped = builder._dedupe_chunks(chunks)
        assert len(deduped) == 1
        assert deduped[0].vector_score == 0.8  # Kept higher score

    def test_dedupe_facts_by_id(self, sample_facts: list[RetrievedFact]) -> None:
        """Test that duplicate facts are deduplicated by ID."""
        builder = ContextBuilder()

        facts = [
            sample_facts[0],
            RetrievedFact(
                fact_id=sample_facts[0].fact_id,
                content=sample_facts[0].content,
                subject="Apple",
                relationship_type="ACQUIRED",
                object="DarwinAI",
                vector_score=0.7,  # Lower score
            ),
        ]

        deduped = builder._dedupe_facts(facts)
        assert len(deduped) == 1
        assert deduped[0].vector_score == 0.9  # Kept original higher score

    def test_apply_limits(self) -> None:
        """Test that limits are applied to each category."""
        builder = ContextBuilder(
            max_high_relevance_chunks=2,
            max_facts=1,
        )

        chunks = [
            RetrievedChunk(
                chunk_id=f"chunk-{i}",
                content=f"Content {i}",
                vector_score=0.9 - i * 0.01,
            )
            for i in range(10)
        ]

        facts = [
            RetrievedFact(
                fact_id=f"fact-{i}",
                content=f"Fact {i}",
                subject="A",
                relationship_type="REL",
                object="B",
                vector_score=0.8,
            )
            for i in range(5)
        ]

        context = builder.build(
            resolved_entities=[],
            resolved_topics=[],
            entity_chunks=chunks,
            neighbor_chunks=[],
            topic_chunks=[],
            global_chunks=[],
            facts=facts,
        )

        assert len(context.high_relevance_chunks) <= 2
        assert len(context.facts) <= 1

    def test_to_prompt_text(
        self,
        sample_chunks: list[RetrievedChunk],
        sample_facts: list[RetrievedFact],
    ) -> None:
        """Test context formatting to prompt text."""
        context = StructuredContext(
            resolved_entities=[
                ResolvedEntity(
                    original_hint="Apple",
                    resolved_name="Apple Inc.",
                    resolved_uuid="uuid-1",
                    summary="Technology company",
                    entity_type="company",
                )
            ],
            high_relevance_chunks=sample_chunks[:1],
            facts=sample_facts,
        )

        text = context.to_prompt_text()
        assert "Apple Inc." in text
        assert "Technology company" in text
        assert "ACQUIRED" in text
        assert "Primary Evidence" in text


# -----------------------------------------------------------------------------
# QueryDecomposer Tests
# -----------------------------------------------------------------------------


class TestQueryDecomposer:
    """Tests for QueryDecomposer."""

    @pytest.mark.asyncio
    async def test_decompose_factual(
        self,
        mock_llm: MagicMock,
        sample_decomposition: QueryDecomposition,
    ) -> None:
        """Test decomposition of a factual question."""
        mock_llm.generate_structured = AsyncMock(return_value=sample_decomposition)

        decomposer = QueryDecomposer(mock_llm)
        result = await decomposer.decompose("What companies did Apple acquire in 2024?")

        assert result.question_type == QuestionType.ENUMERATION
        assert len(result.sub_queries) > 0
        assert len(result.entity_hints) > 0

    @pytest.mark.asyncio
    async def test_decompose_fallback(self, mock_llm: MagicMock) -> None:
        """Test fallback when LLM fails."""
        mock_llm.generate_structured = AsyncMock(side_effect=Exception("API Error"))

        decomposer = QueryDecomposer(mock_llm)
        result = await decomposer.decompose("What happened with Microsoft and Apple?")

        # Fallback should still produce a result
        assert result is not None
        assert result.confidence < 1.0  # Lower confidence for fallback
        assert len(result.sub_queries) > 0

    @pytest.mark.asyncio
    async def test_decompose_limits_subqueries(
        self,
        mock_llm: MagicMock,
        config: KGConfig,
    ) -> None:
        """Test that sub-queries are limited by config."""
        # Create decomposition with many sub-queries
        many_subqueries = QueryDecomposition(
            required_info=["info"] * 10,
            sub_queries=[
                SubQuery(
                    query_text=f"Query {i}",
                    target_info=f"Target {i}",
                )
                for i in range(10)
            ],
            entity_hints=[],
            topic_hints=[],
            question_type=QuestionType.FACTUAL,
        )
        mock_llm.generate_structured = AsyncMock(return_value=many_subqueries)

        decomposer = QueryDecomposer(mock_llm, config)
        result = await decomposer.decompose("Complex question", max_subqueries=3)

        assert len(result.sub_queries) <= 3

    def test_fallback_extracts_entities(self, mock_llm: MagicMock) -> None:
        """Test that fallback extracts capitalized words as entities."""
        decomposer = QueryDecomposer(mock_llm)
        result = decomposer._fallback_decomposition(
            "How does Microsoft compare to Apple in revenue?",
            "Test error",
        )

        entity_names = [h.name for h in result.entity_hints]
        assert "Microsoft" in entity_names or "Apple" in entity_names

    def test_fallback_detects_comparison(self, mock_llm: MagicMock) -> None:
        """Test that fallback detects comparison question type."""
        decomposer = QueryDecomposer(mock_llm)
        result = decomposer._fallback_decomposition(
            "Compare Microsoft and Apple",
            "Test error",
        )

        assert result.question_type == QuestionType.COMPARISON

    def test_fallback_detects_causal(self, mock_llm: MagicMock) -> None:
        """Test that fallback detects causal question type."""
        decomposer = QueryDecomposer(mock_llm)
        result = decomposer._fallback_decomposition(
            "Why did the stock price increase?",
            "Test error",
        )

        assert result.question_type == QuestionType.CAUSAL


# -----------------------------------------------------------------------------
# Researcher Tests
# -----------------------------------------------------------------------------


class TestResearcher:
    """Tests for Researcher."""

    @pytest.mark.asyncio
    async def test_ensure_ontology_index_uses_ontology_group(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
        tmp_path,
    ) -> None:
        """Ontology index should be initialized with group_id='ontology'."""
        mock_storage.kb_path = tmp_path

        with patch("vanna_kg.query.researcher.LanceDBIndices") as mock_indices_cls:
            mock_index = MagicMock()
            mock_index.initialize = AsyncMock()
            mock_indices_cls.return_value = mock_index

            researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
            index = await researcher._ensure_ontology_index()

        assert index is mock_index
        called_args = mock_indices_cls.call_args
        assert called_args is not None
        assert called_args.kwargs["group_id"] == "ontology"
        assert called_args.args[0] == tmp_path / "lancedb"

    @pytest.mark.asyncio
    async def test_resolve_topics_two_stage_ontology_then_kb_lookup(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Topic resolution should search ontology first, then map names in KB."""
        ontology_index = MagicMock()
        ontology_index.search_topics = AsyncMock(
            return_value=[
                (
                    {
                        "uuid": "ontology-1",
                        "name": "Inflation",
                        "definition": "General increase in prices.",
                        "group_id": "ontology",
                    },
                    0.93,
                ),
                (
                    {
                        "uuid": "ontology-2",
                        "name": "Inflation",
                        "definition": "Duplicate lower score.",
                        "group_id": "ontology",
                    },
                    0.72,
                ),
            ]
        )

        kb_topic = MagicMock()
        kb_topic.name = "Inflation"
        kb_topic.uuid = "kb-topic-1"
        kb_topic.definition = "Inflation in the ingested KB."
        mock_storage.get_topics_by_names = AsyncMock(return_value=[kb_topic])

        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        researcher._topic_llm_verification = False
        researcher._ensure_ontology_index = AsyncMock(return_value=ontology_index)

        result = await researcher.resolve_topics(["prices"], [0.1] * 3072)

        ontology_index.search_topics.assert_awaited_once()
        mock_storage.get_topics_by_names.assert_awaited_once_with(["Inflation"])
        assert len(result) == 1
        assert result[0].resolved_name == "Inflation"
        assert result[0].resolved_uuid == "kb-topic-1"

    @pytest.mark.asyncio
    async def test_resolve_topics_returns_empty_when_kb_lookup_empty(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Ontology matches should not resolve if KB default-group lookup has no rows."""
        ontology_index = MagicMock()
        ontology_index.search_topics = AsyncMock(
            return_value=[
                (
                    {
                        "uuid": "ontology-1",
                        "name": "Labor Market",
                        "definition": "Employment conditions.",
                        "group_id": "ontology",
                    },
                    0.89,
                ),
            ]
        )
        mock_storage.get_topics_by_names = AsyncMock(return_value=[])

        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        researcher._topic_llm_verification = False
        researcher._ensure_ontology_index = AsyncMock(return_value=ontology_index)

        result = await researcher.resolve_topics(["employment"], [0.1] * 3072)

        ontology_index.search_topics.assert_awaited_once()
        mock_storage.get_topics_by_names.assert_awaited_once_with(["Labor Market"])
        assert result == []

    @pytest.mark.asyncio
    async def test_research_caches_resolution(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that entity resolution is cached."""
        # Mock entity search to return a candidate
        mock_entity = MagicMock()
        mock_entity.name = "Apple Inc."
        mock_entity.uuid = "uuid-1"
        mock_entity.entity_type = "company"
        mock_entity.summary = "Tech company"
        mock_storage.search_entities = AsyncMock(return_value=[(mock_entity, 0.9)])

        # Mock LLM to return resolution
        from vanna_kg.query.types import EntityResolutionResponse, ResolvedNode
        mock_llm.generate_structured = AsyncMock(
            return_value=EntityResolutionResponse(
                resolved_entities=[ResolvedNode(name="Apple Inc.")]
            )
        )

        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        embedding = [0.1] * 3072

        # First resolution
        result1 = await researcher.resolve_entities(["Apple"], embedding)

        # Second resolution (should use cache)
        result2 = await researcher.resolve_entities(["Apple"], embedding)

        # LLM should only be called once
        assert mock_llm.generate_structured.call_count == 1
        assert len(result1) > 0
        assert len(result2) > 0

    @pytest.mark.asyncio
    async def test_research_handles_empty_hints(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that empty hints return empty results."""
        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        embedding = [0.1] * 3072

        entities = await researcher.resolve_entities([], embedding)
        topics = await researcher.resolve_topics([], embedding)

        assert entities == []
        assert topics == []

    @pytest.mark.asyncio
    async def test_clear_cache(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test cache clearing."""
        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)

        # Add something to cache
        researcher._entity_cache["test"] = [
            ResolvedEntity(
                original_hint="test",
                resolved_name="Test Entity",
                resolved_uuid="uuid-1",
            )
        ]

        researcher.clear_cache()

        assert len(researcher._entity_cache) == 0
        assert len(researcher._topic_cache) == 0

    @pytest.mark.asyncio
    async def test_get_entity_chunks_maps_chunk_id_and_doc_name(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test entity chunk mapping uses current retrieval keys."""
        mock_storage.get_entity_chunks = AsyncMock(return_value=[
            {
                "chunk_id": "chunk-123",
                "content": "Apple acquired DarwinAI.",
                "header_path": "Acquisitions",
                "document_uuid": "doc-1",
                "doc_name": "Apple Filing",
                "document_date": "2024-03-01",
            }
        ])

        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        chunks = await researcher._get_entity_chunks("Apple", [0.1] * 3072)

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chunk-123"
        assert chunks[0].doc_name == "Apple Filing"

    @pytest.mark.asyncio
    async def test_get_entity_chunks_supports_legacy_keys(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test entity chunk mapping remains backward-compatible with legacy keys."""
        mock_storage.get_entity_chunks = AsyncMock(return_value=[
            {
                "uuid": "legacy-chunk-1",
                "content": "Legacy chunk content.",
                "document_uuid": "doc-legacy",
                "document_name": "Legacy Doc",
            }
        ])

        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        chunks = await researcher._get_entity_chunks("Apple", [0.1] * 3072)

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "legacy-chunk-1"
        assert chunks[0].doc_name == "Legacy Doc"

    @pytest.mark.asyncio
    async def test_retrieved_chunk_ids_support_context_dedupe(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test retrieval maps real chunk IDs so context dedupe can work correctly."""
        mock_storage.get_entity_chunks = AsyncMock(return_value=[
            {
                "chunk_id": "shared-chunk-1",
                "content": "Apple acquired DarwinAI.",
                "document_uuid": "doc-1",
                "doc_name": "Apple News",
            }
        ])
        mock_storage.get_entity_neighbors = AsyncMock(return_value=[{"name": "Beats"}])

        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        entity_chunks = await researcher._get_entity_chunks("Apple", [0.1] * 3072)
        neighbor_chunks = await researcher._get_neighbor_chunks("Apple", [0.1] * 3072)

        context = ContextBuilder().build(
            resolved_entities=[],
            resolved_topics=[],
            entity_chunks=entity_chunks,
            neighbor_chunks=neighbor_chunks,
            topic_chunks=[],
            global_chunks=[],
            facts=[],
        )

        all_chunks = context.high_relevance_chunks + context.low_relevance_chunks
        assert len(all_chunks) == 1
        assert all_chunks[0].chunk_id == "shared-chunk-1"

    @pytest.mark.asyncio
    async def test_get_topic_chunks_maps_chunk_id_and_doc_name(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test topic chunk mapping uses current retrieval keys."""
        mock_storage.get_topic_chunks = AsyncMock(return_value=[
            {
                "chunk_id": "topic-chunk-1",
                "content": "M&A activity increased in 2024.",
                "document_uuid": "doc-topic",
                "doc_name": "Market Report",
            }
        ])

        researcher = Researcher(mock_storage, mock_llm, mock_embeddings)
        chunks = await researcher._get_topic_chunks("M&A")

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "topic-chunk-1"
        assert chunks[0].doc_name == "Market Report"


# -----------------------------------------------------------------------------
# Synthesizer Tests
# -----------------------------------------------------------------------------


class TestSynthesizer:
    """Tests for Synthesizer."""

    @pytest.mark.asyncio
    async def test_synthesize_sub_answer_empty_context(
        self, mock_llm: MagicMock
    ) -> None:
        """Test sub-answer synthesis with empty context."""
        synthesizer = Synthesizer(mock_llm)
        sub_query = SubQuery(
            query_text="Test query",
            target_info="Test target",
        )
        context = StructuredContext()

        result = await synthesizer.synthesize_sub_answer(sub_query, context)

        assert "Insufficient information" in result.answer
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_sub_answer_with_context(
        self,
        mock_llm: MagicMock,
        sample_chunks: list[RetrievedChunk],
    ) -> None:
        """Test sub-answer synthesis with context."""
        from vanna_kg.query.types import SubAnswerSynthesis
        mock_llm.generate_structured = AsyncMock(
            return_value=SubAnswerSynthesis(
                answer="Apple acquired DarwinAI in 2024.",
                confidence=0.85,
                entities_mentioned=["Apple", "DarwinAI"],
            )
        )

        synthesizer = Synthesizer(mock_llm)
        sub_query = SubQuery(
            query_text="Apple acquisitions 2024",
            target_info="List of acquisitions",
        )
        context = StructuredContext(high_relevance_chunks=sample_chunks[:1])

        result = await synthesizer.synthesize_sub_answer(sub_query, context)

        assert "DarwinAI" in result.answer
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_synthesize_final_no_valid_answers(
        self, mock_llm: MagicMock
    ) -> None:
        """Test final synthesis with no valid sub-answers."""
        synthesizer = Synthesizer(mock_llm)

        sub_answers = [
            SubAnswer(
                sub_query="Query 1",
                target_info="Target 1",
                answer="Failed",
                confidence=0.0,
            ),
        ]

        answer, confidence = await synthesizer.synthesize_final_answer(
            "Test question",
            sub_answers,
            QuestionType.FACTUAL,
        )

        assert "Unable to find" in answer
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_final_single_answer(
        self, mock_llm: MagicMock
    ) -> None:
        """Test final synthesis with single valid sub-answer."""
        synthesizer = Synthesizer(mock_llm)

        sub_answers = [
            SubAnswer(
                sub_query="Query 1",
                target_info="Target 1",
                answer="The answer is 42.",
                confidence=0.9,
            ),
        ]

        answer, confidence = await synthesizer.synthesize_final_answer(
            "Test question",
            sub_answers,
            QuestionType.FACTUAL,
        )

        # Single answer should be returned as-is
        assert answer == "The answer is 42."
        assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_synthesize_final_multiple_answers(
        self, mock_llm: MagicMock
    ) -> None:
        """Test final synthesis with multiple sub-answers."""
        from vanna_kg.query.types import FinalSynthesis
        mock_llm.generate_structured = AsyncMock(
            return_value=FinalSynthesis(
                answer="Combined answer from multiple sources.",
                confidence=0.85,
            )
        )

        synthesizer = Synthesizer(mock_llm)

        sub_answers = [
            SubAnswer(
                sub_query="Query 1",
                target_info="Target 1",
                answer="Answer 1",
                confidence=0.8,
            ),
            SubAnswer(
                sub_query="Query 2",
                target_info="Target 2",
                answer="Answer 2",
                confidence=0.9,
            ),
        ]

        answer, confidence = await synthesizer.synthesize_final_answer(
            "Test question",
            sub_answers,
            QuestionType.COMPARISON,
        )

        assert "Combined" in answer or mock_llm.generate_structured.called

    def test_question_type_instructions(self, mock_llm: MagicMock) -> None:
        """Test that question type instructions are provided."""
        synthesizer = Synthesizer(mock_llm)

        for qtype in QuestionType:
            instructions = synthesizer._get_question_type_instructions(qtype)
            assert len(instructions) > 0


# -----------------------------------------------------------------------------
# GraphRAGPipeline Integration Tests
# -----------------------------------------------------------------------------


class TestGraphRAGPipeline:
    """Integration tests for GraphRAGPipeline."""

    @pytest.mark.asyncio
    async def test_query_end_to_end(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
        config: KGConfig,
        sample_decomposition: QueryDecomposition,
    ) -> None:
        """Test full pipeline execution."""
        # Mock decomposition
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                sample_decomposition,  # Decomposition
                # Resolution responses (may be called)
                MagicMock(resolved_entities=[], no_match=True),
                MagicMock(resolved_topics=[], no_match=True),
                # Synthesis responses
                MagicMock(answer="Test answer", confidence=0.8, entities_mentioned=[]),
                MagicMock(answer="Final answer", confidence=0.85),
            ]
        )

        pipeline = GraphRAGPipeline(mock_storage, mock_llm, mock_embeddings, config)
        result = await pipeline.query("What companies did Apple acquire in 2024?")

        assert isinstance(result, PipelineResult)
        assert result.question == "What companies did Apple acquire in 2024?"
        assert "decomposition" in result.timing
        assert "research" in result.timing
        assert "synthesis" in result.timing

    @pytest.mark.asyncio
    async def test_query_handles_errors(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
        config: KGConfig,
    ) -> None:
        """Test that pipeline handles errors gracefully."""
        # First call for decomposition fails, triggering fallback
        mock_llm.generate_structured = AsyncMock(
            side_effect=Exception("LLM Error")
        )

        pipeline = GraphRAGPipeline(mock_storage, mock_llm, mock_embeddings, config)

        # Should not raise, should use fallback
        result = await pipeline.query("Test question")

        assert isinstance(result, PipelineResult)
        # Should still have timing info
        assert "decomposition" in result.timing

    @pytest.mark.asyncio
    async def test_query_with_expansion_disabled(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
        config: KGConfig,
        sample_decomposition: QueryDecomposition,
    ) -> None:
        """Test query with neighbor expansion disabled."""
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                sample_decomposition,
                MagicMock(answer="Answer", confidence=0.8, entities_mentioned=[]),
            ]
        )

        pipeline = GraphRAGPipeline(mock_storage, mock_llm, mock_embeddings, config)
        result = await pipeline.query(
            "Test question",
            enable_expansion=False,
        )

        assert isinstance(result, PipelineResult)
        # Neighbor retrieval should not be called
        mock_storage.get_entity_neighbors.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_includes_sources(
        self,
        mock_storage: MagicMock,
        mock_llm: MagicMock,
        mock_embeddings: MagicMock,
        config: KGConfig,
        sample_decomposition: QueryDecomposition,
    ) -> None:
        """Test that sources are included when requested."""
        from vanna_kg.query.types import SubAnswerSynthesis
        mock_llm.generate_structured = AsyncMock(
            side_effect=[
                sample_decomposition,
                SubAnswerSynthesis(
                    answer="Test",
                    confidence=0.8,
                    entities_mentioned=["Apple", "DarwinAI"],
                ),
            ]
        )

        pipeline = GraphRAGPipeline(mock_storage, mock_llm, mock_embeddings, config)
        result = await pipeline.query("Test", include_sources=True)

        # Sources should be populated from entities_mentioned
        assert isinstance(result.sources, list)


# -----------------------------------------------------------------------------
# Test Helpers
# -----------------------------------------------------------------------------


class TestStructuredContext:
    """Tests for StructuredContext."""

    def test_is_empty_true(self) -> None:
        """Test is_empty returns True for empty context."""
        context = StructuredContext()
        assert context.is_empty()

    def test_is_empty_false_with_entities(self) -> None:
        """Test is_empty returns False when entities exist."""
        context = StructuredContext(
            resolved_entities=[
                ResolvedEntity(
                    original_hint="test",
                    resolved_name="Test",
                    resolved_uuid="uuid",
                )
            ]
        )
        assert not context.is_empty()

    def test_is_empty_false_with_chunks(self) -> None:
        """Test is_empty returns False when chunks exist."""
        context = StructuredContext(
            high_relevance_chunks=[
                RetrievedChunk(chunk_id="1", content="test")
            ]
        )
        assert not context.is_empty()

    def test_to_prompt_text_empty(self) -> None:
        """Test prompt text for empty context."""
        context = StructuredContext()
        text = context.to_prompt_text()
        assert "No context available" in text


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_total_time_ms(self) -> None:
        """Test total time calculation."""
        result = PipelineResult(
            question="Test",
            answer="Answer",
            timing={
                "decomposition": 100,
                "research": 500,
                "synthesis": 200,
            },
        )

        assert result.total_time_ms == 800

    def test_total_time_ms_empty(self) -> None:
        """Test total time with no timing."""
        result = PipelineResult(question="Test", answer="Answer")
        assert result.total_time_ms == 0
