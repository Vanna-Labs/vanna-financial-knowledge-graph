"""
Tests for KnowledgeGraph facade class.

Tests cover:
- Instantiation and lazy initialization
- Stats method
- Context manager support
- CLI commands
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vanna_kg.api.knowledge_graph import KnowledgeGraph


class TestKnowledgeGraphInstantiation:
    """Tests for KnowledgeGraph instantiation."""

    def test_instantiation_with_path_string(self):
        """Test instantiation with string path."""
        kg = KnowledgeGraph("./test_kb")
        assert kg.path == Path("./test_kb").resolve()
        assert kg.is_initialized is False

    def test_instantiation_with_path_object(self):
        """Test instantiation with Path object."""
        kg = KnowledgeGraph(Path("./test_kb"))
        assert kg.path == Path("./test_kb").resolve()

    def test_instantiation_with_custom_config(self):
        """Test instantiation with custom config."""
        from vanna_kg.config import KGConfig

        config = KGConfig(llm_provider="openai", llm_model="gpt-4o")
        kg = KnowledgeGraph("./test_kb", config=config)
        assert kg.config.llm_model == "gpt-4o"

    def test_instantiation_default_config(self):
        """Test instantiation uses default config."""
        kg = KnowledgeGraph("./test_kb")
        assert kg.config is not None
        assert kg.config.llm_provider == "openai"

    def test_instantiation_does_not_initialize(self):
        """Test that instantiation doesn't trigger initialization."""
        kg = KnowledgeGraph("./test_kb")
        assert kg.is_initialized is False
        assert kg._storage is None
        assert kg._llm is None
        assert kg._embeddings is None


class TestKnowledgeGraphLazyInit:
    """Tests for lazy initialization."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_creates_directory(self):
        """Test that _ensure_initialized creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "new_kb"
            kg = KnowledgeGraph(kb_path)

            assert not kb_path.exists()

            # Mock the storage and providers to avoid real initialization
            mock_storage_instance = AsyncMock()
            mock_storage_instance.initialize = AsyncMock()

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage_instance), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()):

                await kg._ensure_initialized()

                assert kb_path.exists()
                assert kg.is_initialized is True

    @pytest.mark.asyncio
    async def test_ensure_initialized_raises_if_not_exists_and_create_false(self):
        """Test that _ensure_initialized raises if path doesn't exist and create=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "nonexistent_kb"
            kg = KnowledgeGraph(kb_path, create=False)

            with pytest.raises(FileNotFoundError, match="Knowledge base not found"):
                await kg._ensure_initialized()

    @pytest.mark.asyncio
    async def test_ensure_initialized_idempotent(self):
        """Test that _ensure_initialized is idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage_instance = AsyncMock()
            mock_storage_instance.initialize = AsyncMock()

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage_instance) as mock_storage, \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()):

                await kg._ensure_initialized()
                await kg._ensure_initialized()  # Second call should be no-op

                # Storage should only be created once
                assert mock_storage.call_count == 1


class TestKnowledgeGraphProviderFactory:
    """Tests for provider factory methods."""

    def test_create_llm_provider_openai(self):
        """Test OpenAI LLM provider creation."""
        from vanna_kg.config import KGConfig

        config = KGConfig(llm_provider="openai", openai_api_key="test-key")
        kg = KnowledgeGraph("./test_kb", config=config)

        with patch('vanna_kg.providers.llm.openai.OpenAILLMProvider') as mock_provider:
            mock_provider.return_value = MagicMock()
            provider = kg._create_llm_provider()
            mock_provider.assert_called_once_with(
                api_key="test-key",
                model=config.llm_model,
            )

    def test_create_llm_provider_unknown_raises(self):
        """Test that unknown LLM provider raises ValueError."""
        from vanna_kg.config import KGConfig

        config = KGConfig(llm_provider="unknown")
        kg = KnowledgeGraph("./test_kb", config=config)

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            kg._create_llm_provider()

    def test_create_embedding_provider_openai(self):
        """Test OpenAI embedding provider creation."""
        from vanna_kg.config import KGConfig

        config = KGConfig(embedding_provider="openai", openai_api_key="test-key")
        kg = KnowledgeGraph("./test_kb", config=config)

        with patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider') as mock_provider:
            mock_provider.return_value = MagicMock()
            provider = kg._create_embedding_provider()
            mock_provider.assert_called_once()

    def test_create_embedding_provider_unknown_raises(self):
        """Test that unknown embedding provider raises ValueError."""
        from vanna_kg.config import KGConfig

        config = KGConfig(embedding_provider="unknown")
        kg = KnowledgeGraph("./test_kb", config=config)

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            kg._create_embedding_provider()


class TestKnowledgeGraphStats:
    """Tests for stats method."""

    @pytest.mark.asyncio
    async def test_stats_returns_counts(self):
        """Test that stats returns correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            # Mock storage with count methods
            mock_storage = AsyncMock()
            mock_storage.count_documents = AsyncMock(return_value=5)
            mock_storage.count_entities = AsyncMock(return_value=100)
            mock_storage.count_chunks = AsyncMock(return_value=50)
            mock_storage.count_facts = AsyncMock(return_value=200)
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()):

                stats = await kg.stats()

                assert stats == {
                    "documents": 5,
                    "entities": 100,
                    "chunks": 50,
                    "facts": 200,
                }

                await kg.close()

    def test_stats_sync_wrapper(self):
        """Test sync wrapper for stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.count_documents = AsyncMock(return_value=1)
            mock_storage.count_entities = AsyncMock(return_value=2)
            mock_storage.count_chunks = AsyncMock(return_value=3)
            mock_storage.count_facts = AsyncMock(return_value=4)
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()):

                stats = kg.stats_sync()

                assert stats["documents"] == 1


class TestKnowledgeGraphLifecycle:
    """Tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_releases_resources(self):
        """Test that close releases all resources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()):

                await kg._ensure_initialized()
                assert kg.is_initialized is True

                await kg.close()

                assert kg.is_initialized is False
                assert kg._storage is None
                assert kg._llm is None
                assert kg._embeddings is None
                mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager support."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()):

                async with KnowledgeGraph(tmpdir) as kg:
                    assert kg.is_initialized is True

                # After exiting context, resources should be released
                mock_storage.close.assert_called()


class TestKnowledgeGraphQuery:
    """Tests for query methods."""

    @pytest.mark.asyncio
    async def test_query_uses_pipeline(self):
        """Test that query uses GraphRAGPipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=MagicMock(
                answer="Test answer",
                confidence=0.9,
                sources=[],
                sub_answers=[],
                question_type=None,
                timing={"total": 100},
            ))

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()), \
                 patch('vanna_kg.query.GraphRAGPipeline', return_value=mock_pipeline):

                result = await kg.query("What is the answer?")

                assert result.answer == "Test answer"
                assert result.confidence == 0.9
                mock_pipeline.query.assert_called_once_with(
                    "What is the answer?",
                    include_sources=True,
                )

                await kg.close()

    @pytest.mark.asyncio
    async def test_query_caches_pipeline(self):
        """Test that query caches the pipeline instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=MagicMock(
                answer="Answer",
                confidence=0.8,
                sources=[],
                sub_answers=[],
                question_type=None,
                timing={},
            ))

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()), \
                 patch('vanna_kg.query.GraphRAGPipeline', return_value=mock_pipeline) as mock_cls:

                await kg.query("Question 1")
                await kg.query("Question 2")

                # Pipeline should only be created once
                assert mock_cls.call_count == 1

                await kg.close()

    @pytest.mark.asyncio
    async def test_query_maps_sub_answers_and_question_type(self):
        """Test query result mapping for sub-answer fields and question type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(return_value=MagicMock(
                answer="Final answer",
                confidence=0.87,
                sources=[{"entity": "Apple"}],
                sub_answers=[
                    MagicMock(
                        sub_query="Apple acquisitions in 2024",
                        answer="Apple acquired DarwinAI in 2024.",
                        confidence=0.8,
                    )
                ],
                question_type="enumeration",
                timing={"total": 42},
            ))

            with patch('vanna_kg.storage.parquet.backend.ParquetBackend', return_value=mock_storage), \
                 patch('vanna_kg.providers.llm.openai.OpenAILLMProvider', return_value=MagicMock()), \
                 patch('vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider', return_value=MagicMock()), \
                 patch('vanna_kg.query.GraphRAGPipeline', return_value=mock_pipeline):

                result = await kg.query("What did Apple acquire in 2024?")

                assert result.sub_answers == [
                    {
                        "query": "Apple acquisitions in 2024",
                        "answer": "Apple acquired DarwinAI in 2024.",
                        "confidence": 0.8,
                    }
                ]
                assert result.question_type == "enumeration"

                await kg.close()

    @pytest.mark.asyncio
    async def test_query_cost_debug_attaches_breakdown(self):
        """Query should include cost_debug report when enabled."""
        from vanna_kg.types.results import CostUsageRecord
        from vanna_kg.utils.cost_telemetry import record_usage

        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()

            async def query_with_usage(*args, **kwargs):
                record_usage(
                    CostUsageRecord(
                        provider="openai",
                        model="gpt-5-mini",
                        operation="generate_structured",
                        stage="decomposition",
                        input_tokens=120,
                        output_tokens=30,
                        total_tokens=150,
                        estimated_cost_usd=0.001,
                        latency_ms=25,
                        estimated=False,
                    )
                )
                return MagicMock(
                    answer="Test answer",
                    confidence=0.9,
                    sources=[],
                    sub_answers=[],
                    question_type=None,
                    timing={"total": 100},
                )

            mock_pipeline = AsyncMock()
            mock_pipeline.query = AsyncMock(side_effect=query_with_usage)

            with patch(
                "vanna_kg.storage.parquet.backend.ParquetBackend", return_value=mock_storage
            ), patch(
                "vanna_kg.providers.llm.openai.OpenAILLMProvider", return_value=MagicMock()
            ), patch(
                "vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider",
                return_value=MagicMock(),
            ), patch("vanna_kg.query.GraphRAGPipeline", return_value=mock_pipeline):
                result = await kg.query("What is the answer?", cost_debug=True)

            assert result.cost_debug is not None
            assert result.cost_debug.breakdown.total_calls == 1
            assert result.cost_debug.breakdown.total_tokens == 150
            assert result.cost_debug.breakdown.by_stage[0].stage == "decomposition"

            await kg.close()

    @pytest.mark.asyncio
    async def test_search_chunks_returns_chunk_matches(self):
        """Test search_chunks returns scored chunk matches."""
        from vanna_kg.types.chunks import Chunk

        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_storage.search_chunks = AsyncMock(return_value=[
                (
                    Chunk(
                        uuid="chunk-1",
                        content="Apple reported stronger iPhone demand.",
                        header_path="Earnings",
                        position=0,
                        document_uuid="doc-1",
                    ),
                    0.91,
                )
            ])

            mock_embeddings = MagicMock()
            mock_embeddings.embed_single = AsyncMock(return_value=[0.1] * 3072)

            with patch(
                "vanna_kg.storage.parquet.backend.ParquetBackend", return_value=mock_storage
            ), patch(
                "vanna_kg.providers.llm.openai.OpenAILLMProvider", return_value=MagicMock()
            ), patch(
                "vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider",
                return_value=mock_embeddings,
            ):
                matches = await kg.search_chunks("iphone demand", limit=5, threshold=0.4)

            assert len(matches) == 1
            assert matches[0].chunk.uuid == "chunk-1"
            assert matches[0].score == 0.91
            mock_embeddings.embed_single.assert_awaited_once_with("iphone demand")
            mock_storage.search_chunks.assert_awaited_once_with([0.1] * 3072, limit=5, threshold=0.4)

    def test_search_chunks_sync_wrapper(self):
        """Test sync wrapper for search_chunks."""
        from vanna_kg.types.chunks import Chunk

        with tempfile.TemporaryDirectory() as tmpdir:
            kg = KnowledgeGraph(tmpdir)

            mock_storage = AsyncMock()
            mock_storage.initialize = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_storage.search_chunks = AsyncMock(return_value=[
                (
                    Chunk(
                        uuid="chunk-sync",
                        content="Labor market remained tight.",
                        header_path="Labor",
                        position=0,
                        document_uuid="doc-1",
                    ),
                    0.83,
                )
            ])

            mock_embeddings = MagicMock()
            mock_embeddings.embed_single = AsyncMock(return_value=[0.2] * 3072)

            with patch(
                "vanna_kg.storage.parquet.backend.ParquetBackend", return_value=mock_storage
            ), patch(
                "vanna_kg.providers.llm.openai.OpenAILLMProvider", return_value=MagicMock()
            ), patch(
                "vanna_kg.providers.embedding.openai.OpenAIEmbeddingProvider",
                return_value=mock_embeddings,
            ):
                matches = kg.search_chunks_sync("labor")

            assert len(matches) == 1
            assert matches[0].chunk.uuid == "chunk-sync"
            assert matches[0].score == 0.83


class TestKnowledgeGraphIngestionSchemaAlignment:
    """Tests for ingestion model construction field alignment."""

    @pytest.mark.asyncio
    async def test_ingest_pdf_and_markdown_share_pipeline_payload_shape(self):
        """ingest_pdf and ingest_markdown should build equivalent shared-stage payloads."""
        from vanna_kg.types import (
            AssemblyResult,
            CanonicalEntity,
            ChainOfThoughtResult,
            ChunkInput,
            EntityDeduplicationOutput,
            EntityResolutionResult,
            EnumeratedEntity,
            ExtractedFact,
        )
        from vanna_kg.types.topics import TopicResolution, TopicResolutionResult

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            md_path = kb_path / "doc.md"
            pdf_path = kb_path / "doc.pdf"
            md_path.write_text("# Title\n\nApple acquired Beats.", encoding="utf-8")
            pdf_path.write_bytes(b"%PDF-1.7")

            kg = KnowledgeGraph(kb_path)
            kg._initialized = True
            kg._storage = MagicMock()
            kg._llm = MagicMock()
            kg._embeddings = MagicMock()

            chunk_inputs = [
                ChunkInput(
                    doc_id="doc-id",
                    content="Apple acquired Beats.",
                    header_path="Title",
                    position=0,
                )
            ]

            extraction_results = [
                ChainOfThoughtResult(
                    entities=[
                        EnumeratedEntity(
                            name="Apple",
                            entity_type="Company",
                            summary="Tech company",
                        ),
                        EnumeratedEntity(
                            name="Beats",
                            entity_type="Company",
                            summary="Audio brand",
                        ),
                    ],
                    facts=[
                        ExtractedFact(
                            fact="Apple acquired Beats.",
                            subject="Apple",
                            subject_type="Company",
                            object="Beats",
                            object_type="Company",
                            relationship="acquired",
                            date_context="Document date: 2014-05-28",
                            topics=["M&A"],
                        )
                    ],
                )
            ]

            canonical_entities = [
                CanonicalEntity(
                    uuid="entity-apple",
                    name="Apple",
                    entity_type="Company",
                    summary="Apple summary",
                    source_indices=[0],
                ),
                CanonicalEntity(
                    uuid="entity-beats",
                    name="Beats",
                    entity_type="Company",
                    summary="Beats summary",
                    source_indices=[1],
                ),
            ]

            dedup_result = EntityDeduplicationOutput(
                canonical_entities=canonical_entities,
                index_to_canonical={0: 0, 1: 1},
                merge_history=[],
                canonical_entity_embeddings=[
                    [0.11] * 3072,
                    [0.22] * 3072,
                ],
            )
            entity_resolution = EntityResolutionResult(
                new_entities=canonical_entities,
                uuid_remap={},
                summary_updates={},
            )
            topic_resolution = TopicResolutionResult(
                resolved_topics=[
                    TopicResolution(
                        uuid="topic-ma",
                        canonical_label="M&A",
                        is_new=False,
                        definition="Mergers and acquisitions",
                    )
                ],
                uuid_remap={"M&A": "topic-ma"},
                new_topics=[],
            )

            mock_entity_registry_markdown = MagicMock()
            mock_entity_registry_markdown.resolve = AsyncMock(return_value=entity_resolution)
            mock_entity_registry_pdf = MagicMock()
            mock_entity_registry_pdf.resolve = AsyncMock(return_value=entity_resolution)

            mock_topic_resolver_markdown = MagicMock()
            mock_topic_resolver_markdown.resolve = AsyncMock(return_value=topic_resolution)
            mock_topic_resolver_pdf = MagicMock()
            mock_topic_resolver_pdf.resolve = AsyncMock(return_value=topic_resolution)

            mock_assembler_markdown = MagicMock()
            mock_assembler_markdown.assemble = AsyncMock(
                return_value=AssemblyResult(
                    document_written=True,
                    chunks_written=1,
                    entities_written=2,
                    facts_written=1,
                    topics_written=1,
                    relationships_written=1,
                )
            )
            mock_assembler_pdf = MagicMock()
            mock_assembler_pdf.assemble = AsyncMock(
                return_value=AssemblyResult(
                    document_written=True,
                    chunks_written=1,
                    entities_written=2,
                    facts_written=1,
                    topics_written=1,
                    relationships_written=1,
                )
            )

            markdown_progress: list[tuple[str, float]] = []
            pdf_progress: list[tuple[str, float]] = []

            with patch(
                "vanna_kg.ingestion.chunking.chunk_markdown", return_value=chunk_inputs
            ), patch(
                "vanna_kg.ingestion.chunking.chunk_pdf", AsyncMock(return_value=chunk_inputs)
            ), patch(
                "vanna_kg.ingestion.extraction.extract_from_chunks",
                AsyncMock(side_effect=[extraction_results, extraction_results]),
            ), patch(
                "vanna_kg.ingestion.resolution.deduplicate_entities",
                AsyncMock(side_effect=[dedup_result, dedup_result]),
            ), patch(
                "vanna_kg.ingestion.resolution.EntityRegistry",
                side_effect=[mock_entity_registry_markdown, mock_entity_registry_pdf],
            ), patch(
                "vanna_kg.ingestion.resolution.TopicResolver",
                side_effect=[mock_topic_resolver_markdown, mock_topic_resolver_pdf],
            ), patch(
                "vanna_kg.ingestion.assembly.Assembler",
                side_effect=[mock_assembler_markdown, mock_assembler_pdf],
            ), patch.object(
                KnowledgeGraph, "_create_ontology_index", AsyncMock(return_value=MagicMock())
            ):
                markdown_result = await kg.ingest_markdown(
                    md_path,
                    document_date="2014-05-28",
                    on_progress=lambda stage, value: markdown_progress.append((stage, value)),
                )
                pdf_result = await kg.ingest_pdf(
                    pdf_path,
                    document_date="2014-05-28",
                    on_progress=lambda stage, value: pdf_progress.append((stage, value)),
                )

            assert markdown_result.chunks == 1
            assert pdf_result.chunks == 1
            assert markdown_result.entities == 2
            assert pdf_result.entities == 2
            assert markdown_result.facts == 1
            assert pdf_result.facts == 1
            assert markdown_result.topics == 1
            assert pdf_result.topics == 1

            markdown_topic_definitions = mock_topic_resolver_markdown.resolve.await_args.args[0]
            pdf_topic_definitions = mock_topic_resolver_pdf.resolve.await_args.args[0]
            assert [td.topic for td in markdown_topic_definitions] == [td.topic for td in pdf_topic_definitions]

            markdown_registry_call = mock_entity_registry_markdown.resolve.await_args
            pdf_registry_call = mock_entity_registry_pdf.resolve.await_args
            assert markdown_registry_call.kwargs["embeddings"] == dedup_result.canonical_entity_embeddings
            assert pdf_registry_call.kwargs["embeddings"] == dedup_result.canonical_entity_embeddings

            markdown_assembly_input = mock_assembler_markdown.assemble.await_args.args[0]
            pdf_assembly_input = mock_assembler_pdf.assemble.await_args.args[0]

            assert markdown_assembly_input.document.name == "doc.md"
            assert pdf_assembly_input.document.name == "doc.pdf"
            assert markdown_assembly_input.document.document_date == "2014-05-28"
            assert pdf_assembly_input.document.document_date == "2014-05-28"

            assert len(markdown_assembly_input.chunks) == len(pdf_assembly_input.chunks) == 1
            assert markdown_assembly_input.chunks[0].content == pdf_assembly_input.chunks[0].content
            assert markdown_assembly_input.chunks[0].header_path == pdf_assembly_input.chunks[0].header_path
            assert markdown_assembly_input.chunks[0].position == pdf_assembly_input.chunks[0].position

            assert [e.name for e in markdown_assembly_input.entities] == [e.name for e in pdf_assembly_input.entities]
            assert [t.uuid for t in markdown_assembly_input.topics] == [t.uuid for t in pdf_assembly_input.topics]
            assert [t.name for t in markdown_assembly_input.topics] == [t.name for t in pdf_assembly_input.topics]
            assert markdown_assembly_input.entity_embeddings == dedup_result.canonical_entity_embeddings
            assert pdf_assembly_input.entity_embeddings == dedup_result.canonical_entity_embeddings

            assert len(markdown_assembly_input.facts) == len(pdf_assembly_input.facts) == 1
            markdown_fact = markdown_assembly_input.facts[0]
            pdf_fact = pdf_assembly_input.facts[0]
            assert markdown_fact.content == pdf_fact.content
            assert markdown_fact.subject_name == pdf_fact.subject_name
            assert markdown_fact.object_name == pdf_fact.object_name
            assert markdown_fact.relationship_type == pdf_fact.relationship_type
            assert markdown_fact.object_type == pdf_fact.object_type
            assert markdown_fact.date_context == pdf_fact.date_context

            expected_progress = [
                ("chunking", 0.0),
                ("chunking", 1.0),
                ("extraction", 0.0),
                ("extraction", 1.0),
                ("deduplication", 0.0),
                ("deduplication", 0.5),
                ("resolution", 0.0),
                ("resolution", 1.0),
                ("assembly", 0.0),
                ("assembly", 1.0),
            ]
            assert markdown_progress == expected_progress
            assert pdf_progress == expected_progress

    @pytest.mark.asyncio
    async def test_ingest_markdown_uses_current_schema_fields(self):
        """ingest_markdown should build models with current field names."""
        from vanna_kg.types import (
            AssemblyResult,
            CanonicalEntity,
            ChainOfThoughtResult,
            ChunkInput,
            EntityResolutionResult,
            EntityDeduplicationOutput,
            EnumeratedEntity,
            ExtractedFact,
        )
        from vanna_kg.types.topics import TopicResolution, TopicResolutionResult

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            md_path = kb_path / "doc.md"
            md_path.write_text("# Title\n\nApple acquired Beats.", encoding="utf-8")

            kg = KnowledgeGraph(kb_path)
            kg._initialized = True
            kg._storage = MagicMock()
            kg._storage._lancedb = MagicMock()
            kg._llm = MagicMock()
            kg._embeddings = MagicMock()

            chunk_inputs = [
                ChunkInput(
                    doc_id="doc-id",
                    content="Apple acquired Beats.",
                    header_path="Title",
                    position=0,
                )
            ]

            extraction_results = [
                ChainOfThoughtResult(
                    entities=[
                        EnumeratedEntity(
                            name="Apple",
                            entity_type="Company",
                            summary="Tech company",
                        ),
                        EnumeratedEntity(
                            name="Beats",
                            entity_type="Company",
                            summary="Audio brand",
                        ),
                    ],
                    facts=[
                        ExtractedFact(
                            fact="Apple acquired Beats.",
                            subject="Apple",
                            subject_type="Company",
                            object="Beats",
                            object_type="Company",
                            relationship="acquired",
                            date_context="Document date: 2014-05-28",
                            topics=["M&A"],
                        )
                    ],
                )
            ]

            canonical_entities = [
                CanonicalEntity(
                    uuid="entity-apple",
                    name="Apple",
                    entity_type="Company",
                    summary="Apple summary",
                    source_indices=[0],
                ),
                CanonicalEntity(
                    uuid="entity-beats",
                    name="Beats",
                    entity_type="Company",
                    summary="Beats summary",
                    source_indices=[1],
                ),
            ]

            dedup_result = EntityDeduplicationOutput(
                canonical_entities=canonical_entities,
                index_to_canonical={0: 0, 1: 1},
                merge_history=[],
                canonical_entity_embeddings=[
                    [0.11] * 3072,
                    [0.22] * 3072,
                ],
            )
            entity_resolution = EntityResolutionResult(
                new_entities=canonical_entities,
                uuid_remap={},
                summary_updates={},
            )
            topic_resolution = TopicResolutionResult(
                resolved_topics=[
                    TopicResolution(
                        uuid="topic-ma",
                        canonical_label="M&A",
                        is_new=False,
                        definition="Mergers and acquisitions",
                    )
                ],
                uuid_remap={"M&A": "topic-ma"},
                new_topics=[],
            )

            mock_entity_registry = MagicMock()
            mock_entity_registry.resolve = AsyncMock(return_value=entity_resolution)

            mock_topic_resolver = MagicMock()
            mock_topic_resolver.resolve = AsyncMock(return_value=topic_resolution)

            mock_assembler = MagicMock()
            mock_assembler.assemble = AsyncMock(return_value=AssemblyResult(
                document_written=True,
                chunks_written=1,
                entities_written=2,
                facts_written=1,
                topics_written=1,
                relationships_written=1,
            ))

            with patch('vanna_kg.ingestion.chunking.chunk_markdown', return_value=chunk_inputs), \
                 patch('vanna_kg.ingestion.extraction.extract_from_chunks', AsyncMock(return_value=extraction_results)), \
                 patch('vanna_kg.ingestion.resolution.deduplicate_entities', AsyncMock(return_value=dedup_result)), \
                 patch('vanna_kg.ingestion.resolution.EntityRegistry', return_value=mock_entity_registry), \
                 patch('vanna_kg.ingestion.resolution.TopicResolver', return_value=mock_topic_resolver), \
                 patch('vanna_kg.ingestion.assembly.Assembler', return_value=mock_assembler):

                result = await kg.ingest_markdown(md_path, document_date="2014-05-28")

            assert result.chunks == 1
            assert result.entities == 2
            assert result.facts == 1
            assert result.topics == 1

            topic_definitions = mock_topic_resolver.resolve.await_args.args[0]
            assert topic_definitions[0].topic == "M&A"
            assert mock_entity_registry.resolve.await_args.kwargs["embeddings"] == (
                dedup_result.canonical_entity_embeddings
            )

            assembly_input = mock_assembler.assemble.await_args.args[0]
            assert assembly_input.document.name == "doc.md"
            assert assembly_input.document.document_date == "2014-05-28"
            assert assembly_input.chunks[0].document_uuid == assembly_input.document.uuid
            assert assembly_input.entity_embeddings == dedup_result.canonical_entity_embeddings

            fact = assembly_input.facts[0]
            assert fact.relationship_type == "acquired"
            assert fact.subject_name == "Apple"
            assert fact.object_name == "Beats"
            assert fact.object_type == "entity"

            assert assembly_input.topics[0].uuid == "topic-ma"
            assert result.cost_debug is None

    @pytest.mark.asyncio
    async def test_ingest_markdown_cost_debug_attaches_report(self):
        """Ingest should include cost_debug report when enabled."""
        from vanna_kg.types import (
            AssemblyResult,
            ChunkInput,
            EntityDeduplicationOutput,
            EntityResolutionResult,
        )
        from vanna_kg.types.topics import TopicResolutionResult

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            md_path = kb_path / "doc.md"
            md_path.write_text("# Title\n\nAlpha", encoding="utf-8")

            kg = KnowledgeGraph(kb_path)
            kg._initialized = True
            kg._storage = MagicMock()
            kg._llm = MagicMock()
            kg._embeddings = MagicMock()

            chunk_inputs = [
                ChunkInput(doc_id="doc-id", content="Alpha", header_path="Title", position=0),
            ]
            dedup_result = EntityDeduplicationOutput(
                canonical_entities=[],
                index_to_canonical={},
                merge_history=[],
            )
            entity_resolution = EntityResolutionResult(
                new_entities=[],
                uuid_remap={},
                summary_updates={},
            )
            topic_resolution = TopicResolutionResult(
                resolved_topics=[],
                uuid_remap={},
                new_topics=[],
            )

            mock_entity_registry = MagicMock()
            mock_entity_registry.resolve = AsyncMock(return_value=entity_resolution)
            mock_topic_resolver = MagicMock()
            mock_topic_resolver.resolve = AsyncMock(return_value=topic_resolution)
            mock_assembler = MagicMock()
            mock_assembler.assemble = AsyncMock(
                return_value=AssemblyResult(
                    document_written=True,
                    chunks_written=1,
                    entities_written=0,
                    facts_written=0,
                    topics_written=0,
                    relationships_written=0,
                )
            )

            with patch(
                "vanna_kg.ingestion.chunking.chunk_markdown", return_value=chunk_inputs
            ), patch(
                "vanna_kg.ingestion.extraction.extract_from_chunks", AsyncMock(return_value=[])
            ), patch(
                "vanna_kg.ingestion.resolution.deduplicate_entities",
                AsyncMock(return_value=dedup_result),
            ), patch(
                "vanna_kg.ingestion.resolution.EntityRegistry", return_value=mock_entity_registry
            ), patch(
                "vanna_kg.ingestion.resolution.TopicResolver", return_value=mock_topic_resolver
            ), patch(
                "vanna_kg.ingestion.assembly.Assembler", return_value=mock_assembler
            ), patch.object(
                KnowledgeGraph, "_create_ontology_index", AsyncMock(return_value=MagicMock())
            ):
                result = await kg.ingest_markdown(md_path, cost_debug=True)

            assert result.cost_debug is not None
            assert result.cost_debug.enabled is True
            assert result.cost_debug.breakdown.total_calls == 0

    @pytest.mark.asyncio
    async def test_ingest_markdown_max_chunks_and_progress_callback(self):
        """ingest_markdown should apply max_chunks and report stage progress."""
        from vanna_kg.types import (
            AssemblyResult,
            ChunkInput,
            EntityDeduplicationOutput,
            EntityResolutionResult,
        )
        from vanna_kg.types.topics import TopicResolutionResult

        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            md_path = kb_path / "doc.md"
            md_path.write_text("# Title\n\nAlpha\n\nBeta", encoding="utf-8")

            kg = KnowledgeGraph(kb_path)
            kg._initialized = True
            kg._storage = MagicMock()
            kg._llm = MagicMock()
            kg._embeddings = MagicMock()

            chunk_inputs = [
                ChunkInput(doc_id="doc-id", content="Alpha", header_path="Title", position=0),
                ChunkInput(doc_id="doc-id", content="Beta", header_path="Title", position=1),
            ]

            dedup_result = EntityDeduplicationOutput(
                canonical_entities=[],
                index_to_canonical={},
                merge_history=[],
            )
            entity_resolution = EntityResolutionResult(
                new_entities=[],
                uuid_remap={},
                summary_updates={},
            )
            topic_resolution = TopicResolutionResult(
                resolved_topics=[],
                uuid_remap={},
                new_topics=[],
            )

            mock_entity_registry = MagicMock()
            mock_entity_registry.resolve = AsyncMock(return_value=entity_resolution)
            mock_topic_resolver = MagicMock()
            mock_topic_resolver.resolve = AsyncMock(return_value=topic_resolution)
            mock_assembler = MagicMock()
            mock_assembler.assemble = AsyncMock(
                return_value=AssemblyResult(
                    document_written=True,
                    chunks_written=1,
                    entities_written=0,
                    facts_written=0,
                    topics_written=0,
                    relationships_written=0,
                )
            )
            progress_events: list[tuple[str, float]] = []

            extract_mock = AsyncMock(return_value=[])
            with patch(
                "vanna_kg.ingestion.chunking.chunk_markdown", return_value=chunk_inputs
            ), patch(
                "vanna_kg.ingestion.extraction.extract_from_chunks", extract_mock
            ), patch(
                "vanna_kg.ingestion.resolution.deduplicate_entities",
                AsyncMock(return_value=dedup_result),
            ), patch(
                "vanna_kg.ingestion.resolution.EntityRegistry", return_value=mock_entity_registry
            ), patch(
                "vanna_kg.ingestion.resolution.TopicResolver", return_value=mock_topic_resolver
            ), patch(
                "vanna_kg.ingestion.assembly.Assembler", return_value=mock_assembler
            ), patch.object(
                KnowledgeGraph, "_create_ontology_index", AsyncMock(return_value=MagicMock())
            ):
                await kg.ingest_markdown(
                    md_path,
                    max_chunks=1,
                    on_progress=lambda stage, value: progress_events.append((stage, value)),
                )

            extraction_chunk_inputs = extract_mock.await_args.args[0]
            assert len(extraction_chunk_inputs) == 1
            assert extraction_chunk_inputs[0].content == "Alpha"

            assembly_input = mock_assembler.assemble.await_args.args[0]
            assert len(assembly_input.chunks) == 1
            assert assembly_input.chunks[0].content == "Alpha"

            assert progress_events == [
                ("chunking", 0.0),
                ("chunking", 1.0),
                ("extraction", 0.0),
                ("extraction", 1.0),
                ("deduplication", 0.0),
                ("deduplication", 0.5),
                ("resolution", 0.0),
                ("resolution", 1.0),
                ("assembly", 0.0),
                ("assembly", 1.0),
            ]

    @pytest.mark.asyncio
    async def test_ingest_markdown_raises_for_invalid_max_chunks(self):
        """ingest_markdown should reject non-positive max_chunks values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir)
            md_path = kb_path / "doc.md"
            md_path.write_text("# Title\n\nBody", encoding="utf-8")

            kg = KnowledgeGraph(kb_path)
            kg._initialized = True
            kg._storage = MagicMock()
            kg._llm = MagicMock()
            kg._embeddings = MagicMock()

            with patch("vanna_kg.ingestion.chunking.chunk_markdown", return_value=[]):
                with pytest.raises(ValueError, match="max_chunks must be a positive integer"):
                    await kg.ingest_markdown(md_path, max_chunks=0)


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self):
        """Test CLI help command."""
        from typer.testing import CliRunner
        from vanna_kg.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "ingest" in result.stdout
        assert "query" in result.stdout
        assert "info" in result.stdout
        assert "shell" in result.stdout

    def test_cli_ingest_help(self):
        """Test ingest command help."""
        from typer.testing import CliRunner
        from vanna_kg.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "--kb" in result.stdout
        assert "--pattern" in result.stdout
        assert "--date" in result.stdout

    def test_cli_query_help(self):
        """Test query command help."""
        from typer.testing import CliRunner
        from vanna_kg.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["query", "--help"])

        assert result.exit_code == 0
        assert "--kb" in result.stdout
        assert "--sources" in result.stdout

    def test_cli_info_help(self):
        """Test info command help."""
        from typer.testing import CliRunner
        from vanna_kg.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info", "--help"])

        assert result.exit_code == 0
        assert "--kb" in result.stdout

    def test_cli_shell_not_implemented(self):
        """Test shell command shows not implemented message."""
        from typer.testing import CliRunner
        from vanna_kg.cli import app

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal KB structure
            Path(tmpdir).mkdir(exist_ok=True)
            (Path(tmpdir) / "metadata.json").write_text("{}")

            result = runner.invoke(app, ["shell", "--kb", tmpdir])

            assert "not yet implemented" in result.stdout.lower()
