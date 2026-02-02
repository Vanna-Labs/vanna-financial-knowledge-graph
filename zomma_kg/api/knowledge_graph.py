"""
KnowledgeGraph - Primary Entry Point

The KnowledgeGraph class manages a knowledge base directory and provides
methods for ingestion, querying, and navigation.

A knowledge base is a self-contained directory containing:
    - entities.parquet: Extracted entities
    - chunks.parquet: Source text chunks
    - facts.parquet: Extracted facts/relationships
    - topics.parquet: Topic classifications
    - documents.parquet: Source document metadata
    - relationships.parquet: All graph edges
    - lancedb/: Vector indices
    - metadata.json: KB metadata and version

Example:
    >>> kg = KnowledgeGraph("./my_kb")
    >>> await kg.ingest_pdf("report.pdf")
    >>> result = await kg.query("What were the main findings?")
    >>> print(result.answer)

    # Or with sync API
    >>> kg = KnowledgeGraph("./my_kb")
    >>> kg.ingest_pdf_sync("report.pdf")
    >>> result = kg.query_sync("What were the main findings?")

See Also:
    - docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 3 (Public API Design)
    - docs/architecture/PARQUET_STORAGE_MIGRATION.md Section 4 (Storage Architecture)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from zomma_kg.api.shell import KGShell
    from zomma_kg.config.settings import KGConfig
    from zomma_kg.providers.base import EmbeddingProvider, LLMProvider
    from zomma_kg.query import GraphRAGPipeline
    from zomma_kg.storage.lancedb.indices import LanceDBIndices
    from zomma_kg.storage.parquet.backend import ParquetBackend
    from zomma_kg.types.chunks import Chunk, Document
    from zomma_kg.types.entities import Entity
    from zomma_kg.types.facts import Fact
    from zomma_kg.types.results import IngestResult, QueryResult, SubQuery


class KnowledgeGraph:
    """
    A portable, embedded knowledge graph.

    Args:
        path: Directory for the knowledge base. Created if doesn't exist.
        config: Optional configuration. Uses defaults if not provided.
        create: If True, create directory if missing. Default True.
    """

    def __init__(
        self,
        path: str | Path,
        config: "KGConfig | None" = None,
        create: bool = True,
    ) -> None:
        """Initialize knowledge graph at the specified path."""
        self._path = Path(path).resolve()
        self._create = create

        # Lazy import to avoid circular imports
        if config is None:
            from zomma_kg.config import KGConfig
            config = KGConfig()
        self._config = config

        # Lazy-initialized components
        self._storage: "ParquetBackend | None" = None
        self._llm: "LLMProvider | None" = None
        self._embeddings: "EmbeddingProvider | None" = None
        self._query_pipeline: "GraphRAGPipeline | None" = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of storage and providers on first use."""
        if self._initialized:
            return

        # Create directory if needed
        if self._create:
            self._path.mkdir(parents=True, exist_ok=True)
        elif not self._path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self._path}")

        # Initialize storage
        from zomma_kg.storage.parquet.backend import ParquetBackend
        self._storage = ParquetBackend(self._path, self._config)
        await self._storage.initialize()

        # Initialize providers
        self._llm = self._create_llm_provider()
        self._embeddings = self._create_embedding_provider()

        self._initialized = True

    def _create_llm_provider(self) -> "LLMProvider":
        """Create LLM provider based on config."""
        provider = self._config.llm_provider.lower()

        if provider == "openai":
            from zomma_kg.providers.llm.openai import OpenAILLMProvider
            return OpenAILLMProvider(
                api_key=self._config.openai_api_key,
                model=self._config.llm_model,
            )
        elif provider == "anthropic":
            raise NotImplementedError("Anthropic provider not yet implemented")
        elif provider == "google":
            raise NotImplementedError("Google provider not yet implemented")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _create_embedding_provider(self) -> "EmbeddingProvider":
        """Create embedding provider based on config."""
        provider = self._config.embedding_provider.lower()

        if provider == "openai":
            from zomma_kg.providers.embedding.openai import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(
                api_key=self._config.openai_api_key,
                model=self._config.embedding_model,
            )
        elif provider == "voyage":
            raise NotImplementedError("Voyage provider not yet implemented")
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    async def _create_ontology_index(self) -> "LanceDBIndices":
        """Create a dedicated ontology-group topic index."""
        from zomma_kg.storage.lancedb.indices import LanceDBIndices

        ontology_index = LanceDBIndices(
            self._path / "lancedb",
            self._config,
            group_id="ontology",
        )
        await ontology_index.initialize()
        return ontology_index

    # === Lifecycle ===

    def __enter__(self) -> "KnowledgeGraph":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit with resource cleanup."""
        self.close_sync()

    async def __aenter__(self) -> "KnowledgeGraph":
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Release all resources (async)."""
        if self._storage is not None:
            await self._storage.close()
            self._storage = None
        self._llm = None
        self._embeddings = None
        self._query_pipeline = None
        self._initialized = False

    def close_sync(self) -> None:
        """Release all resources (sync)."""
        if self._initialized:
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(self.close())
            except RuntimeError:
                asyncio.run(self.close())

    # === Properties ===

    @property
    def path(self) -> Path:
        """Path to the knowledge base directory."""
        return self._path

    @property
    def config(self) -> "KGConfig":
        """Current configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Whether the knowledge graph has been initialized."""
        return self._initialized

    # === Ingestion Methods ===

    async def _ingest_from_chunk_inputs(
        self,
        *,
        chunk_inputs: list[Any],
        doc_uuid: str,
        document_name: str,
        document_date: str | None,
        metadata: dict[str, Any] | None,
        start_time: float,
        errors: list[str],
        on_progress: Callable[[str, float], None] | None = None,
    ) -> "IngestResult":
        """Run shared ingestion pipeline once chunk inputs are available."""
        assert self._storage is not None
        assert self._llm is not None
        assert self._embeddings is not None

        import time
        from uuid import uuid4

        from zomma_kg.ingestion.assembly import Assembler
        from zomma_kg.ingestion.extraction import extract_from_chunks
        from zomma_kg.ingestion.resolution import (
            EntityRegistry,
            TopicResolver,
            deduplicate_entities,
        )
        from zomma_kg.types import (
            AssemblyInput,
            Chunk,
            Document,
            Fact,
            IngestResult,
            Topic,
            TopicDefinition,
        )

        def report(stage: str, progress: float) -> None:
            if on_progress:
                on_progress(stage, progress)

        # 1. Extraction: Chunks -> Entities + Facts
        report("extraction", 0.0)
        extraction_results = await extract_from_chunks(
            chunk_inputs,
            self._llm,
            document_date=document_date,
            concurrency=self._config.extraction_concurrency,
        )
        report("extraction", 1.0)

        all_entities = []
        all_facts = []
        for result in extraction_results:
            all_entities.extend(result.entities)
            all_facts.extend(result.facts)

        # 2. In-document deduplication
        report("deduplication", 0.0)
        dedup_result = await deduplicate_entities(
            all_entities,
            self._llm,
            self._embeddings,
            similarity_threshold=self._config.dedup_similarity_threshold,
        )
        report("deduplication", 0.5)

        # 3. Resolution (Entity + Topic in parallel)
        report("resolution", 0.0)
        topic_names = set()
        for fact in all_facts:
            topic_names.update(fact.topics)

        topic_definitions = [
            TopicDefinition(topic=name, definition=name)
            for name in topic_names
        ]

        entity_registry = EntityRegistry(
            self._storage,
            self._llm,
            self._embeddings,
            self._config,
        )
        ontology_index = await self._create_ontology_index()
        topic_resolver = TopicResolver(
            ontology_index,
            self._llm,
            self._embeddings,
            self._config,
        )

        entity_resolution, topic_resolution = await asyncio.gather(
            entity_registry.resolve(dedup_result.canonical_entities),
            topic_resolver.resolve(topic_definitions),
        )
        report("resolution", 1.0)

        # 4. Build final data structures
        document = Document(
            uuid=doc_uuid,
            name=document_name,
            document_date=document_date,
            metadata=metadata or {},
        )

        chunks = [
            Chunk(
                uuid=str(uuid4()),
                document_uuid=doc_uuid,
                content=ci.content,
                header_path=ci.header_path,
                position=ci.position,
            )
            for ci in chunk_inputs
        ]

        entity_name_to_uuid: dict[str, str] = {}
        for ce in dedup_result.canonical_entities:
            final_uuid = entity_resolution.uuid_remap.get(ce.uuid, ce.uuid)
            entity_name_to_uuid[ce.name] = final_uuid
            for alias in ce.aliases:
                entity_name_to_uuid[alias] = final_uuid

        facts: list[Fact] = []
        chunk_idx = 0
        for result in extraction_results:
            chunk_uuid = chunks[chunk_idx].uuid if chunk_idx < len(chunks) else None
            for ef in result.facts:
                subject_uuid = entity_name_to_uuid.get(ef.subject)
                object_uuid = entity_name_to_uuid.get(ef.object)
                if subject_uuid and object_uuid and chunk_uuid:
                    facts.append(
                        Fact(
                            uuid=str(uuid4()),
                            content=ef.fact,
                            subject_uuid=subject_uuid,
                            subject_name=ef.subject,
                            object_uuid=object_uuid,
                            object_name=ef.object,
                            object_type="topic" if ef.object_type == "Topic" else "entity",
                            chunk_uuid=chunk_uuid,
                            relationship_type=ef.relationship,
                            date_context=ef.date_context,
                        )
                    )
            chunk_idx += 1

        topics: list[Topic] = []
        for tr in topic_resolution.resolved_topics:
            topics.append(
                Topic(
                    uuid=tr.uuid,
                    name=tr.canonical_name,
                    definition=tr.definition or tr.canonical_name,
                    group_id="default",
                )
            )

        # 5. Assembly: Write to storage
        report("assembly", 0.0)
        assembler = Assembler(self._storage, self._embeddings)
        entities_to_write = entity_resolution.new_entities

        assembly_result = await assembler.assemble(
            AssemblyInput(
                document=document,
                chunks=chunks,
                entities=entities_to_write,
                facts=facts,
                topics=topics,
            )
        )
        report("assembly", 1.0)

        return IngestResult(
            document_id=doc_uuid,
            chunks=assembly_result.chunks_written,
            entities=assembly_result.entities_written,
            facts=assembly_result.facts_written,
            topics=assembly_result.topics_written,
            duration_seconds=time.time() - start_time,
            errors=errors,
        )

    async def ingest_pdf(
        self,
        path: str | Path,
        *,
        document_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> "IngestResult":
        """
        Ingest a PDF document into the knowledge graph.

        Pipeline:
            1. PDF -> Markdown (Gemini vision model)
            2. Markdown -> Chunks (header-aware splitting)
            3. Chunks -> Entities/Facts (LLM extraction with critique)
            4. Deduplication (embeddings + Union-Find + LLM verification)
            5. Assembly (write to Parquet + LanceDB)

        Args:
            path: Path to PDF file
            document_date: Optional document date (YYYY-MM-DD)
            metadata: Optional metadata dict
            on_progress: Optional callback (stage_name, progress_0_to_1)

        Returns:
            IngestResult with counts of entities, facts, chunks written
        """
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._llm is not None
        assert self._embeddings is not None

        import time
        from uuid import uuid4

        from zomma_kg.ingestion.chunking import chunk_pdf
        from zomma_kg.types import IngestResult

        start_time = time.time()
        path = Path(path)
        doc_uuid = str(uuid4())
        errors: list[str] = []

        # Report progress helper
        def report(stage: str, progress: float) -> None:
            if on_progress:
                on_progress(stage, progress)

        # 1. Chunking: PDF -> Chunks
        report("chunking", 0.0)
        chunk_inputs = await chunk_pdf(
            path,
            doc_id=doc_uuid,
            api_key=self._config.google_api_key,
        )
        report("chunking", 1.0)

        if not chunk_inputs:
            return IngestResult(
                document_id=doc_uuid,
                chunks=0,
                entities=0,
                facts=0,
                topics=0,
                duration_seconds=time.time() - start_time,
                errors=["No chunks extracted from PDF"],
            )
        return await self._ingest_from_chunk_inputs(
            chunk_inputs=chunk_inputs,
            doc_uuid=doc_uuid,
            document_name=path.name,
            document_date=document_date,
            metadata=metadata,
            start_time=start_time,
            errors=errors,
            on_progress=on_progress,
        )

    async def ingest_markdown(
        self,
        path: str | Path,
        *,
        document_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_chunks: int | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> "IngestResult":
        """Ingest a markdown document."""
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._llm is not None
        assert self._embeddings is not None

        import time
        from uuid import uuid4

        from zomma_kg.ingestion.chunking import chunk_markdown
        from zomma_kg.types import IngestResult

        start_time = time.time()
        path = Path(path)
        doc_uuid = str(uuid4())
        errors: list[str] = []

        # Report progress helper
        def report(stage: str, progress: float) -> None:
            if on_progress:
                on_progress(stage, progress)

        # Read markdown content
        report("chunking", 0.0)
        content = path.read_text(encoding="utf-8")

        # Chunk the markdown
        chunk_inputs = chunk_markdown(content, doc_id=doc_uuid)
        if max_chunks is not None:
            if max_chunks <= 0:
                raise ValueError("max_chunks must be a positive integer")
            chunk_inputs = chunk_inputs[:max_chunks]
        report("chunking", 1.0)

        if not chunk_inputs:
            errors.append("No chunks extracted from markdown")
            return IngestResult(
                document_id=doc_uuid,
                chunks=0,
                entities=0,
                facts=0,
                topics=0,
                duration_seconds=time.time() - start_time,
                errors=errors,
            )
        return await self._ingest_from_chunk_inputs(
            chunk_inputs=chunk_inputs,
            doc_uuid=doc_uuid,
            document_name=path.name,
            document_date=document_date,
            metadata=metadata,
            start_time=start_time,
            errors=errors,
            on_progress=on_progress,
        )

    async def ingest_chunks(
        self,
        chunks: list["Chunk"],
        *,
        document_name: str | None = None,
        document_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> "IngestResult":
        """Ingest pre-chunked content into the knowledge graph."""
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._llm is not None
        assert self._embeddings is not None

        import time
        from uuid import uuid4

        from zomma_kg.ingestion.assembly import Assembler
        from zomma_kg.ingestion.extraction import extract_from_chunks
        from zomma_kg.ingestion.resolution import (
            EntityRegistry,
            TopicResolver,
            deduplicate_entities,
        )
        from zomma_kg.types import (
            AssemblyInput,
            Chunk as KGChunk,
            ChunkInput,
            Document,
            Fact,
            IngestResult,
            Topic,
            TopicDefinition,
        )

        start_time = time.time()
        errors: list[str] = []

        if not chunks:
            return IngestResult(
                document_id=str(uuid4()),
                chunks=0,
                entities=0,
                facts=0,
                topics=0,
                duration_seconds=time.time() - start_time,
                errors=["No chunks provided"],
            )

        # Report progress helper
        def report(stage: str, progress: float) -> None:
            if on_progress:
                on_progress(stage, progress)

        doc_uuids = {c.document_uuid for c in chunks if c.document_uuid}
        if len(doc_uuids) > 1:
            raise ValueError("All chunks must share the same document_uuid")

        doc_uuid = next(iter(doc_uuids), str(uuid4()))
        effective_document_date = document_date or next(
            (c.document_date for c in chunks if c.document_date),
            None,
        )

        normalized_chunks: list[KGChunk] = []
        chunk_inputs: list[ChunkInput] = []
        for idx, chunk in enumerate(chunks):
            position = chunk.position if chunk.position is not None else idx
            chunk_uuid = chunk.uuid or str(uuid4())
            normalized_chunks.append(
                KGChunk(
                    uuid=chunk_uuid,
                    document_uuid=doc_uuid,
                    content=chunk.content,
                    header_path=chunk.header_path,
                    position=position,
                    document_date=effective_document_date,
                )
            )
            chunk_inputs.append(
                ChunkInput(
                    doc_id=doc_uuid,
                    content=chunk.content,
                    header_path=chunk.header_path,
                    position=position,
                )
            )

        # 1. Extraction: Chunks -> Entities + Facts
        report("extraction", 0.0)
        extraction_results = await extract_from_chunks(
            chunk_inputs,
            self._llm,
            document_date=effective_document_date,
            concurrency=self._config.extraction_concurrency,
        )
        report("extraction", 1.0)

        all_entities = []
        all_facts = []
        for result in extraction_results:
            all_entities.extend(result.entities)
            all_facts.extend(result.facts)

        # 2. In-document deduplication
        report("deduplication", 0.0)
        dedup_result = await deduplicate_entities(
            all_entities,
            self._llm,
            self._embeddings,
            similarity_threshold=self._config.dedup_similarity_threshold,
        )
        report("deduplication", 0.5)

        # 3. Resolution (Entity + Topic in parallel)
        report("resolution", 0.0)
        topic_names = set()
        for fact in all_facts:
            topic_names.update(fact.topics)

        topic_definitions = [
            TopicDefinition(topic=name, definition=name)
            for name in topic_names
        ]

        entity_registry = EntityRegistry(
            self._storage,
            self._llm,
            self._embeddings,
            self._config,
        )
        ontology_index = await self._create_ontology_index()
        topic_resolver = TopicResolver(
            ontology_index,
            self._llm,
            self._embeddings,
            self._config,
        )

        entity_resolution, topic_resolution = await asyncio.gather(
            entity_registry.resolve(dedup_result.canonical_entities),
            topic_resolver.resolve(topic_definitions),
        )
        report("resolution", 1.0)

        # 4. Build final data structures
        document = Document(
            uuid=doc_uuid,
            name=document_name or f"chunks-{doc_uuid[:8]}",
            document_date=effective_document_date,
            metadata=metadata or {},
        )

        entity_name_to_uuid: dict[str, str] = {}
        for ce in dedup_result.canonical_entities:
            final_uuid = entity_resolution.uuid_remap.get(ce.uuid, ce.uuid)
            entity_name_to_uuid[ce.name] = final_uuid
            for alias in ce.aliases:
                entity_name_to_uuid[alias] = final_uuid

        facts: list[Fact] = []
        chunk_idx = 0
        for result in extraction_results:
            chunk_uuid = (
                normalized_chunks[chunk_idx].uuid if chunk_idx < len(normalized_chunks) else None
            )
            for ef in result.facts:
                subject_uuid = entity_name_to_uuid.get(ef.subject)
                object_uuid = entity_name_to_uuid.get(ef.object)
                if subject_uuid and object_uuid and chunk_uuid:
                    facts.append(
                        Fact(
                            uuid=str(uuid4()),
                            content=ef.fact,
                            subject_uuid=subject_uuid,
                            subject_name=ef.subject,
                            object_uuid=object_uuid,
                            object_name=ef.object,
                            object_type="topic" if ef.object_type == "Topic" else "entity",
                            chunk_uuid=chunk_uuid,
                            relationship_type=ef.relationship,
                            date_context=ef.date_context,
                        )
                    )
            chunk_idx += 1

        topics: list[Topic] = []
        for tr in topic_resolution.resolved_topics:
            topics.append(
                Topic(
                    uuid=tr.uuid,
                    name=tr.canonical_name,
                    definition=tr.definition or tr.canonical_name,
                    group_id="default",
                )
            )

        # 5. Assembly: Write to storage
        report("assembly", 0.0)
        assembler = Assembler(self._storage, self._embeddings)
        entities_to_write = entity_resolution.new_entities

        assembly_result = await assembler.assemble(
            AssemblyInput(
                document=document,
                chunks=normalized_chunks,
                entities=entities_to_write,
                facts=facts,
                topics=topics,
            )
        )
        report("assembly", 1.0)

        return IngestResult(
            document_id=doc_uuid,
            chunks=assembly_result.chunks_written,
            entities=assembly_result.entities_written,
            facts=assembly_result.facts_written,
            topics=assembly_result.topics_written,
            duration_seconds=time.time() - start_time,
            errors=errors,
        )

    async def ingest_directory(
        self,
        path: str | Path,
        *,
        pattern: str = "**/*.pdf",
        recursive: bool = True,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list["IngestResult"]:
        """
        Ingest all matching files in a directory.

        Args:
            path: Directory path
            pattern: Glob pattern (default: **/*.pdf)
            recursive: Whether to search recursively
            on_progress: Callback (filename, current, total)

        Returns:
            List of IngestResult for each file
        """
        await self._ensure_initialized()

        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        files = list(path.glob(pattern))
        results = []

        for i, file_path in enumerate(files):
            if on_progress:
                on_progress(file_path.name, i + 1, len(files))

            if file_path.suffix.lower() == ".pdf":
                result = await self.ingest_pdf(file_path)
            elif file_path.suffix.lower() in (".md", ".markdown"):
                result = await self.ingest_markdown(file_path)
            else:
                continue

            results.append(result)

        return results

    # Sync wrappers
    def ingest_pdf_sync(self, path: str | Path, **kwargs: Any) -> "IngestResult":
        """Sync wrapper for ingest_pdf."""
        return asyncio.run(self.ingest_pdf(path, **kwargs))

    def ingest_markdown_sync(self, path: str | Path, **kwargs: Any) -> "IngestResult":
        """Sync wrapper for ingest_markdown."""
        return asyncio.run(self.ingest_markdown(path, **kwargs))

    def ingest_chunks_sync(self, chunks: list["Chunk"], **kwargs: Any) -> "IngestResult":
        """Sync wrapper for ingest_chunks."""
        return asyncio.run(self.ingest_chunks(chunks, **kwargs))

    def ingest_directory_sync(self, path: str | Path, **kwargs: Any) -> list["IngestResult"]:
        """Sync wrapper for ingest_directory."""
        return asyncio.run(self.ingest_directory(path, **kwargs))

    # === Query Methods ===

    async def query(
        self,
        question: str,
        *,
        include_sources: bool = True,
    ) -> "QueryResult":
        """
        Answer a question using the knowledge graph.

        V7 Pipeline:
            1. Decompose question into sub-queries
            2. Resolve entities/topics (wide-net: one hint -> many nodes)
            3. Retrieve chunks, facts, neighbors in parallel
            4. Assemble context with relevance filtering
            5. Synthesize answer with question-type-aware formatting

        Args:
            question: Natural language question
            include_sources: Whether to include source citations

        Returns:
            QueryResult with answer, confidence, sources, timing
        """
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._llm is not None
        assert self._embeddings is not None

        from zomma_kg.query import GraphRAGPipeline
        from zomma_kg.types import QueryResult

        # Cache pipeline for reuse
        if self._query_pipeline is None:
            self._query_pipeline = GraphRAGPipeline(
                storage=self._storage,
                llm=self._llm,
                embeddings=self._embeddings,
                config=self._config,
            )

        # Execute query
        result = await self._query_pipeline.query(
            question,
            include_sources=include_sources,
        )

        # Map PipelineResult to public QueryResult
        return QueryResult(
            answer=result.answer,
            confidence=result.confidence,
            sources=result.sources if include_sources else [],
            sub_answers=[
                {
                    "query": sa.sub_query,
                    "answer": sa.answer,
                    "confidence": sa.confidence,
                }
                for sa in result.sub_answers
            ],
            question_type=result.question_type,
            timing=result.timing,
        )

    async def decompose(
        self,
        question: str,
        *,
        max_subqueries: int | None = None,
    ) -> list["SubQuery"]:
        """Decompose a question into focused sub-queries."""
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._llm is not None
        assert self._embeddings is not None

        from zomma_kg.query import GraphRAGPipeline

        if self._query_pipeline is None:
            self._query_pipeline = GraphRAGPipeline(
                storage=self._storage,
                llm=self._llm,
                embeddings=self._embeddings,
                config=self._config,
            )

        decomposition = await self._query_pipeline.decomposer.decompose(
            question,
            max_subqueries=max_subqueries,
        )
        return decomposition.sub_queries

    async def search_entities(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list["Entity"]:
        """Search for entities by semantic similarity."""
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._embeddings is not None

        # Generate query embedding
        query_vector = await self._embeddings.embed_single(query)

        # Search
        results = await self._storage.search_entities(
            query_vector,
            limit=limit,
            threshold=threshold,
        )

        return [entity for entity, score in results]

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list["Fact"]:
        """Search for facts by semantic similarity."""
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._embeddings is not None

        query_vector = await self._embeddings.embed_single(query)

        results = await self._storage.search_facts(
            query_vector,
            limit=limit,
            threshold=threshold,
        )

        return [fact for fact, score in results]

    async def search_chunks(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list["Chunk"]:
        """Search for source chunks by semantic similarity."""
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._embeddings is not None

        # Note: chunks don't have embeddings in current schema
        # This would need chunk embeddings to be added to LanceDB
        raise NotImplementedError("Chunk search not yet implemented - requires chunk embeddings")

    # Sync wrappers
    def query_sync(self, question: str, **kwargs: Any) -> "QueryResult":
        """Sync wrapper for query."""
        return asyncio.run(self.query(question, **kwargs))

    def decompose_sync(self, question: str, **kwargs: Any) -> list["SubQuery"]:
        """Sync wrapper for decompose."""
        return asyncio.run(self.decompose(question, **kwargs))

    def search_entities_sync(self, query: str, **kwargs: Any) -> list["Entity"]:
        """Sync wrapper for search_entities."""
        return asyncio.run(self.search_entities(query, **kwargs))

    def search_facts_sync(self, query: str, **kwargs: Any) -> list["Fact"]:
        """Sync wrapper for search_facts."""
        return asyncio.run(self.search_facts(query, **kwargs))

    # === Data Access ===

    async def get_entity(self, name: str) -> "Entity | None":
        """Get an entity by canonical name."""
        await self._ensure_initialized()
        assert self._storage is not None
        return await self._storage.get_entity_by_name(name)

    async def get_entities(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list["Entity"]:
        """List entities with pagination."""
        await self._ensure_initialized()
        assert self._storage is not None
        # Use DuckDB for efficient pagination
        return await self._storage._duckdb.get_all_entities(limit=limit, offset=offset)

    async def get_document(self, document_id: str) -> "Document | None":
        """Get a document by ID."""
        await self._ensure_initialized()
        assert self._storage is not None
        return await self._storage.get_document(document_id)

    async def get_documents(self) -> list["Document"]:
        """List all documents in the knowledge base."""
        await self._ensure_initialized()
        assert self._storage is not None
        return await self._storage.get_all_documents()

    async def get_facts_for_entity(
        self,
        entity_name: str,
        *,
        limit: int = 100,
    ) -> list["Fact"]:
        """Get facts involving an entity."""
        await self._ensure_initialized()
        assert self._storage is not None
        return await self._storage.get_entity_facts(entity_name, limit=limit)

    async def get_neighbors(
        self,
        entity_name: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get entities connected to the given entity."""
        await self._ensure_initialized()
        assert self._storage is not None
        return await self._storage.get_entity_neighbors(entity_name, limit=limit)

    # === Statistics ===

    async def stats(self) -> dict[str, int]:
        """Get knowledge base statistics."""
        await self._ensure_initialized()
        assert self._storage is not None

        return {
            "documents": await self._storage.count_documents(),
            "entities": await self._storage.count_entities(),
            "chunks": await self._storage.count_chunks(),
            "facts": await self._storage.count_facts(),
        }

    def stats_sync(self) -> dict[str, int]:
        """Sync wrapper for stats."""
        return asyncio.run(self.stats())

    # === Shell/Navigation ===

    def shell(self) -> "KGShell":
        """
        Get an interactive shell for navigating the knowledge graph.

        The shell presents the KG as a virtual filesystem navigable
        with familiar commands: ls, cd, cat, grep, find, etc.
        """
        from zomma_kg.api.shell import KGShell
        return KGShell(self)
