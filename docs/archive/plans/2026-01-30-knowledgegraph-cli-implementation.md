# KnowledgeGraph & CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the KnowledgeGraph facade class and Typer-based CLI to expose internal components through a clean public API.

**Architecture:** Lazy initialization pattern where providers/storage are created on first async operation. CLI uses `asyncio.run()` per command with Rich console output.

**Tech Stack:** Python 3.10+, Typer, Rich, asyncio, Pydantic

---

## Task 1: Add Typer Dependency

**Files:**
- Modify: `pyproject.toml:31-48`

**Step 1: Add typer to dependencies**

In `pyproject.toml`, add typer to the dependencies list:

```toml
dependencies = [
    # Storage
    "duckdb>=0.10",
    "lancedb>=0.6",
    "pyarrow>=14.0",
    "filelock>=3.12",
    # Core utilities
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    # Async
    "anyio>=4.0",
    # Types
    "typing-extensions>=4.8",
    "python-dotenv>=1.2.1",
    # Scientific computing (for entity deduplication)
    "numpy>=1.20",
    "scipy>=1.10",
    # CLI
    "typer[all]>=0.9.0",
]
```

**Step 2: Verify installation works**

Run: `pip install -e .`
Expected: Success, typer and rich installed

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add typer[all] dependency for CLI"
```

---

## Task 2: Implement KnowledgeGraph Core Structure

**Files:**
- Modify: `vanna_kg/api/knowledge_graph.py`

**Step 1: Write the implementation**

Replace the entire file with:

```python
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
    from vanna_kg.api.shell import KGShell
    from vanna_kg.config.settings import KGConfig
    from vanna_kg.providers.base import EmbeddingProvider, LLMProvider
    from vanna_kg.query import GraphRAGPipeline
    from vanna_kg.storage.parquet.backend import ParquetBackend
    from vanna_kg.types.chunks import Chunk, Document
    from vanna_kg.types.entities import Entity
    from vanna_kg.types.facts import Fact
    from vanna_kg.types.results import IngestResult, QueryResult


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
            from vanna_kg.config import KGConfig
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
        from vanna_kg.storage.parquet.backend import ParquetBackend
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
            from vanna_kg.providers.llm.openai import OpenAILLMProvider
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
            from vanna_kg.providers.embedding.openai import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(
                api_key=self._config.openai_api_key,
                model=self._config.embedding_model,
            )
        elif provider == "voyage":
            raise NotImplementedError("Voyage provider not yet implemented")
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

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
```

**Step 2: Verify syntax**

Run: `python -c "from vanna_kg.api.knowledge_graph import KnowledgeGraph; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add vanna_kg/api/knowledge_graph.py
git commit -m "feat(api): implement KnowledgeGraph core structure with lazy init"
```

---

## Task 3: Implement Ingestion Methods

**Files:**
- Modify: `vanna_kg/api/knowledge_graph.py` (append to class)

**Step 1: Add ingestion methods**

Add the following methods to the KnowledgeGraph class:

```python
    # === Ingestion Methods ===

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

        from vanna_kg.ingestion.chunking import chunk_pdf
        from vanna_kg.ingestion.extraction import extract_from_chunks
        from vanna_kg.ingestion.resolution import (
            EntityRegistry,
            TopicResolver,
            deduplicate_entities,
        )
        from vanna_kg.ingestion.assembly import Assembler
        from vanna_kg.types import (
            AssemblyInput,
            Document,
            Chunk,
            Fact,
            Topic,
            IngestResult,
            TopicDefinition,
        )

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

        # 2. Extraction: Chunks -> Entities + Facts
        report("extraction", 0.0)
        extraction_results = await extract_from_chunks(
            chunk_inputs,
            self._llm,
            document_date=document_date,
            concurrency=self._config.extraction_concurrency,
        )
        report("extraction", 1.0)

        # Collect all entities and facts
        all_entities = []
        all_facts = []
        for result in extraction_results:
            all_entities.extend(result.entities)
            all_facts.extend(result.facts)

        # 3. In-document deduplication
        report("deduplication", 0.0)
        dedup_result = await deduplicate_entities(
            all_entities,
            self._llm,
            self._embeddings,
            similarity_threshold=self._config.dedup_similarity_threshold,
        )
        report("deduplication", 0.5)

        # 4. Resolution (Entity + Topic in parallel)
        report("resolution", 0.0)

        # Collect unique topics from facts
        topic_names = set()
        for fact in all_facts:
            topic_names.update(fact.topics)

        topic_definitions = [
            TopicDefinition(name=name, definition=name)
            for name in topic_names
        ]

        # Run entity and topic resolution in parallel
        entity_registry = EntityRegistry(
            self._storage,
            self._llm,
            self._embeddings,
            self._config,
        )
        topic_resolver = TopicResolver(
            self._storage._lancedb,
            self._llm,
            self._embeddings,
            self._config,
        )

        entity_resolution, topic_resolution = await asyncio.gather(
            entity_registry.resolve(dedup_result.canonical_entities),
            topic_resolver.resolve(topic_definitions),
        )
        report("resolution", 1.0)

        # 5. Build final data structures
        # Create Document
        document = Document(
            uuid=doc_uuid,
            filename=path.name,
            date=document_date,
            metadata=metadata or {},
        )

        # Create Chunks from ChunkInputs
        chunks = [
            Chunk(
                uuid=str(uuid4()),
                doc_uuid=doc_uuid,
                content=ci.content,
                header_path=ci.header_path,
                position=ci.position,
            )
            for ci in chunk_inputs
        ]

        # Map entity names to UUIDs for fact rewriting
        entity_name_to_uuid: dict[str, str] = {}
        for ce in dedup_result.canonical_entities:
            # Use remapped UUID if entity was merged with existing
            final_uuid = entity_resolution.uuid_remap.get(ce.uuid, ce.uuid)
            entity_name_to_uuid[ce.name] = final_uuid
            for alias in ce.aliases:
                entity_name_to_uuid[alias] = final_uuid

        # Create Facts with proper UUIDs
        facts: list[Fact] = []
        chunk_idx = 0
        for result in extraction_results:
            chunk_uuid = chunks[chunk_idx].uuid if chunk_idx < len(chunks) else None
            for ef in result.facts:
                subject_uuid = entity_name_to_uuid.get(ef.subject)
                object_uuid = entity_name_to_uuid.get(ef.object)
                if subject_uuid and object_uuid and chunk_uuid:
                    facts.append(Fact(
                        uuid=str(uuid4()),
                        content=ef.fact,
                        subject_uuid=subject_uuid,
                        object_uuid=object_uuid,
                        chunk_uuid=chunk_uuid,
                        relationship=ef.relationship,
                        date_context=ef.date_context,
                    ))
            chunk_idx += 1

        # Create Topics
        topics: list[Topic] = []
        for tr in topic_resolution.resolved_topics:
            topics.append(Topic(
                uuid=tr.ontology_uuid,
                name=tr.canonical_name,
                definition=tr.definition or tr.canonical_name,
                group_id="default",
            ))

        # 6. Assembly: Write to storage
        report("assembly", 0.0)
        assembler = Assembler(self._storage, self._embeddings)

        # Combine new entities with those that already exist
        entities_to_write = entity_resolution.new_entities

        assembly_result = await assembler.assemble(AssemblyInput(
            document=document,
            chunks=chunks,
            entities=entities_to_write,
            facts=facts,
            topics=topics,
        ))
        report("assembly", 1.0)

        duration = time.time() - start_time

        return IngestResult(
            document_id=doc_uuid,
            chunks=assembly_result.chunks_written,
            entities=assembly_result.entities_written,
            facts=assembly_result.facts_written,
            topics=assembly_result.topics_written,
            duration_seconds=duration,
            errors=errors,
        )

    async def ingest_markdown(
        self,
        path: str | Path,
        *,
        document_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> "IngestResult":
        """Ingest a markdown document."""
        await self._ensure_initialized()
        assert self._storage is not None
        assert self._llm is not None
        assert self._embeddings is not None

        import time
        from uuid import uuid4

        from vanna_kg.ingestion.chunking import chunk_markdown
        from vanna_kg.ingestion.extraction import extract_from_chunks
        from vanna_kg.ingestion.resolution import (
            EntityRegistry,
            TopicResolver,
            deduplicate_entities,
        )
        from vanna_kg.ingestion.assembly import Assembler
        from vanna_kg.types import (
            AssemblyInput,
            Document,
            Chunk,
            Fact,
            Topic,
            IngestResult,
            TopicDefinition,
        )

        start_time = time.time()
        path = Path(path)
        doc_uuid = str(uuid4())

        # Read markdown content
        content = path.read_text(encoding="utf-8")

        # Chunk the markdown
        chunk_inputs = chunk_markdown(content, doc_id=doc_uuid)

        if not chunk_inputs:
            return IngestResult(
                document_id=doc_uuid,
                chunks=0,
                entities=0,
                facts=0,
                topics=0,
                duration_seconds=time.time() - start_time,
                errors=["No chunks extracted from markdown"],
            )

        # Rest of pipeline is same as PDF (extraction, resolution, assembly)
        extraction_results = await extract_from_chunks(
            chunk_inputs,
            self._llm,
            document_date=document_date,
            concurrency=self._config.extraction_concurrency,
        )

        all_entities = []
        all_facts = []
        for result in extraction_results:
            all_entities.extend(result.entities)
            all_facts.extend(result.facts)

        dedup_result = await deduplicate_entities(
            all_entities,
            self._llm,
            self._embeddings,
            similarity_threshold=self._config.dedup_similarity_threshold,
        )

        topic_names = set()
        for fact in all_facts:
            topic_names.update(fact.topics)

        topic_definitions = [
            TopicDefinition(name=name, definition=name)
            for name in topic_names
        ]

        entity_registry = EntityRegistry(
            self._storage,
            self._llm,
            self._embeddings,
            self._config,
        )
        topic_resolver = TopicResolver(
            self._storage._lancedb,
            self._llm,
            self._embeddings,
            self._config,
        )

        entity_resolution, topic_resolution = await asyncio.gather(
            entity_registry.resolve(dedup_result.canonical_entities),
            topic_resolver.resolve(topic_definitions),
        )

        document = Document(
            uuid=doc_uuid,
            filename=path.name,
            date=document_date,
            metadata=metadata or {},
        )

        chunks = [
            Chunk(
                uuid=str(uuid4()),
                doc_uuid=doc_uuid,
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
                    facts.append(Fact(
                        uuid=str(uuid4()),
                        content=ef.fact,
                        subject_uuid=subject_uuid,
                        object_uuid=object_uuid,
                        chunk_uuid=chunk_uuid,
                        relationship=ef.relationship,
                        date_context=ef.date_context,
                    ))
            chunk_idx += 1

        topics: list[Topic] = []
        for tr in topic_resolution.resolved_topics:
            topics.append(Topic(
                uuid=tr.ontology_uuid,
                name=tr.canonical_name,
                definition=tr.definition or tr.canonical_name,
                group_id="default",
            ))

        assembler = Assembler(self._storage, self._embeddings)
        entities_to_write = entity_resolution.new_entities

        assembly_result = await assembler.assemble(AssemblyInput(
            document=document,
            chunks=chunks,
            entities=entities_to_write,
            facts=facts,
            topics=topics,
        ))

        return IngestResult(
            document_id=doc_uuid,
            chunks=assembly_result.chunks_written,
            entities=assembly_result.entities_written,
            facts=assembly_result.facts_written,
            topics=assembly_result.topics_written,
            duration_seconds=time.time() - start_time,
            errors=[],
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

    def ingest_directory_sync(self, path: str | Path, **kwargs: Any) -> list["IngestResult"]:
        """Sync wrapper for ingest_directory."""
        return asyncio.run(self.ingest_directory(path, **kwargs))
```

**Step 2: Verify syntax**

Run: `python -c "from vanna_kg.api.knowledge_graph import KnowledgeGraph; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add vanna_kg/api/knowledge_graph.py
git commit -m "feat(api): implement ingestion methods with parallel resolution"
```

---

## Task 4: Implement Query Methods

**Files:**
- Modify: `vanna_kg/api/knowledge_graph.py` (append to class)

**Step 1: Add query methods**

Add the following methods to the KnowledgeGraph class:

```python
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

        from vanna_kg.query import GraphRAGPipeline
        from vanna_kg.types import QueryResult

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
                    "query": sa.query_text,
                    "answer": sa.answer,
                    "confidence": sa.confidence,
                }
                for sa in result.sub_answers
            ],
            question_type=result.question_type.value if result.question_type else None,
            timing=result.timing,
        )

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

    def search_entities_sync(self, query: str, **kwargs: Any) -> list["Entity"]:
        """Sync wrapper for search_entities."""
        return asyncio.run(self.search_entities(query, **kwargs))

    def search_facts_sync(self, query: str, **kwargs: Any) -> list["Fact"]:
        """Sync wrapper for search_facts."""
        return asyncio.run(self.search_facts(query, **kwargs))
```

**Step 2: Verify syntax**

Run: `python -c "from vanna_kg.api.knowledge_graph import KnowledgeGraph; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add vanna_kg/api/knowledge_graph.py
git commit -m "feat(api): implement query and search methods"
```

---

## Task 5: Implement Data Access Methods

**Files:**
- Modify: `vanna_kg/api/knowledge_graph.py` (append to class)

**Step 1: Add data access methods**

Add the following methods to the KnowledgeGraph class:

```python
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
        from vanna_kg.api.shell import KGShell
        return KGShell(self)
```

**Step 2: Verify syntax**

Run: `python -c "from vanna_kg.api.knowledge_graph import KnowledgeGraph; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add vanna_kg/api/knowledge_graph.py
git commit -m "feat(api): implement data access and stats methods"
```

---

## Task 6: Implement CLI Commands

**Files:**
- Modify: `vanna_kg/cli/__init__.py`

**Step 1: Write the CLI implementation**

Replace the entire file with:

```python
"""
Command-Line Interface

CLI commands for VannaKG operations.

Commands:
    vanna-kg ingest  - Ingest documents into a knowledge base
    vanna-kg query   - Query a knowledge base
    vanna-kg info    - Display knowledge base information
    vanna-kg shell   - Interactive navigation shell (placeholder)

Usage:
    # Ingest a PDF
    vanna-kg ingest report.pdf --kb ./my_kb

    # Ingest a directory
    vanna-kg ingest ./documents --kb ./my_kb --pattern "**/*.pdf"

    # Query
    vanna-kg query "What were the findings?" --kb ./my_kb

    # Show stats
    vanna-kg info --kb ./my_kb

See: docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 8
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

__all__ = ["main", "app"]

app = typer.Typer(
    name="vanna-kg",
    help="Embedded knowledge graph for document understanding",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(
        ...,
        help="File or directory to ingest",
        exists=True,
    ),
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
    ),
    pattern: str = typer.Option(
        "**/*.pdf",
        "--pattern", "-p",
        help="Glob pattern for directory ingestion",
    ),
    date: Optional[str] = typer.Option(
        None,
        "--date", "-d",
        help="Document date (YYYY-MM-DD)",
    ),
) -> None:
    """Ingest documents into a knowledge base."""

    async def _run() -> None:
        from vanna_kg.api.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(kb)

        try:
            if path.is_dir():
                # Directory ingestion
                files = list(path.glob(pattern))
                if not files:
                    console.print(f"[yellow]No files matching '{pattern}' found in {path}[/]")
                    return

                console.print(f"Found {len(files)} files to ingest")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Ingesting...", total=len(files))

                    total_entities = 0
                    total_facts = 0
                    total_chunks = 0

                    for file_path in files:
                        progress.update(task, description=f"Ingesting {file_path.name}")

                        if file_path.suffix.lower() == ".pdf":
                            result = await kg.ingest_pdf(file_path, document_date=date)
                        elif file_path.suffix.lower() in (".md", ".markdown"):
                            result = await kg.ingest_markdown(file_path, document_date=date)
                        else:
                            progress.advance(task)
                            continue

                        total_entities += result.entities
                        total_facts += result.facts
                        total_chunks += result.chunks
                        progress.advance(task)

                console.print()
                console.print(Panel(
                    f"[green]Successfully ingested {len(files)} files[/]\n\n"
                    f"  Chunks: {total_chunks}\n"
                    f"  Entities: {total_entities}\n"
                    f"  Facts: {total_facts}",
                    title="Ingestion Complete",
                ))
            else:
                # Single file ingestion
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Ingesting {path.name}...")

                    if path.suffix.lower() == ".pdf":
                        result = await kg.ingest_pdf(path, document_date=date)
                    elif path.suffix.lower() in (".md", ".markdown"):
                        result = await kg.ingest_markdown(path, document_date=date)
                    else:
                        console.print(f"[red]Unsupported file type: {path.suffix}[/]")
                        return

                    progress.update(task, completed=True)

                console.print()
                console.print(Panel(
                    f"[green]Successfully ingested {path.name}[/]\n\n"
                    f"  Document ID: {result.document_id}\n"
                    f"  Chunks: {result.chunks}\n"
                    f"  Entities: {result.entities}\n"
                    f"  Facts: {result.facts}\n"
                    f"  Duration: {result.duration_seconds:.1f}s",
                    title="Ingestion Complete",
                ))

                if result.errors:
                    console.print("[yellow]Warnings:[/]")
                    for error in result.errors:
                        console.print(f"  - {error}")
        finally:
            await kg.close()

    asyncio.run(_run())


@app.command()
def query(
    question: str = typer.Argument(
        ...,
        help="Question to ask the knowledge base",
    ),
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
        exists=True,
    ),
    sources: bool = typer.Option(
        True,
        "--sources/--no-sources",
        help="Include source citations",
    ),
) -> None:
    """Query the knowledge base."""

    async def _run() -> None:
        from vanna_kg.api.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(kb, create=False)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Thinking...")
                result = await kg.query(question, include_sources=sources)
                progress.update(task, completed=True)

            # Display answer
            console.print()
            console.print(Panel(
                Markdown(result.answer),
                title=f"Answer (confidence: {result.confidence:.0%})",
                border_style="green" if result.confidence > 0.7 else "yellow",
            ))

            # Display sources if available
            if sources and result.sources:
                console.print()
                table = Table(title="Sources")
                table.add_column("Document", style="cyan")
                table.add_column("Section", style="dim")

                seen = set()
                for source in result.sources[:5]:
                    key = (source.get("document", ""), source.get("section", ""))
                    if key not in seen:
                        seen.add(key)
                        table.add_row(
                            source.get("document", "Unknown"),
                            source.get("section", ""),
                        )

                console.print(table)

            # Display timing
            if result.timing:
                total_ms = sum(result.timing.values())
                console.print(f"\n[dim]Query time: {total_ms}ms[/]")

        finally:
            await kg.close()

    asyncio.run(_run())


@app.command()
def info(
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
        exists=True,
    ),
) -> None:
    """Display knowledge base information."""

    async def _run() -> None:
        from vanna_kg.api.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(kb, create=False)

        try:
            stats = await kg.stats()

            table = Table(title=f"Knowledge Base: {kb}")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right", style="green")

            table.add_row("Documents", str(stats["documents"]))
            table.add_row("Chunks", str(stats["chunks"]))
            table.add_row("Entities", str(stats["entities"]))
            table.add_row("Facts", str(stats["facts"]))

            console.print(table)

        finally:
            await kg.close()

    asyncio.run(_run())


@app.command()
def shell(
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
        exists=True,
    ),
) -> None:
    """Interactive navigation shell (coming soon)."""
    console.print("[yellow]Interactive shell not yet implemented.[/]")
    console.print("Use 'vanna-kg query' for now.")


def main() -> None:
    """Entry point for the CLI."""
    app()
```

**Step 2: Verify CLI works**

Run: `python -m vanna_kg.cli --help`
Expected: Help text showing ingest, query, info, shell commands

**Step 3: Commit**

```bash
git add vanna_kg/cli/__init__.py
git commit -m "feat(cli): implement typer CLI with ingest, query, info commands"
```

---

## Task 7: Add Missing Storage Methods

The KnowledgeGraph uses some storage methods that may not exist. Let's verify and add them if needed.

**Files:**
- Check: `vanna_kg/storage/duckdb/queries.py`
- Check: `vanna_kg/storage/parquet/backend.py`

**Step 1: Check for missing methods**

Verify these methods exist:
- `get_all_entities(limit, offset)`
- `get_all_documents()`
- `get_document(uuid)`

If missing, add them to the appropriate files.

**Step 2: Test the full flow**

Run: `python -c "from vanna_kg import KnowledgeGraph; kg = KnowledgeGraph('./test_kb'); print(kg.path)"`
Expected: Prints the path

**Step 3: Commit if changes made**

```bash
git add vanna_kg/storage/
git commit -m "fix(storage): add missing data access methods"
```

---

## Task 8: Update Package Exports

**Files:**
- Modify: `vanna_kg/__init__.py`

**Step 1: Ensure KnowledgeGraph is exported**

Add or verify this export exists:

```python
from vanna_kg.api.knowledge_graph import KnowledgeGraph

__all__ = [
    "KnowledgeGraph",
    # ... other exports
]
```

**Step 2: Test import**

Run: `python -c "from vanna_kg import KnowledgeGraph; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add vanna_kg/__init__.py
git commit -m "feat: export KnowledgeGraph from package root"
```

---

## Task 9: Final Integration Test

**Step 1: Run full test**

```bash
# Create a test markdown file
echo "# Test Document

Apple Inc. announced a partnership with Microsoft in 2024.

## Details

Tim Cook, CEO of Apple, met with Satya Nadella to discuss the deal.
" > /tmp/test_doc.md

# Test ingestion and query
python -c "
import asyncio
from vanna_kg import KnowledgeGraph

async def test():
    kg = KnowledgeGraph('./test_integration_kb')

    # Skip actual ingestion (requires API keys)
    # result = await kg.ingest_markdown('/tmp/test_doc.md')
    # print(f'Ingested: {result}')

    stats = await kg.stats()
    print(f'Stats: {stats}')

    await kg.close()

asyncio.run(test())
"
```

**Step 2: Test CLI**

```bash
vanna-kg --help
vanna-kg info --kb ./test_integration_kb
```

**Step 3: Clean up and final commit**

```bash
rm -rf ./test_integration_kb /tmp/test_doc.md

git add -A
git commit -m "feat: complete KnowledgeGraph and CLI implementation

Implements:
- KnowledgeGraph facade with lazy initialization
- Full ingestion pipeline (PDF, markdown, directory)
- Query pipeline integration
- Data access methods (entities, facts, documents)
- Typer CLI with ingest, query, info commands
- Rich console output with progress bars

Closes #XX"
```

---

## Summary

| Task | Description | Est. Lines |
|------|-------------|-----------|
| 1 | Add Typer dependency | 1 |
| 2 | KnowledgeGraph core structure | ~150 |
| 3 | Ingestion methods | ~250 |
| 4 | Query methods | ~80 |
| 5 | Data access methods | ~60 |
| 6 | CLI commands | ~200 |
| 7 | Storage method fixes | ~20 |
| 8 | Package exports | ~5 |
| 9 | Integration test | N/A |

**Total:** ~750 lines of new code across 2 files
