# Python Package Design: zomma-kg

## Executive Summary

### Vision

Transform ZommaLabsKG from an infrastructure-dependent knowledge graph system into a **pip-installable Python library** with zero server requirements. Users should be able to run:

```bash
pip install zomma-kg
```

And immediately begin building knowledge graphs from documents without provisioning databases, managing connection strings, or handling cloud service authentication.

### Target Users

1. **Developers building RAG applications** - Need to transform document collections into queryable knowledge structures without infrastructure complexity
2. **Financial analysts** - Require domain-specific entity extraction and relationship mapping for investment research
3. **AI/ML engineers** - Building agent-based systems that need structured knowledge retrieval
4. **Research teams** - Processing large document corpora for systematic analysis

### Key Differentiators

| Feature | Current State (Neo4j) | Target State (Embedded) |
|---------|----------------------|------------------------|
| **Installation** | pip install + Neo4j setup + connection config | pip install only |
| **First query** | 30-60s (cold start) | <1s |
| **Knowledge base** | Remote database | Local directory |
| **Portability** | Tied to database instance | zip and share |
| **Offline operation** | Impossible | Full support |
| **Multi-tenant** | group_id in shared DB | Separate directories |

### Design Principles

1. **Zero infrastructure** - No servers, no containers, no cloud services required for core functionality
2. **Progressive complexity** - Simple defaults that work, with advanced configuration available
3. **Portable knowledge bases** - A KB is just a directory; move it, version it, share it
4. **Provider agnostic** - Support multiple LLM providers with consistent interface
5. **Optional acceleration** - Rust components available but not required

---

## Package Architecture Overview

### Public API Surface

The library exposes a minimal, coherent API designed for discoverability and ease of use.

**Top-level imports:**
```python
from zomma_kg import (
    # Main entry point
    KnowledgeGraph,

    # Ingestion
    ingest_pdf,
    ingest_markdown,
    ingest_chunks,

    # Query
    query,
    decompose,

    # Navigation/Shell
    KGShell,

    # Configuration
    KGConfig,

    # Types
    Chunk,
    Entity,
    Fact,
    Topic,
    QueryResult,
)
```

**Design rationale:**
- Single entry point (`KnowledgeGraph`) for most use cases
- Convenience functions for common operations
- Explicit types for structured data
- Configuration separate from operations

### Internal Module Organization

```
zomma_kg/
├── __init__.py                 # Public API exports
├── py.typed                    # PEP 561 marker for type checking
│
├── api/                        # Public API implementations
│   ├── __init__.py
│   ├── knowledge_graph.py      # KnowledgeGraph class
│   ├── shell.py                # KGShell interactive interface
│   └── convenience.py          # Top-level convenience functions
│
├── config/                     # Configuration system
│   ├── __init__.py
│   ├── settings.py             # KGConfig class
│   ├── providers.py            # Provider configurations
│   └── defaults.py             # Default values and constants
│
├── ingestion/                  # Document processing pipeline
│   ├── __init__.py
│   ├── pipeline.py             # Three-phase ingestion orchestrator
│   ├── chunking/               # Document chunking
│   │   ├── __init__.py
│   │   ├── pdf.py              # PDF to markdown conversion
│   │   ├── markdown.py         # Markdown chunking
│   │   └── text.py             # Plain text chunking
│   ├── extraction/             # LLM-based extraction
│   │   ├── __init__.py
│   │   ├── extractor.py        # Chain-of-thought extraction
│   │   ├── critique.py         # Extraction quality assessment
│   │   └── schemas.py          # Pydantic schemas for extraction
│   ├── resolution/             # Entity and topic resolution
│   │   ├── __init__.py
│   │   ├── entity_dedup.py     # In-document deduplication
│   │   ├── entity_registry.py  # Cross-document entity matching
│   │   └── topic_resolver.py   # Topic ontology resolution
│   └── assembly/               # Knowledge base construction
│       ├── __init__.py
│       └── assembler.py        # Write resolved data to storage
│
├── query/                      # Query pipeline
│   ├── __init__.py
│   ├── pipeline.py             # V7 query pipeline orchestrator
│   ├── decomposer.py           # Question decomposition
│   ├── researcher.py           # Per-subquery research
│   ├── retriever.py            # Multi-modal retrieval
│   └── synthesizer.py          # Answer synthesis
│
├── storage/                    # Storage backends
│   ├── __init__.py
│   ├── base.py                 # Abstract storage interface
│   ├── parquet/                # Primary storage implementation
│   │   ├── __init__.py
│   │   ├── backend.py          # ParquetBackend class
│   │   ├── tables.py           # Table schemas and operations
│   │   └── migrations.py       # Schema versioning
│   ├── lancedb/                # Vector search
│   │   ├── __init__.py
│   │   └── indices.py          # Vector index management
│   └── duckdb/                 # Relational queries
│       ├── __init__.py
│       └── queries.py          # SQL query implementations
│
├── providers/                  # LLM and embedding providers
│   ├── __init__.py
│   ├── base.py                 # Abstract provider interfaces
│   ├── llm/                    # LLM providers
│   │   ├── __init__.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── google.py
│   │   └── local.py            # Ollama, llama.cpp, etc.
│   └── embedding/              # Embedding providers
│       ├── __init__.py
│       ├── openai.py
│       ├── voyage.py
│       └── local.py            # sentence-transformers, etc.
│
├── shell/                      # Filesystem navigation
│   ├── __init__.py
│   ├── commands.py             # ls, cd, cat, grep, find, etc.
│   ├── path_resolver.py        # Virtual path resolution
│   └── formatters.py           # Output formatting
│
├── types/                      # Type definitions
│   ├── __init__.py
│   ├── entities.py             # Entity, EntityType
│   ├── facts.py                # Fact, Relationship
│   ├── chunks.py               # Chunk, Document
│   ├── topics.py               # Topic
│   └── results.py              # QueryResult, SearchResult
│
├── _core/                      # Optional Rust extension (compiled)
│   └── __init__.py             # Imports from compiled extension
│
└── _compat/                    # Compatibility and fallbacks
    ├── __init__.py
    ├── rust_fallback.py        # Pure Python fallbacks for Rust code
    └── legacy.py               # Migration helpers for Neo4j users
```

### Optional Dependencies Architecture

The package uses Python's extras mechanism to enable optional functionality without bloating the base installation.

**Dependency tiers:**

1. **Core (always installed)** - Minimum viable functionality
2. **Provider extras** - LLM/embedding provider clients
3. **Performance extras** - Rust acceleration, GPU support
4. **Development extras** - Testing, linting, documentation

```toml
[project.optional-dependencies]
# LLM providers
openai = ["openai>=1.0", "tiktoken"]
anthropic = ["anthropic>=0.18"]
google = ["google-generativeai>=0.5"]
xai = ["langchain-xai>=0.2"]

# Embedding providers
voyage = ["voyageai>=0.2"]
local-embed = ["sentence-transformers>=2.2", "torch>=2.0"]

# Performance
rust = []  # No extra deps; maturin builds native extension

# Bundles
all-providers = ["zomma-kg[openai,anthropic,google,voyage]"]
all = ["zomma-kg[all-providers,rust]"]

# Development
dev = ["pytest>=7.0", "pytest-asyncio", "ruff", "mypy"]
```

---

## Public API Design

### KnowledgeGraph Class

The `KnowledgeGraph` class is the primary entry point for all operations. It manages a knowledge base directory and provides methods for ingestion, querying, and navigation.

**Design principles:**
- Single class for entire lifecycle (create, ingest, query, export)
- Lazy initialization (don't connect to anything until needed)
- Context manager support for resource cleanup
- Async-first with sync wrappers

```python
class KnowledgeGraph:
    """
    A portable, embedded knowledge graph.

    Example:
        # Create and populate a knowledge graph
        kg = KnowledgeGraph("./my_kb")
        await kg.ingest_pdf("financial_report.pdf")
        result = await kg.query("What were the key risks mentioned?")
        print(result.answer)

        # Or use sync API
        kg = KnowledgeGraph("./my_kb")
        kg.ingest_pdf_sync("financial_report.pdf")
        result = kg.query_sync("What were the key risks mentioned?")

    Args:
        path: Directory for the knowledge base. Created if doesn't exist.
        config: Optional configuration. Uses defaults if not provided.
        create: If True, create directory if missing. Default True.

    The knowledge base directory contains:
        - entities.parquet: Extracted entities
        - chunks.parquet: Source text chunks
        - facts.parquet: Extracted facts/relationships
        - topics.parquet: Topic classifications
        - documents.parquet: Source document metadata
        - relationships.parquet: Graph edges
        - lancedb/: Vector indices
        - metadata.json: KB metadata and version
    """

    def __init__(
        self,
        path: str | Path,
        config: KGConfig | None = None,
        create: bool = True,
    ) -> None:
        """Initialize knowledge graph at the specified path."""
        ...

    # === Lifecycle ===

    def __enter__(self) -> "KnowledgeGraph":
        """Context manager entry."""
        ...

    def __exit__(self, *args) -> None:
        """Context manager exit with resource cleanup."""
        ...

    async def close(self) -> None:
        """Release all resources (async)."""
        ...

    def close_sync(self) -> None:
        """Release all resources (sync)."""
        ...

    # === Properties ===

    @property
    def path(self) -> Path:
        """Path to the knowledge base directory."""
        ...

    @property
    def config(self) -> KGConfig:
        """Current configuration."""
        ...

    @property
    def stats(self) -> KGStats:
        """Statistics about the knowledge base."""
        ...

    # === Ingestion Methods ===

    async def ingest_pdf(
        self,
        path: str | Path,
        *,
        document_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> IngestResult:
        """
        Ingest a PDF document into the knowledge graph.

        The PDF is converted to markdown, chunked, and processed through
        the three-phase extraction pipeline:
        1. Chunking and parallel extraction
        2. Entity/topic resolution and deduplication
        3. Assembly into the knowledge base

        Args:
            path: Path to the PDF file
            document_date: Override document date (ISO format: YYYY-MM-DD)
            metadata: Additional metadata to attach to the document
            on_progress: Callback for progress updates

        Returns:
            IngestResult with statistics about extracted entities, facts, etc.
        """
        ...

    async def ingest_markdown(
        self,
        path: str | Path,
        *,
        document_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> IngestResult:
        """
        Ingest a markdown document.

        Markdown is chunked by headers while preserving hierarchy context.
        """
        ...

    async def ingest_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        document_date: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """
        Ingest raw text content.

        Text is chunked by paragraphs with configurable size limits.
        """
        ...

    async def ingest_chunks(
        self,
        chunks: list[Chunk],
        *,
        document_id: str | None = None,
        skip_chunking: bool = True,
    ) -> IngestResult:
        """
        Ingest pre-chunked content.

        Useful when you have custom chunking logic or pre-processed data.
        """
        ...

    async def ingest_directory(
        self,
        path: str | Path,
        *,
        pattern: str = "**/*.pdf",
        recursive: bool = True,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> list[IngestResult]:
        """
        Ingest all matching files in a directory.

        Processes files in parallel with configurable concurrency.
        """
        ...

    # Sync wrappers
    def ingest_pdf_sync(self, path: str | Path, **kwargs) -> IngestResult: ...
    def ingest_markdown_sync(self, path: str | Path, **kwargs) -> IngestResult: ...
    def ingest_text_sync(self, text: str, **kwargs) -> IngestResult: ...
    def ingest_chunks_sync(self, chunks: list[Chunk], **kwargs) -> IngestResult: ...
    def ingest_directory_sync(self, path: str | Path, **kwargs) -> list[IngestResult]: ...

    # === Query Methods ===

    async def query(
        self,
        question: str,
        *,
        max_chunks: int = 20,
        max_facts: int = 50,
        include_sources: bool = True,
        synthesis_model: str | None = None,
    ) -> QueryResult:
        """
        Answer a question using the knowledge graph.

        Uses the V7 query pipeline:
        1. Decompose question into sub-queries
        2. Research each sub-query (entity resolution, retrieval)
        3. Assemble evidence from multiple sources
        4. Synthesize final answer

        Args:
            question: Natural language question
            max_chunks: Maximum source chunks to retrieve
            max_facts: Maximum facts to consider
            include_sources: Include source citations in result
            synthesis_model: Override model for answer synthesis

        Returns:
            QueryResult with answer, sources, and metadata
        """
        ...

    async def decompose(
        self,
        question: str,
        *,
        max_subqueries: int = 5,
    ) -> list[SubQuery]:
        """
        Decompose a complex question into sub-queries.

        Useful for understanding how a question will be processed
        or for custom query pipelines.
        """
        ...

    async def search_entities(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.3,
        entity_types: list[str] | None = None,
    ) -> list[EntityMatch]:
        """
        Search for entities by semantic similarity.
        """
        ...

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 20,
        threshold: float = 0.3,
        entity_filter: str | None = None,
    ) -> list[FactMatch]:
        """
        Search for facts by semantic similarity.
        """
        ...

    async def search_chunks(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[ChunkMatch]:
        """
        Search for source chunks by semantic similarity.
        """
        ...

    # Sync wrappers
    def query_sync(self, question: str, **kwargs) -> QueryResult: ...
    def decompose_sync(self, question: str, **kwargs) -> list[SubQuery]: ...
    def search_entities_sync(self, query: str, **kwargs) -> list[EntityMatch]: ...
    def search_facts_sync(self, query: str, **kwargs) -> list[FactMatch]: ...
    def search_chunks_sync(self, query: str, **kwargs) -> list[ChunkMatch]: ...

    # === Shell/Navigation ===

    def shell(self) -> "KGShell":
        """
        Get an interactive shell for navigating the knowledge graph.

        The shell presents the KG as a virtual filesystem navigable
        with familiar commands: ls, cd, cat, grep, find, etc.

        Example:
            shell = kg.shell()
            shell.cd("/entities/organizations/")
            print(shell.ls())
            shell.cd("Apple_Inc/")
            print(shell.cat("summary.txt"))
        """
        ...

    async def query_with_shell(
        self,
        question: str,
        *,
        max_steps: int = 20,
    ) -> ShellQueryResult:
        """
        Answer a question using agent-based shell navigation.

        An LLM agent navigates the knowledge graph using filesystem
        commands, building context iteratively before answering.

        This approach is useful for:
        - Complex multi-hop questions
        - Exploratory queries where the path isn't clear
        - Debugging and understanding KG structure
        """
        ...

    # === Data Access ===

    def get_entity(self, name: str) -> Entity | None:
        """Get an entity by canonical name."""
        ...

    def get_entities(
        self,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with optional filtering."""
        ...

    def get_document(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        ...

    def get_documents(self) -> list[Document]:
        """List all documents in the knowledge base."""
        ...

    def get_chunks_for_entity(
        self,
        entity_name: str,
        *,
        limit: int = 20,
    ) -> list[Chunk]:
        """Get source chunks mentioning an entity."""
        ...

    def get_facts_for_entity(
        self,
        entity_name: str,
        *,
        as_subject: bool = True,
        as_object: bool = True,
    ) -> list[Fact]:
        """Get facts involving an entity."""
        ...

    def get_neighbors(
        self,
        entity_name: str,
        *,
        relationship_type: str | None = None,
        max_hops: int = 1,
    ) -> list[Entity]:
        """Get entities connected to the given entity."""
        ...

    # === Export/Import ===

    async def export_to_json(
        self,
        output_path: str | Path,
        *,
        include_embeddings: bool = False,
    ) -> None:
        """Export knowledge base to JSON format."""
        ...

    async def export_to_rdf(
        self,
        output_path: str | Path,
        *,
        format: str = "turtle",
    ) -> None:
        """Export knowledge base to RDF format."""
        ...

    @classmethod
    async def from_neo4j(
        cls,
        neo4j_uri: str,
        username: str,
        password: str,
        output_path: str | Path,
        *,
        group_id: str = "default",
    ) -> "KnowledgeGraph":
        """
        Migrate from a Neo4j-based knowledge graph.

        Exports all data from Neo4j and creates a new embedded
        knowledge base at the specified path.
        """
        ...
```

### KGShell Class

The `KGShell` class provides filesystem-style navigation of the knowledge graph, designed for both interactive exploration and agent-based querying.

```python
class KGShell:
    """
    Interactive shell for navigating a knowledge graph.

    Presents the KG as a virtual filesystem with familiar commands.
    Designed for both human exploration and LLM agent navigation.

    Example:
        shell = kg.shell()
        print(shell.pwd())  # /kg/
        print(shell.ls())   # entities/  topics/  chunks/  documents/  @search/

        shell.cd("entities/organizations/")
        print(shell.ls())   # Alphabet_Inc/  Apple_Inc/  ...

        shell.cd("Alphabet_Inc/")
        print(shell.cat("summary.txt"))  # Entity description
        print(shell.ls("relationships/"))  # ACQUIRED/  PARTNERED_WITH/  ...
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        """Initialize shell for a knowledge graph."""
        ...

    # === Navigation ===

    def pwd(self) -> str:
        """Print working directory."""
        ...

    def cd(self, path: str) -> str:
        """
        Change directory.

        Args:
            path: Absolute (/kg/entities/) or relative (../topics/) path

        Returns:
            New working directory path

        Raises:
            FileNotFoundError: If path doesn't exist
            NotADirectoryError: If path is a file
        """
        ...

    def ls(
        self,
        path: str | None = None,
        *,
        long: bool = False,
        all: bool = False,
        sort_by: str = "name",
    ) -> str:
        """
        List directory contents.

        Args:
            path: Path to list (default: current directory)
            long: Show detailed information (-l flag)
            all: Show hidden entries (-a flag)
            sort_by: Sort order: "name", "time", "size"

        Returns:
            Formatted directory listing
        """
        ...

    # === Content Access ===

    def cat(self, path: str) -> str:
        """
        Read file contents.

        Args:
            path: Path to file

        Returns:
            File contents as string
        """
        ...

    def head(self, path: str, lines: int = 10) -> str:
        """Return first N lines of a file."""
        ...

    def tail(self, path: str, lines: int = 10) -> str:
        """Return last N lines of a file."""
        ...

    # === Search ===

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        *,
        semantic: bool = True,
        case_insensitive: bool = True,
        context_lines: int = 0,
    ) -> str:
        """
        Search for pattern in files.

        Args:
            pattern: Search pattern (semantic query or regex)
            path: Path to search (default: current directory)
            semantic: Use semantic search (default) or literal match
            case_insensitive: Ignore case for literal matches
            context_lines: Lines of context around matches

        Returns:
            Formatted search results with scores and paths
        """
        ...

    def find(
        self,
        path: str | None = None,
        *,
        name: str | None = None,
        type: str | None = None,  # "d" for directory, "f" for file
        entity_type: str | None = None,
        has_relationship: str | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> str:
        """
        Find files matching criteria.

        Args:
            path: Starting path (default: current directory)
            name: Name pattern with wildcards (* and ?)
            type: Entry type: "d" (directory) or "f" (file)
            entity_type: Filter by entity type
            has_relationship: Filter entities with specific relationship
            date_range: Filter by date (ISO format tuple)

        Returns:
            List of matching paths
        """
        ...

    # === Statistics ===

    def wc(self, path: str | None = None) -> str:
        """
        Count items.

        Returns counts of entities, facts, chunks, etc.
        for the specified path or current directory.
        """
        ...

    # === Session Management ===

    def history(self, limit: int = 20) -> list[str]:
        """Get command history for this session."""
        ...

    def back(self) -> str:
        """Go to previous directory (cd -)."""
        ...

    # === Execution ===

    def execute(self, command: str) -> str:
        """
        Execute a shell command string.

        Parses and executes commands like "ls -l entities/" or
        "grep 'inflation' /kg/chunks/". Used by LLM agents.

        Args:
            command: Full command string

        Returns:
            Command output or error message
        """
        ...

    def execute_batch(self, commands: list[str]) -> list[str]:
        """Execute multiple commands, returning all outputs."""
        ...
```

### KGConfig Class

Configuration follows the principle of sensible defaults with full override capability.

```python
class KGConfig:
    """
    Configuration for ZommaKG.

    Configuration sources (in order of precedence):
    1. Explicit arguments to KGConfig()
    2. Environment variables (ZOMMA_* prefix)
    3. Config file (~/.zomma/config.toml or specified path)
    4. Built-in defaults

    Example:
        # Use defaults (reads from environment)
        kg = KnowledgeGraph("./kb")

        # Explicit configuration
        config = KGConfig(
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-5",
            embedding_provider="voyage",
            embedding_model="voyage-finance-2",
        )
        kg = KnowledgeGraph("./kb", config=config)

        # From config file
        config = KGConfig.from_file("./my_config.toml")
    """

    # === LLM Configuration ===

    llm_provider: str = "openai"
    """LLM provider: "openai", "anthropic", "google", "local" """

    llm_model: str = "gpt-4o"
    """Model for extraction and synthesis"""

    llm_model_fast: str = "gpt-4o-mini"
    """Model for quick operations (verification, critique)"""

    llm_model_cheap: str = "gpt-4o-mini"
    """Model for bulk operations (summary merging)"""

    # === Embedding Configuration ===

    embedding_provider: str = "openai"
    """Embedding provider: "openai", "voyage", "local" """

    embedding_model: str = "text-embedding-3-large"
    """Embedding model name"""

    embedding_dimensions: int = 3072
    """Embedding vector dimensions (provider-dependent)"""

    # === API Keys ===
    # These can be set via environment variables:
    # OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, VOYAGE_API_KEY

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    voyage_api_key: str | None = None

    # === Processing Configuration ===

    extraction_concurrency: int = 10
    """Max concurrent LLM extraction calls"""

    embedding_concurrency: int = 5
    """Max concurrent embedding batches"""

    embedding_batch_size: int = 100
    """Texts per embedding API call"""

    dedup_similarity_threshold: float = 0.70
    """Cosine similarity threshold for entity deduplication"""

    # === Query Configuration ===

    query_entity_threshold: float = 0.3
    """Minimum similarity for entity resolution in queries"""

    query_fact_threshold: float = 0.3
    """Minimum similarity for fact retrieval"""

    query_max_subqueries: int = 5
    """Maximum sub-queries for question decomposition"""

    query_enable_expansion: bool = True
    """Enable 1-hop neighbor expansion in queries"""

    # === Storage Configuration ===

    parquet_compression: str = "zstd"
    """Parquet compression: "zstd", "snappy", "gzip", "none" """

    lancedb_index_type: str = "IVF_PQ"
    """Vector index type for LanceDB"""

    # === Rust Acceleration ===

    use_rust_core: bool = True
    """Use Rust extensions if available (falls back to Python)"""

    # === Methods ===

    @classmethod
    def from_file(cls, path: str | Path) -> "KGConfig":
        """Load configuration from TOML file."""
        ...

    @classmethod
    def from_env(cls) -> "KGConfig":
        """Load configuration from environment variables."""
        ...

    def to_file(self, path: str | Path) -> None:
        """Save configuration to TOML file."""
        ...

    def with_overrides(self, **kwargs) -> "KGConfig":
        """Return new config with specified overrides."""
        ...
```

---

## Storage Architecture

### Knowledge Base Directory Structure

A knowledge base is a self-contained directory that can be moved, copied, zipped, and versioned.

```
my_knowledge_base/
├── metadata.json              # KB metadata and schema version
│
├── entities.parquet           # Extracted entities
├── chunks.parquet             # Source text chunks
├── facts.parquet              # Extracted facts/relationships
├── topics.parquet             # Topic classifications
├── documents.parquet          # Source document metadata
├── relationships.parquet      # All graph edges
│
├── lancedb/                   # Vector indices (LanceDB format)
│   ├── entities.lance/        # Entity name + summary embeddings
│   ├── facts.lance/           # Fact content embeddings
│   └── topics.lance/          # Topic definition embeddings
│
├── checkpoints/               # Ingestion checkpoints (optional)
│   └── [document_id].pkl      # Resume state for interrupted ingestion
│
└── cache/                     # Query cache (optional)
    └── [query_hash].json      # Cached query results
```

### Parquet Table Schemas

Each Parquet file stores one type of node or edge with a consistent schema.

#### entities.parquet

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key (UUID4) |
| name | STRING | Canonical entity name |
| summary | STRING | LLM-generated entity description |
| entity_type | STRING | PERSON, ORGANIZATION, LOCATION, CONCEPT, etc. |
| aliases | LIST[STRING] | Alternative names for the entity |
| created_at | TIMESTAMP | When entity was first extracted |
| updated_at | TIMESTAMP | When entity was last modified |

**Design note:** Embeddings are stored in LanceDB, not the Parquet file. This separates vector search concerns from relational queries and allows different embedding models without re-writing entity data.

#### chunks.parquet

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key (UUID4) |
| document_uuid | STRING | Foreign key to documents |
| content | STRING | Full text content |
| header_path | STRING | Breadcrumb path (e.g., "Section 1 > Overview") |
| position | INT32 | Order within document (0-indexed) |
| document_date | STRING | Date from parent document (ISO format) |
| created_at | TIMESTAMP | Extraction timestamp |

#### facts.parquet

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key (UUID4, also used as fact_id in relationships) |
| content | STRING | Fact statement text |
| subject_uuid | STRING | FK to entities (denormalized for fast lookup) |
| subject_name | STRING | Subject entity name (denormalized) |
| object_uuid | STRING | FK to entities or topics |
| object_name | STRING | Object entity/topic name (denormalized) |
| object_type | STRING | "entity" or "topic" |
| relationship_type | STRING | Normalized relationship type (e.g., "ACQUIRED") |
| date_context | STRING | Temporal context for the fact |
| created_at | TIMESTAMP | Extraction timestamp |

**Design note:** Denormalization of subject/object names enables efficient fact retrieval without joining to entities. The relationship_type is normalized from free-form LLM output (e.g., "acquired a majority stake in" -> "ACQUIRED").

#### relationships.parquet

| Column | Type | Description |
|--------|------|-------------|
| id | STRING | Primary key (UUID4) |
| from_uuid | STRING | Source node UUID |
| from_type | STRING | Source type: "entity", "chunk", "document", "topic" |
| to_uuid | STRING | Target node UUID |
| to_type | STRING | Target type |
| rel_type | STRING | Relationship type (e.g., "ACQUIRED", "CONTAINS_CHUNK") |
| fact_id | STRING | FK to facts (nullable, for fact-bearing edges) |
| description | STRING | Free-form relationship description |
| date_context | STRING | Temporal context |
| created_at | TIMESTAMP | Creation timestamp |

**Design rationale:** This table replaces Neo4j's graph edges. The `fact_id` join pattern enables chunk-centric provenance: given an entity, find chunks where it appears via relationships with matching fact_id values.

#### topics.parquet

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key (UUID4) |
| name | STRING | Canonical topic name |
| definition | STRING | Topic definition from ontology |
| parent_topic | STRING | FK to parent topic (nullable, for hierarchy) |
| created_at | TIMESTAMP | Creation timestamp |

#### documents.parquet

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key (UUID4) |
| name | STRING | Document name/title |
| document_date | STRING | Document date (ISO format) |
| source_path | STRING | Original file path (for provenance) |
| file_type | STRING | "pdf", "markdown", "text" |
| metadata | STRING | JSON-encoded additional metadata |
| created_at | TIMESTAMP | Ingestion timestamp |

### Vector Indices (LanceDB)

LanceDB stores embeddings with metadata for filtered vector search.

**entities.lance schema:**
```python
{
    "uuid": str,
    "name": str,
    "summary": str,
    "entity_type": str,
    "vector": list[float],  # embedding_dimensions from config
}
```

**facts.lance schema:**
```python
{
    "uuid": str,
    "content": str,
    "subject_name": str,
    "object_name": str,
    "relationship_type": str,
    "vector": list[float],
}
```

**topics.lance schema:**
```python
{
    "uuid": str,
    "name": str,
    "definition": str,
    "vector": list[float],
}
```

### Metadata and Versioning

The `metadata.json` file tracks knowledge base version and configuration.

```json
{
  "schema_version": "1.0.0",
  "created_at": "2026-01-15T10:30:00Z",
  "updated_at": "2026-01-28T14:22:00Z",
  "zomma_version": "0.3.0",
  "embedding_model": "text-embedding-3-large",
  "embedding_dimensions": 3072,
  "stats": {
    "documents": 42,
    "chunks": 1847,
    "entities": 523,
    "facts": 2891,
    "topics": 67
  }
}
```

**Schema migration:** When schema_version changes between library versions, migration functions in `storage/parquet/migrations.py` transform the data. Migrations are run automatically on first access with user confirmation for destructive changes.

---

## Dependency Strategy

### Core Dependencies (Minimal Required)

These dependencies are always installed. The goal is to keep this list as small as possible while providing full functionality.

```toml
dependencies = [
    # Storage
    "duckdb>=0.10",              # Relational queries on Parquet
    "lancedb>=0.6",              # Vector search
    "pyarrow>=14.0",             # Parquet I/O

    # Core utilities
    "pydantic>=2.0",             # Schema validation
    "pydantic-settings>=2.0",    # Configuration management

    # Async
    "anyio>=4.0",                # Async abstraction

    # Types
    "typing-extensions>=4.8",    # Backported typing features
]
```

**Notably absent from core:**
- No LLM clients (all providers are optional)
- No LangChain (reduces complexity, optional for advanced use)
- No heavy ML frameworks (torch, transformers)

**Consequence:** A knowledge base can be **queried** without any LLM provider installed, using pre-computed embeddings. Only **ingestion** and **synthesis** require LLM access.

### Provider Dependencies

Each LLM/embedding provider is an optional extra.

```toml
[project.optional-dependencies]
# LLM Providers
openai = [
    "openai>=1.0",
    "tiktoken>=0.5",          # Token counting
]

anthropic = [
    "anthropic>=0.18",
]

google = [
    "google-generativeai>=0.5",
]

# Embedding Providers
voyage = [
    "voyageai>=0.2",
]

local-embed = [
    "sentence-transformers>=2.2",
    "torch>=2.0",             # Required by sentence-transformers
]
```

### Performance Dependencies

Optional acceleration components.

```toml
[project.optional-dependencies]
# Rust acceleration (built via maturin)
rust = []  # No Python deps; Rust extension is compiled

# GPU support for local embeddings
gpu = [
    "torch>=2.0",
    "zomma-kg[local-embed]",
]
```

### Convenience Bundles

Pre-configured extras for common setups.

```toml
[project.optional-dependencies]
# All cloud providers
cloud = [
    "zomma-kg[openai,anthropic,google,voyage]",
]

# Everything except GPU
all = [
    "zomma-kg[cloud,rust]",
]

# Full installation
full = [
    "zomma-kg[all,gpu]",
]
```

### Lazy Import Strategy

To avoid import-time dependency on optional packages:

```python
# providers/llm/openai.py
def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI provider requires the 'openai' package. "
            "Install with: pip install zomma-kg[openai]"
        )
    return OpenAI()
```

This pattern:
- Defers import until actually needed
- Provides clear error messages for missing dependencies
- Allows the package to be imported even without all providers

---

## Configuration System

### Configuration Hierarchy

Configuration follows a layered approach where more specific settings override general ones.

**Priority (highest to lowest):**
1. Programmatic (passed to `KGConfig()`)
2. Environment variables (`ZOMMA_*` prefix)
3. Config file (`~/.zomma/config.toml` or `ZOMMA_CONFIG_FILE`)
4. Built-in defaults

### Environment Variables

All configuration options have corresponding environment variables.

| Config Option | Environment Variable | Example |
|--------------|---------------------|---------|
| `llm_provider` | `ZOMMA_LLM_PROVIDER` | `anthropic` |
| `llm_model` | `ZOMMA_LLM_MODEL` | `claude-sonnet-4-5` |
| `embedding_provider` | `ZOMMA_EMBEDDING_PROVIDER` | `voyage` |
| `openai_api_key` | `OPENAI_API_KEY` | `sk-...` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `extraction_concurrency` | `ZOMMA_EXTRACTION_CONCURRENCY` | `20` |

**API key precedence:** Provider-specific environment variables (e.g., `OPENAI_API_KEY`) are checked before `ZOMMA_*` variants for compatibility with existing setups.

### Config File Format

TOML format for human readability:

```toml
# ~/.zomma/config.toml

[llm]
provider = "anthropic"
model = "claude-sonnet-4-5"
model_fast = "claude-haiku-3-5"

[embedding]
provider = "voyage"
model = "voyage-finance-2"
dimensions = 1024

[processing]
extraction_concurrency = 15
embedding_batch_size = 50

[query]
entity_threshold = 0.25
max_subqueries = 7

[storage]
parquet_compression = "zstd"
```

### Programmatic Configuration

```python
from zomma_kg import KnowledgeGraph, KGConfig

# Method 1: Explicit configuration
config = KGConfig(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5",
    anthropic_api_key="sk-ant-...",  # Or set ANTHROPIC_API_KEY env var
)
kg = KnowledgeGraph("./my_kb", config=config)

# Method 2: Override specific settings
config = KGConfig.from_env()  # Load from environment
config = config.with_overrides(
    extraction_concurrency=20,
    query_max_subqueries=10,
)
kg = KnowledgeGraph("./my_kb", config=config)

# Method 3: From file with overrides
config = KGConfig.from_file("./project_config.toml")
config = config.with_overrides(llm_model="claude-opus-4-5")
```

### Model Selection and Defaults

Default models are chosen for balance of quality and cost.

| Use Case | Default Model | Rationale |
|----------|--------------|-----------|
| Extraction | gpt-4o | Best structured output |
| Critique | gpt-4o-mini | Fast, good enough for verification |
| Dedup verification | gpt-4o-mini | Many calls, cost-sensitive |
| Summary merging | gpt-4o-mini | Bulk operation |
| Query synthesis | gpt-4o | Quality matters for user-facing |
| Embeddings | text-embedding-3-large | Best retrieval quality |

**Provider-specific defaults:** When a provider is selected, appropriate model defaults are applied:

```python
# Selecting Anthropic as provider
config = KGConfig(llm_provider="anthropic")
# Automatically sets:
#   llm_model = "claude-sonnet-4-5"
#   llm_model_fast = "claude-haiku-3-5"
#   llm_model_cheap = "claude-haiku-3-5"
```

### Concurrency Settings

Concurrency defaults balance throughput with rate limits.

| Setting | Default | Description |
|---------|---------|-------------|
| `extraction_concurrency` | 10 | Parallel LLM extraction calls |
| `embedding_concurrency` | 5 | Parallel embedding API calls |
| `embedding_batch_size` | 100 | Texts per embedding call |
| `dedup_concurrency` | 10 | Parallel dedup verification calls |

**Rate limit handling:** Providers have different rate limits. When rate-limited:
1. Exponential backoff with jitter
2. Automatic retry up to 5 times
3. Warning logged if approaching limits frequently

---

## Plugin/Provider Architecture

### Provider Abstraction

Providers implement abstract interfaces for LLM and embedding operations.

```python
# providers/base.py

from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
    ) -> BaseModel:
        """Generate a structured response matching the schema."""
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model name."""
        ...


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensions."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model name."""
        ...
```

### LLM Provider Implementations

Each provider implements the abstract interface with provider-specific optimizations.

**OpenAI Provider:**
```python
# providers/llm/openai.py

class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ):
        self._model = model
        self._client = self._get_client(api_key)

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
    ) -> BaseModel:
        # Use OpenAI's response_format for structured output
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_schema", "schema": schema.model_json_schema()},
        )
        return schema.model_validate_json(response.choices[0].message.content)
```

**Anthropic Provider:**
```python
# providers/llm/anthropic.py

class AnthropicProvider(LLMProvider):
    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
    ) -> BaseModel:
        # Use Claude's tool use for structured output
        response = await self._client.messages.create(
            model=self._model,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "name": "structured_output",
                "description": "Output structured data",
                "input_schema": schema.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": "structured_output"},
        )
        return schema.model_validate(response.content[0].input)
```

**Local Provider (Ollama/llama.cpp):**
```python
# providers/llm/local.py

class LocalProvider(LLMProvider):
    """Provider for local LLM inference via Ollama or llama.cpp."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
    ) -> BaseModel:
        # Use JSON mode with schema in prompt
        schema_prompt = f"""
{prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}
"""
        response = await self.generate(schema_prompt, system=system)
        return schema.model_validate_json(response)
```

### Embedding Provider Implementations

```python
# providers/embedding/openai.py

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        api_key: str | None = None,
    ):
        self._model = model
        self._dimensions = dimensions
        self._client = self._get_client(api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]
```

```python
# providers/embedding/local.py

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embeddings via sentence-transformers."""

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Local embeddings require sentence-transformers. "
                "Install with: pip install zomma-kg[local-embed]"
            )
        self._model = SentenceTransformer(model)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # sentence-transformers is sync, run in thread pool
        import asyncio
        embeddings = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True
        )
        return embeddings.tolist()
```

### Storage Backend Abstraction

For future flexibility, storage operations go through an abstract interface.

```python
# storage/base.py

class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    async def write_entities(self, entities: list[Entity]) -> None: ...

    @abstractmethod
    async def write_chunks(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    async def write_facts(self, facts: list[Fact]) -> None: ...

    @abstractmethod
    async def get_entity(self, name: str) -> Entity | None: ...

    @abstractmethod
    async def search_entities(
        self,
        query_embedding: list[float],
        limit: int,
        threshold: float,
    ) -> list[EntityMatch]: ...

    @abstractmethod
    async def get_chunks_for_entity(
        self,
        entity_name: str,
        limit: int,
    ) -> list[Chunk]: ...

    # ... etc
```

**Current implementation:** `ParquetBackend` using DuckDB for relational queries and LanceDB for vector search.

**Future possibilities:**
- `SQLiteBackend` - For simpler single-file storage
- `PostgresBackend` - For server-based deployments
- `RemoteBackend` - For cloud-hosted knowledge bases

---

## CLI Interface

The package includes a command-line interface for common operations.

### Command Overview

```bash
# Ingestion
zomma-kg ingest document.pdf --kb ./my_kb
zomma-kg ingest ./docs/ --pattern "*.md" --kb ./my_kb

# Query
zomma-kg query "What were the key risks?" --kb ./my_kb
zomma-kg query --interactive --kb ./my_kb

# Shell
zomma-kg shell --kb ./my_kb

# Info
zomma-kg info --kb ./my_kb
zomma-kg info --kb ./my_kb --entities
zomma-kg info --kb ./my_kb --documents

# Export
zomma-kg export --format json --output ./export.json --kb ./my_kb
zomma-kg export --format rdf --output ./export.ttl --kb ./my_kb

# Migration
zomma-kg migrate-from-neo4j --uri bolt://... --user neo4j --kb ./my_kb
```

### zomma-kg ingest

```bash
zomma-kg ingest [OPTIONS] PATHS...

Ingest documents into a knowledge base.

Arguments:
  PATHS                 Files or directories to ingest

Options:
  --kb PATH             Knowledge base directory [required]
  --pattern TEXT        Glob pattern for directory ingestion [default: **/*.pdf]
  --date TEXT           Document date override (ISO format)
  --concurrency INT     Max concurrent extractions [default: 10]
  --resume              Resume interrupted ingestion
  --fresh               Force fresh start (ignore checkpoints)
  --verbose             Show detailed progress

Examples:
  # Ingest a single PDF
  zomma-kg ingest report.pdf --kb ./my_kb

  # Ingest all PDFs in a directory
  zomma-kg ingest ./documents/ --kb ./my_kb

  # Ingest markdown files
  zomma-kg ingest ./notes/ --pattern "**/*.md" --kb ./my_kb

  # Resume interrupted ingestion
  zomma-kg ingest ./large_corpus/ --kb ./my_kb --resume
```

### zomma-kg query

```bash
zomma-kg query [OPTIONS] [QUESTION]

Query a knowledge base.

Arguments:
  QUESTION              Natural language question (omit for interactive mode)

Options:
  --kb PATH             Knowledge base directory [required]
  --interactive, -i     Interactive query mode
  --shell               Use shell-based navigation for query
  --json                Output as JSON
  --sources             Include source citations
  --max-chunks INT      Maximum chunks to retrieve [default: 20]

Examples:
  # Single question
  zomma-kg query "What economic conditions were reported?" --kb ./my_kb

  # Interactive mode
  zomma-kg query -i --kb ./my_kb

  # JSON output for scripting
  zomma-kg query "List all companies mentioned" --kb ./my_kb --json

  # With shell-based exploration
  zomma-kg query --shell "How are Apple and Google connected?" --kb ./my_kb
```

### zomma-kg shell

```bash
zomma-kg shell [OPTIONS]

Interactive shell for knowledge base navigation.

Options:
  --kb PATH             Knowledge base directory [required]

Commands available in shell:
  pwd                   Print working directory
  cd PATH               Change directory
  ls [PATH]             List directory contents
  cat FILE              Read file contents
  grep PATTERN [PATH]   Search for pattern
  find [OPTIONS]        Find files matching criteria
  wc [PATH]             Count items
  help                  Show command help
  exit                  Exit shell

Example session:
  $ zomma-kg shell --kb ./my_kb

  /kg$ ls
  entities/  topics/  chunks/  documents/  @search/

  /kg$ cd entities/organizations/
  /kg/entities/organizations$ ls
  Alphabet_Inc/  Apple_Inc/  Microsoft/  ...

  /kg/entities/organizations$ cd Alphabet_Inc/
  /kg/entities/organizations/Alphabet_Inc$ cat summary.txt
  Alphabet Inc. is a multinational technology conglomerate...

  /kg/entities/organizations/Alphabet_Inc$ grep "acquisition"
  facts/ACQUIRED/DeepMind.json: [0.92] Acquired DeepMind in 2014...
```

### zomma-kg info

```bash
zomma-kg info [OPTIONS]

Display knowledge base information.

Options:
  --kb PATH             Knowledge base directory [required]
  --entities            List all entities
  --documents           List all documents
  --topics              List all topics
  --stats               Show detailed statistics
  --json                Output as JSON

Examples:
  # Summary statistics
  zomma-kg info --kb ./my_kb

  Knowledge Base: ./my_kb
  Created: 2026-01-15
  Updated: 2026-01-28

  Statistics:
    Documents: 42
    Chunks: 1,847
    Entities: 523 (312 organizations, 156 people, 55 locations)
    Facts: 2,891
    Topics: 67

  # List all entities
  zomma-kg info --kb ./my_kb --entities

  # JSON output
  zomma-kg info --kb ./my_kb --json --stats
```

---

## Distribution Strategy

### PyPI Package Structure

The package is distributed as a standard Python package with optional native extensions.

**Package metadata (pyproject.toml):**
```toml
[project]
name = "zomma-kg"
version = "0.3.0"
description = "Embedded knowledge graph library for document understanding"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
authors = [{name = "Zomma Labs"}]
keywords = [
    "knowledge-graph",
    "rag",
    "llm",
    "financial-documents",
    "document-understanding",
    "embedded-database",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Database :: Database Engines/Servers",
    "Typing :: Typed",
]

[project.scripts]
zomma-kg = "zomma_kg.cli:main"

[project.urls]
Homepage = "https://github.com/Zomma-Labs/zomma-kg"
Documentation = "https://zomma-kg.readthedocs.io"
Repository = "https://github.com/Zomma-Labs/zomma-kg"
Issues = "https://github.com/Zomma-Labs/zomma-kg/issues"
```

### Wheel Building

**Pure Python wheels:** Built for any platform.

**Native wheels (with Rust):** Built separately for each platform using maturin and GitHub Actions.

```yaml
# .github/workflows/release.yml
jobs:
  build-wheels:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          args: --release --out dist

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python }}
          path: dist/
```

**Distribution strategy:**
1. Pure Python wheels always available
2. Native wheels for common platforms (Linux x86_64, macOS ARM/Intel, Windows x86_64)
3. Source distribution for other platforms (requires Rust toolchain)
4. Fallback to pure Python if native extension unavailable

### Version Compatibility

**Python version support:**
- Minimum: Python 3.10 (for modern typing, match statements)
- Tested: Python 3.10, 3.11, 3.12
- Policy: Support latest 3 Python minor versions

**Dependency version policy:**
- Specify minimum versions in requirements
- Use `>=` not `==` for flexibility
- Test against both minimum and latest versions in CI

**Semantic versioning:**
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Platform Support

| Platform | Pure Python | Native Extension | Notes |
|----------|-------------|------------------|-------|
| Linux x86_64 | Yes | Yes | Primary development platform |
| Linux ARM64 | Yes | Yes | AWS Graviton, etc. |
| macOS x86_64 | Yes | Yes | Intel Macs |
| macOS ARM64 | Yes | Yes | Apple Silicon |
| Windows x86_64 | Yes | Yes | WSL recommended for dev |
| Windows ARM64 | Yes | Source only | Rare platform |

---

## Migration Path

### From Current Neo4j-Based System

Users with existing Neo4j-based ZommaLabsKG deployments can migrate to the embedded architecture.

**Migration command:**
```bash
zomma-kg migrate-from-neo4j \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password secret \
    --group-id default \
    --kb ./migrated_kb
```

**Migration process:**
1. Connect to Neo4j instance
2. Export all nodes (EntityNode, EpisodicNode, FactNode, TopicNode, DocumentNode)
3. Export all relationships with properties
4. Transform to Parquet schema
5. Re-index vectors in LanceDB
6. Validate data integrity

**Code migration:**
```python
# Before (Neo4j-based)
from zomma_kg import Pipeline, QueryEngine

pipeline = Pipeline()
await pipeline.ingest_file("document.pdf")

engine = QueryEngine(group_id="default")
result = await engine.query("What risks were mentioned?")

# After (embedded)
from zomma_kg import KnowledgeGraph

kg = KnowledgeGraph("./my_kb")
await kg.ingest_pdf("document.pdf")
result = await kg.query("What risks were mentioned?")
```

**API mapping:**

| Old API | New API |
|---------|---------|
| `Pipeline.ingest_file()` | `KnowledgeGraph.ingest_pdf()` |
| `Pipeline.ingest_chunks()` | `KnowledgeGraph.ingest_chunks()` |
| `QueryEngine.query()` | `KnowledgeGraph.query()` |
| `Config` class | `KGConfig` class |
| Environment variables | Same, plus `ZOMMA_*` variants |

### Data Export/Import Utilities

**Export from embedded KB:**
```python
kg = KnowledgeGraph("./my_kb")

# To JSON (portable, human-readable)
await kg.export_to_json("./export.json")

# To RDF (interoperable)
await kg.export_to_rdf("./export.ttl", format="turtle")
```

**Import formats:**
- JSON export from other KG systems
- CSV with entity/relationship schemas
- RDF graphs (Turtle, N-Triples, RDF/XML)

### Backwards Compatibility

**Deprecation policy:**
1. Deprecated APIs remain functional for 2 minor versions
2. Deprecation warnings emitted on use
3. Migration guide in changelog

**Legacy module:**
```python
# zomma_kg/_compat/legacy.py

# Import old API names for backwards compatibility
from zomma_kg import KnowledgeGraph as Pipeline
from zomma_kg import KnowledgeGraph

class QueryEngine:
    """Legacy compatibility wrapper."""

    def __init__(self, group_id: str = "default", **kwargs):
        import warnings
        warnings.warn(
            "QueryEngine is deprecated. Use KnowledgeGraph.query() instead.",
            DeprecationWarning,
        )
        # Would need KB path in real implementation
        ...
```

---

## Summary

This document describes the design for transforming ZommaLabsKG into a pip-installable Python package with zero infrastructure requirements. Key architectural decisions:

1. **Embedded storage** using DuckDB + LanceDB + Parquet files
2. **Portable knowledge bases** as self-contained directories
3. **Provider-agnostic architecture** supporting multiple LLM/embedding providers
4. **Optional Rust acceleration** for performance-critical operations
5. **Filesystem navigation interface** for agent-based exploration
6. **Progressive complexity** with sensible defaults and full configurability

The design prioritizes:
- **Developer experience**: Simple installation, intuitive API, helpful errors
- **Portability**: Knowledge bases are just directories
- **Flexibility**: Multiple providers, optional acceleration, extensible architecture
- **Migration path**: Clear path from Neo4j-based system

Implementation should proceed in phases:
1. Core storage layer (Parquet + LanceDB)
2. Public API (KnowledgeGraph class)
3. Ingestion pipeline migration
4. Query pipeline migration
5. Shell/navigation interface
6. CLI implementation
7. Rust acceleration (optional)
8. Migration utilities

The result will be a knowledge graph library that can be installed and used with a single `pip install` command, enabling developers to build sophisticated document understanding systems without infrastructure complexity.
