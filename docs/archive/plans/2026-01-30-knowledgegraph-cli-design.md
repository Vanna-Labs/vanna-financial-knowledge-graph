# KnowledgeGraph & CLI Implementation Design

**Date:** 2026-01-30
**Status:** Approved

## Overview

Implement the `KnowledgeGraph` facade class and Typer-based CLI to expose the existing internal components (ingestion pipeline, query pipeline, storage) through a clean public API.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Provider initialization | Lazy | Faster startup, lower memory if not all features used |
| CLI library | Typer | Type-hint based, less boilerplate, modern API |
| Async bridging | `asyncio.run()` per command | Simple, predictable, standard pattern |
| CLI output | Rich console | Progress bars, colors, formatted tables |

## Architecture

### KnowledgeGraph Class

```
KnowledgeGraph
├── _path: Path
├── _config: KGConfig
├── _storage: ParquetBackend | None (lazy)
├── _llm: LLMProvider | None (lazy)
├── _embeddings: EmbeddingProvider | None (lazy)
├── _query_pipeline: GraphRAGPipeline | None (lazy)
└── _initialized: bool
```

**Lazy initialization pattern:**
- All providers/storage are `None` until first async operation
- `_ensure_initialized()` called at start of each public async method
- Directory created on first use if `create=True`

### Ingestion Flow

```
ingest_pdf(path)
    │
    ├── 1. chunk_pdf() ─────────────────► list[ChunkInput]
    │
    ├── 2. extract_from_chunks() ───────► list[ChainOfThoughtResult]
    │
    ├── 3. deduplicate_entities() ──────► EntityDeduplicationOutput
    │
    ├── 4. asyncio.gather() ────────────┬─► EntityResolutionResult
    │   ├── EntityRegistry.resolve()    │
    │   └── TopicResolver.resolve() ────┴─► TopicResolutionResult
    │
    └── 5. Assembler.assemble() ────────► AssemblyResult
```

**Key points:**
- Entity and topic resolution run in parallel (independent operations)
- UUID remapping applied before assembly
- Progress callback supported for UI feedback

### Query Flow

```
query(question)
    │
    └── GraphRAGPipeline.query()
        ├── 1. QueryDecomposer.decompose()
        ├── 2. Researcher.research() (per sub-query, parallel)
        ├── 3. ContextBuilder.build()
        └── 4. Synthesizer.synthesize_final_answer()
```

**Key points:**
- Pipeline instance cached for reuse across queries
- Internal `PipelineResult` mapped to public `QueryResult` type

### CLI Commands

| Command | Description |
|---------|-------------|
| `vanna-kg ingest <path>` | Ingest PDF/markdown file or directory |
| `vanna-kg query "question"` | Query the knowledge base |
| `vanna-kg info` | Display KB statistics |
| `vanna-kg shell` | Interactive navigation (placeholder) |

**Common options:**
- `--kb, -k PATH` - Knowledge base directory (default: `./kb`)

## Files to Modify

### 1. `vanna_kg/api/knowledge_graph.py`

Full implementation replacing stubs:

```python
class KnowledgeGraph:
    def __init__(self, path, config=None, create=True):
        self._path = Path(path)
        self._config = config or KGConfig()
        self._create = create
        self._storage = None
        self._llm = None
        self._embeddings = None
        self._query_pipeline = None
        self._initialized = False

    async def _ensure_initialized(self):
        if self._initialized:
            return
        if self._create:
            self._path.mkdir(parents=True, exist_ok=True)
        self._storage = ParquetBackend(self._path, self._config)
        await self._storage.initialize()
        self._llm = self._create_llm_provider()
        self._embeddings = self._create_embedding_provider()
        self._initialized = True

    async def ingest_pdf(self, path, *, document_date=None, metadata=None, on_progress=None):
        await self._ensure_initialized()
        # Chunking → Extraction → Dedup → Resolution (parallel) → Assembly
        ...

    async def query(self, question, *, include_sources=True, **kwargs):
        await self._ensure_initialized()
        if self._query_pipeline is None:
            self._query_pipeline = GraphRAGPipeline(...)
        result = await self._query_pipeline.query(question, include_sources=include_sources)
        return QueryResult(...)
```

### 2. `vanna_kg/cli/__init__.py`

Typer-based CLI:

```python
import typer
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(name="vanna-kg")
console = Console()

@app.command()
def ingest(
    path: Path = typer.Argument(...),
    kb: Path = typer.Option(Path("./kb"), "--kb", "-k"),
    pattern: str = typer.Option("**/*.pdf", "--pattern", "-p"),
):
    """Ingest documents into a knowledge base."""
    import asyncio
    asyncio.run(_ingest(path, kb, pattern))

@app.command()
def query(
    question: str = typer.Argument(...),
    kb: Path = typer.Option(Path("./kb"), "--kb", "-k"),
):
    """Query the knowledge base."""
    import asyncio
    asyncio.run(_query(question, kb))

@app.command()
def info(kb: Path = typer.Option(Path("./kb"), "--kb", "-k")):
    """Display knowledge base statistics."""
    import asyncio
    asyncio.run(_info(kb))

def main():
    app()
```

### 3. `pyproject.toml`

Add dependency and entry point:

```toml
[project.dependencies]
typer = {version = ">=0.9.0", extras = ["all"]}

[project.scripts]
vanna-kg = "vanna_kg.cli:main"
```

## Implementation Tasks

1. **Implement KnowledgeGraph.__init__ and _ensure_initialized**
   - Lazy provider/storage initialization
   - Directory creation logic

2. **Implement provider factory methods**
   - `_create_llm_provider()` - based on config.llm_provider
   - `_create_embedding_provider()` - based on config.embedding_provider

3. **Implement ingestion methods**
   - `ingest_pdf()` - full pipeline
   - `ingest_markdown()` - skip PDF conversion
   - `ingest_text()` - skip chunking
   - `ingest_directory()` - batch processing

4. **Implement query methods**
   - `query()` - full GraphRAG pipeline
   - `search_entities()`, `search_facts()`, `search_chunks()` - direct vector search

5. **Implement data access methods**
   - `get_entity()`, `get_entities()`, `get_document()`, etc.
   - Delegate to storage backend

6. **Implement lifecycle methods**
   - `close()` / `close_sync()` - cleanup resources
   - Context manager support

7. **Implement CLI commands**
   - `ingest` with progress bar
   - `query` with formatted output
   - `info` with stats table

8. **Add typer dependency and entry point**
   - Update pyproject.toml

## Testing Strategy

- Unit tests for KnowledgeGraph methods (mock storage/providers)
- Integration test: ingest a small PDF, query it
- CLI tests using typer.testing.CliRunner
