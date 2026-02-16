# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VannaKG is a pip-installable Python library for building embedded knowledge graphs from documents with zero infrastructure requirements. It transforms documents into queryable knowledge structures using DuckDB, LanceDB, and Parquet files.

**Key characteristics:**
- Zero infrastructure: `pip install vanna-kg` and start immediately
- Knowledge bases are portable directories (can be zipped and shared)
- Full offline support with no database server required
- Multi-tenant capable with group_id isolation
- Primary use case: Financial document analysis and entity extraction

## Build and Development Commands

```bash
# Install in development mode
uv pip install -e ".[dev,all]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_entity_dedup.py

# Run single test
pytest tests/test_entity_dedup.py::test_specific_function -v

# Format code
ruff format vanna_kg tests

# Lint code
ruff check vanna_kg tests

# Type check (strict mode)
mypy vanna_kg

# Run CLI
python -m vanna_kg
```

## Architecture

### Three-Phase Ingestion Pipeline

```
PDF/Markdown → Chunking → LLM Extraction → Resolution → Assembly → Storage
```

1. **Chunking & Extraction** (`ingestion/chunking/`, `ingestion/extraction/`): PDF → Markdown → Chunks with chain-of-thought LLM extraction
2. **Resolution** (`ingestion/resolution/`): In-document dedup + cross-document entity matching + topic resolution
3. **Assembly** (`ingestion/assembly/`): Batch embedding and write to Parquet + LanceDB

### GraphRAG Query Pipeline (`query/`)

```
Question → Decomposer → Researcher → Context Builder → Synthesizer → Answer
```

- `decomposer.py`: Break questions into sub-queries
- `researcher.py`: Parallel retrieval with semaphore-bounded concurrency
- `context_builder.py`: Deduplicate and organize results
- `synthesizer.py`: Question-type-aware answer generation

### Storage Layer (`storage/`)

- **Parquet files** (relational): entities, chunks, facts, topics, documents, relationships
- **LanceDB** (vectors): entity/fact/topic embeddings for similarity search
- **DuckDB** (query layer): SQL queries over Parquet files

### Key Modules

| Module | Purpose |
|--------|---------|
| `api/knowledge_graph.py` | Main `KnowledgeGraph` facade class |
| `api/shell.py` | `KGShell` virtual filesystem interface |
| `config/settings.py` | `KGConfig` class (env + TOML + programmatic) |
| `types/` | 31 Pydantic models for all data structures |
| `providers/` | Abstract LLM/embedding provider interfaces |
| `mcp/server.py` | Model Context Protocol server with `kg_execute` tool |

### Provider Abstraction

```python
# providers/base.py defines abstract interfaces
class LLMProvider(ABC):
    async def generate(self, prompt: str, ...) -> str
    async def generate_structured(self, prompt: str, schema: type[BaseModel], ...) -> BaseModel

class EmbeddingProvider(ABC):
    async def embed(self, texts: list[str]) -> list[list[float]]
```

Currently OpenAI is fully implemented; Anthropic/Google are stubs.

## Key Patterns

### Lazy Initialization
Components initialize on first use via `_ensure_initialized()` pattern in `KnowledgeGraph` class.

### Async-First with Sync Wrappers
All core methods are async (`kg.query()`), with sync wrappers (`kg.query_sync()`).

### Chunk-Centric Provenance
Facts link entities to source chunks via `relationships.parquet` with `fact_id` join pattern.

### Multi-Tenancy
All queries filtered by `group_id`. Safe character validation for group IDs.

## Configuration

Environment variables (prefix `VANNA_`):
- `VANNA_LLM_PROVIDER` / `VANNA_LLM_MODEL`
- `VANNA_EMBEDDING_PROVIDER`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`

## Code Style

- Line length: 100 characters
- Python 3.10+ features (match statements, modern typing)
- Strict MyPy type checking enabled
- Ruff rules: E, F, I, N, W, UP (error, undefined, import, naming, warning, upgrade)
