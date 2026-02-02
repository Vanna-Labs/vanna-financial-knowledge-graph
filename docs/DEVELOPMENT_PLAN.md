# ZommaKG Development Plan

## Executive Summary

This document outlines a phased development plan for implementing the ZommaKG Python package - an embedded knowledge graph library with zero infrastructure requirements. The package transforms the existing Neo4j-based system into a pip-installable library using DuckDB, LanceDB, and Parquet for storage.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    KnowledgeGraph API                       │
├─────────────────────────────────────────────────────────────┤
│  Ingestion Pipeline  │  Query Pipeline  │  Shell Interface  │
├──────────────────────┴──────────────────┴───────────────────┤
│                     Storage Layer                           │
│         DuckDB (SQL) + LanceDB (Vectors) + Parquet         │
├─────────────────────────────────────────────────────────────┤
│              Providers (LLM + Embedding)                    │
├─────────────────────────────────────────────────────────────┤
│           Types + Config + Utils (Foundation)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (Week 1-2)

**Goal**: Establish core infrastructure that all other components depend on.

### 1.1 Type System (`types/`)

**Status**: ✅ COMPLETE

**Files implemented**:
- `types/entities.py` - Entity, EntityType, EnumeratedEntity, EntityMatchDecision, EntityGroup, EntityResolution ✅
- `types/facts.py` - Fact, ExtractedFact ✅
- `types/chunks.py` - Chunk, Document, DocumentPayload, ChunkInput ✅
- `types/topics.py` - Topic, TopicResolution, TopicDefinition, BatchTopicDefinitions ✅
- `types/results.py` - QueryResult, IngestResult, SearchResult, ExtractionResult, CritiqueResult, ChainOfThoughtResult, EntityDedupeResult, QuestionType, EntityHint, SubQuery, QueryDecomposition, DateExtraction, CanonicalEntity, EntityDeduplicationOutput, MergeRecord ✅

**Ported from**: `ZommaLabsKG/zomma_kg/schemas/extraction.py`, `query/shared_schemas.py`, `query/schemas.py`

**Deliverable**: 31 Pydantic models with full type hints, validated with mypy. ✅

---

### 1.2 Configuration System (`config/`)

**Status**: ✅ COMPLETE

**Files implemented**:
- `config/settings.py` - KGConfig class with env + TOML file + programmatic overrides ✅

**Key features**:
- Environment variables: `ZOMMA_*` prefix ✅
- TOML config file support: `from_file()` and `to_file()` ✅
- Programmatic overrides via constructor kwargs ✅
- Python 3.10 (tomli) and 3.11+ (tomllib) support ✅

**Deliverable**: KGConfig loading from all three sources with proper precedence. ✅

---

### 1.3 Provider Abstraction (`providers/`)

**Status**: ✅ COMPLETE

**Files implemented**:
- `providers/base.py` - LLMProvider, EmbeddingProvider ABCs ✅
- `providers/llm/openai.py` - OpenAILLMProvider (LangChain-based) ✅
- `providers/embedding/openai.py` - OpenAIEmbeddingProvider (LangChain-based) ✅

**Features**:
- `generate()`, `generate_structured()`, `stream()` for LLM ✅
- `embed()`, `embed_single()` for embeddings ✅
- `with_model()` for easy model switching ✅
- Lazy imports via `__getattr__` in `__init__.py` ✅
- Default model: `gpt-5.1` (configurable) ✅

**Deliverable**: OpenAI provider implementations (minimum viable). ✅

---

## Phase 2: Storage Layer (Week 2-3)

**Status**: ✅ COMPLETE

**Goal**: Implement embedded storage replacing Neo4j.

### 2.1 Storage Base (`storage/base.py`)

**Create abstract interface**:
```python
class StorageBackend(ABC):
    async def write_entities(self, entities: list[Entity]) -> None: ...
    async def write_chunks(self, chunks: list[Chunk]) -> None: ...
    async def write_facts(self, facts: list[Fact]) -> None: ...
    async def get_entity(self, name: str) -> Entity | None: ...
    async def get_chunks_for_entity(self, name: str) -> list[Chunk]: ...
    # ... etc
```

---

### 2.2 Parquet Backend (`storage/parquet/`)

**Files to implement**:
- `backend.py` - ParquetBackend class
- `tables.py` - Table schemas and PyArrow operations
- `migrations.py` - Schema versioning

**Table schemas**:

| Table | Columns |
|-------|---------|
| `entities.parquet` | uuid, name, summary, entity_type, aliases, created_at, updated_at |
| `chunks.parquet` | uuid, document_uuid, content, header_path, position, document_date |
| `facts.parquet` | uuid, content, subject_uuid, subject_name, object_uuid, object_name, relationship_type, date_context |
| `relationships.parquet` | id, from_uuid, from_type, to_uuid, to_type, rel_type, fact_id, description |
| `topics.parquet` | uuid, name, definition, parent_topic |
| `documents.parquet` | uuid, name, document_date, source_path, file_type, metadata |

**Deliverable**: ParquetBackend with CRUD operations for all tables.

---

### 2.3 LanceDB Integration (`storage/lancedb/`)

**Files to implement**:
- `indices.py` - Vector index management

**Vector indices**:
- `entities.lance` - Entity name + summary embeddings
- `facts.lance` - Fact content embeddings
- `topics.lance` - Topic definition embeddings

**Operations**:
- `index_entities(entities, embeddings)` - Add/update entity vectors
- `search_entities(query_vector, limit, threshold)` - Similarity search
- `search_facts(query_vector, limit, threshold)` - Fact search

**Deliverable**: LanceDB wrapper with filtered vector search.

---

### 2.4 DuckDB Query Layer (`storage/duckdb/`)

**Files to implement**:
- `queries.py` - SQL query implementations

**Query translations from Neo4j Cypher**:

| Query | SQL Equivalent |
|-------|----------------|
| Entity lookup | `SELECT * FROM entities WHERE name = ?` |
| Entity chunks | `JOIN relationships ON from_uuid = entity.uuid JOIN chunks ON ...` |
| 1-hop neighbors | Self-join on relationships via fact_id |
| Fact retrieval | LanceDB search + DuckDB join |

**Deliverable**: All query patterns from QUERYING_SYSTEM.md implemented in SQL.

---

## Phase 3: Ingestion Pipeline (Week 3-5)

**Status**: ✅ COMPLETE

**Goal**: Port 3-phase ingestion system to embedded storage.

### 3.1 Chunking System (`ingestion/chunking/`)

**Files to implement**:
- `pdf.py` - PDF → Markdown (Gemini vision)
- `markdown.py` - Markdown → Chunks (header-aware)
- `text.py` - Plain text chunking

**Port from**: `ZommaLabsKG/zomma_kg/chunker/`

**Algorithm**:
1. PDF → Markdown via Gemini 2.5 Pro vision
2. Parse markdown line-by-line tracking header stack
3. Split on paragraph boundaries (blank lines)
4. Keep HTML tables atomic
5. Filter chunks < 50 characters

**Deliverable**: Complete chunking with PDF, markdown, text support.

---

### 3.2 Extraction System (`ingestion/extraction/`)

**Files to implement**:
- `extractor.py` - Chain-of-thought extraction
- `critique.py` - Quality assessment
- `schemas.py` - Extraction Pydantic schemas

**Port from**: `ZommaLabsKG/zomma_kg/pipeline/extractor.py`

**Two-step extraction**:
1. **Entity Enumeration**: Extract ALL entities (name, type, summary)
2. **Relationship Generation**: Generate facts using ONLY enumerated entities

**Critique/Reflexion**: LLM verifies extraction, optional re-extraction on failure.

**Deliverable**: Extractor with critique loop.

---

### 3.3 Resolution System (`ingestion/resolution/`)

**Status**: ✅ COMPLETE

**Files implemented**:
- `entity_dedup.py` - In-document deduplication ✅ COMPLETE
- `entity_registry.py` - Cross-document matching ✅ COMPLETE
- `topic_resolver.py` - Topic ontology resolution ✅ COMPLETE

**Port from**:
- `ZommaLabsKG/zomma_kg/pipeline/entity_dedup.py`
- `ZommaLabsKG/zomma_kg/pipeline/entity_registry.py`

**Deduplication algorithm** (implemented in `entity_dedup.py`):
1. Generate embeddings (via embedding provider) ✅
2. Compute similarity matrix using scipy.spatial.distance.cdist ✅
3. Build edges where similarity >= threshold (default 0.70) ✅
4. Union-Find to find connected components ✅
5. Greedy BFS ordering for batch coherence ✅
6. Overlapping batches for large clusters (>15 entities) ✅
7. LLM verification with subsidiary awareness ✅
8. Build UUID remapping and merge history ✅

**Output types** (added to `types/results.py`):
- `CanonicalEntity` - Deduplicated entity with UUID, aliases, source indices
- `EntityDeduplicationOutput` - Full result with canonical entities and index mapping
- `MergeRecord` - Audit trail of merge decisions

**Tests**: 31 tests in `tests/test_entity_dedup.py` and `tests/test_types.py`

**Deliverable**: Full deduplication pipeline.

**Cross-document resolution** (implemented in `entity_registry.py`):
1. Embed each entity as `{name}: {summary}` ✅
2. Search LanceDB for top 25 candidates (threshold 0.50) ✅
3. LLM verification with subsidiary awareness (gpt-5-mini) ✅
4. Summary merging via LLM on match ✅
5. Return uuid_remap and summary_updates ✅
6. Parallel processing with configurable concurrency (default 10) ✅
7. Error handling with fallback to "new entity" on LLM failure ✅

**Output types** (added to `types/results.py`):
- `EntityRegistryMatch` - LLM match decision with confidence
- `EntityResolutionResult` - new_entities, uuid_remap, summary_updates

**Tests**: 17 tests in `tests/test_entity_registry.py`

**Topic ontology resolution** (implemented in `topic_resolver.py`):
1. Load curated ontology from JSON (`financial_topics.json`, 232 topics) ✅
2. Hash-based change detection for lazy reload ✅
3. Generate embeddings as `{label}: {definition}` + `{synonym}: {definition}` ✅
4. Vector search against ontology in LanceDB ✅
5. Batched LLM verification (~10 per call) with bounded concurrency ✅
6. Dictionary lookup by topic name for robust decision matching ✅
7. Collect unmatched topics in `new_topics` (toggleable) ✅

**Output types** (added to `types/topics.py`):
- `TopicResolutionResult` - resolved_topics, uuid_remap, new_topics
- `TopicMatchDecision` - LLM decision for single topic match
- `BatchTopicMatchResponse` - Batched LLM response

**Tests**: 29 tests in `tests/test_topic_resolver.py`

---

### 3.4 Assembly System (`ingestion/assembly/`)

**Status**: ✅ COMPLETE

**Files implemented**:
- `assembler.py` - Write to storage ✅ COMPLETE

**Port from**: `ZommaLabsKG/zomma_kg/pipeline/bulk_writer.py`

**Write order** (implemented):
1. Documents ✅
2. Chunks (reference document) ✅
3. Entities with embeddings ✅
4. Facts with embeddings ✅
5. Topics with embeddings ✅
6. Relationships (chunk-centric fact pattern) ✅

**Features**:
- Parallel embedding generation via `asyncio.gather()` ✅
- Embedding count validation ✅
- Error handling with progress logging ✅
- Unique relationship IDs ✅

**Output types** (added to `types/results.py`):
- `AssemblyInput` - document, chunks, entities, facts, topics
- `AssemblyResult` - counts of items written

**Tests**: 21 tests in `tests/test_assembler.py`

**Key change**: Write to Parquet + LanceDB instead of Neo4j.

**Deliverable**: Assembler writing to embedded storage. ✅

---

## Phase 4: Query Pipeline (Week 5-6)

**Status**: ✅ COMPLETE

**Goal**: Port V7 GraphRAG query system.

### 4.1 Query Pipeline (`query/`)

**Files implemented**:
- `pipeline.py` - V7 orchestrator (GraphRAGPipeline) ✅
- `decomposer.py` - Question decomposition ✅
- `researcher.py` - Per-subquery research ✅
- `synthesizer.py` - Answer synthesis ✅
- `context_builder.py` - Context assembly ✅
- `types.py` - Pipeline types (PipelineResult, SubAnswer, StructuredContext) ✅

**Port from**: `ZommaLabsKG/zomma_kg/query/`

**Pipeline phases**:
1. **Decompose**: Break question into sub-queries with entity/topic hints ✅
2. **Research**: Parallel retrieval for each sub-query (semaphore-bounded) ✅
3. **Assemble**: Dedupe, filter by relevance, order context ✅
4. **Synthesize**: Question-type-aware answer generation ✅

**Features**:
- Parallel sub-query research with configurable concurrency ✅
- Resolution caching across sub-queries ✅
- Comprehensive timing metrics ✅
- Graceful error handling with fallbacks ✅

**Tests**: `tests/test_query_pipeline.py`

**Key change**: Replace Cypher with DuckDB + LanceDB queries.

**Deliverable**: Complete V7 query pipeline. ✅

---

## Phase 5: Agent Interface (Week 6-7)

**Status**: ✅ COMPLETE (MCP Server approach)

**Goal**: Provide LLM agents with knowledge graph query capabilities.

### 5.1 MCP Server (`mcp/`)

**Approach changed**: Instead of a virtual filesystem shell, we now expose a single
MCP tool (`kg_execute`) that accepts command strings. This aligns with the skill
definition in `zomma_kg/skills/kg-query/SKILL.md`.

**Files implemented**:
- `mcp/__init__.py` - Module init ✅
- `mcp/server.py` - MCP server with `kg_execute` tool ✅
- `mcp/__main__.py` - Entry point for `python -m zomma_kg.mcp` ✅

**Commands (via kg_execute tool)**:
| Command | Purpose | Status |
|---------|---------|--------|
| `find` | Resolve names → canonical entities/topics | ✅ Done |
| `search` | Find connections between nodes | ✅ Done |
| `cat` | Expand fact details (by result number) | ✅ Done |
| `info` | Entity/topic summary | ✅ Done |
| `ls` | Browse entities, topics, documents | ✅ Done |
| `stats` | Knowledge base statistics | ✅ Done |

**Workflow**: Agents use the documented `find → search → cat` pattern.
Session state maintains search results so `cat 1` references previous search.

**Usage**:
```bash
# Run MCP server
python -m zomma_kg.mcp --kb ./my_kb

# Claude Desktop config
{
    "mcpServers": {
        "zomma-kg": {
            "command": "python",
            "args": ["-m", "zomma_kg.mcp", "--kb", "./my_kb"]
        }
    }
}
```

**Deliverable**: MCP server with kg_execute tool. ✅

### 5.2 Legacy Shell (Deprecated)

The `api/shell.py` KGShell class is deprecated in favor of the MCP approach.
It may be removed in a future version.

---

## Phase 6: Public API (Week 7-8)

**Status**: ✅ COMPLETE

**Goal**: Implement KnowledgeGraph class and CLI.

### 6.1 KnowledgeGraph Class (`api/`)

**Files implemented**:
- `knowledge_graph.py` - Main class ✅
- `convenience.py` - Top-level functions ✅
- `shell.py` - Shell interface (stub, see Phase 5) ⏳

**API surface**:
```python
kg = KnowledgeGraph("./my_kb")

# Ingestion
await kg.ingest_pdf("doc.pdf")
await kg.ingest_markdown("doc.md")
await kg.ingest_chunks(chunks)

# Query
result = await kg.query("What were the findings?")
entities = await kg.search_entities("apple")

# Shell
shell = kg.shell()
shell.execute("ls /kg/entities/")

# Sync wrappers
kg.ingest_pdf_sync("doc.pdf")
kg.query_sync("What were the findings?")
```

**Note**: `KGShell` is still a placeholder. The above shell calls are not yet implemented.

**Deliverable**: Complete KnowledgeGraph class (shell interface pending).

---

### 6.2 CLI (`cli/`)

**Status**: ✅ COMPLETE

**Files implemented**:
- `__init__.py` - CLI entry point (typer-based) ✅
- `__main__.py` - Python -m entry point ✅

**Commands**:
```bash
zomma-kg ingest report.pdf --kb ./my_kb     # ✅ Implemented
zomma-kg query "What were the risks?" --kb ./my_kb  # ✅ Implemented
zomma-kg info --kb ./my_kb                  # ✅ Implemented
zomma-kg shell --kb ./my_kb                 # ⏳ Placeholder (KGShell not implemented)
zomma-kg export --format json --kb ./my_kb  # ❌ Not implemented
```

**Deliverable**: CLI with core commands. ✅

---

## Known Issues

### Entity Deduplication Bug (test_kb_refactor)

**Status**: 🐛 Open

**Observed**: When running the MCP server against `test_kb_refactor`, entities appear multiple times with different UUIDs:
- "Federal Reserve System" appears 3 times
- "Federal Reserve" appears 2 times
- "Beige Book" appears 4 times

**Expected**: Each canonical entity should appear exactly once after deduplication.

**Root Cause**: Investigation needed - likely the `deduplicate_entities` function is not being applied correctly during ingestion, or the test KB was created before deduplication was implemented.

**Impact**: Query results may return duplicate entities, inflating result counts.

**Fix**: Re-ingest test documents with proper deduplication or investigate the dedup pipeline.

---

## Phase 7: Testing & Polish (Week 8-9)

**Status**: ⏳ PARTIAL

### 7.1 Unit Tests

- `tests/test_types.py` - Pydantic model validation ✅
- `tests/test_config.py` - Config loading from all sources ❌
- `tests/test_storage.py` - Parquet/LanceDB operations ❌
- `tests/test_entity_dedup.py` - Entity deduplication ✅
- `tests/test_entity_registry.py` - Cross-document entity resolution ✅
- `tests/test_topic_resolver.py` - Topic ontology resolution ✅
- `tests/test_assembler.py` - Assembly pipeline ✅
- `tests/test_query_pipeline.py` - Query pipeline ✅
- `tests/test_knowledge_graph.py` - Main API ✅
- `tests/test_mcp_server.py` - MCP server command flow ✅
- `tests/test_parquet_append_scalability.py` - Parquet append scalability ✅
- `tests/test_shell.py` - Shell commands ❌ (depends on Phase 5)

### 7.2 Integration Tests

- End-to-end: PDF → ingest → query → answer ❌
- Multi-document knowledge base ❌
- Shell navigation scenarios ❌ (depends on Phase 5)

### 7.3 Documentation

- README.md quickstart ✅
- Developer guide (`docs/DEVELOPER_GUIDE.md`) ✅
- API reference (docstrings) ⏳ Partial
- Migration guide from Neo4j ❌

---

## Dependency Graph

```
Phase 1: Foundation
    types/ ─────────────────────────────┐
    config/ ────────────────────────────┤
    providers/ ─────────────────────────┤
                                        │
                                        v
Phase 2: Storage ───────────────────────┤
    storage/parquet/ ───────────────────┤
    storage/lancedb/ ───────────────────┤
    storage/duckdb/ ────────────────────┤
                                        │
            ┌───────────────────────────┘
            │
            v
Phase 3: Ingestion              Phase 4: Query
    chunking/ ──────┐               decomposer.py
    extraction/ ────┼───────────►   researcher.py
    resolution/ ────┤               synthesizer.py
    assembly/ ──────┘                   │
            │                           │
            v                           v
Phase 5: Shell ◄────────────────► Phase 6: API
    commands.py                     knowledge_graph.py
    path_resolver.py                cli/
```

---

## What to Port vs Rewrite

### Port (adapt for new storage):

| Component | Port From | Adaptation Needed |
|-----------|-----------|-------------------|
| Extraction prompts | `pipeline/extractor.py` | Change LLM client |
| Dedup algorithm | `pipeline/entity_dedup.py` | Use LanceDB for similarity |
| Query decomposition | `query/decomposer.py` | Minimal changes |
| Context builder | `query/context_builder.py` | Minimal changes |

### Rewrite:

| Component | Why |
|-----------|-----|
| Storage layer | Neo4j → Parquet/LanceDB/DuckDB |
| Graph queries | Cypher → SQL |
| Bulk writer | Neo4j batching → Parquet append |
| Shell interface | New feature |

---

## Milestone Checkpoints

| Milestone | Week | Deliverable | Status |
|-----------|------|-------------|--------|
| **M1** | 2 | Types + Config + Storage reads/writes working | ✅ Complete |
| **M2** | 4 | Can ingest a PDF end-to-end | ✅ Complete |
| **M3** | 6 | Can query and get answers | ✅ Complete |
| **M4** | 7 | Agent interface (MCP server) working | ✅ Complete |
| **M5** | 8 | KnowledgeGraph API + CLI complete | ✅ Complete |
| **M6** | 9 | Tests passing, docs complete | ⏳ Partial |

---

## Files to Create (Priority Order)

### Week 1-2 (Foundation)
1. `types/*.py` - Complete all type definitions ✅ DONE
2. `config/settings.py` - KGConfig implementation ✅ DONE
3. `providers/llm/openai.py` - OpenAI LLM provider ✅ DONE
4. `providers/embedding/openai.py` - OpenAI embeddings ✅ DONE

### Week 2-3 (Storage)
5. `storage/base.py` - Abstract interface ✅ DONE
6. `storage/parquet/backend.py` - Parquet operations ✅ DONE
7. `storage/lancedb/indices.py` - Vector indices ✅ DONE
8. `storage/duckdb/queries.py` - SQL queries ✅ DONE

### Week 3-5 (Ingestion)
9. `ingestion/chunking/markdown.py` - Markdown chunker ✅ DONE
10. `ingestion/chunking/pdf.py` - PDF converter ✅ DONE
11. `ingestion/extraction/extractor.py` - Chain-of-thought ✅ DONE
12. `ingestion/resolution/entity_dedup.py` - Deduplication ✅ DONE
13. `ingestion/resolution/entity_registry.py` - Cross-document matching ✅ DONE
14. `ingestion/resolution/topic_resolver.py` - Topic ontology resolution ✅ DONE
15. `ingestion/assembly/assembler.py` - Write to storage ✅ DONE

### Week 5-6 (Query)
16. `query/pipeline.py` - V7 orchestrator ✅ DONE
17. `query/decomposer.py` - Question decomposition ✅ DONE
18. `query/researcher.py` - Retrieval ✅ DONE
19. `query/synthesizer.py` - Answer synthesis ✅ DONE
20. `query/context_builder.py` - Context assembly ✅ DONE
21. `query/types.py` - Pipeline types ✅ DONE

### Week 6-7 (Agent Interface - MCP)
22. `mcp/__init__.py` - MCP module init ✅ DONE
23. `mcp/server.py` - MCP server with kg_execute tool ✅ DONE
24. `mcp/__main__.py` - Entry point ✅ DONE

### Week 7-8 (API)
23. `api/knowledge_graph.py` - Main class ✅ DONE
24. `cli/__init__.py` - CLI commands ✅ DONE
25. `cli/__main__.py` - Python -m entry point ✅ DONE

---

## Success Criteria

The package is complete when:

| Criterion | Description | Status |
|-----------|-------------|--------|
| **Zero infrastructure** | `pip install zomma-kg && python -c "from zomma_kg import KnowledgeGraph"` works | ✅ Ready |
| **End-to-end** | Can ingest a PDF and answer questions about it | ✅ Working |
| **Portable** | Knowledge base is a directory that can be zipped and shared | ✅ Working |
| **Agent-friendly** | MCP server with kg_execute tool for LLM agents | ✅ Working |
| **Tested** | Core functionality has test coverage | ⏳ Partial |
| **Documented** | README quickstart + Developer Guide; API docstrings partial | ⏳ Partial |
