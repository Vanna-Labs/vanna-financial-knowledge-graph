# ZommaKG Developer Guide

This guide documents the public API and recommended usage patterns for the ZommaKG Python package.
It is based on the current code in `zomma_kg/api`, `zomma_kg/config`, `zomma_kg/types`, and the CLI.

**Audience**
Developers integrating ZommaKG into applications, scripts, or pipelines.

**Status Notes**
- `KGShell` is a placeholder and not implemented.
- `KnowledgeGraph.search_chunks` is not implemented (requires chunk embeddings).
- Only the OpenAI LLM + embedding providers are implemented; Anthropic/Google/Voyage are stubs.
- PDF ingestion requires Google Gemini via `google-genai`.

## Installation

Core package:

```bash
pip install zomma-kg
```

Recommended extras:

```bash
# OpenAI LLM + embeddings (required for ingestion + query)
pip install "zomma-kg[openai]"

# Google Gemini for PDF -> Markdown conversion
pip install "zomma-kg[google]"
```

Environment variables:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."  # required for PDF ingestion
```

## Quick Start

Async API:

```python
from zomma_kg import KnowledgeGraph

kg = KnowledgeGraph("./my_kb")

# Ingest a PDF (uses Gemini for PDF->Markdown)
result = await kg.ingest_pdf("report.pdf")
print(result.entities, result.facts)

# Query the knowledge graph
answer = await kg.query("What were the key findings?")
print(answer.answer)
```

Sync API:

```python
from zomma_kg import KnowledgeGraph

kg = KnowledgeGraph("./my_kb")
kg.ingest_pdf_sync("report.pdf")
result = kg.query_sync("Summarize pricing changes")
print(result.answer)
```

Convenience functions (short scripts / REPL):

```python
from zomma_kg import ingest_pdf, query

ingest_pdf("report.pdf", kb="./my_kb")
result = query("What were the key findings?", kb="./my_kb")
print(result.answer)
```

## Core Concepts

**Knowledge base (KB)**
A KB is a self-contained directory containing Parquet tables, LanceDB vector indexes, and metadata.
The KB is portable and can be zipped or copied between machines.

**Async-first design**
All core methods are async with sync wrappers. In async apps, call `await kg.query(...)`.
In sync contexts, use `kg.query_sync(...)` or convenience functions.

**Lazy initialization**
`KnowledgeGraph` initializes storage and providers on first use. This makes construction cheap.

## Public API Surface

### `KnowledgeGraph`
Primary entry point for ingestion, query, search, and KB statistics.

Constructor:

```python
from zomma_kg import KnowledgeGraph, KGConfig

config = KGConfig(llm_model="gpt-5.1", embedding_model="text-embedding-3-large")
kg = KnowledgeGraph("./my_kb", config=config, create=True)
```

Key ingestion methods:
- `ingest_pdf(path, document_date=None, metadata=None, on_progress=None)`
- `ingest_markdown(path, document_date=None, metadata=None, max_chunks=None, on_progress=None)`
- `ingest_chunks(chunks, document_name=None, document_date=None, metadata=None, on_progress=None)`
- `ingest_directory(path, pattern="**/*.pdf", recursive=True, on_progress=None)`

Key query methods:
- `query(question, include_sources=True)`
- `decompose(question, max_subqueries=None)`

Search and retrieval:
- `search_entities(query, limit=10, threshold=0.3)`
- `search_facts(query, limit=20, threshold=0.3)`
- `search_chunks(...)` is not implemented

Data access:
- `get_entity(name)`
- `get_entities(limit=100, offset=0)`
- `get_document(document_id)`
- `get_documents()`
- `get_facts_for_entity(entity_name, limit=100)`
- `get_neighbors(entity_name, limit=20)`

Stats:
- `stats()` returns counts for documents, entities, chunks, facts

Resource cleanup:
- `await kg.close()`
- `kg.close_sync()`
- Context managers are supported: `async with KnowledgeGraph(...)` or `with KnowledgeGraph(...)`

### Convenience functions
Convenience functions wrap the sync API and auto-manage the KG lifecycle.

- `ingest_pdf(path, kb=...)`
- `ingest_markdown(path, kb=...)`
- `ingest_chunks(chunks, kb=...)`
- `query(question, kb=...)`
- `decompose(question, kb=...)`

### `KGShell`
Filesystem-style navigation API for KG exploration. This is currently a placeholder.

## Ingestion Pipeline (What Happens Under the Hood)

The ingestion pipeline is a multi-stage process:

1. Chunking
2. Extraction
3. Deduplication
4. Resolution
5. Assembly

Pipeline details:
- PDF ingestion calls Gemini (Google) to convert PDF to Markdown.
- Markdown is split into header-aware chunks.
- Extraction uses LLM prompts to extract entities and facts.
- In-document deduplication uses embeddings + union-find + LLM verification.
- Cross-document resolution merges entities against existing KB.
- Assembly writes Parquet rows and updates LanceDB indexes.

Progress callbacks:

```python
def on_progress(stage: str, progress: float) -> None:
    print(stage, progress)

await kg.ingest_pdf("report.pdf", on_progress=on_progress)
```

## Query Pipeline

`query()` uses a GraphRAG pipeline:

1. Decompose the question into sub-queries.
2. Resolve entity and topic hints to nodes in the graph.
3. Retrieve relevant chunks, facts, and neighbors in parallel.
4. Assemble and deduplicate context.
5. Synthesize an answer with a question-type-aware format.

You can inspect decomposition with `decompose()`.

## Configuration

`KGConfig` supports programmatic configuration, environment variables, and TOML files.

Programmatic:

```python
from zomma_kg import KGConfig

config = KGConfig(
    llm_provider="openai",
    llm_model="gpt-5.1",
    embedding_provider="openai",
    embedding_model="text-embedding-3-large",
    extraction_concurrency=10,
)
```

Environment variables:
- `ZOMMA_LLM_PROVIDER`
- `ZOMMA_LLM_MODEL`
- `ZOMMA_EMBEDDING_PROVIDER`
- `ZOMMA_EXTRACTION_CONCURRENCY`
- `ZOMMA_REGISTRY_CONCURRENCY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `VOYAGE_API_KEY`

TOML configuration:

```python
from zomma_kg import KGConfig

config = KGConfig.from_file("./kg.toml")
kg = KnowledgeGraph("./kb", config=config)
```

Example `kg.toml`:

```toml
[llm]
provider = "openai"
model = "gpt-5.1"

[embedding]
provider = "openai"
model = "text-embedding-3-large"

[processing]
extraction_concurrency = 10
registry_concurrency = 10
```

## Data Model Overview

Key persisted types are defined in `zomma_kg/types`.

- `Document`: document metadata
- `Chunk`: source text segments with header context
- `Entity`: canonical entities with summaries and aliases
- `Fact`: relationships between entities/topics with temporal context
- `Topic`: resolved themes or concepts

Results:
- `IngestResult`: counts and errors for ingestion
- `QueryResult`: answer, confidence, sources, and timing

## Knowledge Base Layout

A knowledge base directory contains:

- `documents.parquet`
- `chunks.parquet`
- `entities.parquet`
- `facts.parquet`
- `topics.parquet`
- `relationships.parquet`
- `lancedb/` (vector indexes)
- `metadata.json`

## CLI Usage

Entry point: `zomma-kg`

Ingest a file:

```bash
zomma-kg ingest report.pdf --kb ./my_kb
```

Ingest a directory:

```bash
zomma-kg ingest ./documents --kb ./my_kb --pattern "**/*.pdf"
```

Query:

```bash
zomma-kg query "What were the findings?" --kb ./my_kb --sources
```

Stats:

```bash
zomma-kg info --kb ./my_kb
```

Shell:

```bash
zomma-kg shell --kb ./my_kb
```

The shell is currently a placeholder and prints a “not implemented” message.

## Limitations and Roadmap Items

Current limitations:
- `KGShell` is not implemented.
- `search_chunks` is not implemented (no chunk embeddings yet).
- Anthropic/Google/Voyage providers are declared but not implemented.
- Query global chunk search is disabled by default.

If you need any of these features, confirm roadmap priority before relying on them.

## Troubleshooting

Common issues:
- `ImportError: langchain-openai`: install `zomma-kg[openai]`.
- `ValueError: Google API key required`: set `GOOGLE_API_KEY` or pass `api_key` when ingesting PDFs.
- `RuntimeError: This event loop is already running`: use async API instead of sync wrappers inside async apps.

## Appendix: Minimal Integration Template

```python
import asyncio
from zomma_kg import KnowledgeGraph, KGConfig

async def main() -> None:
    config = KGConfig(llm_model="gpt-5.1", embedding_model="text-embedding-3-large")
    async with KnowledgeGraph("./kb", config=config) as kg:
        await kg.ingest_markdown("report.md")
        result = await kg.query("What changed in Q4?")
        print(result.answer)

if __name__ == "__main__":
    asyncio.run(main())
```
