# VannaKG

Embedded knowledge graph library for document understanding.

## Installation

```bash
pip install vanna-kg
```

Install directly from GitHub:

```bash
pip install "git+https://github.com/Vanna-Labs/vanna-financial-knowledge-graph.git"
```

Python package imports use underscore:

```python
import vanna_kg
```

## Environment

Copy `.env.example` to `.env` and set at least:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## Quick Start

```python
from vanna_kg import KnowledgeGraph

kg = KnowledgeGraph("./my_kb")

# Ingest a document
await kg.ingest_pdf("report.pdf")

# Query the knowledge graph
result = await kg.query("What were the key findings?")
print(result.answer)
```

## Developer Guide

See `docs/DEVELOPER_GUIDE.md` for a comprehensive API and usage guide.

## Features

- **Zero Infrastructure**: Embedded storage using DuckDB, LanceDB, and Parquet
- **Portable Knowledge Bases**: A directory you can zip and share
- **OpenAI-First Runtime**: OpenAI-backed ingestion and query pipeline in production
- **Roadmap Placeholders**: `KGShell` and query-time global chunk-search fallback in `query()` are not implemented yet
- **Planned Providers**: Anthropic, Google, and alternative embedding providers are roadmap items

## License

Apache-2.0
