# VannaKG

Embedded knowledge graph library for document understanding.

## Installation

```bash
pip install vanna-kg
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
- **Roadmap Placeholders**: `KGShell`, chunk vector search, and global chunk search are not implemented yet
- **Planned Providers**: Anthropic, Google, and alternative embedding providers are roadmap items

## License

Apache-2.0
