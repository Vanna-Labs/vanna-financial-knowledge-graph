"""
Storage Backends

Embedded storage using DuckDB + LanceDB + Parquet files.

Modules:
    base: Abstract storage interface
    parquet/: Primary storage implementation
    lancedb/: Vector search indices
    duckdb/: Relational queries

Knowledge Base Directory Structure:
    my_kb/
    ├── metadata.json           # KB metadata and schema version
    ├── entities.parquet        # Extracted entities
    ├── chunks.parquet          # Source text chunks
    ├── facts.parquet           # Extracted facts/relationships
    ├── topics.parquet          # Topic classifications
    ├── documents.parquet       # Source document metadata
    ├── relationships.parquet   # All graph edges
    └── lancedb/                # Vector indices
        ├── entities.lance/
        ├── facts.lance/
        └── topics.lance/

Design Principles:
    - Zero infrastructure (embedded databases)
    - Portable (knowledge base is just a directory)
    - Fast (DuckDB for relational, LanceDB for vectors)

See: docs/architecture/PARQUET_STORAGE_MIGRATION.md
"""

from vanna_kg.storage.base import StorageBackend
from vanna_kg.storage.parquet.backend import ParquetBackend

__all__ = [
    "StorageBackend",
    "ParquetBackend",
]
