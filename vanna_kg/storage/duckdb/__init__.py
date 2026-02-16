"""
DuckDB Query Layer

Relational queries on Parquet files.

Modules:
    queries: SQL query implementations

Query Patterns:
    - Entity lookup by name
    - Chunk retrieval for entity (1-hop)
    - 1-hop neighbor expansion (2-hop with fact_id join)
    - Fact retrieval by entity
    - Global aggregations

Example Queries (from docs/architecture/PARQUET_STORAGE_MIGRATION.md):

    -- Entity chunks (1-hop)
    SELECT c.* FROM chunks c
    JOIN relationships r ON c.uuid = r.to_uuid
    WHERE r.from_uuid = ? AND r.rel_type != 'CONTAINS_CHUNK'

    -- 1-hop neighbors
    SELECT DISTINCT e2.* FROM relationships r1
    JOIN relationships r2 ON r1.fact_id = r2.fact_id
    JOIN entities e2 ON r2.to_uuid = e2.uuid
    WHERE r1.from_uuid = ? AND e2.uuid != ?

See: docs/architecture/PARQUET_STORAGE_MIGRATION.md Section 5
"""

from vanna_kg.storage.duckdb.queries import DuckDBQueries

__all__ = ["DuckDBQueries"]
