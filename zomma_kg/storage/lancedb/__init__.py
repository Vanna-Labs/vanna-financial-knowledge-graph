"""
LanceDB Vector Indices

Vector search for semantic retrieval.

Modules:
    indices: Vector index management

Index Schemas:
    entities.lance:
        uuid, name, summary, entity_type, vector (3072 dims)

    facts.lance:
        uuid, content, subject_name, object_name, relationship_type, vector

    topics.lance:
        uuid, name, definition, vector

Features:
    - Cosine similarity search
    - Filtered search (by entity_type, etc.)
    - IVF_PQ indexing for scale

See: docs/architecture/PARQUET_STORAGE_MIGRATION.md Section 4
"""

from zomma_kg.storage.lancedb.indices import LanceDBIndices

__all__ = ["LanceDBIndices"]
