"""
Parquet Storage Backend

Primary storage for knowledge base data.

Modules:
    backend: ParquetBackend class
    tables: Table schemas and operations
    migrations: Schema versioning

Table Schemas:
    entities.parquet:
        uuid, name, summary, entity_type, aliases, created_at, updated_at

    chunks.parquet:
        uuid, document_uuid, content, header_path, position, document_date, created_at

    facts.parquet:
        uuid, content, subject_uuid, subject_name, object_uuid, object_name,
        object_type, relationship_type, date_context, created_at

    relationships.parquet:
        id, from_uuid, from_type, to_uuid, to_type, rel_type, fact_id,
        description, date_context, created_at

    topics.parquet:
        uuid, name, definition, parent_topic, created_at

    documents.parquet:
        uuid, name, document_date, source_path, file_type, metadata, created_at

See: docs/architecture/PARQUET_STORAGE_MIGRATION.md Section 4
"""

from vanna_kg.storage.parquet.backend import ParquetBackend

__all__ = ["ParquetBackend"]
