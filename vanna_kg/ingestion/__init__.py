"""
Ingestion Pipeline

Three-phase document processing pipeline that transforms raw documents
into a structured knowledge graph.

Phases:
    Phase 1 - Chunking & Extraction (LLM-heavy):
        - PDF -> Markdown -> Chunks
        - Chain-of-thought entity/fact extraction with critique

    Phase 2 - Resolution (Deduplication):
        - In-document entity deduplication (embeddings + Union-Find + LLM)
        - Cross-document entity resolution
        - Topic resolution against ontology

    Phase 3 - Assembly (Bulk Write):
        - Batch embedding generation
        - Write to Parquet + LanceDB

Modules:
    pipeline: Main ingestion orchestrator
    chunking/: Document chunking (PDF, markdown, text)
    extraction/: LLM-based entity/fact extraction
    resolution/: Entity and topic resolution
    assembly/: Knowledge base construction

See:
    - docs/pipeline/CHUNKING_SYSTEM.md
    - docs/pipeline/ENTITY_TOPIC_EXTRACTION.md
    - docs/pipeline/DEDUPLICATION_SYSTEM.md
    - docs/pipeline/ASSEMBLY_SYSTEM.md
"""

from vanna_kg.ingestion.assembly import Assembler
from vanna_kg.ingestion.resolution import EntityRegistry, deduplicate_entities

__all__ = ["Assembler", "EntityRegistry", "deduplicate_entities"]
