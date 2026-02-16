"""
Knowledge Base Assembly

Final phase that writes resolved data to storage.

Modules:
    assembler: Main assembly logic

Three Sub-Phases:
    3a. Operation Collection:
        - Collect all nodes and relationships into buffer
        - Chunk-centric fact pattern (Subject -> Chunk -> Object)

    3b. Batch Embedding Generation:
        - Entity embeddings (name + summary)
        - Entity name-only embeddings
        - Topic embeddings
        - Fact embeddings
        - All run in parallel with rate limit handling

    3c. Bulk Write:
        - Write to Parquet files
        - Index vectors in LanceDB
        - Write order: documents -> chunks -> entities -> facts -> topics -> relationships

Idempotency:
    - Safe to re-run (MERGE semantics)
    - fact_id for relationship deduplication

See: docs/pipeline/ASSEMBLY_SYSTEM.md
"""

from vanna_kg.ingestion.assembly.assembler import Assembler

__all__ = ["Assembler"]
