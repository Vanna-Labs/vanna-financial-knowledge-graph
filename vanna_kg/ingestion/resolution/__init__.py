"""
Entity and Topic Resolution

Three levels of deduplication to ensure clean knowledge graph:

Modules:
    entity_dedup: In-document deduplication (Phase 2a-c)
    entity_registry: Cross-document entity resolution (Phase 2d)
    topic_resolver: Topic ontology resolution (Phase 2e)

In-Document Deduplication (Phase 2a-c):
    1. Generate embeddings for all entities
    2. Build similarity matrix (cosine similarity)
    3. Find connected components via Union-Find
    4. LLM verification of clusters

Cross-Document Resolution (Phase 2d):
    - Vector search for candidates in existing KB
    - LLM verification of matches
    - UUID reuse for matches, summary merging

Topic Resolution (Phase 2e):
    - Vector search against topic ontology
    - LLM verification of semantic matches

Key Principle: Subsidiary Awareness
    AWS != Amazon (subsidiaries are separate entities)

See: docs/pipeline/DEDUPLICATION_SYSTEM.md
"""

from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities
from vanna_kg.ingestion.resolution.entity_registry import EntityRegistry
from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

__all__ = ["deduplicate_entities", "EntityRegistry", "TopicResolver"]
