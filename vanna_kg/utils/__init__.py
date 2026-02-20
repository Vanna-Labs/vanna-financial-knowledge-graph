"""
Utility Functions

Core algorithms and helper functions used throughout the package.

Note: Cosine similarity is handled by LanceDB's vector search.
We don't implement our own - the vector database does this efficiently.

Modules:
    clustering: Union-Find algorithm for entity deduplication
    text: Text processing and normalization
"""

from vanna_kg.utils.clustering import UnionFind, build_similarity_edges, union_find_components
from vanna_kg.utils.cost_telemetry import CostCollector, telemetry_collector, telemetry_stage
from vanna_kg.utils.embedding_text import format_canonical_entity_text
from vanna_kg.utils.text import clean_entity_name, generate_chunk_id, normalize_relationship_type

__all__ = [
    "UnionFind",
    "union_find_components",
    "build_similarity_edges",
    "CostCollector",
    "telemetry_collector",
    "telemetry_stage",
    "format_canonical_entity_text",
    "normalize_relationship_type",
    "clean_entity_name",
    "generate_chunk_id",
]
