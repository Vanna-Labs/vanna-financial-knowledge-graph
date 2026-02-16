"""
Utility Functions

Core algorithms and helper functions used throughout the package.

Note: Cosine similarity is handled by LanceDB's vector search.
We don't implement our own - the vector database does this efficiently.

Modules:
    clustering: Union-Find algorithm for entity deduplication
    text: Text processing and normalization
"""

from vanna_kg.utils.clustering import UnionFind, union_find_components, build_similarity_edges
from vanna_kg.utils.text import normalize_relationship_type, clean_entity_name, generate_chunk_id

__all__ = [
    "UnionFind",
    "union_find_components",
    "build_similarity_edges",
    "normalize_relationship_type",
    "clean_entity_name",
    "generate_chunk_id",
]
