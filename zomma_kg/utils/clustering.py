"""
Clustering Algorithms

Union-Find (Disjoint Set Union) implementation for entity deduplication.

Used during Phase 2 of ingestion to cluster similar entities before
LLM verification. Similarity scores come from the embedding provider.

See: zomma_kg/ingestion/resolution/README.md
"""

from __future__ import annotations

from collections import defaultdict


class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.

    Used for clustering entities during deduplication. When two entities
    have similarity above threshold (from embedding provider), they are
    unioned together into the same cluster.

    Time Complexity:
        - find(): O(α(n)) amortized (nearly constant)
        - union(): O(α(n)) amortized
        where α is the inverse Ackermann function
    """

    def __init__(self, n: int) -> None:
        """Initialize Union-Find with n elements (0 to n-1)."""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n

    def find(self, x: int) -> int:
        """Find root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union components containing x and y. Returns True if merged."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same component."""
        return self.find(x) == self.find(y)

    def get_components(self) -> list[list[int]]:
        """Get all connected components as lists of indices."""
        components: dict[int, list[int]] = defaultdict(list)
        for i in range(self.n):
            components[self.find(i)].append(i)
        return list(components.values())


def union_find_components(
    n: int,
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    """
    Find connected components given edges.

    Args:
        n: Number of nodes
        edges: List of (i, j) edges indicating similar entities

    Returns:
        List of components (each is a list of node indices)
    """
    uf = UnionFind(n)
    for i, j in edges:
        uf.union(i, j)
    return uf.get_components()


def build_similarity_edges(
    similarities: list[tuple[int, int, float]],
    threshold: float = 0.70,
) -> list[tuple[int, int]]:
    """
    Build edges from similarity scores (from embedding provider).

    Args:
        similarities: List of (i, j, score) from vector search
        threshold: Minimum similarity to create an edge

    Returns:
        List of (i, j) edges where score >= threshold
    """
    return [(i, j) for i, j, score in similarities if score >= threshold]
