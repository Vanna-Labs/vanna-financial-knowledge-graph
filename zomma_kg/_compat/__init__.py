"""
Compatibility Layer

Migration helpers for users transitioning from the Neo4j-based system.

Modules:
    legacy: Deprecated class aliases with migration guidance
"""

from zomma_kg._compat.legacy import Pipeline, QueryEngine

__all__ = ["Pipeline", "QueryEngine"]
