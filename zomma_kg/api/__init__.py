"""
Public API Layer

This module contains the user-facing API classes and functions.

Modules:
    knowledge_graph: KnowledgeGraph class - main entry point
    shell: KGShell class - filesystem-style navigation
    convenience: Top-level convenience functions (ingest_pdf, query, etc.)

Design Principles:
    - Single entry point (KnowledgeGraph) for most operations
    - Async-first with sync wrappers (_sync suffix)
    - Lazy initialization - don't connect until needed
    - Context manager support for resource cleanup
"""

from zomma_kg.api.knowledge_graph import KnowledgeGraph
from zomma_kg.api.shell import KGShell

__all__ = ["KnowledgeGraph", "KGShell"]
