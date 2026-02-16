"""
VannaKG - Embedded Knowledge Graph Library

A pip-installable Python library for building knowledge graphs from documents
with zero infrastructure requirements.

Example:
    >>> from vanna_kg import KnowledgeGraph
    >>> kg = KnowledgeGraph("./my_kb")
    >>> await kg.ingest_pdf("document.pdf")
    >>> result = await kg.query("What were the key findings?")
    >>> print(result.answer)

Main Classes:
    KnowledgeGraph: Primary entry point for all operations
    KGShell: Filesystem-style navigation interface
    KGConfig: Configuration management

See Also:
    - docs/architecture/PYTHON_PACKAGE_DESIGN.md for full API design
    - docs/architecture/PARQUET_STORAGE_MIGRATION.md for storage details
    - docs/architecture/FILESYSTEM_NAVIGATION_ARCHITECTURE.md for shell usage
"""

__version__ = "0.3.0"

# Public API - lazy imports to avoid loading optional dependencies
def __getattr__(name: str):
    """Lazy import public API components."""

    if name == "KnowledgeGraph":
        from vanna_kg.api.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph

    if name == "KGShell":
        from vanna_kg.api.shell import KGShell
        return KGShell

    if name == "KGConfig":
        from vanna_kg.config.settings import KGConfig
        return KGConfig

    # Convenience functions
    if name in ("ingest_pdf", "ingest_markdown", "ingest_chunks", "query", "decompose"):
        from vanna_kg.api import convenience
        return getattr(convenience, name)

    # Types
    if name in ("Chunk", "Entity", "Fact", "Topic", "Document", "QueryResult"):
        from vanna_kg import types
        return getattr(types, name)

    raise AttributeError(f"module 'vanna_kg' has no attribute {name!r}")


__all__ = [
    # Main classes
    "KnowledgeGraph",
    "KGShell",
    "KGConfig",

    # Convenience functions
    "ingest_pdf",
    "ingest_markdown",
    "ingest_chunks",
    "query",
    "decompose",

    # Types
    "Chunk",
    "Entity",
    "Fact",
    "Topic",
    "Document",
    "QueryResult",

    # Version
    "__version__",
]
