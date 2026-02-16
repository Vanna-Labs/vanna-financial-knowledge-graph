"""
Convenience Functions

Top-level functions for common operations without explicit KnowledgeGraph
instantiation. These are designed for quick scripts and REPL usage.

Example:
    >>> from vanna_kg import ingest_pdf, query
    >>> ingest_pdf("report.pdf", kb="./my_kb")
    >>> result = query("What were the findings?", kb="./my_kb")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vanna_kg.types.chunks import Chunk
    from vanna_kg.types.results import IngestResult, QueryResult


def ingest_pdf(
    path: str | Path,
    *,
    kb: str | Path,
    **kwargs,
) -> "IngestResult":
    """
    Ingest a PDF document into a knowledge base.

    Args:
        path: Path to the PDF file
        kb: Path to the knowledge base directory
        **kwargs: Additional arguments passed to KnowledgeGraph.ingest_pdf()
    """
    from vanna_kg.api.knowledge_graph import KnowledgeGraph
    with KnowledgeGraph(kb) as kg:
        return kg.ingest_pdf_sync(path, **kwargs)


def ingest_markdown(
    path: str | Path,
    *,
    kb: str | Path,
    **kwargs,
) -> "IngestResult":
    """Ingest a markdown document into a knowledge base."""
    from vanna_kg.api.knowledge_graph import KnowledgeGraph
    with KnowledgeGraph(kb) as kg:
        return kg.ingest_markdown_sync(path, **kwargs)


def ingest_chunks(
    chunks: list["Chunk"],
    *,
    kb: str | Path,
    **kwargs,
) -> "IngestResult":
    """Ingest pre-chunked content into a knowledge base."""
    from vanna_kg.api.knowledge_graph import KnowledgeGraph

    with KnowledgeGraph(kb) as kg:
        return kg.ingest_chunks_sync(chunks, **kwargs)


def query(
    question: str,
    *,
    kb: str | Path,
    **kwargs,
) -> "QueryResult":
    """
    Query a knowledge base.

    Args:
        question: Natural language question
        kb: Path to the knowledge base directory
        **kwargs: Additional arguments passed to KnowledgeGraph.query()
    """
    from vanna_kg.api.knowledge_graph import KnowledgeGraph
    with KnowledgeGraph(kb, create=False) as kg:
        return kg.query_sync(question, **kwargs)


def decompose(
    question: str,
    *,
    kb: str | Path,
    **kwargs,
) -> list:
    """Decompose a question into sub-queries."""
    from vanna_kg.api.knowledge_graph import KnowledgeGraph

    with KnowledgeGraph(kb, create=False) as kg:
        return kg.decompose_sync(question, **kwargs)
