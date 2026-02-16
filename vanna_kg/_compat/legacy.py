"""
Legacy Compatibility for Neo4j Users

Migration helpers for users transitioning from the Neo4j-based system.

See: docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 10
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vanna_kg.api.knowledge_graph import KnowledgeGraph


# Legacy class aliases with deprecation warnings
class Pipeline:
    """
    DEPRECATED: Use KnowledgeGraph instead.

    Migration:
        # Before
        pipeline = Pipeline()
        await pipeline.ingest_file("doc.pdf")

        # After
        kg = KnowledgeGraph("./my_kb")
        await kg.ingest_pdf("doc.pdf")
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Pipeline is deprecated. Use KnowledgeGraph instead. "
            "See docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 10 for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "Pipeline has been replaced by KnowledgeGraph. "
            "Please update your code."
        )


class QueryEngine:
    """
    DEPRECATED: Use KnowledgeGraph.query() instead.

    Migration:
        # Before
        engine = QueryEngine(group_id="default")
        result = await engine.query("...")

        # After
        kg = KnowledgeGraph("./my_kb")
        result = await kg.query("...")
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "QueryEngine is deprecated. Use KnowledgeGraph.query() instead. "
            "See docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 10 for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "QueryEngine has been replaced by KnowledgeGraph.query(). "
            "Please update your code."
        )


# Legacy function aliases
def get_config():
    """DEPRECATED: Use KGConfig instead."""
    warnings.warn(
        "get_config() is deprecated. Use KGConfig instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from vanna_kg.config import KGConfig
    return KGConfig()
