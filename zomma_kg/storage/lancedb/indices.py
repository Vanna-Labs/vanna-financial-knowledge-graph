"""
LanceDB Vector Indices

Manages vector indices for entities, facts, and topics with multi-tenant support.
"""

import asyncio
import threading
from pathlib import Path
from typing import Any

import lancedb

from zomma_kg.config import KGConfig


class LanceDBIndices:
    """
    Manages LanceDB vector indices for similarity search.

    Tables:
        - entities: Entity embeddings (uuid, name, summary, group_id, vector)
        - facts: Fact embeddings (uuid, content, subject_name, object_name, group_id, vector)
        - topics: Topic embeddings (uuid, name, definition, group_id, vector)

    Multi-tenancy:
        All data is tagged with group_id. Searches are filtered by group_id.

    Thread safety:
        Uses thread-local storage for connections since LanceDB connections
        may not be thread-safe and asyncio.to_thread() may use different threads.

    All vectors are 3072 dimensions (text-embedding-3-large compatible).
    """

    VECTOR_DIM = 3072

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        """Escape single quotes for SQL WHERE clauses."""
        return value.replace("'", "''")

    def __init__(self, lancedb_path: Path, config: KGConfig, group_id: str = "default"):
        self.path = lancedb_path
        self.config = config
        self.group_id = group_id
        self._local = threading.local()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize LanceDB (marks as ready, connections created per-thread)."""
        if self._initialized:
            return

        def _init() -> None:
            self.path.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(_init)
        self._initialized = True

    async def close(self) -> None:
        """Close LanceDB connections."""
        self._initialized = False
        # Clear current thread's connection if it exists
        if hasattr(self._local, 'db'):
            self._local.db = None

    def _get_db(self) -> lancedb.DBConnection:
        """Get thread-local LanceDB connection, creating if needed."""
        if not self._initialized:
            raise RuntimeError("LanceDB not initialized. Call initialize() first.")

        # Check for existing thread-local connection
        db = getattr(self._local, 'db', None)
        if db is None:
            # Create new connection for this thread
            db = lancedb.connect(str(self.path))
            self._local.db = db
        return db

    @staticmethod
    def _table_names(db: lancedb.DBConnection) -> set[str]:
        """
        Return table names across LanceDB API variants.

        Recent LanceDB returns a response object from list_tables() with a
        `tables` attribute, while older versions return a plain list.
        """
        listed = db.list_tables()
        tables = getattr(listed, "tables", listed)
        return {str(name) for name in tables}

    def _has_table(self, db: lancedb.DBConnection, table_name: str) -> bool:
        """Check table existence in a LanceDB-version-safe way."""
        return table_name in self._table_names(db)

    # -------------------------------------------------------------------------
    # Entity Operations
    # -------------------------------------------------------------------------

    async def add_entities(
        self,
        entities: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add entities with embeddings to the index.

        Args:
            entities: List of entity dicts with uuid, name, summary, group_id
            embeddings: List of 3072-dim vectors
        """
        if not entities or not embeddings:
            return

        def _add() -> None:
            db = self._get_db()
            data = [
                {
                    "uuid": e["uuid"],
                    "name": e["name"],
                    "summary": e.get("summary", ""),
                    "group_id": e.get("group_id", self.group_id),
                    "vector": emb,
                }
                for e, emb in zip(entities, embeddings)
            ]

            if self._has_table(db, "entities"):
                table = db.open_table("entities")
                table.add(data)
            else:
                db.create_table("entities", data)

        await asyncio.to_thread(_add)

    async def search_entities(
        self,
        query_vector: list[float],
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Search entities by vector similarity within the current group_id.

        Returns list of (entity_dict, similarity_score) tuples.
        LanceDB returns distance, we convert to similarity (1 - distance).
        """
        def _search() -> list[tuple[dict[str, Any], float]]:
            db = self._get_db()
            if not self._has_table(db, "entities"):
                return []

            table = db.open_table("entities")
            results = (
                table.search(query_vector)
                .metric("cosine")
                .where(f"group_id = '{self.group_id}'")
                .limit(limit)
                .to_arrow()
            )

            output: list[tuple[dict[str, Any], float]] = []
            for i in range(results.num_rows):
                # Cosine distance = 1 - similarity
                distance = results.column("_distance")[i].as_py()
                similarity = 1 - distance

                if similarity >= threshold:
                    output.append((
                        {
                            "uuid": results.column("uuid")[i].as_py(),
                            "name": results.column("name")[i].as_py(),
                            "summary": results.column("summary")[i].as_py(),
                        },
                        similarity,
                    ))
            return output

        return await asyncio.to_thread(_search)

    # -------------------------------------------------------------------------
    # Fact Operations
    # -------------------------------------------------------------------------

    async def add_facts(
        self,
        facts: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Add facts with embeddings to the index."""
        if not facts or not embeddings:
            return

        def _add() -> None:
            db = self._get_db()
            data = [
                {
                    "uuid": f["uuid"],
                    "content": f["content"],
                    "subject_name": f.get("subject_name", ""),
                    "object_name": f.get("object_name", ""),
                    "relationship_type": f.get("relationship_type", ""),
                    "group_id": f.get("group_id", self.group_id),
                    "vector": emb,
                }
                for f, emb in zip(facts, embeddings)
            ]

            if self._has_table(db, "facts"):
                table = db.open_table("facts")
                table.add(data)
            else:
                db.create_table("facts", data)

        await asyncio.to_thread(_add)

    async def search_facts(
        self,
        query_vector: list[float],
        limit: int = 50,
        threshold: float = 0.3,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search facts by vector similarity within the current group_id."""
        def _search() -> list[tuple[dict[str, Any], float]]:
            db = self._get_db()
            if not self._has_table(db, "facts"):
                return []

            table = db.open_table("facts")
            results = (
                table.search(query_vector)
                .metric("cosine")
                .where(f"group_id = '{self.group_id}'")
                .limit(limit)
                .to_arrow()
            )

            output: list[tuple[dict[str, Any], float]] = []
            for i in range(results.num_rows):
                distance = results.column("_distance")[i].as_py()
                similarity = 1 - distance

                if similarity >= threshold:
                    output.append((
                        {
                            "uuid": results.column("uuid")[i].as_py(),
                            "content": results.column("content")[i].as_py(),
                            "subject_name": results.column("subject_name")[i].as_py(),
                            "object_name": results.column("object_name")[i].as_py(),
                            "relationship_type": results.column("relationship_type")[i].as_py(),
                        },
                        similarity,
                    ))
            return output

        return await asyncio.to_thread(_search)

    async def search_facts_by_entity(
        self,
        query_vector: list[float],
        entity_name: str,
        limit: int = 50,
        threshold: float = 0.3,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search facts involving a specific entity within the current group_id."""
        def _search() -> list[tuple[dict[str, Any], float]]:
            db = self._get_db()
            if not self._has_table(db, "facts"):
                return []

            table = db.open_table("facts")
            # Filter by group_id AND entity being subject or object
            # Escape entity_name to prevent SQL injection
            entity_escaped = self._escape_sql_string(entity_name.lower())
            results = (
                table.search(query_vector)
                .metric("cosine")
                .where(
                    f"group_id = '{self.group_id}' AND ("
                    f"LOWER(subject_name) = '{entity_escaped}' OR "
                    f"LOWER(object_name) = '{entity_escaped}')"
                )
                .limit(limit)
                .to_arrow()
            )

            output: list[tuple[dict[str, Any], float]] = []
            for i in range(results.num_rows):
                distance = results.column("_distance")[i].as_py()
                similarity = 1 - distance

                if similarity >= threshold:
                    output.append((
                        {
                            "uuid": results.column("uuid")[i].as_py(),
                            "content": results.column("content")[i].as_py(),
                            "subject_name": results.column("subject_name")[i].as_py(),
                            "object_name": results.column("object_name")[i].as_py(),
                            "relationship_type": results.column("relationship_type")[i].as_py(),
                        },
                        similarity,
                    ))
            return output

        return await asyncio.to_thread(_search)

    async def search_facts_by_uuids(
        self,
        query_vector: list[float],
        fact_uuids: list[str],
        limit: int = 50,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Search facts by semantic similarity, filtered to a specific set of UUIDs.

        This enables two-stage search: structured filter first (DuckDB), then
        semantic ranking within that subset (LanceDB).

        Args:
            query_vector: 3072-dim embedding of the search query
            fact_uuids: List of fact UUIDs to search within
            limit: Maximum results to return
            threshold: Minimum similarity score (default 0.0 = return all)

        Returns:
            List of (uuid, similarity_score) tuples, sorted by similarity descending.
        """
        if not fact_uuids:
            return []

        def _search() -> list[tuple[str, float]]:
            db = self._get_db()
            if not self._has_table(db, "facts"):
                return []

            table = db.open_table("facts")

            # Build UUID filter - use IN clause with escaped UUIDs
            escaped_uuids = [self._escape_sql_string(u) for u in fact_uuids]
            uuid_list = ", ".join(f"'{u}'" for u in escaped_uuids)

            results = (
                table.search(query_vector)
                .metric("cosine")
                .where(f"group_id = '{self.group_id}' AND uuid IN ({uuid_list})")
                .limit(limit)
                .to_arrow()
            )

            output: list[tuple[str, float]] = []
            for i in range(results.num_rows):
                distance = results.column("_distance")[i].as_py()
                similarity = 1 - distance

                if similarity >= threshold:
                    output.append((
                        results.column("uuid")[i].as_py(),
                        similarity,
                    ))
            return output

        return await asyncio.to_thread(_search)

    # -------------------------------------------------------------------------
    # Topic Operations
    # -------------------------------------------------------------------------

    async def add_topics(
        self,
        topics: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Add topics with embeddings to the index."""
        if not topics or not embeddings:
            return

        def _add() -> None:
            db = self._get_db()
            data = [
                {
                    "uuid": t["uuid"],
                    "name": t["name"],
                    "definition": t.get("definition", ""),
                    "group_id": t.get("group_id", self.group_id),
                    "vector": emb,
                }
                for t, emb in zip(topics, embeddings)
            ]

            if self._has_table(db, "topics"):
                table = db.open_table("topics")
                table.add(data)
            else:
                db.create_table("topics", data)

        await asyncio.to_thread(_add)

    async def search_topics(
        self,
        query_vector: list[float],
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search topics by vector similarity within the current group_id."""
        def _search() -> list[tuple[dict[str, Any], float]]:
            db = self._get_db()
            if not self._has_table(db, "topics"):
                return []

            table = db.open_table("topics")
            results = (
                table.search(query_vector)
                .metric("cosine")
                .where(f"group_id = '{self.group_id}'")
                .limit(limit)
                .to_arrow()
            )

            output: list[tuple[dict[str, Any], float]] = []
            for i in range(results.num_rows):
                distance = results.column("_distance")[i].as_py()
                similarity = 1 - distance

                if similarity >= threshold:
                    output.append((
                        {
                            "uuid": results.column("uuid")[i].as_py(),
                            "name": results.column("name")[i].as_py(),
                            "definition": results.column("definition")[i].as_py(),
                            "group_id": results.column("group_id")[i].as_py(),
                        },
                        similarity,
                    ))
            return output

        return await asyncio.to_thread(_search)
