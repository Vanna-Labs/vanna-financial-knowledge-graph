"""
DuckDB Query Layer

SQL queries on Parquet files for graph traversal and lookups with multi-tenant support.
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import duckdb

from zomma_kg.config import KGConfig
from zomma_kg.types import Chunk, Document, Entity, EntityType, Fact, Topic


class DuckDBQueries:
    """
    DuckDB query layer for Parquet files with multi-tenant support.

    Handles all SQL queries including:
    - Entity/chunk/fact lookups by UUID or name
    - Graph traversals (1-hop chunks, 2-hop neighbors)
    - Statistics and aggregations

    Multi-tenancy:
        All queries are filtered by group_id for tenant isolation.

    Thread safety:
        Uses thread-local storage for connections since DuckDB connections
        are not thread-safe and asyncio.to_thread() may use different threads.

    DuckDB reads Parquet files directly without loading into memory.
    """

    def __init__(self, kb_path: Path, config: KGConfig, group_id: str = "default"):
        self.kb_path = kb_path
        self.config = config
        self.group_id = group_id
        self._local = threading.local()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize DuckDB (marks as ready, connections created per-thread)."""
        self._initialized = True

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        """Get thread-local DuckDB connection, creating if needed."""
        if not self._initialized:
            raise RuntimeError("DuckDB not initialized. Call initialize() first.")

        # Check for existing thread-local connection
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            # Create new connection for this thread
            conn = duckdb.connect()
            self._register_views(conn)
            self._local.conn = conn
        return conn

    def _register_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Register Parquet files as views if they exist."""
        tables = ["entities", "chunks", "facts", "topics", "documents", "relationships"]

        for table in tables:
            path = self.kb_path / f"{table}.parquet"
            if path.exists():
                conn.execute(f"""
                    CREATE OR REPLACE VIEW {table} AS
                    SELECT * FROM read_parquet('{path}')
                """)

    async def close(self) -> None:
        """Close DuckDB connections."""
        # Note: Thread-local connections will be garbage collected
        # when threads are recycled. We just mark as uninitialized.
        self._initialized = False
        # Close current thread's connection if it exists
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _refresh_view(self, table: str) -> None:
        """Refresh a view after a Parquet file is updated."""
        conn = self._get_conn()
        path = self.kb_path / f"{table}.parquet"
        if path.exists():
            conn.execute(f"""
                CREATE OR REPLACE VIEW {table} AS
                SELECT * FROM read_parquet('{path}')
            """)

    # -------------------------------------------------------------------------
    # Entity Operations
    # -------------------------------------------------------------------------

    async def get_entity(self, uuid: str) -> Entity | None:
        """Get entity by UUID (filtered by group_id)."""
        def _query() -> Entity | None:
            self._refresh_view("entities")
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "SELECT * FROM entities WHERE uuid = ? AND group_id = ?",
                    [uuid, self.group_id]
                ).fetchone()
            except duckdb.CatalogException:
                return None

            if not result:
                return None

            return self._row_to_entity(result, conn)

        return await asyncio.to_thread(_query)

    async def get_entity_by_name(self, name: str) -> Entity | None:
        """Get entity by name (case-insensitive, filtered by group_id)."""
        def _query() -> Entity | None:
            self._refresh_view("entities")
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "SELECT * FROM entities WHERE LOWER(name) = LOWER(?) AND group_id = ?",
                    [name, self.group_id]
                ).fetchone()
            except duckdb.CatalogException:
                return None

            if not result:
                return None

            return self._row_to_entity(result, conn)

        return await asyncio.to_thread(_query)

    async def get_entities(self, uuids: list[str]) -> list[Entity]:
        """Get multiple entities by UUIDs (filtered by group_id)."""
        if not uuids:
            return []

        def _query() -> list[Entity]:
            self._refresh_view("entities")
            conn = self._get_conn()
            placeholders = ",".join(["?" for _ in uuids])
            try:
                results = conn.execute(
                    f"SELECT * FROM entities WHERE uuid IN ({placeholders}) AND group_id = ?",
                    [*uuids, self.group_id]
                ).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_entity(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    async def get_all_entities(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """Get all entities with pagination (filtered by group_id)."""
        def _query() -> list[Entity]:
            self._refresh_view("entities")
            conn = self._get_conn()
            try:
                results = conn.execute(
                    "SELECT * FROM entities WHERE group_id = ? ORDER BY name LIMIT ? OFFSET ?",
                    [self.group_id, limit, offset]
                ).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_entity(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    def _row_to_entity(self, row: tuple[Any, ...], conn: duckdb.DuckDBPyConnection) -> Entity:
        """Convert DuckDB row to Entity model."""
        # Get column names from the query description
        col_names = [desc[0] for desc in conn.description]
        row_dict = dict(zip(col_names, row))

        # Parse aliases from JSON string
        aliases = row_dict.get("aliases", "[]")
        if isinstance(aliases, str):
            aliases = json.loads(aliases) if aliases else []

        return Entity(
            uuid=row_dict["uuid"],
            name=row_dict["name"],
            summary=row_dict.get("summary", ""),
            entity_type=EntityType(row_dict.get("entity_type", "concept")),
            aliases=aliases,
            created_at=row_dict.get("created_at"),
            updated_at=row_dict.get("updated_at"),
        )

    # -------------------------------------------------------------------------
    # Chunk Operations
    # -------------------------------------------------------------------------

    async def get_chunk(self, uuid: str) -> Chunk | None:
        """Get chunk by UUID (filtered by group_id)."""
        def _query() -> Chunk | None:
            self._refresh_view("chunks")
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "SELECT * FROM chunks WHERE uuid = ? AND group_id = ?",
                    [uuid, self.group_id]
                ).fetchone()
            except duckdb.CatalogException:
                return None

            if not result:
                return None

            return self._row_to_chunk(result, conn)

        return await asyncio.to_thread(_query)

    async def get_chunks(self, uuids: list[str]) -> list[Chunk]:
        """Get multiple chunks by UUIDs (filtered by group_id)."""
        if not uuids:
            return []

        def _query() -> list[Chunk]:
            self._refresh_view("chunks")
            conn = self._get_conn()
            placeholders = ",".join(["?" for _ in uuids])
            try:
                results = conn.execute(
                    f"SELECT * FROM chunks WHERE uuid IN ({placeholders}) AND group_id = ?",
                    [*uuids, self.group_id]
                ).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_chunk(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    def _row_to_chunk(self, row: tuple[Any, ...], conn: duckdb.DuckDBPyConnection) -> Chunk:
        """Convert DuckDB row to Chunk model."""
        col_names = [desc[0] for desc in conn.description]
        row_dict = dict(zip(col_names, row))

        return Chunk(
            uuid=row_dict["uuid"],
            content=row_dict["content"],
            header_path=row_dict.get("header_path", ""),
            position=row_dict.get("position", 0),
            document_uuid=row_dict["document_uuid"],
            document_date=row_dict.get("document_date"),
            created_at=row_dict.get("created_at"),
        )

    # -------------------------------------------------------------------------
    # Fact Operations
    # -------------------------------------------------------------------------

    async def get_fact(self, uuid: str) -> Fact | None:
        """Get fact by UUID (filtered by group_id)."""
        def _query() -> Fact | None:
            self._refresh_view("facts")
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "SELECT * FROM facts WHERE uuid = ? AND group_id = ?",
                    [uuid, self.group_id]
                ).fetchone()
            except duckdb.CatalogException:
                return None

            if not result:
                return None

            return self._row_to_fact(result, conn)

        return await asyncio.to_thread(_query)

    async def get_facts_by_uuids(self, uuids: list[str]) -> list[Fact]:
        """Get multiple facts by UUIDs (filtered by group_id)."""
        if not uuids:
            return []

        def _query() -> list[Fact]:
            self._refresh_view("facts")
            conn = self._get_conn()
            placeholders = ",".join(["?" for _ in uuids])
            try:
                results = conn.execute(
                    f"SELECT * FROM facts WHERE uuid IN ({placeholders}) AND group_id = ?",
                    [*uuids, self.group_id]
                ).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_fact(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    async def get_entity_facts(self, entity_name: str, limit: int = 100) -> list[Fact]:
        """Get facts where entity is subject or object (filtered by group_id)."""
        def _query() -> list[Fact]:
            self._refresh_view("facts")
            conn = self._get_conn()
            try:
                results = conn.execute("""
                    SELECT * FROM facts
                    WHERE (LOWER(subject_name) = LOWER(?)
                       OR LOWER(object_name) = LOWER(?))
                       AND group_id = ?
                    LIMIT ?
                """, [entity_name, entity_name, self.group_id, limit]).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_fact(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    async def get_facts_for_entities(
        self,
        entity_names: list[str],
        limit: int = 100,
    ) -> list[Fact]:
        """
        Get facts where ANY of the given entities is subject or object.

        This enables wide-net search across multiple resolved entities,
        supporting the "one name -> many entities" pattern.
        """
        if not entity_names:
            return []

        def _query() -> list[Fact]:
            self._refresh_view("facts")
            conn = self._get_conn()

            # Build case-insensitive match using LOWER()
            lower_names = [n.lower() for n in entity_names]
            placeholders = ", ".join(["?" for _ in lower_names])

            try:
                results = conn.execute(f"""
                    SELECT * FROM facts
                    WHERE (LOWER(subject_name) IN ({placeholders})
                       OR LOWER(object_name) IN ({placeholders}))
                       AND group_id = ?
                    ORDER BY date_context DESC NULLS LAST
                    LIMIT ?
                """, [*lower_names, *lower_names, self.group_id, limit]).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_fact(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    async def get_facts_by_entities(
        self,
        selected_names: list[str],
        mode: str = "around",
        limit: int = 100,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[Fact]:
        """
        Get facts involving selected nodes.

        Modes:
        - around: endpoint overlap with selected set (subject OR object)
        - between: both endpoints inside selected set (subject AND object)
        """
        if not selected_names:
            return []
        if mode not in {"around", "between"}:
            raise ValueError(f"Unsupported mode: {mode}")

        def _query() -> list[Fact]:
            self._refresh_view("facts")
            self._refresh_view("chunks")
            conn = self._get_conn()

            conditions = ["f.group_id = ?"]
            params: list[Any] = [self.group_id]

            selected_lower = [n.lower() for n in selected_names]
            selected_ph = ", ".join(["?" for _ in selected_lower])

            if mode == "between":
                conditions.append(
                    f"(LOWER(f.subject_name) IN ({selected_ph}) "
                    f"AND LOWER(f.object_name) IN ({selected_ph}))"
                )
                params.extend(selected_lower)
                params.extend(selected_lower)
            else:
                conditions.append(
                    f"(LOWER(f.subject_name) IN ({selected_ph}) "
                    f"OR LOWER(f.object_name) IN ({selected_ph}))"
                )
                params.extend(selected_lower)
                params.extend(selected_lower)

            # Date filtering via chunk's document_date
            if from_date:
                conditions.append("c.document_date >= ?")
                params.append(from_date)
            if to_date:
                conditions.append("c.document_date <= ?")
                params.append(to_date)

            where_clause = " AND ".join(conditions)

            try:
                results = conn.execute(f"""
                    SELECT f.* FROM facts f
                    LEFT JOIN chunks c ON f.chunk_uuid = c.uuid AND c.group_id = ?
                    WHERE {where_clause}
                    ORDER BY c.document_date DESC NULLS LAST
                    LIMIT ?
                """, [self.group_id, *params, limit]).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_fact(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    def _row_to_fact(self, row: tuple[Any, ...], conn: duckdb.DuckDBPyConnection) -> Fact:
        """Convert DuckDB row to Fact model."""
        col_names = [desc[0] for desc in conn.description]
        row_dict = dict(zip(col_names, row))

        return Fact(
            uuid=row_dict["uuid"],
            content=row_dict["content"],
            subject_uuid=row_dict["subject_uuid"],
            subject_name=row_dict["subject_name"],
            object_uuid=row_dict["object_uuid"],
            object_name=row_dict["object_name"],
            object_type=row_dict.get("object_type", "entity"),
            relationship_type=row_dict["relationship_type"],
            date_context=row_dict.get("date_context", ""),
            chunk_uuid=row_dict.get("chunk_uuid"),
            created_at=row_dict.get("created_at"),
        )

    # -------------------------------------------------------------------------
    # Graph Traversal Queries
    # -------------------------------------------------------------------------

    async def get_entity_chunks(
        self,
        entity_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get chunks where an entity appears (as subject or object).

        Uses chunk_uuid edge property for 1-hop traversal.
        """
        def _query() -> list[dict[str, Any]]:
            self._refresh_view("entities")
            self._refresh_view("chunks")
            self._refresh_view("relationships")
            self._refresh_view("documents")
            conn = self._get_conn()

            try:
                result = conn.execute("""
                    SELECT DISTINCT
                        c.uuid AS chunk_id,
                        c.content,
                        c.header_path,
                        c.document_uuid,
                        d.name AS doc_name,
                        d.document_date
                    FROM entities e
                    JOIN relationships r ON (r.from_uuid = e.uuid OR r.to_uuid = e.uuid)
                        AND r.group_id = ?
                    JOIN chunks c ON c.uuid = r.chunk_uuid AND c.group_id = ?
                    LEFT JOIN documents d ON d.uuid = c.document_uuid AND d.group_id = ?
                    WHERE LOWER(e.name) = LOWER(?) AND e.group_id = ?
                    LIMIT ?
                """, [self.group_id, self.group_id, self.group_id,
                      entity_name, self.group_id, limit])
                rows = result.fetchall()
                col_names = [desc[0] for desc in result.description]
            except duckdb.CatalogException:
                return []

            return [dict(zip(col_names, row)) for row in rows]

        return await asyncio.to_thread(_query)

    async def get_entity_neighbors(
        self,
        entity_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get neighboring entities directly connected to this entity.

        1-hop traversal via direct entity-entity edges.
        """
        def _query() -> list[dict[str, Any]]:
            self._refresh_view("entities")
            self._refresh_view("relationships")
            conn = self._get_conn()

            try:
                result = conn.execute("""
                    SELECT
                        neighbor.name,
                        neighbor.summary,
                        neighbor.entity_type,
                        COUNT(*) AS connection_count
                    FROM entities e
                    JOIN relationships r ON r.from_uuid = e.uuid
                        AND r.to_type = 'entity'
                        AND r.group_id = ?
                    JOIN entities neighbor ON neighbor.uuid = r.to_uuid
                        AND neighbor.group_id = ?
                    WHERE LOWER(e.name) = LOWER(?)
                        AND e.group_id = ?
                        AND LOWER(neighbor.name) != LOWER(?)
                    GROUP BY neighbor.name, neighbor.summary, neighbor.entity_type
                    ORDER BY connection_count DESC
                    LIMIT ?
                """, [self.group_id, self.group_id,
                      entity_name, self.group_id, entity_name, limit])
                rows = result.fetchall()
                col_names = [desc[0] for desc in result.description]
            except duckdb.CatalogException:
                return []

            return [dict(zip(col_names, row)) for row in rows]

        return await asyncio.to_thread(_query)

    async def get_topic_chunks(
        self,
        topic_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get chunks that mention a topic (topic is the object of a relationship).

        Uses chunk_uuid edge property for lookup.
        """
        def _query() -> list[dict[str, Any]]:
            self._refresh_view("topics")
            self._refresh_view("chunks")
            self._refresh_view("relationships")
            self._refresh_view("documents")
            conn = self._get_conn()

            try:
                result = conn.execute("""
                    SELECT DISTINCT
                        c.uuid AS chunk_id,
                        c.content,
                        c.header_path,
                        c.document_uuid,
                        d.name AS doc_name,
                        d.document_date
                    FROM topics t
                    JOIN relationships r ON r.to_uuid = t.uuid
                        AND r.to_type = 'topic'
                        AND r.group_id = ?
                    JOIN chunks c ON c.uuid = r.chunk_uuid AND c.group_id = ?
                    LEFT JOIN documents d ON d.uuid = c.document_uuid AND d.group_id = ?
                    WHERE LOWER(t.name) = LOWER(?) AND t.group_id = ?
                    LIMIT ?
                """, [self.group_id, self.group_id, self.group_id,
                      topic_name, self.group_id, limit])
                rows = result.fetchall()
                col_names = [desc[0] for desc in result.description]
            except duckdb.CatalogException:
                return []

            return [dict(zip(col_names, row)) for row in rows]

        return await asyncio.to_thread(_query)

    async def get_topics_by_names(
        self,
        names: list[str],
    ) -> list[Topic]:
        """Get topics by their names (case-insensitive, filtered by group_id)."""
        if not names:
            return []

        def _query() -> list[Topic]:
            self._refresh_view("topics")
            conn = self._get_conn()

            # Build case-insensitive match list
            lower_names = [n.lower() for n in names]
            placeholders = ", ".join(["?" for _ in lower_names])

            try:
                results = conn.execute(
                    f"""
                    SELECT uuid, name, definition, parent_topic
                    FROM topics
                    WHERE LOWER(name) IN ({placeholders})
                    AND group_id = ?
                    """,
                    [*lower_names, self.group_id]
                ).fetchall()
            except duckdb.CatalogException:
                return []

            return [
                Topic(
                    uuid=row[0],
                    name=row[1],
                    definition=row[2],
                    parent_topic=row[3],
                )
                for row in results
            ]

        return await asyncio.to_thread(_query)

    # -------------------------------------------------------------------------
    # Statistics (filtered by group_id)
    # -------------------------------------------------------------------------

    async def count_entities(self) -> int:
        """Count total entities for this group_id."""
        return await self._count_table("entities")

    async def count_chunks(self) -> int:
        """Count total chunks for this group_id."""
        return await self._count_table("chunks")

    async def count_facts(self) -> int:
        """Count total facts for this group_id."""
        return await self._count_table("facts")

    async def count_documents(self) -> int:
        """Count total documents for this group_id."""
        return await self._count_table("documents")

    async def _count_table(self, table: str) -> int:
        """Helper to count rows in a table for current group_id."""
        def _query() -> int:
            self._refresh_view(table)
            conn = self._get_conn()
            try:
                result = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE group_id = ?",
                    [self.group_id]
                ).fetchone()
                return result[0] if result else 0
            except duckdb.CatalogException:
                return 0

        return await asyncio.to_thread(_query)

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    async def get_document(self, uuid: str) -> Document | None:
        """Get document by UUID (filtered by group_id)."""
        def _query() -> Document | None:
            self._refresh_view("documents")
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "SELECT * FROM documents WHERE uuid = ? AND group_id = ?",
                    [uuid, self.group_id]
                ).fetchone()
            except duckdb.CatalogException:
                return None

            if not result:
                return None

            return self._row_to_document(result, conn)

        return await asyncio.to_thread(_query)

    async def get_all_documents(self) -> list[Document]:
        """Get all documents (filtered by group_id)."""
        def _query() -> list[Document]:
            self._refresh_view("documents")
            conn = self._get_conn()
            try:
                results = conn.execute(
                    "SELECT * FROM documents WHERE group_id = ? ORDER BY created_at DESC",
                    [self.group_id]
                ).fetchall()
            except duckdb.CatalogException:
                return []

            return [self._row_to_document(row, conn) for row in results]

        return await asyncio.to_thread(_query)

    def _row_to_document(
        self, row: tuple[Any, ...], conn: duckdb.DuckDBPyConnection
    ) -> Document:
        """Convert DuckDB row to Document model."""
        col_names = [desc[0] for desc in conn.description]
        row_dict = dict(zip(col_names, row))

        # Parse metadata from JSON string
        metadata = row_dict.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}

        return Document(
            uuid=row_dict["uuid"],
            name=row_dict["name"],
            document_date=row_dict.get("document_date"),
            source_path=row_dict.get("source_path"),
            file_type=row_dict.get("file_type", "pdf"),
            metadata=metadata,
            created_at=row_dict.get("created_at"),
        )
