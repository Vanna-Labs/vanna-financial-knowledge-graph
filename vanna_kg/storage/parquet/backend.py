"""
Parquet Storage Backend

Orchestrates Parquet file writing, LanceDB vector indices, and DuckDB queries.
"""

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from vanna_kg.config import KGConfig
from vanna_kg.storage.base import StorageBackend
from vanna_kg.storage.duckdb.queries import DuckDBQueries
from vanna_kg.storage.lancedb.indices import LanceDBIndices
from vanna_kg.types import Chunk, Document, Entity, EntityType, Fact, Topic


class ParquetBackend(StorageBackend):
    """
    Parquet-based storage backend with multi-tenant support.

    Directory structure:
        kb_path/
        ├── entities.parquet
        ├── chunks.parquet
        ├── facts.parquet
        ├── topics.parquet
        ├── documents.parquet
        ├── relationships.parquet
        ├── lancedb/
        │   ├── entities.lance/
        │   ├── facts.lance/
        │   └── topics.lance/
        └── metadata.json

    Multi-tenancy:
        All data is tagged with group_id for tenant isolation.
        Queries are automatically filtered by the backend's group_id.

    Thread safety:
        - Write operations use file locking (.kb.lock)
        - Read operations are concurrent-safe (Parquet is immutable)
    """

    SCHEMA_VERSION = "1.0.0"

    @staticmethod
    def _is_valid_group_id(group_id: str) -> bool:
        """Validate group_id contains only safe characters."""
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', group_id))

    def __init__(
        self,
        kb_path: Path | str,
        config: KGConfig | None = None,
        group_id: str = "default",
    ):
        self._kb_path = Path(kb_path)
        self.config = config or KGConfig()

        # Validate group_id to prevent injection attacks
        if not self._is_valid_group_id(group_id):
            raise ValueError(
                f"Invalid group_id: {group_id}. "
                "Must contain only alphanumeric characters, hyphens, and underscores."
            )
        self._group_id = group_id
        self._lock = FileLock(self._kb_path / ".kb.lock", timeout=30)
        self._lancedb = LanceDBIndices(
            self._kb_path / "lancedb", self.config, group_id=group_id
        )
        self._duckdb = DuckDBQueries(self._kb_path, self.config, group_id=group_id)
        self._initialized = False

    @property
    def group_id(self) -> str:
        """Return the group_id (tenant identifier) for this backend."""
        return self._group_id

    @property
    def kb_path(self) -> Path:
        """Return the path to the knowledge base directory."""
        return self._kb_path

    async def initialize(self) -> None:
        """Initialize storage backend."""
        if self._initialized:
            return

        def _init() -> None:
            self.kb_path.mkdir(parents=True, exist_ok=True)
            self._write_metadata_if_missing()

        await asyncio.to_thread(_init)
        await self._lancedb.initialize()
        await self._duckdb.initialize()
        self._initialized = True

    async def close(self) -> None:
        """Close storage backend."""
        await self._lancedb.close()
        await self._duckdb.close()
        self._initialized = False

    def _write_metadata_if_missing(self) -> None:
        """Create metadata.json if it doesn't exist."""
        meta_path = self.kb_path / "metadata.json"
        if not meta_path.exists():
            metadata = {
                "schema_version": self.SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding_dimensions": 3072,
            }
            meta_path.write_text(json.dumps(metadata, indent=2))

    # -------------------------------------------------------------------------
    # Parquet Schemas (all include group_id for multi-tenancy)
    # -------------------------------------------------------------------------

    @staticmethod
    def _entity_schema() -> pa.Schema:
        return pa.schema([
            ("uuid", pa.string()),
            ("name", pa.string()),
            ("summary", pa.string()),
            ("entity_type", pa.string()),
            ("aliases", pa.string()),  # JSON-encoded list
            ("group_id", pa.string()),
            ("created_at", pa.string()),
            ("updated_at", pa.string()),
        ])

    @staticmethod
    def _chunk_schema() -> pa.Schema:
        return pa.schema([
            ("uuid", pa.string()),
            ("content", pa.string()),
            ("header_path", pa.string()),
            ("position", pa.int32()),
            ("document_uuid", pa.string()),
            ("document_date", pa.string()),
            ("group_id", pa.string()),
            ("created_at", pa.string()),
        ])

    @staticmethod
    def _fact_schema() -> pa.Schema:
        return pa.schema([
            ("uuid", pa.string()),
            ("content", pa.string()),
            ("subject_uuid", pa.string()),
            ("subject_name", pa.string()),
            ("object_uuid", pa.string()),
            ("object_name", pa.string()),
            ("object_type", pa.string()),
            ("relationship_type", pa.string()),
            ("date_context", pa.string()),
            ("chunk_uuid", pa.string()),
            ("group_id", pa.string()),
            ("created_at", pa.string()),
        ])

    @staticmethod
    def _topic_schema() -> pa.Schema:
        return pa.schema([
            ("uuid", pa.string()),
            ("name", pa.string()),
            ("definition", pa.string()),
            ("parent_topic", pa.string()),
            ("group_id", pa.string()),
            ("created_at", pa.string()),
        ])

    @staticmethod
    def _document_schema() -> pa.Schema:
        return pa.schema([
            ("uuid", pa.string()),
            ("name", pa.string()),
            ("document_date", pa.string()),
            ("source_path", pa.string()),
            ("file_type", pa.string()),
            ("metadata", pa.string()),  # JSON-encoded dict
            ("group_id", pa.string()),
            ("created_at", pa.string()),
        ])

    @staticmethod
    def _relationship_schema() -> pa.Schema:
        return pa.schema([
            ("id", pa.string()),
            ("from_uuid", pa.string()),
            ("from_type", pa.string()),
            ("to_uuid", pa.string()),
            ("to_type", pa.string()),
            ("rel_type", pa.string()),
            ("chunk_uuid", pa.string()),  # Source chunk reference for provenance
            ("fact_id", pa.string()),
            ("description", pa.string()),
            ("date_context", pa.string()),
            ("group_id", pa.string()),
            ("created_at", pa.string()),
        ])

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    async def write_document(self, document: Document) -> None:
        """Write a single document."""
        def _write() -> None:
            with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                data = {
                    "uuid": [document.uuid],
                    "name": [document.name],
                    "document_date": [document.document_date or ""],
                    "source_path": [document.source_path or ""],
                    "file_type": [document.file_type],
                    "metadata": [json.dumps(document.metadata)],
                    "group_id": [self._group_id],
                    "created_at": [document.created_at or now],
                }
                self._append_to_parquet("documents", data, self._document_schema())

        await asyncio.to_thread(_write)

    async def write_chunks(self, chunks: list[Chunk]) -> None:
        """Write chunks in batch."""
        if not chunks:
            return

        def _write() -> None:
            with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                data: dict[str, list[Any]] = {
                    "uuid": [c.uuid for c in chunks],
                    "content": [c.content for c in chunks],
                    "header_path": [c.header_path for c in chunks],
                    "position": [c.position for c in chunks],
                    "document_uuid": [c.document_uuid for c in chunks],
                    "document_date": [c.document_date or "" for c in chunks],
                    "group_id": [self._group_id for _ in chunks],
                    "created_at": [c.created_at or now for c in chunks],
                }
                self._append_to_parquet("chunks", data, self._chunk_schema())

        await asyncio.to_thread(_write)

    async def write_entities(
        self,
        entities: list[Entity],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Write entities with optional embeddings."""
        if not entities:
            return

        def _write() -> None:
            with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                data = {
                    "uuid": [e.uuid for e in entities],
                    "name": [e.name for e in entities],
                    "summary": [e.summary for e in entities],
                    "entity_type": [
                        e.entity_type.value if isinstance(e.entity_type, EntityType)
                        else e.entity_type
                        for e in entities
                    ],
                    "aliases": [json.dumps(e.aliases) for e in entities],
                    "group_id": [self._group_id for _ in entities],
                    "created_at": [e.created_at or now for e in entities],
                    "updated_at": [e.updated_at or now for e in entities],
                }
                self._append_to_parquet("entities", data, self._entity_schema())

        await asyncio.to_thread(_write)

        # Index in LanceDB if embeddings provided
        if embeddings:
            entity_dicts = [
                {
                    "uuid": e.uuid,
                    "name": e.name,
                    "summary": e.summary,
                    "group_id": self._group_id,
                }
                for e in entities
            ]
            await self._lancedb.add_entities(entity_dicts, embeddings)

    async def write_facts(
        self,
        facts: list[Fact],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Write facts with optional embeddings."""
        if not facts:
            return

        def _write() -> None:
            with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                data = {
                    "uuid": [f.uuid for f in facts],
                    "content": [f.content for f in facts],
                    "subject_uuid": [f.subject_uuid for f in facts],
                    "subject_name": [f.subject_name for f in facts],
                    "object_uuid": [f.object_uuid for f in facts],
                    "object_name": [f.object_name for f in facts],
                    "object_type": [f.object_type for f in facts],
                    "relationship_type": [f.relationship_type for f in facts],
                    "date_context": [f.date_context for f in facts],
                    "chunk_uuid": [f.chunk_uuid or "" for f in facts],
                    "group_id": [self._group_id for _ in facts],
                    "created_at": [f.created_at or now for f in facts],
                }
                self._append_to_parquet("facts", data, self._fact_schema())

        await asyncio.to_thread(_write)

        # Index in LanceDB if embeddings provided
        if embeddings:
            fact_dicts = [
                {
                    "uuid": f.uuid,
                    "content": f.content,
                    "subject_name": f.subject_name,
                    "object_name": f.object_name,
                    "relationship_type": f.relationship_type,
                    "group_id": self._group_id,
                }
                for f in facts
            ]
            await self._lancedb.add_facts(fact_dicts, embeddings)

    async def write_topics(
        self,
        topics: list[Topic],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Write topics with optional embeddings."""
        if not topics:
            return

        def _write() -> None:
            with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                data = {
                    "uuid": [t.uuid for t in topics],
                    "name": [t.name for t in topics],
                    "definition": [t.definition or "" for t in topics],
                    "parent_topic": [t.parent_topic or "" for t in topics],
                    "group_id": [self._group_id for _ in topics],
                    "created_at": [t.created_at or now for t in topics],
                }
                self._append_to_parquet("topics", data, self._topic_schema())

        await asyncio.to_thread(_write)

        # Index in LanceDB if embeddings provided
        if embeddings:
            topic_dicts = [
                {
                    "uuid": t.uuid,
                    "name": t.name,
                    "definition": t.definition or "",
                    "group_id": self._group_id,
                }
                for t in topics
            ]
            await self._lancedb.add_topics(topic_dicts, embeddings)

    async def write_relationships(self, relationships: list[dict[str, Any]]) -> None:
        """Write relationships (edges)."""
        if not relationships:
            return

        def _write() -> None:
            with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                data = {
                    "id": [r.get("id", str(uuid4())) for r in relationships],
                    "from_uuid": [r["from_uuid"] for r in relationships],
                    "from_type": [r["from_type"] for r in relationships],
                    "to_uuid": [r["to_uuid"] for r in relationships],
                    "to_type": [r["to_type"] for r in relationships],
                    "rel_type": [r["rel_type"] for r in relationships],
                    "chunk_uuid": [r.get("chunk_uuid", "") for r in relationships],
                    "fact_id": [r.get("fact_id", "") for r in relationships],
                    "description": [r.get("description", "") for r in relationships],
                    "date_context": [r.get("date_context", "") for r in relationships],
                    "group_id": [self._group_id for _ in relationships],
                    "created_at": [r.get("created_at", now) for r in relationships],
                }
                self._append_to_parquet("relationships", data, self._relationship_schema())

        await asyncio.to_thread(_write)

    def _append_to_parquet(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        schema: pa.Schema,
    ) -> None:
        """
        Append data using a Parquet dataset directory.

        Strategy:
        - New writes append immutable part files (no read/concat rewrite).
        - Legacy single-file tables are migrated in-place on first append.
        """
        path = self.kb_path / f"{table_name}.parquet"
        table = pa.Table.from_pydict(data, schema=schema)
        self._ensure_dataset_path(path)

        now_part = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        part_name = f"part-{now_part}-{uuid4().hex}.parquet"
        part_path = path / part_name
        temp_part_path = path / f".{part_name}.tmp"
        pq.write_table(table, temp_part_path, compression="zstd")
        temp_part_path.replace(part_path)

    def _ensure_dataset_path(self, path: Path) -> None:
        """Ensure table path is a dataset directory, migrating legacy file if needed."""
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return

        if path.is_dir():
            return

        # Migrate legacy single-file table to dataset directory.
        migrated_dir = path.with_name(f"{path.name}.migrating-{uuid4().hex}")
        migrated_dir.mkdir(parents=True, exist_ok=False)
        migrated_part = migrated_dir / "part-000000.parquet"
        path.replace(migrated_part)
        migrated_dir.replace(path)

    # -------------------------------------------------------------------------
    # Read Operations (delegate to DuckDB)
    # -------------------------------------------------------------------------

    async def get_entity(self, uuid: str) -> Entity | None:
        return await self._duckdb.get_entity(uuid)

    async def get_entity_by_name(self, name: str) -> Entity | None:
        return await self._duckdb.get_entity_by_name(name)

    async def get_entities(self, uuids: list[str]) -> list[Entity]:
        return await self._duckdb.get_entities(uuids)

    async def get_chunk(self, uuid: str) -> Chunk | None:
        return await self._duckdb.get_chunk(uuid)

    async def get_chunks(self, uuids: list[str]) -> list[Chunk]:
        return await self._duckdb.get_chunks(uuids)

    async def get_fact(self, uuid: str) -> Fact | None:
        return await self._duckdb.get_fact(uuid)

    async def get_document(self, uuid: str) -> Document | None:
        return await self._duckdb.get_document(uuid)

    async def get_all_documents(self) -> list[Document]:
        return await self._duckdb.get_all_documents()

    # -------------------------------------------------------------------------
    # Update Operations
    # -------------------------------------------------------------------------

    async def update_entity_summary(self, uuid: str, summary: str) -> None:
        """Update an entity's summary by rewriting the Parquet file atomically."""
        def _update() -> None:
            with self._lock:
                path = self.kb_path / "entities.parquet"
                if not path.exists():
                    return

                # Read existing table
                table = pq.read_table(path)

                # Find the row to update (must match uuid AND group_id)
                uuid_col = table.column("uuid")
                group_col = table.column("group_id")
                row_idx = None
                for i in range(table.num_rows):
                    if (uuid_col[i].as_py() == uuid and
                            group_col[i].as_py() == self._group_id):
                        row_idx = i
                        break

                if row_idx is None:
                    return

                # Build new columns with updated values
                now = datetime.now(timezone.utc).isoformat()
                new_columns: dict[str, list[Any]] = {}
                for col_name in table.column_names:
                    col = table.column(col_name)
                    values = [col[i].as_py() for i in range(table.num_rows)]

                    if col_name == "summary":
                        values[row_idx] = summary
                    elif col_name == "updated_at":
                        values[row_idx] = now

                    new_columns[col_name] = values

                # Write to temp file, then atomic rename (prevents corruption)
                new_table = pa.Table.from_pydict(new_columns, schema=self._entity_schema())
                if path.is_dir():
                    temp_dir = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
                    temp_dir.mkdir(parents=True, exist_ok=False)
                    pq.write_table(new_table, temp_dir / "part-000000.parquet", compression="zstd")

                    backup_dir = path.with_name(f"{path.name}.bak-{uuid4().hex}")
                    path.replace(backup_dir)
                    temp_dir.replace(path)
                    shutil.rmtree(backup_dir, ignore_errors=True)
                else:
                    temp_path = path.with_suffix(".parquet.tmp")
                    pq.write_table(new_table, temp_path, compression="zstd")
                    temp_path.replace(path)  # Atomic on POSIX systems

        await asyncio.to_thread(_update)

    # -------------------------------------------------------------------------
    # Vector Search Operations (delegate to LanceDB)
    # -------------------------------------------------------------------------

    async def search_entities(
        self,
        query_vector: list[float],
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[tuple[Entity, float]]:
        """Search entities by vector similarity."""
        results = await self._lancedb.search_entities(query_vector, limit, threshold)

        # Enrich with full entity data from Parquet
        if not results:
            return []

        uuids = [r[0]["uuid"] for r in results]
        entities = await self._duckdb.get_entities(uuids)

        # Match scores to entities
        uuid_to_score = {r[0]["uuid"]: r[1] for r in results}
        return [(e, uuid_to_score.get(e.uuid, 0)) for e in entities]

    async def search_facts(
        self,
        query_vector: list[float],
        limit: int = 50,
        threshold: float = 0.3,
    ) -> list[tuple[Fact, float]]:
        """Search facts by vector similarity."""
        results = await self._lancedb.search_facts(query_vector, limit, threshold)

        if not results:
            return []

        uuids = [r[0]["uuid"] for r in results]
        facts = await self._duckdb.get_facts_by_uuids(uuids)

        uuid_to_score = {r[0]["uuid"]: r[1] for r in results}
        return [(f, uuid_to_score.get(f.uuid, 0)) for f in facts]

    async def search_topics(
        self,
        query_vector: list[float],
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[tuple[Topic, float]]:
        """Search topics by vector similarity."""
        results = await self._lancedb.search_topics(query_vector, limit, threshold)

        if not results:
            return []

        # For topics, LanceDB has all the data we need
        return [
            (
                Topic(
                    uuid=r[0]["uuid"],
                    name=r[0]["name"],
                    definition=r[0]["definition"],
                ),
                r[1],
            )
            for r in results
        ]

    # -------------------------------------------------------------------------
    # Graph Query Operations (delegate to DuckDB)
    # -------------------------------------------------------------------------

    async def get_entity_chunks(
        self,
        entity_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._duckdb.get_entity_chunks(entity_name, limit)

    async def get_entity_neighbors(
        self,
        entity_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return await self._duckdb.get_entity_neighbors(entity_name, limit)

    async def get_topic_chunks(
        self,
        topic_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._duckdb.get_topic_chunks(topic_name, limit)

    async def get_topics_by_names(
        self,
        names: list[str],
    ) -> list[Topic]:
        """Get topics by their names (case-insensitive)."""
        return await self._duckdb.get_topics_by_names(names)

    async def get_entity_facts(
        self,
        entity_name: str,
        limit: int = 100,
    ) -> list[Fact]:
        return await self._duckdb.get_entity_facts(entity_name, limit)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def count_entities(self) -> int:
        return await self._duckdb.count_entities()

    async def count_chunks(self) -> int:
        return await self._duckdb.count_chunks()

    async def count_facts(self) -> int:
        return await self._duckdb.count_facts()

    async def count_documents(self) -> int:
        return await self._duckdb.count_documents()
