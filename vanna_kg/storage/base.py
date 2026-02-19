"""
Abstract Storage Backend Interface

Defines the contract for all storage backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vanna_kg.types import Chunk, Document, Entity, Fact, Topic


class StorageBackend(ABC):
    """
    Abstract interface for storage backends.

    All storage implementations (Parquet, Neo4j, etc.) must implement this interface.

    Multi-tenancy:
        Each backend is initialized with a group_id for tenant isolation.
        All operations are scoped to that group_id.

    Lifecycle:
        backend = ParquetBackend(path, config, group_id="tenant-1")
        await backend.initialize()
        # ... operations ...
        await backend.close()

    Or using context manager:
        async with ParquetBackend(path, config, group_id="tenant-1") as backend:
            await backend.write_entities(entities)
    """

    @property
    @abstractmethod
    def group_id(self) -> str:
        """Return the group_id (tenant identifier) for this backend."""
        ...

    @property
    @abstractmethod
    def kb_path(self) -> Path:
        """Return the path to the knowledge base directory."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage (create directories, tables, indices)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close storage and release resources."""
        ...

    async def __aenter__(self) -> "StorageBackend":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def write_document(self, document: "Document") -> None:
        """Write a single document."""
        ...

    @abstractmethod
    async def write_chunks(
        self,
        chunks: list["Chunk"],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Write chunks in batch with optional embeddings for vector search."""
        ...

    @abstractmethod
    async def write_entities(
        self,
        entities: list["Entity"],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Write entities with optional embeddings for vector search."""
        ...

    @abstractmethod
    async def write_facts(
        self,
        facts: list["Fact"],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Write facts with optional embeddings for vector search."""
        ...

    @abstractmethod
    async def write_topics(
        self,
        topics: list["Topic"],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Write topics with optional embeddings for vector search."""
        ...

    @abstractmethod
    async def write_relationships(self, relationships: list[dict[str, Any]]) -> None:
        """
        Write relationships (edges).

        Relationship dict: {id, from_uuid, from_type, to_uuid, to_type,
                           rel_type, fact_id, description, date_context}
        """
        ...

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_entity(self, uuid: str) -> "Entity | None":
        """Get entity by UUID."""
        ...

    @abstractmethod
    async def get_entity_by_name(self, name: str) -> "Entity | None":
        """Get entity by name."""
        ...

    @abstractmethod
    async def get_entities(self, uuids: list[str]) -> list["Entity"]:
        """Get multiple entities by UUIDs."""
        ...

    @abstractmethod
    async def get_chunk(self, uuid: str) -> "Chunk | None":
        """Get chunk by UUID."""
        ...

    @abstractmethod
    async def get_chunks(self, uuids: list[str]) -> list["Chunk"]:
        """Get multiple chunks by UUIDs."""
        ...

    @abstractmethod
    async def get_fact(self, uuid: str) -> "Fact | None":
        """Get fact by UUID."""
        ...

    # -------------------------------------------------------------------------
    # Update Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def update_entity_summary(self, uuid: str, summary: str) -> None:
        """Update an entity's summary (used during entity resolution)."""
        ...

    # -------------------------------------------------------------------------
    # Vector Search Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def search_entities(
        self,
        query_vector: list[float],
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[tuple["Entity", float]]:
        """Search entities by vector similarity. Returns (entity, score) tuples."""
        ...

    @abstractmethod
    async def search_facts(
        self,
        query_vector: list[float],
        limit: int = 50,
        threshold: float = 0.3,
    ) -> list[tuple["Fact", float]]:
        """Search facts by vector similarity. Returns (fact, score) tuples."""
        ...

    @abstractmethod
    async def search_topics(
        self,
        query_vector: list[float],
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[tuple["Topic", float]]:
        """Search topics by vector similarity. Returns (topic, score) tuples."""
        ...

    @abstractmethod
    async def search_chunks(
        self,
        query_vector: list[float],
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[tuple["Chunk", float]]:
        """Search chunks by vector similarity. Returns (chunk, score) tuples."""
        ...

    # -------------------------------------------------------------------------
    # Graph Query Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_entity_chunks(
        self,
        entity_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get chunks where entity appears (1-hop: Entity -> Chunk)."""
        ...

    @abstractmethod
    async def get_entity_neighbors(
        self,
        entity_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get neighboring entities (2-hop: Entity -> Chunk -> Entity)."""
        ...

    @abstractmethod
    async def get_topic_chunks(
        self,
        topic_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get chunks for a topic."""
        ...

    @abstractmethod
    async def get_topics_by_names(
        self,
        names: list[str],
    ) -> list["Topic"]:
        """Get topics by their names (case-insensitive). Returns matching Topic objects."""
        ...

    @abstractmethod
    async def get_entity_facts(
        self,
        entity_name: str,
        limit: int = 100,
    ) -> list["Fact"]:
        """Get facts involving an entity (as subject or object)."""
        ...

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @abstractmethod
    async def count_entities(self) -> int:
        """Return total number of entities."""
        ...

    @abstractmethod
    async def count_chunks(self) -> int:
        """Return total number of chunks."""
        ...

    @abstractmethod
    async def count_facts(self) -> int:
        """Return total number of facts."""
        ...

    @abstractmethod
    async def count_documents(self) -> int:
        """Return total number of documents."""
        ...
