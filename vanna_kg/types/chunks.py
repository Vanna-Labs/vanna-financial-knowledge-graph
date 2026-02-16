"""
Chunk and Document Types

Chunks are segments of source documents with header context.

Storage Models:
    - Chunk: Persisted chunk with UUID and metadata
    - Document: Persisted document metadata

Input Models (used during ingestion):
    - DocumentPayload: Input document with optional bytes for processing
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """
    A persisted text chunk in the knowledge graph.

    Attributes:
        uuid: Unique identifier
        content: The text content
        header_path: Breadcrumb path (e.g., "Section 1 > Overview")
        position: Order within document (0-indexed)
        document_uuid: Parent document ID
        document_date: Date from parent document (ISO format)

    The header_path preserves context from document structure,
    enabling the LLM to understand where the chunk came from.
    """

    uuid: str
    content: str
    header_path: str
    position: int
    document_uuid: str
    document_date: str | None = None
    created_at: str | None = None


class Document(BaseModel):
    """
    A persisted source document in the knowledge graph.

    Attributes:
        uuid: Unique identifier
        name: Document name/title
        document_date: Document date (ISO format)
        source_path: Original file path (for provenance)
        file_type: Type of document (pdf, markdown, text)
        metadata: Additional metadata (JSON-serializable)
    """

    uuid: str
    name: str
    document_date: str | None = None
    source_path: str | None = None
    file_type: str = "pdf"
    metadata: dict[str, Any] = {}
    created_at: str | None = None


# -----------------------------------------------------------------------------
# Input Models (used during ingestion pipeline)
# -----------------------------------------------------------------------------


class DocumentPayload(BaseModel):
    """
    Metadata and optional bytes for a document to be processed.

    Used as input to the chunking pipeline. Can represent either a file path
    or in-memory bytes for processing.
    """

    doc_id: str = Field(..., description="Unique identifier for this document")
    path: Path | None = Field(
        default=None, description="Path to the document file (if on disk)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the document"
    )
    content_bytes: bytes | None = Field(
        default=None, description="Raw bytes of the document (if not using path)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChunkInput(BaseModel):
    """
    Input chunk before UUID assignment.

    Used during the chunking phase before chunks are written to storage.
    """

    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="The text content of the chunk")
    header_path: str = Field(
        default="", description="Breadcrumb path (e.g., 'Section 1 > Overview')"
    )
    position: int = Field(default=0, description="Order within document (0-indexed)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
