"""
Document Chunking

Transforms raw documents into semantic chunks suitable for LLM extraction.

Modules:
    pdf: PDF to markdown conversion (Gemini 2.5 Pro vision)
    markdown: Markdown chunking (header-aware splitting)
    text: Plain text chunking (paragraph-based)

Two-Step Process:
    1. PDF -> Markdown (vision model preserves structure)
    2. Markdown -> Chunks (header-aware semantic splitting)

Key Features:
    - Header hierarchy preservation (breadcrumbs)
    - HTML table handling (kept as atomic chunks)
    - Document date extraction
    - Minimum content thresholds

See: docs/pipeline/CHUNKING_SYSTEM.md
"""

from zomma_kg.ingestion.chunking.markdown import chunk_markdown
from zomma_kg.ingestion.chunking.pdf import chunk_pdf, pdf_to_markdown

__all__ = ["chunk_markdown", "chunk_pdf", "pdf_to_markdown"]
