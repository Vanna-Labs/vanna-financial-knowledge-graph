"""
PDF to Markdown Converter

Converts PDF files to markdown using Gemini vision, then chunks the result.

Uses the google-genai SDK to send the entire PDF to Gemini for conversion.
The model "reads" the PDF visually and outputs structured markdown that
preserves headers, tables, and formatting.

Example:
    >>> markdown = await pdf_to_markdown("report.pdf")
    >>> chunks = chunk_markdown(markdown, "report-id")

    # Or use the convenience wrapper:
    >>> chunks = await chunk_pdf("report.pdf", "report-id")
"""

import os
from pathlib import Path

from zomma_kg.ingestion.chunking.markdown import chunk_markdown
from zomma_kg.types import ChunkInput

# System prompt for PDF to Markdown conversion
_PDF_TO_MARKDOWN_PROMPT = """Convert this PDF document to well-structured markdown.

Requirements:
- Preserve the document's hierarchical structure using markdown headers (#, ##, ###, etc.)
- Convert tables to HTML <table> format (not markdown tables) for better preservation
- Preserve bullet points and numbered lists
- Keep important formatting (bold, italic) where meaningful
- Extract all text content accurately
- Maintain the logical reading order
- Do not add commentary or summaries - just convert the content faithfully

Output only the markdown, no preamble or explanation."""


async def pdf_to_markdown(
    pdf_path: Path | str,
    *,
    api_key: str | None = None,
    model: str = "gemini-3-flash",
) -> str:
    """
    Convert a PDF file to markdown using Gemini vision.

    Sends the entire PDF to Gemini which "reads" it visually and outputs
    structured markdown preserving headers, tables, and formatting.

    Args:
        pdf_path: Path to the PDF file
        api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
        model: Gemini model to use (default: gemini-3-flash)

    Returns:
        Markdown string representing the PDF content

    Raises:
        ImportError: If google-genai package is not installed
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If API key is not provided and not in environment
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "PDF conversion requires the 'google-genai' package. "
            "Install with: pip install google-genai"
        )

    # Resolve path and validate
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Get API key
    resolved_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Google API key required. Provide api_key parameter or set GOOGLE_API_KEY env var."
        )

    # Create client and upload file
    client = genai.Client(api_key=resolved_api_key)

    # Upload the PDF file
    pdf_file = client.files.upload(file=pdf_path)

    # Generate markdown from PDF
    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            _PDF_TO_MARKDOWN_PROMPT,
            pdf_file,
        ],
    )

    return response.text or ""


async def chunk_pdf(
    pdf_path: Path | str,
    doc_id: str,
    *,
    api_key: str | None = None,
    model: str = "gemini-3-flash",
    max_paragraphs_per_chunk: int = 6,
    min_chunk_chars: int = 50,
) -> list[ChunkInput]:
    """
    Convert a PDF to markdown and chunk it in one step.

    Convenience wrapper that combines pdf_to_markdown() and chunk_markdown().

    Args:
        pdf_path: Path to the PDF file
        doc_id: Document ID to associate with chunks
        api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
        model: Gemini model for PDF conversion (default: gemini-3-flash)
        max_paragraphs_per_chunk: Max paragraphs before splitting a section
        min_chunk_chars: Minimum characters for a chunk to be kept

    Returns:
        List of ChunkInput objects ready for UUID assignment
    """
    markdown = await pdf_to_markdown(pdf_path, api_key=api_key, model=model)

    return chunk_markdown(
        markdown,
        doc_id,
        max_paragraphs_per_chunk=max_paragraphs_per_chunk,
        min_chunk_chars=min_chunk_chars,
    )
