"""
Markdown Chunker

Section-based markdown chunking with header breadcrumbs and even paragraph splitting.

Algorithm:
    1. Parse markdown into sections based on headers
    2. Split long sections evenly by paragraph count
    3. Filter small chunks and build ChunkInput objects

See: docs/plans/2026-01-29-markdown-chunker-design.md
"""

import re
from dataclasses import dataclass
from math import ceil

from vanna_kg.types import ChunkInput


@dataclass
class _Section:
    """A parsed section from markdown."""

    header_path: str
    header_level: int  # 1 for #, 2 for ##, etc. (0 = no header/preamble)
    content: str


# Regex patterns
_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
_TABLE_PATTERN = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)


def chunk_markdown(
    content: str,
    doc_id: str,
    *,
    max_paragraphs_per_chunk: int = 6,
    min_chunk_chars: int = 50,
) -> list[ChunkInput]:
    """
    Split markdown into chunks by section, with paragraph-based subdivision.

    Args:
        content: Raw markdown text
        doc_id: Parent document ID for chunk association
        max_paragraphs_per_chunk: Threshold for splitting long sections evenly
        min_chunk_chars: Filter out chunks smaller than this

    Returns:
        List of ChunkInput objects ready for UUID assignment
    """
    # Parse into sections
    sections = _parse_sections(content)

    # Split long sections
    split_sections: list[_Section] = []
    for section in sections:
        if _count_paragraphs(section.content) > max_paragraphs_per_chunk:
            split_sections.extend(
                _split_section_evenly(section, max_paragraphs_per_chunk)
            )
        else:
            split_sections.append(section)

    # Filter and build ChunkInput objects
    chunks: list[ChunkInput] = []
    position = 0

    for section in split_sections:
        text = section.content.strip()
        if len(text) >= min_chunk_chars:
            chunks.append(
                ChunkInput(
                    doc_id=doc_id,
                    content=text,
                    header_path=section.header_path,
                    position=position,
                )
            )
            position += 1

    return chunks


def _parse_sections(content: str) -> list[_Section]:
    """
    Parse markdown into sections based on headers.

    Tracks header hierarchy to build breadcrumb paths like "Intro > Background".
    Content before the first header gets an empty header_path.
    """
    lines = content.split("\n")
    sections: list[_Section] = []

    # Track header stack: [(level, title), ...]
    header_stack: list[tuple[int, str]] = []
    current_content_lines: list[str] = []
    current_header_path = ""
    current_level = 0

    def flush_section() -> None:
        """Save current accumulated content as a section."""
        if current_content_lines:
            content_text = "\n".join(current_content_lines)
            if content_text.strip():
                sections.append(
                    _Section(
                        header_path=current_header_path,
                        header_level=current_level,
                        content=content_text,
                    )
                )

    for line in lines:
        match = _HEADER_PATTERN.match(line)

        if match:
            # Flush previous section
            flush_section()
            current_content_lines = []

            # Parse header
            hashes, title = match.groups()
            level = len(hashes)
            title = title.strip()

            # Update header stack - pop headers at same or deeper level
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()

            header_stack.append((level, title))

            # Build breadcrumb path
            current_header_path = " > ".join(h[1] for h in header_stack)
            current_level = level
        else:
            current_content_lines.append(line)

    # Flush final section
    flush_section()

    return sections


def _split_section_evenly(
    section: _Section,
    max_paragraphs: int,
) -> list[_Section]:
    """
    Split a section into evenly-sized chunks by paragraph.

    For 12 paragraphs with max_paragraphs=5:
        - Need ceil(12/5) = 3 chunks
        - Target size = 12/3 = 4 paragraphs each
    """
    paragraphs = _split_into_paragraphs(section.content)
    n_paragraphs = len(paragraphs)

    if n_paragraphs <= max_paragraphs:
        return [section]

    # Calculate number of chunks needed and target size
    n_chunks = ceil(n_paragraphs / max_paragraphs)
    base_size = n_paragraphs // n_chunks
    remainder = n_paragraphs % n_chunks

    # Distribute paragraphs as evenly as possible
    result: list[_Section] = []
    idx = 0

    for i in range(n_chunks):
        # First 'remainder' chunks get one extra paragraph
        chunk_size = base_size + (1 if i < remainder else 0)
        chunk_paragraphs = paragraphs[idx : idx + chunk_size]
        idx += chunk_size

        chunk_content = "\n\n".join(chunk_paragraphs)
        result.append(
            _Section(
                header_path=section.header_path,
                header_level=section.header_level,
                content=chunk_content,
            )
        )

    return result


def _split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs, keeping tables and code blocks atomic.

    Tables (<table>...</table>) and code blocks (```...```) are treated
    as single paragraph units regardless of internal blank lines.
    """
    # Find all atomic regions (tables and code blocks)
    atomic_regions: list[tuple[int, int, str]] = []

    for pattern in [_TABLE_PATTERN, _CODE_BLOCK_PATTERN]:
        for match in pattern.finditer(text):
            atomic_regions.append((match.start(), match.end(), match.group()))

    # Sort by start position
    atomic_regions.sort(key=lambda x: x[0])

    # If no atomic regions, simple split
    if not atomic_regions:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    # Build list of text segments and atomic blocks
    segments: list[str] = []
    last_end = 0

    for start, end, atomic_text in atomic_regions:
        # Add text before this atomic region
        if start > last_end:
            before_text = text[last_end:start]
            # Split the before_text into paragraphs
            for p in before_text.split("\n\n"):
                if p.strip():
                    segments.append(p.strip())

        # Add the atomic region as a single segment
        segments.append(atomic_text.strip())
        last_end = end

    # Add any remaining text after last atomic region
    if last_end < len(text):
        after_text = text[last_end:]
        for p in after_text.split("\n\n"):
            if p.strip():
                segments.append(p.strip())

    return segments


def _count_paragraphs(text: str) -> int:
    """Count paragraphs, treating tables/code blocks as single units."""
    return len(_split_into_paragraphs(text))
