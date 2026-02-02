# Chunking System Design

**Date:** 2026-01-29
**Status:** Approved
**Components:** `ingestion/chunking/markdown.py`, `ingestion/chunking/pdf.py`

## Overview

Section-based markdown chunker that preserves document structure via header breadcrumbs, with even paragraph splitting for long sections.

## Interface

```python
def chunk_markdown(
    content: str,
    doc_id: str,
    *,
    max_paragraphs_per_chunk: int = 6,
    min_chunk_chars: int = 50,
) -> list[ChunkInput]:
```

## Algorithm

1. **Parse into sections** - Split on headers, track header stack, build breadcrumb paths
2. **Split long sections** - If section exceeds `max_paragraphs_per_chunk`, split evenly
3. **Filter & build output** - Remove tiny chunks, assign positions, return `ChunkInput` list

## Key Behaviors

| Aspect | Handling |
|--------|----------|
| Header hierarchy | `# A` → `## B` → `### C` = "A > B > C" |
| Header reset | `# A` → `## B` → `# C` resets to "C" |
| Content before first header | Gets empty header_path `""` |
| HTML tables | Kept as single paragraph unit |
| Code blocks | Kept as single paragraph unit |
| Even splitting | 12 paragraphs / threshold 5 → 3 chunks of 4 each |

## Internal Structure

```python
@dataclass
class Section:
    header_path: str
    header_level: int
    content: str

def _parse_sections(content: str) -> list[Section]
def _split_section_evenly(section: Section, max_paragraphs: int) -> list[Section]
def _count_paragraphs(text: str) -> int
def _split_into_paragraphs(text: str) -> list[str]
```

---

## PDF Converter (`pdf.py`)

### Interface

```python
async def pdf_to_markdown(
    pdf_path: Path | str,
    *,
    api_key: str | None = None,
    model: str = "gemini-3-flash",
) -> str:

async def chunk_pdf(
    pdf_path: Path | str,
    doc_id: str,
    *,
    api_key: str | None = None,
    model: str = "gemini-3-flash",
    max_paragraphs_per_chunk: int = 6,
    min_chunk_chars: int = 50,
) -> list[ChunkInput]:
```

### Approach

- Sends entire PDF to Gemini in one call (leverages large context window)
- Gemini "reads" PDF visually and outputs structured markdown
- Tables converted to HTML `<table>` format for preservation
- `chunk_pdf()` is convenience wrapper: `pdf_to_markdown()` → `chunk_markdown()`

---

## Extraction System (`ingestion/extraction/extractor.py`)

### Interface

```python
async def extract_from_chunk(
    chunk: ChunkInput,
    llm: LLMProvider,
    *,
    document_date: str | None = None,
    enable_critique: bool = True,
    max_retries: int = 1,
) -> ChainOfThoughtResult:

async def extract_from_chunks(
    chunks: list[ChunkInput],
    llm: LLMProvider,
    *,
    document_date: str | None = None,
    concurrency: int = 50,  # -1 for unlimited
    enable_critique: bool = True,
    max_retries: int = 1,
) -> list[ChainOfThoughtResult]:
```

### Approach

- **Single LLM call** with two-step structured output (`ChainOfThoughtResult`)
- Step 1: Enumerate ALL entities → `list[EnumeratedEntity]`
- Step 2: Generate relationships → `list[ExtractedFact]`
- **Optional critique** step verifies quality, re-extracts if issues found
- **Semaphore concurrency** with `-1` for unlimited
