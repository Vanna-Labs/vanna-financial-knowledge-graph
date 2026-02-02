# ZommaKG Chunking System

## Executive Summary

The chunking system is the foundational preprocessing stage of the ZommaKG knowledge graph pipeline. Its purpose is to transform raw PDF documents into semantically meaningful text segments (chunks) that can be processed by downstream extraction components.

Effective chunking is critical because:

1. **Extraction Quality**: LLMs perform entity and relationship extraction on individual chunks. If a chunk lacks context or splits mid-sentence, the extraction quality degrades significantly.

2. **Semantic Coherence**: Each chunk should represent a complete thought or logical unit. This enables the extraction model to understand the full meaning without needing cross-chunk context.

3. **Header Context Preservation**: Financial and regulatory documents rely heavily on hierarchical structure. A paragraph about "Revenue" has entirely different meaning under "Risk Factors" versus "Financial Highlights."

4. **Downstream Retrieval**: Chunks become the atomic units for vector search and question answering. Well-formed chunks with proper metadata enable accurate retrieval.

The chunking system outputs structured JSON records containing the text content, breadcrumb hierarchy, and document identifiers that flow through the entire pipeline.

---

## Two-Step Conversion Process

The chunking system employs a two-step process to convert PDFs into semantic chunks:

### Step 1: PDF to Markdown Conversion

Raw PDF files are converted to well-structured Markdown using a vision-capable language model (Gemini 2.5 Pro). This step handles:

- Visual layout interpretation
- Text extraction with structure preservation
- Header hierarchy reconstruction
- Table conversion to HTML format
- Removal of headers, footers, and page numbers

### Step 2: Markdown to Semantic Chunks

The Markdown output is then parsed into individual chunks using a paragraph-boundary algorithm that:

- Splits on blank lines (paragraph boundaries)
- Tracks header hierarchy using a stack-based approach
- Generates breadcrumb navigation paths
- Keeps HTML tables as single atomic units
- Filters out short or trivial content

This two-step approach provides flexibility: documents that are already in Markdown format can skip Step 1 and proceed directly to chunking.

---

## PDF Conversion

### Why Vision Models for PDF Parsing

Traditional PDF parsing libraries (PyPDF2, pdfplumber, etc.) extract text in reading order but lose structural information. They cannot reliably determine:

- Which text is a header versus body text
- The hierarchy level of headers
- Table cell boundaries and relationships
- Multi-column layout flow

Vision-capable models like Gemini 2.5 Pro can "see" the document as rendered, understanding:

- Font sizes and weights that indicate headers
- Spatial relationships that define structure
- Table layouts regardless of underlying PDF encoding
- Visual separators and section breaks

This visual understanding produces significantly higher-quality structured output than text-only extraction.

### Conversion Prompt Engineering

The conversion prompt is carefully engineered to produce consistent, parseable output:

```
Convert the contents of this PDF into well-formatted Markdown.
Preserve all structural elements like headings, lists, tables, and paragraphs.
Maintain the original formatting and hierarchy. Ensure the output is clean.
You can remove the headers and footers.
All headers should use # where the number of # indicates their level (# for h1, ## for h2, etc.).
All tables MUST be formatted as HTML tables using <table>, <tr>, <th>, and <td> tags. Do NOT use markdown pipe tables.
```

Key prompt design decisions:

1. **Explicit Header Notation**: Requiring `#` notation for headers ensures consistent parsing downstream. Headers must use 1-6 `#` symbols corresponding to h1-h6 levels.

2. **HTML Tables**: Markdown pipe tables have limited capability for complex structures (merged cells, nested content). HTML tables preserve all structural information and are easier to keep as atomic chunks.

3. **Header/Footer Removal**: Page headers (document titles, dates) and footers (page numbers, disclaimers) are noise that would create meaningless chunks or pollute extraction.

4. **Structure Preservation**: The prompt emphasizes preserving original hierarchy rather than flattening or reorganizing content.

### Model Configuration

The conversion uses specific generation parameters:

- **Temperature**: 0.1 (low) - Faithful reproduction rather than creative interpretation
- **Max Output Tokens**: 100,000 - Sufficient for full document conversion
- **Model**: gemini-2.5-pro - Required for vision capabilities and context length

### Header Hierarchy Maintenance

The vision model reconstructs header hierarchy from visual cues:

1. **Font Size**: Larger text typically indicates higher-level headers
2. **Font Weight**: Bold text often indicates headers
3. **Spacing**: Headers usually have increased spacing before/after
4. **Numbering**: Explicit numbering (1.0, 1.1, 1.1.1) indicates hierarchy
5. **Position**: Top-of-page text after page breaks may indicate section starts

The model outputs headers with appropriate `#` counts:
- `#` (h1) - Document title, major sections
- `##` (h2) - Section headings
- `###` (h3) - Subsection headings
- And so on through `######` (h6)

### Table Handling

Tables are converted to HTML format rather than Markdown pipe tables:

**Input (visual table in PDF):**
```
| Year | Revenue | Employees |
|------|---------|-----------|
| 2023 | $100B   | 10,000    |
| 2024 | $120B   | 12,000    |
```

**Output (HTML):**
```html
<table>
  <thead>
    <tr>
      <th>Year</th>
      <th>Revenue</th>
      <th>Employees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2023</td>
      <td>$100B</td>
      <td>10,000</td>
    </tr>
    <tr>
      <td>2024</td>
      <td>$120B</td>
      <td>12,000</td>
    </tr>
  </tbody>
</table>
```

HTML tables support:
- Complex cell spans (colspan, rowspan)
- Nested tables
- Mixed content (text, lists)
- Styling hints that may aid understanding

---

## Markdown Chunking Algorithm

### Overview

The chunking algorithm processes Markdown text line-by-line, maintaining state about the current header hierarchy and accumulating paragraph content. It produces one chunk per semantic unit (paragraph or table).

### Algorithm Pseudocode

```
INITIALIZE:
  header_stack = []           # Stack of (level, title) tuples
  current_paragraph = []      # Lines being accumulated
  chunks = []                 # Output chunks
  chunk_counter = 0
  inside_table = False

FOR EACH line IN document:

  IF line contains '<table':
    flush_paragraph()
    inside_table = True
    append line to current_paragraph
    IF line also contains '</table>':
      inside_table = False
      flush_paragraph()
    CONTINUE

  IF inside_table:
    append line to current_paragraph
    IF line contains '</table>':
      inside_table = False
      flush_paragraph()
    CONTINUE

  IF line matches header pattern (^#{1,6}\s+.+$):
    flush_paragraph()
    level = count of # symbols
    title = text after # symbols

    # Pop headers at same or deeper level
    WHILE header_stack not empty AND top level >= level:
      pop header_stack

    push (level, title) to header_stack

  ELSE IF line is blank:
    flush_paragraph()

  ELSE:
    append line to current_paragraph

# Flush any remaining content
flush_paragraph()

FUNCTION flush_paragraph():
  IF current_paragraph is empty:
    RETURN

  text = join current_paragraph with newlines, strip whitespace

  # Skip short content
  IF length(text) < MIN_PARAGRAPH_LENGTH:
    clear current_paragraph
    RETURN

  # Skip horizontal rules
  IF text starts with '---':
    clear current_paragraph
    RETURN

  increment chunk_counter
  breadcrumbs = [title for (level, title) in header_stack]
  header_path = join breadcrumbs with ' > '

  create Chunk:
    chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
    doc_id = document identifier
    body = text
    breadcrumbs = breadcrumbs list
    header_path = header_path (or "Document" if empty)

  append Chunk to chunks
  clear current_paragraph
```

### Paragraph Boundary Splitting

The algorithm treats blank lines as paragraph boundaries. This is the natural structure of Markdown where:

- A single blank line separates paragraphs
- Content between blank lines forms a logical unit
- Multiple consecutive blank lines are treated identically to one

This approach works well for:
- Standard prose paragraphs
- Bullet lists (which don't have internal blank lines)
- Code blocks (which maintain internal newlines)

### Header Stack Maintenance

The header stack tracks the current position in the document hierarchy. When a new header is encountered:

1. **Pop Deeper Headers**: Any headers at the same level or deeper are removed from the stack. For example, if the stack contains [h1, h2, h3] and we encounter a new h2, we pop h3 and h2, leaving [h1].

2. **Push New Header**: The new header is added to the stack, making it [h1, h2-new].

This maintains the invariant that the stack always represents a valid path from root to current position, with strictly increasing header levels.

**Example:**

```markdown
# Chapter 1           # Stack: [(1, "Chapter 1")]
## Section 1.1        # Stack: [(1, "Chapter 1"), (2, "Section 1.1")]
### Subsection 1.1.1  # Stack: [(1, "Chapter 1"), (2, "Section 1.1"), (3, "Subsection 1.1.1")]
## Section 1.2        # Pop h3, h2 -> Stack: [(1, "Chapter 1"), (2, "Section 1.2")]
# Chapter 2           # Pop h2, h1 -> Stack: [(1, "Chapter 2")]
```

### Breadcrumb Generation

Breadcrumbs are generated by extracting the titles from the header stack:

```python
breadcrumbs = [title for (level, title) in header_stack]
```

For visualization, the header_path joins breadcrumbs with ` > `:

```
header_path = "Chapter 1 > Section 1.2 > Details"
```

If the stack is empty (content before any headers), the header_path defaults to "Document".

### HTML Table Handling

HTML tables require special handling because they may contain blank lines internally (for readability) that should not trigger paragraph splits.

**Detection:**
- Table start: Line contains `<table` (case-insensitive)
- Table end: Line contains `</table>` (case-insensitive)

**Behavior:**
1. When `<table` is detected, flush any pending paragraph
2. Set `inside_table = True`
3. Accumulate all lines until `</table>` is found
4. The entire table becomes a single chunk

This ensures complex tables with multiple rows and internal formatting remain atomic.

### Minimum Content Thresholds

The algorithm filters out chunks that are too short to be meaningful:

- **Default Threshold**: 50 characters
- **Purpose**: Eliminate noise from:
  - Single-word headers that got orphaned
  - Page numbers
  - Footer remnants
  - Empty table cells

Content below the threshold is silently discarded.

**Additional filters:**
- Horizontal rules (`---`) are skipped regardless of length
- Pure whitespace content is skipped

---

## Chunk Data Structure

Each chunk is represented as a structured object with the following fields:

### chunk_id

**Format:** `{doc_id}_chunk_{sequence:04d}`

**Examples:**
- `alphabet_chunk_0001`
- `beigebook_20251015_chunk_0042`
- `10k_annual_report_chunk_0123`

**Properties:**
- Globally unique when combined with doc_id
- Zero-padded to 4 digits for lexicographic sorting
- Sequential ordering reflects document reading order
- Deterministic: re-chunking same document produces same IDs

### doc_id

**Format:** Lowercase, underscores replacing spaces

**Derivation:** Generated from the source filename stem:
```python
doc_id = filename_stem.replace(" ", "_").lower()
```

**Examples:**
- `BeigeBook_20251015.pdf` -> `beigebook_20251015`
- `Alphabet Inc.pdf` -> `alphabet_inc.`
- `10-K Annual Report.md` -> `10-k_annual_report`

### body

**Content:** The actual text content of the chunk.

**Properties:**
- Stripped of leading/trailing whitespace
- Preserves internal newlines (important for lists)
- May contain HTML (for tables)
- Minimum length threshold applied

### breadcrumbs

**Type:** List of strings

**Content:** Ordered list of header titles from root to current position.

**Examples:**
```json
["Alphabet Inc.", "History"]
["Financial Statements", "Balance Sheet", "Assets"]
[]  // Content before any headers
```

**Usage:** Provides hierarchical context for extraction and retrieval.

### header_path

**Type:** String

**Content:** Human-readable path joining breadcrumbs with ` > `.

**Examples:**
```
"Alphabet Inc. > History"
"Financial Statements > Balance Sheet > Assets"
"Document"  // Default when breadcrumbs is empty
```

**Usage:** Displayed in UIs and prepended to chunk text for LLM context.

### Metadata Fields (Extended Schema)

The validation schema supports additional optional fields:

- **heading**: Primary section header (first breadcrumb)
- **subheading**: Secondary header (second breadcrumb or "Body")
- **relationships**: Extracted relationships (populated by pipeline)
- **metadata**: Arbitrary key-value pairs including:
  - `page_numbers`: Source PDF pages
  - `document_date`: Extracted publication date
  - `content_type`: Classification (text, table, etc.)
  - `origin_filename`: Original source file

---

## Document Date Extraction

### Purpose

Document dates provide critical temporal context for knowledge graph construction. A statement "Revenue increased 20%" is meaningless without knowing when the document was published. The date extraction system identifies the publication or creation date of each document.

### LLM Analysis Approach

The system uses an LLM (Gemini Flash or equivalent "nano" model) to analyze the first and last chunks of each document:

**Why first and last chunks?**
- Title pages typically appear at the beginning
- Publication dates often appear in headers/footers
- Document metadata may appear at the end
- Body text contains historical dates that should be ignored

**Input structure:**
```
--- DOCUMENT TITLE: {title} ---

--- BEGINNING OF DOCUMENT ---
[First 6 chunks joined with "..."]

--- END OF DOCUMENT ---
[Last 6 chunks joined with "..."]
```

### Extraction Prompt

The extraction prompt provides clear instructions:

```
You are a specialist in temporal data extraction.
Your TASK: Determine the Creation Date or Publication Date of this document.

[Document context inserted here]

RULES:
1. LOOK FOR: Title page dates, headers, footers, 'Published on', 'Date:', 'As of'.
2. IGNORE: Historical dates mentioned in the text (e.g. 'In 1990, the company...').
   ONLY extract the date of the document itself.
3. IF NOT FOUND: Return date_found=False.
4. IF PARTIAL: If valid Year/Month found but no Day, return them.
```

### Structured Output

The LLM returns a structured response:

```json
{
  "date_found": true,
  "year": 2024,
  "month": 12,
  "day": 15,
  "reasoning": "Document header states 'As of December 15, 2024'"
}
```

**Field definitions:**
- `date_found`: Boolean indicating if a valid date was located
- `year`: Four-digit year (required if date_found)
- `month`: Month number 1-12 (optional, defaults to 1)
- `day`: Day number 1-31 (optional, defaults to 1)
- `reasoning`: Quote from document supporting the date

### Temporal Context Importance

Document dates are used throughout the pipeline:

1. **Extraction Context**: The extraction prompt receives the document date to help interpret relative time references ("last quarter", "this year").

2. **Fact Timestamps**: Extracted facts can be associated with the document date for temporal reasoning.

3. **Query Filtering**: Users can filter knowledge graph queries by time period.

4. **Versioning**: Multiple versions of the same document (e.g., quarterly reports) are distinguished by date.

### Fallback Strategies

When the LLM cannot determine a date:

1. **Partial Date**: If year and month are found but not day, default day to 1.
2. **Invalid Date**: If the constructed date is invalid (e.g., Feb 30), return None.
3. **Not Found**: If no date is found, return None and let downstream components handle the absence.

The pipeline does not artificially inject today's date as a fallback, as this would be misleading for historical documents.

---

## Configuration Options

### Minimum Paragraph Length

**Parameter:** `skip_short_paragraphs`

**Default:** 50 characters

**Purpose:** Filters out trivial content that would create noise in extraction.

**Tradeoffs:**
- **Too Low (< 20)**: Noise from fragments, page numbers, orphaned words
- **Too High (> 100)**: May lose legitimate short paragraphs, bullet items

**Recommendation:** Start with default, adjust based on document corpus characteristics.

### Batch Sizes

**Embedding Batch Size:** 100 texts per batch

Controls how many chunk texts are sent to the embedding model at once. Larger batches are more efficient but require more memory.

**Dedup Concurrency:** Derived from extraction concurrency

Controls parallel processing during entity deduplication phase.

### Model Selection

**PDF Conversion Model:** `gemini-2.5-pro`

Required for vision capabilities. The model must support:
- PDF input as base64-encoded bytes
- Large context windows (100k+ tokens)
- Structured output generation

**Date Extraction Model:** Configurable "nano" model

Uses cheaper, faster models since the task is straightforward:
- Default: `gemini-2.5-flash-lite`
- Alternative: `gpt-5-mini`

**Extraction Model:** Separate configuration

The main extraction model is configured separately and typically uses more capable models for complex reasoning.

### Document ID Generation

**Function:** `document_id_from_path(path)`

**Algorithm:**
```python
return path.stem.replace(" ", "_").lower()
```

**Ensures:**
- No spaces (filesystem-safe)
- Lowercase (case-insensitive matching)
- Derived from filename (not path) for portability

---

## Quality Considerations

### Why Semantic Chunking Matters

Naive chunking strategies (fixed token count, character limits) create poor extraction results:

**Fixed-size chunks problems:**
1. **Mid-sentence splits**: "The company reported revenue of" ... "[chunk boundary]" ... "$100 million for Q4."
2. **Context loss**: A number without its label is meaningless
3. **Relationship breaks**: Subject and object may end up in different chunks

**Semantic chunking advantages:**
1. **Complete thoughts**: Each chunk contains full paragraphs
2. **Preserved structure**: Tables remain atomic
3. **Header context**: Breadcrumbs provide missing context

### Header Context for LLM Understanding

The extraction pipeline prepends header paths to chunk text:

```
[Financial Statements > Balance Sheet > Current Assets]

Cash and cash equivalents totaled $25.3 billion at year end.
```

This provides critical context:
- "Cash" is clearly a Balance Sheet item, not operational cash flow
- "Current Assets" classifies the nature of the amount
- The LLM can extract proper entity types and relationships

Without headers:
```
Cash and cash equivalents totaled $25.3 billion at year end.
```

The LLM might misclassify this or miss important context about what kind of cash is being discussed.

### Chunk Size Tradeoffs

**Small chunks (< 100 chars):**
- Pros: Precise retrieval, focused extraction
- Cons: Loss of context, overhead per chunk, fragmented facts

**Large chunks (> 2000 chars):**
- Pros: Full context, efficient processing
- Cons: Diluted relevance for retrieval, may exceed context windows

**Optimal range (200-1500 chars):**
- Complete paragraphs maintain coherence
- Headers provide additional context
- Reasonable extraction and retrieval performance

The paragraph-boundary approach naturally produces chunks in this optimal range for most business documents.

### Table Preservation

Tables are kept as single chunks regardless of size because:

1. **Relational Structure**: A table represents relationships between headers and values that cannot be understood in isolation.

2. **Extraction Quality**: LLMs can extract multiple facts from a single table, but need the full table to understand column relationships.

3. **Visual Integrity**: Tables are often the most information-dense parts of financial documents.

**Tradeoff:** Very large tables may exceed ideal chunk sizes, but the alternative (splitting tables) produces worse results.

---

## Implementation Notes for Reimplementation

### Required Dependencies

1. **PDF Processing**: Google Generative AI SDK for Gemini API access
2. **Markdown Parsing**: No external parser needed; line-by-line regex-based parsing
3. **Data Structures**: Dataclass or Pydantic models for chunks
4. **Serialization**: JSON/JSONL output support

### Key Regular Expressions

**Header Pattern:**
```regex
^(#{1,6})\s+(.+)$
```
- Group 1: The `#` symbols (1-6)
- Group 2: The header title text

**Table Detection:**
```python
'<table' in line.lower()  # Start
'</table>' in line.lower()  # End
```

### Edge Cases to Handle

1. **Empty Documents**: Return empty chunk list
2. **No Headers**: Use "Document" as default header_path
3. **Table on Single Line**: Check for both `<table` and `</table>` on same line
4. **Unicode Content**: Ensure UTF-8 encoding throughout
5. **Very Long Lines**: May need truncation for display/storage
6. **Duplicate doc_ids**: Validation should prevent or handle conflicts

### Testing Recommendations

1. **Unit Test Header Stack**: Verify correct popping/pushing behavior
2. **Test Table Handling**: Multi-line tables, nested tables, tables with blank lines
3. **Test Edge Cases**: Empty content, single-line documents, documents with only headers
4. **Integration Test**: Full PDF-to-chunks pipeline with sample documents
5. **Regression Test**: Verify chunk_id stability across runs

### Performance Considerations

1. **PDF Conversion**: Rate-limited by API calls; implement retry logic
2. **Large Documents**: Stream processing if document exceeds memory
3. **Batch Processing**: Process multiple documents concurrently
4. **Caching**: Cache converted Markdown to avoid re-conversion
