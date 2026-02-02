"""
Chain-of-Thought Extractor

Extracts entities and facts from text chunks using a two-step approach:
1. Entity Enumeration: List ALL entities in the text
2. Fact Generation: Generate relationships between enumerated entities

Includes optional critique/reflexion step for quality assurance.

Example:
    >>> from zomma_kg.providers import OpenAILLMProvider
    >>> llm = OpenAILLMProvider()
    >>> result = await extract_from_chunk(chunk, llm)
    >>> print(f"Found {len(result.entities)} entities, {len(result.facts)} facts")
"""

import asyncio
from typing import TYPE_CHECKING

from zomma_kg.types import ChunkInput
from zomma_kg.types.results import ChainOfThoughtResult, CritiqueResult

if TYPE_CHECKING:
    from zomma_kg.providers.base import LLMProvider


# -----------------------------------------------------------------------------
# System Prompts
# -----------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a financial analyst building a knowledge graph from documents.

## Your Task
Extract entities and relationships from the text using a two-step process:

**Step 1 - Entity Enumeration**: List ALL named entities mentioned in the text.
**Step 2 - Fact Generation**: Generate relationships between the enumerated entities.

## Entity Types
- **Company**: Corporate entities. Subsidiaries are SEPARATE (AWS ≠ Amazon)
- **Person**: Named individuals
- **Organization**: Non-profit, governmental bodies (Federal Reserve, SEC, UN)
- **Location**: Geographic entities (countries, regions, cities)
- **Product**: Specific products/services (iPhone, ChatGPT, Azure)
- **Topic**: Abstract concepts, metrics, themes (M&A, Earnings, Inflation, GDP)

## Proper Noun Requirement (CRITICAL)
Entities must be **names**, not **descriptions of unnamed things**.

**The Test:** Is this a NAME or a DESCRIPTION?
- "Federal Reserve Bank of San Francisco" → NAME (find it in a database) ✓
- "Hawaii" → NAME (find it on a map) ✓
- "A manufacturer based in Hawaii" → DESCRIPTION (could be anyone) ✗
- "Some manufacturers" → DESCRIPTION (unnamed group) ✗

**When text references unnamed sources:**
Extract only the named elements within:
- "A manufacturer based in Hawaii" → Extract "Hawaii" (Location)
- "Contacts in the automotive sector" → Extract "Automotive" (Topic)
- "Sources familiar with the deal" → Extract nothing

The fact text preserves full context; entity fields capture only actual names.

## Rules for Entities
- Use clean names only - no parenthetical descriptions
- Keep subsidiaries separate from parent companies
- Include entities from section headers (the ">" denotes hierarchy)
- **Definition vs Summary** (IMPORTANT):
  - **definition**: What the entity IS - stable across all documents
    - Good: "Central banking system of the United States"
    - Bad: "Reported inflation concerns" (that's context-specific)
  - **summary**: What was found about this entity IN THIS CHUNK
    - Good: "Reported concern about inflation in Q3 2024"
    - Bad: "A central bank" (that's the definition)
  - Example for "Federal Reserve" in a Beige Book:
    - definition: "Central banking system of the United States"
    - summary: "Expressed cautious optimism about economic outlook"

## Rules for Facts
- Every fact MUST have a date_context (use document date as fallback)
- Facts must be self-contained propositions (no pronouns like "it", "they")
- Subject and object MUST be from your enumerated entities
- Relationship is free-form (e.g., "acquired", "partnered with")
- Include relevant topics for each fact

## What NOT to Extract as Entities
- URLs, citations, page numbers
- Numeric values alone (extract the metric category instead)
- Descriptions of unnamed things (extract any named locations/topics within)"""

_EXTRACTION_USER_TEMPLATE = """\
DOCUMENT CONTEXT:
Document Date: {document_date}

CHUNK TEXT:
Section: {header_path}

{content}

First enumerate ALL entities (including any from the Section header), then generate
relationships between them. ALL facts MUST have date_context - use "{document_date}"
as fallback if no specific date in text."""


_CRITIQUE_SYSTEM_PROMPT = """\
You are a quality reviewer for knowledge graph extractions.

Review the extraction results against the original text and check for:

1. **Entity Coverage**: Are ALL named entities captured? Check for missed entities.
2. **Sentence Completeness**: Does every substantive clause have a corresponding fact?
3. **Relationship Completeness**: Are all material interactions captured?
4. **Accuracy**: Do entity names match the text exactly? No hallucinated entities?
5. **Self-Containment**: Are facts understandable standalone? No dangling pronouns?
6. **Clean Structure**: Are entity names clean (no parenthetical descriptions)?
7. **Date Context**: Does every fact have temporal context?
8. **Proper Noun Check**: For each entity, ask: Is this a NAME or a DESCRIPTION?
   - Could you find this exact entity in a real-world database? If no → REJECT
   - Is this describing an unnamed thing ("a manufacturer", "some contacts")? → REJECT
   - For rejected entities: extract any named locations/sectors separately.

Be strict but fair. Minor omissions of tangential information are acceptable.
Missing key entities or relationships is NOT acceptable.
Extracting generic descriptions as entities is NOT acceptable."""

_CRITIQUE_USER_TEMPLATE = """ORIGINAL TEXT:
Section: {header_path}

{content}

EXTRACTION RESULTS:
Entities: {entities}

Facts: {facts}

Review the extraction. Is it complete and accurate?"""


_REEXTRACT_USER_TEMPLATE = """\
DOCUMENT CONTEXT:
Document Date: {document_date}

CHUNK TEXT:
Section: {header_path}

{content}

PREVIOUS EXTRACTION HAD ISSUES:
{critique}

MISSED FACTS TO ADD:
{missed_facts}

CORRECTIONS NEEDED:
{corrections}

Re-extract, addressing the issues above. Enumerate ALL entities first, then generate
relationships between them."""


# -----------------------------------------------------------------------------
# Core Extraction Functions
# -----------------------------------------------------------------------------


async def extract_from_chunk(
    chunk: ChunkInput,
    llm: "LLMProvider",
    *,
    document_date: str | None = None,
    enable_critique: bool = True,
    max_retries: int = 1,
) -> ChainOfThoughtResult:
    """
    Extract entities and facts from a single chunk.

    Uses chain-of-thought: enumerate entities first, then generate relationships.
    Optionally runs critique step and re-extracts if quality issues found.

    Args:
        chunk: The text chunk to extract from
        llm: LLM provider for generation
        document_date: Document date for temporal context (ISO format)
        enable_critique: Whether to run critique step
        max_retries: Max re-extraction attempts after failed critique (0 = no retries)

    Returns:
        ChainOfThoughtResult with entities and facts
    """
    doc_date = document_date or "Unknown"

    # Initial extraction
    prompt = _EXTRACTION_USER_TEMPLATE.format(
        document_date=doc_date,
        header_path=chunk.header_path or "(No section)",
        content=chunk.content,
    )

    result = await llm.generate_structured(
        prompt,
        ChainOfThoughtResult,
        system=_EXTRACTION_SYSTEM_PROMPT,
    )

    # Skip critique if disabled or no results
    if not enable_critique or (not result.entities and not result.facts):
        return result

    # Critique step
    critique = await _critique_extraction(chunk, result, llm)

    if critique.is_approved:
        return result

    # Re-extract if critique failed and retries allowed
    retries = 0
    while not critique.is_approved and retries < max_retries:
        result = await _reextract_with_feedback(chunk, critique, llm, doc_date)
        critique = await _critique_extraction(chunk, result, llm)
        retries += 1

    return result


async def _critique_extraction(
    chunk: ChunkInput,
    result: ChainOfThoughtResult,
    llm: "LLMProvider",
) -> CritiqueResult:
    """Run critique step on extraction results."""
    # Format entities and facts for review
    entities_str = "\n".join(
        f"- {e.name} ({e.entity_type}): {e.summary}" for e in result.entities
    ) or "(none)"

    facts_str = "\n".join(
        f"- {f.subject} --[{f.relationship}]--> {f.object}: {f.fact}"
        for f in result.facts
    ) or "(none)"

    prompt = _CRITIQUE_USER_TEMPLATE.format(
        header_path=chunk.header_path or "(No section)",
        content=chunk.content,
        entities=entities_str,
        facts=facts_str,
    )

    return await llm.generate_structured(
        prompt,
        CritiqueResult,
        system=_CRITIQUE_SYSTEM_PROMPT,
    )


async def _reextract_with_feedback(
    chunk: ChunkInput,
    critique: CritiqueResult,
    llm: "LLMProvider",
    document_date: str,
) -> ChainOfThoughtResult:
    """Re-extract with critique feedback."""
    prompt = _REEXTRACT_USER_TEMPLATE.format(
        document_date=document_date,
        header_path=chunk.header_path or "(No section)",
        content=chunk.content,
        critique=critique.critique or "Quality issues detected",
        missed_facts="\n".join(f"- {f}" for f in critique.missed_facts) or "(none)",
        corrections="\n".join(f"- {c}" for c in critique.corrections) or "(none)",
    )

    return await llm.generate_structured(
        prompt,
        ChainOfThoughtResult,
        system=_EXTRACTION_SYSTEM_PROMPT,
    )


# -----------------------------------------------------------------------------
# Batch Extraction with Concurrency
# -----------------------------------------------------------------------------


async def extract_from_chunks(
    chunks: list[ChunkInput],
    llm: "LLMProvider",
    *,
    document_date: str | None = None,
    concurrency: int = 50,
    enable_critique: bool = True,
    max_retries: int = 1,
) -> list[ChainOfThoughtResult]:
    """
    Extract entities and facts from multiple chunks in parallel.

    Args:
        chunks: List of text chunks to extract from
        llm: LLM provider for generation
        document_date: Document date for temporal context
        concurrency: Max concurrent extractions (-1 for unlimited)
        enable_critique: Whether to run critique step
        max_retries: Max re-extraction attempts per chunk

    Returns:
        List of ChainOfThoughtResult, one per chunk (same order as input)
    """
    if not chunks:
        return []

    async def extract_one(chunk: ChunkInput) -> ChainOfThoughtResult:
        try:
            return await extract_from_chunk(
                chunk,
                llm,
                document_date=document_date,
                enable_critique=enable_critique,
                max_retries=max_retries,
            )
        except Exception:
            # Return empty result on error, don't fail entire batch
            return ChainOfThoughtResult()

    # Unlimited concurrency
    if concurrency == -1:
        tasks = [extract_one(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    # Semaphore-limited concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def extract_with_semaphore(chunk: ChunkInput) -> ChainOfThoughtResult:
        async with semaphore:
            return await extract_one(chunk)

    tasks = [extract_with_semaphore(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)
