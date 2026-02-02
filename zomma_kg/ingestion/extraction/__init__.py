"""
LLM-Based Extraction

Chain-of-thought extraction of entities and facts from chunks.

Modules:
    extractor: Main extraction logic (two-step process)
    critique: Extraction quality assessment
    schemas: Pydantic schemas for extraction results

Two-Step Extraction:
    1. Entity Enumeration: Extract ALL entities first
    2. Relationship Generation: Create facts using ONLY enumerated entities

Critique/Reflexion:
    - LLM verifies extraction completeness
    - Checks for missed facts at sentence level
    - One re-extraction retry if issues found

Entity Types:
    - Company (subsidiaries are SEPARATE entities)
    - Person
    - Organization
    - Location
    - Product
    - Topic (metrics, themes, concepts)

See: docs/pipeline/ENTITY_TOPIC_EXTRACTION.md
"""

from zomma_kg.ingestion.extraction.extractor import (
    extract_from_chunk,
    extract_from_chunks,
)

__all__ = ["extract_from_chunk", "extract_from_chunks"]
