#!/usr/bin/env python3
"""
Manual test for entity deduplication.

Run with: uv run python test_dedup_manual.py
"""

import asyncio

from dotenv import load_dotenv

from vanna_kg.ingestion.resolution import deduplicate_entities
from vanna_kg.providers.llm.openai import OpenAILLMProvider
from vanna_kg.providers.embedding.openai import OpenAIEmbeddingProvider
from vanna_kg.types.entities import EnumeratedEntity

load_dotenv()


async def main():
    print("=" * 70)
    print("ENTITY DEDUPLICATION TEST")
    print("=" * 70)
    print()

    # Test entities with duplicates
    entities = [
        EnumeratedEntity(
            name="Apple Inc.",
            entity_type="Company",
            summary="Technology company known for iPhone and Mac computers",
        ),
        EnumeratedEntity(
            name="AAPL",
            entity_type="Company",
            summary="Stock ticker symbol for Apple",
        ),
        EnumeratedEntity(
            name="Apple",
            entity_type="Company",
            summary="Consumer electronics manufacturer",
        ),
        EnumeratedEntity(
            name="Tim Cook",
            entity_type="Person",
            summary="CEO of Apple since 2011",
        ),
        EnumeratedEntity(
            name="Timothy D. Cook",
            entity_type="Person",
            summary="Chief Executive Officer of Apple Inc.",
        ),
        EnumeratedEntity(
            name="Google",
            entity_type="Company",
            summary="Search engine and technology company",
        ),
        EnumeratedEntity(
            name="Alphabet",
            entity_type="Company",
            summary="Parent company of Google",
        ),
    ]

    print("Input entities:")
    for i, e in enumerate(entities):
        print(f"  {i}: {e.name} ({e.entity_type})")
    print()

    # Create providers
    llm = OpenAILLMProvider()
    embeddings = OpenAIEmbeddingProvider()

    print("Running deduplication...")
    print()

    result = await deduplicate_entities(entities, llm, embeddings)

    print("=" * 70)
    print(f"CANONICAL ENTITIES ({len(result.canonical_entities)})")
    print("=" * 70)
    for e in result.canonical_entities:
        print(f"  {e.name} ({e.entity_type})")
        print(f"    UUID: {e.uuid}")
        print(f"    Source indices: {e.source_indices}")
        if e.aliases:
            print(f"    Aliases: {e.aliases}")
        print()

    print("=" * 70)
    print("INDEX MAPPING")
    print("=" * 70)
    for orig_idx, canon_idx in sorted(result.index_to_canonical.items()):
        orig_name = entities[orig_idx].name
        canon_name = result.canonical_entities[canon_idx].name
        print(f"  {orig_idx} ({orig_name}) -> {canon_idx} ({canon_name})")
    print()

    print("=" * 70)
    print(f"MERGE HISTORY ({len(result.merge_history)} merges)")
    print("=" * 70)
    for record in result.merge_history:
        print(f"  {record.canonical_name}")
        print(f"    Merged: {record.merged_names}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Input entities: {len(entities)}")
    print(f"  Canonical entities: {len(result.canonical_entities)}")
    print(f"  Merges performed: {len(result.merge_history)}")
    print()

    # Verify subsidiary awareness
    print("=" * 70)
    print("SUBSIDIARY AWARENESS CHECK")
    print("=" * 70)
    google_idx = result.index_to_canonical[5]  # Google
    alphabet_idx = result.index_to_canonical[6]  # Alphabet
    if google_idx != alphabet_idx:
        print("  ✅ PASS: Google and Alphabet are SEPARATE entities")
    else:
        print("  ❌ FAIL: Google and Alphabet were incorrectly merged!")


if __name__ == "__main__":
    asyncio.run(main())
