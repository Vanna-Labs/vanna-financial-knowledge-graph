#!/usr/bin/env python3
"""
Test script for extraction.

Run with: uv run python test_extraction_proper_nouns.py
"""

import asyncio

from dotenv import load_dotenv

from zomma_kg.ingestion.extraction import extract_from_chunk
from zomma_kg.providers.llm.openai import OpenAILLMProvider
from zomma_kg.types import ChunkInput

load_dotenv()

# Test paragraph from Federal Reserve report
TEST_PARAGRAPH = """\
Federal Reserve Bank of San Francisco Manufacturing activity remained largely stable \
in recent weeks. While demand for automated equipment continued to grow, demand for \
some manufactured furniture weakened further. Some manufacturers noted increased levels \
of excess capacity. Most reports highlighted adequate materials availability but at \
higher costs, including for utilities and imported semiconductor chips. One manufacturer \
based in Hawaii reported higher sale prices, reduced payroll, and lower production \
volumes. A contact in the automotive sector noted that higher tariff costs resulted in \
stricter criteria for moving ahead with investment opportunities."""


async def main():
    print("=" * 70)
    print("EXTRACTION TEST")
    print("=" * 70)
    print()
    print("Test paragraph:")
    print("-" * 70)
    print(TEST_PARAGRAPH)
    print("-" * 70)
    print()

    # Create chunk and LLM provider
    chunk = ChunkInput(
        doc_id="test-doc",
        content=TEST_PARAGRAPH,
        header_path="Manufacturing",
        position=0,
    )

    llm = OpenAILLMProvider()

    print("Extracting with critique enabled...")
    print()

    result = await extract_from_chunk(
        chunk,
        llm,
        document_date="2024-01-15",
        enable_critique=True,
        max_retries=1,
    )

    # Display entities
    print("=" * 70)
    print(f"EXTRACTED ENTITIES ({len(result.entities)})")
    print("=" * 70)
    for e in result.entities:
        print(f"  - {e.name} ({e.entity_type})")
        if e.summary:
            print(f"    {e.summary}")
    print()

    # Display facts
    print("=" * 70)
    print(f"EXTRACTED FACTS ({len(result.facts)})")
    print("=" * 70)
    for f in result.facts:
        print(f"  {f.subject} --[{f.relationship}]--> {f.object}")
        print(f"    Fact: {f.fact}")
        print(f"    Date: {f.date_context}")
        if f.topics:
            print(f"    Topics: {', '.join(f.topics)}")
        print()

    # Summary
    all_topics = set()
    for f in result.facts:
        all_topics.update(f.topics)
    for e in result.entities:
        if e.entity_type == "Topic":
            all_topics.add(e.name)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Entities: {len(result.entities)}")
    print(f"  Facts: {len(result.facts)}")
    print(f"  Topics: {len(all_topics)}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
