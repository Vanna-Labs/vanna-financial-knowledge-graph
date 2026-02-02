#!/usr/bin/env python3
"""
Build Knowledge Graph Script

Thin wrapper around KnowledgeGraph ingestion APIs.

Usage:
    python scripts/build_kg.py test_data/BeigeBook_20251015.md
    python scripts/build_kg.py test_data/BeigeBook_20251015.md --chunks 10
    python scripts/build_kg.py test_data/BeigeBook_20251015.md --output ./my_kb
    python scripts/build_kg.py test_data/BeigeBook_20251015.md --model gpt-5.1
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

from zomma_kg.api.knowledge_graph import KnowledgeGraph
from zomma_kg.config import KGConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from a document"
    )
    parser.add_argument("input", type=Path, help="Path to markdown or PDF file")
    parser.add_argument(
        "--chunks",
        type=int,
        default=None,
        help="For markdown only: ingest first N chunks via KnowledgeGraph.ingest_markdown()",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./test_kb"),
        help="Output directory for knowledge base (default: ./test_kb)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model override (default: from KGConfig)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Document date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean output directory before building (default: true)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    config = KGConfig(llm_model=args.model) if args.model else KGConfig()

    start = time.time()
    async with KnowledgeGraph(args.output, config=config) as kg:
        def on_progress(stage: str, progress: float) -> None:
            print(f"  [{stage}] {progress * 100:.0f}%")

        suffix = args.input.suffix.lower()
        if suffix in (".md", ".markdown"):
            if args.chunks is not None and args.chunks <= 0:
                raise ValueError("--chunks must be a positive integer")
            if args.chunks is None:
                print("Ingesting markdown via KnowledgeGraph.ingest_markdown()...")
            else:
                print(f"Ingesting first {args.chunks} markdown chunks via KnowledgeGraph...")
            result = await kg.ingest_markdown(
                args.input,
                document_date=args.date,
                max_chunks=args.chunks,
                metadata={
                    "source_path": str(args.input.resolve()),
                    "file_type": "markdown",
                },
                on_progress=on_progress,
            )
        elif suffix == ".pdf":
            if args.chunks is not None:
                print(
                    "Warning: --chunks is only supported for markdown and "
                    "will be ignored for PDF."
                )
            print("Ingesting PDF via KnowledgeGraph.ingest_pdf()...")
            result = await kg.ingest_pdf(
                args.input,
                document_date=args.date,
                metadata={
                    "source_path": str(args.input.resolve()),
                    "file_type": "pdf",
                },
                on_progress=on_progress,
            )
        else:
            raise ValueError(f"Unsupported input type: {suffix} (expected .md/.markdown/.pdf)")

    total = time.time() - start
    print("\nIngestion complete")
    print(f"  Document ID: {result.document_id}")
    print(f"  Chunks: {result.chunks}")
    print(f"  Entities: {result.entities}")
    print(f"  Facts: {result.facts}")
    print(f"  Topics: {result.topics}")
    print(f"  Ingest duration: {result.duration_seconds:.2f}s")
    print(f"  Total script duration: {total:.2f}s")
    if result.errors:
        print("  Warnings:")
        for err in result.errors:
            print(f"    - {err}")


if __name__ == "__main__":
    asyncio.run(main())
