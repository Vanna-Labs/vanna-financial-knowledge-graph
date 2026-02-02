#!/usr/bin/env python3
"""Debug script to investigate topic resolution issues."""

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Only show debug for our modules
logging.getLogger("zomma_kg").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import lancedb

from zomma_kg.config import KGConfig
from zomma_kg.storage import ParquetBackend
from zomma_kg.providers.embedding.openai import OpenAIEmbeddingProvider
from zomma_kg.providers.llm.openai import OpenAILLMProvider
from zomma_kg.query import GraphRAGPipeline


async def main():
    kb_path = Path("./test_kb")
    config = KGConfig()

    # 1. Check LanceDB directly
    print("=" * 60)
    print("1. LANCEDB DIRECT INSPECTION")
    print("=" * 60)

    lancedb_path = kb_path / "lancedb"
    db = lancedb.connect(str(lancedb_path))
    print(f"Tables: {db.table_names()}")

    if "topics" in db.table_names():
        topics_table = db.open_table("topics")
        arrow_table = topics_table.to_arrow()
        print(f"\nTopics table has {arrow_table.num_rows} rows")
        print(f"Columns: {arrow_table.column_names}")

        # Get unique group_ids and count
        group_id_counts: dict[str, int] = {}
        for i in range(arrow_table.num_rows):
            gid = arrow_table.column("group_id")[i].as_py()
            group_id_counts[gid] = group_id_counts.get(gid, 0) + 1
        print(f"\nGroup_id counts: {group_id_counts}")

        print(f"\nSample topics by group:")
        for gid in group_id_counts:
            print(f"\n  {gid}:")
            count = 0
            for i in range(arrow_table.num_rows):
                if arrow_table.column("group_id")[i].as_py() == gid:
                    name = arrow_table.column("name")[i].as_py()
                    print(f"    - {name}")
                    count += 1
                    if count >= 3:
                        break
    else:
        print("No topics table found!")

    # 2. Test via storage backend
    print("\n" + "=" * 60)
    print("2. TESTING TWO-STAGE TOPIC RESOLUTION")
    print("=" * 60)

    embeddings = OpenAIEmbeddingProvider()
    query_vec = await embeddings.embed_single("tariffs trade policy economic impact")

    storage = ParquetBackend(kb_path, config)  # Uses default group_id
    await storage.initialize()

    print(f"Storage group_id: {storage.group_id}")

    # Test get_topics_by_names
    print("\nTesting get_topics_by_names(['Tariffs', 'Auto Sales']):")
    kb_topics = await storage.get_topics_by_names(["Tariffs", "Auto Sales", "Trade Policy"])
    print(f"  Found {len(kb_topics)} topics:")
    for t in kb_topics:
        print(f"    - {t.name} (uuid={t.uuid[:8]}...)")

    await storage.close()

    # 3. Test researcher's topic resolution directly
    print("\n" + "=" * 60)
    print("3. TESTING RESEARCHER TOPIC RESOLUTION DIRECTLY")
    print("=" * 60)

    from zomma_kg.query.researcher import Researcher

    storage = ParquetBackend(kb_path, config)
    await storage.initialize()

    llm = OpenAILLMProvider(model=config.llm_model)

    researcher = Researcher(storage, llm, embeddings, config)

    print("\nTesting resolve_topics(['tariffs', 'trade policy']):")
    resolved = await researcher.resolve_topics(
        ["tariffs", "trade policy", "economic uncertainty"],
        query_vec
    )
    print(f"Resolved {len(resolved)} topics:")
    for t in resolved:
        print(f"  - {t.resolved_name} (uuid={t.resolved_uuid[:8]}..., conf={t.confidence:.3f})")

    await storage.close()

    # 4. Test full pipeline
    print("\n" + "=" * 60)
    print("4. TESTING FULL QUERY PIPELINE")
    print("=" * 60)

    storage = ParquetBackend(kb_path, config)
    await storage.initialize()

    pipeline = GraphRAGPipeline(storage, llm, embeddings, config)

    print("\nQuery: What do businesses say about tariffs?")
    result = await pipeline.query("What do businesses say about tariffs?")
    print(f"\nAnswer:\n{result.answer[:800]}...")
    print(f"\nConfidence: {result.confidence}")
    print(f"Sub-answers: {len(result.sub_answers)}")

    if result.sources:
        print(f"\nSources ({len(result.sources)}):")
        for s in result.sources[:3]:
            print(f"  - {s}")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
