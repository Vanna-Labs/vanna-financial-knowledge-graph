# Build KG Script Design

## Overview

A standalone script to run the full VannaKG ingestion pipeline end-to-end, primarily for testing and validation.

## CLI Interface

```bash
# Basic usage (3 chunks, markdown input)
python scripts/build_kg.py test_data/BeigeBook_20251015.md

# More chunks
python scripts/build_kg.py test_data/BeigeBook_20251015.md --chunks 10

# Custom output directory
python scripts/build_kg.py test_data/BeigeBook_20251015.md --output ./test_kb

# Use higher quality model
python scripts/build_kg.py test_data/BeigeBook_20251015.md --model gpt-5.1

# Skip topic resolution
python scripts/build_kg.py test_data/BeigeBook_20251015.md --skip-topics
```

## Pipeline Phases

### Phase 1: Setup
- Load environment variables (`.env`)
- Initialize `KGConfig`
- Initialize `ParquetBackend(output_path)`
- Initialize `OpenAILLMProvider` (default: `gpt-5-mini`)
- Initialize `OpenAIEmbeddingProvider` (`text-embedding-3-large`, 3072 dimensions)

### Phase 2: Chunking
- Read markdown file
- `chunk_markdown(content, doc_id)` → `ChunkInput[]`
- Slice to first N chunks (default: 3)

### Phase 3: Extraction
- For each chunk: `extract_from_chunk(chunk, llm)` → `ChainOfThoughtResult`
- Flatten to `all_entities[]`, `all_facts[]`

### Phase 4: In-Document Deduplication
- `deduplicate_entities(all_entities, llm, embeddings)` → `EntityDedupeResult`
- Output: `canonical_entities[]`, `merge_records[]`

### Phase 5: Cross-Document Resolution
- `EntityRegistry(storage, llm, embeddings).resolve(canonical_entities)`
- Output: `new_entities[]`, `uuid_remap{}`, `summary_updates{}`

### Phase 6: Topic Resolution (skippable)
- Collect topics from all facts
- `TopicResolver(indices, llm, embeddings).resolve(topics)`
- Output: `resolved_topics[]`

### Phase 7: Assembly
- Build `AssemblyInput` (document, chunks, entities, facts, topics)
- `Assembler(storage, embeddings).assemble(input)`
- Write to Parquet + LanceDB

## Output Directory Structure

```
test_kb/
├── metadata.json
├── documents.parquet
├── chunks.parquet
├── entities.parquet
├── facts.parquet
├── topics.parquet
├── relationships.parquet
└── lancedb/
    ├── entities.lance/
    ├── facts.lance/
    └── topics.lance/
```

## Error Handling

- Wrap each phase in try/except
- If extraction fails on a chunk, skip and continue
- Use built-in retry logic for rate limits
- On fatal error, print what succeeded before failing

## Progress Output

```
[1/7] Setup... done (gpt-5-mini, text-embedding-3-large)
[2/7] Chunking... 5 chunks from BeigeBook_20251015.md
[3/7] Extraction... 23 entities, 31 facts (5/5 chunks)
[4/7] Deduplication... 18 canonical entities (5 merged)
[5/7] Cross-doc resolution... 18 new, 0 matched
[6/7] Topic resolution... 12 topics resolved
[7/7] Assembly... written to ./test_kb/
      - 1 document, 5 chunks, 18 entities, 31 facts, 47 relationships

Done in 34.2s
```

## Configuration

| Setting | Default | Flag |
|---------|---------|------|
| Chunks | 3 | `--chunks N` |
| Output | `./test_kb/` | `--output PATH` |
| Model | `gpt-5-mini` | `--model NAME` |
| Skip topics | False | `--skip-topics` |

## Test Data

Using October 2025 Beige Book:
- Location: `test_data/BeigeBook_20251015.md`
- 555 lines, well-structured with Federal Reserve district headers
- Financial content matches `financial_topics.json` ontology
