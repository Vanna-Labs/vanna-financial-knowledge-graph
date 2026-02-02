# Entity Deduplication Module Design

**Date**: 2026-01-29
**Status**: Ready for implementation
**Scope**: In-document entity deduplication (Phase 2a-c)

---

## Overview

The `entity_dedup.py` module handles in-document entity deduplication - merging entity mentions that refer to the same real-world entity within a single document (e.g., "Apple Inc.", "AAPL", "Apple" → single canonical entity).

This is Phase 2a-c of the ZommaKG ingestion pipeline, sitting between extraction (Phase 1) and cross-document resolution (Phase 2d).

---

## Module Interface

```python
async def deduplicate_entities(
    entities: list[EnumeratedEntity],
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    *,
    similarity_threshold: float = 0.70,
    max_batch_size: int = 15,
) -> EntityDeduplicationOutput:
    """
    Deduplicate entities extracted from a single document.

    Args:
        entities: Entities from extraction (Phase 1 output)
        llm: LLM provider for verification decisions
        embeddings: Embedding provider for similarity computation
        similarity_threshold: Minimum cosine similarity to consider as candidates (default 0.70)
        max_batch_size: Max entities per LLM verification call (default 15)

    Returns:
        EntityDeduplicationOutput with canonical entities, index mapping, and merge history
    """
```

**Design principle**: The function is stateless - operates on a single document's entities without touching storage. Cross-document resolution (Phase 2d) handles KB matching separately.

---

## Algorithm Flow

```
Step 1: Embed
   entities[] → embedding texts → OpenAI API → vectors[]
   Embedding text: f"{name}: {summary}"

Step 2: Cluster
   vectors[] → cosine similarity matrix → edges where sim ≥ threshold → Union-Find → components[]

Step 3: Verify
   For each component with 2+ entities:
     - If size ≤ max_batch_size: single LLM call
     - If size > max_batch_size: overlapping batches → merge results

   LLM returns: which entities are truly the same, canonical names, merged summaries

Step 4: Build Output
   - Assign UUIDs to canonical entities
   - Build index mapping (original → canonical)
   - Record merge history
```

### Key Design Decisions

1. **Embedding text formula**: `f"{name}: {summary}"` includes summary for semantic context (distinguishes "Apple Inc" from "Apple Records")

2. **Threshold 0.70**: Deliberately low to catch candidates. False positives filtered by LLM; false negatives are permanent losses.

3. **Union-Find for transitivity**: If A~B and B~C, all three become candidates even if A and C aren't directly similar.

4. **Singletons pass through**: Entities in components of size 1 get UUIDs without LLM calls.

---

## Data Types

Add to `types/results.py`:

```python
class MergeRecord(BaseModel):
    """Record of entities merged into one canonical entity."""
    canonical_uuid: str
    canonical_name: str
    merged_indices: list[int]      # Original indices that were merged
    merged_names: list[str]        # Names of merged entities
    original_summaries: list[str]  # Summaries before merge
    final_summary: str             # Combined summary after merge


class CanonicalEntity(BaseModel):
    """A deduplicated entity with assigned UUID."""
    uuid: str
    name: str
    entity_type: EntityTypeLabel
    summary: str
    source_indices: list[int]      # Which original entities this represents
    aliases: list[str]             # Other names (from merged entities)


class EntityDeduplicationOutput(BaseModel):
    """Complete output from in-document deduplication."""
    canonical_entities: list[CanonicalEntity]
    index_to_canonical: dict[int, int]  # original index → canonical index
    merge_history: list[MergeRecord]
```

---

## LLM Verification Prompts

### System Prompt

```
You are deduplicating entities for a knowledge graph.

MERGE - same real-world entity, different names:
- Ticker and company: "AAPL" = "Apple Inc."
- Abbreviations: "Fed" = "Federal Reserve"
- Name variants: "Tim Cook" = "Timothy D. Cook"

DO NOT MERGE - related but distinct:
- Parent/subsidiary: "Alphabet" ≠ "Google" ≠ "YouTube"
- Person/company: "Tim Cook" ≠ "Apple"
- Product/company: "iPhone" ≠ "Apple"
- Competitors: "Goldman Sachs" ≠ "Morgan Stanley"

Decision test: In a knowledge graph, would these be the same node? Would facts about one apply to the other?
```

### User Prompt Template

```
ENTITIES TO ANALYZE:
{numbered list of entities with name, type, summary}

Group entities that refer to the SAME real-world entity.
For each group, provide:
- The canonical (most formal/complete) name
- Which entity indices belong to this group
- A merged summary combining information from all members

Entities that are distinct should each be their own group.
```

### Output Schema

Uses existing `EntityDedupeResult` with `EntityGroup` items containing:
- `canonical`: Best/most complete name
- `members`: Other names/aliases (not including canonical)
- `entity_type`: Type classification
- `reasoning`: Why these are the same entity

---

## Large Component Batching

When a component exceeds `max_batch_size`, use overlapping batches:

### Step 1: Similarity-Order Traversal (Greedy BFS)

Order entities so similar ones are adjacent:

```python
def _similarity_order(n: int, similarity_matrix: np.ndarray) -> list[int]:
    """Order entities so similar ones are adjacent."""
    visited = [False] * n
    order = []

    visited[0] = True
    order.append(0)

    while len(order) < n:
        best_next = -1
        best_sim = -1.0

        # Check last 10 visited for efficiency
        for visited_idx in order[-10:]:
            for j in range(n):
                if not visited[j] and similarity_matrix[visited_idx, j] > best_sim:
                    best_sim = similarity_matrix[visited_idx, j]
                    best_next = j

        visited[best_next] = True
        order.append(best_next)

    return order
```

### Step 2: Overlapping Batches

Process in windows with 5-entity overlap:

```
Component size: 25 entities (ordered by similarity)
Batch 1: indices 0-14  (15 entities)
Batch 2: indices 10-24 (15 entities, 5 overlap with batch 1)
```

### Step 3: Merge Batch Results

When an entity appears in multiple batches with different group assignments:
- Compute embedding similarity between group representatives
- If similarity > 0.80: auto-merge the groups
- Otherwise: keep entity in its first-assigned group

---

## File Structure

```
zomma_kg/
├── types/
│   └── results.py              # Add: MergeRecord, CanonicalEntity, EntityDeduplicationOutput
│
└── ingestion/
    └── resolution/
        ├── __init__.py         # Update exports
        └── entity_dedup.py     # NEW: ~400-500 lines
```

---

## Implementation Order

Within `entity_dedup.py`:

1. **Imports and constants** - thresholds, prompt templates
2. **`_generate_embeddings()`** - batch embed entities via provider
3. **`_compute_similarity_matrix()`** - cosine similarity with numpy
4. **`_similarity_order()`** - greedy BFS for batch ordering
5. **`_verify_cluster()`** - single LLM call for cluster ≤ max_batch_size
6. **`_verify_large_cluster()`** - overlapping batch logic
7. **`_merge_batch_results()`** - conflict resolution for overlaps
8. **`deduplicate_entities()`** - main orchestrator function

---

## Dependencies

- `numpy` - for similarity matrix computation
- `uuid` - for generating entity UUIDs
- Existing: `UnionFind` from `zomma_kg.utils.clustering`
- Existing: `LLMProvider`, `EmbeddingProvider` from `zomma_kg.providers.base`
- Existing: `EnumeratedEntity`, `EntityDedupeResult`, `EntityGroup` from types

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.70 | Minimum cosine similarity for clustering candidates |
| `max_batch_size` | 15 | Maximum entities per LLM verification call |
| `overlap_size` | 5 | Overlap between batches for large components |
| `merge_threshold` | 0.80 | Similarity threshold for auto-merging groups across batches |

---

## Example Usage

```python
from zomma_kg.providers import OpenAILLMProvider, OpenAIEmbeddingProvider
from zomma_kg.ingestion.extraction import extract_from_chunk
from zomma_kg.ingestion.resolution import deduplicate_entities

# Phase 1: Extract
llm = OpenAILLMProvider()
embeddings = OpenAIEmbeddingProvider()
extraction_result = await extract_from_chunk(chunk, llm)

# Phase 2a-c: Deduplicate
dedup_result = await deduplicate_entities(
    extraction_result.entities,
    llm,
    embeddings,
    similarity_threshold=0.70,
    max_batch_size=15,
)

print(f"Reduced {len(extraction_result.entities)} entities to {len(dedup_result.canonical_entities)} canonical")
```
