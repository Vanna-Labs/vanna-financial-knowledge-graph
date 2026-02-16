# Entity Registry & Assembler Design

**Date**: 2026-01-29
**Status**: Approved
**Modules**: `ingestion/resolution/entity_registry.py`, `ingestion/assembly/assembler.py`

## Overview

Two modules that complete the ingestion pipeline by resolving entities against the existing KB and writing data to storage.

```
Extraction Output → Entity Registry → Assembler → Storage (Parquet + LanceDB)
```

### Module Split

| Module | Responsibility |
|--------|----------------|
| `entity_registry.py` | Cross-document entity matching against KB |
| `assembler.py` | Batch writes resolved data to storage |

---

## Module 1: Entity Registry

**File**: `vanna_kg/ingestion/resolution/entity_registry.py`

### Purpose

Match new entities against existing KB entities using vector similarity + LLM verification. This is the cross-document deduplication layer (in-document dedup happens earlier in `entity_dedup.py`).

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EntityRegistry                          │
├─────────────────────────────────────────────────────────────┤
│  Input: List[Entity] (from extraction + in-doc dedup)       │
│                                                             │
│  Step 1: Embed each entity ("{name}: {summary}")            │
│  Step 2: Search LanceDB for top-K candidates per entity     │
│  Step 3: LLM verification for each candidate set            │
│  Step 4: Build UUID remap table (duplicate → canonical)     │
│                                                             │
│  Output: EntityResolutionResult                             │
│    - new_entities: List[Entity]  (truly new, to be written) │
│    - uuid_remap: dict[str, str]  (merged UUID → canonical)  │
│    - summary_updates: dict[str, str] (UUID → merged summary)│
└─────────────────────────────────────────────────────────────┘
```

### Constants

```python
CANDIDATE_LIMIT = 25              # Top candidates from vector search
SIMILARITY_DISPLAY_THRESHOLD = 0.50  # Show in LLM prompt if above this
HIGH_SIMILARITY_THRESHOLD = 0.90     # Flag as "likely same" in prompt
```

### Algorithm

```python
async def resolve(entities: list[Entity]) -> EntityResolutionResult:
    new_entities = []
    uuid_remap = {}
    summary_updates = {}

    # Batch embed all entities
    texts = [f"{e.name}: {e.summary}" for e in entities]
    embeddings = await embedding_provider.embed(texts)

    for entity, embedding in zip(entities, embeddings):
        # Search KB for candidates
        candidates = await storage.search_entities(embedding, limit=CANDIDATE_LIMIT)

        # Filter to meaningful similarity
        candidates = [(e, score) for e, score in candidates
                      if score >= SIMILARITY_DISPLAY_THRESHOLD]

        if not candidates:
            # No matches - truly new entity
            new_entities.append(entity)
            continue

        # LLM verification
        decision = await llm_verify_match(entity, candidates)

        if decision.matches_existing:
            # Map this entity's UUID to the canonical one
            uuid_remap[entity.uuid] = decision.matched_uuid
            # Queue summary update for existing entity
            summary_updates[decision.matched_uuid] = decision.merge_summary
        else:
            new_entities.append(entity)

    return EntityResolutionResult(
        new_entities=new_entities,
        uuid_remap=uuid_remap,
        summary_updates=summary_updates,
    )
```

### LLM Verification

**Model**: `gpt-5-mini`

**Prompt Structure**:

```
You are resolving whether a new entity matches any existing KB entities.

NEW ENTITY:
  Name: {name}
  Summary: {summary}

CANDIDATES (sorted by similarity):
  1. "Apple Inc." (92% similar) - Technology company...
  2. "Apple Records" (67% similar) - Record label founded by Beatles...

Rules:
- MATCH if same real-world entity (different names OK: "Apple" = "AAPL")
- DISTINCT if related but separate (subsidiaries: AWS ≠ Amazon)
- When uncertain, prefer DISTINCT (avoid incorrect merges)

Respond with your decision.
```

**Output Schema**:

```python
class EntityMatchDecision(BaseModel):
    matches_existing: bool
    matched_uuid: str | None      # UUID of matched KB entity
    canonical_name: str           # Best name to use
    merge_summary: str            # Combined summary if matched
    confidence: float             # 0.0 to 1.0
    reasoning: str                # Why this decision
```

### Summary Merging

When an entity matches, summaries are merged via LLM to:
- Combine information intelligently
- Deduplicate redundant facts
- Preserve source annotations (`[Source: chunk-uuid]`)

**Merge prompt** (separate LLM call):

```
Merge these two entity summaries into one coherent summary.
Preserve all [Source: ...] annotations. Remove redundant information.

EXISTING: {existing_summary}
NEW: {new_summary}

Output the merged summary.
```

### Public API

```python
class EntityRegistry:
    def __init__(
        self,
        storage: ParquetBackend,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
    ):
        self.storage = storage
        self.llm = llm_provider
        self.embeddings = embedding_provider

    async def resolve(
        self,
        entities: list[Entity],
        embeddings: list[list[float]] | None = None,
    ) -> EntityResolutionResult:
        """
        Resolve entities against existing KB.

        Args:
            entities: Entities from extraction (after in-doc dedup)
            embeddings: Pre-computed embeddings, or None to generate

        Returns:
            EntityResolutionResult with new_entities, uuid_remap, summary_updates
        """
```

---

## Module 2: Assembler

**File**: `vanna_kg/ingestion/assembly/assembler.py`

### Purpose

Take fully-resolved extraction output and write to storage in efficient batches.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Assembler                             │
├─────────────────────────────────────────────────────────────┤
│  Input: AssemblyInput                                       │
│    - document: Document                                     │
│    - chunks: list[Chunk]                                    │
│    - entities: list[Entity]  (already resolved via registry)│
│    - facts: list[Fact]       (UUIDs already remapped)       │
│    - topics: list[Topic]                                    │
│                                                             │
│  Step 1: Generate embeddings (entities, facts, topics)      │
│  Step 2: Write in order:                                    │
│          document → chunks → entities → facts → topics      │
│  Step 3: Build & write relationships                        │
│                                                             │
│  Output: AssemblyResult                                     │
│    - entities_written: int                                  │
│    - facts_written: int                                     │
│    - chunks_written: int                                    │
└─────────────────────────────────────────────────────────────┘
```

### Batching Strategy

**Embedding Generation** (parallel batch calls):

```python
# All embeddings generated in parallel batches - NOT one call per item
entity_texts = [f"{e.name}: {e.summary}" for e in entities]
fact_texts = [f.content for f in facts]
topic_texts = [f"{t.name}: {t.definition}" for t in topics]

entity_embeddings, fact_embeddings, topic_embeddings = await asyncio.gather(
    embedding_provider.embed(entity_texts),
    embedding_provider.embed(fact_texts),
    embedding_provider.embed(topic_texts),
)
```

**Storage Writes** (one call per table):

```python
# Batch writes - NOT one write per item
await storage.write_document(document)
await storage.write_chunks(chunks)
await storage.write_entities(entities, entity_embeddings)
await storage.write_facts(facts, fact_embeddings)
await storage.write_topics(topics, topic_embeddings)
await storage.write_relationships(relationships)
```

### Write Order

Order matters due to foreign key relationships:

1. **Document** - No dependencies
2. **Chunks** - Reference document UUID
3. **Entities** - No dependencies (UUIDs already resolved)
4. **Facts** - Reference entity UUIDs and chunk UUID
5. **Topics** - No dependencies
6. **Relationships** - Reference all of the above

### Relationship Building

Relationships follow the chunk-centric fact pattern from the original:

```python
relationships = []
for fact in facts:
    # Subject Entity -> Chunk (where fact was extracted)
    relationships.append({
        "from_uuid": fact.subject_uuid,
        "from_type": "entity",
        "to_uuid": fact.chunk_uuid,
        "to_type": "chunk",
        "rel_type": "MENTIONED_IN",
        "fact_id": fact.uuid,
    })

    # Chunk -> Object Entity
    relationships.append({
        "from_uuid": fact.chunk_uuid,
        "from_type": "chunk",
        "to_uuid": fact.object_uuid,
        "to_type": "entity",
        "rel_type": fact.relationship_type,
        "fact_id": fact.uuid,
    })

# Single batch write for all relationships
await storage.write_relationships(relationships)
```

### Public API

```python
@dataclass
class AssemblyInput:
    document: Document
    chunks: list[Chunk]
    entities: list[Entity]
    facts: list[Fact]
    topics: list[Topic]


@dataclass
class AssemblyResult:
    document_written: bool
    chunks_written: int
    entities_written: int
    facts_written: int
    topics_written: int
    relationships_written: int


class Assembler:
    def __init__(
        self,
        storage: ParquetBackend,
        embedding_provider: EmbeddingProvider,
    ):
        self.storage = storage
        self.embeddings = embedding_provider

    async def assemble(self, input: AssemblyInput) -> AssemblyResult:
        """
        Write resolved extraction output to storage in batches.

        Args:
            input: AssemblyInput with document, chunks, entities, facts, topics
                   (entities should already be resolved via EntityRegistry,
                    fact UUIDs should already be remapped)

        Returns:
            AssemblyResult with counts of items written
        """
```

---

## Integration: Full Pipeline Flow

```python
async def ingest_document(doc_path: str, storage: ParquetBackend):
    # Phase 1: Chunking
    chunks = await chunker.chunk(doc_path)

    # Phase 2: Extraction
    extraction = await extractor.extract(chunks)

    # Phase 3a: In-document deduplication
    dedup_result = await entity_dedup.deduplicate(extraction.entities)

    # Phase 3b: Cross-document resolution (Entity Registry)
    registry = EntityRegistry(storage, llm, embeddings)
    resolution = await registry.resolve(dedup_result.canonical_entities)

    # Apply UUID remapping to facts
    for fact in extraction.facts:
        fact.subject_uuid = resolution.uuid_remap.get(fact.subject_uuid, fact.subject_uuid)
        fact.object_uuid = resolution.uuid_remap.get(fact.object_uuid, fact.object_uuid)

    # Update existing entity summaries
    for uuid, summary in resolution.summary_updates.items():
        await storage.update_entity_summary(uuid, summary)

    # Phase 4: Assembly (write to storage)
    assembler = Assembler(storage, embeddings)
    result = await assembler.assemble(AssemblyInput(
        document=extraction.document,
        chunks=chunks,
        entities=resolution.new_entities,  # Only truly new entities
        facts=extraction.facts,
        topics=extraction.topics,
    ))

    return result
```

---

## Types Summary

### New Types to Add

```python
# In types/results.py

@dataclass
class EntityResolutionResult:
    """Result from EntityRegistry.resolve()"""
    new_entities: list[Entity]        # Truly new, not in KB
    uuid_remap: dict[str, str]        # old_uuid -> canonical_uuid
    summary_updates: dict[str, str]   # uuid -> merged_summary


class EntityMatchDecision(BaseModel):
    """LLM output for entity matching"""
    matches_existing: bool
    matched_uuid: str | None
    canonical_name: str
    merge_summary: str
    confidence: float                 # 0.0 to 1.0
    reasoning: str
```

### Existing Types Used

- `Entity`, `Fact`, `Chunk`, `Document`, `Topic` from `types/`
- `ParquetBackend` from `storage/parquet/backend.py`
- `LLMProvider`, `EmbeddingProvider` from `providers/`

---

## File Checklist

- [ ] `vanna_kg/ingestion/resolution/entity_registry.py`
- [ ] `vanna_kg/ingestion/assembly/assembler.py`
- [ ] Update `vanna_kg/types/results.py` with new types
- [ ] Tests: `tests/test_entity_registry.py`
- [ ] Tests: `tests/test_assembler.py`
