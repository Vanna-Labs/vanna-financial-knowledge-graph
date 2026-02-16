# Topic Resolver Design

## Overview

The topic resolver matches extracted topic strings against a curated financial ontology using vector similarity search and batched LLM verification. This completes Phase 3 of the ingestion pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TopicResolver                            │
├─────────────────────────────────────────────────────────────┤
│  1. Load ontology (lazy, hash-checked)                      │
│     financial_topics.json → LanceDB topics table            │
│     Embeddings: "label: def" + "synonym: def" patterns      │
├─────────────────────────────────────────────────────────────┤
│  2. Resolve topics                                          │
│     a. Embed "topic: definition" (from extractor)           │
│     b. Vector search against ontology in LanceDB            │
│     c. Batch LLM verification (~10 per call, parallel)      │
├─────────────────────────────────────────────────────────────┤
│  3. Return TopicResolutionResult                            │
│     - resolved_topics: list[TopicResolution]                │
│     - uuid_remap: dict[str, str]                            │
│     - new_topics: list[str] (toggleable)                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Dual Storage Model

- **JSON file** (`vanna_kg/data/topics/financial_topics.json`): Source of truth, editable, version-controlled
- **LanceDB**: Search index loaded from JSON, used for vector similarity search

### 2. Embedding Pattern

Matches entity registry pattern: `"{name}: {definition}"`

For ontology loading, each topic generates multiple embeddings:
```
"Mergers And Acquisitions: Transactions in which ownership..."  (label)
"M&A: Transactions in which ownership..."                        (synonym)
"Acquisitions: Transactions in which ownership..."               (synonym)
```

All embeddings point to the same UUID.

For resolution, extracted topics use the same pattern:
```
"M&A: Corporate mergers and acquisitions activity"
```

### 3. Ontology Loading & Caching

**Lazy initialization with hash-based change detection:**

```
First resolve() call
    │
    ▼
Check .ontology_hash file exists?
    │
    ├─ No → Load JSON, compute hash, embed, write to LanceDB, save hash
    │
    └─ Yes → Compare hash with current JSON
                │
                ├─ Match → Use existing LanceDB index
                │
                └─ Different → Clear old entries, reload, update hash
```

**Hash file** (stored in LanceDB directory):
```json
{"hash": "sha256...", "loaded_at": "2026-01-29T..."}
```

**Manual reload**: `await resolver.reload_ontology()` forces reload regardless of hash.

### 4. Resolution Flow

**Input**: `list[TopicDefinition]` from extractor (topic + definition pairs)

**Steps**:
1. Embed topics using `"{topic}: {definition}"` pattern
2. Vector search against ontology in LanceDB (filtered by `ontology_group_id`)
3. Batch LLM verification (~10 topics per call, parallel with bounded concurrency)
4. Build `TopicResolutionResult`

### 5. Unmatched Topic Handling

- Unmatched topics collected in `new_topics` list for ontology review
- Toggleable via `collect_unmatched=True` (default) or `False`
- Unmatched topics excluded from `uuid_remap`

### 6. LLM Verification

**Batched for efficiency**: ~10 topics per LLM call

**Parallel with configurable concurrency**:
- `concurrency=3` (default): 3 concurrent batch calls
- `concurrency=-1`: unlimited parallelism

**Prompt structure**:
```
TOPICS TO VERIFY:

1. Extracted: "M&A" (def: Corporate mergers and acquisitions activity)
   Candidates:
   - Mergers And Acquisitions (89% similar): Transactions in which ownership...
   - Antitrust (45% similar): Laws and regulations designed to...

2. Extracted: "Fed Policy" (def: Federal Reserve monetary policy decisions)
   Candidates:
   - Monetary Policy (92% similar): Central bank actions to control...

For each, return the matching candidate number or null if no match.
```

## Types

### New types for `types/topics.py`:

```python
class TopicResolutionResult(BaseModel):
    """Result of resolving topics against the ontology."""

    resolved_topics: list[TopicResolution] = Field(
        default_factory=list,
        description="Successfully resolved topics"
    )
    uuid_remap: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from extracted topic name to ontology UUID"
    )
    new_topics: list[str] = Field(
        default_factory=list,
        description="Topics not found in ontology (for review)"
    )


class TopicMatchDecision(BaseModel):
    """LLM decision for a single topic match."""
    topic: str = Field(description="The extracted topic name")
    selected_number: int | None = Field(
        description="Candidate number (1-indexed) or null if no match"
    )
    reasoning: str = Field(description="Brief explanation of decision")


class BatchTopicMatchResponse(BaseModel):
    """Batched LLM response for topic verification."""
    decisions: list[TopicMatchDecision]
```

## Class Interface

```python
class TopicResolver:
    """
    Resolves extracted topics against the curated ontology.

    Usage:
        resolver = TopicResolver(lancedb_indices, llm, embeddings)
        result = await resolver.resolve(topics)

        # result.resolved_topics - matched to ontology
        # result.uuid_remap - {topic_name: ontology_uuid}
        # result.new_topics - unmatched (if collect_unmatched=True)
    """

    def __init__(
        self,
        indices: LanceDBIndices,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        config: KGConfig | None = None,
        # Behavior toggles
        collect_unmatched: bool = True,
        # Batching & concurrency
        batch_size: int = 10,
        concurrency: int = 3,                 # -1 = unlimited
        # Search tuning
        candidate_limit: int = 15,
        similarity_threshold: float = 0.40,
        high_similarity_flag: float = 0.85,
        # Ontology
        ontology_group_id: str = "ontology",
    ): ...

    async def resolve(
        self,
        topics: list[TopicDefinition],
        embeddings: list[list[float]] | None = None,
    ) -> TopicResolutionResult:
        """Resolve topics against ontology."""
        ...

    async def reload_ontology(self) -> None:
        """Force reload ontology from JSON, ignoring hash."""
        ...

    async def _ensure_ontology_loaded(self) -> None:
        """Lazy load with hash check."""
        ...
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `vanna_kg/data/topics/financial_topics.json` | Create | Copy from VannaLabsKG (~140 topics) |
| `vanna_kg/ingestion/resolution/topic_resolver.py` | Create | Main resolver class |
| `vanna_kg/types/topics.py` | Modify | Add `TopicResolutionResult`, `TopicMatchDecision`, `BatchTopicMatchResponse` |
| `vanna_kg/ingestion/resolution/__init__.py` | Modify | Export `TopicResolver` |
| `tests/test_topic_resolver.py` | Create | Unit tests |
| `pyproject.toml` | Modify | Include `data/topics/*.json` in package |

## Dependencies

All already present in project:
- `LanceDBIndices` from `storage/lancedb/indices.py`
- `LLMProvider`, `EmbeddingProvider` from `providers/base.py`
- `KGConfig` from `config/settings.py`

## Test Plan

1. **Unit tests** (`tests/test_topic_resolver.py`):
   - Ontology loading from JSON
   - Hash-based reload detection
   - Single topic resolution (match found)
   - Single topic resolution (no match)
   - Batch resolution with mixed results
   - `collect_unmatched=False` behavior
   - Concurrency settings (-1 unlimited)

2. **Mock strategy**:
   - Mock LLM provider for deterministic responses
   - Mock embedding provider with fixed vectors
   - Small test ontology (5-10 topics)

## Constants (Tunable Defaults)

| Constant | Default | Description |
|----------|---------|-------------|
| `candidate_limit` | 15 | Max candidates from vector search |
| `similarity_threshold` | 0.40 | Minimum similarity to consider |
| `high_similarity_flag` | 0.85 | Flag as "likely match" in LLM prompt |
| `batch_size` | 10 | Topics per LLM verification call |
| `concurrency` | 3 | Parallel batches (-1 = unlimited) |
| `ontology_group_id` | "ontology" | LanceDB group_id for ontology entries |
