# Assembly and Bulk Write System (Phase 3)

## Executive Summary

Phase 3 of the ZommaKG ingestion pipeline transforms resolved entities, topics, and extracted facts into a complete knowledge graph stored in the embedded runtime (Parquet + DuckDB + LanceDB). This phase receives the outputs of Phase 1 (extraction) and Phase 2 (resolution/deduplication) and performs three critical sub-phases:

1. **Phase 3a: Operation Collection** - Gathers all nodes and relationships into a centralized buffer
2. **Phase 3b: Batch Embedding Generation** - Generates vector embeddings for all nodes in parallel
3. **Phase 3c: Bulk Write to Embedded Storage** - Writes all operations to Parquet tables and vector indices in optimized batches

The assembly phase is designed for efficiency (batched operations minimize round-trips), idempotency (safe reruns via MERGE semantics), and multi-tenancy (all nodes tagged with group_id).

---

## Three Sub-Phases

### Phase 3a: Operation Collection into Buffer

The first sub-phase iterates through all successful extractions and collects nodes and relationships into a `BulkWriteBuffer`. This buffer serves as an in-memory staging area that accumulates all storage operations before any writes occur.

**Algorithm:**

```
For each extraction in successful_extractions:
    1. Create EpisodicNode entry (chunk representation)
    2. For each entity in entity_lookup:
        - If entity.is_new: Add to entity_nodes list
        - If entity exists: Queue summary update
    3. For each topic in topic_lookup:
        - If topic.is_new and not already in buffer: Add to topic_nodes
    4. For each fact in extraction.facts:
        - Resolve subject and object from lookups
        - Generate stable fact_uuid from (group_id, chunk_uuid, subject, rel_type, object, fact_text, date_context)
        - Create FactNode entry
        - Create three relationships per fact (chunk-centric pattern)
        - Create DISCUSSES relationships for associated topics
```

**Key Design Decision:** Embeddings are NOT generated during collection. All embedding fields are set to `None` and populated in Phase 3b. This allows the collection phase to complete quickly and enables batch embedding generation for maximum throughput.

### Phase 3b: Batch Embedding Generation

The second sub-phase generates vector embeddings for all nodes that require them. This is performed asynchronously with controlled concurrency to maximize throughput while respecting API rate limits.

**Embedding Types Generated:**

| Node Type | Embedding Field | Text Used | Embedding Client |
|-----------|-----------------|-----------|------------------|
| EntityNode | `name_embedding` | "{name}: {summary}" | Primary embeddings |
| EntityNode | `name_only_embedding` | "{name}" | Primary embeddings |
| TopicNode | `embedding` | "{name}" | Primary embeddings |
| FactNode | `embedding` | "{fact content}" | Dedup embeddings |

**Concurrency Model:**

The system uses separate semaphores for different embedding clients to prevent conflicts:
- `sem_main`: Controls concurrency for entity and topic embeddings
- `sem_dedup`: Controls concurrency for fact embeddings

All four embedding types (entity text, entity name-only, topic, fact) are generated in parallel using `asyncio.gather()`, with internal batching within each type.

### Phase 3c: Bulk Write to Embedded Storage

The final sub-phase executes all buffered operations against Parquet-backed storage and LanceDB indices in a specific order with optimized batch sizes.

**Write Order (Critical):**

1. DocumentNode (single MERGE)
2. EpisodicNodes (batched, with CONTAINS_CHUNK edges to document)
3. EntityNodes (batched CREATE for new entities only)
4. Entity summary updates (batched, with LLM merge for existing entities)
5. FactNodes (batched MERGE)
6. TopicNodes (batched MERGE)
7. Relationships (batched, grouped by type)

This order ensures referential integrity - parent nodes exist before child nodes and relationships reference them.

---

## BulkWriteBuffer Structure

The `BulkWriteBuffer` dataclass is the central data structure for Phase 3. It collects all operations that will be written to embedded storage.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `document_uuid` | str | UUID of the parent DocumentNode |
| `document_name` | str | Human-readable document identifier |
| `document_date` | Optional[str] | Document date in ISO format |
| `group_id` | str | Tenant identifier for multi-tenant isolation |
| `episodic_nodes` | List[Dict] | Chunk nodes to create |
| `entity_nodes` | List[Dict] | NEW entities only (not existing) |
| `entity_updates` | List[Dict] | Summary updates for existing entities |
| `fact_nodes` | List[Dict] | Facts to create |
| `topic_nodes` | List[Dict] | Topics to create |
| `relationships` | List[Dict] | All edges to create |
| `_created_topics` | Dict[str, str] | Internal tracker: label -> uuid |

### EpisodicNodes (Chunks)

Each chunk from the document becomes an EpisodicNode:

```
{
    "uuid": "<stable-uuid-from-group-doc-chunk>",
    "content": "<raw chunk text with headers prepended>",
    "header_path": "<breadcrumb path e.g. 'Report > Summary > Q3'>"
}
```

The UUID is deterministic, generated from `stable_uuid(group_id, doc_id, chunk_id)`. This ensures the same chunk in a rerun produces the same UUID.

### EntityNodes (New Entities Only)

Only entities where `EntityResolution.is_new == True` are added here:

```
{
    "uuid": "<uuid-from-resolution>",
    "name": "<canonical_name>",
    "summary": "<updated_summary>",
    "group_id": "<tenant-id>",
    "embedding": None,  # Populated in Phase 3b
    "name_only_embedding": None  # Populated in Phase 3b
}
```

### Entity Summary Updates (Existing Entities)

For entities that already exist in the graph, only summary updates are queued:

```
{
    "uuid": "<existing-entity-uuid>",
    "new_summary": "<summary-from-this-extraction>"
}
```

These updates require LLM-based summary merging during the write phase to combine old and new information intelligently.

### FactNodes

Each extracted fact becomes a FactNode:

```
{
    "uuid": "<stable-fact-uuid>",
    "content": "<atomic fact statement>",
    "group_id": "<tenant-id>",
    "embedding": None,  # Populated in Phase 3b
    "subject": "<subject canonical name>",  # For semantic fact indexing
    "object": "<object canonical name>",    # For semantic fact indexing
    "edge_type": "<normalized relationship type>"  # For semantic fact indexing
}
```

The fact UUID is deterministic, generated from: `stable_uuid(group_id, chunk_uuid, subject_name, rel_type, object_name, fact_text, date_context)`.

### TopicNodes

Topics resolved against the ontology:

```
{
    "uuid": "<topic-uuid-from-resolution>",
    "name": "<canonical_label from ontology>",
    "definition": "<definition from ontology>",
    "group_id": "<tenant-id>",
    "embedding": None  # Populated in Phase 3b
}
```

The `_created_topics` dict prevents duplicates across chunks when the same topic appears multiple times.

### Relationships (All Edges)

All edges are collected into a unified list:

```
{
    "from_uuid": "<source-node-uuid>",
    "to_uuid": "<target-node-uuid>",
    "rel_type": "<UPPER_SNAKE_CASE type>",
    "properties": {  # Optional, may be None
        "fact_id": "<linking-fact-uuid>",
        "description": "<original relationship text>",
        "date_context": "<temporal context>"
    }
}
```

---

## Relationship Collection Pattern

### Chunk-Centric Fact Pattern

ZommaKG uses a **chunk-centric fact pattern** where relationships flow through the EpisodicNode (chunk) that contains them. This design enables precise source attribution - every fact can be traced back to its exact source chunk.

**Pattern Structure:**

```
Subject Entity ----[REL_TYPE]----> EpisodicNode ----[REL_TYPE_TARGET]----> Object Entity
                                        |
                                        |----[CONTAINS_FACT]----> FactNode
```

### Three Relationships Per Fact

For each extracted fact, exactly three relationships are created:

#### 1. Subject to EpisodicNode

```
(SubjectEntity)-[REL_TYPE {fact_id, description, date_context}]->(EpisodicNode)
```

This edge connects the subject entity to the chunk where the relationship was mentioned. The relationship type is the normalized version of the extraction's relationship field (e.g., "acquired majority stake in" becomes `ACQUIRED_MAJORITY_STAKE_IN`).

#### 2. EpisodicNode to Object

```
(EpisodicNode)-[REL_TYPE_TARGET {fact_id, description, date_context}]->(ObjectEntity)
```

This edge completes the path to the object entity. The `_TARGET` suffix distinguishes this edge from the subject edge, enabling directional traversal.

#### 3. EpisodicNode to FactNode

```
(EpisodicNode)-[CONTAINS_FACT]->(FactNode)
```

This edge links the chunk to the atomic fact statement, enabling fact retrieval and semantic search.

### Topic Relationships (DISCUSSES)

When a fact is associated with topics, additional edges are created:

```
(EpisodicNode)-[DISCUSSES]->(TopicNode)
```

These edges have no properties and use simple MERGE semantics (no duplicates if the same chunk discusses the same topic multiple times).

### fact_id Linking

The `fact_id` property appears on both the Subject->Chunk and Chunk->Object relationships. This enables:

1. **Traversal**: Find both endpoints of a fact relationship
2. **Deduplication**: MERGE on fact_id prevents duplicate relationships on reruns
3. **Attribution**: Link relationships back to their source FactNode

---

## Batch Embedding Generation

### Parallel Embedding Types

Phase 3b generates embeddings for four distinct purposes:

| Type | Purpose | Text Formula |
|------|---------|--------------|
| Entity (full) | Semantic search on entity content | `{name}: {summary}` |
| Entity (name-only) | Direct name matching | `{name}` |
| Topic | Topic similarity search | `{name}` |
| Fact | Semantic fact retrieval | `{fact content}` |

Entity and topic embeddings use the primary embedding client. Fact embeddings use the dedup embedding client (potentially a different model optimized for short texts).

### Batch Size Strategies

**Default batch size: 128 texts per API call**

The batch size is configurable via the `batch_size` parameter. Larger batches reduce API overhead but increase memory usage and latency per request.

**Embedding batching algorithm:**

```
function embed_in_batches_async(texts, embed_fn, batch_sz, semaphore):
    batches = split texts into groups of batch_sz
    tasks = []
    for each (batch, batch_index):
        task = async embed_batch(batch, batch_index)
        tasks.append(task)
    results = await gather(tasks)
    sort results by batch_index
    flatten and return all embeddings
```

### Rate Limit Handling with Backoff

The embedding system includes automatic retry logic for rate limits:

```
async function embed_batch(batch, batch_idx):
    async with semaphore:
        try:
            result = await embed_documents(batch)
            return (batch_idx, result)
        except error:
            if "rate limit" in error:
                await sleep(30 seconds)
                result = await embed_documents(batch)
                return (batch_idx, result)
            raise error
```

The 30-second backoff provides time for rate limit windows to reset. In production, this could be enhanced with exponential backoff.

### Concurrency Controls

**Default concurrency: 10 concurrent embedding API calls per client**

Concurrency is controlled via `asyncio.Semaphore`:

- `sem_main` (concurrency=10): Limits concurrent calls to primary embedding client
- `sem_dedup` (concurrency=10): Limits concurrent calls to dedup embedding client

Since entity and topic embeddings share `sem_main`, their combined concurrency is capped at 10. Fact embeddings use a separate semaphore, allowing an additional 10 concurrent calls.

**Total maximum concurrent embedding API calls: 20** (10 main + 10 dedup)

---

## Embedded Bulk Write Strategy

### Batch Write Pattern for Storage Backend

All bulk writes use backend-specific batch APIs for efficient processing:

```python
storage.write_entities_batch(nodes)
```

This pattern sends batched operations instead of issuing one write per item.

### MERGE vs CREATE Decision Logic

| Node Type | Operation | Reasoning |
|-----------|-----------|-----------|
| DocumentNode | MERGE | Idempotent - same doc rerun safe |
| EpisodicNode | MERGE | Same chunk should not duplicate |
| EntityNode (new) | CREATE | Only new entities; UUIDs guaranteed unique |
| FactNode | MERGE | Same fact may be mentioned again |
| TopicNode | MERGE on name | Topics are identified by canonical label |

**Relationships:**

| Relationship Category | Operation | Key |
|-----------------------|-----------|-----|
| Simple (no properties) | MERGE | (from_uuid, rel_type, to_uuid) |
| With fact_id property | MERGE on fact_id | fact_id ensures uniqueness |
| With properties, no fact_id | CREATE | Each occurrence is distinct |

The fact_id-based MERGE prevents duplicate relationships when rerunning the pipeline on the same document.

### Relationship Grouping by Type

Relationships are grouped by type before writing. This optimization:

1. Reduces query compilation overhead (same query structure per type)
2. Enables type-specific handling (properties vs. no properties)
3. Improves batch efficiency (similar operations together)

**Grouping algorithm:**

```
by_type = {}
for each relationship:
    by_type[rel_type].append(relationship)

for each (rel_type, rels) in by_type:
    with_props = filter(rels where properties exist)
    without_props = filter(rels where properties is None)

    # Batch write without_props using simple MERGE
    # Batch write with_props using MERGE on fact_id (or CREATE if no fact_id)
```

### Batch Size Optimization

**Default batch size: 250 items per query**

This value balances:
- **Transaction overhead**: Fewer queries = less overhead
- **Memory usage**: Larger batches consume more memory
- **Timeout risk**: Very large batches may timeout
- **Recovery granularity**: Smaller batches mean less work lost on failure

The batch size is configurable via the `batch_size` parameter in `bulk_write_all()`.

### Order of Operations

The write order is critical for referential integrity:

```
1. DocumentNode
   └── Establishes document root

2. EpisodicNodes + CONTAINS_CHUNK edges
   └── Creates chunks and links to document
   └── Must exist before relationships reference them

3. EntityNodes (CREATE new only)
   └── Creates new entity nodes
   └── Must exist before relationship edges

4. Entity Summary Updates (MATCH + SET)
   └── Updates existing entities with LLM-merged summaries
   └── Requires entities to already exist

5. FactNodes (MERGE)
   └── Creates/updates fact nodes
   └── Must exist before CONTAINS_FACT edges

6. TopicNodes (MERGE)
   └── Creates/updates topic nodes
   └── Must exist before DISCUSSES edges

7. Relationships (grouped by type)
   └── All edges created last
   └── References nodes from steps 2-6
```

**Post-Write: LanceDB Fact Indexing**

After facts are written to storage, they are also indexed to LanceDB for semantic fact search:

```
for each fact in buffer.fact_nodes:
    qdrant_fact = {
        fact_id, embedding, group_id,
        subject, object, edge_type, content
    }
    fact_store.index_facts_batch(qdrant_facts)
```

---

## Idempotency and Reruns

### How MERGE Ensures Safe Reruns

The pipeline is designed for safe reruns on the same document:

**DocumentNode:** MERGE on (uuid, group_id) - Same document produces same UUID

**EpisodicNode:** MERGE on (uuid, group_id) - Stable UUID from (group_id, doc_id, chunk_id)
- ON MATCH updates content and header_path (handles modified chunks)

**EntityNode:** Only new entities are CREATEd; existing entities receive summary updates via separate MATCH query

**FactNode:** MERGE on (uuid, group_id) - Stable UUID from fact signature
- ON MATCH updates content and embedding (handles refined extractions)

**TopicNode:** MERGE on (name, group_id) - Same topic name maps to same node
- ON MATCH preserves existing UUID, updates definition if provided

### fact_id for Relationship Deduplication

Without fact_id, rerunning the pipeline would create duplicate relationships:

```cypher
-- Without fact_id (BAD):
MERGE (a)-[:ACQUIRED]->(b)  -- Dedupes on (a, type, b) only
SET rel.description = "new value"  -- Overwrites previous fact!

-- With fact_id (GOOD):
MERGE (a)-[:ACQUIRED {fact_id: $fid}]->(b)  -- Dedupes on (a, type, b, fact_id)
SET rel += $properties  -- Each fact gets its own edge
```

The fact_id is a deterministic UUID generated from the fact's content signature, ensuring:
1. Same fact content = same fact_id = same edge (no duplicates)
2. Different facts between same entities = different edges (preserved)

### Checkpoint Integration

The checkpoint system saves state after each phase:

| Phase | Checkpoint File | Contents |
|-------|-----------------|----------|
| 1 | phase1_extraction.pkl | extractions, document metadata, chunks |
| 2 | phase2_resolution.pkl | entity_lookup, topic_lookup, dedup maps |
| 3 | phase3_buffer.pkl | BulkWriteBuffer, embeddings_done flag |

**Recovery Flow:**

```
If checkpoint exists for (input_file, group_id):
    Load checkpoint metadata
    Resume from (last_phase + 1)

    If last_phase >= 3 and embeddings_done:
        Skip to storage write only
    If last_phase == 2:
        Reconstruct buffer from checkpointed lookups
        Generate embeddings
        Write to storage backend
```

The `embeddings_done` flag in Phase 3 checkpoint enables partial recovery within the assembly phase itself.

---

## Vector Index Management

### Index Creation/Verification

Before any writes, the pipeline ensures required vector indexes exist:

```python
ensure_vector_indexes(neo4j):
    for index_name in [
        "entity_name_embeddings",
        "entity_name_only_embeddings",
        "topic_embeddings"
    ]:
        result = SHOW INDEXES WHERE name = index_name
        if not result:
            CREATE VECTOR INDEX ...
```

**Indexes Created:**

| Index Name | Node Label | Property | Purpose |
|------------|------------|----------|---------|
| entity_name_embeddings | EntityNode | name_embedding | Full entity search |
| entity_name_only_embeddings | EntityNode | name_only_embedding | Name-only lookup |
| topic_embeddings | TopicNode | embedding | Topic similarity |

### Embedding Dimensions

**All indexes use 3072 dimensions** (OpenAI text-embedding-3-large model)

This dimension count is hardcoded in the index creation queries:

```cypher
OPTIONS {indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
}}
```

### Similarity Function

**All indexes use cosine similarity**

Cosine similarity is appropriate for normalized embeddings and provides:
- Scale-invariant comparison (magnitude doesn't affect similarity)
- Efficient computation in LanceDB vector indices
- Interpretable scores in [0, 1] range (after normalization)

---

## Multi-Tenant Support

### group_id Tagging on All Nodes

Every node type includes a `group_id` property:

| Node Type | group_id Usage |
|-----------|----------------|
| DocumentNode | `{uuid: $uuid, group_id: $group_id}` |
| EpisodicNode | `{uuid: $uuid, group_id: $group_id}` |
| EntityNode | `{group_id: $group_id}` as property |
| FactNode | `{uuid: $uuid, group_id: $group_id}` |
| TopicNode | `{group_id: $group_id}` as property |

The group_id is set during Pipeline initialization and propagated through all operations.

### Isolation Guarantees

**Query-Level Isolation:**

All queries filter by group_id:
```cypher
MATCH (e:EntityNode {uuid: $uuid, group_id: $group_id})
```

**Vector Search Isolation:**

Vector searches include group_id as a filter:
```python
neo4j.vector_search(
    index_name="entity_name_embeddings",
    query_vector=embedding,
    filters={"group_id": self.group_id}
)
```

**MERGE Isolation:**

MERGE operations include group_id in the match criteria, ensuring:
- Tenant A's "Apple" entity is distinct from Tenant B's "Apple" entity
- Same document ingested by different tenants creates separate graphs

**Write Isolation:**

The BulkWriteBuffer carries group_id as a top-level field, ensuring all collected operations are tagged with the correct tenant identifier.

---

## Summary: AI Agent Reference

For an AI agent implementing or debugging this system:

### Critical Write Order
1. DocumentNode (MERGE)
2. EpisodicNodes (MERGE + CONTAINS_CHUNK)
3. EntityNodes (CREATE new)
4. Entity summaries (MATCH + SET with LLM merge)
5. FactNodes (MERGE + LanceDB index)
6. TopicNodes (MERGE)
7. Relationships (grouped MERGE/CREATE)

### Batch Sizes
- Embedding API calls: 128 texts per batch
- Storage writes: 250 items per batch
- Embedding concurrency: 10 per client (20 total)

### UUID Generation
- Documents: `stable_uuid(group_id, doc_id)`
- Chunks: `stable_uuid(group_id, doc_id, chunk_id)`
- Facts: `stable_uuid(group_id, chunk_uuid, subject, rel_type, object, fact_text, date_context)`
- Entities: UUID4 for new, existing UUID from graph for matched

### Relationship Pattern Per Fact
```
Subject --[REL_TYPE {fact_id}]--> Chunk --[REL_TYPE_TARGET {fact_id}]--> Object
                                   |
                                   +--[CONTAINS_FACT]--> FactNode
                                   +--[DISCUSSES]--> TopicNode (for each topic)
```

### Idempotency Keys
- Nodes: (uuid, group_id)
- Topics: (name, group_id)
- Relationships with properties: (from_uuid, rel_type, to_uuid, fact_id)
- Simple relationships: (from_uuid, rel_type, to_uuid)
