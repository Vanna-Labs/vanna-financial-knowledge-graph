# Entity Deduplication System

## Phase 2 of the VannaKG Pipeline

---

## 1. Executive Summary

### Why Deduplication is Critical

Entity deduplication is the foundation of knowledge graph quality. Without it, the same real-world entity would exist as multiple nodes, fragmenting knowledge and destroying the graph's ability to answer questions accurately.

Consider a financial document that mentions "Apple", "Apple Inc.", "AAPL", and "Apple headquarters" across different paragraphs. Without deduplication:
- A query about Apple's relationships would miss 75% of the connections
- Traversal algorithms would fail to connect related facts
- The graph would grow unbounded with redundant nodes
- Semantic search would return incomplete results

The deduplication system ensures that every real-world entity has exactly one canonical representation in the graph, while tracking all aliases and merging information from multiple mentions.

### Design Philosophy

The system operates on three principles:

1. **Wide Net, Precise Filter**: Use embedding similarity to cast a wide net for potential duplicates, then use LLM verification for precise decisions
2. **Subsidiaries Are Separate**: Parent companies and subsidiaries maintain distinct identities (Google is not Alphabet; Instagram is not Meta)
3. **Traceability**: Every merge is recorded with full history for debugging and auditing

---

## 2. Three Levels of Deduplication

The pipeline implements deduplication at three distinct levels, each targeting a different source of duplication:

### Phase 2a-c: In-Document Deduplication

**Purpose**: Merge entity mentions that refer to the same real-world entity within a single document.

**When**: After extraction (Phase 1), before graph resolution (Phase 2d).

**Example**: Within an earnings call transcript, "Tim Cook", "Mr. Cook", "Apple's CEO Tim Cook", and "Timothy D. Cook" should all resolve to a single entity.

**Key Components**:
- Embedding generation for all extracted entities
- Cosine similarity matrix computation
- Union-Find connected components
- LLM verification of candidate clusters

### Phase 2d: Graph Entity Resolution

**Purpose**: Match new (deduplicated) entities against entities already stored in the knowledge base.

**When**: After in-document deduplication, in parallel with topic resolution.

**Example**: A new document mentions "Federal Reserve". The graph already contains a "Federal Reserve" node from previous documents. Phase 2d ensures UUID reuse rather than creating a duplicate.

**Key Components**:
- Vector search in LanceDB entity index for candidates (top 25)
- LLM verification of semantic match
- UUID reuse for matches, new UUID for novel entities
- Summary merging for enrichment

### Phase 2e: Topic Resolution Against Ontology

**Purpose**: Map extracted topic terms to canonical concepts in the curated topic ontology index.

**When**: In parallel with entity resolution (Phase 2d).

**Example**: Extracted topics like "M&A", "Mergers & Acquisitions", and "merger activity" should all resolve to the canonical "Mergers and Acquisitions" topic.

**Key Components**:
- Vector search in ontology-group LanceDB index
- LLM verification of semantic match
- Topic definition enrichment for better matching

---

## 3. In-Document Deduplication Algorithm

This section describes the complete algorithm for Phases 2a-c in sufficient detail for reimplementation.

### 3.1 Embedding Generation

All extracted entities are converted to embeddings for similarity comparison.

**Input**: List of entities, each with:
- `name`: Entity name as extracted
- `summary`: 1-2 sentence description from context

**Embedding Text Formula**:
```
embedding_text = f"{name}: {summary}" if summary else name
```

**Output**: List of embedding vectors (typically 1536 dimensions for OpenAI embeddings).

**Rationale**: Including the summary provides semantic context that helps distinguish entities with similar names but different meanings (e.g., "Apple Inc." vs "Apple Records").

### 3.2 Similarity Matrix Computation

A pairwise cosine similarity matrix is computed for all entity embeddings.

**Algorithm**:
```
1. Let E be the matrix of embeddings (n x d)
2. Normalize each row: E_norm[i] = E[i] / ||E[i]||
3. Compute similarity: S = E_norm * E_norm.T
4. S[i,j] = cosine similarity between entity i and entity j
```

**Cosine Similarity Formula**:
```
cosine_sim(a, b) = (a . b) / (||a|| * ||b||)
```

For normalized vectors, this simplifies to the dot product.

**Implementation Note**: Handle zero-norm vectors by replacing with 1 to avoid division by zero.

### 3.3 Threshold Selection

**Threshold Value**: 0.70 (configurable)

**Rationale**: The threshold is deliberately set lower than typical classification thresholds (0.85-0.90) because:
- This is a **candidate generation** step, not final classification
- LLM verification provides the precision filter
- False negatives (missed duplicates) are permanent; false positives (incorrect candidates) are filtered by LLM
- A threshold of 0.70 casts a wide net to ensure potential duplicates are not missed

**Empirical Basis**: At 0.70 threshold:
- "Apple Inc." and "AAPL" are connected (genuine duplicates)
- "Apple" and "Apple Records" may be connected (LLM will separate)
- "Google" and "Microsoft" are not connected (clearly distinct)

### 3.4 Connected Components via Union-Find

The similarity matrix is converted to a graph where edges connect entities with similarity above threshold. Connected components are found using Union-Find.

**Why Connected Components?**: Two entities A and C might not be directly similar, but if A is similar to B and B is similar to C, they form a candidate cluster for LLM verification. The transitive nature of Union-Find captures these chains.

---

## 4. Union-Find Algorithm Details

The Union-Find (Disjoint Set Union) data structure efficiently tracks connected components as similarity edges are processed.

### 4.1 Data Structure

```
parent: array[0..n-1]  -- parent[i] = parent of element i (or self if root)
rank: array[0..n-1]    -- rank[i] = upper bound on tree height rooted at i
```

**Initialization**:
```
for i in 0..n-1:
    parent[i] = i    -- Each element is its own root
    rank[i] = 0      -- All trees have height 0
```

### 4.2 Find Operation with Path Compression

The Find operation returns the root of the tree containing element x, while flattening the tree.

```
function find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  -- Path compression: point directly to root
    return parent[x]
```

**Path Compression Explanation**: After find(x), every node on the path from x to root points directly to root. This amortizes future operations to nearly O(1).

**Example**:
```
Before: 0 -> 1 -> 2 -> 3 (root)
find(0) returns 3
After:  0 -> 3, 1 -> 3, 2 -> 3
```

### 4.3 Union Operation with Union by Rank

The Union operation merges two trees, keeping the result balanced.

```
function union(x, y):
    px = find(x)  -- Root of x's tree
    py = find(y)  -- Root of y's tree

    if px == py:
        return  -- Already in same component

    -- Union by rank: attach smaller tree under larger
    if rank[px] < rank[py]:
        swap(px, py)

    parent[py] = px  -- py's tree goes under px's tree

    if rank[px] == rank[py]:
        rank[px] += 1  -- Only increment when ranks were equal
```

**Union by Rank Explanation**: By always attaching the shorter tree under the taller tree, we keep the combined tree balanced. This ensures find operations remain fast.

### 4.4 Extracting Components

After all unions are performed, extract the final components:

```
function get_components():
    components = {}  -- map: root -> list of members

    for i in 0..n-1:
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    return components
```

### 4.5 Why Union-Find is Efficient

**Time Complexity**: With both path compression and union by rank, any sequence of m operations on n elements runs in O(m * alpha(n)) time, where alpha is the inverse Ackermann function. For all practical purposes, alpha(n) <= 4, making operations essentially O(1) amortized.

**Space Complexity**: O(n) for the parent and rank arrays.

**Comparison to Alternatives**:
- BFS/DFS per component: O(n^2) worst case
- Graph adjacency matrix: O(n^2) space
- Union-Find: O(n) space, nearly O(1) per operation

---

## 5. LLM Verification Process

After Union-Find identifies candidate clusters, an LLM makes the final deduplication decisions.

### 5.1 What the LLM Decides

For each connected component with 2+ entities, the LLM determines:
1. **How many distinct real-world entities** exist in the cluster
2. **Which entities should be merged** (member indices)
3. **The canonical name** for each distinct entity
4. **A merged summary** combining information from all members

### 5.2 Subsidiary Awareness Enforcement

The LLM prompt explicitly instructs:

**MERGE - same entity, different names**:
- Ticker and company: "AAPL" = "Apple Inc." = "Apple"
- Abbreviation and full name: "Fed" = "Federal Reserve"
- Person name variants: "Tim Cook" = "Timothy D. Cook"

**DO NOT MERGE - related but distinct**:
- Parent and subsidiary: "Alphabet" is not "Google" is not "YouTube"
- Person and company: "Tim Cook" is not "Apple"
- Product and company: "iPhone" is not "Apple"
- Competitors: "Goldman Sachs" is not "Morgan Stanley"

**Decision Test**: "In a knowledge graph, would these be the same node? Would facts about one apply to the other?"

### 5.3 Structured Output (DeduplicationResult)

The LLM returns structured output conforming to:

```
DeduplicationResult:
    distinct_entities: List[DistinctEntity]

DistinctEntity:
    canonical_name: str       -- Best/most complete name
    member_indices: List[int] -- 0-based indices of input entities
    merged_summary: str       -- Combined summary from all members
```

**Validation**: The system verifies that all input indices are covered. Missing indices are added as singletons.

### 5.4 Handling Large Components with Batching

When a connected component exceeds the batch size (default: 15 entities), special handling is required.

**Strategy**:

1. **Order by Similarity Traversal**: Use greedy BFS starting from the first entity, always picking the most similar unvisited entity next. This ensures similar entities are adjacent in the ordering.

2. **Process in Overlapping Batches**: Process batches of 15 with 5-entity overlap to catch entities that might span batch boundaries.

3. **Merge Results with Conflict Resolution**:
   - If an entity appears in multiple batch results:
     - If groups have >80% embedding similarity: auto-merge
     - Otherwise: keep entity in its first-assigned group

**Similarity Traversal Algorithm**:
```
visited = [False] * n
order = []
current = 0
visited[0] = True
order.append(0)

while len(order) < n:
    best_next = -1
    best_sim = -1

    for visited_idx in order[-10:]:  -- Check last 10 for efficiency
        for j in 0..n-1:
            if not visited[j] and similarity[visited_idx, j] > best_sim:
                best_sim = similarity[visited_idx, j]
                best_next = j

    visited[best_next] = True
    order.append(best_next)
```

### 5.5 Batch Conflict Resolution

When processing large components in batches, overlapping entities may be assigned to different groups in different batches.

**Resolution Logic**:
```
for each distinct_entity in batch_result:
    orig_indices = map batch indices to original indices

    existing_key = None
    for idx in orig_indices:
        if idx in entity_to_canonical:
            existing_key = entity_to_canonical[idx]
            break

    if existing_key and existing_key != new_key:
        -- Conflict: entity assigned to two different groups

        similarity = compute similarity between group representatives

        if similarity > 0.80:
            -- Auto-merge: high similarity suggests same entity
            merge new indices into existing group
        else:
            -- Keep entity in first-assigned group
            -- Add only new entities to new group
```

---

## 6. Graph Entity Resolution (Phase 2d)

After in-document deduplication, canonical entities are resolved against the existing knowledge graph.

### 6.1 Vector Search in LanceDB

For each canonical entity, perform a vector similarity search:

**Index Used**: `entity_name_embeddings`

**Query Vector**: Embedding of `"{canonical_name}: {merged_summary}"`

**Filters**: `group_id = {current_group_id}` (multi-tenant isolation)

**Top-k**: 25 candidates

**Rationale for 25 Candidates**: High enough to catch semantic matches that might not be in top-5, but low enough for efficient LLM verification.

### 6.2 LLM Verification Against Existing Entities

The LLM receives:
- New entity name, type, and summary
- List of 25 candidates with their names, types, summaries, and aliases

**Decision Output** (EntityMatchDecision):
```
is_same: bool        -- True if new entity matches an existing one
match_index: int     -- 1-based index of matching candidate (or null)
reasoning: str       -- Explanation of the decision
```

### 6.3 UUID Reuse for Matches

When a match is found:
```
matched = candidates[match_index - 1]  -- Convert to 0-based

resolution = EntityResolution(
    uuid = matched.uuid,        -- REUSE existing UUID
    canonical_name = matched.name,
    is_new = False,
    updated_summary = merge_summaries(matched.summary, new_summary),
    source_chunks = matched.source_chunks + [current_chunk],
    aliases = matched.aliases + [new_name] if new_name != matched.name
)
```

When no match is found:
```
resolution = EntityResolution(
    uuid = generate_new_uuid(),  -- New UUID
    canonical_name = entity_name,
    is_new = True,
    updated_summary = f"{summary}\n[Source: {chunk_uuid}]",
    source_chunks = [chunk_uuid],
    aliases = []
)
```

### 6.4 Summary Merging Strategy

When merging summaries from matched entities:

1. **Empty Check**: If existing summary is empty, use new summary with source annotation
2. **Containment Check**: If new summary is contained in existing (or vice versa), keep the longer one
3. **LLM Merge**: For distinct information, use LLM to combine:
   - Preserve existing source annotations `[Source: ...]`
   - Add new source annotation
   - Avoid redundancy
   - Maintain factual accuracy

**LLM Merge Prompt Context**:
```
EXISTING SUMMARY:
{existing_summary}

NEW INFORMATION FROM CHUNK {chunk_uuid}:
{new_summary}

Merge these into a single comprehensive summary.
```

---

## 7. Topic Resolution (Phase 2e)

Topics are resolved against a curated ontology index, rather than the free-form entity index.

### 7.1 Ontology Index Structure

The topic ontology is stored in a dedicated LanceDB table/group with:

**Collection Name**: `topic_ontology`

**Vector Dimension**: 1536 (Voyage AI embeddings)

**Payload Fields**:
- `uri`: Unique identifier for the topic
- `label`: Canonical topic name
- `definition`: Description of the topic's meaning
- `synonyms`: Alternative names/phrases for the topic

**Example Entry**:
```
{
    "uri": "fibo:mergers_and_acquisitions",
    "label": "Mergers and Acquisitions",
    "definition": "Corporate transactions involving the combining of two companies or the purchase of one company by another",
    "synonyms": "M&A, mergers, acquisitions, corporate combinations"
}
```

### 7.2 Vector Search for Topic Candidates

**Query Process**:
1. Generate embedding for extracted topic (optionally enriched with definition)
2. Query ontology LanceDB index for top-k similar topics (default k=15)
3. Filter candidates below threshold (default 0.40)
4. Pass viable candidates to LLM

**Enrichment for Better Matching**:
```
if topic has extracted definition:
    search_text = f"{topic_name}: {definition}"
else:
    search_text = topic_name
```

### 7.3 LLM Verification of Semantic Matches

The LLM receives:
- Extracted topic name
- Extracted definition (if available)
- Source context where topic appeared
- Candidate topics with their definitions and synonyms

**Prompt Structure**:
```
TASK: Match an extracted topic to its canonical form from our ontology.

EXTRACTED TOPIC: "{topic_name}"
EXTRACTED DEFINITION: "{definition}"

SOURCE CONTEXT:
"{context}"

CANDIDATE TOPICS FROM ONTOLOGY:
1. {label}: {definition} (e.g., {synonyms})
2. ...

Return the matching candidate number (1-N), or null if no reliable match.
```

**Decision Output** (TopicResolutionResponse):
```
selected_number: int or null  -- 1-indexed candidate, or null if no match
```

### 7.4 Topic Definition Enrichment

When multiple topics are extracted from a single chunk, definitions are generated in batch:

**Purpose**: Provide context for better semantic matching against the ontology.

**Batch Definition Request**:
```
Define each financial/business topic in one sentence.

CONTEXT:
"{chunk_text}"

TOPICS TO DEFINE:
- Inflation
- M&A
- Rate Cuts

For each topic, provide a concise one-sentence definition.
```

**Output** (BatchTopicDefinitions):
```
definitions: [
    { topic: "Inflation", definition: "The rate at which general price levels increase over time" },
    { topic: "M&A", definition: "Corporate transactions involving company mergers or acquisitions" },
    { topic: "Rate Cuts", definition: "Federal Reserve reductions in the federal funds target rate" }
]
```

---

## 8. Resolution Output

### 8.1 UUID Remapping Structure

The deduplication system maintains a UUID remapping dictionary:

```
uuid_remap: Dict[str, str]
    key: original UUID (from initial extraction)
    value: canonical UUID (after deduplication)
```

**Usage During Fact Assembly**:
```
def get_remapped_uuid(original_uuid):
    return uuid_remap.get(original_uuid, original_uuid)

# When writing facts:
fact.subject_uuid = get_remapped_uuid(fact.subject_uuid)
fact.object_uuid = get_remapped_uuid(fact.object_uuid)
```

### 8.2 Merge History for Traceability

Every merge is recorded as a MergeRecord:

```
MergeRecord:
    canonical_uuid: str          -- UUID of the surviving entity
    canonical_name: str          -- Name chosen as canonical
    merged_uuids: List[str]      -- All UUIDs that were merged
    merged_names: List[str]      -- All names that were merged
    original_summaries: List[str] -- Summaries before merge
    final_summary: str           -- Combined summary after merge
```

**Use Cases**:
- Debugging unexpected merges
- Auditing deduplication decisions
- Understanding provenance of entity information

### 8.3 Canonical Name Selection

When multiple names exist for the same entity, the canonical name is selected by:

1. **LLM Preference**: The LLM explicitly chooses the "best/most complete name"
2. **Fallback Heuristics**:
   - Prefer formal names: "Apple Inc." over "Apple"
   - Prefer full names: "Federal Reserve" over "Fed"
   - Prefer names with more context: "Tim Cook" over "Cook"

**Canonical Entity Selection for UUID**:
When merging entities, the entity with the longest summary is selected as canonical (for UUID preservation), but the name is determined by the LLM.

```
canonical_entity = max(member_entities, key=lambda e: len(e.summary))
canonical_entity.name = llm_chosen_canonical_name  -- Override name
canonical_entity.summary = llm_merged_summary       -- Override summary
```

---

## 9. Concurrency Patterns

The deduplication system uses careful concurrency to maximize throughput while respecting rate limits.

### 9.1 Parallel Entity vs Topic Resolution

Entity resolution (Phase 2d) and topic resolution (Phase 2e) run in parallel:

```
entity_lookup, topic_lookup = await asyncio.gather(
    self._resolve_entities(...),
    self._resolve_topics(...)
)
```

**Rationale**: These operations are independent (different data stores, different LLM prompts) and both are I/O-bound.

### 9.2 Semaphore Controls

Concurrency is controlled via asyncio.Semaphore at multiple levels:

**Extraction Concurrency** (Phase 1):
- Default: 5 concurrent extractions
- Controlled by main `concurrency` parameter

**Embedding Concurrency** (Phase 2a):
- Default: `concurrency // 2` (at least 2)
- Batched to reduce API calls

**Deduplication LLM Concurrency** (Phase 2b-c):
- Default: `concurrency * 4` (e.g., 20 for concurrency=5)
- Higher because dedup LLM calls are smaller/faster

**Resolution Concurrency** (Phase 2d-e):
- Default: `concurrency * 10` (e.g., 50 for concurrency=5)
- Highest because these are simple verification calls

**Implementation Pattern**:
```
sem = asyncio.Semaphore(concurrency)

async def process_item(item):
    async with sem:
        return await asyncio.to_thread(blocking_operation, item)

tasks = [process_item(i) for i in items]
results = await asyncio.gather(*tasks)
```

### 9.3 Rate Limit Handling

Rate limits are handled through:

1. **Semaphore-based throttling**: Limits concurrent requests
2. **Batching**: Groups API calls (especially embeddings) to reduce request count
3. **Error fallback**: On LLM failure, defaults to treating entities as distinct (safe failure mode)

**Embedding Batching**:
```
batch_size = 100  -- Texts per embedding API call

async def embed_in_batches(texts, embeddings, batch_sz, sem):
    results = []
    for i in range(0, len(texts), batch_sz):
        batch = texts[i:i+batch_sz]
        async with sem:
            batch_embeddings = await asyncio.to_thread(
                embeddings.embed_documents,
                batch
            )
        results.extend(batch_embeddings)
    return results
```

---

## 10. Complete Algorithm Summary

For an AI agent to reimplement the full deduplication pipeline:

### Phase 2a: Embedding Generation
1. Collect all entities from extraction results
2. Generate embedding text: `f"{name}: {summary}"`
3. Batch-embed all entity texts

### Phase 2b: Clustering
1. Compute cosine similarity matrix
2. Initialize Union-Find with n elements
3. For each pair (i,j) where similarity[i,j] > 0.70: union(i, j)
4. Extract connected components

### Phase 2c: LLM Verification
1. For each component with 2+ entities:
   - If size <= 15: process directly
   - If size > 15: process in overlapping batches
2. LLM identifies distinct entities within each component
3. Build UUID remapping from merged entities
4. Record merge history

### Phase 2d: Graph Entity Resolution
1. For each canonical entity (after in-doc dedup):
   - Vector search LanceDB entity index for top-25 candidates
   - LLM verifies match
   - If match: reuse UUID, merge summary
   - If no match: create new UUID

### Phase 2e: Topic Resolution
1. For each extracted topic:
   - Generate definition (batch for efficiency)
   - Vector search ontology-group LanceDB index
   - LLM verifies semantic match
   - Map to canonical topic or skip if not in ontology

### Output
- `entity_lookup`: Map from original name to EntityResolution
- `topic_lookup`: Map from original name to TopicResolution
- Both contain UUIDs that facts will reference in Phase 3

---

## Appendix: Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.70 | Minimum cosine similarity for clustering |
| `ENTITY_SEARCH_LIMIT` | 25 | Top-k candidates for graph resolution |
| `max_batch_size` | 15 | Max entities per LLM dedup call |
| `candidate_threshold` | 0.40 | Minimum similarity for topic candidates |
| `top_k` (topics) | 15 | Candidates retrieved from ontology |
| `dedup_concurrency` | concurrency * 4 | Max concurrent dedup LLM calls |
| `resolve_concurrency` | concurrency * 10 | Max concurrent resolution calls |
| `embedding_batch_size` | 100 | Texts per embedding API call |
