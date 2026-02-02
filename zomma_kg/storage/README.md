# Parquet Storage Migration: From Neo4j to Embedded Architecture

## Executive Summary

### Why Migrate Away from Neo4j?

The current ZommaLabsKG system uses Neo4j Aura as its graph database, which presents significant barriers to the project's goal of becoming a pip-installable Python library with zero infrastructure requirements.

**Current Pain Points:**

1. **Infrastructure Dependency**: Users must provision and maintain a Neo4j Aura instance (or self-host Neo4j). Even the free tier requires account creation, connection string management, and handling cold-start latency when instances sleep.

2. **Operational Overhead**: The codebase includes extensive retry logic, connection warmup procedures, and error handling specifically for Neo4j's transient failures (see `Neo4jClient.warmup()` with up to 10 attempts at 5-second intervals).

3. **Vendor Lock-in**: Cypher queries, vector index syntax, and Neo4j-specific features (like `db.index.vector.queryNodes`) tie the codebase to a specific database vendor.

4. **Under-utilization of Graph Features**: Analysis of actual query patterns reveals that the system primarily performs:
   - Vector similarity search
   - Simple 1-hop neighbor lookups
   - Property-based filtering
   - Join operations via `fact_id` linkages

   These operations do not require a graph database's advanced traversal capabilities (PageRank, shortest path, community detection, etc.).

5. **Cost at Scale**: Neo4j Aura pricing increases with data volume and query throughput, while embedded solutions have zero marginal cost.

**Migration Thesis**: The knowledge graph's actual query patterns map naturally to relational joins and vector search operations. By migrating to DuckDB (embedded analytics) + LanceDB (embedded vector search) with Parquet file storage, we achieve:
- Zero infrastructure requirements
- Portable knowledge bases (just a directory)
- No cold-start delays
- Simpler debugging (query local files)
- Full pip-installability

---

## Current Neo4j Usage Analysis

### Node Types and Their Properties

Based on the codebase analysis, the current Neo4j schema contains five node types:

| Node Type | Key Properties | Vector Index | Purpose |
|-----------|---------------|--------------|---------|
| **DocumentNode** | uuid, name, group_id, document_date | None | Source document metadata |
| **EpisodicNode** | uuid, content, header_path, group_id, document_date | None | Text chunks (the central node in star schema) |
| **EntityNode** | uuid, name, summary, group_id, name_embedding, name_only_embedding | entity_name_embeddings, entity_name_only_embeddings | Extracted entities (companies, people, etc.) |
| **FactNode** | uuid, content, group_id, embedding | fact_embeddings | Atomic fact statements |
| **TopicNode** | uuid, name, definition, group_id, embedding | topic_embeddings | Topic/concept nodes |

### Relationship Types

The system uses a **chunk-centric star schema** where EpisodicNode (chunks) serves as the central hub:

```
                    [DocumentNode]
                          |
                    CONTAINS_CHUNK
                          |
                          v
[EntityNode] --REL_TYPE--> [EpisodicNode] --REL_TYPE_TARGET--> [EntityNode/TopicNode]
                          |
                    DISCUSSES
                          |
                          v
                    [TopicNode]
```

**Key Relationship Pattern**: Facts are encoded as relationship pairs with a shared `fact_id`:
- `(Subject:EntityNode)-[REL_TYPE {fact_id, description, date_context}]->(chunk:EpisodicNode)`
- `(chunk:EpisodicNode)-[REL_TYPE_TARGET {fact_id, description, date_context}]->(Object:EntityNode|TopicNode)`

This pattern enables chunk-centric retrieval: given an entity, find all chunks where it appears, then find related entities through those chunks.

### Actual Query Patterns

Analysis of `graph_store.py` reveals the complete set of production queries:

#### 1. Entity Candidate Resolution (Vector Search)
```cypher
CALL db.index.vector.queryNodes('entity_name_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
RETURN node.name as name, node.summary as summary, score
ORDER BY score DESC
```
**Pattern**: Vector search with property filter, returning node properties.

#### 2. Topic Candidate Resolution (Vector Search)
```cypher
CALL db.index.vector.queryNodes('topic_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
RETURN node.name as name, score
ORDER BY score DESC
```
**Pattern**: Identical to entity resolution, different index.

#### 3. Entity Chunks Retrieval (1-Hop from Entity to Chunk)
```cypher
MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
      -[r]->(c:EpisodicNode {group_id: $uid})
WHERE r.fact_id IS NOT NULL
OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
RETURN DISTINCT c.uuid as chunk_id, c.content as content, ...
```
**Pattern**: Property lookup + 1-hop traversal to chunks + optional document join.

#### 4. Topic Chunks Retrieval (1-Hop from Chunk to Topic)
```cypher
MATCH (c:EpisodicNode {group_id: $uid})-[:DISCUSSES]->(t:TopicNode {name: $topic_name, group_id: $uid})
OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
RETURN DISTINCT c.uuid as chunk_id, c.content as content, ...
```
**Pattern**: 1-hop from chunk to topic (reversed direction), with document join.

#### 5. 1-Hop Neighbor Expansion (Entity to Entity via Chunk)
```cypher
MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
      -[r]->(c:EpisodicNode {group_id: $uid})
      -[r2]->(neighbor:EntityNode {group_id: $uid})
WHERE r.fact_id = r2.fact_id AND neighbor.name <> $entity_name
RETURN DISTINCT neighbor.name as name, neighbor.summary as summary, count(*) as connection_count
```
**Pattern**: 2-hop traversal (Entity -> Chunk -> Entity) with fact_id join condition.

#### 6. Entity Facts Retrieval (Vector Search + Traversal)
```cypher
CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
MATCH (subj:EntityNode {name: $entity_name, group_id: $uid})
      -[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
      -[r2 {fact_id: node.uuid}]->(obj)
WHERE (obj:EntityNode OR obj:TopicNode) AND obj.group_id = $uid
RETURN DISTINCT node.uuid as fact_id, node.content as content, ...
```
**Pattern**: Vector search for facts, then filter by entity involvement via fact_id join.

#### 7. Global Chunk Search (Vector Search + Chunk Lookup)
```cypher
CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
WHERE subj.group_id = $uid
OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
RETURN DISTINCT c.uuid as chunk_id, c.content as content, max(score) as score
```
**Pattern**: Vector search for facts, then join to chunks via relationship fact_id.

### Why These Patterns Don't Need a Graph Database

Every query above falls into one of these categories:

1. **Vector similarity search** with property filters -> LanceDB handles this natively
2. **Property-based lookup** (find node by name/uuid) -> Standard SQL `WHERE` clause
3. **1-hop traversal** with fact_id join -> SQL JOIN on relationships table
4. **2-hop traversal** (Entity -> Chunk -> Entity) -> Two SQL JOINs

The system never uses:
- Variable-length path traversal (`-[*1..5]->`)
- Graph algorithms (PageRank, Louvain, shortest path)
- Pattern matching with unbounded depth
- Cypher's `COLLECT` or `UNWIND` for graph aggregations

**Key Insight**: The `fact_id` property on relationships effectively denormalizes the graph into a relational model. Every "graph traversal" is actually a join on `fact_id`.

---

## Proposed Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Relational Queries** | DuckDB | Embedded analytics database, excellent at joins and aggregations |
| **Vector Search** | LanceDB | Embedded vector database with native Parquet support |
| **Data Storage** | Apache Parquet | Columnar format, portable, compressed |
| **Python Interface** | PyArrow + DuckDB | Zero-copy data access |

### Directory Structure

A knowledge base becomes a self-contained directory:

```
knowledge_base/
├── entities.parquet      # EntityNode equivalent
├── chunks.parquet        # EpisodicNode equivalent
├── facts.parquet         # FactNode equivalent
├── topics.parquet        # TopicNode equivalent
├── relationships.parquet # All edges (the key table!)
├── documents.parquet     # DocumentNode equivalent
├── lancedb/              # LanceDB vector index directory
│   ├── entities.lance/   # Entity embeddings
│   ├── facts.lance/      # Fact embeddings
│   └── topics.lance/     # Topic embeddings
└── metadata.json         # Version, group_id, creation date
```

### Why This Stack?

**DuckDB**:
- Embedded (runs in-process, no server)
- Reads Parquet files directly without loading into memory
- Excellent JOIN performance on columnar data
- SQL interface (familiar to most developers)
- Apache Arrow integration for zero-copy data exchange

**LanceDB**:
- Embedded vector database (no server)
- Native Parquet-like columnar storage (Lance format)
- Supports filtered vector search
- Persists to disk (unlike FAISS which requires manual serialization)
- Written in Rust, Python bindings available

**Parquet**:
- Self-describing (schema embedded in file)
- Efficient compression (especially for text columns)
- Columnar (fast scans for analytics)
- Portable (can be read by Spark, DuckDB, Pandas, etc.)
- Industry standard for data interchange

---

## Schema Design

### entities.parquet

Stores all EntityNode data. Vector embeddings are stored in LanceDB for search, but a reference is kept here.

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key, UUID4 |
| name | STRING | Canonical entity name |
| summary | STRING | Entity description/summary |
| group_id | STRING | Multi-tenant isolation |
| entity_type | STRING | PERSON, ORGANIZATION, LOCATION, etc. |
| created_at | TIMESTAMP | Creation timestamp |
| name_embedding_ref | STRING | Reference to LanceDB vector (optional, for debugging) |

**Indexes**: Primary key on `uuid`, secondary on `(group_id, name)`.

**Note**: Embeddings (`name_embedding`, `name_only_embedding`) are stored in LanceDB's `entities.lance` table, not in the Parquet file. This separates vector search from relational queries.

### chunks.parquet

Stores all EpisodicNode (chunk) data. This is the central table in the star schema.

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key, UUID4 |
| document_uuid | STRING | Foreign key to documents.parquet |
| content | STRING | Full text content of the chunk |
| header_path | STRING | Breadcrumb path (e.g., "Section 1 > Overview") |
| group_id | STRING | Multi-tenant isolation |
| document_date | STRING | Date from parent document (ISO format) |
| created_at | TIMESTAMP | Creation timestamp |

**Indexes**: Primary key on `uuid`, secondary on `(group_id, document_uuid)`.

### facts.parquet

Stores all FactNode data. Facts are atomic statements extracted from chunks.

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key, UUID4 (this is the fact_id used in relationships) |
| content | STRING | The fact statement text |
| group_id | STRING | Multi-tenant isolation |
| subject_uuid | STRING | Foreign key to entities.parquet (denormalized for fast lookup) |
| subject_name | STRING | Subject entity name (denormalized) |
| object_uuid | STRING | Foreign key to entities or topics |
| object_name | STRING | Object entity/topic name (denormalized) |
| object_type | STRING | "entity" or "topic" |
| edge_type | STRING | Relationship type (e.g., "ACQUIRED", "PARTNERED_WITH") |
| date_context | STRING | Temporal context for the fact |
| created_at | TIMESTAMP | Creation timestamp |

**Indexes**: Primary key on `uuid`, secondary on `(group_id, subject_uuid)`, `(group_id, object_uuid)`.

**Note**: Embeddings are stored in LanceDB's `facts.lance` table.

**Design Decision**: We denormalize subject/object names and include edge_type in the facts table. This allows fact retrieval without joining to entities, which matches the current query pattern where facts are searched and then displayed with their relationship context.

### relationships.parquet

**This is the key table that replaces Neo4j's graph edges.** Every edge in the original graph becomes a row here.

| Column | Type | Description |
|--------|------|-------------|
| id | STRING | Primary key, UUID4 |
| from_uuid | STRING | Source node UUID |
| from_type | STRING | Source node type: "entity", "chunk", "document", "topic" |
| to_uuid | STRING | Target node UUID |
| to_type | STRING | Target node type |
| rel_type | STRING | Relationship type (e.g., "ACQUIRED", "CONTAINS_CHUNK", "DISCUSSES") |
| fact_id | STRING | Foreign key to facts.parquet (nullable, only for fact-bearing edges) |
| description | STRING | Free-form relationship description |
| date_context | STRING | Temporal context |
| group_id | STRING | Multi-tenant isolation |
| created_at | TIMESTAMP | Creation timestamp |

**Indexes**:
- Primary key on `id`
- Composite on `(group_id, from_uuid, rel_type)`
- Composite on `(group_id, to_uuid, rel_type)`
- Index on `(group_id, fact_id)` for fact-based joins

**Relationship Types Stored**:
- `CONTAINS_CHUNK`: Document -> Chunk
- `DISCUSSES`: Chunk -> Topic
- `{EDGE_TYPE}`: Entity -> Chunk (with fact_id)
- `{EDGE_TYPE}_TARGET`: Chunk -> Entity/Topic (with fact_id)

**Query Optimization**: The key insight is that most traversals are "find chunks for entity" or "find entities in chunk". By indexing on `(from_uuid, rel_type)` and `(to_uuid, rel_type)`, these become simple index scans.

### topics.parquet

Stores all TopicNode data.

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key, UUID4 |
| name | STRING | Canonical topic name |
| definition | STRING | Topic definition from ontology |
| group_id | STRING | Multi-tenant isolation |
| created_at | TIMESTAMP | Creation timestamp |

**Indexes**: Primary key on `uuid`, secondary on `(group_id, name)`.

**Note**: Embeddings are stored in LanceDB's `topics.lance` table.

### documents.parquet

Stores source document metadata.

| Column | Type | Description |
|--------|------|-------------|
| uuid | STRING | Primary key, UUID4 |
| name | STRING | Document name/title |
| document_date | STRING | Document date (ISO format) |
| group_id | STRING | Multi-tenant isolation |
| source_path | STRING | Original file path (for provenance) |
| created_at | TIMESTAMP | Creation timestamp |

**Indexes**: Primary key on `uuid`, secondary on `(group_id, name)`.

---

## Query Translation

This section provides exact translations from current Cypher queries to SQL equivalents.

### 1. Entity Candidate Resolution

**Current Cypher:**
```cypher
CALL db.index.vector.queryNodes('entity_name_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
RETURN node.name as name, node.summary as summary, score
ORDER BY score DESC
```

**LanceDB + DuckDB Equivalent:**
```python
# Step 1: Vector search in LanceDB
import lancedb

db = lancedb.connect("knowledge_base/lancedb")
entities_table = db.open_table("entities")

results = (
    entities_table
    .search(query_vector)
    .where(f"group_id = '{group_id}'")
    .limit(top_k)
    .to_pandas()
)

# Filter by threshold
results = results[results['_distance'] < (1 - threshold)]  # LanceDB uses distance, not similarity

# Step 2: Get full entity data from Parquet (if needed)
# Often the LanceDB table includes all needed fields, making this optional
```

**Alternative (LanceDB stores all fields):**
```python
# Configure LanceDB table to include name, summary during indexing
# Then no Parquet join needed for this query
results = (
    entities_table
    .search(query_vector)
    .where(f"group_id = '{group_id}'")
    .select(["uuid", "name", "summary"])
    .limit(top_k)
    .to_pandas()
)
```

### 2. Topic Candidate Resolution

**Current Cypher:**
```cypher
CALL db.index.vector.queryNodes('topic_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
RETURN node.name as name, score
ORDER BY score DESC
```

**LanceDB Equivalent:**
```python
topics_table = db.open_table("topics")
results = (
    topics_table
    .search(query_vector)
    .where(f"group_id = '{group_id}'")
    .select(["uuid", "name", "definition"])
    .limit(top_k)
    .to_pandas()
)
```

### 3. Entity Chunks Retrieval

**Current Cypher:**
```cypher
MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
      -[r]->(c:EpisodicNode {group_id: $uid})
WHERE r.fact_id IS NOT NULL
OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
RETURN DISTINCT c.uuid as chunk_id, c.content as content,
       c.header_path as header_path, d.name as doc_id,
       d.document_date as document_date
```

**DuckDB SQL Equivalent:**
```sql
SELECT DISTINCT
    c.uuid AS chunk_id,
    c.content,
    c.header_path,
    d.name AS doc_id,
    d.document_date
FROM entities e
JOIN relationships r ON r.from_uuid = e.uuid
    AND r.from_type = 'entity'
    AND r.to_type = 'chunk'
    AND r.fact_id IS NOT NULL
JOIN chunks c ON c.uuid = r.to_uuid
LEFT JOIN documents d ON d.uuid = c.document_uuid
WHERE e.name = $entity_name
    AND e.group_id = $group_id
    AND c.group_id = $group_id
LIMIT $top_k
```

**Python with DuckDB:**
```python
import duckdb

conn = duckdb.connect()
conn.execute("CREATE VIEW entities AS SELECT * FROM 'knowledge_base/entities.parquet'")
conn.execute("CREATE VIEW chunks AS SELECT * FROM 'knowledge_base/chunks.parquet'")
conn.execute("CREATE VIEW relationships AS SELECT * FROM 'knowledge_base/relationships.parquet'")
conn.execute("CREATE VIEW documents AS SELECT * FROM 'knowledge_base/documents.parquet'")

result = conn.execute("""
    SELECT DISTINCT
        c.uuid AS chunk_id,
        c.content,
        c.header_path,
        d.name AS doc_id,
        d.document_date
    FROM entities e
    JOIN relationships r ON r.from_uuid = e.uuid
        AND r.from_type = 'entity'
        AND r.to_type = 'chunk'
        AND r.fact_id IS NOT NULL
    JOIN chunks c ON c.uuid = r.to_uuid
    LEFT JOIN documents d ON d.uuid = c.document_uuid
    WHERE e.name = ?
        AND e.group_id = ?
        AND c.group_id = ?
    LIMIT ?
""", [entity_name, group_id, group_id, top_k]).fetchall()
```

### 4. Topic Chunks Retrieval

**Current Cypher:**
```cypher
MATCH (c:EpisodicNode {group_id: $uid})-[:DISCUSSES]->(t:TopicNode {name: $topic_name, group_id: $uid})
OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
RETURN DISTINCT c.uuid as chunk_id, c.content as content, ...
```

**DuckDB SQL Equivalent:**
```sql
SELECT DISTINCT
    c.uuid AS chunk_id,
    c.content,
    c.header_path,
    d.name AS doc_id,
    d.document_date
FROM topics t
JOIN relationships r ON r.to_uuid = t.uuid
    AND r.to_type = 'topic'
    AND r.rel_type = 'DISCUSSES'
JOIN chunks c ON c.uuid = r.from_uuid
LEFT JOIN documents d ON d.uuid = c.document_uuid
WHERE t.name = $topic_name
    AND t.group_id = $group_id
LIMIT $top_k
```

### 5. 1-Hop Neighbor Expansion

**Current Cypher:**
```cypher
MATCH (e:EntityNode {name: $entity_name, group_id: $uid})
      -[r]->(c:EpisodicNode {group_id: $uid})
      -[r2]->(neighbor:EntityNode {group_id: $uid})
WHERE r.fact_id = r2.fact_id AND neighbor.name <> $entity_name
RETURN DISTINCT neighbor.name as name, neighbor.summary as summary,
       count(*) as connection_count
ORDER BY connection_count DESC
LIMIT $max_neighbors
```

**DuckDB SQL Equivalent:**
```sql
SELECT DISTINCT
    neighbor.name,
    neighbor.summary,
    COUNT(*) AS connection_count
FROM entities e
-- First hop: entity -> chunk
JOIN relationships r1 ON r1.from_uuid = e.uuid
    AND r1.from_type = 'entity'
    AND r1.to_type = 'chunk'
    AND r1.fact_id IS NOT NULL
JOIN chunks c ON c.uuid = r1.to_uuid
-- Second hop: chunk -> entity (via same fact_id)
JOIN relationships r2 ON r2.from_uuid = c.uuid
    AND r2.from_type = 'chunk'
    AND r2.to_type = 'entity'
    AND r2.fact_id = r1.fact_id  -- Key join condition!
JOIN entities neighbor ON neighbor.uuid = r2.to_uuid
WHERE e.name = $entity_name
    AND e.group_id = $group_id
    AND neighbor.name <> $entity_name
GROUP BY neighbor.name, neighbor.summary
ORDER BY connection_count DESC
LIMIT $max_neighbors
```

**Explanation**: The `r1.fact_id = r2.fact_id` condition is the key insight. This is exactly how the Cypher query works, we just make the join explicit.

### 6. Entity Facts Retrieval

**Current Cypher:**
```cypher
CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
MATCH (subj:EntityNode {name: $entity_name, group_id: $uid})
      -[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
      -[r2 {fact_id: node.uuid}]->(obj)
WHERE (obj:EntityNode OR obj:TopicNode) AND obj.group_id = $uid
RETURN DISTINCT node.uuid as fact_id, node.content as content, ...
```

**LanceDB + DuckDB Equivalent:**
```python
# Step 1: Vector search for relevant facts
facts_table = db.open_table("facts")
fact_candidates = (
    facts_table
    .search(query_vector)
    .where(f"group_id = '{group_id}'")
    .limit(top_k * 2)  # Get more candidates for filtering
    .to_pandas()
)

# Step 2: Filter to facts involving the entity
fact_uuids = fact_candidates['uuid'].tolist()

result = conn.execute("""
    SELECT DISTINCT
        f.uuid AS fact_id,
        f.content,
        f.subject_name AS subject,
        f.edge_type,
        f.object_name AS object,
        r.to_uuid AS chunk_id
    FROM facts f
    JOIN relationships r ON r.fact_id = f.uuid
        AND r.from_type = 'entity'
        AND r.to_type = 'chunk'
    WHERE f.uuid IN (SELECT unnest(?::varchar[]))
        AND f.group_id = ?
        AND (f.subject_name = ? OR f.object_name = ?)
""", [fact_uuids, group_id, entity_name, entity_name]).fetchall()
```

**Optimization**: Since we denormalized `subject_name` and `object_name` into facts.parquet, we can filter directly without joining to entities.

### 7. Global Chunk Search

**Current Cypher:**
```cypher
CALL db.index.vector.queryNodes('fact_embeddings', $top_k, $vec)
YIELD node, score
WHERE node.group_id = $uid AND score > $threshold
MATCH (subj)-[r1 {fact_id: node.uuid}]->(c:EpisodicNode {group_id: $uid})
WHERE subj.group_id = $uid
OPTIONAL MATCH (d:DocumentNode {group_id: $uid})-[:CONTAINS_CHUNK]->(c)
RETURN DISTINCT c.uuid as chunk_id, c.content as content, max(score) as score
```

**LanceDB + DuckDB Equivalent:**
```python
# Step 1: Vector search for relevant facts
facts_table = db.open_table("facts")
fact_results = (
    facts_table
    .search(query_vector)
    .where(f"group_id = '{group_id}'")
    .limit(top_k * 2)
    .to_pandas()
)

fact_uuids = fact_results['uuid'].tolist()
scores_by_fact = dict(zip(fact_results['uuid'], fact_results['_distance']))

# Step 2: Get chunks for these facts
result = conn.execute("""
    SELECT DISTINCT
        c.uuid AS chunk_id,
        c.content,
        c.header_path,
        d.name AS doc_id,
        d.document_date,
        f.uuid AS fact_uuid
    FROM facts f
    JOIN relationships r ON r.fact_id = f.uuid
        AND r.to_type = 'chunk'
    JOIN chunks c ON c.uuid = r.to_uuid
    LEFT JOIN documents d ON d.uuid = c.document_uuid
    WHERE f.uuid IN (SELECT unnest(?::varchar[]))
        AND f.group_id = ?
""", [fact_uuids, group_id]).fetchall()

# Step 3: Aggregate scores by chunk (max score wins)
chunk_scores = {}
for row in result:
    chunk_id = row['chunk_id']
    score = 1 - scores_by_fact.get(row['fact_uuid'], 1)  # Convert distance to similarity
    if chunk_id not in chunk_scores or score > chunk_scores[chunk_id]:
        chunk_scores[chunk_id] = score
```

---

## Performance Comparison

### Expected Performance Characteristics

| Operation | Neo4j (Current) | DuckDB + LanceDB (Proposed) | Notes |
|-----------|-----------------|----------------------------|-------|
| **Cold Start** | 30-60s (Aura free tier wakeup) | ~100ms (open files) | Massive improvement for interactive use |
| **Vector Search (1k entities)** | ~50ms | ~10ms | LanceDB is optimized for this |
| **Vector Search (100k entities)** | ~100ms | ~30ms | LanceDB uses IVF-PQ indexing |
| **1-Hop Traversal** | ~10ms | ~5ms | DuckDB JOIN on indexed columns |
| **2-Hop Traversal** | ~30ms | ~15ms | Two JOINs, still fast |
| **Batch Write (1000 nodes)** | ~500ms | ~50ms | Parquet append is fast |
| **Memory Usage** | N/A (server) | ~50MB base + data | DuckDB memory-maps files |

### Cold Start Time (Critical for Library Use)

The biggest improvement is eliminating cold start latency:

**Current Flow (Neo4j Aura Free Tier):**
1. User runs `from zomma import KnowledgeGraph` - instant
2. User runs `kg.query("...")` - triggers connection
3. Neo4j Aura instance may be sleeping
4. First query waits 30-60 seconds for wakeup
5. Subsequent queries are fast (~50ms)

**Proposed Flow (Embedded):**
1. User runs `from zomma import KnowledgeGraph` - instant
2. User runs `kg.query("...")` - triggers file opens
3. DuckDB opens Parquet files (~10ms)
4. LanceDB opens vector indices (~50ms)
5. First query executes (~50ms total)
6. Subsequent queries are equally fast

### Vector Search Latency

LanceDB uses modern vector indexing (IVF-PQ) and is optimized for the exact use case:
- Small to medium datasets (thousands to millions of vectors)
- Filtered search (by group_id)
- Python-native integration

Benchmarks on similar workloads show LanceDB performs within 2x of FAISS while providing persistence and filtering that FAISS lacks.

### Join Performance

DuckDB excels at analytical joins on columnar data:
- Columnar storage means reading only needed columns
- Hash joins for equality conditions
- Vectorized execution (SIMD)
- Memory-mapped I/O (no explicit loading)

For the relationships table (~100k rows typical), join operations complete in single-digit milliseconds.

### Batch Write Performance

**Current Flow (Neo4j):**
```python
# Each batch requires:
# 1. Serialize to Cypher parameters
# 2. Network round-trip
# 3. Transaction commit
# Typical: 250 nodes per batch, ~100ms per batch
```

**Proposed Flow (Parquet):**
```python
# Accumulate in memory, then:
# 1. Convert to Arrow table
# 2. Write Parquet file (append or replace)
# Typical: All nodes at once, ~10ms per 1000 rows
```

The Parquet approach also enables atomic writes (write new file, rename to replace old).

---

## Benefits for Library Distribution

### Zero Infrastructure Requirements

**Before:**
```bash
pip install zomma-kg
# Then: Sign up for Neo4j Aura, get connection string, configure .env
# Or: docker run neo4j, configure ports, wait for startup
```

**After:**
```bash
pip install zomma-kg
# Ready to use immediately
```

### Portable Knowledge Bases

A knowledge base becomes a simple directory:
```bash
# Share a knowledge base
zip -r my_kb.zip my_knowledge_base/
# Send to colleague

# Colleague uses it
unzip my_kb.zip
kg = KnowledgeGraph("my_knowledge_base/")
kg.query("What happened in Q3?")
```

No database migrations, connection strings, or server provisioning.

### Dependency Simplicity

**Current dependencies:**
- neo4j (Python driver)
- neo4j-aura or docker/server instance (infrastructure)
- qdrant (additional vector store, currently used for fact search)

**Proposed dependencies:**
- duckdb (single binary, bundled)
- lancedb (pip-installable)
- pyarrow (for Parquet, often already installed)

All dependencies are pip-installable with no native code compilation required on common platforms.

### Debugging and Inspection

With Parquet files, users can inspect their data with familiar tools:
```python
import pandas as pd

# See all entities
entities = pd.read_parquet("my_kb/entities.parquet")
print(entities.head())

# Query without the library
import duckdb
duckdb.sql("SELECT * FROM 'my_kb/chunks.parquet' WHERE content LIKE '%Apple%'")
```

This transparency builds trust and aids debugging.

### Version Control Integration

Parquet files can be versioned (with Git LFS for large files):
```bash
# Track knowledge base versions
git lfs track "*.parquet"
git add my_kb/
git commit -m "Add Q3 2024 earnings data"
```

---

## Migration Strategy

### Phase 1: Parallel Implementation (Non-Breaking)

1. Create new storage backend interface (`StorageBackend` ABC)
2. Implement `Neo4jBackend` wrapping current code
3. Implement `ParquetBackend` with DuckDB + LanceDB
4. Both backends pass the same test suite
5. Feature flag to switch between backends

### Phase 2: Data Migration Tooling

1. Create `migrate_neo4j_to_parquet.py` script
2. Export all nodes and relationships from Neo4j
3. Generate Parquet files in new schema
4. Index vectors in LanceDB
5. Validate data integrity (row counts, sample queries)

### Phase 3: Query Layer Adaptation

1. Update `GraphStore` to use `StorageBackend` interface
2. Replace Cypher queries with SQL/LanceDB calls
3. Maintain identical return types (existing tests should pass)
4. Performance benchmarking to verify improvements

### Phase 4: Default Swap and Neo4j Deprecation

1. Make ParquetBackend the default
2. Document migration path for existing Neo4j users
3. Keep Neo4jBackend available but mark as deprecated
4. Remove Neo4j code after deprecation period (v2.0)

### Migration Script Outline

```python
def migrate_neo4j_to_parquet(neo4j_client, output_dir):
    """Migrate a Neo4j knowledge graph to Parquet format."""

    # 1. Export all nodes
    entities = neo4j_client.query("MATCH (e:EntityNode) RETURN e")
    chunks = neo4j_client.query("MATCH (c:EpisodicNode) RETURN c")
    facts = neo4j_client.query("MATCH (f:FactNode) RETURN f")
    topics = neo4j_client.query("MATCH (t:TopicNode) RETURN t")
    documents = neo4j_client.query("MATCH (d:DocumentNode) RETURN d")

    # 2. Export all relationships
    relationships = neo4j_client.query("""
        MATCH (a)-[r]->(b)
        RETURN
            id(r) as id,
            a.uuid as from_uuid,
            labels(a)[0] as from_type,
            b.uuid as to_uuid,
            labels(b)[0] as to_type,
            type(r) as rel_type,
            r.fact_id as fact_id,
            r.description as description,
            r.date_context as date_context
    """)

    # 3. Convert to DataFrames and write Parquet
    pd.DataFrame(entities).to_parquet(f"{output_dir}/entities.parquet")
    pd.DataFrame(chunks).to_parquet(f"{output_dir}/chunks.parquet")
    # ... etc

    # 4. Index vectors in LanceDB
    db = lancedb.connect(f"{output_dir}/lancedb")

    entity_vectors = [
        {"uuid": e["uuid"], "vector": e["name_embedding"], ...}
        for e in entities if e.get("name_embedding")
    ]
    db.create_table("entities", entity_vectors)
    # ... etc
```

---

## Tradeoffs and What You Lose

### Things That Become Harder

1. **Variable-Length Path Queries**: Neo4j's `MATCH (a)-[*1..5]->(b)` has no direct SQL equivalent. However, the current system never uses this pattern.

2. **Real-Time Multi-User Writes**: Parquet files don't support concurrent writers well. For multi-user scenarios, you'd need:
   - File locking (simple but limiting)
   - Write-ahead log pattern
   - Periodic compaction

   For a library use case (single-user, batch ingestion), this is not a concern.

3. **ACID Transactions Across Operations**: Neo4j provides full ACID. Parquet writes are atomic per-file but not across files. The mitigation is to:
   - Write new files to temp location
   - Rename atomically (POSIX guarantees)
   - This gives consistency but not rollback on partial failure

4. **Graph Visualization Tools**: Neo4j's built-in browser and ecosystem tools (Bloom, etc.) won't work. Alternative: export to NetworkX for visualization.

5. **Cypher Query Language**: Teams familiar with Cypher will need to learn SQL. However, the queries in this system are simple enough that this is a minor learning curve.

### Things That Get Better

1. **Startup Time**: From 30-60 seconds to under 100 milliseconds
2. **Portability**: Knowledge bases become simple directories
3. **Dependency Management**: Pure Python, no server infrastructure
4. **Debugging**: Parquet files are inspectable with standard tools
5. **Cost**: Zero marginal cost vs. Neo4j pricing at scale
6. **Testability**: In-memory DuckDB for unit tests, no test server needed
7. **Deployment**: `pip install` is the entire deployment story

### When to Keep Neo4j

If the system evolves to need:
- Variable-length path traversals (e.g., "find all entities connected within 5 hops")
- Graph algorithms (PageRank, community detection)
- Real-time multi-user collaborative editing
- Integration with Neo4j ecosystem tools

Then Neo4j remains the right choice. The proposed architecture specifically addresses the current query patterns, which are relational in nature.

---

## Appendix: Complete Schema DDL

### DuckDB Schema (for reference, actual storage is Parquet)

```sql
-- These are views over Parquet files, but show the logical schema

CREATE TABLE documents (
    uuid VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    document_date VARCHAR,
    group_id VARCHAR NOT NULL,
    source_path VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
    uuid VARCHAR PRIMARY KEY,
    document_uuid VARCHAR REFERENCES documents(uuid),
    content TEXT NOT NULL,
    header_path VARCHAR,
    group_id VARCHAR NOT NULL,
    document_date VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE entities (
    uuid VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    summary TEXT,
    group_id VARCHAR NOT NULL,
    entity_type VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE topics (
    uuid VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    definition TEXT,
    group_id VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE facts (
    uuid VARCHAR PRIMARY KEY,
    content TEXT NOT NULL,
    group_id VARCHAR NOT NULL,
    subject_uuid VARCHAR REFERENCES entities(uuid),
    subject_name VARCHAR,
    object_uuid VARCHAR,  -- Can reference entities or topics
    object_name VARCHAR,
    object_type VARCHAR,  -- 'entity' or 'topic'
    edge_type VARCHAR NOT NULL,
    date_context VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE relationships (
    id VARCHAR PRIMARY KEY,
    from_uuid VARCHAR NOT NULL,
    from_type VARCHAR NOT NULL,  -- 'entity', 'chunk', 'document', 'topic'
    to_uuid VARCHAR NOT NULL,
    to_type VARCHAR NOT NULL,
    rel_type VARCHAR NOT NULL,
    fact_id VARCHAR REFERENCES facts(uuid),
    description VARCHAR,
    date_context VARCHAR,
    group_id VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes (these would be created on the Parquet files via DuckDB's indexing)
CREATE INDEX idx_relationships_from ON relationships(group_id, from_uuid, rel_type);
CREATE INDEX idx_relationships_to ON relationships(group_id, to_uuid, rel_type);
CREATE INDEX idx_relationships_fact ON relationships(group_id, fact_id);
CREATE INDEX idx_entities_name ON entities(group_id, name);
CREATE INDEX idx_topics_name ON topics(group_id, name);
CREATE INDEX idx_chunks_doc ON chunks(group_id, document_uuid);
```

### LanceDB Table Schemas

```python
# entities.lance
{
    "uuid": str,
    "name": str,
    "summary": str,
    "group_id": str,
    "vector": list[float],  # 3072 dimensions (voyage-finance-2)
}

# topics.lance
{
    "uuid": str,
    "name": str,
    "group_id": str,
    "vector": list[float],  # 3072 dimensions
}

# facts.lance
{
    "uuid": str,
    "content": str,
    "group_id": str,
    "subject_name": str,
    "object_name": str,
    "edge_type": str,
    "vector": list[float],  # 3072 dimensions
}
```

---

## Conclusion

The migration from Neo4j to a Parquet-based storage architecture is well-suited to ZommaLabsKG's actual usage patterns and strategic goal of becoming a pip-installable library. The current system uses Neo4j as an overqualified key-value store with vector search, paying the infrastructure tax of a distributed graph database without utilizing its advanced features.

By migrating to DuckDB + LanceDB with Parquet storage, the project gains:
- Zero infrastructure requirements
- Sub-second cold start times
- Portable, inspectable knowledge bases
- Simplified dependency management
- Identical query capabilities for the patterns actually in use

The tradeoffs (loss of ACID transactions, Cypher syntax, graph algorithms) are acceptable given that these features are not used by the current system. Should requirements evolve to need true graph traversals, the architecture can be extended or the migration reconsidered.
