# Relationship Model Refactor Plan

## Overview

Simplify the graph structure from a chunk-centric model (2 edges per fact) to a direct entity-entity model (1 edge per fact) with chunk reference as an edge property.

## Current vs Proposed Structure

### Current Structure
```
Subject Entity -[MENTIONED_IN]-> Chunk -[relationship_type]-> Object Entity
Chunk -[DISCUSSES]-> Topic
```

**Example:**
```
Fed -[MENTIONED_IN]-> Chunk123 -[REPORTED_INCREASE_IN]-> Inflation
Chunk123 -[DISCUSSES]-> Tariffs
```

- 2 edges per fact + 1 edge per topic-chunk mapping
- Generic `MENTIONED_IN` loses semantic meaning
- `DISCUSSES` is separate from fact relationships

### Proposed Structure
```
Entity -[relationship_type, chunk_uuid]-> Entity
Entity -[relationship_type, chunk_uuid]-> Topic
```

**Example:**
```
Fed -[REPORTED_INCREASE_IN, chunk_uuid=Chunk123]-> Inflation
Businesses -[EXPRESSED_CONCERN_ABOUT, chunk_uuid=Chunk123]-> Tariffs
```

- 1 edge per fact
- Topics are just entities (or treated similarly)
- Chunk provenance preserved as edge property

---

## Schema Changes

### 1. Relationships Parquet Schema

**Current:**
```python
("id", pa.string()),
("from_uuid", pa.string()),
("from_type", pa.string()),      # "entity", "chunk"
("to_uuid", pa.string()),
("to_type", pa.string()),        # "entity", "chunk", "topic"
("rel_type", pa.string()),       # "MENTIONED_IN", "DISCUSSES", or fact relationship
("fact_id", pa.string()),
("description", pa.string()),
("date_context", pa.string()),
("group_id", pa.string()),
("created_at", pa.string()),
```

**Proposed:**
```python
("id", pa.string()),
("from_uuid", pa.string()),      # Always entity UUID
("from_type", pa.string()),      # "entity"
("to_uuid", pa.string()),        # Entity or Topic UUID
("to_type", pa.string()),        # "entity" or "topic"
("rel_type", pa.string()),       # Semantic relationship from fact
("chunk_uuid", pa.string()),     # NEW: Reference to source chunk
("fact_id", pa.string()),        # Keep for fact lookup
("description", pa.string()),    # Fact content
("date_context", pa.string()),
("group_id", pa.string()),
("created_at", pa.string()),
```

**Changes:**
- Add `chunk_uuid` field
- Remove need for `from_type = "chunk"` (always entity now)
- `to_type` is either "entity" or "topic"
- No more `MENTIONED_IN` or `DISCUSSES` relationship types

---

## Files to Modify

### Phase 1: Schema & Storage Layer

#### 1.1 `zomma_kg/storage/parquet/backend.py`
- [ ] Update `RELATIONSHIP_SCHEMA` to add `chunk_uuid` field
- [ ] Update `write_relationships()` to handle new schema

#### 1.2 `zomma_kg/storage/README.md`
- [ ] Update relationship schema documentation
- [ ] Update query examples

### Phase 2: Assembly Layer

#### 2.1 `zomma_kg/ingestion/assembly/assembler.py`
- [ ] Rewrite `_build_relationships()` to create single edges:
  ```python
  def _build_relationships(self, input: AssemblyInput) -> list[dict]:
      relationships = []

      for fact in input.facts:
          # Single edge: Subject -> Object with chunk reference
          relationships.append({
              "id": str(uuid4()),
              "from_uuid": fact.subject_uuid,
              "from_type": "entity",
              "to_uuid": fact.object_uuid,
              "to_type": self._get_object_type(fact),  # "entity" or "topic"
              "rel_type": fact.relationship_type,
              "chunk_uuid": fact.chunk_uuid,  # NEW
              "fact_id": fact.uuid,
              "description": fact.content,
              "date_context": fact.date_context or "",
          })

      return relationships
  ```
- [ ] Remove topic-specific DISCUSSES relationship creation
- [ ] Add helper to determine if object is entity or topic

#### 2.2 `zomma_kg/types/results.py`
- [ ] Remove `topic_chunk_mappings` from `AssemblyInput` (no longer needed)

#### 2.3 `scripts/build_kg.py`
- [ ] Remove `topic_name_to_chunk_indices` tracking
- [ ] Remove `topic_chunk_mappings` construction
- [ ] Facts with topic objects should have `object_type = "topic"`

### Phase 3: Query Layer

#### 3.1 `zomma_kg/storage/duckdb/queries.py`

**Update `get_entity_chunks()`:**
```python
async def get_entity_chunks(self, entity_name: str, limit: int = 50):
    """Get chunks where entity appears (as subject or object)."""
    # Query relationships where entity is from_uuid or to_uuid
    # Join with chunks table using chunk_uuid
    results = conn.execute("""
        SELECT DISTINCT
            c.uuid AS chunk_id,
            c.content,
            c.header_path,
            c.document_uuid,
            d.name AS doc_name,
            r.rel_type,
            r.description AS fact_content
        FROM entities e
        JOIN relationships r ON (r.from_uuid = e.uuid OR r.to_uuid = e.uuid)
            AND r.group_id = ?
        JOIN chunks c ON c.uuid = r.chunk_uuid AND c.group_id = ?
        LEFT JOIN documents d ON d.uuid = c.document_uuid
        WHERE LOWER(e.name) = LOWER(?) AND e.group_id = ?
        LIMIT ?
    """, [group_id, group_id, entity_name, group_id, limit])
```

**Update `get_topic_chunks()`:**
```python
async def get_topic_chunks(self, topic_name: str, limit: int = 50):
    """Get chunks that mention a topic (topic is the object of a relationship)."""
    results = conn.execute("""
        SELECT DISTINCT
            c.uuid AS chunk_id,
            c.content,
            c.header_path,
            r.rel_type,
            r.description AS fact_content
        FROM topics t
        JOIN relationships r ON r.to_uuid = t.uuid
            AND r.to_type = 'topic'
            AND r.group_id = ?
        JOIN chunks c ON c.uuid = r.chunk_uuid AND c.group_id = ?
        WHERE LOWER(t.name) = LOWER(?) AND t.group_id = ?
        LIMIT ?
    """, [group_id, group_id, topic_name, group_id, limit])
```

**Update `get_entity_neighbors()`:**
```python
async def get_entity_neighbors(self, entity_name: str, limit: int = 20):
    """Get entities directly connected to this entity."""
    # Simpler now - just 1-hop query
    results = conn.execute("""
        SELECT DISTINCT
            neighbor.uuid,
            neighbor.name,
            neighbor.summary,
            r.rel_type,
            r.description
        FROM entities e
        JOIN relationships r ON r.from_uuid = e.uuid AND r.group_id = ?
        JOIN entities neighbor ON neighbor.uuid = r.to_uuid
            AND neighbor.group_id = ?
            AND r.to_type = 'entity'
        WHERE LOWER(e.name) = LOWER(?) AND e.group_id = ?
        LIMIT ?
    """, [group_id, group_id, entity_name, group_id, limit])
```

#### 3.2 `zomma_kg/storage/base.py`
- [ ] Update docstrings for query methods to reflect new structure

### Phase 4: Testing & Validation

#### 4.1 Update Tests
- [ ] `tests/test_storage.py` - Update relationship tests
- [ ] `tests/test_assembly.py` - Update assembly tests
- [ ] `tests/test_query_pipeline.py` - Ensure queries still work

#### 4.2 Integration Test
- [ ] Run `scripts/build_kg.py` with new structure
- [ ] Verify relationship counts (should be ~half of before)
- [ ] Run query pipeline and verify answers

---

## Migration Notes

### Breaking Changes
- Existing KBs will need to be rebuilt (relationship schema changed)
- No migration path for existing data (clean rebuild required)

### Backwards Compatibility
- None needed for now (development phase)

---

## Verification Checklist

After refactor, verify:
- [ ] `build_kg.py` creates correct number of relationships (1 per fact)
- [ ] No `MENTIONED_IN` or `DISCUSSES` relationship types exist
- [ ] All relationships have `chunk_uuid` populated
- [ ] `get_entity_chunks()` returns correct chunks
- [ ] `get_topic_chunks()` returns correct chunks
- [ ] `get_entity_neighbors()` returns connected entities
- [ ] Query pipeline returns meaningful answers
- [ ] Facts table still works independently

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Edges per fact | 2 | 1 |
| Relationship types | MENTIONED_IN, DISCUSSES, + semantic | Semantic only |
| Query complexity | 2-hop for entity relations | 1-hop |
| Storage size | ~2x relationships | ~1x relationships |

---

## Open Questions

1. **Topics as entities?** Should topics be stored in the entities table with `entity_type = "topic"` instead of a separate topics table? This would further simplify the model.

2. **Bidirectional edges?** Currently edges are directional (subject -> object). Do we need reverse lookups? The current query handles this with `OR r.to_uuid = e.uuid`.

3. **Multiple facts same entities?** If two facts relate the same entity pair with different relationship types, we'll have multiple edges. This is correct behavior but worth noting.
