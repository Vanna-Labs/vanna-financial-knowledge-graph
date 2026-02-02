# Relationship Model Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify the graph from chunk-centric (2 edges per fact) to direct entity-entity edges (1 edge per fact) with chunk_uuid as an edge property.

**Architecture:** Replace the current `Subject → Chunk → Object` pattern with direct `Subject → Object` edges. Store `chunk_uuid` as a relationship property for provenance. Remove `MENTIONED_IN` and `DISCUSSES` relationship types — all relationships are now semantic (from facts).

**Tech Stack:** Python, Parquet (PyArrow), DuckDB, pytest

---

## Task 1: Update Relationship Schema

**Files:**
- Modify: `zomma_kg/storage/parquet/backend.py:196-209`

**Step 1: Add chunk_uuid field to relationship schema**

In `_relationship_schema()`, add the `chunk_uuid` field after `rel_type`:

```python
@staticmethod
def _relationship_schema() -> pa.Schema:
    return pa.schema([
        ("id", pa.string()),
        ("from_uuid", pa.string()),
        ("from_type", pa.string()),
        ("to_uuid", pa.string()),
        ("to_type", pa.string()),
        ("rel_type", pa.string()),
        ("chunk_uuid", pa.string()),  # NEW: Source chunk reference
        ("fact_id", pa.string()),
        ("description", pa.string()),
        ("date_context", pa.string()),
        ("group_id", pa.string()),
        ("created_at", pa.string()),
    ])
```

**Step 2: Commit**

```bash
git add zomma_kg/storage/parquet/backend.py
git commit -m "feat(storage): add chunk_uuid field to relationship schema"
```

---

## Task 2: Remove topic_chunk_mappings from AssemblyInput

**Files:**
- Modify: `zomma_kg/types/results.py:417-436`

**Step 1: Remove the topic_chunk_mappings field**

The new model doesn't need separate topic-chunk tracking — topics are linked via facts like entities.

```python
class AssemblyInput(BaseModel):
    """
    Input for the Assembler.

    Contains all resolved data ready to be written to storage.
    """

    document: Any = Field(..., description="Document to write")
    chunks: list[Any] = Field(default_factory=list, description="Chunks to write")
    entities: list[CanonicalEntity] = Field(default_factory=list, description="Resolved entities")
    facts: list[Any] = Field(default_factory=list, description="Facts to write")
    topics: list[Any] = Field(default_factory=list, description="Topics to write")

    class Config:
        arbitrary_types_allowed = True
```

**Step 2: Commit**

```bash
git add zomma_kg/types/results.py
git commit -m "refactor(types): remove topic_chunk_mappings from AssemblyInput"
```

---

## Task 3: Rewrite _build_relationships in Assembler

**Files:**
- Modify: `zomma_kg/ingestion/assembly/assembler.py:189-254`

**Step 1: Write the new _build_relationships method**

Replace the entire method with the new direct-edge model:

```python
def _build_relationships(self, input: AssemblyInput) -> list[dict]:
    """
    Build relationships as direct entity-entity edges.

    For each fact:
        Subject Entity -> Object Entity/Topic (relationship_type from fact)
        with chunk_uuid as edge property for provenance
    """
    relationships = []

    for fact in input.facts:
        if not fact.chunk_uuid:
            continue

        # Determine object type: "entity" or "topic"
        object_type = "topic" if fact.object_type == "topic" else "entity"

        # Single direct edge: Subject -> Object
        relationships.append(
            {
                "id": str(uuid4()),
                "from_uuid": fact.subject_uuid,
                "from_type": "entity",
                "to_uuid": fact.object_uuid,
                "to_type": object_type,
                "rel_type": fact.relationship_type,
                "chunk_uuid": fact.chunk_uuid,
                "fact_id": fact.uuid,
                "description": fact.content,
                "date_context": fact.date_context or "",
            }
        )

    return relationships
```

**Step 2: Update the module docstring at the top of the file**

Change line 5-6 from:
```python
"""
Knowledge Base Assembler

Final phase that writes resolved data to storage in efficient batches.

Write order (due to foreign key relationships):
    1. Document - No dependencies
    2. Chunks - Reference document UUID
    3. Entities - No dependencies (UUIDs already resolved)
    4. Facts - Reference entity UUIDs and chunk UUID
    5. Topics - No dependencies
    6. Relationships - Reference all of the above
"""
```

To:
```python
"""
Knowledge Base Assembler

Final phase that writes resolved data to storage in efficient batches.

Write order (due to foreign key relationships):
    1. Document - No dependencies
    2. Chunks - Reference document UUID
    3. Entities - No dependencies (UUIDs already resolved)
    4. Facts - Reference entity UUIDs and chunk UUID
    5. Topics - No dependencies
    6. Relationships - Direct entity-entity edges with chunk_uuid for provenance
"""
```

**Step 3: Commit**

```bash
git add zomma_kg/ingestion/assembly/assembler.py
git commit -m "refactor(assembler): simplify to direct entity-entity edges"
```

---

## Task 4: Update build_kg.py to Remove topic_chunk_mappings

**Files:**
- Modify: `scripts/build_kg.py:155-165, 335-353`

**Step 1: Remove topic_name_to_chunk_indices tracking**

Delete lines 155 and the topic tracking logic in the loop (lines 161-165):

Before (lines 151-169):
```python
# Flatten entities and facts, tracking which chunk each topic came from
all_entities = []
all_facts = []
chunk_fact_map = {}  # Track which facts came from which chunk
topic_name_to_chunk_indices: dict[str, list[int]] = {}  # Track topic name -> chunk indices

for i, result in enumerate(extraction_results):
    for entity in result.entities:
        all_entities.append(entity)
        # Track which chunk this topic came from
        if entity.entity_type.lower() == "topic":
            key = entity.name.lower().strip()
            if key not in topic_name_to_chunk_indices:
                topic_name_to_chunk_indices[key] = []
            topic_name_to_chunk_indices[key].append(i)

    for fact in result.facts:
        all_facts.append(fact)
        chunk_fact_map[id(fact)] = i  # Map fact to chunk index
```

After:
```python
# Flatten entities and facts
all_entities = []
all_facts = []
chunk_fact_map = {}  # Track which facts came from which chunk

for i, result in enumerate(extraction_results):
    for entity in result.entities:
        all_entities.append(entity)

    for fact in result.facts:
        all_facts.append(fact)
        chunk_fact_map[id(fact)] = i  # Map fact to chunk index
```

**Step 2: Remove topic_chunk_mappings construction**

Delete lines 335-343:
```python
# Build topic-chunk mappings (topic UUID -> chunk UUIDs)
topic_chunk_mappings: dict[str, list[str]] = {}
for topic in resolved_topics:
    topic_key = topic.name.lower().strip()
    chunk_indices = topic_name_to_chunk_indices.get(topic_key, [])
    if chunk_indices:
        topic_chunk_mappings[topic.uuid] = [
            chunk_uuids[idx] for idx in chunk_indices if idx < len(chunk_uuids)
        ]
```

**Step 3: Update AssemblyInput creation**

Change lines 345-353 from:
```python
# Build assembly input
assembly_input = AssemblyInput(
    document=doc,
    chunks=chunks,
    entities=resolution_result.new_entities,
    facts=facts,
    topics=resolved_topics,
    topic_chunk_mappings=topic_chunk_mappings,
)
```

To:
```python
# Build assembly input
assembly_input = AssemblyInput(
    document=doc,
    chunks=chunks,
    entities=resolution_result.new_entities,
    facts=facts,
    topics=resolved_topics,
)
```

**Step 4: Commit**

```bash
git add scripts/build_kg.py
git commit -m "refactor(build_kg): remove topic_chunk_mappings tracking"
```

---

## Task 5: Update get_entity_chunks Query

**Files:**
- Modify: `zomma_kg/storage/duckdb/queries.py:316-360`

**Step 1: Rewrite the query for direct edges**

Replace the method with:

```python
async def get_entity_chunks(
    self,
    entity_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Get chunks where an entity appears (as subject or object).

    Uses chunk_uuid edge property for 1-hop traversal.
    """
    def _query() -> list[dict[str, Any]]:
        self._refresh_view("entities")
        self._refresh_view("chunks")
        self._refresh_view("relationships")
        self._refresh_view("documents")
        conn = self._get_conn()

        try:
            results = conn.execute("""
                SELECT DISTINCT
                    c.uuid AS chunk_id,
                    c.content,
                    c.header_path,
                    c.document_uuid,
                    d.name AS doc_name,
                    d.document_date
                FROM entities e
                JOIN relationships r ON (r.from_uuid = e.uuid OR r.to_uuid = e.uuid)
                    AND r.group_id = ?
                JOIN chunks c ON c.uuid = r.chunk_uuid AND c.group_id = ?
                LEFT JOIN documents d ON d.uuid = c.document_uuid AND d.group_id = ?
                WHERE LOWER(e.name) = LOWER(?) AND e.group_id = ?
                LIMIT ?
            """, [self.group_id, self.group_id, self.group_id,
                  entity_name, self.group_id, limit]).fetchall()
        except duckdb.CatalogException:
            return []

        col_names = [desc[0] for desc in conn.description]
        return [dict(zip(col_names, row)) for row in results]

    return await asyncio.to_thread(_query)
```

**Step 2: Commit**

```bash
git add zomma_kg/storage/duckdb/queries.py
git commit -m "refactor(queries): update get_entity_chunks for direct edges"
```

---

## Task 6: Update get_entity_neighbors Query

**Files:**
- Modify: `zomma_kg/storage/duckdb/queries.py:362-413`

**Step 1: Simplify to 1-hop traversal**

Replace the method with:

```python
async def get_entity_neighbors(
    self,
    entity_name: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Get neighboring entities directly connected to this entity.

    1-hop traversal via direct entity-entity edges.
    """
    def _query() -> list[dict[str, Any]]:
        self._refresh_view("entities")
        self._refresh_view("relationships")
        conn = self._get_conn()

        try:
            results = conn.execute("""
                SELECT
                    neighbor.name,
                    neighbor.summary,
                    neighbor.entity_type,
                    COUNT(*) AS connection_count
                FROM entities e
                JOIN relationships r ON r.from_uuid = e.uuid
                    AND r.to_type = 'entity'
                    AND r.group_id = ?
                JOIN entities neighbor ON neighbor.uuid = r.to_uuid
                    AND neighbor.group_id = ?
                WHERE LOWER(e.name) = LOWER(?)
                    AND e.group_id = ?
                    AND LOWER(neighbor.name) != LOWER(?)
                GROUP BY neighbor.name, neighbor.summary, neighbor.entity_type
                ORDER BY connection_count DESC
                LIMIT ?
            """, [self.group_id, self.group_id,
                  entity_name, self.group_id, entity_name, limit]).fetchall()
        except duckdb.CatalogException:
            return []

        col_names = [desc[0] for desc in conn.description]
        return [dict(zip(col_names, row)) for row in results]

    return await asyncio.to_thread(_query)
```

**Step 2: Commit**

```bash
git add zomma_kg/storage/duckdb/queries.py
git commit -m "refactor(queries): simplify get_entity_neighbors to 1-hop"
```

---

## Task 7: Update get_topic_chunks Query

**Files:**
- Modify: `zomma_kg/storage/duckdb/queries.py:415-454`

**Step 1: Update query for new edge model**

Replace the method with:

```python
async def get_topic_chunks(
    self,
    topic_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Get chunks that mention a topic (topic is the object of a relationship).

    Uses chunk_uuid edge property for lookup.
    """
    def _query() -> list[dict[str, Any]]:
        self._refresh_view("topics")
        self._refresh_view("chunks")
        self._refresh_view("relationships")
        self._refresh_view("documents")
        conn = self._get_conn()

        try:
            results = conn.execute("""
                SELECT DISTINCT
                    c.uuid AS chunk_id,
                    c.content,
                    c.header_path,
                    c.document_uuid,
                    d.name AS doc_name,
                    d.document_date
                FROM topics t
                JOIN relationships r ON r.to_uuid = t.uuid
                    AND r.to_type = 'topic'
                    AND r.group_id = ?
                JOIN chunks c ON c.uuid = r.chunk_uuid AND c.group_id = ?
                LEFT JOIN documents d ON d.uuid = c.document_uuid AND d.group_id = ?
                WHERE LOWER(t.name) = LOWER(?) AND t.group_id = ?
                LIMIT ?
            """, [self.group_id, self.group_id, self.group_id,
                  topic_name, self.group_id, limit]).fetchall()
        except duckdb.CatalogException:
            return []

        col_names = [desc[0] for desc in conn.description]
        return [dict(zip(col_names, row)) for row in results]

    return await asyncio.to_thread(_query)
```

**Step 2: Commit**

```bash
git add zomma_kg/storage/duckdb/queries.py
git commit -m "refactor(queries): update get_topic_chunks for direct edges"
```

---

## Task 8: Update Relationship Tests

**Files:**
- Modify: `tests/test_assembler.py:134-289`

**Step 1: Rewrite test_build_relationships_basic**

The test should now expect 1 relationship per fact, not 2:

```python
def test_build_relationships_basic(self):
    """Relationships are direct entity-entity edges with chunk_uuid."""
    assembler = Assembler.__new__(Assembler)

    doc_uuid = str(uuid4())
    chunk_uuid = str(uuid4())
    subject_uuid = str(uuid4())
    object_uuid = str(uuid4())
    fact_uuid = str(uuid4())

    doc = Document(uuid=doc_uuid, name="test.pdf", file_type="pdf")
    fact = Fact(
        uuid=fact_uuid,
        content="Apple acquired Beats.",
        subject_uuid=subject_uuid,
        subject_name="Apple",
        object_uuid=object_uuid,
        object_name="Beats",
        object_type="entity",
        relationship_type="ACQUIRED",
        date_context="2014",
        chunk_uuid=chunk_uuid,
    )

    input = AssemblyInput(
        document=doc,
        chunks=[],
        entities=[],
        facts=[fact],
        topics=[],
    )

    relationships = assembler._build_relationships(input)

    # Now only 1 relationship per fact (direct edge)
    assert len(relationships) == 1

    # Direct edge: Subject -> Object with chunk_uuid
    rel = relationships[0]
    assert rel["from_uuid"] == subject_uuid
    assert rel["from_type"] == "entity"
    assert rel["to_uuid"] == object_uuid
    assert rel["to_type"] == "entity"
    assert rel["rel_type"] == "ACQUIRED"
    assert rel["chunk_uuid"] == chunk_uuid
    assert rel["fact_id"] == fact_uuid
    assert rel["description"] == "Apple acquired Beats."
```

**Step 2: Update test_build_relationships_preserves_date_context**

```python
def test_build_relationships_preserves_date_context(self):
    """Date context is preserved in relationships."""
    assembler = Assembler.__new__(Assembler)

    doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
    fact = Fact(
        uuid=str(uuid4()),
        content="Revenue in Q4.",
        subject_uuid=str(uuid4()),
        subject_name="Company",
        object_uuid=str(uuid4()),
        object_name="Revenue",
        object_type="entity",
        relationship_type="REPORTED",
        chunk_uuid=str(uuid4()),
        date_context="Q4 2025",
    )

    input = AssemblyInput(
        document=doc,
        chunks=[],
        entities=[],
        facts=[fact],
        topics=[],
    )

    relationships = assembler._build_relationships(input)

    assert len(relationships) == 1
    assert relationships[0]["date_context"] == "Q4 2025"
```

**Step 3: Update test_build_relationships_includes_id**

```python
def test_build_relationships_includes_id(self):
    """Each relationship should have a unique id field."""
    assembler = Assembler.__new__(Assembler)

    doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
    fact1 = Fact(
        uuid=str(uuid4()),
        content="Test fact 1.",
        subject_uuid=str(uuid4()),
        subject_name="Subject1",
        object_uuid=str(uuid4()),
        object_name="Object1",
        object_type="entity",
        relationship_type="RELATED_TO",
        chunk_uuid=str(uuid4()),
        date_context="2025",
    )
    fact2 = Fact(
        uuid=str(uuid4()),
        content="Test fact 2.",
        subject_uuid=str(uuid4()),
        subject_name="Subject2",
        object_uuid=str(uuid4()),
        object_name="Object2",
        object_type="entity",
        relationship_type="ACQUIRED",
        chunk_uuid=str(uuid4()),
        date_context="2025",
    )

    input = AssemblyInput(
        document=doc,
        chunks=[],
        entities=[],
        facts=[fact1, fact2],
        topics=[],
    )

    relationships = assembler._build_relationships(input)

    # 2 facts = 2 relationships
    assert len(relationships) == 2

    # Both relationships should have id field
    assert "id" in relationships[0]
    assert "id" in relationships[1]

    # IDs should be unique
    assert relationships[0]["id"] != relationships[1]["id"]

    # IDs should be valid UUIDs (36 chars with hyphens)
    assert len(relationships[0]["id"]) == 36
    assert len(relationships[1]["id"]) == 36
```

**Step 4: Add test for topic object type**

Add a new test after `test_build_relationships_includes_id`:

```python
def test_build_relationships_topic_object(self):
    """Facts with object_type='topic' create edges to topics."""
    assembler = Assembler.__new__(Assembler)

    doc = Document(uuid=str(uuid4()), name="test.pdf", file_type="pdf")
    chunk_uuid = str(uuid4())
    topic_uuid = str(uuid4())

    fact = Fact(
        uuid=str(uuid4()),
        content="Company discussed tariffs.",
        subject_uuid=str(uuid4()),
        subject_name="Company",
        object_uuid=topic_uuid,
        object_name="Tariffs",
        object_type="topic",
        relationship_type="DISCUSSED",
        chunk_uuid=chunk_uuid,
        date_context="2025",
    )

    input = AssemblyInput(
        document=doc,
        chunks=[],
        entities=[],
        facts=[fact],
        topics=[],
    )

    relationships = assembler._build_relationships(input)

    assert len(relationships) == 1
    assert relationships[0]["to_type"] == "topic"
    assert relationships[0]["to_uuid"] == topic_uuid
    assert relationships[0]["chunk_uuid"] == chunk_uuid
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_assembler.py::TestBuildRelationships -v
```

Expected: All 5 tests pass

**Step 6: Commit**

```bash
git add tests/test_assembler.py
git commit -m "test(assembler): update tests for direct entity-entity edges"
```

---

## Task 9: Update Integration Test Assertions

**Files:**
- Modify: `tests/test_assembler.py:340-406` (in TestAssemblerIntegration)

**Step 1: Find and update relationship count assertions**

Search for `relationships_written == 2` (or similar) and update to expect half the relationships.

The test at approximately line 399 asserts `relationships_written == 2` for one fact — change to `== 1`.

Also update any comment that says "Subject->Chunk, Chunk->Object" to "Subject->Object".

**Step 2: Run full test suite**

```bash
pytest tests/test_assembler.py -v
```

Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_assembler.py
git commit -m "test(assembler): update integration test for new relationship count"
```

---

## Task 10: Final Verification

**Files:**
- None (verification only)

**Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: (Optional) Run build_kg.py on a test document**

If you have test data available:

```bash
python scripts/build_kg.py --input test_data/sample.pdf --output test_kb
```

Verify:
- Relationship count is roughly half of what it was before
- No `MENTIONED_IN` or `DISCUSSES` relationship types in output
- All relationships have `chunk_uuid` populated

**Step 3: Create summary commit**

```bash
git log --oneline -10
```

Review that all commits are present and logical.

---

## Verification Checklist

After completing all tasks, verify:

- [ ] Schema has `chunk_uuid` field
- [ ] No `topic_chunk_mappings` in `AssemblyInput`
- [ ] `_build_relationships` creates 1 edge per fact
- [ ] No `MENTIONED_IN` or `DISCUSSES` relationship types
- [ ] All relationships have `chunk_uuid` populated
- [ ] `get_entity_chunks()` uses `chunk_uuid` join
- [ ] `get_entity_neighbors()` is 1-hop (not 2-hop)
- [ ] `get_topic_chunks()` uses `chunk_uuid` join
- [ ] All tests pass
