# Bidirectional `-to` Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `-to` flag to search command for bidirectional N-to-N entity/topic queries.

**Architecture:** Replace `-topic` with `-to` flag. When `-to` is provided, find all edges between the `-entity` group and `-to` group (bidirectional). When only `-entity` is provided, find edges where entity appears as subject OR object (current behavior).

**Tech Stack:** Python, DuckDB, argparse

---

### Task 1: Update DuckDB Query Method

**Files:**
- Modify: `zomma_kg/storage/duckdb/queries.py:366-449`

**Step 1: Rename and update the query method**

Replace `get_facts_for_entities_and_topics` with `get_facts_by_entities`:

```python
async def get_facts_by_entities(
    self,
    entity_names: list[str],
    to_names: list[str] | None = None,
    limit: int = 100,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[Fact]:
    """
    Get facts involving specified entities.

    - If only entity_names: facts where entity is subject OR object
    - If entity_names + to_names: facts where edge connects the two groups (bidirectional)

    Bidirectional means: (subject IN entities AND object IN to) OR (subject IN to AND object IN entities)
    """
    if not entity_names:
        return []

    def _query() -> list[Fact]:
        self._refresh_view("facts")
        self._refresh_view("chunks")
        conn = self._get_conn()

        conditions = ["f.group_id = ?"]
        params: list = []

        entity_lower = [n.lower() for n in entity_names]
        entity_ph = ", ".join(["?" for _ in entity_lower])

        if to_names:
            # Bidirectional: edges between the two groups
            to_lower = [n.lower() for n in to_names]
            to_ph = ", ".join(["?" for _ in to_lower])

            conditions.append(
                f"((LOWER(f.subject_name) IN ({entity_ph}) AND LOWER(f.object_name) IN ({to_ph})) "
                f"OR (LOWER(f.subject_name) IN ({to_ph}) AND LOWER(f.object_name) IN ({entity_ph})))"
            )
            params.extend(entity_lower)
            params.extend(to_lower)
            params.extend(to_lower)
            params.extend(entity_lower)
        else:
            # No -to: entity as subject OR object
            conditions.append(
                f"(LOWER(f.subject_name) IN ({entity_ph}) "
                f"OR LOWER(f.object_name) IN ({entity_ph}))"
            )
            params.extend(entity_lower)
            params.extend(entity_lower)

        # Date filtering via chunk's document_date
        if from_date:
            conditions.append("c.document_date >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("c.document_date <= ?")
            params.append(to_date)

        params.append(self.group_id)

        where_clause = " AND ".join(conditions)

        try:
            results = conn.execute(f"""
                SELECT f.* FROM facts f
                LEFT JOIN chunks c ON f.chunk_uuid = c.uuid AND c.group_id = ?
                WHERE {where_clause}
                ORDER BY c.document_date DESC NULLS LAST
                LIMIT ?
            """, [self.group_id, *params, limit]).fetchall()
        except duckdb.CatalogException:
            return []

        return [self._row_to_fact(row, conn) for row in results]

    return await asyncio.to_thread(_query)
```

**Step 2: Verify no syntax errors**

Run: `python -c "from zomma_kg.storage.duckdb.queries import DuckDBQueries; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add zomma_kg/storage/duckdb/queries.py
git commit -m "feat: add bidirectional get_facts_by_entities query"
```

---

### Task 2: Update Server Search Command

**Files:**
- Modify: `zomma_kg/mcp/server.py:169-290`

**Step 1: Update argparse to replace `-topic` with `-to`**

Change the argument parser in `cmd_search`:

```python
async def cmd_search(args: list[str]) -> str:
    """
    Find connections between nodes.

    - Only -entity: edges where entity appears (subject OR object)
    - With -to: edges between -entity group and -to group (bidirectional)

    Usage:
        search -entity "Apple Inc"
        search -entity "Apple Inc" -to "BRK-B, Revenue"
        search -entity "Fed Boston, Fed Atlanta" -to "Manufacturing, Employment"
    """
    kg = await get_kg()
    session = get_session()
    session.clear_search()

    parser = argparse.ArgumentParser(prog="search", add_help=False)
    parser.add_argument("-entity", "--entity", type=str, default="")
    parser.add_argument("-to", "--to", dest="to_targets", type=str, default="")
    parser.add_argument("--query", "-q", type=str, default="")
    parser.add_argument("--from", dest="from_date", type=str, default="")
    parser.add_argument("--to-date", dest="to_date", type=str, default="")
    parser.add_argument("--limit", type=int, default=20)

    try:
        parsed, _ = parser.parse_known_args(args)
    except SystemExit:
        return "Usage: search -entity <names> [-to <names>] [--query \"text\"] [--from DATE] [--to-date DATE]"

    entity_names = parse_csv_list(parsed.entity) if parsed.entity else []
    to_names = parse_csv_list(parsed.to_targets) if parsed.to_targets else []

    if not entity_names:
        return "Usage: search requires -entity"

    assert kg._storage is not None

    # Stage 1: Get facts from DuckDB
    fetch_limit = parsed.limit * 3 if parsed.query else parsed.limit

    facts = await kg._storage._duckdb.get_facts_by_entities(
        entity_names,
        to_names=to_names if to_names else None,
        limit=fetch_limit,
        from_date=parsed.from_date or None,
        to_date=parsed.to_date or None,
    )

    # Stage 2: If semantic query provided, rank facts by similarity
    fact_scores: dict[str, float] = {}
    if parsed.query and facts:
        query_vector = await kg._embeddings.embed_single(parsed.query)
        fact_uuids = [f.uuid for f in facts]
        ranked_results = await kg._storage._lancedb.search_facts_by_uuids(
            query_vector,
            fact_uuids,
            limit=parsed.limit,
            threshold=0.0,
        )
        fact_scores = {uuid: score for uuid, score in ranked_results}
        ranked_uuids = [uuid for uuid, _ in ranked_results]
        uuid_to_fact = {f.uuid: f for f in facts}
        facts = [uuid_to_fact[uuid] for uuid in ranked_uuids if uuid in uuid_to_fact]

    if not facts:
        info = f"entities: {', '.join(entity_names)}"
        if to_names:
            info += f" → {', '.join(to_names)}"
        return f"No connections found for {info}"

    # Format output
    output_lines = []
    header_parts = []
    if entity_names:
        header_parts.append(f"entities: {', '.join(entity_names)}")
    if to_names:
        header_parts.append(f"to: {', '.join(to_names)}")
    if parsed.query:
        header_parts.append(f'query: "{parsed.query}"')
    if header_parts:
        output_lines.append(f"Searching {' + '.join(header_parts)}")
        output_lines.append("")

    for i, fact in enumerate(facts[:parsed.limit], 1):
        date_str = fact.date_context or "Unknown"
        doc_name = "Document"
        score = fact_scores.get(fact.uuid, 0.0)

        result = SearchResult(
            index=i,
            date=date_str,
            from_entity=fact.subject_name,
            relationship=fact.relationship_type,
            to_entity=fact.object_name,
            document=doc_name,
            score=score,
            fact_uuid=fact.uuid,
            chunk_uuid=fact.chunk_uuid or "",
        )
        session.search_results.append(result)

        output_lines.append(
            f"[{i}] {date_str}  {fact.subject_name} ──[{fact.relationship_type}]──▶ {fact.object_name}"
        )
        score_str = f" | score: {score:.2f}" if score > 0 else ""
        output_lines.append(f"    📄 {doc_name}{score_str}")
        output_lines.append("")

    return "\n".join(output_lines)
```

**Step 2: Update help command**

Find the help text and update:

```python
if cmd_name == "help":
    return """ZommaKG Commands:
    find -entity <names> -to <names>     Resolve names to canonical nodes
    search -entity <n> [-to <n>] [--query "action"] [--from DATE] [--to-date DATE]
                                          Find connections between nodes
    cat <number>                          Expand fact details
    info <name>                           Entity/topic summary
    ls [entities|topics|documents]        Browse available nodes
    stats                                 Knowledge base statistics

Workflow: find → search → cat
-entity alone: edges where entity appears anywhere
-entity + -to: edges between the two groups (bidirectional)
--query ranks by relationship/action (e.g., "acquired", "reported")"""
```

**Step 3: Update MCP tool docstring**

Update the `kg_execute` docstring to reflect new syntax.

**Step 4: Verify no syntax errors**

Run: `python -c "from zomma_kg.mcp.server import cmd_search; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add zomma_kg/mcp/server.py
git commit -m "feat: replace -topic with -to flag, bidirectional search"
```

---

### Task 3: Update find Command

**Files:**
- Modify: `zomma_kg/mcp/server.py:107-166`

**Step 1: Replace `-topic` with `-to` in find command**

```python
async def cmd_find(args: list[str]) -> str:
    """
    Resolve names to canonical entities/topics.

    Usage:
        find -entity "Apple, Tim Cook"
        find -to "inflation, BRK-B"
        find -entity Apple -to inflation
    """
    parser = argparse.ArgumentParser(prog="find", add_help=False)
    parser.add_argument("-entity", "--entity", type=str, default="")
    parser.add_argument("-to", "--to", dest="to_names", type=str, default="")

    try:
        parsed, _ = parser.parse_known_args(args)
    except SystemExit:
        return "Usage: find -entity <names> -to <names>"

    entity_names = parse_csv_list(parsed.entity) if parsed.entity else []
    to_names = parse_csv_list(parsed.to_names) if parsed.to_names else []

    if not entity_names and not to_names:
        return "Usage: find -entity <names> and/or -to <names>"

    kg = await get_kg()
    output_lines = []

    # Resolve entities
    if entity_names:
        output_lines.append("ENTITIES:")
        for name in entity_names:
            results = await kg.search_entities(name, limit=5, threshold=0.3)
            if results:
                output_lines.append(f'  "{name}"')
                for entity in results:
                    entity_type = entity.entity_type.upper() if hasattr(entity.entity_type, 'upper') else str(entity.entity_type).upper()
                    output_lines.append(f"    → [{entity_type}] {entity.name}")
            else:
                output_lines.append(f'  "{name}"')
                output_lines.append("    → (no matches found)")
        output_lines.append("")

    # Resolve -to targets (could be entities or topics)
    if to_names:
        output_lines.append("TARGETS:")
        for name in to_names:
            # Try entities first
            entity_results = await kg.search_entities(name, limit=3, threshold=0.3)
            # Try topics
            assert kg._storage is not None
            topic_results = await kg._storage.get_topics_by_names([name])

            output_lines.append(f'  "{name}"')
            found = False
            for entity in entity_results:
                entity_type = entity.entity_type.upper() if hasattr(entity.entity_type, 'upper') else str(entity.entity_type).upper()
                output_lines.append(f"    → [{entity_type}] {entity.name}")
                found = True
            for topic in topic_results:
                output_lines.append(f"    → [TOPIC] {topic.name}")
                found = True
            if not found:
                output_lines.append("    → (no matches found)")

    return "\n".join(output_lines)
```

**Step 2: Verify no syntax errors**

Run: `python -c "from zomma_kg.mcp.server import cmd_find; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add zomma_kg/mcp/server.py
git commit -m "feat: update find command to use -to instead of -topic"
```

---

### Task 4: Update SKILL.md Documentation

**Files:**
- Modify: `zomma_kg/skills/kg-query/SKILL.md`

**Step 1: Replace all `-topic` with `-to` throughout the document**

Key changes:
- Command examples
- Flags tables
- Workflow examples
- Example sessions

**Step 2: Update the search command section**

```markdown
### 2. `search` — Find Connections Between Nodes

Returns matching connections with dates and provenance.

- **Only `-entity`**: edges where entity appears (subject OR object)
- **With `-to`**: edges between `-entity` group and `-to` group (bidirectional)

```bash
search -entity "Apple Inc"
search -entity "Apple Inc" -to "BRK-B, Revenue"
search -entity "Fed Boston, Fed Atlanta" -to "Manufacturing, Employment"
search -entity "Apple Inc" -to Revenue --query "reported" --from 2024-01
```

**Flags:**
| Flag | Description |
|------|-------------|
| `-entity <names>` | Source entities (comma-separated) |
| `-to <names>` | Target entities/topics (comma-separated) |
| `--query "<text>"` | Semantic query to rank by relationship/action |
| `--from <date>` | Start date filter (YYYY-MM or YYYY-MM-DD) |
| `--to-date <date>` | End date filter (YYYY-MM or YYYY-MM-DD) |
| `--limit <n>` | Max results (default: 20) |
```

**Step 3: Update workflow section**

```markdown
## Workflow: find → search → cat

### Step 1: Resolve names
```bash
find -entity "Apple, Federal Reserve" -to "BRK-B, inflation"
```

### Step 2: Search for connections
```bash
# All edges involving Apple
search -entity "Apple Inc"

# Edges between Apple and specific targets
search -entity "Apple Inc" -to "BRK-B, Inflation"

# N-to-N between groups
search -entity "Fed Boston, Fed Atlanta" -to "Manufacturing, Employment"
```
```

**Step 4: Update common mistakes table**

Remove the row about `-topic` vs `--query`, update to reflect new `-to` usage.

**Step 5: Commit**

```bash
git add zomma_kg/skills/kg-query/SKILL.md
git commit -m "docs: update SKILL.md for -to flag and bidirectional search"
```

---

### Task 5: Manual Testing

**Step 1: Test the system end-to-end**

If you have a test KB, run these commands manually:

```bash
# Start a Python REPL or test script
python -c "
import asyncio
from zomma_kg.mcp.server import init_kg, execute_command

async def test():
    await init_kg('./test_kb')  # or your KB path

    # Test find
    print(await execute_command('find -entity \"Federal Reserve\"'))

    # Test search without -to
    print(await execute_command('search -entity \"Federal Reserve\"'))

    # Test search with -to
    print(await execute_command('search -entity \"Federal Reserve\" -to \"Inflation\"'))

asyncio.run(test())
"
```

**Step 2: Verify bidirectional behavior**

Confirm that edges go both ways (A→B and B→A are both returned).

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete -to flag implementation with bidirectional search"
```

---

## Summary

| Task | What Changes |
|------|--------------|
| 1 | DuckDB query method - bidirectional logic |
| 2 | Server search command - `-to` flag |
| 3 | Server find command - `-to` flag |
| 4 | SKILL.md documentation |
| 5 | Manual testing |
