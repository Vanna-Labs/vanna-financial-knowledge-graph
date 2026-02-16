# Ticket H - P1 MCP Query Contract Realignment (No Backward Compatibility)

Status: Completed

Date: 2026-02-02

## Priority

P1

## Depends on

None

## Scope

Realign MCP query commands to the intended product contract:

1. `find` resolves canonical names by type (`-entity`, `-topic`)
2. `search` finds connections with explicit graph mode (`around` vs `between`)
3. `cat` expands numbered search results
4. `info` provides entity/topic summary

This ticket explicitly removes the accidental `-to` contract and standardizes
around/between semantics for fresh agent workflows.

## Why

Current contract drift caused confusion:

- `find` was expected to be resolve-only with `-entity`/`-topic`, but currently
  uses `-to`
- `search` behavior is ambiguous between:
  - "search around X node(s)"
  - "search only within selected node subset"
- Docs/skill text no longer cleanly reflect intended usage

We need a deterministic, agent-friendly command API with explicit semantics.

## Required command contract

### `find`

- Flags: `-entity <names>`, `-topic <names>`
- Purpose: resolve user-provided names to canonical nodes
- No graph retrieval in this step
- **No backward compatibility for `-to`**

### `search`

- Flags:
  - `-entity <names>` (optional)
  - `-topic <names>` (optional)
  - `--mode around|between` (default: `around`)
  - `--query`, `--from`, `--to-date`, `--limit` (existing behavior retained)
- Must require at least one selector (`-entity` or `-topic`)

#### `around` semantics (default)

Let `S` be all selected nodes (resolved entity + topic names).

Return edges where either endpoint is in `S`:
- `(subject in S OR object in S)`

Use case: "search around X node(s)".

#### `between` semantics

Let `S` be all selected nodes (resolved entity + topic names).

Return only edges fully inside selected subset:
- `(subject in S AND object in S)`

Use case: "search only connections among selected nodes".

Examples:
- `search -entity "Apple Inc, Microsoft Corp" --mode between`
  - Should allow `Apple <-> Microsoft` edges
- `search -entity "Apple Inc" -topic "Inflation" --mode between`
  - Should return only edges where both endpoints are inside `{Apple Inc, Inflation}`

Important: `between` is **not** limited to entity->topic-only edges; it is
node-subgraph filtering over the selected set.

### `cat`

- Keep existing behavior:
  - `cat <result_number> [result_number ...]`
  - Uses latest `search` session result indices

### `info`

- Keep existing summary behavior for entity/topic name input

## Target files

- `vanna_kg/mcp/server.py`
- `vanna_kg/storage/duckdb/queries.py`
- `tests/test_mcp_server.py`
- `vanna_kg/skills/kg-query/SKILL.md`
- `vanna_kg/skills/kg-query/references/examples.md`
- `vanna_kg/skills/kg-query/references/troubleshooting.md`
- `vanna_kg/mcp/__init__.py`

## Required implementation changes

1. Update `cmd_find` parser/usage:
   - Replace `-to` with `-topic`
   - Output typed resolution sections for entities/topics
   - Remove `-to` path entirely (no alias/deprecation)

2. Update `cmd_search` parser/usage:
   - Replace `-to` with `-topic`
   - Add `--mode` with choices `around|between`, default `around`
   - Require at least one selector
   - Reflect mode and selectors in output header

3. Update DuckDB query helper(s):
   - Support node-set filtering for both modes:
     - around: endpoint overlap with selected set
     - between: both endpoints inside selected set
   - Preserve date filtering via `document_date`
   - Preserve limit and ordering behavior

4. Keep semantic reranking behavior:
   - Structural filter first
   - `--query` reranks filtered facts only

5. Update help/docstrings/skill docs:
   - Remove `-to` usage
   - Document `--mode` semantics clearly and consistently

6. Update/expand tests:
   - `find` with `-entity`/`-topic`
   - `find` rejects old `-to` contract
   - `search` default `around`
   - `search --mode between` with:
     - entity-only subset (`Apple, Microsoft`)
     - mixed subset (`Apple`, `Inflation`)
   - `cat` session-index behavior unchanged

## Acceptance criteria

- `find -entity "Apple" -topic "Inflation"` works as canonical resolution.
- `find -to ...` is no longer supported.
- `search -entity "Apple Inc"` defaults to `around`.
- `search -entity "Apple Inc, Microsoft Corp" --mode between` filters to
  edges whose endpoints are both inside selected subset.
- `search -entity "Apple Inc" -topic "Inflation" --mode between` filters to
  edges whose endpoints are both inside selected subset.
- `cat` behavior remains stable with search-session indexing.
- MCP help text and skill docs exactly match implemented command contract.

## Verify

```bash
uv run pytest -q tests/test_mcp_server.py
uv run pytest -q
```

## Notes / non-goals

- **No backward compatibility** for legacy `-to`.
- Decision to remove `ls` from public command surface is out of scope for this
  ticket; keep or remove in a separate explicit ticket.
