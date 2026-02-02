# Ticket I - P1 MCP `find` Structured Selector Contract (Agent-Facing)

Status: Completed

Date: 2026-02-02

## Priority

P1

## Depends on

- Ticket H (`2026-02-02-ticket-h-mcp-query-contract-realignment.md`)

## Scope

Update MCP `find` to require structured selector objects for entity/topic
resolution. Each selector must include both:

- `name`
- `definition`

This is an agent-facing contract and must avoid ambiguous free-form parsing
such as `"name: definition"`.

## Why

`find` currently accepts free-form strings. For agents, this causes:

- weaker disambiguation for non-exact names
- inconsistency with dedup/resolution embedding strategy (`name + definition`)
- fragile parsing when meaning is packed into a single unstructured string

We want deterministic, machine-friendly inputs with stronger resolution quality.

## Required command contract

### `find` (new required input shape)

`find` must accept repeatable structured selector flags:

- `--entity '<json object>'`
- `--topic '<json object>'`

Selector object schema:

- required: `name` (string), `definition` (string)
- optional: `id` (string, passthrough only for client correlation)

Rules:

1. At least one selector (`--entity` or `--topic`) is required.
2. Each selector value must parse as a JSON object.
3. `name` and `definition` must exist and be non-empty after trim.
4. `id`, if present, must be a non-empty string.
5. **No backward compatibility** for old `find` input styles:
   - `-entity "A, B"`
   - `-topic "X, Y"`
   - paired flag form (`--entity-name/--entity-def`, `--topic-name/--topic-def`)
   - embedded `"name: definition"` contract

Examples:

```bash
find --entity '{"name":"Apple Inc","definition":"US technology company focused on consumer devices"}'

find \
  --entity '{"name":"Apple","definition":"Consumer electronics and services company"}' \
  --entity '{"name":"Microsoft","definition":"Enterprise software and cloud services company"}' \
  --topic  '{"name":"Inflation","definition":"General increase in price levels over time"}'

find \
  --entity '{"id":"e1","name":"Federal Reserve","definition":"US central banking system"}' \
  --topic  '{"id":"t1","name":"Labor Market","definition":"Employment, hiring, and wage conditions"}'
```

## Resolution semantics for `find`

For each selector object, construct query text as:

- `"{name}: {definition}"`

Then resolve candidates via vector search.

### Entity resolution

- Use entity semantic search with the constructed query text.
- Keep top-k behavior as existing `find` output currently does (or explicitly set
  to 5 for consistency).

### Topic resolution

- Use topic semantic search with the constructed query text.
- Do not rely only on exact name lookup.

### Output shape

Keep typed sections:

- `ENTITIES:`
- `TOPICS:`

For each input selector, print the searched `name` and matched canonical
candidates. If `id` is provided, include it in the output line for traceability.

## Target files

- `zomma_kg/mcp/server.py`
- `tests/test_mcp_server.py`
- `zomma_kg/skills/kg-query/SKILL.md`
- `zomma_kg/skills/kg-query/references/examples.md`
- `zomma_kg/skills/kg-query/references/troubleshooting.md`
- `zomma_kg/mcp/__init__.py`

## Required implementation changes

1. Update `cmd_find` parser and usage text:
   - Add repeatable `--entity` and `--topic` args.
   - Parse each arg as JSON object.
   - Validate required fields (`name`, `definition`) and optional `id`.
   - Remove legacy `find` selector parsing paths.

2. Add deterministic query-text construction per selector:
   - `query_text = f"{name}: {definition}"`

3. Use vector-backed resolution for both entities and topics:
   - Entities via existing semantic entity search path.
   - Topics via topic embedding search path.

4. Keep `search`, `cat`, `info`, and `ls` behavior unchanged.

5. Update MCP help/docstrings/skill docs to reflect the new `find` contract
   exactly.

6. Update/expand tests in `tests/test_mcp_server.py`:
   - valid entity selector object
   - valid topic selector object
   - mixed multi-selector request
   - invalid JSON rejection
   - missing `name` or `definition` rejection
   - empty-definition rejection
   - legacy `find -entity ...` rejection
   - legacy `find -topic ...` rejection

## Acceptance criteria

- `find --entity '{"name":"Apple","definition":"Technology company"}'` works.
- `find --topic '{"name":"Inflation","definition":"General price increases"}'` works.
- Repeated `--entity`/`--topic` selectors work in one command.
- `find` rejects invalid JSON selector input.
- `find` rejects missing/empty required fields.
- `find -entity "Apple"` is no longer supported.
- `find -topic "Inflation"` is no longer supported.
- Skill/help docs match implemented contract exactly.

## Verify

```bash
uv run pytest -q tests/test_mcp_server.py
uv run pytest -q
```

## Notes / non-goals

- This ticket changes only the `find` selector input contract.
- `search` contract from Ticket H remains unchanged.
- Do not add backward compatibility shims for old `find` input styles.
