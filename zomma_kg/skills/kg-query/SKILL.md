---
name: kg-query
description: Use when issuing ZommaKG MCP `kg_execute` shell-style commands to resolve names (`find`), search graph connections (`search`), inspect fact details (`cat`), and perform temporal/provenance analysis over financial-document knowledge bases.
---

# ZommaKG Graph Query Skill

Query the knowledge graph through the MCP tool `kg_execute` using the workflow:
`find -> search -> cat`.

Run commands only through:

```python
kg_execute(command='find --entity {"name":"Apple","definition":"US technology company"}')
kg_execute(command='find --topic {"name":"Inflation","definition":"General price increases"}')
kg_execute(command='find --entity {"name":"Apple","definition":"Consumer electronics company"} --topic {"name":"Inflation","definition":"General price increases"}')
kg_execute(command="search -entity \"Apple Inc\" -topic Inflation --mode around --query \"reported\"")
kg_execute(command="cat 1")
```

## Use This Core Workflow

1. Resolve names with `find` before searching.
2. Search connections with `search` using canonical names from `find`.
3. Expand specific results with `cat <result_number>`.

For more worked sessions, read `references/examples.md`.
For failure handling and edge cases, read `references/troubleshooting.md`.

## Command Semantics

### `find`

Resolve input strings to canonical entities/topics.

- Use repeatable `--entity` for entity selectors.
- Use repeatable `--topic` for topic selectors.
- Each selector is a JSON object with `name` and `definition` (required), plus optional `id`.
- Treat multiple matches as distinct candidates, not duplicates.
- Include all relevant resolved names in later `search` calls.

Examples:

```bash
find --entity {"name":"Apple","definition":"Consumer electronics company"}
find --topic {"name":"Inflation","definition":"General price increases"}
find --entity {"name":"Federal Reserve","definition":"US central banking system"} --topic {"name":"Economic Conditions","definition":"Macro conditions in the economy"}
find --entity {"id":"e1","name":"Apple","definition":"Consumer electronics company"} --entity {"id":"e2","name":"Microsoft","definition":"Enterprise software and cloud services company"}
```

### `search`

Find facts and assign numbered results for `cat`.

- Requires at least one selector: `-entity` and/or `-topic`.
- `--mode around` (default): match edges where either endpoint is in the selected node set.
- `--mode between`: match edges where both endpoints are in the selected node set.
- `--query`: semantic reranking of already filtered facts (not global search).
- `--from` / `--to-date`: filter by document date (publication date).

Examples:

```bash
search -entity "Apple Inc"
search -topic "Inflation"
search -entity "Apple Inc, Microsoft Corp" --mode between
search -entity "Fed Boston, Fed Atlanta" -topic "Manufacturing, Employment" --mode around
search -entity "Apple Inc" -topic "Interest Rates" --query "affected by" --from 2024-01
```

### `cat`

Expand one or more result numbers from the latest `search`.

```bash
cat 1
cat 1 2 3
```

### `info`

Return summary data for one entity or topic.

```bash
info "Apple Inc"
info "Interest Rates"
```

### `ls`

Browse available nodes/documents.

```bash
ls entities
ls entities -type company
ls topics
ls documents
```

### `stats`

Return high-level KB counts.

```bash
stats
```

## Time and Provenance Guardrails

Always disambiguate these time fields:

- `document_date`: publication date used for `search --from/--to-date` filtering.
- `date_context`: event period stored on facts and shown in `search` output lines.

Key implication: a document published in January 2025 can contain facts about Q4 2024.

When validating claims, open results with `cat` to inspect extracted fact text and source chunk metadata.

## Session-State Rules

- `cat` works only after `search` in the same session.
- Each new `search` resets result numbering from `[1]`.
- If `cat` fails with "Result N not found," rerun `search` and use current indices.

## Quoting Rules

- Quote multi-word names: `"Federal Reserve"`.
- Quote comma-separated lists: `"Apple, Microsoft, Google"`.
- Single-word names may be unquoted: `Apple`, `Inflation`.
- `find` selectors are raw JSON objects; do not wrap them in quotes unless your client requires it.

## Compact Command Reference

| Command | Purpose | Key flags |
| --- | --- | --- |
| `find` | Resolve names to canonical nodes | `--entity`, `--topic` |
| `search` | Find connections and create result indices | `-entity`, `-topic`, `--mode`, `--query`, `--from`, `--to-date`, `--limit` |
| `cat` | Expand result details from latest search | Result numbers only |
| `info` | Summarize entity or topic | Name |
| `ls` | Browse entities/topics/documents | `entities`, `topics`, `documents`, `-type`, `--limit` |
| `stats` | Show KB totals | None |
| `help` | Show usage summary | None |
