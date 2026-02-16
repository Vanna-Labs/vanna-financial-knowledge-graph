# Troubleshooting

## Common Failures and Fixes

### "No connections found for mode: ... + entities/topics: ..."

Try:

1. Run `find` again and use canonical names from its output.
2. Expand scope by switching to `--mode around` or widening dates.
3. Remove `--query` temporarily to inspect raw matches first.

### "No search results. Run 'search' first."

`cat` only works after `search` in the same session.

Run:

```bash
search -entity "..."
cat 1
```

### "Result N not found. Valid: 1-M"

A newer `search` reset result numbering.

Fix:

1. Re-run the search you care about.
2. Use the current displayed indices.

### Too many or ambiguous entity matches from `find`

This is expected with aliases and overlapping names.

Fix:

1. Keep only context-relevant matches.
2. Include multiple resolved names when uncertain.
3. Tighten selector definitions or pair with narrower `-topic` filters or date filters.

### Weak relevance with `--query`

`--query` reranks already filtered facts. It does not retrieve outside the current filter set.

Fix:

1. Improve structure first with `-entity`/`-topic` and the right mode (`around` vs `between`).
2. Use action/relationship wording in `--query` (for example: `acquired`, `reported decline`, `hedging`).
3. Use `cat` to verify factual grounding.

## Current Output Expectations

- `search` line date reflects fact `date_context`.
- `search --from/--to-date` filters by publication date (`document_date`).
- `cat` shows fact text + source metadata + chunk UUID (not full document metadata).
- `info` for entities includes aliases/fact-count/connections; topic output is definition-focused.
- `stats` reports aggregate counts (entities, facts, chunks, documents).

## Practical Notes

- Prefer `find -> search -> cat -> info` over ad-hoc guessing.
- For temporal questions, run separate searches per time window and compare `cat` outputs.
- Treat extracted facts as model-derived; confirm sensitive conclusions in source chunk context.
