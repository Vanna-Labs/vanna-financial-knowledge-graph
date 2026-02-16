# Ticket F - P1 Ingestion Pipeline Consolidation (Single Canonical Path)

Status: Completed

Date: 2026-02-02

## Priority

P1

## Depends on

Ticket C merged

## Scope

Eliminate duplicate ingestion logic by making `scripts/build_kg.py` call the
`KnowledgeGraph` API instead of re-implementing
chunk/extract/dedup/resolve/assemble.

## Why

Current duplication causes behavior drift (for example, topic ontology wiring
differed between API vs script), which creates inconsistent runtime behavior.

## Target files

- `scripts/build_kg.py`
- `vanna_kg/api/knowledge_graph.py` (only if small hooks/callbacks are needed)
- `tests/test_knowledge_graph.py`
- `tests/` (new script-wrapper test file if needed)

## Required changes

1. Refactor `scripts/build_kg.py` into a thin wrapper around:
   - `KnowledgeGraph.ingest_markdown(...)`
   - `KnowledgeGraph.ingest_pdf(...)` (if script supports PDF path)
2. Keep script-only UX (timing/progress prints) in wrapper-level callbacks/logging,
   not in duplicated pipeline logic.
3. Remove duplicate topic/entity resolution code from the script path.
4. Ensure script and API now use identical topic resolution behavior.
5. Keep CLI interface/arguments backward-compatible where practical.
6. Preserve working behavior while consolidating (no intentional semantic changes
   to successful ingestion outputs).

## Acceptance criteria

- Only one ingestion implementation path exists for core logic (API path).
- Running `scripts/build_kg.py` and direct `KnowledgeGraph.ingest_*` yields
  consistent topic/entity write behavior.
- No regression in ingestion tests.
- Existing successful ingestion behavior is preserved (same output shape/count
  expectations for standard fixtures).

## Verify

```bash
uv run pytest -q tests/test_knowledge_graph.py tests/test_topic_resolver.py tests/test_query_pipeline.py
uv run pytest -q
```

## Notes / non-goals

- Do not redesign extraction/dedup algorithms in this ticket.
- Do not expand scope beyond consolidation and parity.
