# Ticket G - P1 KnowledgeGraph Ingestion Internal Dedup (Preserve Behavior)

Status: Completed

Date: 2026-02-02

## Priority

P1

## Depends on

None (can run independently), but best after Ticket C.

## Scope

Remove duplicated ingestion orchestration inside `KnowledgeGraph` by extracting
shared logic from `ingest_pdf(...)` and `ingest_markdown(...)` into one internal
code path.

## Why

`ingest_pdf` and `ingest_markdown` currently duplicate most of extraction,
entity deduplication, entity/topic resolution, fact/topic construction, and
assembly logic. This increases drift risk and makes fixes error-prone.

## Target files

- `vanna_kg/api/knowledge_graph.py`
- `tests/test_knowledge_graph.py`
- optional new focused test file under `tests/` for ingestion parity

## Required changes

1. Introduce a shared internal ingestion core in `KnowledgeGraph` for the
   common pipeline after chunk inputs are available.
2. Keep `ingest_pdf(...)` and `ingest_markdown(...)` as thin entry points:
   - input-specific chunking/loading only
   - then delegate to shared internal core
3. Preserve working behavior while refactoring:
   - public method signatures unchanged
   - `IngestResult` fields/meaning unchanged
   - progress callback stage semantics unchanged where currently used
   - no intentional semantic changes to successful ingestion outputs
4. Add tests that guard behavior parity between PDF/Markdown ingestion paths
   for shared stages (resolution, fact/topic construction, assembly payload).
5. Keep changes scoped; no unrelated API redesign.

## Acceptance criteria

- `KnowledgeGraph.ingest_pdf(...)` and `KnowledgeGraph.ingest_markdown(...)`
  use a shared internal orchestration path for common logic.
- Existing ingestion tests pass.
- New tests prevent drift between PDF/Markdown paths.
- Working ingestion behavior is preserved for standard fixtures.

## Verify

```bash
uv run pytest -q tests/test_knowledge_graph.py
uv run pytest -q tests/test_topic_resolver.py tests/test_query_pipeline.py
uv run pytest -q
```

## Notes / non-goals

- This ticket is internal refactor + safety coverage, not a feature expansion.
- If a behavior change is desired, do it in a separate ticket with explicit
  before/after expectations.
