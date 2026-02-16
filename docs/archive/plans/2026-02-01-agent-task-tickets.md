# Agent Task Tickets - System Stabilization

Status: Completed

Date: 2026-02-01  
Primary roadmap: `docs/plans/2026-02-01-system-stabilization-roadmap.md`

## How to use this file

- Assign one ticket per agent.
- Agents should only touch files listed in their ticket unless explicitly noted.
- Merge order: all `P0` tickets first, then `P1/P2`, then optional `P3`.

---

## Ticket A - P0.1 Ingestion Facade Schema Alignment

**Priority:** P0 (blocker)  
**Owner:** Agent A  
**Depends on:** none

## Scope

Fix ingestion model construction in `KnowledgeGraph` so it matches current Pydantic schemas.

## Target files

- `vanna_kg/api/knowledge_graph.py`
- `tests/test_knowledge_graph.py` (and/or new focused ingestion test file)

## Required changes

1. Replace outdated constructor fields:
   - `TopicDefinition(name=...)` -> `TopicDefinition(topic=...)`
   - `Document(filename/date)` -> `Document(name/document_date)`
   - `Chunk(doc_uuid)` -> `Chunk(document_uuid)`
   - `Fact(relationship)` -> `Fact(relationship_type)`
2. Ensure `Fact` creation provides required fields:
   - `subject_name`, `object_name`, `object_type`
3. Fix topic resolution mapping:
   - use `tr.uuid` (not `tr.ontology_uuid`)

## Acceptance criteria

- `ingest_markdown` path no longer throws Pydantic validation errors from field mismatches.
- tests cover these constructor paths.
- no regressions in existing tests.

## Verify

```bash
uv run pytest -q tests/test_knowledge_graph.py
uv run pytest -q
```

---

## Ticket B - P0.2 + P0.3 Query Mapping and Retrieval Key Fixes

**Priority:** P0 (blocker)  
**Owner:** Agent B  
**Depends on:** none

## Scope

Fix runtime field mismatches in query-result mapping and chunk-key mapping in researcher retrieval.

## Target files

- `vanna_kg/api/knowledge_graph.py`
- `vanna_kg/query/researcher.py`
- `tests/test_query_pipeline.py`
- (optional) `tests/test_knowledge_graph.py`

## Required changes

1. `KnowledgeGraph.query(...)` mapping fixes:
   - `sa.query_text` -> `sa.sub_query`
   - `result.question_type.value` -> `result.question_type` (string passthrough)
2. Researcher retrieval key fixes:
   - read `chunk_id` (not `uuid`)
   - read `doc_name` (not `document_name`)
   - optionally keep backward-compatible fallback to legacy keys

## Acceptance criteria

- no `AttributeError` on sub-answer or question type mapping.
- retrieved chunks have valid `chunk_id` and doc name.
- context dedupe uses real chunk IDs.

## Verify

```bash
uv run pytest -q tests/test_query_pipeline.py tests/test_knowledge_graph.py
uv run pytest -q
```

---

## Ticket C - P0.4 Topic Resolver Ontology Wiring

**Priority:** P0 (blocker)  
**Owner:** Agent C  
**Depends on:** none

## Scope

Ensure topic resolution consistently searches ontology-group vectors and maps to KB topics correctly.

## Target files

- `vanna_kg/api/knowledge_graph.py`
- `vanna_kg/ingestion/resolution/topic_resolver.py`
- `vanna_kg/query/researcher.py` (only if needed for consistency)
- `tests/test_topic_resolver.py`
- `tests/test_query_pipeline.py` (topic resolution behavior)

## Required changes

1. Remove ambiguity around index group usage:
   - ingestion-time topic resolver must use an explicit `LanceDBIndices(..., group_id="ontology")` path (not storage default index).
2. Validate that resolution pipeline still returns KB topic UUID/name for final use.
3. Add/adjust tests to protect against empty-result regressions due to wrong group wiring.
4. Ensure query-time two-stage behavior remains explicit:
   - stage 1 ontology candidate search
   - stage 2 KB topic lookup by matched names (default group)

## Acceptance criteria

- ontology candidates are returned when expected.
- resolved topics map to KB topics correctly.
- tests explicitly guard ontology/default group behavior.
- ingestion path no longer silently returns `topics=0` due to ontology/default index mismatch.

## Verify

```bash
uv run pytest -q tests/test_topic_resolver.py tests/test_query_pipeline.py
uv run pytest -q
```

---

## Ticket D - P1 API Coherence + P2 Docs/Surface Truthfulness

**Priority:** P1/P2  
**Owner:** Agent D  
**Depends on:** all P0 tickets merged

## Scope

Clean up public API drift and make docs/features truthful.

## Target files

- `vanna_kg/api/convenience.py`
- `vanna_kg/api/knowledge_graph.py` (if adding missing methods)
- `README.md`
- `vanna_kg/ingestion/extraction/README.md`
- `vanna_kg/ingestion/resolution/README.md`
- `vanna_kg/ingestion/assembly/README.md`
- optional: `vanna_kg/api/shell.py`, CLI help text where needed

## Required changes

1. Resolve convenience API mismatch:
   - either implement missing `KnowledgeGraph.ingest_chunks` / `KnowledgeGraph.decompose`
   - or remove/replace convenience calls to non-existent methods.
2. Update stale docs (remove Neo4j/Qdrant descriptions where obsolete).
3. Clearly mark unimplemented features (`KGShell`, chunk/global chunk search) as roadmap/placeholder.

## Acceptance criteria

- no dead convenience entry points.
- docs match current storage/runtime architecture.
- user-facing feature status is explicit.

## Verify

```bash
uv run pytest -q
```

---

## Ticket E - P3 Storage Scalability Hardening (Optional After Stabilization)

**Priority:** P3  
**Owner:** Agent E  
**Depends on:** all P0/P1 complete

## Scope

Design and prototype a better write path than full parquet rewrite-on-append.

## Target files

- `vanna_kg/storage/parquet/backend.py`
- optional new module(s) under `vanna_kg/storage/parquet/`
- performance/behavior tests (new)

## Required changes

1. Propose concrete append strategy (partitioned dataset append, staged writes, or compaction model).
2. Implement minimally safe version preserving current correctness.
3. Add tests/benchmarks for correctness + basic performance regression guardrails.

## Acceptance criteria

- correctness preserved vs existing behavior.
- measurable improvement or clear migration path documented.

## Verify

```bash
uv run pytest -q
```

---

## Cross-Agent Rules

1. Do not rewrite unrelated modules.
2. Keep each PR scoped to its ticket.
3. Add tests with each fix; do not rely on manual validation.
4. If a ticket blocks on another ticket, stop and document the blocker.

---

## PR Handoff Template (Copy/Paste)

```md
## Ticket
- ID: <A/B/C/D/E>
- Priority: <P0/P1/P2/P3>

## What changed
- <bullet list>

## Why
- <runtime bug / alignment / docs drift / scalability>

## Files touched
- <path list>

## Tests added/updated
- <test names>

## Verification
- `uv run pytest -q ...` output summary

## Risks / follow-ups
- <if any>
```
