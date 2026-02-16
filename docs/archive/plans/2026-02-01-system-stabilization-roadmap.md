# VannaKG Stabilization Roadmap (Priority-Ordered)

Date: 2026-02-01

## Purpose

This document consolidates the highest-impact integration issues found in the current codebase and defines a strict, execution-ready fix plan for multiple agents.

Use this as the single source of truth for stabilization work.

---

## Executive Assessment

The architecture is strong and modular, and tests currently pass, but there are critical interface mismatches between layers (facade ↔ types, facade ↔ query models, researcher ↔ storage query outputs) that can break runtime paths.

**Core risk:** schema/interface drift across modules.

---

## Confirmed Issues (Consolidated)

## 1) Ingestion facade constructs models with outdated field names (runtime break risk)

- File: `vanna_kg/api/knowledge_graph.py`
- Evidence:
  - `TopicDefinition(name=...)` is used, but model expects `topic` (`vanna_kg/types/topics.py`).
  - `Document(filename=..., date=...)` is used, but model expects `name`, `document_date` (`vanna_kg/types/chunks.py`).
  - `Chunk(doc_uuid=...)` is used, but model expects `document_uuid`.
  - `Fact(... relationship=...)` is used, but model expects `relationship_type`, plus required `subject_name`, `object_name`, `object_type`.
  - `TopicResolution` is read with `ontology_uuid`, but model field is `uuid`.

## 2) Query facade output mapping mismatches current query models (runtime break risk)

- File: `vanna_kg/api/knowledge_graph.py`
- Evidence:
  - Uses `sa.query_text`; current model field is `sa.sub_query` (`vanna_kg/query/types.py`).
  - Uses `result.question_type.value`; current model stores string (`question_type: str`), not enum.

## 3) Retrieval key mapping mismatch drops chunk metadata/context

- File: `vanna_kg/query/researcher.py`
- Evidence:
  - Reads `cd.get("uuid")` and `cd.get("document_name")`.
  - DuckDB query returns `chunk_id` and `doc_name` (`vanna_kg/storage/duckdb/queries.py`).
- Impact:
  - `chunk_id` can become empty.
  - context dedupe quality degrades.
  - source attribution quality degrades.

## 4) Topic resolver wiring likely incorrect for ontology-vs-default groups

- Files: `vanna_kg/api/knowledge_graph.py`, `vanna_kg/ingestion/resolution/topic_resolver.py`, `vanna_kg/storage/lancedb/indices.py`
- Evidence:
  - `TopicResolver` gets `self._storage._lancedb` (default group), while resolver expects ontology candidates and filters by ontology group.
  - LanceDB `search_topics` already filters by index `group_id`; wrong index group will return empty ontology candidates.

## 5) Public API / convenience drift

- File: `vanna_kg/api/convenience.py`
- Evidence:
  - Calls `kg.ingest_chunks(...)` and `kg.decompose(...)`, but these methods are not present in `KnowledgeGraph`.

## 6) Feature promises vs implementation gaps

- `KGShell` methods are mostly `NotImplemented`.
- `search_chunks` in `KnowledgeGraph` is `NotImplemented`.
- Global chunk search in researcher is not implemented.

## 7) Documentation drift

- Several ingestion/resolution/assembly READMEs still describe Neo4j/Qdrant pipeline while code is Parquet + DuckDB + LanceDB.

## 8) Scale risk in parquet append strategy

- File: `vanna_kg/storage/parquet/backend.py`
- Current append path reads whole parquet, concatenates, rewrites whole file per append.
- Works functionally; will degrade at larger KB sizes.

---

## Strict Priority Fix Roadmap

**Rule:** do not start P1 until all P0 acceptance criteria pass.

## P0 - Runtime Correctness (Blockers)

### P0.1 Align ingestion facade with current typed models

- Update `vanna_kg/api/knowledge_graph.py`:
  - `TopicDefinition(topic=..., definition=...)`
  - `Document(name=..., document_date=..., ...)`
  - `Chunk(document_uuid=..., ...)`
  - `Fact(relationship_type=..., subject_name=..., object_name=..., object_type=..., ...)`
  - topic resolution reads `tr.uuid` not `tr.ontology_uuid`
- Acceptance criteria:
  - `ingest_markdown` path succeeds end-to-end on sample input with real model objects.
  - no Pydantic validation errors from field mismatches.

### P0.2 Fix query result mapping in facade

- Update `vanna_kg/api/knowledge_graph.py` query mapping:
  - `sa.sub_query` (not `sa.query_text`)
  - `question_type=result.question_type` (string passthrough)
- Acceptance criteria:
  - `KnowledgeGraph.query(...)` returns valid `QueryResult`.
  - no attribute errors on question type or sub-answer mapping.

### P0.3 Fix researcher chunk key mapping

- Update `vanna_kg/query/researcher.py`:
  - Use `chunk_id` and `doc_name` keys from storage responses.
  - Optional backward-compatible fallback for legacy keys.
- Acceptance criteria:
  - retrieved chunks carry non-empty `chunk_id` and correct `doc_name`.
  - context dedupe works on real IDs.

### P0.4 Fix topic resolver wiring for ontology index

- Ensure topic resolution uses ontology-group index explicitly.
- Preferred: instantiate dedicated ontology `LanceDBIndices(..., group_id="ontology")` where needed, or refactor resolver to own that split clearly.
- Acceptance criteria:
  - topic resolution returns ontology matches where expected.
  - no empty-result regression due to group mismatch.

### P0.5 Add/repair integration tests for these runtime paths

- Add tests covering:
  - `ingest_markdown` real model construction path.
  - `KnowledgeGraph.query` mapping path.
  - researcher chunk mapping with storage query output shape.
  - topic resolution with ontology/default split.
- Acceptance criteria:
  - tests fail before fixes, pass after.

---

## P1 - API Coherence + Developer Safety

### P1.1 Reconcile convenience API with actual `KnowledgeGraph` methods

Choose one:
- implement missing `KnowledgeGraph.ingest_chunks` and `KnowledgeGraph.decompose`, or
- remove/replace convenience calls to non-existent methods.

Acceptance:
- no dead public entry points.

### P1.2 Fix sync-wrapper behavior inside running event loops

- `close_sync` currently fails when called from an active loop.
- make behavior explicit and safe (clear exception or alternate mechanism).

Acceptance:
- no unawaited coroutine warnings.

### P1.3 Reduce private-attribute coupling

- Replace call sites using `._storage`, `._duckdb`, `._lancedb` directly where feasible with explicit storage API methods.

Acceptance:
- fewer cross-layer private accesses.

---

## P2 - Truthful Product Surface

### P2.1 Update stale docs (Neo4j/Qdrant references)

- Refresh:
  - `vanna_kg/ingestion/extraction/README.md`
  - `vanna_kg/ingestion/resolution/README.md`
  - `vanna_kg/ingestion/assembly/README.md`

Acceptance:
- docs describe current Parquet + DuckDB + LanceDB system.

### P2.2 Mark unimplemented features clearly

- `KGShell`, global chunk search, chunk vector search:
  - either implement, or explicitly document as roadmap/experimental.

Acceptance:
- no misleading “available” implication in user-facing docs.

---

## P3 - Scalability Hardening

### P3.1 Improve parquet append strategy

- Move away from full-table rewrite per append operation.
- Options: partitioned writes, append datasets, periodic compaction.

Acceptance:
- large-ingest performance baseline improves.

---

## Agent Work Packages (Parallelizable)

## Agent A (P0.1)
- Fix ingestion model-field mismatches in `vanna_kg/api/knowledge_graph.py`.
- Add focused ingestion integration tests.

## Agent B (P0.2 + P0.3)
- Fix query result mapping and researcher chunk key mapping.
- Add mapping-focused tests in query pipeline suite.

## Agent C (P0.4)
- Resolve ontology/default topic index wiring.
- Add resolution tests for ontology match behavior.

## Agent D (P1 + P2)
- Reconcile convenience API drift.
- Update docs and feature-status accuracy.

## Agent E (P3, optional after stabilization)
- Design and prototype parquet append/compaction strategy.

---

## Suggested Verification Commands

Run after each P0 package merges:

```bash
uv run pytest -q
```

For targeted checks during development:

```bash
uv run pytest -q tests/test_knowledge_graph.py tests/test_query_pipeline.py tests/test_topic_resolver.py
```

---

## Definition of Done (Stabilization Milestone)

- All P0 items complete with tests.
- `uv run pytest -q` passes.
- Ingestion and query facade paths run without schema/attribute runtime errors.
- Topic resolution works with ontology matching.
- Retrieval context includes valid chunk/document identifiers.

