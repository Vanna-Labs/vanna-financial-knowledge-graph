# Ticket C Handoff Notes (Observed During 10-Chunk Ingestion Runs)

Status: Completed

## What I ran

1. **Manual script path (pipeline modules directly)**
   ```bash
   uv run python scripts/build_kg.py test_data/BeigeBook_20251015.md --chunks 10 --output ./test_kb_debug_10
   ```

2. **Facade path (`KnowledgeGraph.ingest_markdown`) capped to 10 chunks**
   - Used a temporary monkey-patch in a one-off runner to limit `chunk_markdown(...)[:10]`.
   - Called:
     - `KnowledgeGraph(...).ingest_markdown("test_data/BeigeBook_20251015.md", document_date="2025-10-15")`

## Key outputs and logs

- **Script path output** (successful resolution):
  - `63 topics matched, 22 new`
  - Assembly succeeded and wrote document/chunks/entities/facts/relationships.

- **Facade path output**:
  - `INGEST_RESULT ... chunks=10, entities=119, facts=132, topics=0, errors=[]`
  - `KB_STATS ... documents=1, entities=119, chunks=10, facts=132`
  - Warnings:
    - `Duplicate decision for topic 'Real estate'`
    - `Duplicate decision for topic 'Economic Conditions'`
    - `Duplicate decision for topic 'Consumer Spending'`

## Ticket C territory findings (likely root causes)

1. **Ontology/default LanceDB group wiring mismatch in facade path**
   - Facade constructs `TopicResolver` using storage's LanceDB index:
     - `vanna_kg/api/knowledge_graph.py:325`
     - `vanna_kg/api/knowledge_graph.py:521`
   - That index is created in `ParquetBackend` with backend `group_id` defaulting to `"default"`:
     - `vanna_kg/storage/parquet/backend.py:77`
   - But topic resolver ontology flow expects ontology-group data:
     - Ontology rows are inserted with `group_id="ontology"`:
       - `vanna_kg/ingestion/resolution/topic_resolver.py:218`
     - Candidate search path filters by index group first:
       - `vanna_kg/storage/lancedb/indices.py:400`
     - Then resolver filters candidates to ontology group again:
       - `vanna_kg/ingestion/resolution/topic_resolver.py:344`

   **Consequence:** if resolver runs on default-group index, ontology candidates can be empty, leading to `topics=0` despite no hard failure.

2. **Evidence from control path**
   - `scripts/build_kg.py` explicitly uses:
     - `LanceDBIndices(..., group_id="ontology")` at `scripts/build_kg.py:218`
   - Same document/chunk count then yields many matched topics (`63`) in script path.

3. **Duplicate topic decision warnings**
   - Emitted from decision map collision logic:
     - `vanna_kg/ingestion/resolution/topic_resolver.py:413-419`
   - This is likely secondary, but can still degrade match quality by overwriting decisions for repeated normalized topic keys.

## Additional nuance relevant to C acceptance

- During ontology load, resolver stores topic row names as `text.split(":")[0]`:
  - `vanna_kg/ingestion/resolution/topic_resolver.py:216`
- Because embeddings include both labels and synonyms, returned `matched["name"]` may be synonym text rather than canonical label.
- This can affect "map to KB topics correctly" expectations in Ticket C.

## Working tree status check (2026-02-01)

I reviewed current unstaged diffs in Ticket C-adjacent files:

- `vanna_kg/api/knowledge_graph.py`: contains Ticket A/B style schema/query mapping fixes, but still constructs `TopicResolver(self._storage._lancedb, ...)` in ingestion paths.
- `vanna_kg/ingestion/resolution/topic_resolver.py`: no Ticket C wiring change yet.
- `vanna_kg/query/researcher.py` + `tests/test_query_pipeline.py`: chunk key mapping updates are present (Ticket B), not Ticket C ontology/default wiring tests.
- `tests/test_topic_resolver.py`: no new ingestion-wiring regression tests yet.

Net: Ticket C still open.

## Updated execution checklist for Ticket C

1. **Fix ingestion-time ontology index wiring in `KnowledgeGraph`**
   - In both ingestion paths (`ingest_pdf`, `ingest_markdown`), instantiate a dedicated `LanceDBIndices` with:
     - path: `self._storage.kb_path / "lancedb"` (or equivalent storage kb path)
     - config: `self._config`
     - `group_id="ontology"`
   - Initialize it and pass that index to `TopicResolver`.
   - Keep storage write path unchanged (resolved topics still written to default KB group via storage backend).

2. **Keep resolution contract stable**
   - `TopicResolver.resolve(...)` should still output `TopicResolution(uuid, canonical_label, definition, ...)`.
   - Ingestion should continue using `tr.uuid` and `tr.canonical_name` when creating `Topic` objects.

3. **Add explicit ontology/default group regression tests**
   - `tests/test_topic_resolver.py`:
     - Add a test that fails if resolver search is run against default group and passes with ontology group wiring.
   - `tests/test_query_pipeline.py`:
     - Add tests for two-stage topic resolution behavior:
       - Stage 1 ontology candidates are selected.
       - Stage 2 maps candidate names to KB topics via `get_topics_by_names`.
       - Empty KB lookup returns empty resolved topic list.

4. **(Optional but recommended) canonical-label guard**
   - Add a targeted test for synonym candidate output behavior to avoid writing synonym strings as canonical topic names by mistake.
   - If this is fixed in Ticket C, do it minimally and keep scope to resolver/topic tests only.

## Verify commands (ticket-required)

```bash
uv run pytest -q tests/test_topic_resolver.py tests/test_query_pipeline.py
uv run pytest -q
```
