# Entity Definition Field Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `definition` field to entities that captures stable meaning (what the entity IS), separate from `summary` (what was found about it), and use definition for deduplication embeddings.

**Architecture:** The extraction prompt will instruct the LLM to output both `definition` (stable, reusable across chunks) and `summary` (context-specific findings). Deduplication will embed `{name}: {definition}` instead of `{name}: {summary}`, ensuring entities with the same name get similar embeddings regardless of chunk-specific context.

**Tech Stack:** Python, Pydantic, pytest

---

## Task 1: Add `definition` Field to EnumeratedEntity

**Files:**
- Modify: `vanna_kg/types/entities.py:71-87`
- Test: `tests/test_types.py`

**Step 1: Write the failing test**

Add to `tests/test_types.py`:

```python
def test_enumerated_entity_has_definition():
    """EnumeratedEntity should have both definition and summary fields."""
    from vanna_kg.types.entities import EnumeratedEntity

    entity = EnumeratedEntity(
        name="Federal Reserve",
        entity_type="Organization",
        definition="The central banking system of the United States",
        summary="Reported concern about inflation in October 2025",
    )
    assert entity.definition == "The central banking system of the United States"
    assert entity.summary == "Reported concern about inflation in October 2025"
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_types.py::test_enumerated_entity_has_definition -v`
Expected: FAIL with "unexpected keyword argument 'definition'"

**Step 3: Write minimal implementation**

Edit `vanna_kg/types/entities.py`, update `EnumeratedEntity` class:

```python
class EnumeratedEntity(BaseModel):
    """
    An entity discovered during the enumeration step of chain-of-thought extraction.

    This is the first pass output before relationship generation.
    """

    name: str = Field(..., description="Entity name as it appears in the text")
    entity_type: EntityTypeLabel = Field(
        ...,
        description="Entity type: Company, Person, Organization, Location, Product, or Topic",
    )
    definition: str = Field(
        default="",
        description="Stable definition: what the entity IS (e.g., 'US central bank'). Should be the same across all mentions.",
    )
    summary: str = Field(
        default="",
        description="Context-specific findings: what was learned about this entity in this chunk",
    )
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_types.py::test_enumerated_entity_has_definition -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/types/entities.py tests/test_types.py
git commit -m "$(cat <<'EOF'
feat(types): add definition field to EnumeratedEntity

Separate stable entity definition from context-specific summary.
Definition captures what the entity IS (stable across chunks).
Summary captures what was found about it (varies per chunk).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update Extraction Prompt to Output Definition + Summary

**Files:**
- Modify: `vanna_kg/ingestion/extraction/extractor.py:31-84`

**Step 1: Update the extraction system prompt**

Replace lines 65-72 in `_EXTRACTION_SYSTEM_PROMPT`:

```python
## Rules for Entities
- Use clean names only - no parenthetical descriptions
- Keep subsidiaries separate from parent companies
- Include entities from section headers (the ">" denotes hierarchy)
- **Definition vs Summary** (IMPORTANT):
  - **definition**: What the entity IS - stable across all documents
    - Good: "Central banking system of the United States"
    - Bad: "Reported inflation concerns" (that's context-specific)
  - **summary**: What was found about this entity IN THIS CHUNK
    - Good: "Reported concern about inflation in Q3 2024"
    - Bad: "A central bank" (that's the definition)
  - Example for "Federal Reserve" in a Beige Book:
    - definition: "Central banking system of the United States"
    - summary: "Expressed cautious optimism about economic outlook"
```

**Step 2: Run existing extraction tests**

Run: `source .venv/bin/activate && pytest tests/ -k extraction -v`
Expected: PASS (prompt change doesn't break existing tests)

**Step 3: Commit**

```bash
git add vanna_kg/ingestion/extraction/extractor.py
git commit -m "$(cat <<'EOF'
feat(extraction): update prompt to extract definition + summary

Instructs LLM to separate:
- definition: stable meaning (what entity IS)
- summary: context-specific findings (what was learned)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update Deduplication to Use Definition for Embeddings

**Files:**
- Modify: `vanna_kg/ingestion/resolution/entity_dedup.py:43-54`
- Test: `tests/test_entity_dedup.py`

**Step 1: Write the failing test**

Add to `tests/test_entity_dedup.py`:

```python
def test_embedding_text_uses_definition():
    """Embedding text should use definition, not summary."""
    from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text
    from vanna_kg.types.entities import EnumeratedEntity

    entity = EnumeratedEntity(
        name="Federal Reserve",
        entity_type="Organization",
        definition="Central banking system of the United States",
        summary="Reported inflation concerns in October 2025",
    )
    text = _embedding_text(entity)
    # Should use definition, not summary
    assert text == "Federal Reserve: Central banking system of the United States"
    assert "inflation" not in text  # Summary content should NOT be in embedding


def test_embedding_text_falls_back_to_summary_if_no_definition():
    """If no definition, fall back to summary for backwards compatibility."""
    from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text
    from vanna_kg.types.entities import EnumeratedEntity

    entity = EnumeratedEntity(
        name="Apple Inc.",
        entity_type="Company",
        definition="",  # No definition
        summary="Technology company known for iPhone",
    )
    text = _embedding_text(entity)
    assert text == "Apple Inc.: Technology company known for iPhone"
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_entity_dedup.py::test_embedding_text_uses_definition -v`
Expected: FAIL (currently uses summary)

**Step 3: Write minimal implementation**

Update `_embedding_text` in `vanna_kg/ingestion/resolution/entity_dedup.py`:

```python
def _embedding_text(entity: EnumeratedEntity) -> str:
    """
    Generate embedding text for an entity.

    Format: "{name}: {definition}" if definition exists,
    else "{name}: {summary}" if summary exists,
    else just "{name}".

    Uses definition (stable meaning) rather than summary (context-specific)
    to ensure entities with the same name get similar embeddings.
    """
    # Prefer definition (stable) over summary (context-specific)
    definition = entity.definition.strip() if entity.definition else ""
    if definition:
        return f"{entity.name}: {definition}"

    # Fall back to summary for backwards compatibility
    summary = entity.summary.strip() if entity.summary else ""
    if summary:
        return f"{entity.name}: {summary}"

    return entity.name
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_entity_dedup.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/entity_dedup.py tests/test_entity_dedup.py
git commit -m "$(cat <<'EOF'
feat(dedup): use definition instead of summary for embeddings

Definition is stable across chunks, so entities with the same name
will have similar embeddings regardless of context-specific findings.
Falls back to summary for backwards compatibility.

Fixes entity deduplication bug where "Beige Book" appeared 4 times.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update Existing Dedup Tests

**Files:**
- Modify: `tests/test_entity_dedup.py:14-48`

**Step 1: Update existing tests to include definition field**

Update the existing `TestEmbeddingTextGeneration` tests to use both fields:

```python
class TestEmbeddingTextGeneration:
    """Tests for _embedding_text helper."""

    def test_embedding_text_with_definition(self):
        """Embedding text prefers definition over summary."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple Inc.",
            entity_type="Company",
            definition="American multinational technology company",
            summary="Reported strong Q3 earnings",
        )
        text = _embedding_text(entity)
        assert text == "Apple Inc.: American multinational technology company"

    def test_embedding_text_falls_back_to_summary(self):
        """Embedding text falls back to summary if no definition."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple Inc.",
            entity_type="Company",
            definition="",
            summary="Technology company known for iPhone",
        )
        text = _embedding_text(entity)
        assert text == "Apple Inc.: Technology company known for iPhone"

    def test_embedding_text_without_definition_or_summary(self):
        """Embedding text falls back to name only."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="AAPL",
            entity_type="Company",
            definition="",
            summary="",
        )
        text = _embedding_text(entity)
        assert text == "AAPL"

    def test_embedding_text_whitespace_definition(self):
        """Whitespace-only definition treated as empty, falls back to summary."""
        from vanna_kg.ingestion.resolution.entity_dedup import _embedding_text

        entity = EnumeratedEntity(
            name="Apple",
            entity_type="Company",
            definition="   ",
            summary="Tech company",
        )
        text = _embedding_text(entity)
        assert text == "Apple: Tech company"
```

**Step 2: Run all dedup tests**

Run: `source .venv/bin/activate && pytest tests/test_entity_dedup.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_entity_dedup.py
git commit -m "$(cat <<'EOF'
test(dedup): update tests for definition-first embedding

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: End-to-End Verification

**Files:**
- None (verification only)

**Step 1: Run the full test suite**

Run: `source .venv/bin/activate && pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Build a fresh test KB and verify no duplicates**

Run:
```bash
source .venv/bin/activate
rm -rf ./test_kb
python scripts/build_kg.py test_data/BeigeBook_20251015.md --chunks 10 --output ./test_kb
```

Expected: Ingestion completes successfully

**Step 3: Verify no duplicate entities**

Run:
```bash
source .venv/bin/activate && python -c "
import asyncio
from collections import Counter
from vanna_kg import KnowledgeGraph

async def main():
    kg = KnowledgeGraph('./test_kb')
    await kg._ensure_initialized()
    entities = await kg.get_entities(limit=200)

    names = [e.name for e in entities]
    counts = Counter(names)
    duplicates = [(n, c) for n, c in counts.items() if c > 1]

    if duplicates:
        print('FAIL: Duplicate entities found:')
        for name, count in duplicates:
            print(f'  - \"{name}\" appears {count} times')
    else:
        print('SUCCESS: No duplicate entity names!')

    await kg.close()

asyncio.run(main())
"
```

Expected: "SUCCESS: No duplicate entity names!"

**Step 4: Final commit (if all verification passes)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: verify entity definition field implementation

All tests pass, no duplicate entities in test KB.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Add `definition` field to `EnumeratedEntity` | `types/entities.py`, `tests/test_types.py` |
| 2 | Update extraction prompt | `ingestion/extraction/extractor.py` |
| 3 | Use definition for dedup embeddings | `ingestion/resolution/entity_dedup.py` |
| 4 | Update existing dedup tests | `tests/test_entity_dedup.py` |
| 5 | End-to-end verification | (none - verification only) |
