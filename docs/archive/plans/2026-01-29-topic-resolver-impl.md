# Topic Resolver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement topic ontology resolution to complete Phase 3 of the ingestion pipeline.

**Architecture:** TopicResolver matches extracted topics against a curated ontology using LanceDB vector search + batched LLM verification. JSON file is source of truth, LanceDB is search index with hash-based cache invalidation.

**Tech Stack:** Python 3.10+, Pydantic, LanceDB, asyncio, pytest

---

## Task 1: Add Types to `types/topics.py`

**Files:**
- Modify: `vanna_kg/types/topics.py`
- Test: `tests/test_types.py`

**Step 1: Write failing test for TopicResolutionResult**

Add to `tests/test_types.py`:

```python
class TestTopicResolutionResult:
    """Test TopicResolutionResult structure."""

    def test_empty_result(self):
        """Empty result should have empty collections."""
        from vanna_kg.types.topics import TopicResolutionResult

        result = TopicResolutionResult()
        assert result.resolved_topics == []
        assert result.uuid_remap == {}
        assert result.new_topics == []

    def test_result_with_data(self):
        """Result should correctly store all fields."""
        from vanna_kg.types.topics import TopicResolution, TopicResolutionResult

        topic = TopicResolution(
            uuid="topic-uuid",
            canonical_label="Inflation",
            is_new=False,
            definition="A general increase in prices.",
        )
        result = TopicResolutionResult(
            resolved_topics=[topic],
            uuid_remap={"CPI": "topic-uuid"},
            new_topics=["Unknown Topic"],
        )
        assert len(result.resolved_topics) == 1
        assert result.uuid_remap["CPI"] == "topic-uuid"
        assert "Unknown Topic" in result.new_topics
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py::TestTopicResolutionResult -v`
Expected: FAIL with ImportError for TopicResolutionResult

**Step 3: Add TopicResolutionResult to types/topics.py**

Add after `BatchTopicDefinitions` class in `vanna_kg/types/topics.py`:

```python
class TopicResolutionResult(BaseModel):
    """Result of resolving topics against the ontology."""

    resolved_topics: list[TopicResolution] = Field(
        default_factory=list,
        description="Successfully resolved topics",
    )
    uuid_remap: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from extracted topic name to ontology UUID",
    )
    new_topics: list[str] = Field(
        default_factory=list,
        description="Topics not found in ontology (for review)",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py::TestTopicResolutionResult -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/types/topics.py tests/test_types.py
git commit -m "feat(types): add TopicResolutionResult"
```

---

## Task 2: Add LLM Response Types

**Files:**
- Modify: `vanna_kg/types/topics.py`
- Test: `tests/test_types.py`

**Step 1: Write failing test for TopicMatchDecision and BatchTopicMatchResponse**

Add to `tests/test_types.py`:

```python
class TestTopicMatchTypes:
    """Test topic match decision types."""

    def test_topic_match_decision_with_match(self):
        """TopicMatchDecision should store match decision."""
        from vanna_kg.types.topics import TopicMatchDecision

        decision = TopicMatchDecision(
            topic="M&A",
            selected_number=1,
            reasoning="Exact semantic match to Mergers And Acquisitions.",
        )
        assert decision.topic == "M&A"
        assert decision.selected_number == 1
        assert "semantic match" in decision.reasoning

    def test_topic_match_decision_no_match(self):
        """TopicMatchDecision should allow null for no match."""
        from vanna_kg.types.topics import TopicMatchDecision

        decision = TopicMatchDecision(
            topic="Random Noise",
            selected_number=None,
            reasoning="No candidates match this topic.",
        )
        assert decision.selected_number is None

    def test_batch_topic_match_response(self):
        """BatchTopicMatchResponse should contain list of decisions."""
        from vanna_kg.types.topics import BatchTopicMatchResponse, TopicMatchDecision

        decisions = [
            TopicMatchDecision(topic="M&A", selected_number=1, reasoning="Match."),
            TopicMatchDecision(topic="Unknown", selected_number=None, reasoning="No match."),
        ]
        response = BatchTopicMatchResponse(decisions=decisions)
        assert len(response.decisions) == 2
        assert response.decisions[0].selected_number == 1
        assert response.decisions[1].selected_number is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py::TestTopicMatchTypes -v`
Expected: FAIL with ImportError

**Step 3: Add types to topics.py**

Add after `TopicResolutionResult` in `vanna_kg/types/topics.py`:

```python
class TopicMatchDecision(BaseModel):
    """LLM decision for a single topic match."""

    topic: str = Field(description="The extracted topic name")
    selected_number: int | None = Field(
        default=None,
        description="Candidate number (1-indexed) or null if no match",
    )
    reasoning: str = Field(description="Brief explanation of decision")


class BatchTopicMatchResponse(BaseModel):
    """Batched LLM response for topic verification."""

    decisions: list[TopicMatchDecision] = Field(
        description="Match decisions for each topic in the batch"
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py::TestTopicMatchTypes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/types/topics.py tests/test_types.py
git commit -m "feat(types): add TopicMatchDecision and BatchTopicMatchResponse"
```

---

## Task 3: Copy Financial Topics JSON

**Files:**
- Create: `vanna_kg/data/topics/financial_topics.json`
- Modify: `pyproject.toml`

**Step 1: Create data directory and copy JSON**

```bash
mkdir -p vanna_kg/data/topics
cp /home/rithv/Programming/Startups/VannaLabsKG/vanna_kg/data/topics/financial_topics.json vanna_kg/data/topics/
```

**Step 2: Verify file exists and has content**

Run: `python -c "import json; data = json.load(open('vanna_kg/data/topics/financial_topics.json')); print(f'Loaded {len(data)} topics')"`
Expected: "Loaded 140 topics" (approximately)

**Step 3: Update pyproject.toml to include data files**

Add after `[tool.hatch.build.targets.wheel]` section in `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["vanna_kg"]

[tool.hatch.build.targets.sdist]
include = [
    "vanna_kg/**/*.py",
    "vanna_kg/data/**/*.json",
]

[tool.hatch.build.targets.wheel.force-include]
"vanna_kg/data" = "vanna_kg/data"
```

**Step 4: Verify package includes data**

Run: `python -c "from pathlib import Path; p = Path('vanna_kg/data/topics/financial_topics.json'); print(f'Found: {p.exists()}, Size: {p.stat().st_size} bytes')"`
Expected: Found: True, Size: ~100KB

**Step 5: Commit**

```bash
git add vanna_kg/data/topics/financial_topics.json pyproject.toml
git commit -m "feat: add financial topics ontology JSON"
```

---

## Task 4: Create TopicResolver Skeleton with Constructor

**Files:**
- Create: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for TopicResolver construction**

Create `tests/test_topic_resolver.py`:

```python
"""Tests for TopicResolver topic ontology resolution."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestTopicResolverConstruction:
    """Test TopicResolver can be constructed with dependencies."""

    def test_constructor_with_defaults(self):
        """TopicResolver should accept dependencies and use defaults."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        indices = MagicMock()
        llm = MagicMock()
        embeddings = MagicMock()

        resolver = TopicResolver(indices, llm, embeddings)

        assert resolver.indices is indices
        assert resolver.llm is llm
        assert resolver.embeddings is embeddings
        assert resolver.collect_unmatched is True
        assert resolver.batch_size == 10
        assert resolver.concurrency == 3
        assert resolver.candidate_limit == 15
        assert resolver.similarity_threshold == 0.40
        assert resolver.high_similarity_flag == 0.85
        assert resolver.ontology_group_id == "ontology"

    def test_constructor_with_custom_values(self):
        """TopicResolver should accept custom configuration."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        indices = MagicMock()
        llm = MagicMock()
        embeddings = MagicMock()

        resolver = TopicResolver(
            indices,
            llm,
            embeddings,
            collect_unmatched=False,
            batch_size=5,
            concurrency=-1,
            candidate_limit=20,
            similarity_threshold=0.50,
            high_similarity_flag=0.90,
            ontology_group_id="custom",
        )

        assert resolver.collect_unmatched is False
        assert resolver.batch_size == 5
        assert resolver.concurrency == -1
        assert resolver.candidate_limit == 20
        assert resolver.similarity_threshold == 0.50
        assert resolver.high_similarity_flag == 0.90
        assert resolver.ontology_group_id == "custom"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestTopicResolverConstruction -v`
Expected: FAIL with ImportError

**Step 3: Create topic_resolver.py with constructor**

Create `vanna_kg/ingestion/resolution/topic_resolver.py`:

```python
"""
Topic Resolver - Topic Ontology Resolution

Resolves extracted topics against a curated ontology using:
1. Vector similarity search (LanceDB)
2. Batched LLM verification

The ontology is loaded from a JSON file and cached in LanceDB.
Hash-based change detection triggers automatic reload when JSON changes.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from vanna_kg.config import KGConfig
from vanna_kg.providers.base import EmbeddingProvider, LLMProvider
from vanna_kg.storage.lancedb.indices import LanceDBIndices
from vanna_kg.types.topics import (
    BatchTopicMatchResponse,
    TopicDefinition,
    TopicResolution,
    TopicResolutionResult,
)

logger = logging.getLogger(__name__)


class TopicResolver:
    """
    Resolves extracted topics against the curated ontology.

    Usage:
        resolver = TopicResolver(lancedb_indices, llm, embeddings)
        result = await resolver.resolve(topics)

        # result.resolved_topics - matched to ontology
        # result.uuid_remap - {topic_name: ontology_uuid}
        # result.new_topics - unmatched (if collect_unmatched=True)
    """

    def __init__(
        self,
        indices: LanceDBIndices,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        config: KGConfig | None = None,
        # Behavior toggles
        collect_unmatched: bool = True,
        # Batching & concurrency
        batch_size: int = 10,
        concurrency: int = 3,  # -1 = unlimited
        # Search tuning
        candidate_limit: int = 15,
        similarity_threshold: float = 0.40,
        high_similarity_flag: float = 0.85,
        # Ontology
        ontology_group_id: str = "ontology",
    ):
        self.indices = indices
        self.llm = llm_provider
        self.embeddings = embedding_provider
        self.config = config or KGConfig()

        # Behavior toggles
        self.collect_unmatched = collect_unmatched

        # Batching & concurrency
        self.batch_size = batch_size
        self.concurrency = concurrency

        # Search tuning
        self.candidate_limit = candidate_limit
        self.similarity_threshold = similarity_threshold
        self.high_similarity_flag = high_similarity_flag

        # Ontology
        self.ontology_group_id = ontology_group_id

        # Internal state
        self._ontology_loaded = False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestTopicResolverConstruction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): add TopicResolver skeleton with constructor"
```

---

## Task 5: Implement Ontology Loading with Hash Detection

**Files:**
- Modify: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for ontology loading**

Add to `tests/test_topic_resolver.py`:

```python
class TestOntologyLoading:
    """Test ontology loading from JSON."""

    @pytest.fixture
    def small_ontology(self, tmp_path):
        """Create a small test ontology JSON."""
        ontology = [
            {
                "uri": "https://kg.vannalabs.com/topic/Inflation",
                "label": "Inflation",
                "definition": "A general increase in prices.",
                "synonyms": ["CPI", "Price Increases"],
            },
            {
                "uri": "https://kg.vannalabs.com/topic/GDP",
                "label": "GDP",
                "definition": "Gross domestic product.",
                "synonyms": ["Economic Output"],
            },
        ]
        ontology_file = tmp_path / "test_topics.json"
        ontology_file.write_text(json.dumps(ontology))
        return ontology_file

    def test_load_ontology_entries(self, small_ontology):
        """_load_ontology_entries should parse JSON correctly."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        entries = resolver._load_ontology_entries(small_ontology)

        assert len(entries) == 2
        assert entries[0]["label"] == "Inflation"
        assert "CPI" in entries[0]["synonyms"]

    def test_compute_ontology_hash(self, small_ontology):
        """_compute_ontology_hash should return consistent hash."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        hash1 = resolver._compute_ontology_hash(small_ontology)
        hash2 = resolver._compute_ontology_hash(small_ontology)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_generate_embedding_texts(self, small_ontology):
        """_generate_embedding_texts should create label:def and synonym:def pairs."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        entries = resolver._load_ontology_entries(small_ontology)
        texts, uuids = resolver._generate_embedding_texts(entries)

        # Inflation has label + 2 synonyms = 3 texts
        # GDP has label + 1 synonym = 2 texts
        assert len(texts) == 5
        assert len(uuids) == 5

        # Check format: "name: definition"
        assert "Inflation: A general increase in prices." in texts
        assert "CPI: A general increase in prices." in texts
        assert "GDP: Gross domestic product." in texts


import json
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestOntologyLoading -v`
Expected: FAIL with AttributeError

**Step 3: Implement ontology loading methods**

Add to `TopicResolver` class in `topic_resolver.py`:

```python
    def _get_ontology_path(self) -> Path:
        """Get path to the ontology JSON file."""
        # Use package data path
        import vanna_kg
        package_dir = Path(vanna_kg.__file__).parent
        return package_dir / "data" / "topics" / "financial_topics.json"

    def _get_hash_file_path(self) -> Path:
        """Get path to the ontology hash file."""
        return self.indices.path / ".ontology_hash"

    def _load_ontology_entries(self, ontology_path: Path) -> list[dict]:
        """Load ontology entries from JSON file."""
        with open(ontology_path, "r") as f:
            return json.load(f)

    def _compute_ontology_hash(self, ontology_path: Path) -> str:
        """Compute SHA256 hash of ontology file."""
        content = ontology_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _generate_embedding_texts(
        self, entries: list[dict]
    ) -> tuple[list[str], list[str]]:
        """
        Generate embedding texts for ontology entries.

        For each entry, creates:
        - "label: definition"
        - "synonym: definition" for each synonym

        Returns:
            texts: List of embedding texts
            uuids: List of corresponding UUIDs (from uri)
        """
        texts: list[str] = []
        uuids: list[str] = []

        for entry in entries:
            label = entry["label"]
            definition = entry.get("definition", "")
            uri = entry["uri"]
            # Extract UUID from URI (last segment)
            uuid = uri.split("/")[-1]
            synonyms = entry.get("synonyms", [])

            # Add label: definition
            texts.append(f"{label}: {definition}")
            uuids.append(uuid)

            # Add synonym: definition for each synonym
            for synonym in synonyms:
                texts.append(f"{synonym}: {definition}")
                uuids.append(uuid)

        return texts, uuids
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestOntologyLoading -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): add ontology loading and hash computation"
```

---

## Task 6: Implement Hash-Based Cache and Reload

**Files:**
- Modify: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for hash-based caching**

Add to `tests/test_topic_resolver.py`:

```python
class TestOntologyCache:
    """Test hash-based ontology caching."""

    @pytest.fixture
    def mock_indices(self, tmp_path):
        """Create mock indices with tmp path."""
        indices = MagicMock()
        indices.path = tmp_path
        indices.add_topics = AsyncMock()
        indices.initialize = AsyncMock()
        return indices

    @pytest.fixture
    def small_ontology_path(self, tmp_path):
        """Create a small test ontology JSON."""
        ontology = [
            {
                "uri": "https://kg.vannalabs.com/topic/Inflation",
                "label": "Inflation",
                "definition": "A general increase in prices.",
                "synonyms": [],
            },
        ]
        ontology_file = tmp_path / "test_topics.json"
        ontology_file.write_text(json.dumps(ontology))
        return ontology_file

    def test_should_reload_no_hash_file(self, mock_indices, small_ontology_path):
        """Should reload when no hash file exists."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.indices = mock_indices

        assert resolver._should_reload_ontology(small_ontology_path) is True

    def test_should_reload_hash_matches(self, mock_indices, small_ontology_path):
        """Should not reload when hash matches."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.indices = mock_indices

        # Write matching hash
        current_hash = resolver._compute_ontology_hash(small_ontology_path)
        hash_file = mock_indices.path / ".ontology_hash"
        hash_file.write_text(json.dumps({"hash": current_hash}))

        assert resolver._should_reload_ontology(small_ontology_path) is False

    def test_should_reload_hash_differs(self, mock_indices, small_ontology_path):
        """Should reload when hash differs."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.indices = mock_indices

        # Write different hash
        hash_file = mock_indices.path / ".ontology_hash"
        hash_file.write_text(json.dumps({"hash": "different_hash"}))

        assert resolver._should_reload_ontology(small_ontology_path) is True

    def test_save_ontology_hash(self, mock_indices, small_ontology_path):
        """_save_ontology_hash should persist hash to file."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.indices = mock_indices

        test_hash = "abc123"
        resolver._save_ontology_hash(test_hash)

        hash_file = mock_indices.path / ".ontology_hash"
        assert hash_file.exists()
        data = json.loads(hash_file.read_text())
        assert data["hash"] == test_hash
        assert "loaded_at" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestOntologyCache -v`
Expected: FAIL with AttributeError

**Step 3: Implement caching methods**

Add to `TopicResolver` class:

```python
    def _should_reload_ontology(self, ontology_path: Path) -> bool:
        """Check if ontology needs reloading based on hash comparison."""
        hash_file = self._get_hash_file_path()

        if not hash_file.exists():
            return True

        try:
            stored = json.loads(hash_file.read_text())
            stored_hash = stored.get("hash", "")
            current_hash = self._compute_ontology_hash(ontology_path)
            return stored_hash != current_hash
        except (json.JSONDecodeError, KeyError):
            return True

    def _save_ontology_hash(self, hash_value: str) -> None:
        """Save ontology hash to file."""
        hash_file = self._get_hash_file_path()
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hash": hash_value,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }
        hash_file.write_text(json.dumps(data))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestOntologyCache -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): add hash-based ontology caching"
```

---

## Task 7: Implement _ensure_ontology_loaded and reload_ontology

**Files:**
- Modify: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for ensure_ontology_loaded**

Add to `tests/test_topic_resolver.py`:

```python
class TestEnsureOntologyLoaded:
    """Test lazy ontology loading."""

    @pytest.fixture
    def mock_resolver_deps(self, tmp_path):
        """Create mock dependencies for resolver."""
        indices = MagicMock()
        indices.path = tmp_path
        indices.add_topics = AsyncMock()
        indices.initialize = AsyncMock()

        llm = MagicMock()
        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=[[0.1] * 3072])

        return indices, llm, embeddings

    @pytest.fixture
    def test_ontology(self, tmp_path):
        """Create test ontology file."""
        ontology = [
            {
                "uri": "https://kg.vannalabs.com/topic/Test",
                "label": "Test Topic",
                "definition": "A test topic.",
                "synonyms": [],
            },
        ]
        ontology_file = tmp_path / "topics.json"
        ontology_file.write_text(json.dumps(ontology))
        return ontology_file

    @pytest.mark.asyncio
    async def test_ensure_loads_on_first_call(
        self, mock_resolver_deps, test_ontology, monkeypatch
    ):
        """_ensure_ontology_loaded should load on first call."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        indices, llm, embeddings = mock_resolver_deps
        resolver = TopicResolver(indices, llm, embeddings)

        # Patch to use test ontology
        monkeypatch.setattr(resolver, "_get_ontology_path", lambda: test_ontology)

        assert resolver._ontology_loaded is False

        await resolver._ensure_ontology_loaded()

        assert resolver._ontology_loaded is True
        indices.add_topics.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_skips_when_loaded(
        self, mock_resolver_deps, test_ontology, monkeypatch
    ):
        """_ensure_ontology_loaded should skip if already loaded and hash matches."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        indices, llm, embeddings = mock_resolver_deps
        resolver = TopicResolver(indices, llm, embeddings)
        monkeypatch.setattr(resolver, "_get_ontology_path", lambda: test_ontology)

        # First call loads
        await resolver._ensure_ontology_loaded()
        first_call_count = indices.add_topics.call_count

        # Second call should skip
        await resolver._ensure_ontology_loaded()
        assert indices.add_topics.call_count == first_call_count

    @pytest.mark.asyncio
    async def test_reload_ontology_forces_reload(
        self, mock_resolver_deps, test_ontology, monkeypatch
    ):
        """reload_ontology should reload even when hash matches."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        indices, llm, embeddings = mock_resolver_deps
        resolver = TopicResolver(indices, llm, embeddings)
        monkeypatch.setattr(resolver, "_get_ontology_path", lambda: test_ontology)

        # First load
        await resolver._ensure_ontology_loaded()
        first_call_count = indices.add_topics.call_count

        # Force reload
        await resolver.reload_ontology()
        assert indices.add_topics.call_count == first_call_count + 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestEnsureOntologyLoaded -v`
Expected: FAIL with AttributeError

**Step 3: Implement _ensure_ontology_loaded and reload_ontology**

Add to `TopicResolver` class:

```python
    async def _ensure_ontology_loaded(self) -> None:
        """Lazy load ontology with hash-based change detection."""
        ontology_path = self._get_ontology_path()

        # Skip if already loaded and hash matches
        if self._ontology_loaded and not self._should_reload_ontology(ontology_path):
            return

        await self._load_ontology(ontology_path)

    async def reload_ontology(self) -> None:
        """Force reload ontology from JSON, ignoring hash."""
        ontology_path = self._get_ontology_path()
        await self._load_ontology(ontology_path)

    async def _load_ontology(self, ontology_path: Path) -> None:
        """Load ontology into LanceDB."""
        logger.info(f"Loading topic ontology from {ontology_path}")

        # Load entries
        entries = self._load_ontology_entries(ontology_path)

        # Generate embedding texts
        texts, uuids = self._generate_embedding_texts(entries)

        # Generate embeddings
        embeddings_list = await self.embeddings.embed(texts)

        # Build topic dicts for LanceDB
        # Create lookup for definitions
        entry_lookup = {e["uri"].split("/")[-1]: e for e in entries}

        topics = []
        for text, uuid in zip(texts, uuids):
            entry = entry_lookup[uuid]
            topics.append({
                "uuid": uuid,
                "name": text.split(":")[0],  # Extract name from "name: def"
                "definition": entry.get("definition", ""),
                "group_id": self.ontology_group_id,
            })

        # Write to LanceDB
        await self.indices.add_topics(topics, embeddings_list)

        # Save hash
        current_hash = self._compute_ontology_hash(ontology_path)
        self._save_ontology_hash(current_hash)

        self._ontology_loaded = True
        logger.info(f"Loaded {len(entries)} topics ({len(texts)} embeddings)")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestEnsureOntologyLoaded -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): add lazy ontology loading with hash detection"
```

---

## Task 8: Implement Topic-to-Text Helper

**Files:**
- Modify: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for _topic_to_text**

Add to `tests/test_topic_resolver.py`:

```python
class TestTopicToText:
    """Test topic text generation for embeddings."""

    def test_topic_to_text_format(self):
        """Text format should be 'topic: definition'."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import TopicDefinition

        topic = TopicDefinition(
            topic="M&A",
            definition="Corporate mergers and acquisitions activity.",
        )

        resolver = TopicResolver.__new__(TopicResolver)
        actual = resolver._topic_to_text(topic)
        expected = "M&A: Corporate mergers and acquisitions activity."
        assert actual == expected

    def test_topic_to_text_empty_definition(self):
        """Empty definition should still produce valid text."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import TopicDefinition

        topic = TopicDefinition(topic="Unknown", definition="")

        resolver = TopicResolver.__new__(TopicResolver)
        actual = resolver._topic_to_text(topic)
        assert actual == "Unknown: "
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestTopicToText -v`
Expected: FAIL with AttributeError

**Step 3: Implement _topic_to_text**

Add to `TopicResolver` class:

```python
    def _topic_to_text(self, topic: TopicDefinition) -> str:
        """Convert topic to embedding text: '{topic}: {definition}'."""
        return f"{topic.topic}: {topic.definition}"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestTopicToText -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): add _topic_to_text helper"
```

---

## Task 9: Implement Batch LLM Verification Prompt Builder

**Files:**
- Modify: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for _build_batch_verification_prompt**

Add to `tests/test_topic_resolver.py`:

```python
class TestBatchVerificationPrompt:
    """Test batch verification prompt building."""

    def test_prompt_contains_all_topics(self):
        """Prompt should list all topics with their candidates."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import TopicDefinition

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.high_similarity_flag = 0.85

        topics = [
            TopicDefinition(topic="M&A", definition="Mergers and acquisitions."),
            TopicDefinition(topic="Fed Policy", definition="Federal Reserve policy."),
        ]
        candidates_list = [
            [
                ({"uuid": "uuid-1", "name": "Mergers And Acquisitions", "definition": "Corporate transactions."}, 0.89),
                ({"uuid": "uuid-2", "name": "Antitrust", "definition": "Competition law."}, 0.45),
            ],
            [
                ({"uuid": "uuid-3", "name": "Monetary Policy", "definition": "Central bank actions."}, 0.92),
            ],
        ]

        prompt = resolver._build_batch_verification_prompt(topics, candidates_list)

        # Check topics are listed
        assert "M&A" in prompt
        assert "Mergers and acquisitions" in prompt
        assert "Fed Policy" in prompt
        assert "Federal Reserve policy" in prompt

        # Check candidates are listed
        assert "Mergers And Acquisitions" in prompt
        assert "89%" in prompt
        assert "Monetary Policy" in prompt
        assert "92%" in prompt

    def test_prompt_flags_high_similarity(self):
        """High similarity candidates should be flagged."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import TopicDefinition

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.high_similarity_flag = 0.85

        topics = [TopicDefinition(topic="Test", definition="Test topic.")]
        candidates_list = [
            [({"uuid": "uuid-1", "name": "Test Topic", "definition": "A test."}, 0.95)],
        ]

        prompt = resolver._build_batch_verification_prompt(topics, candidates_list)

        assert "[LIKELY MATCH]" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestBatchVerificationPrompt -v`
Expected: FAIL with AttributeError

**Step 3: Implement _build_batch_verification_prompt**

Add to `TopicResolver` class:

```python
    def _build_batch_verification_prompt(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
    ) -> str:
        """Build prompt for batch topic verification."""
        lines = [
            "Match each extracted topic to its best candidate from the ontology.",
            "Return the candidate number (1-indexed) or null if no good match.",
            "",
            "TOPICS TO VERIFY:",
            "",
        ]

        for i, (topic, candidates) in enumerate(zip(topics, candidates_list), 1):
            lines.append(f"{i}. Extracted: \"{topic.topic}\"")
            lines.append(f"   Definition: {topic.definition}")
            lines.append("   Candidates:")

            if not candidates:
                lines.append("   (no candidates found)")
            else:
                for j, (cand, score) in enumerate(candidates, 1):
                    pct = int(score * 100)
                    flag = " [LIKELY MATCH]" if score >= self.high_similarity_flag else ""
                    lines.append(
                        f"   {j}. {cand['name']} ({pct}% similar){flag}"
                    )
                    lines.append(f"      Definition: {cand['definition']}")

            lines.append("")

        lines.extend([
            "RULES:",
            "- Match if the extracted topic and candidate refer to the same concept",
            "- Different names are OK if meaning is the same (e.g., 'M&A' = 'Mergers And Acquisitions')",
            "- Return null if no candidate is a good semantic match",
            "",
            "Return a decision for each topic.",
        ])

        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestBatchVerificationPrompt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): add batch verification prompt builder"
```

---

## Task 10: Implement Batch LLM Verification

**Files:**
- Modify: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for _verify_batch**

Add to `tests/test_topic_resolver.py`:

```python
class TestVerifyBatch:
    """Test batch LLM verification."""

    @pytest.mark.asyncio
    async def test_verify_batch_returns_decisions(self):
        """_verify_batch should return decisions for all topics."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        llm = AsyncMock()
        llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="M&A", selected_number=1, reasoning="Match."),
                    TopicMatchDecision(topic="Unknown", selected_number=None, reasoning="No match."),
                ]
            )
        )

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.llm = llm
        resolver.high_similarity_flag = 0.85

        topics = [
            TopicDefinition(topic="M&A", definition="Mergers."),
            TopicDefinition(topic="Unknown", definition="Unknown topic."),
        ]
        candidates_list = [
            [({"uuid": "uuid-1", "name": "Mergers And Acquisitions", "definition": "Transactions."}, 0.89)],
            [],
        ]

        decisions = await resolver._verify_batch(topics, candidates_list)

        assert len(decisions) == 2
        assert decisions[0].selected_number == 1
        assert decisions[1].selected_number is None

    @pytest.mark.asyncio
    async def test_verify_batch_handles_llm_failure(self):
        """_verify_batch should return no-match decisions on LLM failure."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import TopicDefinition

        llm = AsyncMock()
        llm.generate_structured = AsyncMock(side_effect=Exception("LLM error"))

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.llm = llm
        resolver.high_similarity_flag = 0.85

        topics = [TopicDefinition(topic="Test", definition="Test.")]
        candidates_list = [[({"uuid": "u1", "name": "Test", "definition": "Test."}, 0.9)]]

        decisions = await resolver._verify_batch(topics, candidates_list)

        assert len(decisions) == 1
        assert decisions[0].selected_number is None
        assert "error" in decisions[0].reasoning.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestVerifyBatch -v`
Expected: FAIL with AttributeError

**Step 3: Implement _verify_batch**

Add to `TopicResolver` class:

```python
    async def _verify_batch(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
    ) -> list[TopicMatchDecision]:
        """Verify a batch of topics using LLM."""
        from vanna_kg.types.topics import TopicMatchDecision

        prompt = self._build_batch_verification_prompt(topics, candidates_list)

        system = (
            "You are a topic resolution expert. Match extracted topics to their "
            "canonical forms in a curated ontology. Be precise - only match if "
            "the concepts are semantically equivalent."
        )

        try:
            response = await self.llm.generate_structured(
                prompt, BatchTopicMatchResponse, system=system
            )
            return response.decisions
        except Exception as e:
            logger.warning(f"Batch LLM verification failed: {e}")
            # Return no-match decisions for all topics
            return [
                TopicMatchDecision(
                    topic=t.topic,
                    selected_number=None,
                    reasoning=f"LLM verification error: {e}",
                )
                for t in topics
            ]
```

Also add import at top of file:

```python
from vanna_kg.types.topics import (
    BatchTopicMatchResponse,
    TopicDefinition,
    TopicMatchDecision,
    TopicResolution,
    TopicResolutionResult,
)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestVerifyBatch -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): add batch LLM verification"
```

---

## Task 11: Implement Main resolve() Method

**Files:**
- Modify: `vanna_kg/ingestion/resolution/topic_resolver.py`
- Test: `tests/test_topic_resolver.py`

**Step 1: Write failing test for resolve()**

Add to `tests/test_topic_resolver.py`:

```python
class TestResolve:
    """Test main resolve() method."""

    @pytest.fixture
    def mock_resolver(self, tmp_path):
        """Create resolver with all mocks."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicMatchDecision,
        )

        indices = MagicMock()
        indices.path = tmp_path
        indices.add_topics = AsyncMock()
        indices.search_topics = AsyncMock(return_value=[])

        llm = AsyncMock()
        llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(decisions=[])
        )

        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=[[0.1] * 3072])

        resolver = TopicResolver(indices, llm, embeddings)
        resolver._ontology_loaded = True  # Skip loading for tests

        return resolver

    @pytest.mark.asyncio
    async def test_resolve_empty_list(self, mock_resolver):
        """Empty topic list returns empty result."""
        result = await mock_resolver.resolve([])

        assert result.resolved_topics == []
        assert result.uuid_remap == {}
        assert result.new_topics == []

    @pytest.mark.asyncio
    async def test_resolve_with_match(self, mock_resolver):
        """Matching topic should be in resolved_topics and uuid_remap."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        # Set up search to return a candidate
        mock_resolver.indices.search_topics = AsyncMock(
            return_value=[
                ({"uuid": "inflation-uuid", "name": "Inflation", "definition": "Price increases."}, 0.89),
            ]
        )

        # Set up LLM to match
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="CPI", selected_number=1, reasoning="Match."),
                ]
            )
        )

        topics = [TopicDefinition(topic="CPI", definition="Consumer price index.")]
        result = await mock_resolver.resolve(topics)

        assert len(result.resolved_topics) == 1
        assert result.resolved_topics[0].uuid == "inflation-uuid"
        assert result.resolved_topics[0].canonical_label == "Inflation"
        assert result.uuid_remap["CPI"] == "inflation-uuid"
        assert result.new_topics == []

    @pytest.mark.asyncio
    async def test_resolve_no_match_collected(self, mock_resolver):
        """Unmatched topic should be in new_topics when collect_unmatched=True."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        mock_resolver.collect_unmatched = True
        mock_resolver.indices.search_topics = AsyncMock(return_value=[])
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="Unknown", selected_number=None, reasoning="No match."),
                ]
            )
        )

        topics = [TopicDefinition(topic="Unknown", definition="Unknown topic.")]
        result = await mock_resolver.resolve(topics)

        assert result.resolved_topics == []
        assert result.uuid_remap == {}
        assert "Unknown" in result.new_topics

    @pytest.mark.asyncio
    async def test_resolve_no_match_not_collected(self, mock_resolver):
        """Unmatched topic should NOT be in new_topics when collect_unmatched=False."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        mock_resolver.collect_unmatched = False
        mock_resolver.indices.search_topics = AsyncMock(return_value=[])
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="Unknown", selected_number=None, reasoning="No match."),
                ]
            )
        )

        topics = [TopicDefinition(topic="Unknown", definition="Unknown topic.")]
        result = await mock_resolver.resolve(topics)

        assert result.new_topics == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_resolver.py::TestResolve -v`
Expected: FAIL with AttributeError

**Step 3: Implement resolve()**

Add to `TopicResolver` class:

```python
    async def resolve(
        self,
        topics: list[TopicDefinition],
        embeddings: list[list[float]] | None = None,
    ) -> TopicResolutionResult:
        """
        Resolve topics against the ontology.

        Args:
            topics: Topics from extraction (with definitions)
            embeddings: Pre-computed embeddings, or None to generate

        Returns:
            TopicResolutionResult with resolved_topics, uuid_remap, new_topics
        """
        if not topics:
            return TopicResolutionResult()

        # Ensure ontology is loaded
        await self._ensure_ontology_loaded()

        # Generate embeddings if not provided
        if embeddings is None:
            texts = [self._topic_to_text(t) for t in topics]
            embeddings = await self.embeddings.embed(texts)

        # Search for candidates for each topic
        candidates_list: list[list[tuple[dict, float]]] = []
        for embedding in embeddings:
            candidates = await self.indices.search_topics(
                embedding,
                limit=self.candidate_limit,
                threshold=self.similarity_threshold,
            )
            # Filter to only ontology group
            candidates = [
                (c, s) for c, s in candidates
                if c.get("group_id", self.ontology_group_id) == self.ontology_group_id
            ]
            candidates_list.append(candidates)

        # Batch verification with concurrency
        all_decisions = await self._verify_all_batches(topics, candidates_list)

        # Build result
        return self._build_result(topics, candidates_list, all_decisions)

    async def _verify_all_batches(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
    ) -> list[TopicMatchDecision]:
        """Verify all topics in batches with concurrency control."""
        from vanna_kg.types.topics import TopicMatchDecision

        # Split into batches
        batches: list[tuple[list[TopicDefinition], list[list[tuple[dict, float]]]]] = []
        for i in range(0, len(topics), self.batch_size):
            batch_topics = topics[i : i + self.batch_size]
            batch_candidates = candidates_list[i : i + self.batch_size]
            batches.append((batch_topics, batch_candidates))

        # Set up concurrency
        if self.concurrency == -1:
            # Unlimited parallelism
            tasks = [self._verify_batch(bt, bc) for bt, bc in batches]
            batch_results = await asyncio.gather(*tasks)
        else:
            # Bounded concurrency
            semaphore = asyncio.Semaphore(self.concurrency)

            async def bounded_verify(
                bt: list[TopicDefinition], bc: list[list[tuple[dict, float]]]
            ) -> list[TopicMatchDecision]:
                async with semaphore:
                    return await self._verify_batch(bt, bc)

            tasks = [bounded_verify(bt, bc) for bt, bc in batches]
            batch_results = await asyncio.gather(*tasks)

        # Flatten results
        all_decisions: list[TopicMatchDecision] = []
        for decisions in batch_results:
            all_decisions.extend(decisions)

        return all_decisions

    def _build_result(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
        decisions: list[TopicMatchDecision],
    ) -> TopicResolutionResult:
        """Build TopicResolutionResult from decisions."""
        resolved_topics: list[TopicResolution] = []
        uuid_remap: dict[str, str] = {}
        new_topics: list[str] = []

        for topic, candidates, decision in zip(topics, candidates_list, decisions):
            if decision.selected_number is not None and candidates:
                # Match found
                idx = decision.selected_number - 1  # Convert to 0-indexed
                if 0 <= idx < len(candidates):
                    matched = candidates[idx][0]
                    resolution = TopicResolution(
                        uuid=matched["uuid"],
                        canonical_label=matched["name"],
                        is_new=False,
                        definition=matched.get("definition", ""),
                    )
                    resolved_topics.append(resolution)
                    uuid_remap[topic.topic] = matched["uuid"]
                    continue

            # No match
            if self.collect_unmatched:
                new_topics.append(topic.topic)

        return TopicResolutionResult(
            resolved_topics=resolved_topics,
            uuid_remap=uuid_remap,
            new_topics=new_topics,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_resolver.py::TestResolve -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vanna_kg/ingestion/resolution/topic_resolver.py tests/test_topic_resolver.py
git commit -m "feat(topic_resolver): implement main resolve() method"
```

---

## Task 12: Export TopicResolver and Update __init__.py

**Files:**
- Modify: `vanna_kg/ingestion/resolution/__init__.py`
- Modify: `vanna_kg/types/__init__.py` (if needed)

**Step 1: Update resolution __init__.py**

Edit `vanna_kg/ingestion/resolution/__init__.py`:

```python
"""
Entity and Topic Resolution

Three levels of deduplication to ensure clean knowledge graph:

Modules:
    entity_dedup: In-document deduplication (Phase 2a-c)
    entity_registry: Cross-document entity resolution (Phase 2d)
    topic_resolver: Topic ontology resolution (Phase 2e)

In-Document Deduplication (Phase 2a-c):
    1. Generate embeddings for all entities
    2. Build similarity matrix (cosine similarity)
    3. Find connected components via Union-Find
    4. LLM verification of clusters

Cross-Document Resolution (Phase 2d):
    - Vector search for candidates in existing KB
    - LLM verification of matches
    - UUID reuse for matches, summary merging

Topic Resolution (Phase 2e):
    - Vector search against topic ontology
    - LLM verification of semantic matches

Key Principle: Subsidiary Awareness
    AWS != Amazon (subsidiaries are separate entities)

See: docs/pipeline/DEDUPLICATION_SYSTEM.md
"""

from vanna_kg.ingestion.resolution.entity_dedup import deduplicate_entities
from vanna_kg.ingestion.resolution.entity_registry import EntityRegistry
from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

__all__ = ["deduplicate_entities", "EntityRegistry", "TopicResolver"]
```

**Step 2: Verify import works**

Run: `python -c "from vanna_kg.ingestion.resolution import TopicResolver; print('Import successful')"`
Expected: "Import successful"

**Step 3: Commit**

```bash
git add vanna_kg/ingestion/resolution/__init__.py
git commit -m "feat(resolution): export TopicResolver from module"
```

---

## Task 13: Run Full Test Suite

**Step 1: Run all topic resolver tests**

Run: `pytest tests/test_topic_resolver.py -v`
Expected: All tests PASS

**Step 2: Run full test suite to ensure no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 3: Run type checking**

Run: `mypy vanna_kg/ingestion/resolution/topic_resolver.py --ignore-missing-imports`
Expected: No errors (or only minor ones)

**Step 4: Run linting**

Run: `ruff check vanna_kg/ingestion/resolution/topic_resolver.py`
Expected: No errors

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(topic_resolver): complete topic ontology resolution

Implements Phase 3.3e of development plan:
- TopicResolver with lazy ontology loading
- Hash-based cache invalidation with reload_ontology()
- Batched LLM verification with configurable concurrency
- All thresholds tunable via constructor

Closes Phase 3 of ingestion pipeline."
```

---

## Summary

**Total Tasks:** 13
**Files Created:**
- `vanna_kg/data/topics/financial_topics.json`
- `vanna_kg/ingestion/resolution/topic_resolver.py`
- `tests/test_topic_resolver.py`

**Files Modified:**
- `vanna_kg/types/topics.py`
- `vanna_kg/ingestion/resolution/__init__.py`
- `pyproject.toml`

**Key Test Commands:**
- Single test: `pytest tests/test_topic_resolver.py::TestClassName::test_name -v`
- All topic resolver tests: `pytest tests/test_topic_resolver.py -v`
- Full suite: `pytest tests/ -v`
