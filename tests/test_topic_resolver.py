"""Tests for TopicResolver topic ontology resolution."""

import json
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

    @pytest.mark.asyncio
    async def test_resolve_with_decision_count_mismatch(self, mock_resolver):
        """Resolver should handle when LLM returns fewer decisions than topics."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        # Set up search to return candidates for both topics
        mock_resolver.indices.search_topics = AsyncMock(
            return_value=[
                ({"uuid": "uuid-1", "name": "Topic A", "definition": "Def A.", "group_id": "ontology"}, 0.85),
            ]
        )

        # LLM returns only one decision for two topics
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="TopicA", selected_number=1, reasoning="Match."),
                    # Missing decision for TopicB
                ]
            )
        )
        mock_resolver.embeddings.embed = AsyncMock(return_value=[[0.1] * 3072, [0.2] * 3072])

        topics = [
            TopicDefinition(topic="TopicA", definition="Topic A definition."),
            TopicDefinition(topic="TopicB", definition="Topic B definition."),
        ]
        result = await mock_resolver.resolve(topics)

        # TopicA should match, TopicB should be unmatched (no decision found)
        assert len(result.resolved_topics) == 1
        assert result.uuid_remap.get("TopicA") == "uuid-1"
        assert "TopicB" in result.new_topics

    @pytest.mark.asyncio
    async def test_resolve_with_invalid_candidate_index(self, mock_resolver):
        """Resolver should handle when LLM returns invalid candidate index."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        # Set up search to return one candidate
        mock_resolver.indices.search_topics = AsyncMock(
            return_value=[
                ({"uuid": "uuid-1", "name": "Only Candidate", "definition": "Def.", "group_id": "ontology"}, 0.85),
            ]
        )

        # LLM returns invalid index (5 when only 1 candidate exists)
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="Test", selected_number=5, reasoning="Picked #5."),
                ]
            )
        )

        topics = [TopicDefinition(topic="Test", definition="Test topic.")]
        result = await mock_resolver.resolve(topics)

        # Should treat as unmatched due to invalid index
        assert result.resolved_topics == []
        assert result.uuid_remap == {}
        assert "Test" in result.new_topics

    @pytest.mark.asyncio
    async def test_resolve_filters_by_ontology_group_id(self, mock_resolver):
        """Resolver should filter candidates by ontology_group_id."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        # Return candidates from different groups
        mock_resolver.indices.search_topics = AsyncMock(
            return_value=[
                ({"uuid": "uuid-1", "name": "Wrong Group", "definition": "Def.", "group_id": "other"}, 0.95),
                ({"uuid": "uuid-2", "name": "Right Group", "definition": "Def.", "group_id": "ontology"}, 0.85),
            ]
        )

        # LLM would match candidate 1, but it should be filtered out
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="Test", selected_number=1, reasoning="Match #1."),
                ]
            )
        )

        topics = [TopicDefinition(topic="Test", definition="Test topic.")]
        result = await mock_resolver.resolve(topics)

        # After filtering, only "Right Group" candidate remains, so LLM selecting #1 gets that
        assert len(result.resolved_topics) == 1
        assert result.resolved_topics[0].uuid == "uuid-2"

    @pytest.mark.asyncio
    async def test_resolve_rejects_default_group_candidates(self, mock_resolver):
        """Resolver should not resolve candidates returned from default group."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        mock_resolver.indices.search_topics = AsyncMock(
            return_value=[
                (
                    {
                        "uuid": "uuid-default",
                        "name": "Default Topic",
                        "definition": "Default-group row.",
                        "group_id": "default",
                    },
                    0.94,
                ),
            ]
        )
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(topic="Test", selected_number=1, reasoning="Match #1."),
                ]
            )
        )

        result = await mock_resolver.resolve(
            [TopicDefinition(topic="Test", definition="Test topic.")]
        )

        assert result.resolved_topics == []
        assert result.uuid_remap == {}
        assert "Test" in result.new_topics

    @pytest.mark.asyncio
    async def test_resolve_deduplicates_case_and_whitespace_variants(self, mock_resolver):
        """Resolver should verify one normalized topic and map all original spellings."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        mock_resolver._ensure_ontology_loaded = AsyncMock()
        mock_resolver.indices.search_topics = AsyncMock(
            return_value=[
                (
                    {
                        "uuid": "econ-uuid",
                        "name": "Economic Conditions",
                        "definition": "Overall economic environment.",
                        "group_id": "ontology",
                    },
                    0.9,
                ),
            ]
        )
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(
                        topic="Economic Conditions",
                        selected_number=1,
                        reasoning="Match.",
                    )
                ]
            )
        )

        topics = [
            TopicDefinition(topic="Economic Conditions", definition="Macro backdrop."),
            TopicDefinition(topic=" economic conditions ", definition="Macro backdrop."),
        ]
        result = await mock_resolver.resolve(topics)

        # Verified once after deduplication.
        assert mock_resolver.indices.search_topics.await_count == 1
        assert mock_resolver.embeddings.embed.await_count == 1
        embed_args = mock_resolver.embeddings.embed.await_args.args[0]
        assert len(embed_args) == 1

        # Both original spellings map to the same ontology UUID.
        assert result.uuid_remap["Economic Conditions"] == "econ-uuid"
        assert result.uuid_remap[" economic conditions "] == "econ-uuid"
        assert len(result.resolved_topics) == 1
        assert result.new_topics == []

    @pytest.mark.asyncio
    async def test_resolve_unmatched_deduped_topic_expands_to_all_originals(self, mock_resolver):
        """Unmatched normalized topic should return all original spellings in new_topics."""
        from vanna_kg.types.topics import (
            BatchTopicMatchResponse,
            TopicDefinition,
            TopicMatchDecision,
        )

        mock_resolver._ensure_ontology_loaded = AsyncMock()
        mock_resolver.collect_unmatched = True
        mock_resolver.indices.search_topics = AsyncMock(return_value=[])
        mock_resolver.llm.generate_structured = AsyncMock(
            return_value=BatchTopicMatchResponse(
                decisions=[
                    TopicMatchDecision(
                        topic="Regional conditions",
                        selected_number=None,
                        reasoning="No match.",
                    )
                ]
            )
        )

        topics = [
            TopicDefinition(topic="Regional conditions", definition="Regional outlook."),
            TopicDefinition(topic="regional conditions", definition="Regional outlook."),
        ]
        result = await mock_resolver.resolve(topics)

        assert result.uuid_remap == {}
        assert result.new_topics == ["Regional conditions", "regional conditions"]


class TestOntologyFileErrors:
    """Test error handling for ontology file issues."""

    def test_load_ontology_file_not_found(self, tmp_path):
        """_load_ontology_entries should raise informative error for missing file."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        nonexistent_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError) as exc_info:
            resolver._load_ontology_entries(nonexistent_path)

        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent.json" in str(exc_info.value)

    def test_load_ontology_invalid_json(self, tmp_path):
        """_load_ontology_entries should raise informative error for invalid JSON."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver

        resolver = TopicResolver.__new__(TopicResolver)
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ not valid json")

        with pytest.raises(ValueError) as exc_info:
            resolver._load_ontology_entries(invalid_file)

        assert "invalid json" in str(exc_info.value).lower()


class TestBuildResultValidation:
    """Test _build_result validation logic."""

    def test_build_result_matches_by_topic_name(self):
        """_build_result should match decisions by topic name, not position."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import TopicDefinition, TopicMatchDecision

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.collect_unmatched = True

        topics = [
            TopicDefinition(topic="Apple", definition="Tech company."),
            TopicDefinition(topic="Banana", definition="Fruit."),
        ]
        candidates_list = [
            [({"uuid": "apple-uuid", "name": "Apple Inc", "definition": "Company."}, 0.9)],
            [({"uuid": "banana-uuid", "name": "Banana Corp", "definition": "Corp."}, 0.8)],
        ]
        # Decisions in REVERSE order - should still match correctly by name
        decisions = [
            TopicMatchDecision(topic="Banana", selected_number=1, reasoning="Match."),
            TopicMatchDecision(topic="Apple", selected_number=1, reasoning="Match."),
        ]

        result = resolver._build_result(topics, candidates_list, decisions)

        # Should match correctly despite order difference
        assert result.uuid_remap.get("Apple") == "apple-uuid"
        assert result.uuid_remap.get("Banana") == "banana-uuid"
        assert len(result.resolved_topics) == 2

    def test_build_result_case_insensitive_matching(self):
        """_build_result should match topic names case-insensitively."""
        from vanna_kg.ingestion.resolution.topic_resolver import TopicResolver
        from vanna_kg.types.topics import TopicDefinition, TopicMatchDecision

        resolver = TopicResolver.__new__(TopicResolver)
        resolver.collect_unmatched = True

        topics = [TopicDefinition(topic="M&A", definition="Mergers.")]
        candidates_list = [
            [({"uuid": "ma-uuid", "name": "Mergers", "definition": "M."}, 0.9)],
        ]
        # Decision has different case
        decisions = [
            TopicMatchDecision(topic="m&a", selected_number=1, reasoning="Match."),
        ]

        result = resolver._build_result(topics, candidates_list, decisions)

        assert result.uuid_remap.get("M&A") == "ma-uuid"
        assert len(result.new_topics) == 0
