"""
Tests for the MCP server command parsing and execution.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vanna_kg.mcp.server import (
    Session,
    cmd_stats,
    execute_command,
    get_session,
    parse_command,
    parse_csv_list,
)


def _make_fact(
    uuid: str,
    subject_name: str,
    relationship_type: str,
    object_name: str,
    date_context: str = "2024-01",
    chunk_uuid: str = "chunk-uuid",
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid,
        subject_name=subject_name,
        relationship_type=relationship_type,
        object_name=object_name,
        date_context=date_context,
        chunk_uuid=chunk_uuid,
    )


@pytest.fixture(autouse=True)
def _clear_search_session() -> None:
    get_session().clear_search()


class TestCommandParsing:
    """Test command string parsing."""

    def test_parse_simple_command(self):
        cmd, args = parse_command(
            'find --entity {"name":"Apple","definition":"US tech company"}'
        )
        assert cmd == "find"
        assert args == ["--entity", '{"name":"Apple","definition":"US tech company"}']

    def test_parse_command_with_quotes(self):
        cmd, args = parse_command('search -entity Apple "earnings impact"')
        assert cmd == "search"
        assert "-entity" in args
        assert "Apple" in args
        assert "earnings impact" in args

    def test_parse_find_raw_json_with_spaces(self):
        cmd, args = parse_command(
            'find --entity {"name":"Federal Reserve","definition":"US central banking system"}'
        )
        assert cmd == "find"
        assert args == [
            "--entity",
            '{"name":"Federal Reserve","definition":"US central banking system"}',
        ]

    def test_parse_empty_command_raises(self):
        with pytest.raises(ValueError, match="Empty command"):
            parse_command("")

    def test_parse_csv_list(self):
        result = parse_csv_list("Apple, Microsoft, Google")
        assert result == ["Apple", "Microsoft", "Google"]

    def test_parse_csv_list_single(self):
        result = parse_csv_list("Apple")
        assert result == ["Apple"]

    def test_parse_csv_list_empty(self):
        result = parse_csv_list("")
        assert result == []


class TestSession:
    """Test session state management."""

    def test_session_singleton(self):
        session1 = get_session()
        session2 = get_session()
        assert session1 is session2

    def test_session_clear_search(self):
        session = Session()
        session.search_results = [MagicMock()]
        session.clear_search()
        assert session.search_results == []


class TestHelpCommand:
    """Test help command."""

    @pytest.mark.asyncio
    async def test_help_returns_usage(self):
        result = await execute_command("help")
        assert "find" in result
        assert "search" in result
        assert "cat" in result
        assert "info" in result
        assert "Core workflow: find → search → cat → info" in result


class TestUnknownCommand:
    """Test unknown command handling."""

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        result = await execute_command("foobar")
        assert "Unknown command" in result
        assert "foobar" in result


class TestStatsCommand:
    """Test stats command with mocked KG."""

    @pytest.mark.asyncio
    async def test_stats_format(self):
        # Mock the KG
        mock_kg = AsyncMock()
        mock_kg.stats = AsyncMock(return_value={
            "entities": 100,
            "facts": 500,
            "chunks": 200,
            "documents": 10,
        })

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_stats([])

        assert "KNOWLEDGE BASE STATISTICS" in result
        assert "100" in result
        assert "500" in result
        assert "200" in result
        assert "10" in result


class TestFindCommand:
    """Test find command with mocked KG."""

    @pytest.mark.asyncio
    async def test_find_entity_and_topic(self):
        from vanna_kg.mcp.server import cmd_find

        # Mock entity
        mock_entity = MagicMock()
        mock_entity.name = "Apple Inc"
        mock_entity.entity_type = "COMPANY"
        mock_entity.uuid = "abc12345-1234-1234-1234-123456789abc"

        mock_topic = MagicMock()
        mock_topic.name = "Inflation"

        mock_storage = MagicMock()
        mock_storage.search_topics = AsyncMock(return_value=[(mock_topic, 0.9)])

        mock_kg = AsyncMock()
        mock_kg.search_entities = AsyncMock(return_value=[mock_entity])
        mock_kg._storage = mock_storage
        mock_kg._embeddings = MagicMock()
        mock_kg._embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2])

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_find(
                [
                    "--entity",
                    '{"name":"Apple","definition":"US tech company"}',
                    "--topic",
                    '{"name":"Inflation","definition":"General price increases"}',
                ]
            )

        assert "ENTITIES:" in result
        assert "TOPICS:" in result
        assert "Apple" in result
        assert "COMPANY" in result
        assert "Inflation" in result

    @pytest.mark.asyncio
    async def test_find_no_args(self):
        from vanna_kg.mcp.server import cmd_find

        result = await cmd_find([])
        assert result == "Usage: find --entity <json> and/or --topic <json>"

    @pytest.mark.asyncio
    async def test_find_invalid_json(self):
        from vanna_kg.mcp.server import cmd_find

        result = await cmd_find(["--entity", "{bad json"])
        assert "Invalid Entity selector JSON" in result

    @pytest.mark.asyncio
    async def test_find_missing_definition(self):
        from vanna_kg.mcp.server import cmd_find

        result = await cmd_find(["--entity", '{"name":"Apple"}'])
        assert "requires non-empty definition" in result

    @pytest.mark.asyncio
    async def test_find_empty_definition(self):
        from vanna_kg.mcp.server import cmd_find

        result = await cmd_find(["--entity", '{"name":"Apple","definition":"  "}'])
        assert "requires non-empty definition" in result

    @pytest.mark.asyncio
    async def test_find_multiple_selectors_with_ids(self):
        from vanna_kg.mcp.server import cmd_find

        mock_entity = MagicMock()
        mock_entity.name = "Apple Inc"
        mock_entity.entity_type = "COMPANY"

        mock_topic = MagicMock()
        mock_topic.name = "Inflation"

        mock_storage = MagicMock()
        mock_storage.search_topics = AsyncMock(return_value=[(mock_topic, 0.9)])

        mock_kg = AsyncMock()
        mock_kg.search_entities = AsyncMock(return_value=[mock_entity])
        mock_kg._storage = mock_storage
        mock_kg._embeddings = MagicMock()
        mock_kg._embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2])

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_find(
                [
                    "--entity",
                    '{"id":"e1","name":"Apple","definition":"US tech company"}',
                    "--entity",
                    '{"id":"e2","name":"Microsoft","definition":"Enterprise software company"}',
                    "--topic",
                    '{"id":"t1","name":"Inflation","definition":"General price increases"}',
                ]
            )

        assert "(id=e1)" in result
        assert "(id=e2)" in result
        assert "(id=t1)" in result

    @pytest.mark.asyncio
    async def test_find_rejects_legacy_to_flag(self):
        from vanna_kg.mcp.server import cmd_find

        result = await cmd_find(["-to", "Inflation"])
        assert result == "Usage: find --entity <json> and/or --topic <json>"

    @pytest.mark.asyncio
    async def test_find_rejects_legacy_entity_flag(self):
        from vanna_kg.mcp.server import cmd_find

        result = await cmd_find(["-entity", "Apple"])
        assert result == "Usage: find --entity <json> and/or --topic <json>"

    @pytest.mark.asyncio
    async def test_find_rejects_legacy_topic_flag(self):
        from vanna_kg.mcp.server import cmd_find

        result = await cmd_find(["-topic", "Inflation"])
        assert result == "Usage: find --entity <json> and/or --topic <json>"


class TestSearchCommand:
    """Test search command contract and workflow behavior."""

    @pytest.mark.asyncio
    async def test_search_default_mode_around(self):
        from vanna_kg.mcp.server import cmd_search

        mock_duckdb = MagicMock()
        mock_duckdb.get_facts_by_entities = AsyncMock(
            return_value=[_make_fact("fact-1", "Apple Inc", "MENTIONED_IN", "Revenue")]
        )

        mock_storage = MagicMock()
        mock_storage._duckdb = mock_duckdb
        mock_storage._lancedb = MagicMock()
        mock_storage.get_chunk = AsyncMock(return_value=None)
        mock_storage.get_document = AsyncMock(return_value=None)

        mock_kg = AsyncMock()
        mock_kg._storage = mock_storage
        mock_kg._embeddings = MagicMock()

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_search(["-entity", "Apple Inc"])

        assert "mode: around" in result
        call = mock_duckdb.get_facts_by_entities.await_args
        assert call.args[0] == ["Apple Inc"]
        assert call.kwargs["mode"] == "around"

    @pytest.mark.asyncio
    async def test_search_requires_selector(self):
        from vanna_kg.mcp.server import cmd_search

        mock_kg = AsyncMock()
        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_search(["--mode", "between"])

        assert result == "Usage: search requires at least one selector: -entity and/or -topic"

    @pytest.mark.asyncio
    async def test_search_between_entity_subset(self):
        from vanna_kg.mcp.server import cmd_search

        mock_duckdb = MagicMock()
        mock_duckdb.get_facts_by_entities = AsyncMock(
            return_value=[_make_fact("fact-2", "Apple Inc", "COMPETES_WITH", "Microsoft Corp")]
        )

        mock_storage = MagicMock()
        mock_storage._duckdb = mock_duckdb
        mock_storage._lancedb = MagicMock()
        mock_storage.get_chunk = AsyncMock(return_value=None)
        mock_storage.get_document = AsyncMock(return_value=None)

        mock_kg = AsyncMock()
        mock_kg._storage = mock_storage
        mock_kg._embeddings = MagicMock()

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_search(
                ["-entity", "Apple Inc, Microsoft Corp", "--mode", "between"]
            )

        assert "mode: between" in result
        call = mock_duckdb.get_facts_by_entities.await_args
        assert call.args[0] == ["Apple Inc", "Microsoft Corp"]
        assert call.kwargs["mode"] == "between"

    @pytest.mark.asyncio
    async def test_search_between_mixed_subset(self):
        from vanna_kg.mcp.server import cmd_search

        mock_duckdb = MagicMock()
        mock_duckdb.get_facts_by_entities = AsyncMock(
            return_value=[_make_fact("fact-3", "Apple Inc", "EXPOSED_TO", "Inflation")]
        )

        mock_storage = MagicMock()
        mock_storage._duckdb = mock_duckdb
        mock_storage._lancedb = MagicMock()
        mock_storage.get_chunk = AsyncMock(return_value=None)
        mock_storage.get_document = AsyncMock(return_value=None)

        mock_kg = AsyncMock()
        mock_kg._storage = mock_storage
        mock_kg._embeddings = MagicMock()

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_search(
                ["-entity", "Apple Inc", "-topic", "Inflation", "--mode", "between"]
            )

        assert "mode: between" in result
        assert "topics: Inflation" in result
        call = mock_duckdb.get_facts_by_entities.await_args
        assert call.args[0] == ["Apple Inc", "Inflation"]
        assert call.kwargs["mode"] == "between"

    @pytest.mark.asyncio
    async def test_search_rejects_legacy_to_flag(self):
        from vanna_kg.mcp.server import cmd_search

        mock_kg = AsyncMock()
        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_search(["-entity", "Apple Inc", "-to", "Inflation"])

        assert result.startswith("Usage: search -entity <names> and/or -topic <names>")

    @pytest.mark.asyncio
    async def test_cat_session_index_behavior_unchanged(self):
        from vanna_kg.mcp.server import cmd_cat, cmd_search

        first_batch = [
            _make_fact("fact-a", "Apple Inc", "AFFECTS", "Inflation", chunk_uuid="chunk-a"),
            _make_fact(
                "fact-b", "Apple Inc", "COMPETES_WITH", "Microsoft Corp", chunk_uuid="chunk-b"
            ),
        ]
        second_batch = [
            _make_fact("fact-c", "Apple Inc", "MENTIONS", "Revenue", chunk_uuid="chunk-c"),
        ]

        mock_duckdb = MagicMock()
        mock_duckdb.get_facts_by_entities = AsyncMock(side_effect=[first_batch, second_batch])

        mock_storage = MagicMock()
        mock_storage._duckdb = mock_duckdb
        mock_storage._lancedb = MagicMock()
        mock_storage.get_chunk = AsyncMock(return_value=None)
        mock_storage.get_document = AsyncMock(return_value=None)
        mock_storage.get_fact = AsyncMock(
            side_effect=lambda uuid: SimpleNamespace(uuid=uuid, content=f"content-{uuid}")
        )
        mock_storage.get_chunk = AsyncMock(
            side_effect=lambda uuid: SimpleNamespace(
                uuid=uuid,
                header_path=f"doc-{uuid}",
                content=f"chunk-{uuid}",
                document_uuid=None,
            )
        )

        mock_kg = AsyncMock()
        mock_kg._storage = mock_storage
        mock_kg._embeddings = MagicMock()

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            await cmd_search(["-entity", "Apple Inc"])
            first_cat = await cmd_cat(["2"])
            await cmd_search(["-entity", "Apple Inc"])
            second_cat = await cmd_cat(["2"])

        assert "Microsoft Corp" in first_cat
        assert "Result 2 not found. Valid: 1-1" in second_cat


class TestLsCommand:
    """Test ls command with mocked KG."""

    @pytest.mark.asyncio
    async def test_ls_entities(self):
        from vanna_kg.mcp.server import cmd_ls

        mock_entity = MagicMock()
        mock_entity.name = "Apple Inc"
        mock_entity.entity_type = "COMPANY"

        mock_kg = AsyncMock()
        mock_kg.get_entities = AsyncMock(return_value=[mock_entity])

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_ls(["entities"])

        assert "ENTITIES:" in result
        assert "Apple" in result

    @pytest.mark.asyncio
    async def test_ls_documents(self):
        from vanna_kg.mcp.server import cmd_ls

        mock_doc = MagicMock()
        mock_doc.name = "report.pdf"
        mock_doc.document_date = "2024-01-15"

        mock_kg = AsyncMock()
        mock_kg.get_documents = AsyncMock(return_value=[mock_doc])

        with patch("vanna_kg.mcp.server.get_kg", return_value=mock_kg):
            result = await cmd_ls(["documents"])

        assert "DOCUMENTS:" in result
        assert "report.pdf" in result
