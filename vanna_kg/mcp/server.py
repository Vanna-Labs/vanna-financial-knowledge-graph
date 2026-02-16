"""
VannaKG MCP Server

Single-tool MCP server that executes knowledge graph commands.

Commands follow the syntax defined in vanna_kg/skills/kg-query/SKILL.md:
    find --entity {"name":"Apple","definition":"US technology company"}
    search -entity Apple_Inc -topic Interest_Rates --mode around --query "earnings" --from 2024-01
    cat 1
    info Apple_Inc
    ls entities -type company
    stats

Session state is maintained for the find → search → cat → info workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env file for API keys
load_dotenv()

# Lazy imports to avoid circular dependencies
_kg_instance: Any = None
_session: Session | None = None


@dataclass
class SearchResult:
    """A single search result for referencing with cat."""
    index: int
    date: str
    from_entity: str
    relationship: str
    to_entity: str
    document: str
    score: float
    fact_uuid: str
    chunk_uuid: str


@dataclass
class Session:
    """Session state for the find → search → cat → info workflow."""
    # Last search results for cat command (only state we need)
    search_results: list[SearchResult] = field(default_factory=list)

    def clear_search(self) -> None:
        """Clear search results."""
        self.search_results = []


def get_session() -> Session:
    """Get or create the session."""
    global _session
    if _session is None:
        _session = Session()
    return _session


async def get_kg() -> Any:
    """Get the KnowledgeGraph instance."""
    global _kg_instance
    if _kg_instance is None:
        raise RuntimeError("KnowledgeGraph not initialized. Call init_kg() first.")
    return _kg_instance


async def init_kg(kb_path: str | Path) -> None:
    """Initialize the KnowledgeGraph instance."""
    global _kg_instance
    from vanna_kg.api.knowledge_graph import KnowledgeGraph
    _kg_instance = KnowledgeGraph(kb_path, create=False)
    await _kg_instance._ensure_initialized()


# =============================================================================
# Command Parsing
# =============================================================================

def _tokenize_find_args(arg_text: str) -> list[str]:
    """Tokenize find args, allowing raw JSON blocks after --entity/--topic."""
    tokens: list[str] = []
    i = 0
    length = len(arg_text)
    expect_json = False

    while i < length:
        while i < length and arg_text[i].isspace():
            i += 1
        if i >= length:
            break

        if expect_json:
            if arg_text[i] in ('"', "'"):
                quote = arg_text[i]
                i += 1
                buf = []
                while i < length and arg_text[i] != quote:
                    if arg_text[i] == "\\" and quote == '"' and i + 1 < length:
                        buf.append(arg_text[i + 1])
                        i += 2
                        continue
                    buf.append(arg_text[i])
                    i += 1
                tokens.append("".join(buf))
                if i < length and arg_text[i] == quote:
                    i += 1
                expect_json = False
                continue

            if arg_text[i] != "{":
                start = i
                while i < length and not arg_text[i].isspace():
                    i += 1
                tokens.append(arg_text[start:i])
                expect_json = False
                continue

            start = i
            depth = 0
            in_string = False
            escape = False
            while i < length:
                ch = arg_text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                i += 1

            tokens.append(arg_text[start:i].strip())
            expect_json = False
            continue

        if arg_text[i] in ('"', "'"):
            quote = arg_text[i]
            i += 1
            buf = []
            while i < length and arg_text[i] != quote:
                if arg_text[i] == "\\" and quote == '"' and i + 1 < length:
                    buf.append(arg_text[i + 1])
                    i += 2
                    continue
                buf.append(arg_text[i])
                i += 1
            tokens.append("".join(buf))
            if i < length and arg_text[i] == quote:
                i += 1
        else:
            start = i
            while i < length and not arg_text[i].isspace():
                i += 1
            tokens.append(arg_text[start:i])

        if tokens[-1] in ("--entity", "--topic"):
            expect_json = True

    return tokens


def parse_command(command: str) -> tuple[str, list[str]]:
    """Parse a command string into (cmd_name, args)."""
    stripped = command.strip()
    if not stripped:
        raise ValueError("Empty command")
    parts = stripped.split(maxsplit=1)
    cmd_name = parts[0].lower()
    arg_text = parts[1] if len(parts) > 1 else ""

    if cmd_name == "find":
        return cmd_name, _tokenize_find_args(arg_text)

    return cmd_name, shlex.split(arg_text)


def parse_csv_list(value: str) -> list[str]:
    """Parse comma-separated values, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def format_entity_type(entity_type: Any) -> str:
    """Format an entity type value as uppercase text."""
    return entity_type.upper() if hasattr(entity_type, "upper") else str(entity_type).upper()


def _wrap_text(text: str, width: int = 63) -> list[str]:
    if not text:
        return [""]
    return [text[i:i + width] for i in range(0, len(text), width)]


def _box_line(text: str) -> str:
    return f"│ {text}".ljust(66) + "│"


def _parse_selector_list(values: list[str], label: str) -> tuple[list[dict[str, str]], str | None]:
    selectors: list[dict[str, str]] = []
    for raw in values:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [], f"Invalid {label} selector JSON: {raw}"
        if not isinstance(parsed, dict):
            return [], f"{label} selector must be a JSON object"
        name = parsed.get("name")
        definition = parsed.get("definition")
        selector_id = parsed.get("id")
        if not isinstance(name, str) or not name.strip():
            return [], f"{label} selector requires non-empty name"
        if not isinstance(definition, str) or not definition.strip():
            return [], f"{label} selector requires non-empty definition"
        if "id" in parsed and (not isinstance(selector_id, str) or not selector_id.strip()):
            return [], f"{label} selector id must be a non-empty string"
        selectors.append(
            {
                "name": name.strip(),
                "definition": definition.strip(),
                "id": selector_id.strip() if isinstance(selector_id, str) else "",
            }
        )
    return selectors, None


def _format_selector_label(name: str, selector_id: str) -> str:
    label = f"\"{name}\""
    if selector_id:
        label += f" (id={selector_id})"
    return label


# =============================================================================
# Command Implementations
# =============================================================================

async def cmd_find(args: list[str]) -> str:
    """
    Resolve names to canonical entities/topics.

    Usage:
        find --entity '{"name":"Apple","definition":"US technology company"}'
        find --topic '{"name":"Inflation","definition":"General price increases"}'
        find --entity '{"name":"Apple","definition":"Consumer electronics company"}' --topic '{"name":"Inflation","definition":"General price increases"}'
    """
    usage = "Usage: find --entity <json> and/or --topic <json>"
    if "-to" in args or "--to" in args:
        return usage

    parser = argparse.ArgumentParser(prog="find", add_help=False, allow_abbrev=False)
    parser.add_argument("--entity", action="append", dest="entities", default=[])
    parser.add_argument("--topic", action="append", dest="topics", default=[])

    try:
        parsed, unknown = parser.parse_known_args(args)
    except SystemExit:
        return usage

    if unknown:
        return usage

    entity_selectors, entity_error = _parse_selector_list(parsed.entities, "Entity")
    if entity_error:
        return entity_error
    topic_selectors, topic_error = _parse_selector_list(parsed.topics, "Topic")
    if topic_error:
        return topic_error

    if not entity_selectors and not topic_selectors:
        return usage

    kg = await get_kg()
    output_lines: list[str] = []

    if entity_selectors:
        output_lines.append("ENTITIES:")
        for selector in entity_selectors:
            name = selector["name"]
            definition = selector["definition"]
            selector_id = selector["id"]
            query_text = f"{name}: {definition}"
            results = await kg.search_entities(query_text, limit=5, threshold=0.3)
            output_lines.append(f"  {_format_selector_label(name, selector_id)}")
            if results:
                for entity in results:
                    entity_type = format_entity_type(entity.entity_type)
                    output_lines.append(f"    → [{entity_type}] {entity.name}")
            else:
                output_lines.append("    → (no matches found)")
        output_lines.append("")

    if topic_selectors:
        assert kg._storage is not None
        output_lines.append("TOPICS:")
        for selector in topic_selectors:
            name = selector["name"]
            definition = selector["definition"]
            selector_id = selector["id"]
            query_text = f"{name}: {definition}"
            query_vec = await kg._embeddings.embed_single(query_text)
            topic_results = await kg._storage.search_topics(query_vec, limit=5, threshold=0.3)
            output_lines.append(f"  {_format_selector_label(name, selector_id)}")
            if topic_results:
                for topic, _score in topic_results:
                    output_lines.append(f"    → [TOPIC] {topic.name}")
            else:
                output_lines.append("    → (no matches found)")

    return "\n".join(output_lines)


async def cmd_search(args: list[str]) -> str:
    """
    Find connections between nodes.

    - --mode around: endpoint overlap with selected node set (subject OR object)
    - --mode between: both endpoints inside selected node set (subject AND object)

    Usage:
        search -entity "Apple Inc"
        search -topic "Inflation"
        search -entity "Apple Inc, Microsoft Corp" --mode between
        search -entity "Fed Boston" -topic "Manufacturing" --query "reported" --from 2024-01
    """
    usage = (
        "Usage: search -entity <names> and/or -topic <names> "
        "[--mode around|between] [--query \"text\"] [--from DATE] [--to-date DATE]"
    )
    if "-to" in args or "--to" in args:
        return usage

    kg = await get_kg()
    session = get_session()
    session.clear_search()

    parser = argparse.ArgumentParser(prog="search", add_help=False, allow_abbrev=False)
    parser.add_argument("-entity", "--entity", type=str, default="")
    parser.add_argument("-topic", "--topic", dest="topic_names", type=str, default="")
    parser.add_argument("--mode", choices=["around", "between"], default="around")
    parser.add_argument("--query", "-q", type=str, default="")
    parser.add_argument("--from", dest="from_date", type=str, default="")
    parser.add_argument("--to-date", dest="to_date", type=str, default="")
    parser.add_argument("--limit", type=int, default=20)

    try:
        parsed, unknown = parser.parse_known_args(args)
    except SystemExit:
        return usage

    entity_names = parse_csv_list(parsed.entity) if parsed.entity else []
    topic_names = parse_csv_list(parsed.topic_names) if parsed.topic_names else []
    selected_names = list(dict.fromkeys([*entity_names, *topic_names]))

    if unknown:
        return usage

    if not selected_names:
        return "Usage: search requires at least one selector: -entity and/or -topic"

    assert kg._storage is not None

    # Stage 1: Get facts from DuckDB
    fetch_limit = parsed.limit * 3 if parsed.query else parsed.limit

    facts = await kg._storage._duckdb.get_facts_by_entities(
        selected_names,
        mode=parsed.mode,
        limit=fetch_limit,
        from_date=parsed.from_date or None,
        to_date=parsed.to_date or None,
    )

    # Stage 2: If semantic query provided, rank facts by similarity
    fact_scores: dict[str, float] = {}
    if parsed.query and facts:
        query_vector = await kg._embeddings.embed_single(parsed.query)
        fact_uuids = [f.uuid for f in facts]
        ranked_results = await kg._storage._lancedb.search_facts_by_uuids(
            query_vector,
            fact_uuids,
            limit=parsed.limit,
            threshold=0.0,
        )
        fact_scores = {uuid: score for uuid, score in ranked_results}
        ranked_uuids = [uuid for uuid, _ in ranked_results]
        uuid_to_fact = {f.uuid: f for f in facts}
        facts = [uuid_to_fact[uuid] for uuid in ranked_uuids if uuid in uuid_to_fact]

    if not facts:
        info_parts = [f"mode: {parsed.mode}"]
        if entity_names:
            info_parts.append(f"entities: {', '.join(entity_names)}")
        if topic_names:
            info_parts.append(f"topics: {', '.join(topic_names)}")
        return f"No connections found for {' + '.join(info_parts)}"

    # Format output
    output_lines = []
    header_parts = [f"mode: {parsed.mode}"]
    if entity_names:
        header_parts.append(f"entities: {', '.join(entity_names)}")
    if topic_names:
        header_parts.append(f"topics: {', '.join(topic_names)}")
    if parsed.query:
        header_parts.append(f'query: "{parsed.query}"')
    if header_parts:
        output_lines.append(f"Searching {' + '.join(header_parts)}")
        output_lines.append("")

    chunk_cache: dict[str, Any] = {}
    document_cache: dict[str, Any] = {}

    for i, fact in enumerate(facts[:parsed.limit], 1):
        date_str = fact.date_context or "Unknown"
        doc_name = "Unknown"
        score = fact_scores.get(fact.uuid, 0.0)
        chunk_uuid = fact.chunk_uuid or ""

        if chunk_uuid:
            chunk = chunk_cache.get(chunk_uuid)
            if chunk is None and hasattr(kg._storage, "get_chunk"):
                chunk = await kg._storage.get_chunk(chunk_uuid)
                chunk_cache[chunk_uuid] = chunk
            if chunk is not None and getattr(chunk, "document_uuid", None):
                doc_uuid = chunk.document_uuid
                document = document_cache.get(doc_uuid)
                if document is None and hasattr(kg._storage, "get_document"):
                    document = await kg._storage.get_document(doc_uuid)
                    document_cache[doc_uuid] = document
                if document is not None and getattr(document, "name", None):
                    doc_name = document.name

        result = SearchResult(
            index=i,
            date=date_str,
            from_entity=fact.subject_name,
            relationship=fact.relationship_type,
            to_entity=fact.object_name,
            document=doc_name,
            score=score,
            fact_uuid=fact.uuid,
            chunk_uuid=chunk_uuid,
        )
        session.search_results.append(result)

        output_lines.append(
            f"[{i}] {date_str}  "
            f"{fact.subject_name} ──[{fact.relationship_type}]──▶ {fact.object_name}"
        )
        score_str = f" | score: {score:.2f}" if score > 0 else ""
        output_lines.append(f"    📄 {doc_name}{score_str}")
        output_lines.append("")

    return "\n".join(output_lines)


async def cmd_cat(args: list[str]) -> str:
    """
    Expand connection details.

    Usage:
        cat 1              # By result number from search
        cat 1 2 3          # Multiple results
    """
    kg = await get_kg()
    session = get_session()

    if not args:
        return "Usage: cat <result_number> [result_number ...]"

    if not session.search_results:
        return "No search results. Run 'search' first."

    output_parts = []

    for arg in args:
        try:
            idx = int(arg)
        except ValueError:
            output_parts.append(f"Invalid result number: {arg}")
            continue

        # Find result by index
        result = None
        for r in session.search_results:
            if r.index == idx:
                result = r
                break

        if result is None:
            output_parts.append(f"Result {idx} not found. Valid: 1-{len(session.search_results)}")
            continue

        # Get full fact details
        assert kg._storage is not None
        fact = await kg._storage.get_fact(result.fact_uuid)

        if fact is None:
            output_parts.append(f"[{idx}] Fact not found: {result.fact_uuid}")
            continue

        # Get chunk for context
        chunk = None
        if result.chunk_uuid:
            chunk = await kg._storage.get_chunk(result.chunk_uuid)
        document = None
        if chunk is not None and getattr(chunk, "document_uuid", None):
            document = await kg._storage.get_document(chunk.document_uuid)

        # Format output
        lines = [
            "┌" + "─" * 65 + "┐",
            (
                f"│ {result.from_entity} ──[{result.relationship}]──▶ {result.to_entity}"
            ).ljust(66) + "│",
            _box_line(f"Date Context: {result.date}"),
            "├" + "─" * 65 + "┤",
        ]

        doc_name = document.name if document is not None and document.name else "Unknown"
        doc_date = (
            document.document_date
            if document is not None and getattr(document, "document_date", None)
            else "Unknown"
        )
        lines.append(_box_line(f"Document Name: {doc_name}"))
        lines.append(_box_line(f"Document Date: {doc_date}"))
        lines.append("├" + "─" * 65 + "┤")

        lines.append(_box_line("FACT:"))
        for line in _wrap_text(fact.content):
            lines.append(_box_line(line))
        lines.append("├" + "─" * 65 + "┤")

        if chunk is not None:
            lines.append(_box_line("CHUNK:"))
            for line in _wrap_text(chunk.content):
                lines.append(_box_line(line))
            lines.append("├" + "─" * 65 + "┤")

        if chunk is not None:
            lines.append(_box_line(f"Source: {chunk.header_path or 'Document'}"))
            lines.append(_box_line(f"Chunk UUID: {chunk.uuid[:12]}"))
        else:
            lines.append(_box_line(f"Fact UUID: {fact.uuid}"))

        lines.append("└" + "─" * 65 + "┘")

        output_parts.append("\n".join(lines))

    return "\n\n".join(output_parts)


async def cmd_info(args: list[str]) -> str:
    """
    Get entity/topic summary.

    Usage:
        info Apple_Inc
        info Interest_Rates
    """
    kg = await get_kg()

    if not args:
        return "Usage: info <entity_or_topic_name>"

    name = " ".join(args).replace("_", " ")

    # Try to find as entity first
    entity = await kg.get_entity(name)

    if entity:
        # Get fact count
        facts = await kg.get_facts_for_entity(entity.name, limit=1000)
        neighbors = await kg.get_neighbors(entity.name, limit=100)

        entity_type = format_entity_type(entity.entity_type)

        lines = [
            "┌" + "─" * 65 + "┐",
            f"│ {entity.name} [{entity_type}]".ljust(66) + "│",
            "├" + "─" * 65 + "┤",
        ]

        # Wrap summary
        summary = entity.summary or "No summary available."
        wrapped = [summary[i:i+63] for i in range(0, len(summary), 63)]
        for line in wrapped[:4]:
            lines.append(f"│ {line}".ljust(66) + "│")

        lines.append("├" + "─" * 65 + "┤")

        aliases = ", ".join(entity.aliases) if entity.aliases else "None"
        lines.append(f"│ Aliases: {aliases[:50]}".ljust(66) + "│")
        lines.append(f"│ Facts: {len(facts)} | Connections: {len(neighbors)}".ljust(66) + "│")
        lines.append("└" + "─" * 65 + "┘")

        return "\n".join(lines)

    # Try as topic
    assert kg._storage is not None
    topics = await kg._storage.get_topics_by_names([name])

    if topics:
        topic = topics[0]
        lines = [
            "┌" + "─" * 65 + "┐",
            f"│ {topic.name} [TOPIC]".ljust(66) + "│",
            "├" + "─" * 65 + "┤",
        ]

        definition = topic.definition or "No definition available."
        wrapped = [definition[i:i+63] for i in range(0, len(definition), 63)]
        for line in wrapped[:4]:
            lines.append(f"│ {line}".ljust(66) + "│")

        lines.append("└" + "─" * 65 + "┘")

        return "\n".join(lines)

    return f"Not found: {name}"


async def cmd_ls(args: list[str]) -> str:
    """
    Browse available nodes.

    Usage:
        ls entities
        ls entities -type company
        ls topics
        ls documents
        ls documents --from 2024-10
    """
    kg = await get_kg()

    parser = argparse.ArgumentParser(prog="ls", add_help=False)
    parser.add_argument("category", nargs="?", default="entities")
    parser.add_argument("-type", "--type", dest="entity_type", type=str, default="")
    parser.add_argument("--from", dest="from_date", type=str, default="")
    parser.add_argument("--limit", type=int, default=20)

    try:
        parsed, _ = parser.parse_known_args(args)
    except SystemExit:
        return "Usage: ls [entities|topics|documents] [-type TYPE] [--from DATE]"

    category = parsed.category.lower()

    if category == "entities":
        entities = await kg.get_entities(limit=parsed.limit)

        # Filter by type if specified
        if parsed.entity_type:
            type_filter = parsed.entity_type.upper()
            entities = [
                e for e in entities
                if str(e.entity_type).upper() == type_filter
            ]

        if not entities:
            return "No entities found."

        lines = [f"ENTITIES: {len(entities)} shown"]
        for e in entities:
            etype = str(e.entity_type).upper() if e.entity_type else "UNKNOWN"
            lines.append(f"  {e.name.replace(' ', '_'):<30} [{etype}]")

        return "\n".join(lines)

    elif category == "topics":
        # Get topics via search with empty query
        assert kg._storage is not None
        # Use a broad search
        query_vec = await kg._embeddings.embed_single("financial topics")
        topic_results = await kg._storage.search_topics(query_vec, limit=parsed.limit)

        if not topic_results:
            return "No topics found."

        lines = [f"TOPICS: {len(topic_results)} shown"]
        for topic, score in topic_results:
            lines.append(f"  {topic.name.replace(' ', '_'):<30} [TOPIC]")

        return "\n".join(lines)

    elif category == "documents":
        docs = await kg.get_documents()

        if not docs:
            return "No documents found."

        lines = [f"DOCUMENTS: {len(docs)} total"]
        for doc in docs[:parsed.limit]:
            date_str = doc.document_date or "Unknown"
            lines.append(f"  {doc.name:<40} {date_str}")

        return "\n".join(lines)

    else:
        return f"Unknown category: {category}. Use: entities, topics, documents"


async def cmd_stats(args: list[str]) -> str:
    """
    Knowledge base statistics.

    Usage:
        stats
    """
    kg = await get_kg()
    stats = await kg.stats()

    lines = [
        "┌" + "─" * 65 + "┐",
        "│ KNOWLEDGE BASE STATISTICS".ljust(66) + "│",
        "├" + "─" * 65 + "┤",
        f"│ Entities:    {stats.get('entities', 0):,}".ljust(66) + "│",
        f"│ Facts:       {stats.get('facts', 0):,}".ljust(66) + "│",
        f"│ Chunks:      {stats.get('chunks', 0):,}".ljust(66) + "│",
        f"│ Documents:   {stats.get('documents', 0):,}".ljust(66) + "│",
        "└" + "─" * 65 + "┘",
    ]

    return "\n".join(lines)


# =============================================================================
# Command Dispatcher
# =============================================================================

COMMANDS = {
    "find": cmd_find,
    "search": cmd_search,
    "cat": cmd_cat,
    "info": cmd_info,
    "ls": cmd_ls,
    "stats": cmd_stats,
}


async def execute_command(command: str) -> str:
    """Execute a knowledge graph command."""
    try:
        cmd_name, args = parse_command(command)
    except ValueError as e:
        return f"Error: {e}"

    if cmd_name == "help":
        return """VannaKG Commands:
    find [--entity <json>] [--topic <json>]
                                          Resolve names to canonical nodes
    search -entity <n> [-topic <n>] [--mode around|between]
           [--query "action"] [--from DATE] [--to-date DATE]
                                          Find connections between nodes
    cat <number>                          Expand fact details
    info <name>                           Entity/topic summary
    ls [entities|topics|documents]        Browse available nodes
    stats                                 Knowledge base statistics

Core workflow: find → search → cat → info
around mode: edges touching selected nodes (subject OR object)
between mode: edges fully inside selected nodes (subject AND object)
--query ranks by relationship/action (e.g., "acquired", "reported")"""

    handler = COMMANDS.get(cmd_name)
    if handler is None:
        return f"Unknown command: {cmd_name}. Type 'help' for available commands."

    try:
        return await handler(args)
    except Exception as e:
        return f"Error executing {cmd_name}: {e}"


# =============================================================================
# MCP Server
# =============================================================================

def create_server(name: str = "vanna-kg") -> FastMCP:
    """Create the MCP server with the kg_execute tool."""
    mcp = FastMCP(name)

    @mcp.tool()
    async def kg_execute(command: str) -> str:
        """
        Execute a VannaKG knowledge graph command.

        Commands follow the find → search → cat → info workflow:

        1. find: Resolve names to canonical entities/topics
           find --entity {"name":"Apple","definition":"Consumer electronics company"}
           find --topic {"name":"Inflation","definition":"General price increases"}

        2. search: Find connections between selected nodes
           search -entity "Apple Inc" --mode around
           search -entity "Apple Inc, Microsoft Corp" --mode between
           search -entity "Fed Boston" -topic "Manufacturing" --query "reported" --from 2024-01

        3. cat: Expand fact details (by result number from search)
           cat 1

        4. info: Entity/topic summary
           info "Apple Inc"

        Additional commands:
           ls entities             - Browse entities
           ls topics               - Browse topics
           ls documents            - Browse documents
           stats                   - Knowledge base statistics
           help                    - Show command help

        Args:
            command: The command string to execute

        Returns:
            Command output as formatted text
        """
        return await execute_command(command)

    return mcp


async def run_server(kb_path: str | Path) -> None:
    """Initialize KG and run the MCP server."""
    await init_kg(kb_path)
    mcp = create_server()
    await mcp.run_async()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """CLI entry point for the MCP server."""
    import sys

    parser = argparse.ArgumentParser(
        description="VannaKG MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m vanna_kg.mcp --kb ./my_kb

Claude Desktop config:
    {
        "mcpServers": {
            "vanna-kg": {
                "command": "python",
                "args": ["-m", "vanna_kg.mcp", "--kb", "./my_kb"]
            }
        }
    }
""",
    )
    parser.add_argument(
        "--kb", "-k",
        type=Path,
        required=True,
        help="Path to knowledge base directory",
    )

    args = parser.parse_args()

    if not args.kb.exists():
        print(f"Error: Knowledge base not found: {args.kb}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_server(args.kb))


if __name__ == "__main__":
    main()
