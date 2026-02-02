"""
KGShell - Filesystem-Style Navigation Interface

Presents the knowledge graph as a virtual filesystem navigable with
familiar CLI commands. Designed for both human exploration and LLM
agent-based querying.

Status:
    Placeholder API only. Core shell commands are not implemented yet and
    currently raise NotImplementedError. Use KnowledgeGraph query/search APIs
    directly until shell execution is shipped.

Virtual Filesystem Structure:
    /kg/
    ├── entities/
    │   ├── @search           # Semantic search endpoint
    │   ├── Apple_Inc/
    │   │   ├── summary.txt
    │   │   ├── facts/
    │   │   ├── connections/
    │   │   └── chunks/
    │   └── ...
    ├── topics/
    │   ├── @search
    │   ├── Interest_Rates/
    │   │   ├── definition.txt
    │   │   └── chunks/
    │   └── ...
    ├── chunks/
    │   ├── @search
    │   └── chunk_*.txt
    └── documents/
        └── ...

Commands:
    pwd     - Print working directory
    cd      - Change directory
    ls      - List directory contents
    cat     - Read file contents
    grep    - Semantic search (vector + keyword)
    find    - Find files matching criteria
    head    - First N lines
    tail    - Last N lines
    wc      - Count items

Example:
    >>> shell = kg.shell()
    >>> shell.cd("/kg/entities/")
    >>> print(shell.grep("apple"))
    /kg/entities/Apple_Inc/: Technology company...
    >>> shell.cd("Apple_Inc/")
    >>> print(shell.cat("summary.txt"))

See: docs/architecture/FILESYSTEM_NAVIGATION_ARCHITECTURE.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zomma_kg.api.knowledge_graph import KnowledgeGraph


class KGShell:
    """
    Interactive shell for navigating a knowledge graph.

    Presents the KG as a virtual filesystem with familiar commands.
    Designed for both human exploration and LLM agent navigation.
    Placeholder status: command methods are roadmap items.
    """

    def __init__(self, kg: "KnowledgeGraph") -> None:
        """Initialize shell for a knowledge graph."""
        self._kg = kg
        self._cwd = "/kg/"
        self._history: list[str] = []
        self._prev_dir: str | None = None

    # === Navigation ===

    def pwd(self) -> str:
        """Print working directory."""
        return self._cwd

    def cd(self, path: str) -> str:
        """
        Change directory.

        Args:
            path: Absolute (/kg/entities/) or relative (../topics/) path

        Returns:
            New working directory path
        """
        raise NotImplementedError

    def ls(
        self,
        path: str | None = None,
        *,
        long: bool = False,
        all: bool = False,
        sort_by: str = "name",
    ) -> str:
        """
        List directory contents.

        Args:
            path: Path to list (default: current directory)
            long: Show detailed information (-l flag)
            all: Show hidden entries (-a flag)
            sort_by: Sort order: "name", "time", "size"
        """
        raise NotImplementedError

    # === Content Access ===

    def cat(self, path: str) -> str:
        """Read file contents."""
        raise NotImplementedError

    def head(self, path: str, lines: int = 10) -> str:
        """Return first N lines of a file."""
        raise NotImplementedError

    def tail(self, path: str, lines: int = 10) -> str:
        """Return last N lines of a file."""
        raise NotImplementedError

    # === Search ===

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        *,
        semantic: bool = True,
        case_insensitive: bool = True,
        context_lines: int = 0,
    ) -> str:
        """
        Search for pattern in files.

        When semantic=True (default), performs vector search.
        Otherwise performs literal pattern matching.

        See: docs/architecture/FILESYSTEM_NAVIGATION_ARCHITECTURE.md Section 5
        """
        raise NotImplementedError

    def find(
        self,
        path: str | None = None,
        *,
        name: str | None = None,
        type: str | None = None,
        entity_type: str | None = None,
        has_relationship: str | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> str:
        """
        Find files matching criteria.

        Args:
            path: Starting path (default: current directory)
            name: Name pattern with wildcards (* and ?)
            type: Entry type: "d" (directory) or "f" (file)
            entity_type: Filter by entity type
            has_relationship: Filter entities with specific relationship
            date_range: Filter by date (ISO format tuple)
        """
        raise NotImplementedError

    # === Statistics ===

    def wc(self, path: str | None = None) -> str:
        """Count items at the specified path."""
        raise NotImplementedError

    # === Session Management ===

    def history(self, limit: int = 20) -> list[str]:
        """Get command history for this session."""
        return self._history[-limit:]

    def back(self) -> str:
        """Go to previous directory (cd -)."""
        if self._prev_dir:
            return self.cd(self._prev_dir)
        return self._cwd

    # === Execution ===

    def execute(self, command: str) -> str:
        """
        Execute a shell command string.

        Parses and executes commands like "ls -l entities/" or
        "grep 'inflation' /kg/chunks/". Used by LLM agents.

        Maps commands to underlying DuckDB/LanceDB queries.
        See: docs/architecture/FILESYSTEM_NAVIGATION_ARCHITECTURE.md Section 5
        """
        self._history.append(command)
        raise NotImplementedError

    def execute_batch(self, commands: list[str]) -> list[str]:
        """Execute multiple commands, returning all outputs."""
        return [self.execute(cmd) for cmd in commands]
