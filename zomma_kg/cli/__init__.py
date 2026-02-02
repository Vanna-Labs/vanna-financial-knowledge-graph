"""
Command-Line Interface

CLI commands for ZommaKG operations.

Commands:
    zomma-kg ingest  - Ingest documents into a knowledge base
    zomma-kg query   - Query a knowledge base
    zomma-kg info    - Display knowledge base information
    zomma-kg shell   - Interactive navigation shell (placeholder)

Usage:
    # Ingest a PDF
    zomma-kg ingest report.pdf --kb ./my_kb

    # Ingest a directory
    zomma-kg ingest ./documents --kb ./my_kb --pattern "**/*.pdf"

    # Query
    zomma-kg query "What were the findings?" --kb ./my_kb

    # Show stats
    zomma-kg info --kb ./my_kb

See: docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 8
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

__all__ = ["main", "app"]

app = typer.Typer(
    name="zomma-kg",
    help="Embedded knowledge graph for document understanding",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(
        ...,
        help="File or directory to ingest",
        exists=True,
    ),
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
    ),
    pattern: str = typer.Option(
        "**/*.pdf",
        "--pattern", "-p",
        help="Glob pattern for directory ingestion",
    ),
    date: Optional[str] = typer.Option(
        None,
        "--date", "-d",
        help="Document date (YYYY-MM-DD)",
    ),
) -> None:
    """Ingest documents into a knowledge base."""

    async def _run() -> None:
        from zomma_kg.api.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(kb)

        try:
            if path.is_dir():
                # Directory ingestion
                files = list(path.glob(pattern))
                if not files:
                    console.print(f"[yellow]No files matching '{pattern}' found in {path}[/]")
                    return

                console.print(f"Found {len(files)} files to ingest")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Ingesting...", total=len(files))

                    total_entities = 0
                    total_facts = 0
                    total_chunks = 0

                    for file_path in files:
                        progress.update(task, description=f"Ingesting {file_path.name}")

                        if file_path.suffix.lower() == ".pdf":
                            result = await kg.ingest_pdf(file_path, document_date=date)
                        elif file_path.suffix.lower() in (".md", ".markdown"):
                            result = await kg.ingest_markdown(file_path, document_date=date)
                        else:
                            progress.advance(task)
                            continue

                        total_entities += result.entities
                        total_facts += result.facts
                        total_chunks += result.chunks
                        progress.advance(task)

                console.print()
                console.print(Panel(
                    f"[green]Successfully ingested {len(files)} files[/]\n\n"
                    f"  Chunks: {total_chunks}\n"
                    f"  Entities: {total_entities}\n"
                    f"  Facts: {total_facts}",
                    title="Ingestion Complete",
                ))
            else:
                # Single file ingestion
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Ingesting {path.name}...")

                    if path.suffix.lower() == ".pdf":
                        result = await kg.ingest_pdf(path, document_date=date)
                    elif path.suffix.lower() in (".md", ".markdown"):
                        result = await kg.ingest_markdown(path, document_date=date)
                    else:
                        console.print(f"[red]Unsupported file type: {path.suffix}[/]")
                        return

                    progress.update(task, completed=True)

                console.print()
                console.print(Panel(
                    f"[green]Successfully ingested {path.name}[/]\n\n"
                    f"  Document ID: {result.document_id}\n"
                    f"  Chunks: {result.chunks}\n"
                    f"  Entities: {result.entities}\n"
                    f"  Facts: {result.facts}\n"
                    f"  Duration: {result.duration_seconds:.1f}s",
                    title="Ingestion Complete",
                ))

                if result.errors:
                    console.print("[yellow]Warnings:[/]")
                    for error in result.errors:
                        console.print(f"  - {error}")
        finally:
            await kg.close()

    asyncio.run(_run())


@app.command()
def query(
    question: str = typer.Argument(
        ...,
        help="Question to ask the knowledge base",
    ),
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
        exists=True,
    ),
    sources: bool = typer.Option(
        True,
        "--sources/--no-sources",
        help="Include source citations",
    ),
) -> None:
    """Query the knowledge base."""

    async def _run() -> None:
        from zomma_kg.api.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(kb, create=False)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Thinking...")
                result = await kg.query(question, include_sources=sources)
                progress.update(task, completed=True)

            # Display answer
            console.print()
            console.print(Panel(
                Markdown(result.answer),
                title=f"Answer (confidence: {result.confidence:.0%})",
                border_style="green" if result.confidence > 0.7 else "yellow",
            ))

            # Display sources if available
            if sources and result.sources:
                console.print()
                table = Table(title="Sources")
                table.add_column("Document", style="cyan")
                table.add_column("Section", style="dim")

                seen = set()
                for source in result.sources[:5]:
                    key = (source.get("document", ""), source.get("section", ""))
                    if key not in seen:
                        seen.add(key)
                        table.add_row(
                            source.get("document", "Unknown"),
                            source.get("section", ""),
                        )

                console.print(table)

            # Display timing
            if result.timing:
                total_ms = sum(result.timing.values())
                console.print(f"\n[dim]Query time: {total_ms}ms[/]")

        finally:
            await kg.close()

    asyncio.run(_run())


@app.command()
def info(
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
        exists=True,
    ),
) -> None:
    """Display knowledge base information."""

    async def _run() -> None:
        from zomma_kg.api.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(kb, create=False)

        try:
            stats = await kg.stats()

            table = Table(title=f"Knowledge Base: {kb}")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right", style="green")

            table.add_row("Documents", str(stats["documents"]))
            table.add_row("Chunks", str(stats["chunks"]))
            table.add_row("Entities", str(stats["entities"]))
            table.add_row("Facts", str(stats["facts"]))

            console.print(table)

        finally:
            await kg.close()

    asyncio.run(_run())


@app.command()
def shell(
    kb: Path = typer.Option(
        Path("./kb"),
        "--kb", "-k",
        help="Knowledge base directory",
        exists=True,
    ),
) -> None:
    """Interactive navigation shell (coming soon)."""
    console.print("[yellow]Interactive shell not yet implemented.[/]")
    console.print("Use 'zomma-kg query' for now.")


def main() -> None:
    """Entry point for the CLI."""
    app()
