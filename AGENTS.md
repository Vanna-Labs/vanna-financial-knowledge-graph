# Repository Guidelines

## Project Structure & Module Organization
- Core package code lives in `vanna_kg/`.
- Major domains: `ingestion/` (chunking, extraction, resolution, assembly), `query/` (GraphRAG pipeline), `storage/` (Parquet, DuckDB, LanceDB), `api/` (public facade), and `cli/` (Typer commands).
- Shared schemas and config are in `vanna_kg/types/` and `vanna_kg/config/`.
- Automated tests live in `tests/` (note: `pytest` is configured to run this folder by default).
- Utility/dev artifacts: `scripts/` for local workflows, `docs/` for design plans, and `test_data/` for sample inputs.

## Build, Test, and Development Commands
- Install dev environment: `uv pip install -e ".[dev,all]"`.
- Run all tests: `uv run pytest -q`.
- Run one module/test: `uv run pytest tests/test_entity_dedup.py -q`.
- Lint: `uv run ruff check vanna_kg tests`.
- Format: `uv run ruff format vanna_kg tests`.
- Type-check (strict): `uv run mypy vanna_kg`.
- CLI smoke check: `uv run vanna-kg --help`.
- End-to-end ingestion script example: `python scripts/build_kg.py test_data/BeigeBook_20251015.md --chunks 10`.

## Coding Style & Naming Conventions
- Target Python `>=3.10`, 4-space indentation, and explicit type hints.
- Ruff line length is `100`; keep imports sorted and style clean before opening PRs.
- MyPy runs in strict mode; avoid untyped public functions.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.

## Testing Guidelines
- Test framework: `pytest` with `pytest-asyncio` (`asyncio_mode = auto`).
- File and test names follow `test_*.py` and `test_*`.
- Prefer focused unit tests with mocked LLM/embedding/storage boundaries; avoid network-dependent tests.
- Add or update tests with every behavior change, especially in ingestion, query, and storage paths.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history, e.g. `feat(dedup): ...`, `fix(storage): ...`, `test(query): ...`.
- Keep commits/PRs scoped to one concern and include tests in the same change.
- PRs should include: what changed, why, files touched, verification commands/results, and risks/follow-ups.

## Configuration & Security Tips
- Keep secrets in `.env` (`OPENAI_API_KEY`, provider settings); never commit credentials.
- Use local KB output directories (for example `./test_kb`) for experiments and avoid committing large generated artifacts unless intentionally updating fixtures.
