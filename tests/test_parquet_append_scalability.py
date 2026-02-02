from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from zomma_kg.storage.parquet.backend import ParquetBackend
from zomma_kg.types import Chunk


def _chunk(uuid: str, position: int) -> Chunk:
    return Chunk(
        uuid=uuid,
        content=f"content-{uuid}",
        header_path="Section > Subsection",
        position=position,
        document_uuid="doc-1",
        document_date="2026-02-01",
    )


async def _init_backend(tmp_path: Path) -> ParquetBackend:
    backend = ParquetBackend(tmp_path / "kb")
    backend.kb_path.mkdir(parents=True, exist_ok=True)
    backend._write_metadata_if_missing()
    await backend._duckdb.initialize()
    return backend


@pytest.mark.asyncio
async def test_chunk_appends_use_dataset_parts_and_preserve_reads(tmp_path: Path) -> None:
    backend = await _init_backend(tmp_path)

    await backend.write_chunks([_chunk("chunk-1", 0)])
    await backend.write_chunks([_chunk("chunk-2", 1)])

    chunks_path = backend.kb_path / "chunks.parquet"
    assert chunks_path.is_dir()
    assert len(list(chunks_path.glob("*.parquet"))) == 2

    assert await backend.count_chunks() == 2
    chunk = await backend.get_chunk("chunk-2")
    assert chunk is not None
    assert chunk.content == "content-chunk-2"

    await backend._duckdb.close()


@pytest.mark.asyncio
async def test_chunk_append_migrates_legacy_single_file_to_dataset(tmp_path: Path) -> None:
    backend = await _init_backend(tmp_path)

    legacy_path = backend.kb_path / "chunks.parquet"
    legacy_data = {
        "uuid": ["chunk-legacy"],
        "content": ["legacy content"],
        "header_path": ["Legacy"],
        "position": [0],
        "document_uuid": ["doc-legacy"],
        "document_date": ["2026-01-31"],
        "group_id": [backend.group_id],
        "created_at": ["2026-01-31T00:00:00+00:00"],
    }
    pq.write_table(
        pa.Table.from_pydict(legacy_data, schema=ParquetBackend._chunk_schema()),
        legacy_path,
        compression="zstd",
    )
    assert legacy_path.is_file()

    await backend.write_chunks([_chunk("chunk-new", 1)])

    assert legacy_path.is_dir()
    assert len(list(legacy_path.glob("*.parquet"))) == 2
    assert await backend.count_chunks() == 2

    await backend._duckdb.close()


@pytest.mark.asyncio
async def test_chunk_append_guardrail_never_reads_existing_table(
    monkeypatch, tmp_path: Path
) -> None:
    backend = await _init_backend(tmp_path)
    await backend.write_chunks([_chunk("chunk-1", 0)])

    def _fail_read_table(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("append path should not read/concat existing parquet data")

    monkeypatch.setattr("zomma_kg.storage.parquet.backend.pq.read_table", _fail_read_table)
    await backend.write_chunks([_chunk("chunk-2", 1)])

    assert await backend.count_chunks() == 2

    await backend._duckdb.close()
