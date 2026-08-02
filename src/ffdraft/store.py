"""Local parquet storage. One dataset per file, no database needed."""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def write(name: str, df: pl.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(name)
    df.write_parquet(path)
    return path


def read(name: str) -> pl.DataFrame:
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path}. "
            f"Run the ingest step that produces it first."
        )
    return pl.read_parquet(path)


def exists(name: str) -> bool:
    return _path(name).exists()
