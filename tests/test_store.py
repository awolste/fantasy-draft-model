import polars as pl
import pytest

from ffdraft import store


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def test_write_then_read_roundtrips(tmp_data_dir):
    df = pl.DataFrame({"player_id": ["A", "B"], "fantasy_points": [1.5, 2.5]})
    store.write("weekly_stats", df)
    out = store.read("weekly_stats")
    assert out.to_dicts() == df.to_dicts()


def test_write_creates_the_data_directory(tmp_path, monkeypatch):
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(store, "DATA_DIR", nested)
    store.write("thing", pl.DataFrame({"x": [1]}))
    assert (nested / "thing.parquet").exists()


def test_read_missing_dataset_raises_with_helpful_message(tmp_data_dir):
    with pytest.raises(FileNotFoundError, match="weekly_stats"):
        store.read("weekly_stats")


def test_exists_reports_presence(tmp_data_dir):
    assert store.exists("weekly_stats") is False
    store.write("weekly_stats", pl.DataFrame({"x": [1]}))
    assert store.exists("weekly_stats") is True
