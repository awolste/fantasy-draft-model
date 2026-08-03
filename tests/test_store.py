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


# ---------------------------------------------------------------------------
# fingerprint / check_cache_fresh: cache-staleness detection (Stage 3 Task 1
# Fix 1). A `load_or_fit_*` caches fitted parameters keyed on file existence
# alone unless it also records a fingerprint of its input data and checks it
# on load -- these tests exercise that fingerprint/check machinery directly
# (the per-module `load_or_fit_*` integration is tested alongside each of
# those functions instead, since it also depends on file writes).


def _weekly_like(points: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"player_id": [f"p{i}" for i in range(len(points))], "fantasy_points": points})


def test_fingerprint_is_stable_for_identical_content():
    a = _weekly_like([1.0, 2.0, 3.0])
    b = _weekly_like([1.0, 2.0, 3.0])
    assert store.fingerprint(a) == store.fingerprint(b)


def test_fingerprint_changes_when_row_values_change():
    a = _weekly_like([1.0, 2.0, 3.0])
    b = _weekly_like([1.0, 2.0, 4.0])  # same shape, one value differs
    assert store.fingerprint(a) != store.fingerprint(b)


def test_fingerprint_changes_when_row_count_changes():
    a = _weekly_like([1.0, 2.0, 3.0])
    b = _weekly_like([1.0, 2.0, 3.0, 4.0])
    assert store.fingerprint(a) != store.fingerprint(b)


def test_fingerprint_changes_when_schema_changes():
    a = pl.DataFrame({"x": [1, 2, 3]})
    b = pl.DataFrame({"x": [1.0, 2.0, 3.0]})  # same values, different dtype
    assert store.fingerprint(a) != store.fingerprint(b)


def test_fingerprint_combines_multiple_frames_positionally():
    a = _weekly_like([1.0, 2.0])
    b = _weekly_like([3.0, 4.0])
    assert store.fingerprint(a, b) != store.fingerprint(a)
    assert store.fingerprint(a, b) != store.fingerprint(b, a)


def test_fingerprint_is_stable_across_processes():
    """Must not rely on `id()`, Python's randomized `hash()`, or any other
    per-process/per-run source of nondeterminism -- a cache fingerprint
    written by one process has to be recognized by a different process
    reading it back later."""
    import subprocess
    import sys

    script = (
        "import polars as pl, sys; sys.path.insert(0, 'src'); "
        "from ffdraft import store; "
        "df = pl.DataFrame({'player_id': ['a', 'b', 'c'], 'fantasy_points': [1.0, 2.0, 3.0]}); "
        "print(store.fingerprint(df))"
    )
    results = [
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip()
        for _ in range(2)
    ]
    assert results[0] == results[1]
    assert results[0] != ""


def test_write_fingerprint_then_read_fingerprint_roundtrips(tmp_data_dir):
    store.write_fingerprint("thing", "abc123")
    assert store.read_fingerprint("thing") == "abc123"


def test_read_fingerprint_missing_returns_none(tmp_data_dir):
    assert store.read_fingerprint("thing") is None


def test_check_cache_fresh_does_nothing_when_no_fingerprint_stored(tmp_data_dir):
    # A cache written before this fingerprint machinery existed has no
    # fingerprint sidecar at all -- treated as trusted rather than forced
    # to refit, so upgrading this project doesn't invalidate every existing
    # cache on the next run.
    store.check_cache_fresh("thing", "any-fingerprint")  # must not raise


def test_check_cache_fresh_does_nothing_when_fingerprint_matches(tmp_data_dir):
    store.write_fingerprint("thing", "abc123")
    store.check_cache_fresh("thing", "abc123")  # must not raise


def test_check_cache_fresh_raises_with_actionable_message_on_mismatch(tmp_data_dir):
    store.write_fingerprint("thing", "abc123")
    with pytest.raises(store.CacheStaleError, match="force_refit"):
        store.check_cache_fresh("thing", "different-fingerprint")


# ---------------------------------------------------------------------------
# cache_namespace: two legitimate input contexts (Stage 3 Task 1 Fix 2)
# coexist under one base name instead of evicting each other.


def test_cache_namespace_is_stable_for_identical_context_frames():
    a = _weekly_like([1.0, 2.0])
    b = _weekly_like([1.0, 2.0])
    assert store.cache_namespace("thing", a) == store.cache_namespace("thing", b)


def test_cache_namespace_differs_for_different_context_frames():
    a = _weekly_like([1.0, 2.0])
    b = _weekly_like([3.0, 4.0])
    assert store.cache_namespace("thing", a) != store.cache_namespace("thing", b)


def test_cache_namespace_starts_with_base_name():
    a = _weekly_like([1.0, 2.0])
    ns = store.cache_namespace("thing", a)
    assert ns.startswith("thing__")


def test_two_contexts_coexist_without_evicting_each_other(tmp_data_dir):
    context_a = _weekly_like([1.0, 2.0])
    context_b = _weekly_like([3.0, 4.0, 5.0])
    name_a = store.cache_namespace("thing", context_a)
    name_b = store.cache_namespace("thing", context_b)

    store.write(name_a, pl.DataFrame({"value": ["a"]}))
    store.write_fingerprint(name_a, "fp-a")
    store.write(name_b, pl.DataFrame({"value": ["b"]}))
    store.write_fingerprint(name_b, "fp-b")

    # Writing/checking context B must not have disturbed context A's file
    # or fingerprint.
    store.check_cache_fresh(name_a, "fp-a")  # must not raise
    store.check_cache_fresh(name_b, "fp-b")  # must not raise
    assert store.read(name_a)["value"].to_list() == ["a"]
    assert store.read(name_b)["value"].to_list() == ["b"]


def test_changed_input_within_one_context_still_raises_for_that_context_only(tmp_data_dir):
    context_a = _weekly_like([1.0, 2.0])
    context_b = _weekly_like([3.0, 4.0, 5.0])
    name_a = store.cache_namespace("thing", context_a)
    name_b = store.cache_namespace("thing", context_b)

    store.write_fingerprint(name_a, "fp-a")
    store.write_fingerprint(name_b, "fp-b")

    with pytest.raises(store.CacheStaleError, match="force_refit"):
        store.check_cache_fresh(name_a, "fp-a-changed")
    store.check_cache_fresh(name_b, "fp-b")  # context B is untouched, still fresh
