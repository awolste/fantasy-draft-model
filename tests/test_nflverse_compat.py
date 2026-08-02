import sys
import types

import pandas as pd
import polars as pl
import pytest

from ffdraft.sources._nflverse_compat import load_from_nflverse


def test_uses_nflreadpy_when_available():
    """nflreadpy is installed in this environment, so it must be preferred
    over nfl_data_py -- the nfl_data_py fetch must never be called."""
    calls = []

    def nflreadpy_fetch(nfl):
        calls.append("nflreadpy")
        return pl.DataFrame({"a": [1, 2]})

    def nfl_data_py_fetch(mod):
        raise AssertionError("nfl_data_py fetch must not be called when nflreadpy is present")

    out = load_from_nflverse(nflreadpy_fetch, nfl_data_py_fetch)
    assert calls == ["nflreadpy"]
    assert isinstance(out, pl.DataFrame)
    assert out["a"].to_list() == [1, 2]


def test_converts_non_polars_nflreadpy_result_via_to_polars():
    """nflreadpy can return an object that isn't already a polars
    DataFrame (e.g. a lazy/other-typed frame); it must be normalized via
    `.to_polars()`."""

    class FakeFrame:
        def to_polars(self) -> pl.DataFrame:
            return pl.DataFrame({"b": [3]})

    out = load_from_nflverse(lambda nfl: FakeFrame(), lambda mod: pd.DataFrame({"b": [0]}))
    assert out["b"].to_list() == [3]


def test_falls_back_to_nfl_data_py_when_nflreadpy_missing(monkeypatch):
    """When nflreadpy is not installed (ImportError on the import itself),
    fall back to nfl_data_py and convert its pandas result to polars."""
    monkeypatch.setitem(sys.modules, "nflreadpy", None)  # forces ImportError
    fake_nfl_data_py = types.ModuleType("nfl_data_py")
    monkeypatch.setitem(sys.modules, "nfl_data_py", fake_nfl_data_py)

    def nflreadpy_fetch(nfl):
        raise AssertionError("nflreadpy fetch must not be called when nflreadpy is missing")

    def nfl_data_py_fetch(mod):
        assert mod is fake_nfl_data_py
        return pd.DataFrame({"a": [1, 2]})

    out = load_from_nflverse(nflreadpy_fetch, nfl_data_py_fetch)
    assert isinstance(out, pl.DataFrame)
    assert out["a"].to_list() == [1, 2]


def test_import_error_inside_fetch_call_is_not_swallowed():
    """The except ImportError must be scoped to only the `import nflreadpy`
    statement. An ImportError raised inside the fetch callback (e.g. a
    transitive dependency missing) must propagate as-is, not be misread as
    'nflreadpy itself is not installed' and silently fall back to
    nfl_data_py."""

    def nflreadpy_fetch(nfl):
        raise ImportError("some transitive import failed inside the fetch call")

    def nfl_data_py_fetch(mod):
        raise AssertionError("must not fall back when the ImportError came from inside the fetch")

    with pytest.raises(ImportError, match="transitive import failed"):
        load_from_nflverse(nflreadpy_fetch, nfl_data_py_fetch)
