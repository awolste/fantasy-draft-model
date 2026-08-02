"""Shared nflreadpy/nfl_data_py fallback logic.

Both `sources/nflverse.py` (weekly stats) and `ids.py` (the ID crosswalk)
need to fetch from whichever nflverse package happens to be installed,
preferring nflreadpy and falling back to nfl_data_py. Centralized here so a
future API change or version pin only has to be handled in one place.
"""

from typing import Any, Callable

import polars as pl


def _to_polars(data: Any) -> pl.DataFrame:
    return data if isinstance(data, pl.DataFrame) else data.to_polars()


def load_from_nflverse(
    nflreadpy_fetch: Callable[[Any], Any],
    nfl_data_py_fetch: Callable[[Any], Any],
) -> pl.DataFrame:
    """Fetch a dataset from whichever nflverse package is installed.

    Tries nflreadpy first. `nflreadpy_fetch` receives the imported
    `nflreadpy` module and must return either a polars DataFrame or an
    object with a `.to_polars()` method.

    If nflreadpy is not installed, falls back to nfl_data_py:
    `nfl_data_py_fetch` receives the imported `nfl_data_py` module and must
    return a pandas DataFrame, which is converted to polars.

    The `except ImportError` is scoped to only the `import nflreadpy`
    statement itself -- deliberately narrow, so an ImportError raised
    inside `nflreadpy_fetch` (e.g. a transitive dependency missing) is
    never misread as "nflreadpy is not installed" and silently swallowed
    into a fallback.
    """
    try:
        import nflreadpy as nfl
    except ImportError:
        import nfl_data_py

        return pl.from_pandas(nfl_data_py_fetch(nfl_data_py))

    return _to_polars(nflreadpy_fetch(nfl))
