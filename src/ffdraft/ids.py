"""Reconcile the four player-ID systems in play.

nflverse uses gsis_id, ESPN uses an integer playerId, Sleeper uses its own
string id, and FantasyPros publishes names only. A silent mismatch here does
not raise -- it drops players or misattributes projections, and every
downstream number stays plausible while being wrong. Hence explicit tests
and an unmatched-rate check in validate.py.
"""

import re

import polars as pl

from .store import write

CROSSWALK_COLUMNS = ("gsis_id", "espn_id", "sleeper_id", "name", "position")

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and generational suffixes, squash spaces."""
    cleaned = name.lower().replace("-", " ")
    cleaned = re.sub(r"[^a-z\s]", "", cleaned)
    tokens = [t for t in cleaned.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def _with_key(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("_key")
    )


def match_by_name(df: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Left-join `df` (needs `name` and `position`) onto crosswalk IDs.

    Unmatched rows are kept with null IDs so callers can measure the miss rate
    rather than silently losing players.
    """
    left = _with_key(df)
    right = _with_key(crosswalk).select(
        ["_key", "position", "gsis_id", "espn_id", "sleeper_id"]
    )
    return left.join(right, on=["_key", "position"], how="left").drop("_key")


def load_crosswalk() -> pl.DataFrame:
    """Build the crosswalk from nflverse's ID table plus Sleeper's player dump."""
    try:
        import nflreadpy as nfl

        raw = nfl.load_ff_playerids()
        ids = raw if isinstance(raw, pl.DataFrame) else raw.to_polars()
    except ImportError:
        import nfl_data_py

        ids = pl.from_pandas(nfl_data_py.import_ids())

    ids = ids.select(
        pl.col("gsis_id"),
        pl.col("espn_id").cast(pl.Int64, strict=False),
        pl.col("sleeper_id").cast(pl.String, strict=False),
        pl.col("name"),
        pl.col("position"),
    ).filter(pl.col("name").is_not_null() & pl.col("position").is_not_null())

    # `unique(subset=["gsis_id"])` would treat every null gsis_id as an equal
    # duplicate of every other, collapsing thousands of real players (mostly
    # rookies/prospects not yet on an NFL roster) down to a single row. Only
    # dedupe rows that actually share a real gsis_id; keep every null-gsis_id
    # row so those players stay matchable by name/espn_id/sleeper_id.
    has_gsis = ids.filter(pl.col("gsis_id").is_not_null()).unique(subset=["gsis_id"])
    no_gsis = ids.filter(pl.col("gsis_id").is_null())
    return pl.concat([has_gsis, no_gsis])


def ingest() -> pl.DataFrame:
    df = load_crosswalk()
    write("id_crosswalk", df)
    return df
