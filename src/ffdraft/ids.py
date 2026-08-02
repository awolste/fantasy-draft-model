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
    rather than silently losing players. The join is on (`_key`, `position`),
    so it is only cardinality-preserving if `crosswalk` has at most one row
    per (`_key`, `position`) pair -- `load_crosswalk` guarantees that via
    `_dedupe_by_name_position`. If a caller passes a crosswalk that still has
    duplicates, fail loudly rather than silently inflating the output, per
    this project's data-layer principle (see `_resolve` in
    `sources/nflverse.py`).
    """
    left = _with_key(df)
    right = _with_key(crosswalk).select(
        ["_key", "position", "gsis_id", "espn_id", "sleeper_id"]
    )
    out = left.join(right, on=["_key", "position"], how="left")
    if out.height != left.height:
        dupe_keys = (
            right.group_by(["_key", "position"])
            .len()
            .filter(pl.col("len") > 1)
            .join(left.select(["_key", "position"]), on=["_key", "position"], how="inner")
            .unique(["_key", "position"])
        )
        raise ValueError(
            f"match_by_name produced {out.height} rows from {left.height} input rows "
            f"-- the crosswalk has duplicate (name, position) keys for at least "
            f"{dupe_keys.height} of the input rows: "
            f"{dupe_keys.select(['_key', 'position']).to_dicts()[:10]}"
        )
    return out.drop("_key")


def _dedupe_by_name_position(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse crosswalk rows that share a normalized (name, position) key.

    `normalize_name` strips generational suffixes (Jr./Sr./II/III/...), so
    father-son pairs collide: e.g. "Marvin Harrison Jr." and "Marvin
    Harrison Sr." both normalize to "marvin harrison"/WR. `match_by_name`
    joins on exactly that pair, so an un-deduped crosswalk silently
    multiplies output rows -- a single input row for Marvin Harrison Jr.
    joined to 3 crosswalk rows and produced 3 output rows with no error.

    Preference order, most to least authoritative:
      1. Highest `draft_year` -- these collisions are overwhelmingly
         father/son pairs where the older namesake has retired and the
         younger one is the currently fantasy-relevant player (Harrison Sr.
         drafted 1996, Jr. drafted 2024; Beckham Sr. -- there is only one
         Odell Beckham here, but the same logic applies generally). A more
         recent draft year is the strongest signal of "this is the active
         player," and it's populated for all but ~1% of crosswalk rows.
      2. Non-null `gsis_id` -- means the player has actually appeared in an
         NFL game (nflverse only assigns gsis_id on roster appearance), so
         it's a good proxy for relevance when draft_year ties or is missing.
      3. Non-null `espn_id` -- same idea, weaker signal.
      4. `gsis_id`, then `espn_id`, then `name` ascending -- pure tiebreak so
         the result is reproducible across runs regardless of upstream row
         order, rather than depending on whatever order the source data
         happens to arrive in.
    """
    ranked = df.with_columns(
        pl.col("draft_year").fill_null(-1).alias("_draft_rank"),
        pl.col("gsis_id").is_not_null().alias("_has_gsis"),
        pl.col("espn_id").is_not_null().alias("_has_espn"),
    ).sort(
        ["_key", "position", "_draft_rank", "_has_gsis", "_has_espn", "gsis_id", "espn_id", "name"],
        descending=[False, False, True, True, True, False, False, False],
        nulls_last=True,
    )
    return ranked.unique(
        subset=["_key", "position"], keep="first", maintain_order=True
    ).drop(["_draft_rank", "_has_gsis", "_has_espn"])


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
        pl.col("draft_year").cast(pl.Int64, strict=False),
    ).filter(pl.col("name").is_not_null() & pl.col("position").is_not_null())

    # `unique(subset=["gsis_id"])` would treat every null gsis_id as an equal
    # duplicate of every other, collapsing thousands of real players (mostly
    # rookies/prospects not yet on an NFL roster) down to a single row. Only
    # dedupe rows that actually share a real gsis_id; keep every null-gsis_id
    # row so those players stay matchable by name/espn_id/sleeper_id.
    has_gsis = ids.filter(pl.col("gsis_id").is_not_null()).unique(subset=["gsis_id"])
    no_gsis = ids.filter(pl.col("gsis_id").is_null())
    ids = pl.concat([has_gsis, no_gsis])

    ids = _with_key(ids)
    ids = _dedupe_by_name_position(ids)
    return ids.drop(["_key", "draft_year"]).select(list(CROSSWALK_COLUMNS))


def ingest() -> pl.DataFrame:
    df = load_crosswalk()
    write("id_crosswalk", df)
    return df
