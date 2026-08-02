"""Weekly NFL stat lines from nflverse, normalized to a canonical schema.

Upstream column names differ between nflreadpy and nfl_data_py and between
versions of each. All that variation is absorbed here so nothing downstream
ever sees an upstream column name.
"""

import polars as pl

from ..league import STATS_SEASONS
from ..scoring import add_fantasy_points
from ..store import write

CANONICAL_COLUMNS = (
    "player_id", "player_name", "position", "team", "season", "week",
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "special_teams_tds",
    "fumbles_lost", "two_pt_conversions",
    "fantasy_points",
)

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K"}

# Upstream name -> canonical name. Extra aliases are harmless; the first one
# present in the frame wins.
ALIASES: dict[str, tuple[str, ...]] = {
    "player_id": ("player_id", "gsis_id"),
    "player_name": ("player_display_name", "player_name"),
    "position": ("position",),
    "team": ("recent_team", "team"),
    "season": ("season",),
    "week": ("week",),
    "passing_yards": ("passing_yards",),
    "passing_tds": ("passing_tds",),
    "interceptions": ("interceptions", "passing_interceptions"),
    "rushing_yards": ("rushing_yards",),
    "rushing_tds": ("rushing_tds",),
    "receptions": ("receptions",),
    "targets": ("targets",),
    "receiving_yards": ("receiving_yards",),
    "receiving_tds": ("receiving_tds",),
    "special_teams_tds": ("special_teams_tds",),
}

# Summed rather than aliased, because upstream splits fumbles by play type.
FUMBLE_PARTS = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
TWO_PT_PARTS = (
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
)


def _resolve(df: pl.DataFrame, canonical: str) -> pl.Expr:
    for candidate in ALIASES[canonical]:
        if candidate in df.columns:
            return pl.col(candidate).alias(canonical)
    raise ValueError(
        f"No upstream column found for {canonical!r}. "
        f"Tried {ALIASES[canonical]}; available columns: {sorted(df.columns)}"
    )


def _sum_parts(df: pl.DataFrame, parts: tuple[str, ...], name: str) -> pl.Expr:
    present = [pl.col(p).fill_null(0.0) for p in parts if p in df.columns]
    if not present:
        return pl.lit(0.0).alias(name)
    return pl.sum_horizontal(present).alias(name)


def normalize_weekly(raw: pl.DataFrame) -> pl.DataFrame:
    df = raw.select(
        [_resolve(raw, c) for c in ALIASES]
        + [_sum_parts(raw, FUMBLE_PARTS, "fumbles_lost"),
           _sum_parts(raw, TWO_PT_PARTS, "two_pt_conversions")]
    )
    df = df.filter(
        pl.col("position").is_in(list(FANTASY_POSITIONS))
        & pl.col("player_id").is_not_null()
    )
    df = df.with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
    )
    df = add_fantasy_points(df)
    return df.select(list(CANONICAL_COLUMNS))


def load_raw(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Fetch from whichever nflverse package is installed."""
    try:
        import nflreadpy as nfl
    except ImportError:
        import nfl_data_py

        return pl.from_pandas(nfl_data_py.import_weekly_data(list(seasons)))

    data = nfl.load_player_stats(seasons=list(seasons))
    return data if isinstance(data, pl.DataFrame) else data.to_polars()


def ingest(seasons: tuple[int, ...] = STATS_SEASONS) -> pl.DataFrame:
    df = normalize_weekly(load_raw(seasons))
    write("weekly_stats", df)
    return df
