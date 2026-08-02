"""ADP (average draft position) from Fantasy Football Calculator's free,
public JSON API.

FantasyPros fences its ADP report behind registration (anonymous requests
return only the first 5 rows). Fantasy Football Calculator publishes the
same kind of data with no auth required, so that's the source used here.

Verified live against the real endpoint:
- Player fields are exactly: adp, adp_formatted, bye, high, low, name,
  player_id, position, stdev, team, times_drafted.
- Position values are "DEF" and "PK" for defense/kicker (not "DST"/"K"),
  and the skill positions ("QB", "RB", "WR", "TE") are already bare.
- The `teams` query parameter does not filter results -- teams=10 and
  teams=12 return identical payloads. Nothing here depends on it.
- 2025 has no data upstream: every scoring type returns
  {"status": "Error", "errors": "No ADP data found.", "players": []} for
  year=2025. `parse_adp` raises rather than silently returning an empty
  frame for a season like that -- see `_resolve` in sources/nflverse.py for
  why silent-empty is the failure mode this project guards against.
"""

import re

import polars as pl
import requests

from ..store import write

ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
HEADERS = {"User-Agent": "Mozilla/5.0"}

ADP_COLUMNS = ("name", "position", "team", "adp", "adp_stdev", "times_drafted", "season")

# Upstream position codes that differ from this project's canonical set.
POSITION_ALIASES = {"DEF": "DST", "PK": "K"}
POSITION_PATTERN = re.compile(r"^([A-Z]+)")

# History seasons: 2018-2024 inclusive. 2025 is skipped -- the upstream API
# has no ADP data for it (every scoring type errors for year=2025). 2026 is
# the current draft year and is written separately as `adp_2026`.
HISTORY_SEASONS: tuple[int, ...] = tuple(range(2018, 2025))
CURRENT_SEASON = 2026


def _bare_position(value: str) -> str:
    """Normalize an upstream position code to this project's canonical set."""
    match = POSITION_PATTERN.match((value or "").upper())
    bare = match.group(1) if match else ""
    return POSITION_ALIASES.get(bare, bare)


def parse_adp(payload: dict, season: int) -> pl.DataFrame:
    """Parse an already-fetched JSON payload from the FFC ADP API.

    Pure function: takes parsed JSON so it's testable from a saved fixture
    with no network call. Raises ValueError, naming the season, if the
    response isn't a success or has no players -- e.g. the 2025 gap, where
    the API returns status "Error" with an empty players list.
    """
    if payload.get("status") != "Success":
        raise ValueError(
            f"FantasyFootballCalculator ADP request for season {season} did not "
            f"succeed: status={payload.get('status')!r}, "
            f"errors={payload.get('errors')!r}"
        )
    players = payload.get("players") or []
    if not players:
        raise ValueError(
            f"FantasyFootballCalculator ADP response for season {season} has no "
            f"players. Silent empty frames hide real ingestion failures, so this "
            f"raises instead."
        )

    df = pl.DataFrame(
        {
            "name": [p.get("name") for p in players],
            "position": [_bare_position(p.get("position", "")) for p in players],
            "team": [p.get("team") for p in players],
            "adp": [float(p["adp"]) if p.get("adp") is not None else None for p in players],
            "adp_stdev": [
                float(p["stdev"]) if p.get("stdev") is not None else None for p in players
            ],
            "times_drafted": [p.get("times_drafted") for p in players],
        },
        schema={
            "name": pl.String, "position": pl.String, "team": pl.String,
            "adp": pl.Float64, "adp_stdev": pl.Float64, "times_drafted": pl.Int64,
        },
    ).with_columns(pl.lit(season).alias("season").cast(pl.Int64))

    return df.select(list(ADP_COLUMNS)).sort("adp")


def fetch_adp(season: int) -> dict:
    response = requests.get(
        ADP_URL, params={"teams": 12, "year": season}, headers=HEADERS, timeout=30
    )
    response.raise_for_status()
    return response.json()


def ingest(seasons: tuple[int, ...] = HISTORY_SEASONS) -> dict[str, pl.DataFrame]:
    history = pl.concat(
        [parse_adp(fetch_adp(season), season) for season in seasons]
    )
    current = parse_adp(fetch_adp(CURRENT_SEASON), CURRENT_SEASON)

    frames = {"adp_history": history, f"adp_{CURRENT_SEASON}": current}
    for name, df in frames.items():
        write(name, df)
    return frames
