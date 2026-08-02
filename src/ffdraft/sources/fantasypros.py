"""FantasyPros expert consensus rankings (ECR).

Rankings are embedded as a JSON blob in a script tag, not in the HTML table.
Note that FantasyPros PPR rankings assume 4-point passing touchdowns, so
quarterback ranks are systematically low for this league. That correction is
applied downstream in the player model, not here -- this module reports what
the source actually says.

ADP used to be scraped from FantasyPros's ADP report too, but that page
fences anonymous requests to the first 5 rows (`registrationFence: true` in
the payload). ADP now comes from Fantasy Football Calculator's free public
API instead -- see sources/ffcalculator.py.
"""

import json
import re

import polars as pl
import requests

from ..store import write

ECR_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

RANKING_COLUMNS = ("rank", "name", "position", "team", "bye", "tier")

ECR_PATTERN = re.compile(r"var\s+ecrData\s*=\s*(\{.*?\});", re.DOTALL)
POSITION_PATTERN = re.compile(r"^([A-Z]+)")


def _bare_position(value: str) -> str:
    """'WR12' -> 'WR'. FantasyPros suffixes positional rank onto the position."""
    match = POSITION_PATTERN.match((value or "").upper())
    return match.group(1) if match else ""


def parse_ecr(html: str) -> pl.DataFrame:
    match = ECR_PATTERN.search(html)
    if not match:
        raise ValueError(
            "Could not find `var ecrData` in the FantasyPros page. "
            "The page structure changed; inspect the saved HTML fixture."
        )
    players = json.loads(match.group(1))["players"]
    df = pl.DataFrame(
        {
            "rank": [int(p["rank_ecr"]) for p in players],
            "name": [p["player_name"] for p in players],
            "position": [_bare_position(p.get("player_position_id", "")) for p in players],
            "team": [p.get("player_team_id") for p in players],
            "bye": [p.get("player_bye_week") for p in players],
            "tier": [p.get("tier") for p in players],
        },
        schema={
            "rank": pl.Int64, "name": pl.String, "position": pl.String,
            "team": pl.String, "bye": pl.String, "tier": pl.Int64,
        },
    )
    return df.select(list(RANKING_COLUMNS)).sort("rank")


def _get(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def ingest() -> dict[str, pl.DataFrame]:
    frames = {
        "rankings_2026": parse_ecr(_get(ECR_URL)),
    }
    for name, df in frames.items():
        write(name, df)
    return frames
