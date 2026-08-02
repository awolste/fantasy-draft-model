"""FantasyPros expert consensus rankings (ECR) and consensus ADP.

Rankings are embedded as a JSON blob in a script tag, not in the HTML table.
Note that FantasyPros PPR rankings assume 4-point passing touchdowns, so
quarterback ranks are systematically low for this league. That correction is
applied downstream in the player model, not here -- this module reports what
the source actually says.

The ADP page embeds its data the same way, as `window.FP.reportConfig = {...}`
rather than a plain HTML table. As of this writing that page also applies a
"registration fence": anonymous (unauthenticated) requests only receive the
first 5 rows of the report, with `registrationFence: true` in the payload
flagging the truncation. `parse_adp` reports exactly what the source returns;
it does not attempt to authenticate or otherwise work around the fence.
"""

import json
import re

import polars as pl
import requests

from ..store import write

ECR_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
ADP_URL = "https://www.fantasypros.com/nfl/adp/ppr-overall.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

RANKING_COLUMNS = ("rank", "name", "position", "team", "bye", "tier")
ADP_COLUMNS = ("name", "position", "team", "adp")

ECR_PATTERN = re.compile(r"var\s+ecrData\s*=\s*(\{.*?\});", re.DOTALL)
ADP_PATTERN = re.compile(r"window\.FP\.reportConfig\s*=\s*(\{.*?\});", re.DOTALL)
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


def _split_team_bye(team_and_bye: str) -> str:
    """'DET (6)' -> 'DET'. The ADP report packs bye week into the team field."""
    if not team_and_bye:
        return None
    return team_and_bye.split(" (")[0].strip() or None


def parse_adp(html: str) -> pl.DataFrame:
    """ADP is embedded as JSON in `window.FP.reportConfig`, like ECR's ecrData.

    Note: FantasyPros applies a registration fence to this report -- an
    unauthenticated request only gets the first few rows (the payload marks
    this explicitly with `registrationFence: true`). This function returns
    whatever rows the source actually provided; it does not paper over a
    truncated result.
    """
    match = ADP_PATTERN.search(html)
    if not match:
        raise ValueError(
            "Could not find `window.FP.reportConfig` in the FantasyPros ADP "
            "page. The page structure changed; inspect the saved HTML fixture."
        )
    config = json.loads(match.group(1))
    rows = config["table"]["rows"]

    names, positions, teams, adps = [], [], [], []
    for row in rows:
        player = row.get("player") or {}
        names.append(player.get("name"))
        teams.append(_split_team_bye(player.get("team", "")))
        positions.append(_bare_position(row.get("pos", "")))
        adp = row.get("avg")
        adps.append(float(adp) if adp is not None else None)

    return pl.DataFrame(
        {"name": names, "position": positions, "team": teams, "adp": adps},
        schema={"name": pl.String, "position": pl.String, "team": pl.String, "adp": pl.Float64},
    ).select(list(ADP_COLUMNS))


def _get(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def ingest() -> dict[str, pl.DataFrame]:
    frames = {
        "rankings_2026": parse_ecr(_get(ECR_URL)),
        "adp_2026": parse_adp(_get(ADP_URL)),
    }
    for name, df in frames.items():
        write(name, df)
    return frames
