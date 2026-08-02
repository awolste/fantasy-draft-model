"""Turn raw ESPN season JSON into canonical frames.

Managers are keyed by owner SWID, never by team id or team name: ESPN team
ids are reassigned between seasons and names change constantly, but the
opponent model needs to follow the same human across eight drafts.
"""

import polars as pl

from ..league import LEAGUE_SEASONS, N_TEAMS
from ..store import write
from .espn import EspnClient

DRAFT_COLUMNS = ("season", "overall_pick", "round", "round_pick", "team_id", "espn_player_id")
MANAGER_COLUMNS = ("season", "team_id", "manager_id", "team_name")
RESULT_COLUMNS = ("season", "team_id", "wins", "losses", "points_for", "playoff_seed", "final_rank")


class MissingOwner(ValueError):
    """Raised when an ESPN team has no owners in the payload.

    A null manager_id would silently break the opponent model's per-manager
    tracking across seasons -- better to fail loudly here than to let a
    team with no owner quietly disappear from tenure/opponent analysis.
    """


def parse_draft(payload: dict, season: int) -> pl.DataFrame:
    picks = payload["draftDetail"]["picks"]
    return pl.DataFrame(
        {
            "season": [season] * len(picks),
            "overall_pick": [p["overallPickNumber"] for p in picks],
            "round": [p["roundId"] for p in picks],
            "round_pick": [p["roundPickNumber"] for p in picks],
            "team_id": [p["teamId"] for p in picks],
            "espn_player_id": [p["playerId"] for p in picks],
        },
        schema={
            "season": pl.Int64, "overall_pick": pl.Int64, "round": pl.Int64,
            "round_pick": pl.Int64, "team_id": pl.Int64, "espn_player_id": pl.Int64,
        },
    ).sort("overall_pick")


def _primary_owner(team: dict, season: int) -> str:
    owners = team.get("owners") or []
    if not owners:
        raise MissingOwner(
            f"Team id {team.get('id')} in season {season} has no owners in the "
            f"ESPN payload. Refusing to emit a null manager_id -- the opponent "
            f"model requires every team to be traceable to a person."
        )
    return owners[0]


def _team_name(team: dict) -> str:
    if team.get("name"):
        return team["name"]
    return f"{team.get('location', '')} {team.get('nickname', '')}".strip()


def parse_managers(payload: dict, season: int) -> pl.DataFrame:
    teams = payload["teams"]
    return pl.DataFrame(
        {
            "season": [season] * len(teams),
            "team_id": [t["id"] for t in teams],
            "manager_id": [_primary_owner(t, season) for t in teams],
            "team_name": [_team_name(t) for t in teams],
        },
        schema={
            "season": pl.Int64, "team_id": pl.Int64,
            "manager_id": pl.String, "team_name": pl.String,
        },
    )


def parse_results(payload: dict, season: int) -> pl.DataFrame:
    teams = payload["teams"]
    return pl.DataFrame(
        {
            "season": [season] * len(teams),
            "team_id": [t["id"] for t in teams],
            "wins": [t.get("record", {}).get("overall", {}).get("wins") for t in teams],
            "losses": [t.get("record", {}).get("overall", {}).get("losses") for t in teams],
            "points_for": [t.get("record", {}).get("overall", {}).get("pointsFor") for t in teams],
            "playoff_seed": [t.get("playoffSeed") for t in teams],
            "final_rank": [t.get("rankCalculatedFinal") for t in teams],
        },
        schema={
            "season": pl.Int64, "team_id": pl.Int64, "wins": pl.Int64, "losses": pl.Int64,
            "points_for": pl.Float64, "playoff_seed": pl.Int64, "final_rank": pl.Int64,
        },
    )


def ingest(seasons: tuple[int, ...] = LEAGUE_SEASONS) -> dict[str, pl.DataFrame]:
    client = EspnClient.from_env()
    current_season = max(LEAGUE_SEASONS)
    drafts, managers, results = [], [], []
    for season in seasons:
        payload = client.fetch_and_cache(
            season, ["mDraftDetail", "mTeam", "mSettings"], current_season=current_season
        )
        drafts.append(parse_draft(payload, season))
        managers.append(parse_managers(payload, season))
        results.append(parse_results(payload, season))

    frames = {
        "league_drafts": pl.concat(drafts),
        "league_managers": pl.concat(managers),
        "league_results": pl.concat(results),
    }
    for name, df in frames.items():
        write(name, df)
    return frames
