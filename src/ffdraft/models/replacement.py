"""Waiver-wire replacement level: what a manager can realistically start when
a rostered player is unavailable (Task 4) and the bench cannot cover it.

## The central trap, and why this module does not fall into it

Replacement level is *not* the average of deep-ranked players. Task 3's
per-player model correctly projects a 176th-ranked WR at ~1.2 ppg -- that is
the right expectation for *that specific player*. But a manager scanning
waivers on a Tuesday does not pick a random deep player; they pick the best
plausible option using injury news, snap-count trends, and matchup context.
That is a dramatically better outcome than the mean of everyone left on the
wire, and it is an **order-statistic** question, not a mean.

This module measures replacement level directly from history: for each
season/week/position, who was *not* rostered in this league's 10-team,
18-man-roster depth, and what did the best plausible few of them actually
score that week.

## Step 1: who was "rostered" (the allocation across positions)

180 total roster spots (10 teams x 18), of which 20 are K/DST (1 each per
team), leaving ~160 skill (QB/RB/WR/TE) spots. The plan doc asks this
allocation to be decided and justified -- rather than guess a split, this
module measures it directly from **this league's own 8 seasons of real
drafts** (`league_drafts`, 2018-2025): every pick is matched via
`espn_player_id` -> crosswalk `espn_id` -> `gsis_id`, and each drafted
player's position is read off `weekly_stats` (their actual, empirically
observed position -- not the crosswalk's own `position` field, which labels
kickers "PK" and would otherwise require a second normalization step this
sidesteps entirely). This gives the *real* per-season, per-position count of
players this league actually rostered, not an assumed formula -- e.g.
2022's actual counts were QB 18, RB 52, WR 60, TE 17, K 11 (roster size grew
from 17 to 18 rounds in 2023, so counts are computed **per season**, not
pooled, so the fitted "rostered" set automatically reflects each season's
real roster depth).

A player counts as "rostered" for the whole season once drafted (an ex-ante,
static definition -- see `PlayerAvailability`'s docstring in
`availability.py` for why ex-ante selection avoids survivorship bias here
too). In-season waiver churn means some of the "available" pool below is
occasionally a player later added off waivers elsewhere in the league; that
is not a flaw, it is realistic -- the best waiver adds *do* get rostered
during the season, and this module is measuring exactly that kind of
opportunity.

## Step 2: the estimator, and the manager-skill assumption

Naive approaches both fail:
- **max** of the available pool each week assumes a manager has perfect
  foresight of who is about to have a spike week -- unrealistic.
- **mean** of the available pool assumes a manager picks at random --
  also unrealistic, and this is exactly the trap the plan doc warns about
  (verified directly: the raw available-pool mean for RB is ~4.3 ppg,
  nowhere close to what an attentive manager captures).
- **top-K of that week's own realized scores** (tried first, and reported
  here as a cautionary finding) is *also* a form of look-ahead: ranking
  ~90-140 available players by what they actually scored *that week* and
  averaging the top 3-5 is closer to "manager has a crystal ball for this
  week's box scores" than to real waiver behavior -- it produced WR
  replacement above 20 ppg, comfortably *above* a rostered WR2's median.
  That is the same order of error as the trap in the other direction, and
  it is why this module does not use it.

**What this module actually does**: for each (season, week, position),
among the available pool, rank candidates by their **trailing 3-week
average** (`TRAILING_WEEKS`) heading into that week -- information a real
manager plausibly has (recent role, recent production) -- then take the
**top 5** (`TOP_K`) of that ex-ante ranking and average their *actual*
score in the current week. This directly encodes the manager-skill
assumption stated plainly: **a manager identifies a short list of ~5
in-form, currently-productive available players using recent performance,
and their actual pickup lands among that short list -- not always the
single best (they don't have next week's box score), but reliably better
than an average shot in the dark.** Players with no games in the trailing
window are excluded from the candidate ranking (a manager cannot rank a
totally unknown quantity as one of their top plausible adds), which also
means the first week of a season (no trailing history) is excluded from
the fitted sample -- see `weekly_replacement_values`.

This estimator was checked against three sanity anchors before being
accepted (see the module-level report in the Task 5 writeup): it lands
clearly between the deep-pool mean and the rostered-starter median for
every position, it is never negative, and it reproduces the plan doc's
qualitative streaming finding for K (see `replacement_by_position`).

## Interface

`ReplacementLevelDistribution` satisfies `WeeklyDistribution` (`models/
base.py`) via bootstrap resampling of the empirical (season, week)
replacement-level observations -- this is deliberately non-parametric
(no invented tail shape) and satisfies Step 2's "vary by week, not a
single constant" requirement: two `sample()` calls draw different
week-like values because real historical weeks varied, not because the
literal calendar week is tracked. `sample` is a single vectorized
`rng.choice` call (no Python loop), matching every other distribution in
this package's performance contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..league import N_TEAMS, REGULAR_SEASON_WEEKS, STARTERS
from ..store import exists, read, write
from .base import WeeklyDistribution

POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K")

TRAILING_WEEKS = 3  # L: how many prior weeks define a candidate's "recent form"
TOP_K = 5  # K: size of the ex-ante shortlist a manager is assumed to work from

MIN_SEASON_WEEK_OBS = 30  # else the fitted empirical distribution is too thin to trust

# "Starter" comparison count per position for reporting/tests: starting-slot
# count x N_TEAMS, ignoring FLEX's cross-position share -- a deliberately
# simple baseline (see `median_rostered_starter`), not a load-bearing model
# input.
STARTER_COMPARISON_N: dict[str, int] = {
    pos: STARTERS[pos] * N_TEAMS for pos in POSITIONS if pos in STARTERS
}


# ---------------------------------------------------------------------------
# Step 1: who was rostered, per season, per position -- from this league's
# actual draft history, not an assumed split.


def _player_positions(weekly: pl.DataFrame) -> pl.DataFrame:
    """(player_id, position) for every player, using each player's most
    common `weekly_stats` position label -- avoids the crosswalk's own
    "PK" vs "K" labeling mismatch entirely, since it never touches the
    crosswalk's `position` column."""
    counts = weekly.group_by(["player_id", "position"]).agg(pl.len().alias("n"))
    return (
        counts.sort("n", descending=True)
        .unique(subset=["player_id"], keep="first")
        .select(["player_id", "position"])
    )


def drafted_player_ids(
    league_drafts: pl.DataFrame, crosswalk: pl.DataFrame, weekly: pl.DataFrame
) -> pl.DataFrame:
    """(season, position, player_id) for every player actually drafted onto
    one of this league's 10 rosters that season.

    `league_drafts.espn_player_id` -> crosswalk `espn_id` -> `gsis_id`
    resolves identity; position comes from `weekly_stats` (see
    `_player_positions`). DST picks (ESPN's negative sentinel player IDs for
    team defenses) and any pick that fails to resolve a `gsis_id` or never
    appears in `weekly_stats` are dropped -- this module only covers the
    five positions in `POSITIONS`.
    """
    resolved = league_drafts.join(
        crosswalk.select(["espn_id", "gsis_id"]),
        left_on="espn_player_id",
        right_on="espn_id",
        how="inner",
    ).filter(pl.col("gsis_id").is_not_null())

    positions = _player_positions(weekly)
    with_position = resolved.join(
        positions, left_on="gsis_id", right_on="player_id", how="inner"
    ).filter(pl.col("position").is_in(POSITIONS))

    return with_position.select(
        pl.col("season"), pl.col("position"), pl.col("gsis_id").alias("player_id")
    ).unique()


# ---------------------------------------------------------------------------
# Step 2: the weekly replacement-level estimator


def weekly_replacement_values(
    weekly: pl.DataFrame,
    drafted: pl.DataFrame,
    positions: tuple[str, ...] = POSITIONS,
    trailing_weeks: int = TRAILING_WEEKS,
    top_k: int = TOP_K,
    max_week: int = REGULAR_SEASON_WEEKS,
) -> pl.DataFrame:
    """One row per (season, week, position) with the fitted replacement
    value: the mean actual score of the top `top_k` available players,
    ranked by their trailing `trailing_weeks`-week average heading into
    that week. See the module docstring, Step 2, for the estimator and the
    manager-skill assumption it encodes.

    Week 1 of every season is excluded (no trailing history exists yet to
    rank candidates ex-ante) -- `ReplacementLevelDistribution` is a pooled
    empirical distribution across weeks, not indexed by literal calendar
    week, so this does not leave a gap a caller could hit; see the module
    docstring's Interface section.
    """
    weekly = weekly.filter(pl.col("week") <= max_week).select(
        ["season", "week", "player_id", "position", "fantasy_points"]
    )

    # Seasons this league actually has draft data for, independent of
    # position -- distinguishes "no draft happened this season" (skip; we
    # cannot define "available" at all) from "zero players of this position
    # were drafted this season" (a legitimate, if unusual, empty roster set
    # that should still leave every appearing player "available").
    seasons_with_draft_data = set(drafted["season"].unique().to_list())

    rows: list[dict] = []
    for position in positions:
        pos_drafted = drafted.filter(pl.col("position") == position)
        pos_weekly = weekly.filter(pl.col("position") == position)
        for season in sorted(pos_weekly["season"].unique().to_list()):
            if seasons_with_draft_data and season not in seasons_with_draft_data:
                continue
            rostered_ids = set(
                pos_drafted.filter(pl.col("season") == season)["player_id"].to_list()
            )
            season_weekly = pos_weekly.filter(pl.col("season") == season)
            available = season_weekly.filter(~pl.col("player_id").is_in(rostered_ids))

            for week in range(2, max_week + 1):
                current = available.filter(pl.col("week") == week)
                if current.height == 0:
                    continue
                trailing = available.filter(
                    (pl.col("week") < week) & (pl.col("week") >= week - trailing_weeks)
                )
                if trailing.height == 0:
                    continue
                trailing_form = trailing.group_by("player_id").agg(
                    pl.col("fantasy_points").mean().alias("trailing_avg")
                )
                candidates = current.join(trailing_form, on="player_id", how="inner")
                if candidates.height == 0:
                    continue
                shortlist = candidates.sort("trailing_avg", descending=True).head(top_k)
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "position": position,
                        "value": float(shortlist["fantasy_points"].mean()),
                        "n_candidates": candidates.height,
                    }
                )

    if not rows:
        raise ValueError("weekly_replacement_values produced no observations at all")
    return pl.DataFrame(rows)


def load_or_fit_replacement_values(
    weekly: pl.DataFrame | None = None,
    league_drafts: pl.DataFrame | None = None,
    crosswalk: pl.DataFrame | None = None,
    force_refit: bool = False,
) -> pl.DataFrame:
    if not force_refit and exists("replacement_weekly_values"):
        return read("replacement_weekly_values")

    if weekly is None:
        weekly = read("weekly_stats")
    if league_drafts is None:
        league_drafts = read("league_drafts")
    if crosswalk is None:
        crosswalk = read("id_crosswalk")

    drafted = drafted_player_ids(league_drafts, crosswalk, weekly)
    table = weekly_replacement_values(weekly, drafted)

    for position in POSITIONS:
        n = table.filter(pl.col("position") == position).height
        if n < MIN_SEASON_WEEK_OBS:
            raise ValueError(
                f"Only {n} (season, week) replacement observations for {position} -- "
                f"too few to trust a fitted distribution."
            )

    write("replacement_weekly_values", table)
    return table


# ---------------------------------------------------------------------------
# The distribution itself


@dataclass(frozen=True)
class ReplacementLevelDistribution:
    """A position's replacement-level weekly points, as an empirical
    distribution over historical (season, week) observations.

    `sample` bootstrap-resamples from the fitted historical values -- a
    deliberately non-parametric choice given ~100-140 observations per
    position; no tail shape is assumed beyond what history actually showed.
    This satisfies Step 2's "varies by week" requirement: successive samples
    are not a constant, because real historical weeks were not constant,
    even though no single sample is pinned to a specific calendar week.
    """

    position: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"{self.position}: no replacement-level observations to sample from")

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(np.asarray(self.values, dtype=float), size=size, replace=True)


def replacement_by_position(
    weekly: pl.DataFrame | None = None,
    league_drafts: pl.DataFrame | None = None,
    crosswalk: pl.DataFrame | None = None,
    force_refit: bool = False,
) -> dict[str, ReplacementLevelDistribution]:
    """Every position's fitted `ReplacementLevelDistribution`, keyed by
    position."""
    table = load_or_fit_replacement_values(
        weekly=weekly, league_drafts=league_drafts, crosswalk=crosswalk, force_refit=force_refit
    )
    out = {}
    for position in POSITIONS:
        values = tuple(table.filter(pl.col("position") == position)["value"].to_list())
        out[position] = ReplacementLevelDistribution(position=position, values=values)
    return out


# ---------------------------------------------------------------------------
# Reporting / regression-guard helpers: the three-number comparison
# (replacement vs. median rostered starter vs. deep-pool average) that makes
# the central trap visible rather than assumed away.


def median_rostered_starter(weekly: pl.DataFrame, max_week: int = REGULAR_SEASON_WEEKS) -> pl.DataFrame:
    """Median weekly score of each position's top `STARTER_COMPARISON_N`
    players that season, ranked by season-long average -- a simple "what a
    typical starter actually scores" baseline for comparison. Deliberately
    not the same cohort as `drafted_player_ids` (which is the whole 18-man
    roster, bench included): this is restricted to the starter-count tier
    specifically, so it is not diluted by inactive bench/IR depth."""
    weekly = weekly.filter((pl.col("week") <= max_week) & pl.col("position").is_in(POSITIONS))
    season_avg = (
        weekly.group_by(["season", "position", "player_id"])
        .agg(pl.col("fantasy_points").mean().alias("avg"), pl.len().alias("g"))
        .filter(pl.col("g") >= 4)
    )
    season_avg = season_avg.with_columns(
        pl.col("avg").rank(method="ordinal", descending=True).over(["season", "position"]).alias("rk")
    )

    rows = []
    for position, n in STARTER_COMPARISON_N.items():
        top_ids = season_avg.filter((pl.col("position") == position) & (pl.col("rk") <= n)).select(
            ["season", "player_id"]
        )
        matched = weekly.filter(pl.col("position") == position).join(
            top_ids, on=["season", "player_id"], how="inner"
        )
        rows.append({"position": position, "median_starter": float(matched["fantasy_points"].median())})
    return pl.DataFrame(rows)


def deep_pool_average(weekly: pl.DataFrame, drafted: pl.DataFrame, max_week: int = REGULAR_SEASON_WEEKS) -> pl.DataFrame:
    """Plain mean weekly score of everyone *not* rostered -- "the trap"
    number: what replacement level would look like if a manager picked at
    random off the whole waiver wire instead of choosing well."""
    weekly = weekly.filter((pl.col("week") <= max_week) & pl.col("position").is_in(POSITIONS))
    rows = []
    for position in POSITIONS:
        pos_drafted = drafted.filter(pl.col("position") == position)
        pos_weekly = weekly.filter(pl.col("position") == position)
        vals = []
        for season in sorted(pos_weekly["season"].unique().to_list()):
            rostered_ids = set(pos_drafted.filter(pl.col("season") == season)["player_id"].to_list())
            if not rostered_ids:
                continue
            season_weekly = pos_weekly.filter(pl.col("season") == season)
            avail = season_weekly.filter(~pl.col("player_id").is_in(rostered_ids))
            if avail.height:
                vals.append(float(avail["fantasy_points"].mean()))
        rows.append({"position": position, "deep_pool_avg": float(np.mean(vals)) if vals else None})
    return pl.DataFrame(rows)
