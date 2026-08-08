"""Task 8, Stage 2: correlation impact measurement.

The season simulator (`sim/season.py`) samples every player's weekly score
**independently**. In reality, a QB and his own top receiver score
together (the same TD pass credits both), a shootout lifts both teams'
skill players, and a run-heavy blowout lifts one team's RB while
suppressing its own passing game. Independence understates the variance of
a *stacked* roster (one QB plus his own pass-catchers, or concentrated in
few NFL teams) relative to a *diversified* one with equal projected mean.

This script does **not** implement correlation in the production model. It
measures two things and reports whether the omission is material at the
precision Stage 3 needs (a few percentage points of championship
probability):

1. Real historical weekly-score correlations (from `weekly_stats` + game
   schedules), with a selection-bias-avoidance discipline described below.
2. The swing in simulated championship probability between a deliberately
   stacked roster and a deliberately diversified one of equal projected
   strength, first under the (production, unmodified) independent sampler,
   then under a correlated sampler built *only for this experiment*.

## Avoiding selection-bias in "who is WR1"

Naively defining "WR1" as the receiver with the most points *in the same
season* used to measure the correlation is circular: some of what makes a
receiver finish as WR1 (rather than WR2) in a given season is that he had
big games alongside his QB -- selecting on that outcome and then measuring
"his" correlation with the QB in those same games inflates the coefficient
with an artifact of the selection, not of week-to-week game dynamics.

This script avoids that by splitting each season in two: roles (QB1, WR1,
WR2, TE1 per team) are assigned from weeks 1-8 only (by total fantasy
points, minimum 3 games played), and the correlation is then measured on
weeks 9+ only, using that fixed, out-of-sample role assignment. The two
week ranges never overlap, so no correlation-measurement week was used to
decide who counts as "WR1." This is still a *within-season* role
definition (not a preseason depth chart), so some residual selection
remains -- e.g. a receiver who wins the WR1 role in the first half by
outplaying a teammate is, on average, having a better year, which could
itself correlate with QB play. That residual is far weaker than the
same-week circularity this design rules out, and is called out again in
the report.

## The correlated-sampling experiment: confined to this script

`_zig_ppf`, `_sample_correlated`, and `_simulate_season_correlated` below
implement a QB-anchored Gaussian-copula factor model, used only by this
script's `main()`. **None of this is imported by, or wired into,
`sim/season.py`'s production sampling path** (`_sample_all`) -- the
production simulator is used completely unmodified for the "independent"
half of the comparison, via its public `simulate_season` entry point.

Personal-data rule: this script never reads `.env` or `data/manager_labels.
csv`, and never names a real league team/manager (real 2026 NFL player and
team names are fine -- they are public data, not this league's roster
manager identities).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

import nflreadpy as nfl

from ffdraft.league import (
    N_TEAMS,
    PLAYOFF_BYES,
    PLAYOFF_ROUNDS,
    PLAYOFF_TEAMS,
    REGULAR_SEASON_WEEKS,
)
from ffdraft.models import distribution as dist_mod
from ffdraft.models.availability import availability_by_position, sample_availability_batch
from ffdraft.models.defense import dst_distribution
from ffdraft.models.distribution import ZeroInflatedGammaDistribution
from ffdraft.models.replacement import replacement_by_position
from ffdraft.models.roster import DraftPick, build_roster
from ffdraft.sim.lineup import build_replacement_means
from ffdraft.sim.season import (
    SeasonRosterPlayer,
    SeasonSimResult,
    _run_playoffs,
    _seed_teams,
    _team_weekly_totals,
    _validate_bracket_shape,
    build_regular_season_schedule,
    simulate_season,
)
from ffdraft.store import read

# ---------------------------------------------------------------------------
# Part 1: measured correlations
# ---------------------------------------------------------------------------

FIRST_HALF_MAX_WEEK = 8  # role definition window: weeks 1-8
MIN_GAMES_FOR_ROLE = 3  # exclude cameo/injury-shortened first halves
SKILL = ("QB", "RB", "WR", "TE")


def _team_game_table(seasons: list[int]) -> pl.DataFrame:
    """(season, week, team, opponent, game_id) for every regular-season team-game."""
    sched = nfl.load_schedules(seasons=seasons)
    sched = sched if isinstance(sched, pl.DataFrame) else sched.to_polars()
    reg = sched.filter(pl.col("game_type") == "REG")
    home = reg.select(
        pl.col("season"), pl.col("week"),
        pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent"),
        pl.col("game_id"),
    )
    away = reg.select(
        pl.col("season"), pl.col("week"),
        pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent"),
        pl.col("game_id"),
    )
    return pl.concat([home, away])


def _weekly_with_opponent(weekly: pl.DataFrame) -> pl.DataFrame:
    seasons = sorted(weekly["season"].unique().to_list())
    team_game = _team_game_table(seasons)
    return weekly.join(team_game, on=["season", "week", "team"], how="inner")


def _assign_roles(weekly_g: pl.DataFrame) -> pl.DataFrame:
    """Out-of-sample role table: (season, team, position, player_id, pos_rank),
    from weeks 1-`FIRST_HALF_MAX_WEEK` only. `pos_rank` 1 = that team-season's
    top scorer at that position over the first half."""
    first = weekly_g.filter(pl.col("week") <= FIRST_HALF_MAX_WEEK)
    agg = (
        first.group_by(["season", "team", "position", "player_id"])
        .agg(pl.col("fantasy_points").sum().alias("total_pts"), pl.len().alias("games"))
        .filter(pl.col("games") >= MIN_GAMES_FOR_ROLE)
    )
    return agg.with_columns(
        pl.col("total_pts")
        .rank(method="ordinal", descending=True)
        .over(["season", "team", "position"])
        .alias("pos_rank")
    )


@dataclass(frozen=True)
class CorrResult:
    label: str
    r: float
    n: int


def _teammate_correlation(
    weekly_g: pl.DataFrame, roles: pl.DataFrame,
    pos_a: str, rank_a: int, pos_b: str, rank_b: int, label: str,
) -> CorrResult:
    """Pearson r between two same-team players' weekly points, measured on
    weeks > FIRST_HALF_MAX_WEEK only, using roles assigned from the
    non-overlapping first half (see module docstring)."""
    second = weekly_g.filter(pl.col("week") > FIRST_HALF_MAX_WEEK)
    a_ids = roles.filter((pl.col("position") == pos_a) & (pl.col("pos_rank") == rank_a)).select(
        ["season", "team", "player_id"]
    )
    b_ids = roles.filter((pl.col("position") == pos_b) & (pl.col("pos_rank") == rank_b)).select(
        ["season", "team", "player_id"]
    )
    a_scores = second.join(a_ids, on=["season", "team", "player_id"], how="inner").select(
        ["season", "team", "week", "fantasy_points"]
    ).rename({"fantasy_points": "a_pts"})
    b_scores = second.join(b_ids, on=["season", "team", "player_id"], how="inner").select(
        ["season", "team", "week", "fantasy_points"]
    ).rename({"fantasy_points": "b_pts"})
    paired = a_scores.join(b_scores, on=["season", "team", "week"], how="inner")
    r, _p = stats.pearsonr(paired["a_pts"].to_numpy(), paired["b_pts"].to_numpy())
    return CorrResult(label, float(r), paired.height)


def _opposing_team_skill_total_correlation(weekly_g: pl.DataFrame) -> CorrResult:
    """Pearson r between the two teams' combined QB+RB+WR+TE points in the
    same game, weeks > FIRST_HALF_MAX_WEEK. No role selection at all here
    (whole-team totals), so this is immune to the WR1-definition selection
    concern by construction -- a useful cross-check on the QB-vs-QB number
    below, which does use role assignment."""
    second = weekly_g.filter(pl.col("week") > FIRST_HALF_MAX_WEEK)
    skill = second.filter(pl.col("position").is_in(list(SKILL)))
    team_week = skill.group_by(["season", "team", "week", "opponent", "game_id"]).agg(
        pl.col("fantasy_points").sum().alias("team_total")
    )
    a = team_week.rename({"team": "team_a", "team_total": "total_a", "opponent": "team_b"})
    b = team_week.rename({"team": "team_b2", "team_total": "total_b"})
    paired = a.join(
        b, left_on=["season", "week", "game_id", "team_b"],
        right_on=["season", "week", "game_id", "team_b2"], how="inner",
    ).filter(pl.col("team_a") < pl.col("team_b"))  # each game counted once
    r, _p = stats.pearsonr(paired["total_a"].to_numpy(), paired["total_b"].to_numpy())
    return CorrResult("opposing team skill-point totals (same game)", float(r), paired.height)


def _opposing_qb1_correlation(weekly_g: pl.DataFrame, roles: pl.DataFrame) -> CorrResult:
    second = weekly_g.filter(pl.col("week") > FIRST_HALF_MAX_WEEK)
    qb1 = roles.filter((pl.col("position") == "QB") & (pl.col("pos_rank") == 1)).select(
        ["season", "team", "player_id"]
    )
    qb_scores = second.join(qb1, on=["season", "team", "player_id"], how="inner").select(
        ["season", "team", "week", "opponent", "game_id", "fantasy_points"]
    )
    a = qb_scores.rename({"team": "team_a", "fantasy_points": "pts_a", "opponent": "team_b"})
    b = qb_scores.rename({"team": "team_b2", "fantasy_points": "pts_b"})
    paired = a.join(
        b, left_on=["season", "week", "game_id", "team_b"],
        right_on=["season", "week", "game_id", "team_b2"], how="inner",
    ).filter(pl.col("team_a") < pl.col("team_b"))
    r, _p = stats.pearsonr(paired["pts_a"].to_numpy(), paired["pts_b"].to_numpy())
    return CorrResult("opposing QB1 vs QB1 (same game)", float(r), paired.height)


def measure_correlations() -> dict[str, CorrResult]:
    weekly = read("weekly_stats")
    weekly_g = _weekly_with_opponent(weekly)
    roles = _assign_roles(weekly_g)

    results = {
        "qb_wr1": _teammate_correlation(weekly_g, roles, "QB", 1, "WR", 1, "QB1-WR1 (same team)"),
        "qb_te1": _teammate_correlation(weekly_g, roles, "QB", 1, "TE", 1, "QB1-TE1 (same team)"),
        "qb_rb1": _teammate_correlation(weekly_g, roles, "QB", 1, "RB", 1, "QB1-RB1 (same team)"),
        "wr1_wr2": _teammate_correlation(weekly_g, roles, "WR", 1, "WR", 2, "WR1-WR2 (same team)"),
        "opp_skill_total": _opposing_team_skill_total_correlation(weekly_g),
        "opp_qb1_qb1": _opposing_qb1_correlation(weekly_g, roles),
    }
    return results


def print_correlations(results: dict[str, CorrResult]) -> None:
    print("Measured weekly fantasy-point correlations")
    print(f"(roles assigned from weeks 1-{FIRST_HALF_MAX_WEEK}, min {MIN_GAMES_FOR_ROLE} games;")
    print(f" correlation measured on weeks {FIRST_HALF_MAX_WEEK + 1}+ only -- no overlap, see module docstring)")
    print(f"{'label':<42}{'r':>8}{'n':>8}")
    for res in results.values():
        print(f"{res.label:<42}{res.r:>8.3f}{res.n:>8}")


# ---------------------------------------------------------------------------
# Part 2: rosters -- a stacked one and a projected-equal-strength diversified
# one, built from the real 2026 player pool, plus 8 filler opponents.
# ---------------------------------------------------------------------------


def _pool_by_name(pool: dict[str, dist_mod.PlayerDistribution]) -> dict[tuple[str, str], dist_mod.PlayerDistribution]:
    return {(p.name, p.position): p for p in pool.values()}


# (name, position, team) -- team is documentation only, not read back by
# the simulator, which never sees NFL team identity (see league rule: never
# refer to a fantasy roster by name either -- these are real, public 2026 NFL
# player names, not this league's manager identities).
STACKED_SKILL = [
    ("Joe Burrow", "QB", "CIN"),
    ("Chase Brown", "RB", "CIN"),
    ("Jahmyr Gibbs", "RB", "DET"),
    ("Ja'Marr Chase", "WR", "CIN"),
    ("Amon-Ra St. Brown", "WR", "DET"),
    ("Sam LaPorta", "TE", "DET"),
    ("Tee Higgins", "WR", "CIN"),       # FLEX
    ("Jameson Williams", "WR", "DET"),  # FLEX
]  # 4 CIN + 4 DET -- every skill starter concentrated in exactly two NFL teams

DIVERSIFIED_SKILL = [
    ("Jalen Hurts", "QB", "PHI"),
    ("Christian McCaffrey", "RB", "SF"),
    ("Jonathan Taylor", "RB", "IND"),
    ("Justin Jefferson", "WR", "MIN"),
    ("CeeDee Lamb", "WR", "DAL"),
    ("Trey McBride", "TE", "ARI"),
    ("Zay Flowers", "WR", "BAL"),   # FLEX
    ("Rashee Rice", "WR", "KC"),    # FLEX
]  # 8 distinct NFL teams, projected mean within ~0.2% of the stacked roster


def _make_roster_picks(
    specs: list[tuple[str, str, str]],
    by_name: dict[tuple[str, str], dist_mod.PlayerDistribution],
) -> list[DraftPick]:
    """(name, position, team) specs -> `(player_id, position)` picks for
    `build_roster`. Every spec here is a real, current 2026 pool entry (see
    `STACKED_SKILL`/`DIVERSIFIED_SKILL`), so this never hits build_roster's
    fallback path -- it exists only to resolve the human-readable name/team
    spec down to the pool's own `player_id`."""
    return [(by_name[(name, position)].player_id, position) for name, position, _team in specs]


def _draft_filler_picks(
    pool: dict[str, dist_mod.PlayerDistribution],
    used_names: set[str],
    n_teams: int = 8,
) -> list[list[DraftPick]]:
    """8 reasonably strong, reasonably distinct opponent rosters' worth of
    picks, drafted best-available-by-position (round-robin across the
    `n_teams` teams) from the remaining pool. Not part of the measurement
    itself -- just a fixed, plausible backdrop of opponents so the
    championship-probability comparison isn't against an empty or
    degenerate league."""
    by_pos: dict[str, list[dist_mod.PlayerDistribution]] = {p: [] for p in SKILL}
    for pd_ in pool.values():
        if pd_.position in SKILL and pd_.name not in used_names:
            by_pos[pd_.position].append(pd_)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p.rank)

    picks: list[list[DraftPick]] = [[] for _ in range(n_teams)]

    def take(pos: str, count: int) -> None:
        for _ in range(count):
            for t in range(n_teams):
                pd_ = by_pos[pos].pop(0)
                picks[t].append((pd_.player_id, pos))

    take("QB", 1)
    take("RB", 2)
    take("WR", 2)
    take("TE", 1)

    flex_pool = sorted(by_pos["RB"] + by_pos["WR"] + by_pos["TE"], key=lambda p: p.rank)
    for _ in range(2):
        for t in range(n_teams):
            pd_ = flex_pool.pop(0)
            picks[t].append((pd_.player_id, pd_.position))

    return picks


def build_experiment_rosters() -> tuple[list[list[SeasonRosterPlayer]], dict[str, float], float]:
    """Returns (rosters, replacement_means, projected_mean_gap). `rosters[0]`
    is the stacked roster, `rosters[1]` the diversified roster, `rosters[2:]`
    are 8 filler opponents."""
    pool = dist_mod.build_player_pool()
    by_name = _pool_by_name(pool)
    availability = availability_by_position()
    dst_dist = dst_distribution()
    replacement = replacement_by_position()
    # build_roster's contract: replacement_by_position must carry a "DST"
    # entry, since every DST shares one distribution -- see models/roster.py.
    replacement_with_dst = {**replacement, "DST": dst_dist}

    kickers = sorted((p for p in pool.values() if p.position == "K"), key=lambda p: p.rank)

    stacked_picks = _make_roster_picks(STACKED_SKILL, by_name) + [(kickers[0].player_id, "K"), ("DST::stacked", "DST")]
    diversified_picks = _make_roster_picks(DIVERSIFIED_SKILL, by_name) + [
        (kickers[1].player_id, "K"), ("DST::diversified", "DST"),
    ]

    used_names = {n for n, _, _ in STACKED_SKILL} | {n for n, _, _ in DIVERSIFIED_SKILL}
    filler_picks = _draft_filler_picks(pool, used_names, n_teams=N_TEAMS - 2)
    for i, picks in enumerate(filler_picks):
        picks.append((kickers[2 + i].player_id, "K"))
        picks.append((f"DST::filler{i}", "DST"))

    all_picks = [stacked_picks, diversified_picks] + filler_picks
    rosters = [build_roster(picks, pool, replacement_with_dst, availability).players for picks in all_picks]
    stacked, diversified = rosters[0], rosters[1]

    replacement_means = build_replacement_means(replacement, dst_dist.mean)

    proj_stacked = sum(p.distribution.mean for p in stacked)
    proj_diversified = sum(p.distribution.mean for p in diversified)
    gap = proj_stacked - proj_diversified

    return rosters, replacement_means, gap


# ---------------------------------------------------------------------------
# Part 3: the correlated-sampling experiment (this script only -- see module
# docstring's "confined to this script" section).
# ---------------------------------------------------------------------------

# Loadings: correlation of each pass-catcher position's weekly score with its
# own team's QB, taken directly from Part 1's measured QB1-vs-position
# figures. RB uses the measured QB1-RB1 correlation. Two teammates who are
# not the QB (e.g. WR1 and WR2) are NOT given an independent shared factor --
# their induced correlation falls out as rho_pos_a * rho_pos_b (both riding
# the same QB latent). See module docstring / report for why this overstates
# WR-WR correlation specifically (measured ~0.03, implied ~0.14) and is a
# known, reported limitation of this single-QB-anchored factor model.
RHO_POS = {"WR": 0.372, "TE": 0.281, "RB": 0.070}

# Cross-team factor: correlation between the two opposing teams' QBs in a
# given game (measured directly, Part 1). Applied to regular-season weeks
# only -- playoff opponents are not known until standings are computed from
# the very scores being sampled, so this script does not attempt to inject
# cross-team correlation into playoff weeks (see `_draw_qb_latents`).
RHO_GAME = 0.187


def _zig_ppf(dist: ZeroInflatedGammaDistribution, q: np.ndarray) -> np.ndarray:
    """Inverse CDF of `ZeroInflatedGammaDistribution`, vectorized over `q` in
    (0, 1). Feeding `q = rng.random(...)` reproduces `dist.sample` exactly
    (same three-branch structure, same parameters) -- this is only a
    different *entry point* into the same distribution, so it changes
    nothing about a player's marginal weekly-score distribution. What
    changes in this experiment is what generates `q`: independent uniforms
    in production, correlated ones here (see `_sample_correlated`)."""
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)

    neg_mask = q < dist.p_negative
    zero_mask = (~neg_mask) & (q < dist.p_negative + dist.p_zero)
    pos_mask = ~(neg_mask | zero_mask)

    if neg_mask.any():
        qn = q[neg_mask] / dist.p_negative
        magnitude = stats.gamma.ppf(1.0 - qn, dist.neg_shape, scale=dist.neg_scale)
        out[neg_mask] = -magnitude
    out[zero_mask] = 0.0
    if pos_mask.any():
        p_positive = 1.0 - dist.p_negative - dist.p_zero
        qp = (q[pos_mask] - dist.p_negative - dist.p_zero) / p_positive
        out[pos_mask] = stats.gamma.ppf(qp, dist.gamma_shape, scale=dist.gamma_scale)
    return out


def _draw_qb_latents(
    n_teams: int, n_sims: int, n_weeks: int, regular_season_weeks: int,
    schedule: list[list[tuple[int, int]]], rho_game: float, rng: np.random.Generator,
) -> np.ndarray:
    """`(n_teams, n_sims, n_weeks)` standard-normal latents, one per
    (team, week): each team's own QB "form" that week, correlated `rho_game`
    with its scheduled opponent's during regular-season weeks (pairing known
    upfront), independent during playoff weeks (pairing not known until
    standings are computed from these same scores)."""
    idio = rng.standard_normal((n_teams, n_sims, n_weeks))
    latents = idio.copy()
    for w in range(regular_season_weeks):
        for a, b in schedule[w]:
            g = rng.standard_normal(n_sims)
            latents[a, :, w] = np.sqrt(rho_game) * g + np.sqrt(1 - rho_game) * idio[a, :, w]
            latents[b, :, w] = np.sqrt(rho_game) * g + np.sqrt(1 - rho_game) * idio[b, :, w]
    return latents


def _sample_correlated(
    rosters: list[list[SeasonRosterPlayer]],
    n_sims: int, n_weeks: int, regular_season_weeks: int,
    schedule: list[list[tuple[int, int]]],
    rng: np.random.Generator,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Same return shape/contract as `sim.season._sample_all`, but QB/RB/WR/TE
    scores are drawn from the QB-anchored copula factor model instead of
    independently. Availability, K, and DST are sampled exactly as
    production does (`sample_availability_batch` / `dist.sample`) --
    correlation is not modeled for those."""
    n_teams = len(rosters)
    qb_idx = [next((i for i, p in enumerate(r) if p.position == "QB"), None) for r in rosters]

    qb_latents = _draw_qb_latents(n_teams, n_sims, n_weeks, regular_season_weeks, schedule, RHO_GAME, rng)

    flat: list[tuple[int, int, SeasonRosterPlayer]] = [
        (t, i, p) for t, roster in enumerate(rosters) for i, p in enumerate(roster)
    ]
    avail_flat_idx = [k for k, (_, _, p) in enumerate(flat) if p.availability is not None]
    if avail_flat_idx:
        p_avail = np.array([flat[k][2].availability.p_available for k in avail_flat_idx])
        persistence = np.array([flat[k][2].availability.persistence for k in avail_flat_idx])
        avail_batch = sample_availability_batch(p_avail, persistence, rng, n_sims, n_weeks)
    else:
        avail_batch = np.empty((0, n_sims, n_weeks), dtype=bool)

    scores: list[list[np.ndarray | None]] = [[None] * len(r) for r in rosters]
    available: list[list[np.ndarray | None]] = [[None] * len(r) for r in rosters]

    avail_cursor = 0
    for t, i, player in flat:
        if player.position == "QB" and isinstance(player.distribution, ZeroInflatedGammaDistribution):
            q = stats.norm.cdf(qb_latents[t])
            scores[t][i] = _zig_ppf(player.distribution, q)
        elif player.position in RHO_POS and isinstance(player.distribution, ZeroInflatedGammaDistribution) and qb_idx[t] is not None:
            rho = RHO_POS[player.position]
            idio = rng.standard_normal((n_sims, n_weeks))
            latent = rho * qb_latents[t] + np.sqrt(1 - rho**2) * idio
            q = stats.norm.cdf(latent)
            scores[t][i] = _zig_ppf(player.distribution, q)
        else:
            # K, DST, a QB-less team, or a non-ZIG distribution (e.g. a
            # replacement-level fallback) -- sampled independently, exactly
            # as production does. None of this experiment's rosters hit
            # this branch for a skill position, but real rosters could.
            scores[t][i] = player.distribution.sample(rng, n_sims * n_weeks).reshape(n_sims, n_weeks)

        if player.availability is not None:
            available[t][i] = avail_batch[avail_cursor]
            avail_cursor += 1
        else:
            available[t][i] = np.ones((n_sims, n_weeks), dtype=bool)

    return scores, available  # type: ignore[return-value]


def simulate_season_correlated(
    rosters: list[list[SeasonRosterPlayer]],
    n_sims: int,
    seed: int,
    replacement_means: dict[str, float],
) -> SeasonSimResult:
    """Same orchestration as `sim.season.simulate_season` (schedule build,
    lineup solve, standings, playoffs -- all reused, unmodified, from
    production), with only the sampling step swapped for
    `_sample_correlated`. This function lives entirely in this script; it is
    not part of the package and Stage 3 does not import it."""
    from ffdraft.league import FLEX_ELIGIBLE, STARTERS

    n_teams = len(rosters)
    _validate_bracket_shape(PLAYOFF_TEAMS, PLAYOFF_ROUNDS, PLAYOFF_BYES, n_teams)
    schedule = build_regular_season_schedule(n_teams, REGULAR_SEASON_WEEKS)
    total_weeks = REGULAR_SEASON_WEEKS + PLAYOFF_ROUNDS
    rng = np.random.default_rng(seed)

    scores, available = _sample_correlated(rosters, n_sims, total_weeks, REGULAR_SEASON_WEEKS, schedule, rng)
    team_totals = _team_weekly_totals(
        rosters, scores, available, replacement_means, n_sims, total_weeks, STARTERS, FLEX_ELIGIBLE, hindsight=True,
    )
    team_totals = team_totals + rng.uniform(0, 1e-9, size=team_totals.shape)

    wins = np.zeros((n_teams, n_sims), dtype=float)
    points_for = np.zeros((n_teams, n_sims), dtype=float)
    for week, pairs in enumerate(schedule):
        for a, b in pairs:
            sa, sb = team_totals[a, :, week], team_totals[b, :, week]
            wins[a] += sa > sb
            wins[b] += sb > sa
            points_for[a] += sa
            points_for[b] += sb

    seeds = _seed_teams(wins, points_for, rng)
    champion = _run_playoffs(team_totals, seeds, REGULAR_SEASON_WEEKS, PLAYOFF_BYES)
    champion_counts = np.bincount(champion, minlength=n_teams)
    champion_seed = np.argmax(seeds == champion[None, :], axis=0)
    champion_seed_counts = np.bincount(champion_seed, minlength=PLAYOFF_TEAMS)[:PLAYOFF_TEAMS]

    return SeasonSimResult(
        champion_counts=tuple(int(c) for c in champion_counts),
        n_sims=n_sims,
        champion_seed_counts=tuple(int(c) for c in champion_seed_counts),
    )


def _calibration_check(pool: dict[str, dist_mod.PlayerDistribution], seed: int = 777) -> None:
    """Empirically verify what Pearson r the copula factor model actually
    produces for a representative QB-WR pair against the target `RHO_POS`
    value -- the copula's ppf transform is nonlinear (heavy-tailed mixture),
    so achieved correlation is not guaranteed to equal the input parameter
    exactly. Reported, not just assumed."""
    by_name = _pool_by_name(pool)
    qb = by_name[("Joe Burrow", "QB")].distribution
    wr = by_name[("Ja'Marr Chase", "WR")].distribution
    rng = np.random.default_rng(seed)
    n = 200_000
    rho = RHO_POS["WR"]
    z_qb = rng.standard_normal(n)
    z_wr = rho * z_qb + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    qb_scores = _zig_ppf(qb, stats.norm.cdf(z_qb))
    wr_scores = _zig_ppf(wr, stats.norm.cdf(z_wr))
    achieved_r, _ = stats.pearsonr(qb_scores, wr_scores)
    print(
        f"\nCopula calibration check (Burrow/Chase, n={n}): target rho={rho:.3f} "
        f"(input to the Gaussian copula) -> achieved Pearson r={achieved_r:.3f} "
        f"(measured on the output fantasy-point samples)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_SIMS = 50_000
SEED = 80200801


def _slot_averaged_gap(rosters, replacement_means, sim_fn, seed: int) -> tuple[float, float, float]:
    """`build_regular_season_schedule`'s circle method fixes team-index 0's
    "seat" and rotates everyone else (see `sim/season.py`'s schedule
    docstring) -- so which of the 10 roster-list slots a team occupies is
    not schedule-neutral. A first run of this experiment found a ~5pp
    stacked-vs-diversified gap under the *independent* model, which the task
    predicts should be ~0 -- swapping which slot (0 or 1) held which roster
    moved the gap by more than that, confirming a schedule-slot artifact,
    not a real roster-strength difference (both rosters' projected means are
    within 0.4%). Fixed here by running both slot assignments and averaging
    the (stacked - diversified) gap, which cancels the artifact since each
    roster spends equal simulated time in each slot.

    Returns `(gap_slot01, gap_slot10, averaged_gap)`.
    """
    swapped = [rosters[1], rosters[0]] + rosters[2:]
    res_01 = sim_fn(rosters, n_sims=N_SIMS, seed=seed, replacement_means=replacement_means)
    res_10 = sim_fn(swapped, n_sims=N_SIMS, seed=seed, replacement_means=replacement_means)
    gap_01 = res_01.championship_probabilities[0] - res_01.championship_probabilities[1]
    gap_10 = res_10.championship_probabilities[1] - res_10.championship_probabilities[0]
    return gap_01, gap_10, (gap_01 + gap_10) / 2.0


def main() -> None:
    print("=" * 78)
    print("PART 1: measured historical correlations")
    print("=" * 78)
    corr_results = measure_correlations()
    print_correlations(corr_results)

    print("\n" + "=" * 78)
    print("PART 2: stacked vs diversified roster, independent vs correlated sampling")
    print("=" * 78)
    rosters, replacement_means, proj_gap = build_experiment_rosters()
    stacked_mean = sum(p.distribution.mean for p in rosters[0])
    diversified_mean = sum(p.distribution.mean for p in rosters[1])
    print(f"Stacked roster projected weekly mean (8 skill starters + K + DST):      {stacked_mean:.2f}")
    print(f"Diversified roster projected weekly mean (8 skill starters + K + DST):  {diversified_mean:.2f}")
    print(f"Gap: {proj_gap:+.2f} points/week ({100 * proj_gap / diversified_mean:+.2f}%)")

    pool = dist_mod.build_player_pool()
    _calibration_check(pool)

    t0 = time.perf_counter()
    gap01_i, gap10_i, avg_gap_i = _slot_averaged_gap(rosters, replacement_means, simulate_season, SEED)
    t1 = time.perf_counter()
    gap01_c, gap10_c, avg_gap_c = _slot_averaged_gap(rosters, replacement_means, simulate_season_correlated, SEED)
    t2 = time.perf_counter()
    print(f"\nIndependent sims: {t1 - t0:.2f}s, correlated sims: {t2 - t1:.2f}s  (n_sims={N_SIMS} x 2 slot orders each)")

    print(f"\n{'':<28}{'slot(0,1)':>12}{'slot(1,0)':>12}{'slot-averaged':>16}")
    print(f"{'independent, stacked-div':<28}{100*gap01_i:>11.2f}%{100*gap10_i:>11.2f}%{100*avg_gap_i:>+15.2f}pp")
    print(f"{'correlated, stacked-div':<28}{100*gap01_c:>11.2f}%{100*gap10_c:>11.2f}%{100*avg_gap_c:>+15.2f}pp")

    # Conservative (p=0.5 worst case, ignoring the negative covariance between
    # two teams' championship indicators within a sim, which only shrinks
    # this further) binomial SE bound for the slot-averaged gap.
    se_bound = 0.5 / (N_SIMS ** 0.5)
    print(f"\nApprox. conservative binomial SE bound of a slot-averaged gap (n_sims={N_SIMS} per slot order): {100*se_bound:.2f}pp")

    swing = avg_gap_c - avg_gap_i
    print(f"\nCorrelation's effect on the slot-averaged stacked-vs-diversified championship-probability gap: {100*swing:+.2f} percentage points")
    print(f"(independent model: {100*avg_gap_i:+.2f}pp  ->  correlated model: {100*avg_gap_c:+.2f}pp)")


if __name__ == "__main__":
    main()
