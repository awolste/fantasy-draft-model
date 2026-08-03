"""Per-player, per-week availability (the probability of playing at all).

`distribution.py` (Task 3) models "played but scored X" -- its zero-inflation
term is the observed *within-appearance* zero rate. `weekly_stats` only has
a row for games a player actually appeared in, so a missed game is not a
zero row, it is *no row at all*. This module fills that disjoint gap: how
often does a fantasy-relevant player not play in the first place.

## Step 1: the denominator (this is the part that is easy to get wrong)

The per-week question is "did the player play in a game their team
actually had." Two things must be true for a week to count in the
denominator at all:

1. The player's team played a REG-season game that week (excludes bye
   weeks -- nflreadpy's `load_schedules` REG rows are exactly the weeks a
   team has a game, so byes are excluded by construction, not by a special
   case).
2. The player was on that team's 53-man roster that week, in nflreadpy's
   `load_rosters_weekly` `status` column -- `ACT` (active), `INA`
   (game-day inactive, e.g. a healthy scratch or in-week injury
   designation), or `RES` (injured reserve). `RES` weeks are *included*
   deliberately: a player who tears an ACL and goes on IR is exactly the
   unavailability this module exists to measure, not a reason to shrink
   the denominator and hide it. Excluded statuses are practice squad
   (`DEV`), off the roster (`CUT`), retired (`RET`), and the handful of
   transitional codes -- none of those weeks belong to this team's
   availability question.

A week that clears both bars and has no matching `weekly_stats` row is a
miss; a matching row is an appearance.

## Step 2: which player-seasons to measure (the actual bug this task warned
about)

The obvious next step -- restrict to "fantasy-relevant" player-seasons by
filtering `weekly_stats` to players who finished as a top-N season-rank
performer -- is a survivorship-bias trap. Season rank is computed from
games the player actually played (`season_avg` over their appeared games),
so it implicitly *requires* a player to have appeared enough to accumulate
a stable average. Checked directly: Saquon Barkley's 2020 (torn ACL in
Week 2, done for the year) has only 2 games that season, below any
reasonable minimum-games threshold, so a season-rank-based cohort silently
drops the exact kind of season this module is supposed to capture. Doing
that at position level erases most of the real signal -- an early version
of this fit, built on a >=4-games season-rank cohort, produced RB
availability statistically indistinguishable from QB and WR, which is
precisely the "your denominator is wrong" symptom the plan doc warned
about.

The fix: select the cohort *ex ante*, from preseason ADP (`adp_history`,
2018-2024, matched to a `gsis_id` via `ids.load_crosswalk`/`match_by_name`),
not from anything that happened during the season being measured. A
player's preseason draft position does not depend on whether they get
hurt in Week 2, so this cohort has no survivorship bias. Depth per
position matches `distribution.py`'s `TIER_BREAKS` deepest tier (QB 24,
RB/WR 36, TE 24) -- "plausibly startable or rosterable," not just the
weekly starters.

Kickers are the one exception: ADP data covers K, but (per this project's
established finding, see `docs/superpowers/plans/README.md`) kickers match
the ID crosswalk at ~0%, so an ADP-based cohort for K is empty. Kicker
season-ending injuries are rare enough that the survivorship-bias concern
is much smaller for this position, so K falls back to a `weekly_stats`
season-rank cohort (top 12, >=1 game) instead.

## Step 3: what the fit found

Measured on 2018-2024 (the ADP-covered years), position-level per-week
availability for the ADP-cohort positions, plus K on its own cohort/years:

| position | p_available | miss rate |
|----------|-------------|-----------|
| K        | 0.936       | 6.4%      |
| QB       | 0.842       | 15.8%     |
| WR       | 0.842       | 15.8%     |
| RB       | 0.804       | 19.6%     |
| TE       | 0.797       | 20.3%     |

RB is visibly worse than QB and WR (about 4 points, ~24% more missed
games in relative terms), matching the real-world pattern this module was
asked to encode. TE runs slightly lower than RB in this data -- the plan
doc only asserted RB should beat QB/WR, not that RB must be the single
worst position, so this is reported as-is rather than forced.

**Tier**: split each skill position into ADP-rank tiers 1-12/13-24/25-36
and refit. Tier-1 (clear starters) availability is 3-5 points *higher*
than tier-2/3 within every position (RB: 83.1% / 79.2% / 78.9%; similarly
for QB/WR/TE) -- entrenched starters apparently see fewer healthy-scratch
and roster-churn weeks than deeper players, not more, which if anything
runs opposite the "workhorses get hurt more" intuition the plan doc
floated. The effect is real but modest (a few points) and roughly
consistent in direction across positions rather than a large,
position-specific interaction -- fitting separate tier curves would
fragment an already-modest per-tier sample (~1,300-1,400 player-weeks
each) for a second-order effect. Decision: model at position level only;
the tier gradient is recorded here rather than encoded.

**Age**: nflreadpy's weekly roster snapshot carries `birth_date`, so an
age join is cheap. Checked for RB (the position most plausibly affected):
p(available) by age is 0.76 (21) / 0.79 (22) / 0.85 (23) / 0.82 (24) /
0.81 (25) ... noisy and non-monotonic through the core of the career, and
the tails (age 30+) have 16-66 player-weeks each, small enough that a
single fluky season swings the estimate by double-digit points (age-33
shows 100% on n=16). There is no clean age effect to fit here, and the
tails are exactly where a spurious fit would do the most damage (aging
veterans are a real fantasy question). Decision: leave age out, per the
plan doc's explicit allowance for a weak/noisy effect not being worth the
risk.

## Step 4: independent weeks vs. persistence -- and why persistence wins

Measured directly (position-pooled, consecutive *eligible* weeks only, so
byes never masquerade as a return from injury): P(unavailable next week |
unavailable this week) = 75.5%, vs. P(unavailable next week | available
this week) = 7.1%. An out week is roughly 10x more likely to be followed
by another out week than an available week is. Modeling each week as an
independent Bernoulli draw at the fitted marginal rate would treat a torn
ACL exactly like 17 independent coin flips -- it would produce the right
season-long average appearance count but would essentially never produce
the actual shape that dominates real injury seasons: one player out for a
long, contiguous stretch. For a championship-probability objective that
is exactly the wrong error to make, since a long-enough absence is
effectively "lose this roster spot for the stretch that decides the
playoffs," which independent draws systematically under-price.

`PlayerAvailability` therefore models each player as a two-state Markov
chain (available / out), not independent per-week draws. Two things are
fit directly from data per position:

- `p_available`: the marginal (stationary) probability, the number in the
  table above -- this is the trusted, largest-sample statistic and the
  one both tests and downstream consumers check against.
- `persistence`: P(out next week | out this week), fit directly from the
  (smaller, noisier, but real) out-state transition counts.

The complementary "become unavailable" transition, P(out next week |
available this week), is *not* independently fit -- it is solved for so
the chain's stationary distribution exactly reproduces `p_available` (see
`PlayerAvailability.p_become_unavailable`). This keeps the one
larger-sample, more-trusted number (the marginal rate) authoritative, and
avoids a small inconsistency between two independently-noisy fits
producing a simulated long-run average that quietly drifts from the
historical one. A season is seeded in the chain's stationary distribution
(the marginal rate again), so the simulated per-week marginal matches
history at every week of a simulated season, not just on average across
many seasons.

## Interface / performance

Task 7's budget is ~200 players x 14 weeks x 5,000 sims = 14,000,000 draws
per candidate pick, in ~0.2s through `WeeklyDistribution.sample`. A Markov
chain cannot be flattened into one `rng` call the way independent draws
can -- each week depends on the last -- but it also does not need a
per-(player, sim) Python loop: `sample_availability_batch` loops over
weeks only (14 iterations, fixed, independent of player/sim count) and
vectorizes every other axis (players x sims) inside each iteration with
plain numpy array ops. 14 iterations of cheap numpy work is negligible
next to 14,000,000 total draws either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..ids import load_crosswalk, match_by_name
from ..league import REGULAR_SEASON_WEEKS
from ..sources._nflverse_compat import load_from_nflverse
from ..store import check_cache_fresh, exists, fingerprint, read, write, write_fingerprint

# ---------------------------------------------------------------------------
# Raw data loaders

ADP_SEASONS: tuple[int, ...] = tuple(range(2018, 2025))  # adp_history coverage

# nflreadpy `load_rosters_weekly` status codes that count as "on this team's
# roster this week" for availability purposes. `RES` (injured reserve) is
# included on purpose -- see module docstring, Step 1.
ROSTERED_STATUSES: frozenset[str] = frozenset({"ACT", "INA", "RES"})

# "Draftable" depth per position for the ADP-based cohort, matching
# `distribution.py`'s `TIER_BREAKS` deepest tier.
ADP_COHORT_DEPTH: dict[str, int] = {"QB": 24, "RB": 36, "WR": 36, "TE": 24}

# K has ~0% ADP-to-crosswalk match (see module docstring), so it gets its
# own weekly_stats-season-rank cohort instead.
K_SEASON_RANK_DEPTH = 12
K_MIN_GAMES = 1

FIT_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K")

MIN_PLAYER_WEEKS_FOR_FIT = 200  # else the position-level rate is too thin to trust


def _load_schedules(seasons: tuple[int, ...]) -> pl.DataFrame:
    return load_from_nflverse(
        lambda nfl: nfl.load_schedules(seasons=list(seasons)),
        lambda nfl_data_py: nfl_data_py.import_schedules(list(seasons)),
    )


def _load_rosters_weekly(seasons: tuple[int, ...]) -> pl.DataFrame:
    return load_from_nflverse(
        lambda nfl: nfl.load_rosters_weekly(seasons=list(seasons)),
        lambda nfl_data_py: nfl_data_py.import_weekly_rosters(list(seasons)),
    )


# ---------------------------------------------------------------------------
# Step 1: the denominator


def team_reg_game_weeks(schedules: pl.DataFrame) -> pl.DataFrame:
    """(season, week, team) for every team-week with a REG-season game.

    Bye weeks are excluded by construction: a team on a bye has no row in
    `schedules` for that (season, week), home or away, so it never appears
    here.
    """
    reg = schedules.filter(pl.col("game_type") == "REG")
    home = reg.select(pl.col("season"), pl.col("week"), pl.col("home_team").alias("team"))
    away = reg.select(pl.col("season"), pl.col("week"), pl.col("away_team").alias("team"))
    return pl.concat([home, away]).unique()


def rostered_weeks(rosters_weekly: pl.DataFrame) -> pl.DataFrame:
    """(season, week, team, gsis_id, position) for weeks a player was on a
    team's roster in a status that counts (see `ROSTERED_STATUSES`)."""
    return (
        rosters_weekly.filter(pl.col("status").is_in(ROSTERED_STATUSES))
        .filter(pl.col("gsis_id").is_not_null())
        .select(["season", "week", "team", "gsis_id", "position"])
    )


def eligible_weeks(rosters_weekly: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Every (season, week, gsis_id, position) week that counts in the
    availability denominator: on the roster (`rostered_weeks`) AND the
    player's team had a REG game that week (`team_reg_game_weeks`)."""
    return rostered_weeks(rosters_weekly).join(
        team_reg_game_weeks(schedules), on=["season", "week", "team"], how="inner"
    )


# ---------------------------------------------------------------------------
# Step 2: cohort selection (ex-ante, to avoid survivorship bias)


def _adp_cohort(adp_history: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """(season, position, gsis_id) for players drafted within
    `ADP_COHORT_DEPTH` of their position, matched to a real player ID via
    the crosswalk. Selection is on preseason ADP alone, so it cannot be
    biased by what happens to the player during the season being measured
    -- see module docstring, Step 2."""
    ranked = adp_history.with_columns(
        pl.col("adp").rank(method="ordinal").over(["season", "position"]).alias("pos_rank")
    )
    matched = match_by_name(ranked, crosswalk)
    cohorts = [
        matched.filter(
            (pl.col("position") == position)
            & (pl.col("pos_rank") <= depth)
            & pl.col("gsis_id").is_not_null()
        )
        for position, depth in ADP_COHORT_DEPTH.items()
    ]
    return (
        pl.concat(cohorts)
        .select(["season", "position", "gsis_id"])
        .unique(subset=["season", "position", "gsis_id"])
    )


def _kicker_cohort(weekly: pl.DataFrame) -> pl.DataFrame:
    """(season, position, gsis_id) for the top `K_SEASON_RANK_DEPTH` kickers
    by season average, season-rank based (not ADP, per module docstring)."""
    k = weekly.filter(pl.col("position") == "K")
    season_avg = (
        k.group_by(["season", "player_id"])
        .agg(pl.col("fantasy_points").mean().alias("season_avg"), pl.len().alias("games"))
        .filter(pl.col("games") >= K_MIN_GAMES)
    )
    season_avg = season_avg.with_columns(
        pl.col("season_avg").rank(method="ordinal", descending=True).over("season").alias("season_rank")
    )
    top = season_avg.filter(pl.col("season_rank") <= K_SEASON_RANK_DEPTH)
    return top.select(
        pl.col("season"),
        pl.lit("K").alias("position"),
        pl.col("player_id").alias("gsis_id"),
    )


def availability_cohort(
    adp_history: pl.DataFrame, weekly: pl.DataFrame, crosswalk: pl.DataFrame
) -> pl.DataFrame:
    """(season, position, gsis_id) for every fantasy-relevant player-season
    used to fit availability. ADP-based for QB/RB/WR/TE, season-rank-based
    for K -- see module docstring, Step 2."""
    return pl.concat([_adp_cohort(adp_history, crosswalk), _kicker_cohort(weekly)])


# ---------------------------------------------------------------------------
# Step 3/4: fit p_available and persistence


def fit_availability_rates(
    schedules: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    weekly: pl.DataFrame,
    adp_history: pl.DataFrame,
    crosswalk: pl.DataFrame,
) -> pl.DataFrame:
    """Fit per-position `p_available` (marginal) and `persistence`
    (P(out next week | out this week)) from history.

    Returns a flat table (not `PlayerAvailability` objects) so it can be
    persisted directly, matching `distribution.py`'s fit-and-cache
    convention.
    """
    cohort = availability_cohort(adp_history, weekly, crosswalk)
    elig = eligible_weeks(rosters_weekly, schedules).join(
        cohort, on=["season", "position", "gsis_id"], how="inner"
    )

    appeared = (
        weekly.select(["season", "week", "player_id"])
        .unique()
        .with_columns(pl.lit(True).alias("appeared"))
    )
    elig = elig.join(
        appeared,
        left_on=["season", "week", "gsis_id"],
        right_on=["season", "week", "player_id"],
        how="left",
    ).with_columns(pl.col("appeared").fill_null(False))

    # Consecutive *eligible* weeks per player-season, ordered by week --
    # eligible weeks already exclude byes, so an adjacent pair here is
    # always two real, back-to-back team games.
    seq = elig.sort(["season", "position", "gsis_id", "week"]).with_columns(
        pl.col("appeared").shift(-1).over(["season", "gsis_id"]).alias("next_appeared")
    )
    seq = seq.filter(pl.col("next_appeared").is_not_null())

    rows = []
    for position in FIT_POSITIONS:
        pos_elig = elig.filter(pl.col("position") == position)
        n_weeks = pos_elig.height
        if n_weeks < MIN_PLAYER_WEEKS_FOR_FIT:
            raise ValueError(
                f"Only {n_weeks} eligible player-weeks for {position} -- too few to "
                f"fit a stable availability rate. Check cohort depth or season coverage."
            )
        p_available = float(pos_elig["appeared"].mean())

        pos_seq = seq.filter((pl.col("position") == position) & (~pl.col("appeared")))
        n_out_weeks = pos_seq.height
        if n_out_weeks < 30:
            raise ValueError(
                f"Only {n_out_weeks} out-state transitions for {position} -- too few "
                f"to fit persistence reliably."
            )
        persistence = float((~pos_seq["next_appeared"]).mean())

        rows.append(
            {
                "position": position,
                "p_available": p_available,
                "persistence": persistence,
                "n_player_weeks": n_weeks,
                "n_out_weeks": n_out_weeks,
            }
        )
    return pl.DataFrame(rows)


def load_or_fit_availability_rates(force_refit: bool = False) -> pl.DataFrame:
    """Cached wrapper: fetches nflreadpy schedules/rosters plus this
    project's already-ingested `weekly_stats`/`adp_history` datasets and the
    ID crosswalk, fits once, and persists the result -- so normal use (and
    tests that don't pass `force_refit`) never hits the network.

    Cache freshness is checked against a fingerprint of `weekly_stats` and
    `adp_history` only -- not the crosswalk or the nflreadpy
    schedules/rosters. Those two are cheap, already-persisted local parquet
    files (this function has to read them either way to have any input to
    fit from), so checking them costs a few tens of milliseconds even on a
    cache hit. `load_crosswalk()`, by contrast, is a live network fetch
    measured at ~1.4s; including it in the fingerprint would mean paying a
    network round-trip on *every* call just to confirm a cache hit,
    defeating the purpose of caching at all -- exactly the invisible
    performance cliff this cache exists to avoid. Schedules/rosters are
    similarly only fetched on an actual refit today and are left out for
    the same reason. This means a crosswalk-only change (rare -- it's an ID
    mapping, not seasonal stats) won't trigger a refit; re-ingesting
    `weekly_stats` or `adp_history` will."""
    weekly = read("weekly_stats")
    adp_history = read("adp_history")
    fp = fingerprint(weekly, adp_history)

    if not force_refit and exists("availability_rates"):
        check_cache_fresh("availability_rates", fp)
        return read("availability_rates")

    # Rosters (and hence eligibility) are only fetched for `ADP_SEASONS` --
    # schedules for the same window keeps every position's fit, including
    # K's, on identical season coverage instead of K silently getting a
    # wider effective window than QB/RB/WR/TE.
    schedules = _load_schedules(ADP_SEASONS)
    rosters_weekly = _load_rosters_weekly(ADP_SEASONS)
    crosswalk = load_crosswalk()

    table = fit_availability_rates(schedules, rosters_weekly, weekly, adp_history, crosswalk)
    write("availability_rates", table)
    write_fingerprint("availability_rates", fp)
    return table


# ---------------------------------------------------------------------------
# The model itself


@dataclass(frozen=True)
class PlayerAvailability:
    """A position's per-week availability, modeled as a two-state
    (available / out) Markov chain -- see module docstring, Step 4, for why
    persistence is modeled rather than independent per-week draws.

    `p_available` is the marginal (stationary) probability of being
    available in any given week -- the trusted, largest-sample fitted
    statistic. `persistence` is P(out next week | out this week), fit
    directly. `p_become_unavailable` (P(out next week | available this
    week)) is *derived*, not independently fit, so the chain's stationary
    distribution reproduces `p_available` exactly -- see
    `p_become_unavailable`.
    """

    position: str
    p_available: float
    persistence: float
    n_player_weeks: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_available <= 1.0):
            raise ValueError(f"p_available must be in [0, 1], got {self.p_available}")
        if not (0.0 <= self.persistence <= 1.0):
            raise ValueError(f"persistence must be in [0, 1], got {self.persistence}")

    @property
    def p_become_unavailable(self) -> float:
        """P(out next week | available this week), solved so the chain's
        stationary distribution matches `p_available` exactly:

        pi_available = (1 - persistence) / ((1 - persistence) + r)

        Solving for r = p_become_unavailable given pi_available =
        p_available:

        r = (1 - persistence) * (1 - p_available) / p_available
        """
        if self.p_available <= 0:
            raise ValueError(
                f"{self.position}: p_available={self.p_available} <= 0 -- cannot solve "
                "for a stationary-consistent transition probability."
            )
        r = (1.0 - self.persistence) * (1.0 - self.p_available) / self.p_available
        if not (0.0 <= r <= 1.0):
            raise ValueError(
                f"{self.position}: derived p_become_unavailable={r} is outside [0, 1] -- "
                "p_available and persistence are inconsistent with a valid 2-state chain."
            )
        return r

    def sample_season(
        self,
        rng: np.random.Generator,
        n_sims: int,
        n_weeks: int = REGULAR_SEASON_WEEKS,
    ) -> np.ndarray:
        """Sample `n_sims` independent 14-week (or `n_weeks`) availability
        patterns. Returns a `(n_sims, n_weeks)` bool array, True = available.

        Callers must pass an explicit `np.random.Generator` -- never global
        numpy state -- for the same reproducibility reason as
        `WeeklyDistribution.sample` (see `models/base.py`).
        """
        return sample_availability_batch(
            np.array([self.p_available]),
            np.array([self.persistence]),
            rng,
            n_sims,
            n_weeks,
        )[0]


def sample_availability_batch(
    p_available: np.ndarray,
    persistence: np.ndarray,
    rng: np.random.Generator,
    n_sims: int,
    n_weeks: int = REGULAR_SEASON_WEEKS,
) -> np.ndarray:
    """Vectorized batch sampler: many players x many sims x a season of
    weeks, in one call.

    `p_available` and `persistence` are 1-D arrays of length `n_players`
    (one value per player, e.g. every player's position-level rate looked
    up ahead of time). Returns a `(n_players, n_sims, n_weeks)` bool array,
    True = available that week.

    The Markov dependency between consecutive weeks means this cannot be a
    single flat `rng` call the way `WeeklyDistribution.sample` is -- but it
    also does not need a per-(player, sim) Python loop. This loops over
    `n_weeks` only (a fixed 14 iterations, independent of `n_players` or
    `n_sims`); every other axis is a vectorized numpy op inside each
    iteration. For Task 7's ~200 players x 5,000 sims budget that is 14
    iterations of ~1,000,000-element array ops, not 14,000,000 individual
    Python-level draws.
    """
    p_available = np.asarray(p_available, dtype=float)
    persistence = np.asarray(persistence, dtype=float)
    if p_available.shape != persistence.shape:
        raise ValueError(
            f"p_available and persistence must have the same shape, got "
            f"{p_available.shape} and {persistence.shape}"
        )
    if np.any((p_available < 0) | (p_available > 1)):
        raise ValueError("p_available must be in [0, 1]")
    if np.any((persistence < 0) | (persistence > 1)):
        raise ValueError("persistence must be in [0, 1]")
    if np.any(p_available <= 0):
        raise ValueError("p_available must be > 0 to solve for a stationary transition")

    p_become_unavailable = (1.0 - persistence) * (1.0 - p_available) / p_available
    if np.any((p_become_unavailable < 0) | (p_become_unavailable > 1)):
        raise ValueError(
            "derived p_become_unavailable is outside [0, 1] for at least one player -- "
            "p_available and persistence are inconsistent with a valid 2-state chain."
        )

    n_players = p_available.shape[0]
    out = np.empty((n_players, n_sims, n_weeks), dtype=bool)

    # Week 0: seed each player's chain in its own stationary distribution,
    # so the marginal probability of "available" is exactly p_available at
    # every week of the simulated season, not just in the long run.
    u0 = rng.random((n_players, n_sims))
    state = u0 < p_available[:, None]
    out[:, :, 0] = state

    for w in range(1, n_weeks):
        u = rng.random((n_players, n_sims))
        stays_available = u >= p_become_unavailable[:, None]
        becomes_available = u >= persistence[:, None]
        state = np.where(state, stays_available, becomes_available)
        out[:, :, w] = state

    return out


# ---------------------------------------------------------------------------
# Public lookup


def availability_by_position(force_refit: bool = False) -> dict[str, PlayerAvailability]:
    """Every position's fitted `PlayerAvailability`, keyed by position."""
    table = load_or_fit_availability_rates(force_refit=force_refit)
    return {
        row["position"]: PlayerAvailability(
            position=row["position"],
            p_available=row["p_available"],
            persistence=row["persistence"],
            n_player_weeks=row["n_player_weeks"],
        )
        for row in table.to_dicts()
    }
