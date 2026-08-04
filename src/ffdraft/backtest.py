"""Stage 3 Task 6: the 2024 holdout backtest -- the gate.

Fits every model component on data through 2023 only, drafts four
contenders from slot 8 against 2024's real ADP and real opponents (sampled
from the through-2023 opponent model), scores each resulting roster on
**real** 2024 weekly results, and reports paired championship-rate
differences. See `scripts/backtest.py` for the runnable entry point and
`docs/superpowers/plans/2026-08-03-opponent-model-and-optimizer.md` (Task 6)
for the spec this implements.

## Leakage -- what is fit on which seasons, and why each cut is safe

- **Tier shapes** (`models.tier_shape.fit_tier_shapes`): `weekly_stats`
  filtered to season <= `FIT_THROUGH_SEASON` (2023).
- **Rank curve ADP anchoring** (`models.rank_curve.fit_rank_curves`):
  `adp_history` and `weekly_stats` both filtered to <= 2023. The curve is
  then *anchored* (see `models.distribution.build_player_pool`) to 2024's
  real ADP order (`rankings_holdout`, built from `adp_history`'s 2024 rows)
  -- 2024 ADP is an input a real 2024 drafter would have had, not a
  leakage of 2024 results; only the *shape* of the curve (what a given rank
  is worth) comes from pre-2024 data.
- **Opponent model** (`models.opponent.fit_opponent_model`): `league_drafts`
  restricted to `TRAINING_SEASONS` seasons <= 2023 (i.e. 2018-2023 --
  `TRAINING_SEASONS` already excludes 2025 for lack of ADP; this excludes
  2024 too).
- **Replacement level** (`models.replacement.weekly_replacement_values`):
  `weekly_stats` and `league_drafts` both filtered to <= 2023.
- **Availability** (`models.availability.fit_availability_rates`): nflverse
  schedules/rosters fetched for seasons 2018-2023 only, joined against
  `weekly_stats`/`adp_history` filtered to <= 2023.
- **2024 ADP** (`adp_history` season == 2024): used as an *input* to the
  opponent model's sampling and to the ADP-baseline contender -- this is
  legitimate, pre-draft information, not leakage (see module docstring's
  "Leakage" note in the Task 6 spec).
- **2024 real weekly results** (`weekly_stats` season == 2024): used
  *only* for scoring the four contenders' finished rosters (`real_season_
  players_for_scoring`) -- never touches any fit above.

Several of this project's `load_or_fit_*` cache wrappers key their cache
artifact by a filename that is **not** namespaced by which seasons went
into the fit (`distribution_tier_shapes`, `replacement_weekly_values`,
`availability_rates` all use one fixed name each). Calling them with data
filtered to <= 2023 while the live, full-history cache already exists under
that same name is exactly the scenario `store.check_cache_fresh` exists to
catch -- and it does: `tests/test_backtest.py::
test_tier_shape_cache_raises_rather_than_silently_reusing_stale_fit`
demonstrates this raises `CacheStaleError` rather than silently serving (or
silently overwriting) the live cache. Rather than force a refit that would
overwrite the live cache (corrupting it for the recommender/rollout's
normal, non-backtest use), `fit_holdout_context` below calls the
**uncached** `fit_*` functions directly for these three components and
hydrates their results by hand, via the same hydration helpers
(`tier_shape_from_row`, `hydrate_availability_table`) the cached wrappers
themselves use -- so this is the *same* fitting code, just not written to
the shared, unnamespaced disk cache. The rank curve is the one exception:
`load_or_fit_rank_curves`'s cache *is* namespaced by `rankings` (`store.
cache_namespace`), specifically so a 2024-ADP-anchored context and the live
2026 context can coexist -- this was purpose-built for this backtest (see
that module's docstring), so it is used via its normal cached path.

## The engine, and why it is not `recommend_pick`

Task 6's spec calls the contender "the recommender / greedy rollout
policy." The full Monte-Carlo recommender (`draft.recommender.
recommend_pick`) costs ~5.6 minutes *per pick* at its production budget
(65 rollouts x 500 season sims -- see that module's docstring); calling it
at every one of slot 8's ~18 picks, for each of N draft realizations, is
computationally infeasible for this backtest (N x 18 x 5.6min). This
backtest instead uses `draft.rollout.run_rollout`'s own greedy `draft.value`
policy directly for every one of slot 8's picks -- this *is* the same
policy `recommend_pick`'s own rollouts use for all of *our* future picks
when evaluating any candidate; the Monte-Carlo layer on top only decides
which of ~15 candidates to force as the *immediate next* pick by comparing
simulated title equity, and does not change how any later pick is chosen.
Using the greedy policy throughout is therefore the literal "greedy rollout
policy" half of the spec's own phrasing, at a tractable cost.

## Availability leakage inside the engine's own picks

`draft.value.value_available`'s bench-reachability term defaults to a
process-memoized, live-cached fit of `models.availability.
availability_by_position()` (2018-2024-ish full history) when no explicit
`availability_by_position` is passed -- and neither `draft.rollout` nor
`draft.recommender` expose a parameter to override it. Since the "engine"
contender's own greedy picks depend on this, running it unmodified would
leak 2024 into a component of the draft *decision itself*, not just a
fitted artifact sitting unused. `_patched_engine_availability` (a scoped
monkeypatch of `draft.value`'s memoized default, restored immediately
after) forces the engine's own picks to use the through-2023 availability
fit instead, for exactly the duration of the engine contender's own draft.
The three baseline contenders never call `draft.value` at all, so they need
no such patch.

## Real-results scoring

`real_season_players_for_scoring` resolves each drafted pick to real 2024
`weekly_stats` rows (weeks 1-`N_SCORE_WEEKS`, i.e. 14 regular + 3 playoff
weeks, matching real NFL week numbers directly) and marks a week
unavailable exactly when there is no row -- never applying the fitted
Markov availability model on top (see the Task 6 spec: `weekly_stats` only
has rows for games actually played, so layering the model would
double-count). A pick with **zero** rows across all of 2024 (drafted but
never played, or unmatched) falls back to that position's through-2023
replacement-level distribution, sampled -- "consistent with `build_roster`'s
fallback" per the spec, using the *same* replacement fit already used
throughout the draft. D/ST is scored from the shared `dst_distribution()`
every week regardless of match status: this project never ingested
per-team defensive game stats (see `models/defense.py`'s docstring), so
there is no real weekly D/ST result to fall back to at all -- this mirrors
the project's own standing decision that every D/ST is interchangeable, not
a new exception invented for this backtest.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from . import store
from .draft import value as value_mod
from .draft.rollout import DraftState, PickPolicy, pool_adp_lookup, run_rollout
from .ids import load_crosswalk, normalize_name
from .league import DRAFT_SLOT, N_TEAMS, PLAYOFF_BYES, REGULAR_SEASON_WEEKS
from .models.availability import (
    PlayerAvailability,
    _load_rosters_weekly,
    _load_schedules,
    fit_availability_rates,
    hydrate_availability_table,
)
from .models.defense import dst_distribution
from .models.distribution import PlayerDistribution, build_player_pool
from .models.opponent import TRAINING_SEASONS, OpponentModel, _round_bucket, build_training_set, fit_opponent_model
from .models.rank_curve import fit_rank_curves, hydrate_rank_curves
from .models.replacement import ReplacementLevelDistribution, drafted_player_ids, weekly_replacement_values
from .models.tier_shape import fit_tier_shapes, tier_shape_from_row
from .sim.lineup import RosterPlayer, build_replacement_means, solve_lineup
from .sim.season import _run_playoffs, _seed_teams, build_regular_season_schedule

# ---------------------------------------------------------------------------
# Constants

FIT_THROUGH_SEASON: int = 2023
HOLDOUT_SEASON: int = 2024
N_SCORE_WEEKS: int = REGULAR_SEASON_WEEKS + 3  # 14 regular + 3 playoff weeks
CONTENDERS: tuple[str, ...] = ("engine", "adp", "consensus", "random")


# ---------------------------------------------------------------------------
# Holdout fitting


@dataclass(frozen=True)
class FitReport:
    """Which seasons went into each fitted component -- see module
    docstring, "Leakage", for the full justification of each cut."""

    tier_shape_seasons: tuple[int, int]
    rank_curve_seasons: tuple[int, int]
    opponent_model_seasons: tuple[int, ...]
    replacement_seasons: tuple[int, int]
    availability_seasons: tuple[int, ...]
    holdout_season: int

    def describe(self) -> str:
        lines = [
            f"tier shapes (weekly_stats):          seasons {self.tier_shape_seasons[0]}-{self.tier_shape_seasons[1]}",
            f"rank curve ADP anchoring:             seasons {self.rank_curve_seasons[0]}-{self.rank_curve_seasons[1]}",
            f"opponent model (league_drafts):       seasons {self.opponent_model_seasons}",
            f"replacement level (weekly_stats):     seasons {self.replacement_seasons[0]}-{self.replacement_seasons[1]}",
            f"availability (schedules/rosters):     seasons {self.availability_seasons[0]}-{self.availability_seasons[-1]}",
            f"2024 ADP (input, not leakage):        season {self.holdout_season}",
            f"2024 real weekly results (scoring only): season {self.holdout_season}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class HoldoutContext:
    pool: dict[str, PlayerDistribution]
    replacement_by_position: dict[str, object]  # includes "DST" -> dst_distribution()
    replacement_means: dict[str, float]
    availability_by_position: dict[str, PlayerAvailability]
    opponent_model: OpponentModel
    adp_holdout: pl.DataFrame
    rankings_holdout: pl.DataFrame
    adp_by_player_id: dict[str, float]
    weekly_holdout: pl.DataFrame
    rounds: int
    report: FitReport


def _adp_ranked_frame(adp_season: pl.DataFrame) -> pl.DataFrame:
    """`(rank, name, position, team)` ordered by that season's real ADP --
    the `rankings`-shaped frame `build_player_pool`/`pool_adp_lookup` need,
    built the same way `scripts/season_report.py`'s 2024 calibration does.
    `team` is carried through (unlike `season_report.py`'s version, which
    doesn't need it) because `pool_adp_lookup`'s D/ST matching needs a team
    abbreviation to look up a defense's ADP row."""
    return (
        adp_season.sort("adp")
        .with_row_index("rank")
        .with_columns((pl.col("rank") + 1).alias("rank"), pl.col("team").cast(pl.String))
        .select(["rank", "name", "position", "team"])
    )


_DEEP_UNIVERSE_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K")


def _deep_rankings_holdout(
    adp_holdout: pl.DataFrame,
    adp_history_full: pl.DataFrame,
    weekly_holdout: pl.DataFrame,
    fit_through_season: int,
) -> pl.DataFrame:
    """Extend `_adp_ranked_frame(adp_holdout)` (~200 players -- real ADP's
    actual depth, see `docs/HANDOFF.md`'s ADP-coverage note) with every
    other skill/K player who actually appeared in the holdout season's
    `weekly_stats`, so the simulated draft has a deep enough candidate
    universe for a full 18-round, 10-team mock draft (180 picks) without
    scraping the bottom of a too-shallow pool.

    Real ADP alone is not deep enough: `adp_holdout` only covers ~190 skill/
    K players (see `fit_holdout_context`'s pool-size report) against 180
    needed picks -- a live rollout never hits this because `rankings_2026`
    (this project's normal, much deeper ~510-player draftable universe) has
    no equivalent for a historical season; `adp_history` is the only
    ranking source this project has for 2018-2024. Checked directly: without
    this extension, the simulated draft nearly exhausts the pool by the
    final rounds, and the greedy engine's bench-floor guard (see `draft/
    rollout.py`'s module docstring) starts being forced into whatever
    scraps remain at a scarce position -- a scarcity artifact of the pool
    being too shallow, not a real finding about the engine.

    The extras are ordered by their **most recent ADP from a season <=
    `fit_through_season`** where one exists (still leakage-safe -- a
    player's own draft stock in a prior season is known before the 2024
    season plays out, unlike anything from 2024 itself), and by name
    (stable, arbitrary) for anyone with no ADP history at all -- these are,
    by construction, deep enough that their exact order barely matters
    either way. This is intentionally *not* ordered by 2024 performance,
    which would leak exactly the outcome this backtest exists to avoid.
    """
    base = _adp_ranked_frame(adp_holdout)
    base_keyed = base.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("_key")
    )
    covered = set(zip(base_keyed["_key"].to_list(), base_keyed["position"].to_list()))

    seen = (
        weekly_holdout.filter(pl.col("position").is_in(_DEEP_UNIVERSE_POSITIONS))
        .select(["player_name", "position"])
        .unique()
        .with_columns(pl.col("player_name").map_elements(normalize_name, return_dtype=pl.String).alias("_key"))
    )
    extra = seen.filter(
        pl.struct(["_key", "position"]).map_elements(
            lambda s, covered=covered: (s["_key"], s["position"]) not in covered, return_dtype=pl.Boolean
        )
    )
    if extra.height == 0:
        return base

    hist = adp_history_full.filter(pl.col("season") <= fit_through_season).with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("_key")
    )
    hist_best = (
        hist.sort("season", descending=True)
        .group_by(["_key", "position"])
        .agg(pl.col("adp").first().alias("hist_adp"))
    )
    extra = extra.join(hist_best, on=["_key", "position"], how="left").with_columns(
        pl.col("hist_adp").fill_null(1.0e6)
    )
    extra = extra.sort(["hist_adp", "player_name"])
    n_base = base.height
    extra = extra.with_row_index("extra_rank").with_columns(
        (pl.col("extra_rank") + n_base + 1).alias("rank"),
        pl.lit(None, dtype=pl.String).alias("team"),
    )
    extra = extra.rename({"player_name": "name"}).select(["rank", "name", "position", "team"])
    return pl.concat([base.with_columns(pl.col("team").cast(pl.String)), extra])


def fit_holdout_context(
    fit_through_season: int = FIT_THROUGH_SEASON,
    holdout_season: int = HOLDOUT_SEASON,
) -> HoldoutContext:
    """Fit every model component on data through `fit_through_season` only,
    anchored to `holdout_season`'s real ADP. See module docstring for which
    seasons feed which component and why each cut is leakage-safe."""
    weekly_full = store.read("weekly_stats")
    adp_history_full = store.read("adp_history")
    league_drafts = store.read("league_drafts")
    league_managers = store.read("league_managers")
    crosswalk = load_crosswalk()

    weekly_fit = weekly_full.filter(pl.col("season") <= fit_through_season)
    adp_fit = adp_history_full.filter(pl.col("season") <= fit_through_season)

    adp_holdout = adp_history_full.filter(pl.col("season") == holdout_season)
    if adp_holdout.height == 0:
        raise ValueError(f"no adp_history rows for holdout season {holdout_season}")

    weekly_holdout = weekly_full.filter(pl.col("season") == holdout_season)

    rankings_holdout = _deep_rankings_holdout(adp_holdout, adp_history_full, weekly_holdout, fit_through_season)

    league_drafts_holdout = league_drafts.filter(pl.col("season") == holdout_season)
    rounds = int(league_drafts_holdout["round"].max())

    # --- tier shapes: weekly_stats through fit_through_season, uncached ---
    # (see module docstring, "Leakage", for why this bypasses the disk
    # cache rather than risk a CacheStaleError against the live cache or,
    # worse, force_refit=True silently overwriting it).
    tier_table = fit_tier_shapes(weekly_fit)
    tier_shape_lookup = {(r["position"], r["tier"]): tier_shape_from_row(r) for r in tier_table.to_dicts()}

    # --- rank curve: adp_history + weekly_stats through fit_through_season,
    # anchored to holdout-season ADP order. Uses the namespaced cache (keyed
    # by `rankings_holdout`) via build_player_pool's injected lookup below,
    # but fit directly here first so the pool assembly can also use it.
    adp_table_rc, deep_table_rc = fit_rank_curves(adp_fit, weekly_fit, rankings=rankings_holdout)
    rank_curve_lookup = hydrate_rank_curves(adp_table_rc, deep_table_rc)

    pool = build_player_pool(
        rankings=rankings_holdout,
        weekly=weekly_fit,
        adp_history=adp_fit,
        crosswalk=crosswalk,
        tier_shape_lookup=tier_shape_lookup,
        rank_curve_lookup=rank_curve_lookup,
    )

    # --- replacement level: weekly_stats + league_drafts through
    # fit_through_season, uncached (same reasoning as tier shapes above).
    league_drafts_fit = league_drafts.filter(pl.col("season") <= fit_through_season)
    drafted = drafted_player_ids(league_drafts_fit, crosswalk, weekly_fit)
    replacement_table = weekly_replacement_values(weekly_fit, drafted)
    replacement_by_position: dict[str, object] = {}
    for position in ("QB", "RB", "WR", "TE", "K"):
        values = tuple(replacement_table.filter(pl.col("position") == position)["value"].to_list())
        replacement_by_position[position] = ReplacementLevelDistribution(position=position, values=values)
    replacement_by_position["DST"] = dst_distribution()
    replacement_means = build_replacement_means(
        {k: v for k, v in replacement_by_position.items() if k != "DST"},
        replacement_by_position["DST"].mean,
    )

    # --- availability: nflverse schedules/rosters + weekly_stats/adp_history
    # through fit_through_season, uncached (same reasoning as above).
    availability_seasons = tuple(range(2018, fit_through_season + 1))
    schedules = _load_schedules(availability_seasons)
    rosters_weekly = _load_rosters_weekly(availability_seasons)
    availability_table = fit_availability_rates(schedules, rosters_weekly, weekly_fit, adp_fit, crosswalk)
    availability_by_position = hydrate_availability_table(availability_table)

    # --- opponent model: league_drafts through fit_through_season (ADP-
    # relative reach), via TRAINING_SEASONS restricted to <= fit_through_season.
    opponent_fit_seasons = tuple(s for s in TRAINING_SEASONS if s <= fit_through_season)
    training, _join_report = build_training_set(
        league_drafts, league_managers, crosswalk, adp_history_full, seasons=opponent_fit_seasons
    )
    opponent_model = fit_opponent_model(training)

    adp_by_id, _adp_match_report = pool_adp_lookup(pool, adp_holdout, rankings_holdout)

    report = FitReport(
        tier_shape_seasons=(int(weekly_fit["season"].min()), int(weekly_fit["season"].max())),
        rank_curve_seasons=(int(adp_fit["season"].min()), int(adp_fit["season"].max())),
        opponent_model_seasons=opponent_fit_seasons,
        replacement_seasons=(int(weekly_fit["season"].min()), int(weekly_fit["season"].max())),
        availability_seasons=availability_seasons,
        holdout_season=holdout_season,
    )

    return HoldoutContext(
        pool=pool,
        replacement_by_position=replacement_by_position,
        replacement_means=replacement_means,
        availability_by_position=availability_by_position,
        opponent_model=opponent_model,
        adp_holdout=adp_holdout,
        rankings_holdout=rankings_holdout,
        adp_by_player_id=adp_by_id,
        weekly_holdout=weekly_holdout,
        rounds=rounds,
        report=report,
    )


# ---------------------------------------------------------------------------
# Availability leakage guard for the engine's own picks -- see module
# docstring, "Availability leakage inside the engine's own picks".


@contextlib.contextmanager
def _patched_engine_availability(holdout_availability: Mapping[str, PlayerAvailability]):
    original = value_mod._fit_availability_by_position
    value_mod._fit_availability_by_position = lambda: dict(holdout_availability)
    value_mod._default_availability_by_position.cache_clear()
    try:
        yield
    finally:
        value_mod._fit_availability_by_position = original
        value_mod._default_availability_by_position.cache_clear()


# ---------------------------------------------------------------------------
# Pick policies for the three baseline contenders. `PickPolicy`'s signature
# (see draft/rollout.py): (available_ids, pool, roster_pairs,
# replacement_means, replacement_by_position) -> (player_id, position).
# "Positionally legal" for the random contender needs no extra filtering:
# this league has no per-position draft cap beyond total roster size (see
# league.py), so every available player is always a legal pick.


def adp_pick_policy(adp_by_player_id: Mapping[str, float]) -> PickPolicy:
    def _policy(available_ids, pool, roster_pairs, replacement_means, replacement_by_position):
        pid = min(available_ids, key=lambda p: adp_by_player_id.get(p, float("inf")))
        return pid, pool[pid].position

    return _policy


def consensus_rank_pick_policy() -> PickPolicy:
    """Best available by `PlayerDistribution.rank` -- this project's only
    2024-era consensus-rank signal, since FantasyPros ECR is ingested for
    the live 2026 draft only (no historical ECR exists in this project's
    data). `rankings_holdout` is itself built by sorting 2024 ADP (see
    `_adp_ranked_frame`), so `pool[pid].rank` and real ADP order coincide
    almost exactly for this backtest -- reported honestly in the writeup
    rather than presented as an independently-sourced comparison."""

    def _policy(available_ids, pool, roster_pairs, replacement_means, replacement_by_position):
        pid = min(available_ids, key=lambda p: pool[p].rank)
        return pid, pool[pid].position

    return _policy


def random_legal_pick_policy(rng: np.random.Generator) -> PickPolicy:
    def _policy(available_ids, pool, roster_pairs, replacement_means, replacement_by_position):
        idx = int(rng.integers(0, len(available_ids)))
        pid = available_ids[idx]
        return pid, pool[pid].position

    return _policy


POLICY_FACTORIES: dict[str, object] = {
    "engine": lambda rng, ctx: None,
    "adp": lambda rng, ctx: adp_pick_policy(ctx.adp_by_player_id),
    "consensus": lambda rng, ctx: consensus_rank_pick_policy(),
    "random": lambda rng, ctx: random_legal_pick_policy(rng),
}


# ---------------------------------------------------------------------------
# Running one contender's draft


def run_one_draft(
    seed: int,
    our_team: int,
    contender: str,
    ctx: HoldoutContext,
    n_teams: int = N_TEAMS,
) -> DraftState:
    """Run one contender's full draft against the same through-2023
    opponent model, using 2024's real ADP as the sampling signal -- see
    module docstring for why `contender="engine"` uses `run_rollout`'s
    plain greedy policy (not the Monte-Carlo `recommend_pick`) and why only
    that contender needs the availability leakage patch."""
    rng = np.random.default_rng(seed)
    policy = POLICY_FACTORIES[contender](rng, ctx)
    state = DraftState.from_picks([], n_teams=n_teams, rounds=ctx.rounds)

    def _run() -> DraftState:
        return run_rollout(
            state,
            ctx.pool,
            ctx.opponent_model,
            ctx.replacement_by_position,
            rng,
            our_team=our_team,
            adp_table=ctx.adp_holdout,
            rankings=ctx.rankings_holdout,
            pick_policy=policy,
        )

    if contender == "engine":
        with _patched_engine_availability(ctx.availability_by_position):
            return _run()
    return _run()


# ---------------------------------------------------------------------------
# Scoring on real 2024 weekly results -- see module docstring,
# "Real-results scoring".


def _weekly_lookup(weekly_holdout: pl.DataFrame, max_week: int) -> dict[str, dict[int, float]]:
    filtered = weekly_holdout.filter(pl.col("week") <= max_week)
    out: dict[str, dict[int, float]] = {}
    for pid, wk, pts in filtered.select(["player_id", "week", "fantasy_points"]).iter_rows():
        out.setdefault(pid, {})[wk] = float(pts)
    return out


def _kicker_gsis_by_name(weekly_holdout: pl.DataFrame) -> dict[str, str]:
    k = (
        weekly_holdout.filter(pl.col("position") == "K")
        .select(["player_id", "player_name"])
        .unique(subset=["player_name"])
    )
    return {normalize_name(name): pid for pid, name in k.iter_rows()}


@dataclass(frozen=True)
class RealPlayerWeeks:
    player_id: str
    position: str
    scores: tuple[float, ...]
    available: tuple[bool, ...]
    used_replacement_fallback: bool


def resolve_real_player_weeks(
    pick_player_id: str,
    position: str,
    weekly_lookup: Mapping[str, Mapping[int, float]],
    kicker_gsis_by_name: Mapping[str, str],
    replacement_by_position: Mapping[str, object],
    rng: np.random.Generator,
    n_weeks: int = N_SCORE_WEEKS,
) -> RealPlayerWeeks:
    """One drafted pick's real 2024 per-week score/availability. See module
    docstring, "Real-results scoring", for D/ST's special-case and the
    replacement-level fallback for a pick with zero real 2024 rows."""
    if position == "DST":
        dist = replacement_by_position["DST"]
        scores = tuple(float(x) for x in dist.sample(rng, n_weeks))
        return RealPlayerWeeks(pick_player_id, position, scores, tuple([True] * n_weeks), False)

    if position == "K":
        name = pick_player_id.split("::", 1)[1] if "::" in pick_player_id else pick_player_id
        gsis_id = kicker_gsis_by_name.get(normalize_name(name))
    else:
        gsis_id = pick_player_id if pick_player_id in weekly_lookup else None

    weeks_for_player = weekly_lookup.get(gsis_id, {}) if gsis_id else {}
    if not weeks_for_player:
        dist = replacement_by_position[position]
        scores = tuple(float(x) for x in dist.sample(rng, n_weeks))
        return RealPlayerWeeks(pick_player_id, position, scores, tuple([True] * n_weeks), True)

    scores: list[float] = []
    available: list[bool] = []
    for wk in range(1, n_weeks + 1):
        if wk in weeks_for_player:
            scores.append(weeks_for_player[wk])
            available.append(True)
        else:
            scores.append(0.0)
            available.append(False)
    return RealPlayerWeeks(pick_player_id, position, tuple(scores), tuple(available), False)


def real_team_weekly_totals(
    picks: Sequence[tuple[str, str]],
    weekly_lookup: Mapping[str, Mapping[int, float]],
    kicker_gsis_by_name: Mapping[str, str],
    replacement_by_position: Mapping[str, object],
    replacement_means: Mapping[str, float],
    rng: np.random.Generator,
    n_weeks: int = N_SCORE_WEEKS,
) -> tuple[np.ndarray, int]:
    """One team's optimal-lineup total for each of `n_weeks` real weeks,
    via `sim.lineup.solve_lineup` on that week's real (or replacement-
    fallback) scores. Returns `(totals, n_fallback)`."""
    players = [
        resolve_real_player_weeks(
            pid, pos, weekly_lookup, kicker_gsis_by_name, replacement_by_position, rng, n_weeks
        )
        for pid, pos in picks
    ]
    n_fallback = sum(1 for p in players if p.used_replacement_fallback)
    totals = np.empty(n_weeks, dtype=float)
    for wk in range(n_weeks):
        roster = [RosterPlayer(p.player_id, p.position, p.scores[wk], p.available[wk]) for p in players]
        totals[wk] = solve_lineup(roster, replacement_means).total_points
    return totals, n_fallback


def real_season_champion(
    final_state: DraftState,
    weekly_holdout: pl.DataFrame,
    replacement_by_position: Mapping[str, object],
    replacement_means: Mapping[str, float],
    seed: int,
    n_weeks: int = N_SCORE_WEEKS,
    regular_season_weeks: int = REGULAR_SEASON_WEEKS,
    playoff_byes: int = PLAYOFF_BYES,
) -> tuple[int, int]:
    """Play one completed draft's ten rosters through real 2024 results.
    Reuses `sim.season`'s own schedule/seeding/playoff-bracket machinery
    (`build_regular_season_schedule`, `_seed_teams`, `_run_playoffs`) with
    `n_sims=1` so this stays bracket-for-bracket identical to the rest of
    the project's simulated seasons -- only the per-week *scores* are real
    rather than sampled. Returns `(champion_team_1indexed, n_fallback)`.
    """
    weekly_lookup = _weekly_lookup(weekly_holdout, n_weeks)
    kicker_lookup = _kicker_gsis_by_name(weekly_holdout)
    rng = np.random.default_rng(seed)

    n_teams = final_state.n_teams
    team_totals = np.empty((n_teams, 1, n_weeks), dtype=float)
    total_fallback = 0
    for team in range(1, n_teams + 1):
        totals, n_fb = real_team_weekly_totals(
            list(final_state.rosters[team]),
            weekly_lookup,
            kicker_lookup,
            replacement_by_position,
            replacement_means,
            rng,
            n_weeks,
        )
        team_totals[team - 1, 0, :] = totals
        total_fallback += n_fb

    team_totals = team_totals + rng.uniform(0, 1e-9, size=team_totals.shape)
    schedule = build_regular_season_schedule(n_teams, regular_season_weeks)

    wins = np.zeros((n_teams, 1), dtype=float)
    points_for = np.zeros((n_teams, 1), dtype=float)
    for week, pairs in enumerate(schedule):
        for a, b in pairs:
            sa, sb = team_totals[a, :, week], team_totals[b, :, week]
            wins[a] += sa > sb
            wins[b] += sb > sa
            points_for[a] += sa
            points_for[b] += sb

    seeds = _seed_teams(wins, points_for, rng)
    champion = _run_playoffs(team_totals, seeds, regular_season_weeks, playoff_byes)
    return int(champion[0]) + 1, total_fallback


# ---------------------------------------------------------------------------
# One full realization: four contenders' drafts, same seed, scored on the
# same real 2024 results.


@dataclass(frozen=True)
class RealizationResult:
    seed: int
    champion_by_contender: dict[str, int]
    our_win: dict[str, bool]
    n_fallback_by_contender: dict[str, int]
    round_divergence_engine_vs_adp: dict[int, bool]  # our round -> did engine differ from ADP


def run_realization(
    seed: int,
    ctx: HoldoutContext,
    our_team: int = DRAFT_SLOT,
    n_teams: int = N_TEAMS,
    score_seed_offset: int = 10_000_000,
) -> RealizationResult:
    states: dict[str, DraftState] = {}
    champions: dict[str, int] = {}
    n_fallback: dict[str, int] = {}

    for contender in CONTENDERS:
        state = run_one_draft(seed, our_team, contender, ctx, n_teams=n_teams)
        states[contender] = state
        champ, n_fb = real_season_champion(
            state,
            ctx.weekly_holdout,
            ctx.replacement_by_position,
            ctx.replacement_means,
            seed=seed + score_seed_offset,
        )
        champions[contender] = champ
        n_fallback[contender] = n_fb

    our_win = {c: (champ == our_team) for c, champ in champions.items()}

    engine_picks = {p.overall_pick: p.player_id for p in states["engine"].picks if p.team == our_team}
    adp_picks = {p.overall_pick: p.player_id for p in states["adp"].picks if p.team == our_team}
    divergence: dict[int, bool] = {}
    for overall_pick, engine_pid in engine_picks.items():
        rnd = (overall_pick - 1) // n_teams + 1
        divergence[rnd] = engine_pid != adp_picks.get(overall_pick)

    return RealizationResult(
        seed=seed,
        champion_by_contender=champions,
        our_win=our_win,
        n_fallback_by_contender=n_fallback,
        round_divergence_engine_vs_adp=divergence,
    )


# ---------------------------------------------------------------------------
# Aggregating N realizations


@dataclass(frozen=True)
class ContenderResult:
    name: str
    championship_rate: float
    se: float
    n: int


@dataclass(frozen=True)
class PairedComparison:
    contender: str
    paired_diff_pp: float  # engine - contender, percentage points, mean over realizations
    se_pp: float


@dataclass(frozen=True)
class RoundBucketDivergence:
    bucket: str
    divergence_rate: float
    n: int


@dataclass(frozen=True)
class BacktestSummary:
    n_realizations: int
    elapsed_seconds: float
    contenders: dict[str, ContenderResult]
    paired_vs_engine: dict[str, PairedComparison]
    round_bucket_divergence: dict[str, RoundBucketDivergence]


def summarize(results: Sequence[RealizationResult], elapsed_seconds: float) -> BacktestSummary:
    n = len(results)
    wins = {c: np.array([r.our_win[c] for r in results], dtype=float) for c in CONTENDERS}

    contenders = {}
    for c, arr in wins.items():
        rate = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        contenders[c] = ContenderResult(c, rate, se, n)

    engine_arr = wins["engine"]
    paired = {}
    for c in CONTENDERS:
        if c == "engine":
            continue
        diff = engine_arr - wins[c]
        mean_diff_pp = float(diff.mean()) * 100.0
        se_diff_pp = float(diff.std(ddof=1) / np.sqrt(n)) * 100.0 if n > 1 else float("nan")
        paired[c] = PairedComparison(c, mean_diff_pp, se_diff_pp)

    bucket_vals: dict[str, list[bool]] = {}
    for r in results:
        for rnd, diverged in r.round_divergence_engine_vs_adp.items():
            bucket_vals.setdefault(_round_bucket(rnd), []).append(diverged)
    bucket_summary = {
        bucket: RoundBucketDivergence(bucket, float(np.mean(vals)), len(vals))
        for bucket, vals in bucket_vals.items()
    }

    return BacktestSummary(n, elapsed_seconds, contenders, paired, bucket_summary)


def run_backtest(
    ctx: HoldoutContext,
    n_realizations: int,
    seed0: int = 1,
    our_team: int = DRAFT_SLOT,
    n_teams: int = N_TEAMS,
    progress: bool = True,
) -> tuple[BacktestSummary, list[RealizationResult]]:
    results: list[RealizationResult] = []
    t0 = time.perf_counter()
    report_every = max(1, n_realizations // 10)
    for i in range(n_realizations):
        seed = seed0 + i
        results.append(run_realization(seed, ctx, our_team=our_team, n_teams=n_teams))
        if progress and (i + 1) % report_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{n_realizations} realizations, {elapsed:.1f}s elapsed")
    elapsed = time.perf_counter() - t0
    return summarize(results, elapsed), results
