"""Role-tier weekly-scoring shape, fitted once per (position, tier).

**Shape** (`TierShape`): how spread out and skewed a role tier's weekly
scoring is -- e.g. a WR1's week-to-week profile is genuinely different
from a WR5's, not just a scaled copy. Fitted once per (position, tier)
from 2015-2025 `weekly_stats`, cached to disk.

See `models/distribution.py` for how a `TierShape` is combined with a
rank-anchored mean (`models/rank_curve.py`) into a concrete sampleable
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import optimize, stats

from ..store import check_cache_fresh, exists, fingerprint, read, write, write_fingerprint

# ---------------------------------------------------------------------------
# Step 2: role tiers
#
# Tier boundaries follow the shape of a fantasy roster in this exact league
# (10 teams, 1 QB / 2 RB / 2 WR / 1 TE / 2 FLEX): RB and WR both feed the
# FLEX slots, so "starter-relevant" extends well past the position's own
# starting-slot count -- roughly 2 dedicated + a share of 2 FLEX per team,
# i.e. up to ~30-40 RBs/WRs are plausibly rostered as starters across the
# league in a given week, while QB (1 slot, no FLEX) and TE (1 slot + FLEX
# share) taper off faster. Breaks are round-number approximations of that
# roster math, not a precise formula -- the point is coarse bands that
# separate "clear starter" from "streamer" from "deep bench/waiver," which
# is exactly where Step 1's tier-level fits showed the shape genuinely
# changing (see the fit comparison in the task report).
TIER_BREAKS: dict[str, tuple[int, ...]] = {
    "QB": (12, 24),
    "RB": (12, 24, 36),
    "WR": (12, 24, 36),
    "TE": (12, 24),
    "K": (12,),
}

MIN_GAMES_FOR_SEASON_RANK = 4  # exclude injury-shortened/cameo seasons from tiering
MIN_NEGATIVE_SAMPLES_FOR_OWN_FIT = 10  # else fall back to the position-wide negative fit


def tier_for_rank(position: str, position_rank: int) -> int:
    """1-indexed tier: 1 = best tier (e.g. QB1-12), increasing = deeper."""
    breaks = TIER_BREAKS[position]
    for i, b in enumerate(breaks, start=1):
        if position_rank <= b:
            return i
    return len(breaks) + 1


def n_tiers(position: str) -> int:
    return len(TIER_BREAKS[position]) + 1


# ---------------------------------------------------------------------------
# Step 1/2: fitted shape per (position, tier)


@dataclass(frozen=True)
class TierShape:
    """Mean-independent weekly-scoring shape for one (position, tier).

    `p_negative` and `p_zero` are the fitted probabilities of a bust
    (negative) week and an exact-zero (inactive/no-stats) week. The
    remaining probability mass is a Gamma fit (`gamma_shape`, MLE with
    location fixed at 0) to the positive weeks -- `gamma_shape` alone is
    scale-free and captures the tier's relative skew/tail weight;
    `anchor_tier_shape` solves for the scale that hits a target mean.
    `neg_shape`/`neg_scale` are a Gamma fit to the *magnitude* of negative
    weeks (sampled and negated), letting genuinely bad weeks go below zero
    rather than collapsing them into the zero mass.
    """

    position: str
    tier: int
    p_negative: float
    p_zero: float
    gamma_shape: float
    neg_shape: float
    neg_scale: float
    n_weeks: int

    @property
    def neg_mean_magnitude(self) -> float:
        return self.neg_shape * self.neg_scale


TAIL_MATCH_QUANTILE = 0.9  # p90, matching the acceptance criterion's primary tail check


def _fit_gamma_shape(
    positive_values: np.ndarray, target_quantile: float = TAIL_MATCH_QUANTILE
) -> float:
    """Solve for the Gamma shape `k` whose p90/median ratio matches the
    *empirical* p90/median ratio of `positive_values`, rather than an MLE fit.

    This replaced a plain `scipy.stats.gamma.fit` (MLE) after a calibration
    check found MLE systematically overpredicts the upper tail relative to
    the acceptance criterion (Step 1: "judged primarily on upper-tail fit").
    Concretely, for QB tier 1 (2015-2025, top-12-by-season-finish, weeks
    normalized to each player's own season average): empirical p90/median
    was 1.53, matching independently-measured observed starter behavior
    (1.54) almost exactly, but the MLE fit implied 1.74 -- MLE optimizes
    overall log-likelihood, which is dominated by the bulk of the
    distribution and a handful of extreme outlier games, not by the p90
    quantile specifically. Quantile-matching targets the number the
    acceptance criterion actually checks.

    Gamma's p90/median ratio is scale-free and strictly decreasing in `k`
    (more symmetric as k grows), so this is a single monotonic equation in
    one unknown -- solved by bisection.
    """
    p50, target = np.quantile(positive_values, [0.5, target_quantile])
    if p50 <= 0:
        raise ValueError("Cannot tail-match a Gamma shape when the empirical median is <= 0")
    target_ratio = target / p50

    def residual(k: float) -> float:
        q50, qtarget = stats.gamma.ppf([0.5, target_quantile], k)
        return qtarget / q50 - target_ratio

    lo, hi = 0.02, 100.0
    if residual(lo) < 0 or residual(hi) > 0:
        # Ratio is outside what any Gamma shape can produce in this range
        # (extremely tight or extremely skewed data) -- fall back to MLE
        # rather than raising, but this should be rare and is worth knowing
        # about if it happens.
        shape, _loc, _scale = stats.gamma.fit(positive_values, floc=0)
        return float(shape)
    return float(optimize.brentq(residual, lo, hi))


def _demeaned_positive_ratios(df: pl.DataFrame) -> np.ndarray:
    """Each positive week's fantasy_points divided by that player-season's
    own average, isolating within-player week-to-week variance from
    between-player differences in skill level. Fitting the tail shape on
    pooled *raw* points conflates the two: a tier pools many different
    players' whole seasons together, so some of the apparent spread is
    really "this tier has a range of talent," not "any one player's week
    is this unpredictable" -- and only the latter is what the shape should
    capture, since the mean is anchored separately per player in Step 3.
    """
    positive = df.filter(pl.col("fantasy_points") > 0)
    ratios = (positive["fantasy_points"] / positive["season_avg"]).to_numpy()
    return ratios


def _fit_negative_gamma(negative_values: np.ndarray) -> tuple[float, float]:
    magnitudes = -negative_values
    shape, _loc, scale = stats.gamma.fit(magnitudes, floc=0)
    return float(shape), float(scale)


def _season_position_rank(
    weekly: pl.DataFrame, min_games: int = MIN_GAMES_FOR_SEASON_RANK
) -> pl.DataFrame:
    """Rank each player's season by their own average points, within
    (season, position). Basis for Step 2's tiering -- "that season's finish"
    rather than preseason rank, since it is directly available from
    `weekly_stats` with no extra join/matching noise, and captures who
    actually *played like* a WR1 that year regardless of preseason hype.

    `min_games` is a floor on games played before a player-season counts
    toward ranking, excluding injury-shortened/cameo seasons -- callers with
    a stricter cohort definition (e.g. the calibration starter cohort) can
    pass a higher value than the tier-shape fit's own default."""
    season_avg = (
        weekly.group_by(["season", "position", "player_id"])
        .agg(pl.col("fantasy_points").mean().alias("season_avg"), pl.len().alias("games"))
        .filter(pl.col("games") >= min_games)
    )
    return season_avg.with_columns(
        pl.col("season_avg")
        .rank(method="ordinal", descending=True)
        .over(["season", "position"])
        .alias("season_rank")
    )


def fit_tier_shapes(weekly: pl.DataFrame) -> pl.DataFrame:
    """Fit `TierShape` for every (position, tier) with enough data.

    Returns a flat parameter table (not `TierShape` objects) so it can be
    persisted directly.
    """
    ranked = _season_position_rank(weekly)
    joined = weekly.join(
        ranked.select(["season", "position", "player_id", "season_rank", "season_avg"]),
        on=["season", "position", "player_id"],
        how="inner",
    )

    rows = []
    for position in TIER_BREAKS:
        pos_all = joined.filter(pl.col("position") == position)
        all_x = pos_all["fantasy_points"].to_numpy()
        fallback_neg = None
        if (all_x < 0).sum() >= MIN_NEGATIVE_SAMPLES_FOR_OWN_FIT:
            fallback_neg = _fit_negative_gamma(all_x[all_x < 0])

        pos_all = pos_all.with_columns(
            pl.col("season_rank")
            .map_elements(lambda r, p=position: tier_for_rank(p, r), return_dtype=pl.Int64)
            .alias("tier")
        )
        for tier in range(1, n_tiers(position) + 1):
            tier_df = pos_all.filter(pl.col("tier") == tier)
            x = tier_df["fantasy_points"].to_numpy()
            if len(x) < 30:
                raise ValueError(
                    f"Only {len(x)} weeks for {position} tier {tier} -- too few to fit "
                    f"a stable shape. Check TIER_BREAKS or the historical data range."
                )
            p_negative = float((x < 0).mean())
            p_zero = float((x == 0).mean())
            positive = x[x > 0]
            if len(positive) < 20:
                raise ValueError(
                    f"Only {len(positive)} positive weeks for {position} tier {tier} -- "
                    f"cannot fit a Gamma shape reliably."
                )
            demeaned_ratios = _demeaned_positive_ratios(tier_df)
            gamma_shape = _fit_gamma_shape(demeaned_ratios)

            negative = x[x < 0]
            if len(negative) >= MIN_NEGATIVE_SAMPLES_FOR_OWN_FIT:
                neg_shape, neg_scale = _fit_negative_gamma(negative)
            elif fallback_neg is not None:
                neg_shape, neg_scale = fallback_neg
            else:
                # p_negative will be ~0 in this branch (too few negative
                # weeks position-wide to fit anything), so these values are
                # essentially unused -- placeholders to keep the dataclass
                # well-formed.
                neg_shape, neg_scale = 1.0, 0.5

            rows.append(
                {
                    "position": position,
                    "tier": tier,
                    "p_negative": p_negative,
                    "p_zero": p_zero,
                    "gamma_shape": gamma_shape,
                    "neg_shape": neg_shape,
                    "neg_scale": neg_scale,
                    "n_weeks": len(x),
                }
            )
    return pl.DataFrame(rows)


def load_or_fit_tier_shapes(
    weekly: pl.DataFrame, force_refit: bool = False
) -> dict[tuple[str, int], TierShape]:
    """Cached wrapper around `fit_tier_shapes`. A cache hit is checked
    against a fingerprint of `weekly` (see `store.fingerprint`) recorded
    alongside the cache -- if `weekly` has changed (e.g. re-ingested with
    new data) since the cache was written, this raises `CacheStaleError`
    rather than silently serving parameters fit from the old data. Pass
    `force_refit=True` to refit intentionally.

    Returns hydrated `TierShape` objects keyed by (position, tier), not the
    raw parameter table -- callers shouldn't have to know about
    `tier_shape_from_row` themselves (mirrors `load_or_fit_rank_curves`,
    which returns hydrated `RankCurve` objects)."""
    fp = fingerprint(weekly)
    if not force_refit and exists("distribution_tier_shapes"):
        check_cache_fresh("distribution_tier_shapes", fp)
        table = read("distribution_tier_shapes")
    else:
        table = fit_tier_shapes(weekly)
        write("distribution_tier_shapes", table)
        write_fingerprint("distribution_tier_shapes", fp)
    return {(r["position"], r["tier"]): tier_shape_from_row(r) for r in table.to_dicts()}


def tier_shape_from_row(row: dict) -> TierShape:
    return TierShape(
        position=row["position"],
        tier=row["tier"],
        p_negative=row["p_negative"],
        p_zero=row["p_zero"],
        gamma_shape=row["gamma_shape"],
        neg_shape=row["neg_shape"],
        neg_scale=row["neg_scale"],
        n_weeks=row["n_weeks"],
    )
