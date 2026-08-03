"""Per-player weekly fantasy-points distributions, anchored to 2026 consensus.

Two independent things get fitted from history and then combined:

- **Shape** (`TierShape`): how spread out and skewed a role tier's weekly
  scoring is -- e.g. a WR1's week-to-week profile is genuinely different
  from a WR5's, not just a scaled copy. Fitted once per (position, tier)
  from 2015-2025 `weekly_stats`, cached to disk.
- **Mean** (the rank -> points curve): what a specific 2026 player, at a
  specific *ordinal* rank, is expected to average per week. Fitted once per
  position from `adp_history` joined to actual season performance, cached
  to disk.

`build_player_pool()` combines the two: every 2026 player gets the shape of
their (position, tier) and a mean read off their position's rank curve, via
`anchor_tier_shape`, which solves for the one free scale parameter so the
mixture's expectation matches the anchored mean exactly.

See `models/base.py` for the `WeeklyDistribution` Protocol every
distribution here satisfies, including the flatten-weeks-into-one-`size`
performance note that Task 7's simulator depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import optimize, stats

from ..ids import load_crosswalk, match_by_name, normalize_name
from ..store import check_cache_fresh, exists, fingerprint, read, write, write_fingerprint
from .base import WeeklyDistribution
from .defense import dst_distribution

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


def _season_position_rank(weekly: pl.DataFrame) -> pl.DataFrame:
    """Rank each player's season by their own average points, within
    (season, position). Basis for Step 2's tiering -- "that season's finish"
    rather than preseason rank, since it is directly available from
    `weekly_stats` with no extra join/matching noise, and captures who
    actually *played like* a WR1 that year regardless of preseason hype."""
    season_avg = (
        weekly.group_by(["season", "position", "player_id"])
        .agg(pl.col("fantasy_points").mean().alias("season_avg"), pl.len().alias("games"))
        .filter(pl.col("games") >= MIN_GAMES_FOR_SEASON_RANK)
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


def load_or_fit_tier_shapes(weekly: pl.DataFrame, force_refit: bool = False) -> pl.DataFrame:
    """Cached wrapper around `fit_tier_shapes`. A cache hit is checked
    against a fingerprint of `weekly` (see `store.fingerprint`) recorded
    alongside the cache -- if `weekly` has changed (e.g. re-ingested with
    new data) since the cache was written, this raises `CacheStaleError`
    rather than silently serving parameters fit from the old data. Pass
    `force_refit=True` to refit intentionally."""
    fp = fingerprint(weekly)
    if not force_refit and exists("distribution_tier_shapes"):
        check_cache_fresh("distribution_tier_shapes", fp)
        return read("distribution_tier_shapes")
    table = fit_tier_shapes(weekly)
    write("distribution_tier_shapes", table)
    write_fingerprint("distribution_tier_shapes", fp)
    return table


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


# ---------------------------------------------------------------------------
# Step 3: rank -> weekly-mean curve, per position


def _decay_curve(rank: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a * rank^-b + c: steeply decreasing for small rank, flattening to a
    floor `c` as rank grows -- the shape Step 3 asks for."""
    return a * np.power(rank, -b) + c


# Deep tail (past the ADP-fit range) is built from finish-rank bins, not
# extrapolated from the ADP power law. See `fit_rank_curves` for why.
DEEP_TAIL_N_BINS = 8
MIN_PROJECTED_MEAN = 0.5  # no rosterable player realistically projects to ~0 ppg


@dataclass(frozen=True)
class RankCurve:
    """Two-piece rank -> weekly-mean-points curve.

    Ranks at or below `boundary_rank` use the ADP-fitted power law
    (`a*rank^-b + c`) -- this is the range the curve was actually validated
    against real preseason-ADP-to-performance data. Ranks beyond it use a
    separate finish-rank-based tail (`deep_knot_ranks`/`deep_knot_ppg`),
    rescaled so it's exactly continuous with the ADP piece at the boundary.
    See `fit_rank_curves` for how the tail is built and why it isn't just
    the ADP power law extended further.

    `max_supported_rank` is the deepest rank this curve was actually built
    to cover (max of the historical finish-rank data available and the
    current pool's depth at fit time). Requesting a rank beyond it raises --
    see `mean_for_rank`.
    """

    position: str
    a: float
    b: float
    c: float
    n_players: int
    boundary_rank: int
    deep_knot_ranks: tuple[float, ...]
    deep_knot_ppg: tuple[float, ...]
    max_supported_rank: int

    def mean_for_rank(self, position_rank: int) -> float:
        if position_rank > self.max_supported_rank:
            raise ValueError(
                f"{self.position} rank {position_rank} exceeds this curve's fitted range "
                f"(max_supported_rank={self.max_supported_rank}). Refit with a rankings frame "
                f"covering this depth rather than silently extrapolating -- see fit_rank_curves."
            )
        if position_rank <= self.boundary_rank:
            value = float(_decay_curve(np.array([position_rank], dtype=float), self.a, self.b, self.c)[0])
        else:
            value = float(np.interp(position_rank, self.deep_knot_ranks, self.deep_knot_ppg))
        return max(value, MIN_PROJECTED_MEAN)


def fit_rank_curves(
    adp_history: pl.DataFrame, weekly: pl.DataFrame, rankings: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit the two-piece rank -> weekly-mean curve described in `RankCurve`.

    **ADP piece** (unchanged from the original single-curve version): for
    each past season (`adp_history`, 2018-2024), rank players within
    position by preseason ADP, join to their actual that-season per-game
    average from `weekly_stats`, and fit `a*rank^-b + c` via nonlinear
    least squares. ADP (not FantasyPros ECR) is used as the preseason
    signal because it's the only historical, multi-season ordinal ranking
    in this dataset -- `rankings_2026` is 2026-only. This piece was
    independently validated against observed mean ppg bucketed by ADP rank
    and holds up well within its range (see the calibration report); it is
    used *only* up to the deepest rank it was actually fit on
    (`boundary_rank` = the max ADP-observed position rank, e.g. QB~27,
    RB~65, WR~74, TE~22, K~15) -- extending it past that range with an
    unconstrained power law was the bug this function now fixes (a QB50
    projected at 16.5 ppg, versus a real career-backup level closer to 3).

    **Deep piece**: past `boundary_rank`, most 2026-ranked players have no
    ADP at all (they're undrafted in the historical ADP data), so there is
    no preseason signal to regress against. Instead this uses each
    historical player-season's *finish* rank (season-long average points,
    ranked within position -- the same signal `_season_position_rank`
    computes for tiering) as the deep-rank ground truth, binned into
    `DEEP_TAIL_N_BINS` windows from `boundary_rank` out to the deeper of
    (deepest finish rank observed, deepest rank `rankings` actually needs),
    with bin means forced non-increasing (a deeper finish can't mean higher
    expected points -- protects against per-bin noise producing a local
    uptick).

    Finish rank is *ex-post*: a player who finishes rank 120 is drawn from
    a population that includes both preseason unknowns who broke out
    partway (removed from this bucket once they climb higher) and
    preseason-plausible players who busted -- a different, generally
    narrower population than "every player who *will end up* ranked
    around 120," which is what an ex-ante projection needs. Rather than
    use the deep bins' absolute levels directly (which would inherit that
    bias), only their *shape* is used: the whole binned deep curve is
    rescaled by one constant so it exactly matches the ADP piece's
    (validated) value at `boundary_rank`. That constant is an implicit,
    empirically-anchored correction for the ex-post/ex-ante gap, assumed
    roughly proportional across the deep range -- there's no ex-ante data
    for undrafted-depth players in this dataset to check that assumption
    further, which is the part of this fix to trust least.
    """
    adp = adp_history.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("_key")
    )
    adp = adp.with_columns(
        pl.col("adp").rank(method="ordinal").over(["season", "position"]).alias("pos_rank")
    )

    season_avg_adp = (
        weekly.group_by(["season", "position", "player_name"])
        .agg(pl.col("fantasy_points").mean().alias("avg_pts"), pl.len().alias("games"))
        .filter(pl.col("games") >= 2)
    )
    season_avg_adp = season_avg_adp.with_columns(
        pl.col("player_name").map_elements(normalize_name, return_dtype=pl.String).alias("_key")
    )

    joined = adp.join(
        season_avg_adp.select(["season", "position", "_key", "avg_pts"]),
        on=["season", "position", "_key"],
        how="inner",
    )

    required_depth: dict[str, int] = {}
    if rankings is not None:
        ranked = rankings.with_columns(
            pl.col("rank").rank(method="ordinal").over("position").alias("position_rank")
        )
        required_depth = {
            r["position"]: int(r["position_rank"])
            for r in ranked.group_by("position")
            .agg(pl.col("position_rank").max().alias("position_rank"))
            .to_dicts()
        }

    ranked_hist = _season_position_rank(weekly)  # season, position, player_id, season_avg, season_rank

    adp_rows = []
    deep_rows = []
    for position in TIER_BREAKS:
        d = joined.filter(pl.col("position") == position)
        r = d["pos_rank"].to_numpy().astype(float)
        y = d["avg_pts"].to_numpy().astype(float)
        if len(r) < 20:
            raise ValueError(
                f"Only {len(r)} matched (adp, season-performance) pairs for {position} -- "
                f"too few to fit a rank curve."
            )
        p0 = [max(y.max(), 1.0), 0.5, max(float(y.min()), 0.5)]
        popt, _ = optimize.curve_fit(
            _decay_curve, r, y, p0=p0, bounds=([0, 0.01, 0], [200, 5, 50]), maxfev=20000
        )
        a, b, c = (float(v) for v in popt)
        boundary_rank = int(r.max())

        deep = ranked_hist.filter(
            (pl.col("position") == position) & (pl.col("season_rank") >= boundary_rank)
        )
        needed_depth = max(boundary_rank, required_depth.get(position, boundary_rank))
        max_supported_rank = int(max(deep["season_rank"].max(), needed_depth))

        edges = np.linspace(boundary_rank, max_supported_rank, DEEP_TAIL_N_BINS + 1)
        centers, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            bucket = deep.filter((pl.col("season_rank") >= lo) & (pl.col("season_rank") < hi))
            if bucket.height == 0:
                continue
            centers.append((lo + hi) / 2)
            means.append(float(bucket["season_avg"].mean()))
        if len(centers) < 2:
            raise ValueError(
                f"Only {len(centers)} deep-rank bins with data for {position} beyond rank "
                f"{boundary_rank} -- too few to build a deep-tail curve."
            )
        means_arr = np.minimum.accumulate(np.array(means))  # enforce non-increasing
        boundary_adp_value = float(_decay_curve(np.array([boundary_rank]), a, b, c)[0])
        scale = boundary_adp_value / means_arr[0]
        scaled_means = means_arr * scale

        adp_rows.append(
            {
                "position": position,
                "a": a,
                "b": b,
                "c": c,
                "n_players": len(r),
                "boundary_rank": boundary_rank,
                "max_supported_rank": max_supported_rank,
                "deep_scale": float(scale),
            }
        )
        deep_rows.extend(
            {"position": position, "knot_rank": float(cr), "knot_ppg": float(m)}
            for cr, m in zip(centers, scaled_means)
        )

    return pl.DataFrame(adp_rows), pl.DataFrame(deep_rows)


def load_or_fit_rank_curves(
    adp_history: pl.DataFrame,
    weekly: pl.DataFrame,
    rankings: pl.DataFrame | None = None,
    force_refit: bool = False,
) -> dict[str, RankCurve]:
    """Cached wrapper around `fit_rank_curves`. `rankings` only affects how
    deep the tail is built (`required_depth`), so it is folded into the
    fingerprint whenever it's provided -- a cache fit with one `rankings`
    frame and reused with a materially different one (e.g. next year's
    pool going deeper) would otherwise silently reuse a tail that doesn't
    cover the new depth. See `load_or_fit_tier_shapes` for the staleness
    contract (`CacheStaleError` on mismatch, `force_refit=True` to refit
    intentionally)."""
    fp = fingerprint(adp_history, weekly, rankings) if rankings is not None else fingerprint(adp_history, weekly)
    cache_name = "distribution_rank_curves"
    if (
        not force_refit
        and exists("distribution_rank_curves")
        and exists("distribution_deep_rank_curve")
    ):
        check_cache_fresh(cache_name, fp)
        adp_table = read("distribution_rank_curves")
        deep_table = read("distribution_deep_rank_curve")
    else:
        adp_table, deep_table = fit_rank_curves(adp_history, weekly, rankings=rankings)
        write("distribution_rank_curves", adp_table)
        write("distribution_deep_rank_curve", deep_table)
        write_fingerprint(cache_name, fp)

    curves = {}
    for row in adp_table.to_dicts():
        position = row["position"]
        knots = (
            deep_table.filter(pl.col("position") == position)
            .sort("knot_rank")
        )
        curves[position] = RankCurve(
            position=position,
            a=row["a"],
            b=row["b"],
            c=row["c"],
            n_players=row["n_players"],
            boundary_rank=row["boundary_rank"],
            deep_knot_ranks=tuple(knots["knot_rank"].to_list()),
            deep_knot_ppg=tuple(knots["knot_ppg"].to_list()),
            max_supported_rank=row["max_supported_rank"],
        )
    return curves


# ---------------------------------------------------------------------------
# The distribution itself


@dataclass(frozen=True)
class ZeroInflatedGammaDistribution:
    """A mean-anchored, tier-shaped weekly points distribution.

    Three regimes, drawn each call from one shared uniform draw so the
    probabilities partition correctly:
      - negative (probability `p_negative`): magnitude ~ Gamma(neg_shape,
        neg_scale), negated -- lets genuinely bad weeks (INT-and-fumble
        QB weeks, a lost fumble with no yards) go below zero.
      - exact zero (probability `p_zero`): inactive/no-stats weeks.
      - positive (remaining probability): Gamma(gamma_shape, gamma_scale).

    `gamma_scale` is the only parameter not taken directly from the tier's
    fitted shape -- it is solved (see `anchor_tier_shape`) so this
    distribution's `mean` equals a specific 2026 player's rank-curve
    anchor, while `gamma_shape`/`p_negative`/`p_zero`/`neg_*` stay fixed at
    the tier's fitted values. That split is the whole point: two players
    with the same mean but different tiers get different `gamma_shape` (and
    different `p_zero`), hence different tails.
    """

    mean: float
    p_negative: float
    p_zero: float
    gamma_shape: float
    gamma_scale: float
    neg_shape: float
    neg_scale: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        u = rng.random(size)
        out = np.zeros(size, dtype=float)
        neg_mask = u < self.p_negative
        zero_mask = (~neg_mask) & (u < self.p_negative + self.p_zero)
        pos_mask = ~(neg_mask | zero_mask)

        n_neg = int(neg_mask.sum())
        n_pos = int(pos_mask.sum())
        if n_neg:
            out[neg_mask] = -rng.gamma(self.neg_shape, self.neg_scale, size=n_neg)
        if n_pos:
            out[pos_mask] = rng.gamma(self.gamma_shape, self.gamma_scale, size=n_pos)
        return out


MIN_GAMMA_SCALE = 1e-3


def anchor_tier_shape(shape: TierShape, target_mean: float) -> ZeroInflatedGammaDistribution:
    """Build a concrete distribution: `shape`'s fixed parameters, plus a
    Gamma scale solved so E[X] == target_mean.

    E[X] = p_negative * (-neg_mean_magnitude) + p_positive * (gamma_shape *
    gamma_scale). Everything except gamma_scale is fixed by `shape`, so this
    is one linear equation in gamma_scale.
    """
    p_positive = 1.0 - shape.p_negative - shape.p_zero
    if p_positive <= 0:
        raise ValueError(f"{shape.position} tier {shape.tier} has no positive-week mass to anchor")
    numerator = target_mean + shape.p_negative * shape.neg_mean_magnitude
    gamma_scale = numerator / (p_positive * shape.gamma_shape)
    gamma_scale = max(gamma_scale, MIN_GAMMA_SCALE)
    return ZeroInflatedGammaDistribution(
        mean=target_mean,
        p_negative=shape.p_negative,
        p_zero=shape.p_zero,
        gamma_shape=shape.gamma_shape,
        gamma_scale=gamma_scale,
        neg_shape=shape.neg_shape,
        neg_scale=shape.neg_scale,
    )


# ---------------------------------------------------------------------------
# Step 5: public pool interface


@dataclass(frozen=True)
class PlayerDistribution:
    """One draftable 2026 player: identity plus a sampleable distribution."""

    player_id: str
    name: str
    position: str
    rank: int
    tier: int | None  # None for DST, which has no tiering
    distribution: WeeklyDistribution


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def _player_id_from_crosswalk_row(row: dict) -> str | None:
    if row.get("gsis_id"):
        return row["gsis_id"]
    if row.get("espn_id"):
        return f"espn_{row['espn_id']}"
    if row.get("sleeper_id"):
        return f"sleeper_{row['sleeper_id']}"
    return None


def build_player_pool(
    rankings: pl.DataFrame | None = None,
    weekly: pl.DataFrame | None = None,
    adp_history: pl.DataFrame | None = None,
    crosswalk: pl.DataFrame | None = None,
    force_refit: bool = False,
) -> dict[str, PlayerDistribution]:
    """Build the draftable 2026 player pool.

    Every skill-position (QB/RB/WR/TE) player is matched to the ID
    crosswalk by name; unmatched players are excluded and reported (never
    silently defaulted). Kickers bypass ID matching entirely -- the
    crosswalk uses "PK" where every other source in this project uses "K",
    so kickers match at ~0% -- and are assigned a distribution by rank
    directly. Defenses use the single shared `dst_distribution()` for every
    team, per the Task 2 owner decision.

    `rankings`/`weekly`/`adp_history`/`crosswalk` default to this project's
    persisted datasets (`store.read` / `ids.load_crosswalk`) so this can be
    called with no arguments in normal use; the parameters exist so tests
    (and any future backtest against a different season) can inject
    synthetic or historical data instead.
    """
    if rankings is None:
        rankings = read("rankings_2026")
    if weekly is None:
        weekly = read("weekly_stats")
    if adp_history is None:
        adp_history = read("adp_history")
    if crosswalk is None:
        crosswalk = load_crosswalk()

    tier_shapes = load_or_fit_tier_shapes(weekly, force_refit=force_refit)
    rank_curve_lookup = load_or_fit_rank_curves(
        adp_history, weekly, rankings=rankings, force_refit=force_refit
    )

    tier_shape_lookup = {
        (r["position"], r["tier"]): tier_shape_from_row(r) for r in tier_shapes.to_dicts()
    }

    ranked = rankings.with_columns(
        pl.col("rank").rank(method="ordinal").over("position").alias("position_rank")
    )

    pool: dict[str, PlayerDistribution] = {}
    excluded: list[dict] = []

    # --- skill positions: match to crosswalk, exclude on failure ---
    skill = ranked.filter(pl.col("position").is_in(SKILL_POSITIONS))
    matched = match_by_name(skill, crosswalk)
    for row in matched.to_dicts():
        player_id = _player_id_from_crosswalk_row(row)
        if player_id is None:
            excluded.append({"name": row["name"], "position": row["position"], "rank": row["rank"]})
            continue
        position = row["position"]
        position_rank = int(row["position_rank"])
        tier = tier_for_rank(position, position_rank)
        curve = rank_curve_lookup[position]
        target_mean = curve.mean_for_rank(position_rank)
        shape = tier_shape_lookup[(position, tier)]
        dist = anchor_tier_shape(shape, target_mean)
        pool[player_id] = PlayerDistribution(
            player_id=player_id,
            name=row["name"],
            position=position,
            rank=int(row["rank"]),
            tier=tier,
            distribution=dist,
        )

    # --- kickers: rank-anchored directly, no crosswalk ---
    kickers = ranked.filter(pl.col("position") == "K")
    for row in kickers.to_dicts():
        position_rank = int(row["position_rank"])
        tier = tier_for_rank("K", position_rank)
        curve = rank_curve_lookup["K"]
        target_mean = curve.mean_for_rank(position_rank)
        shape = tier_shape_lookup[("K", tier)]
        dist = anchor_tier_shape(shape, target_mean)
        player_id = f"K::{row['name']}"
        pool[player_id] = PlayerDistribution(
            player_id=player_id,
            name=row["name"],
            position="K",
            rank=int(row["rank"]),
            tier=tier,
            distribution=dist,
        )

    # --- defenses: one shared distribution for every team ---
    dsts = ranked.filter(pl.col("position") == "DST")
    shared_dst = dst_distribution()
    for row in dsts.to_dicts():
        player_id = f"DST::{row['name']}"
        pool[player_id] = PlayerDistribution(
            player_id=player_id,
            name=row["name"],
            position="DST",
            rank=int(row["rank"]),
            tier=None,
            distribution=shared_dst,
        )

    if excluded:
        print(f"build_player_pool: excluded {len(excluded)} unmatched skill players:")
        for e in excluded[:20]:
            print(f"  rank {e['rank']:>4}  {e['position']:<3}  {e['name']}")
        if len(excluded) > 20:
            print(f"  ... and {len(excluded) - 20} more")

    return pool


def excluded_players(
    rankings: pl.DataFrame,
    crosswalk: pl.DataFrame | None = None,
) -> list[dict]:
    """Recompute just the excluded-skill-player list (name, position, rank)
    without building the whole pool -- used by reporting/tests that only
    need the exclusion count."""
    if crosswalk is None:
        crosswalk = load_crosswalk()
    ranked = rankings.with_columns(
        pl.col("rank").rank(method="ordinal").over("position").alias("position_rank")
    )
    skill = ranked.filter(pl.col("position").is_in(SKILL_POSITIONS))
    matched = match_by_name(skill, crosswalk)
    out = []
    for row in matched.to_dicts():
        if _player_id_from_crosswalk_row(row) is None:
            out.append({"name": row["name"], "position": row["position"], "rank": row["rank"]})
    return out


# ---------------------------------------------------------------------------
# Calibration: the "starter cohort" ground truth used to check the fitted
# distributions' tail against reality, not just against themselves.
#
# Defined once here (not duplicated between the calibration test and any
# reporting script) so both always compare modeled output against an
# identical definition of "a starter." Top-N thresholds match
# `TIER_BREAKS`' tier-1 boundaries; MIN_GAMES=8 and MIN_SEASON=2021 pick a
# stricter, more recent slice than the tier-shape fit itself uses (which
# pools 2015-2025 at >=4 games) specifically to get an independent check,
# not a tautological one.

STARTER_COHORT_TOP_N: dict[str, int] = {"QB": 12, "TE": 12, "K": 12, "RB": 24, "WR": 24}
STARTER_COHORT_MIN_SEASON = 2021
STARTER_COHORT_MIN_GAMES = 8


def observed_starter_quantiles(weekly: pl.DataFrame) -> pl.DataFrame:
    """Empirical p50/p90/p95/p99 weekly fantasy points for each position's
    starter cohort (see `STARTER_COHORT_TOP_N`/`_MIN_SEASON`/`_MIN_GAMES`).

    This is measured directly off raw weekly points for players who
    actually finished as starters -- no tiering, no rank-curve anchoring,
    no fitted distribution involved. It is the "reality" side of the
    calibration test.

    p99 in particular is estimated from very few order statistics -- with
    ~900-1900 starter-weeks per position, the top 1% is only ~9-19 weeks
    (`n_weeks * 0.01`, reported per-row). Callers computing a tolerance
    against this value should account for that noise rather than treating
    it as exact; see the calibration test for a bootstrap-based estimate.
    """
    recent = weekly.filter(pl.col("season") >= STARTER_COHORT_MIN_SEASON)
    season_avg = (
        recent.group_by(["season", "position", "player_id"])
        .agg(pl.col("fantasy_points").mean().alias("season_avg"), pl.len().alias("games"))
        .filter(pl.col("games") >= STARTER_COHORT_MIN_GAMES)
    )
    season_avg = season_avg.with_columns(
        pl.col("season_avg")
        .rank(method="ordinal", descending=True)
        .over(["season", "position"])
        .alias("season_rank")
    )
    joined = recent.join(
        season_avg.select(["season", "position", "player_id", "season_rank"]),
        on=["season", "position", "player_id"],
        how="inner",
    )

    rows = []
    for position, top_n in STARTER_COHORT_TOP_N.items():
        x = joined.filter((pl.col("position") == position) & (pl.col("season_rank") <= top_n))[
            "fantasy_points"
        ].to_numpy()
        if len(x) == 0:
            continue
        p50, p90, p95, p99 = (float(v) for v in np.quantile(x, [0.5, 0.9, 0.95, 0.99]))
        rows.append(
            {
                "position": position,
                "n_weeks": len(x),
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "p99": p99,
                "ratio90": p90 / p50,
                "ratio95": p95 / p50,
                "ratio99": p99 / p50,
            }
        )
    return pl.DataFrame(rows)


def modeled_starter_quantile_ratios(
    pool: dict[str, PlayerDistribution],
    n_seeds: int = 5,
    samples_per_seed: int = 100_000,
) -> pl.DataFrame:
    """The modeled counterpart to `observed_starter_quantiles`: for each
    position's starter cohort (top-N *by 2026 rank* in `pool`), sample each
    player's fitted distribution across several seeds and report the mean
    p90/median, p95/median, and p99/median ratio.

    Seed-averaged (not a single draw) so the reported ratio isn't itself
    noisy enough to mask or manufacture a calibration gap. Unlike the
    observed side, the modeled p99 is cheap to make precise -- it's drawn
    from a closed-form fitted distribution, not counted off a few hundred
    real games -- so `samples_per_seed` is large enough that MC noise here
    is negligible next to the observed side's sampling noise.
    """
    by_pos: dict[str, list[PlayerDistribution]] = {}
    for p in pool.values():
        by_pos.setdefault(p.position, []).append(p)

    rows = []
    for position, top_n in STARTER_COHORT_TOP_N.items():
        players = sorted(by_pos.get(position, []), key=lambda p: p.rank)[:top_n]
        if not players:
            continue
        ratio90s, ratio95s, ratio99s = [], [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            for player in players:
                samples = player.distribution.sample(rng, samples_per_seed)
                median = float(np.median(samples))
                if median <= 0.5:
                    continue  # median collapses near zero for very deep/replacement tiers
                p90, p95, p99 = np.quantile(samples, [0.9, 0.95, 0.99])
                ratio90s.append(float(p90) / median)
                ratio95s.append(float(p95) / median)
                ratio99s.append(float(p99) / median)
        rows.append(
            {
                "position": position,
                "n_player_seed_obs": len(ratio90s),
                "ratio90": float(np.mean(ratio90s)),
                "ratio95": float(np.mean(ratio95s)),
                "ratio99": float(np.mean(ratio99s)),
            }
        )
    return pl.DataFrame(rows)
