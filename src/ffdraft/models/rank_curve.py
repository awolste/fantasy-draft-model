"""Rank -> weekly-mean-points curve, fitted once per position.

**Mean** (the rank -> points curve): what a specific 2026 player, at a
specific *ordinal* rank, is expected to average per week. Fitted once per
position from `adp_history` joined to actual season performance, cached
to disk.

See `models/distribution.py` for how a `RankCurve`-derived mean is combined
with a `models/tier_shape.py` `TierShape` into a concrete sampleable
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import optimize

from ..ids import normalize_name
from ..store import cache_namespace, check_cache_fresh, exists, fingerprint, read, write, write_fingerprint
from .tier_shape import TIER_BREAKS, _season_position_rank


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


def hydrate_rank_curves(adp_table: pl.DataFrame, deep_table: pl.DataFrame) -> dict[str, RankCurve]:
    """Turn `fit_rank_curves`'s two raw parameter tables into hydrated
    `RankCurve` objects keyed by position -- the same assembly
    `load_or_fit_rank_curves` does after a cache hit or a fresh fit,
    factored out so a caller that deliberately bypasses the cache (e.g.
    Stage 3's Task 6 backtest, which fits a leakage-safe rank curve from
    data restricted to seasons through 2023 and must not write that fit
    into the live, unnamespaced-by-season cache slot) can still reuse this
    exact assembly logic instead of duplicating it."""
    curves = {}
    for row in adp_table.to_dicts():
        position = row["position"]
        knots = deep_table.filter(pl.col("position") == position).sort("knot_rank")
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
    intentionally).

    `rankings` also legitimately varies *by context*, not just by season:
    Stage 3 calls this with the default 2026 `rankings` for the live
    recommender and with a 2024-ADP-anchored `rankings` for the historical
    backtest (see `scripts/season_report.py`), and alternates between the
    two repeatedly. Both are equally valid, current fits -- neither is
    "stale" relative to the other -- so the cache artifacts are namespaced
    by a hash of `rankings` alone (`store.cache_namespace`) so the two
    contexts get their own cache slot instead of one evicting the other.
    Genuine staleness (the *same* context's `adp_history`/`weekly`/
    `rankings` changing underneath it) is still caught: `fp` below is the
    full fingerprint of every input, checked against that namespaced slot's
    own stored fingerprint."""
    fp = fingerprint(adp_history, weekly, rankings) if rankings is not None else fingerprint(adp_history, weekly)
    if rankings is not None:
        adp_name = cache_namespace("distribution_rank_curves", rankings)
        deep_name = cache_namespace("distribution_deep_rank_curve", rankings)
    else:
        adp_name = "distribution_rank_curves"
        deep_name = "distribution_deep_rank_curve"
    cache_name = adp_name
    if not force_refit and exists(adp_name) and exists(deep_name):
        check_cache_fresh(cache_name, fp)
        adp_table = read(adp_name)
        deep_table = read(deep_name)
    else:
        adp_table, deep_table = fit_rank_curves(adp_history, weekly, rankings=rankings)
        write(adp_name, adp_table)
        write(deep_name, deep_table)
        write_fingerprint(cache_name, fp)

    return hydrate_rank_curves(adp_table, deep_table)
