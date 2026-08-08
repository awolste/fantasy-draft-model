"""Calibration: the "starter cohort" ground truth used to check the fitted
distributions' tail against reality, not just against themselves.

Defined once here (not duplicated between the calibration test and any
reporting script) so both always compare modeled output against an
identical definition of "a starter." Top-N thresholds match
`tier_shape.TIER_BREAKS`' tier-1 boundaries; MIN_GAMES=8 and MIN_SEASON=2021
pick a stricter, more recent slice than the tier-shape fit itself uses
(which pools 2015-2025 at >=4 games) specifically to get an independent
check, not a tautological one.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .distribution import PlayerDistribution
from .tier_shape import _season_position_rank

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
    season_avg = _season_position_rank(recent, min_games=STARTER_COHORT_MIN_GAMES)
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
