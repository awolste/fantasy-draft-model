"""Per-player weekly fantasy-points distributions, anchored to 2026 consensus.

Two independent things get fitted from history and then combined:

- **Shape** (`TierShape`, `models/tier_shape.py`): how spread out and skewed
  a role tier's weekly scoring is -- e.g. a WR1's week-to-week profile is
  genuinely different from a WR5's, not just a scaled copy. Fitted once per
  (position, tier) from 2015-2025 `weekly_stats`, cached to disk.
- **Mean** (`RankCurve`, `models/rank_curve.py`): what a specific 2026
  player, at a specific *ordinal* rank, is expected to average per week.
  Fitted once per position from `adp_history` joined to actual season
  performance, cached to disk.

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

from ..ids import load_crosswalk, match_by_name
from ..store import read
from .base import WeeklyDistribution
from .defense import dst_distribution
from .rank_curve import load_or_fit_rank_curves
from .tier_shape import TierShape, load_or_fit_tier_shapes, tier_for_rank

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


def _matched_skill_players(rankings: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Rank `rankings` by within-position ordinal rank, restrict to
    `SKILL_POSITIONS`, and match to the ID crosswalk by name.

    Shared setup for `build_player_pool` and `excluded_players` -- both need
    the same (ranked, skill-filtered, name-matched) frame to decide which
    skill players have a usable ID."""
    ranked = rankings.with_columns(
        pl.col("rank").rank(method="ordinal").over("position").alias("position_rank")
    )
    skill = ranked.filter(pl.col("position").is_in(SKILL_POSITIONS))
    return match_by_name(skill, crosswalk)


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

    tier_shape_lookup = load_or_fit_tier_shapes(weekly, force_refit=force_refit)
    rank_curve_lookup = load_or_fit_rank_curves(
        adp_history, weekly, rankings=rankings, force_refit=force_refit
    )

    ranked = rankings.with_columns(
        pl.col("rank").rank(method="ordinal").over("position").alias("position_rank")
    )

    pool: dict[str, PlayerDistribution] = {}
    excluded: list[dict] = []

    # --- skill positions: match to crosswalk, exclude on failure ---
    matched = _matched_skill_players(rankings, crosswalk)
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
    matched = _matched_skill_players(rankings, crosswalk)
    out = []
    for row in matched.to_dicts():
        if _player_id_from_crosswalk_row(row) is None:
            out.append({"name": row["name"], "position": row["position"], "rank": row["rank"]})
    return out
