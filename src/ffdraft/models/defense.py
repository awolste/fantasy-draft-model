"""Shared weekly DST (team defense/special teams) distribution.

Owner decision, 2026-08-02: every team in this league starts one DST, and
every team's DST draws from the *same* weekly distribution. Defensive
scoring is largely luck, there is little a manager controls at the draft,
and it is not what this project is trying to answer. Because all ten teams
share one distribution, DST contributes equal expected points to every
roster and cancels out of head-to-head and championship-probability
comparisons -- which is the point.

This module deliberately does NOT implement this league's full DST scoring
rules. Those rules -- points-allowed bands, yards-allowed bands, sacks,
turnovers, and return touchdowns -- are recorded in the design spec
(`docs/superpowers/specs/2026-08-02-ff-draft-2026-design.md`) but are out of
scope here. Implementing them properly would require a team-game-level
ingest (points allowed and yards allowed are properties of a team's game,
not of any player row, and are not derivable from the existing
`weekly_stats` table). That ingest is exactly the work this decision avoids.
If a future stage ever wants per-team defensive modeling, the rules are
already written down -- the missing piece is that ingest, not the scoring
logic.

Parameters below are hand-set, not fitted from data, per the design note's
explicit allowance ("a normal distribution with a hand-set mean and standard
deviation is acceptable here -- this is deliberately the least-precise part
of the model"). Reasoning:

- Mean of 7.5: the middle of this league's ~6-9 points/week anchor for a
  *starting* (roughly top-10) defense, not the league-wide average across
  all 32 teams.
- Standard deviation of 6.0: chosen so a blowout week can go negative (this
  league's points/yards-allowed bands both bottom out negative) without
  making negative weeks the norm, and so elite weeks reach the low-to-mid
  20s and beyond. At mean=7.5, stdev=6.0, a Normal distribution puts ~10.5%
  of weeks below zero and comfortably clears 20+ point weeks in any
  100,000-sample draw -- both matching the anchors in the design note.
"""

from dataclasses import dataclass

import numpy as np

from .base import WeeklyDistribution

DST_MEAN: float = 7.5
DST_STDEV: float = 6.0


@dataclass(frozen=True)
class DstDistribution:
    """A Normal(mean, stdev) weekly DST-points distribution.

    Deliberately has no per-team parameter. The shared distribution is the
    decision this class encodes -- a per-team hook would invite someone to
    fill it with noise later and reintroduce a distinction the league owner
    explicitly judged uninteresting.
    """

    mean: float = DST_MEAN
    stdev: float = DST_STDEV

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(self.mean, self.stdev, size)


def dst_distribution() -> WeeklyDistribution:
    """Return the single shared weekly DST distribution used by every team."""
    return DstDistribution()
