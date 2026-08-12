"""The two recommendation budgets, and the measured basis for the live one.

There is exactly one `recommend_pick` call site in this package (see
`live.cache.recommend`); these objects are the only thing that differs
between the precomputed and the live-fallback path. Keeping the difference
as *data* rather than as two code paths is what makes the equivalence test
in `tests/test_live_cache.py` possible at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..draft.recommender import (
    DEFAULT_N_CANDIDATES,
    DEFAULT_N_ROLLOUTS,
    DEFAULT_N_SIMS_PER_ROLLOUT,
)


@dataclass(frozen=True)
class Budget:
    n_candidates: int
    n_rollouts: int
    n_sims_per_rollout: int
    seed: int

    @property
    def work(self) -> int:
        """Season simulations implied. Runtime is roughly linear in this."""
        return self.n_candidates * self.n_rollouts * self.n_sims_per_rollout

    @property
    def label(self) -> str:
        return f"{self.n_candidates}x{self.n_rollouts}x{self.n_sims_per_rollout}"


FULL_BUDGET = Budget(
    n_candidates=DEFAULT_N_CANDIDATES,
    n_rollouts=DEFAULT_N_ROLLOUTS,
    n_sims_per_rollout=DEFAULT_N_SIMS_PER_ROLLOUT,
    seed=20260804,
)

# MEASURED, not chosen. `scripts/measure_live_budget.py` on the native arm64
# venv, live 2026 context (510 players), at overall pick 8 -- the worst case,
# since later picks have fewer available players and run faster:
#
#   budget          work      elapsed   leader SE   leader
#   10x8x200      16,000       13.9s      1.12pp    Christian McCaffrey
#   10x12x250     30,000       21.5s      0.99pp    Christian McCaffrey
#   12x16x300     57,600       35.6s      1.17pp    Christian McCaffrey   <- LIVE
#   15x20x400    120,000       59.0s      0.92pp    Christian McCaffrey
#   15x30x500    225,000       94.0s      0.67pp    Christian McCaffrey
#   15x65x500    487,500      197.4s      0.45pp    Christian McCaffrey   <- FULL
#
# The leader is IDENTICAL at every budget, so going cheap costs precision on
# the probability, not the recommendation itself -- at least at this state.
#
# Why 12x16x300 rather than the 15x20x400 that also fits under 60s: the 90s
# pick clock is not 90s of compute. It has to cover entering the picks made
# since our last turn, reading the result, and deciding. 59s leaves ~30s of
# that; 35.6s leaves ~54s. The precision given up is 1.17pp vs 0.92pp of
# standard error, which is immaterial next to the model's own between-season
# SE of 4.15pp (HANDOFF section 10). Buying 0.25pp of simulation precision

# with half the human clock is a bad trade.

# NOTE (2026-08-11): the timings recorded above were measured BEFORE the
# `value.dominant_candidates` reduction and the `pick_probabilities` /
# rollout allocation fixes, which together made `recommend_pick` ~2.7x
# faster with bit-identical output. Re-measured at the same budgets:
#
#     LIVE_BUDGET (12x16x300)   35.6s  ->  12.6s
#
# The budgets themselves were left alone rather than quietly widened: the
# leader was already identical at every budget measured, so the headroom is
# available to spend on precision if wanted, but spending it is a decision
# to take deliberately and re-measure, not a side effect of an optimisation.
LIVE_BUDGET = Budget(n_candidates=12, n_rollouts=16, n_sims_per_rollout=300, seed=20260804)
