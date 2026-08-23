"""Tests for `ffdraft.live.budget`."""

from __future__ import annotations

from ffdraft.live.budget import FULL_BUDGET, LIVE_BUDGET, Budget


def test_full_budget_matches_recommender_defaults():
    from ffdraft.draft.recommender import (
        DEFAULT_N_CANDIDATES,
        DEFAULT_N_ROLLOUTS,
        DEFAULT_N_SIMS_PER_ROLLOUT,
    )

    assert FULL_BUDGET.n_candidates == DEFAULT_N_CANDIDATES
    assert FULL_BUDGET.n_rollouts == DEFAULT_N_ROLLOUTS
    assert FULL_BUDGET.n_sims_per_rollout == DEFAULT_N_SIMS_PER_ROLLOUT


def test_live_budget_no_longer_sacrifices_precision_for_the_clock():
    """This used to assert the opposite, and the inversion is the point.

    `LIVE_BUDGET` was a *reduced* budget: the most simulation that fit a
    draft clock, deliberately cheaper than `FULL_BUDGET`. After the two
    speedup passes (HANDOFF section 15) the live budget was widened to
    15x300x150 and now does **more** work than `FULL_BUDGET` while still
    finishing in ~16s at our worst pick.

    `FULL_BUDGET` is pinned to `recommend_pick`'s own defaults, which were
    derived from per-unit costs measured before those speedups (a rollout
    cost 0.267s; it now costs ~12ms). It is kept as the historical
    reference it is, and deliberately not re-derived here -- changing the
    recommender's defaults would silently change the backtest too.
    """
    assert LIVE_BUDGET.work >= FULL_BUDGET.work


def test_budget_is_hashable_so_it_can_key_a_cache():
    assert hash(Budget(1, 2, 3, seed=4)) == hash(Budget(1, 2, 3, seed=4))


def test_work_is_the_product_of_the_three_dimensions():
    assert Budget(2, 3, 5, seed=1).work == 30
