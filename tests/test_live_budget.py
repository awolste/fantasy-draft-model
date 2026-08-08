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


def test_live_budget_is_strictly_cheaper_than_full():
    assert LIVE_BUDGET.work < FULL_BUDGET.work


def test_budget_is_hashable_so_it_can_key_a_cache():
    assert hash(Budget(1, 2, 3, seed=4)) == hash(Budget(1, 2, 3, seed=4))


def test_work_is_the_product_of_the_three_dimensions():
    assert Budget(2, 3, 5, seed=1).work == 30
