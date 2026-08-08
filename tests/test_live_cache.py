"""Tests for `ffdraft.live.cache`.

Two things here carry the Stage 4 architecture:

1. `test_cache_and_live_paths_agree_at_equal_budget` is the entire safety
   argument for having a precomputed path and a live path at all.
2. The key tests pin a deliberate approximation (top-N availability only).
   `scripts/validate_cache_key.py` checks that approximation against real
   full-budget recommendations; these only pin its mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ffdraft.live.cache import TOP_N_FOR_KEY, RecommendationCache, state_key
from ffdraft.store import CacheStaleError


@dataclass(frozen=True)
class _P:
    rank: int
    position: str


def _pool(n: int = 200) -> dict[str, _P]:
    return {f"p{i}": _P(rank=i, position="RB") for i in range(1, n + 1)}


def test_key_ignores_players_outside_the_top_n():
    """The whole point of the approximation: who is gone below the cutoff
    cannot change our pick, because we would never take them."""
    pool = _pool()
    a = state_key(8, {"QB": 1}, drafted_ids={"p150"}, pool=pool)
    b = state_key(8, {"QB": 1}, drafted_ids={"p151"}, pool=pool)
    assert a == b


def test_key_distinguishes_top_n_availability():
    pool = _pool()
    a = state_key(8, {"QB": 1}, drafted_ids={"p1"}, pool=pool)
    b = state_key(8, {"QB": 1}, drafted_ids={"p2"}, pool=pool)
    assert a != b


def test_key_distinguishes_pick_number_and_roster_shape():
    pool = _pool()
    base = state_key(8, {"QB": 1}, drafted_ids=set(), pool=pool)
    assert state_key(13, {"QB": 1}, drafted_ids=set(), pool=pool) != base
    assert state_key(8, {"QB": 2}, drafted_ids=set(), pool=pool) != base


def test_key_is_order_independent_and_hashable():
    pool = _pool()
    a = state_key(8, {"QB": 1, "RB": 2}, drafted_ids={"p3", "p5"}, pool=pool)
    b = state_key(8, {"RB": 2, "QB": 1}, drafted_ids={"p5", "p3"}, pool=pool)
    assert a == b
    assert hash(a) == hash(b)


def test_key_is_stable_when_the_pool_has_more_players_below_the_cutoff():
    """A pool that grows only below the cutoff must not change the key --
    otherwise a re-ingest that adds deep sleepers would invalidate every
    cached entry for no reason."""
    small, large = _pool(100), _pool(400)
    assert state_key(8, {}, {"p1"}, small) == state_key(8, {}, {"p1"}, large)


def test_top_n_is_the_documented_value():
    assert TOP_N_FOR_KEY == 60


def test_key_requires_a_pool_at_least_as_large_as_the_cutoff():
    """A pool smaller than the cutoff means the caller passed the wrong
    thing (a filtered subset, say). Silently keying on whatever is there
    would produce a key that collides across genuinely different boards."""
    with pytest.raises(ValueError, match="smaller than the cache-key cutoff"):
        state_key(8, {}, set(), _pool(10))


# ---------------------------------------------------------------------------
# Persistence and the staleness guard


def test_cache_round_trips_an_entry(tmp_path):
    c = RecommendationCache(path=tmp_path / "cache.json", fingerprint="fp1")
    key = state_key(8, {}, set(), _pool())
    c.put(key, [{"name": "X", "championship_probability": 0.11}])
    c.save()

    reloaded = RecommendationCache.load(tmp_path / "cache.json", fingerprint="fp1")
    assert reloaded.get(key)[0]["name"] == "X"
    assert len(reloaded) == 1


def test_loading_with_a_different_fingerprint_raises_rather_than_serving_stale(tmp_path):
    """A cache built against a stale player pool is confident, plausible and
    wrong -- the exact failure mode this project keeps hitting. It must be
    loud."""
    c = RecommendationCache(path=tmp_path / "cache.json", fingerprint="fp1")
    c.put(state_key(8, {}, set(), _pool()), [{"name": "X"}])
    c.save()

    with pytest.raises(CacheStaleError, match="Rebuild it"):
        RecommendationCache.load(tmp_path / "cache.json", fingerprint="fp2")


def test_missing_key_returns_none_so_the_caller_can_fall_back(tmp_path):
    c = RecommendationCache(path=tmp_path / "cache.json", fingerprint="fp1")
    assert c.get(state_key(13, {}, set(), _pool())) is None


# ---------------------------------------------------------------------------
# The consistency guarantee -- the reason two paths are safe at all


def test_cache_and_live_paths_agree_at_equal_budget():
    """Precompute and live fallback differ ONLY in their `Budget`. At the
    same budget they must produce identical output.

    If this fails, `recommend_pick` is not deterministic at a fixed seed and
    the precompute+fallback architecture is unsound. Investigate; do NOT add
    a tolerance to make it pass.
    """
    from ffdraft.backtest import fit_holdout_context
    from ffdraft.draft.rollout import DraftState, Pick, team_for_pick
    from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
    from ffdraft.live.budget import Budget
    from ffdraft.live.cache import recommend
    from ffdraft.live.context import DraftContext

    ctx = DraftContext.from_holdout(fit_holdout_context())

    taken = sorted(ctx.pool, key=lambda pid: ctx.pool[pid].rank)[: DRAFT_SLOT - 1]
    picks = [
        Pick(
            overall_pick=i + 1,
            team=team_for_pick(i + 1, N_TEAMS),
            player_id=pid,
            position=ctx.pool[pid].position,
        )
        for i, pid in enumerate(taken)
    ]
    state = DraftState.from_picks(picks, n_teams=N_TEAMS, rounds=DRAFT_ROUNDS)
    assert state.team_on_clock == DRAFT_SLOT

    budget = Budget(n_candidates=5, n_rollouts=3, n_sims_per_rollout=40, seed=7)
    a = recommend(state, ctx, budget)
    b = recommend(state, ctx, budget)

    assert [c.player_id for c in a.candidates] == [c.player_id for c in b.candidates]
    assert [c.championship_probability for c in a.candidates] == [
        c.championship_probability for c in b.candidates
    ]
    assert [c.standard_error for c in a.candidates] == [
        c.standard_error for c in b.candidates
    ]


def test_a_saved_cache_entry_equals_a_fresh_live_run_at_the_same_budget(tmp_path):
    """The end-to-end version of the guarantee.

    The test above proves `recommend_pick` is deterministic; the single
    call site in `live.cache` proves both paths run the same code. This
    proves the remaining link: that going through serialisation, disk, and
    reload does not alter what the UI ends up displaying. Without it, a
    float losing precision in JSON could make a cached pick disagree with a
    live one and nothing would notice.
    """
    from ffdraft.backtest import fit_holdout_context
    from ffdraft.draft.rollout import DraftState, Pick, team_for_pick
    from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
    from ffdraft.live.budget import Budget
    from ffdraft.live.cache import candidates_to_rows, recommend
    from ffdraft.live.context import DraftContext

    ctx = DraftContext.from_holdout(fit_holdout_context())
    taken = sorted(ctx.pool, key=lambda pid: ctx.pool[pid].rank)[: DRAFT_SLOT - 1]
    picks = [
        Pick(
            overall_pick=i + 1,
            team=team_for_pick(i + 1, N_TEAMS),
            player_id=pid,
            position=ctx.pool[pid].position,
        )
        for i, pid in enumerate(taken)
    ]
    state = DraftState.from_picks(picks, n_teams=N_TEAMS, rounds=DRAFT_ROUNDS)
    budget = Budget(n_candidates=5, n_rollouts=3, n_sims_per_rollout=40, seed=7)

    key = state_key(state.next_overall_pick, {}, state.drafted_ids, ctx.pool)
    precomputed = RecommendationCache(path=tmp_path / "c.json", fingerprint="fp")
    precomputed.put(key, candidates_to_rows(recommend(state, ctx, budget)))
    precomputed.save()

    from_cache = RecommendationCache.load(tmp_path / "c.json", fingerprint="fp").get(key)
    from_live = candidates_to_rows(recommend(state, ctx, budget))

    assert from_cache == from_live
