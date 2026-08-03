"""Tests for `draft.rollout`.

Most tests use a small, fully synthetic league (few teams, few rounds, a
hand-built pool) so they are fast and independent of the real data
pipeline -- same style as `tests/test_value.py`/`tests/test_roster.py`. A
final group runs against the real 2026 pool/replacement/opponent-model data
to produce the timing and plausibility report the task asks for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest

from ffdraft.draft.rollout import (
    DraftState,
    Pick,
    run_rollout,
    slot_pick_numbers,
    team_for_pick,
)
from ffdraft.models.distribution import PlayerDistribution
from ffdraft.models.opponent import DEFAULT_ROSTER_DECAY, DEFAULT_TEMPERATURE, OpponentModel


@dataclass(frozen=True)
class ConstantDist:
    value: float

    @property
    def mean(self) -> float:
        return self.value

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, self.value)


def _pd(player_id: str, position: str, rank: int, value: float) -> PlayerDistribution:
    return PlayerDistribution(
        player_id=player_id, name=player_id, position=position, rank=rank, tier=1,
        distribution=ConstantDist(value),
    )


REPLACEMENT_BY_POSITION = {
    "QB": ConstantDist(18.3),
    "RB": ConstantDist(9.6),
    "WR": ConstantDist(9.9),
    "TE": ConstantDist(8.1),
    "K": ConstantDist(7.9),
    "DST": ConstantDist(7.5),
}

# A trivial opponent model: no positional bias, no residual noise reported.
FLAT_MODEL = OpponentModel(league_mu=0.0, pos_effect={}, sigma2=0.01)


def _big_synthetic_pool(n: int = 200) -> dict[str, PlayerDistribution]:
    positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    pool = {}
    for i in range(n):
        pos = positions[i % len(positions)]
        pool[f"p{i}"] = _pd(f"p{i}", pos, rank=i + 1, value=30.0 - i * 0.05)
    return pool


# ---------------------------------------------------------------------------
# Snake order helper


def test_team_for_pick_round1_is_straight_order():
    for pick in range(1, 11):
        assert team_for_pick(pick, n_teams=10) == pick


def test_team_for_pick_round2_reverses():
    # Round 2 (picks 11-20) goes 10 -> 1.
    assert team_for_pick(11, n_teams=10) == 10
    assert team_for_pick(20, n_teams=10) == 1


def test_slot_8_pick_numbers_match_expected_overall_picks():
    # From slot 8 in a 10-team snake draft, our picks land at 8, 13, 28,
    # 33, 48, 53, ... -- verified explicitly since an off-by-one in snake
    # direction is easy to write and invisible in aggregate results.
    picks = slot_pick_numbers(draft_slot=8, n_teams=10, rounds=6)
    assert picks == (8, 13, 28, 33, 48, 53)


def test_team_for_pick_every_slot_round3_straight_again():
    # Round 3 (picks 21-30) is straight order again (odd round).
    assert team_for_pick(21, n_teams=10) == 1
    assert team_for_pick(28, n_teams=10) == 8
    assert team_for_pick(30, n_teams=10) == 10


# ---------------------------------------------------------------------------
# DraftState construction


def test_draft_state_from_no_picks_is_at_pick_one():
    state = DraftState.from_picks([], n_teams=4, rounds=3)
    assert state.next_overall_pick == 1
    assert state.team_on_clock == 1
    assert not state.is_complete


def test_draft_state_from_partial_picks_respects_prior_picks():
    picks = [
        Pick(overall_pick=1, team=1, player_id="p0", position="RB"),
        Pick(overall_pick=2, team=2, player_id="p1", position="WR"),
    ]
    state = DraftState.from_picks(picks, n_teams=4, rounds=3)
    assert state.next_overall_pick == 3
    assert state.team_on_clock == 3
    assert state.drafted_ids == frozenset({"p0", "p1"})
    assert state.rosters[1] == (("p0", "RB"),)
    assert state.rosters[2] == (("p1", "WR"),)
    assert state.rosters[3] == ()


def test_draft_state_rejects_duplicate_player_id():
    picks = [
        Pick(overall_pick=1, team=1, player_id="p0", position="RB"),
        Pick(overall_pick=2, team=2, player_id="p0", position="RB"),
    ]
    with pytest.raises(ValueError):
        DraftState.from_picks(picks, n_teams=4, rounds=3)


def test_draft_state_rejects_wrong_team_for_pick_number():
    # Overall pick 2 in a 4-team snake belongs to team 2, not team 3.
    picks = [Pick(overall_pick=2, team=3, player_id="p0", position="RB")]
    with pytest.raises(ValueError):
        DraftState.from_picks(picks, n_teams=4, rounds=3)


def test_draft_state_rejects_out_of_order_picks():
    picks = [
        Pick(overall_pick=1, team=1, player_id="p0", position="RB"),
        Pick(overall_pick=3, team=3, player_id="p1", position="WR"),  # gap at 2
    ]
    with pytest.raises(ValueError):
        DraftState.from_picks(picks, n_teams=4, rounds=3)


def test_draft_state_complete_when_all_slots_filled():
    picks = [
        Pick(overall_pick=i, team=team_for_pick(i, n_teams=2), player_id=f"p{i}", position="RB")
        for i in range(1, 5)  # 2 teams x 2 rounds
    ]
    state = DraftState.from_picks(picks, n_teams=2, rounds=2)
    assert state.is_complete


# ---------------------------------------------------------------------------
# run_rollout: completion, no dupes, snake order, reproducibility


def test_rollout_fills_every_roster_to_configured_size():
    pool = _big_synthetic_pool(200)
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    rng = np.random.default_rng(1)
    final = run_rollout(state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, rng, our_team=1)
    assert final.is_complete
    for team in range(1, 5):
        assert len(final.rosters[team]) == 6


def test_rollout_never_drafts_a_player_twice():
    pool = _big_synthetic_pool(200)
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    rng = np.random.default_rng(2)
    final = run_rollout(state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, rng, our_team=1)
    all_ids = [pid for team in range(1, 5) for pid, _ in final.rosters[team]]
    assert len(all_ids) == len(set(all_ids))


def test_rollout_snake_order_is_correct_including_turn_between_rounds():
    pool = _big_synthetic_pool(200)
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    rng = np.random.default_rng(3)
    final = run_rollout(state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, rng, our_team=1)
    for pick in final.picks:
        assert pick.team == team_for_pick(pick.overall_pick, n_teams=4)


def test_rollout_our_picks_from_slot_8_land_at_expected_numbers():
    pool = _big_synthetic_pool(500)
    state = DraftState.from_picks([], n_teams=10, rounds=6)
    rng = np.random.default_rng(4)
    final = run_rollout(state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, rng, our_team=8)
    our_picks = tuple(sorted(p.overall_pick for p in final.picks if p.team == 8))
    assert our_picks == (8, 13, 28, 33, 48, 53)


def test_rollout_identical_seeds_produce_identical_drafts():
    pool = _big_synthetic_pool(200)
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    final_a = run_rollout(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(42), our_team=1
    )
    final_b = run_rollout(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(42), our_team=1
    )
    assert final_a.picks == final_b.picks


def test_rollout_different_seeds_produce_different_drafts():
    pool = _big_synthetic_pool(200)
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    final_a = run_rollout(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(1), our_team=1
    )
    final_b = run_rollout(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(2), our_team=1
    )
    assert final_a.picks != final_b.picks


def test_rollout_from_partial_state_respects_prior_picks():
    pool = _big_synthetic_pool(200)
    prior_picks = [
        Pick(overall_pick=1, team=1, player_id="p0", position="QB"),
        Pick(overall_pick=2, team=2, player_id="p1", position="RB"),
        Pick(overall_pick=3, team=3, player_id="p2", position="WR"),
        Pick(overall_pick=4, team=4, player_id="p3", position="TE"),
    ]
    state = DraftState.from_picks(prior_picks, n_teams=4, rounds=6)
    rng = np.random.default_rng(5)
    final = run_rollout(state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, rng, our_team=1)

    # The prior picks must survive untouched, in their original slots.
    assert final.picks[:4] == tuple(prior_picks)
    assert final.rosters[1][0] == ("p0", "QB")
    assert final.rosters[2][0] == ("p1", "RB")
    assert final.rosters[3][0] == ("p2", "WR")
    assert final.rosters[4][0] == ("p3", "TE")
    # And none of those players get redrafted later.
    all_ids = [pid for team in range(1, 5) for pid, _ in final.rosters[team]]
    assert all_ids.count("p0") == 1


def test_rollout_supports_17_round_configuration():
    pool = _big_synthetic_pool(200)
    state = DraftState.from_picks([], n_teams=10, rounds=17)
    rng = np.random.default_rng(6)
    final = run_rollout(state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, rng, our_team=8)
    assert final.is_complete
    for team in range(1, 11):
        assert len(final.rosters[team]) == 17


def test_rollout_does_not_touch_global_numpy_state():
    pool = _big_synthetic_pool(200)
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    before = np.random.get_state()
    run_rollout(state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(7), our_team=1)
    after = np.random.get_state()
    assert before[1].tolist() == after[1].tolist()


# ---------------------------------------------------------------------------
# Real 2026 pool: timing and plausibility (see task report for narrative)


@pytest.fixture(scope="module")
def real_fixtures():
    from ffdraft.league import DRAFT_SLOT, N_TEAMS, ROSTER_SIZE
    from ffdraft.models.defense import dst_distribution
    from ffdraft.models.distribution import build_player_pool
    from ffdraft.models.opponent import build_training_set, fit_opponent_model
    from ffdraft.models.replacement import replacement_by_position

    pool = build_player_pool()
    replacement = replacement_by_position()
    replacement = {**replacement, "DST": dst_distribution()}
    training, _ = build_training_set()
    model = fit_opponent_model(training)
    return pool, replacement, model, N_TEAMS, DRAFT_SLOT, ROSTER_SIZE


def test_real_rollout_from_pick_one_timing_and_shape(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds = real_fixtures
    state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
    rng = np.random.default_rng(100)

    start = time.perf_counter()
    final = run_rollout(state, pool, model, replacement, rng, our_team=draft_slot)
    elapsed = time.perf_counter() - start

    assert final.is_complete
    print(f"\nfull rollout from pick 1: {elapsed:.3f}s")
    # Generous ceiling: catches an accidental O(n^2)/refit-per-pick
    # regression, not meant to pin an exact number.
    assert elapsed < 15.0


def test_real_rollout_from_late_draft_state_timing(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds = real_fixtures
    # Build a partial state covering all but the last round by running one
    # rollout and truncating its picks.
    seed_state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
    seeded = run_rollout(seed_state, pool, model, replacement, np.random.default_rng(101), our_team=draft_slot)
    n_before_last_round = n_teams * (rounds - 1)
    late_state = DraftState.from_picks(
        seeded.picks[:n_before_last_round], n_teams=n_teams, rounds=rounds
    )

    start = time.perf_counter()
    final = run_rollout(late_state, pool, model, replacement, np.random.default_rng(102), our_team=draft_slot)
    elapsed = time.perf_counter() - start

    assert final.is_complete
    print(f"late-draft-state rollout (last round only): {elapsed:.3f}s")
    assert elapsed < 5.0


def test_real_rollout_first_two_rounds_are_reported(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds = real_fixtures
    state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
    rng = np.random.default_rng(200)
    final = run_rollout(state, pool, model, replacement, rng, our_team=draft_slot)

    id_to_name = {pid: pd.name for pid, pd in pool.items()}
    first_two_rounds = [p for p in final.picks if p.overall_pick <= 2 * n_teams]
    print("\nFirst two rounds of a sampled rollout:")
    for p in first_two_rounds:
        name = id_to_name.get(p.player_id, p.player_id)
        print(f"  pick {p.overall_pick:>3} (team {p.team:>2}): {name} ({p.position})")

    assert len(first_two_rounds) == 2 * n_teams
    # Loose plausibility guard: round 1 should be dominated by RB/WR (this
    # league's top-of-draft consensus), not a flood of K/DST/QB.
    round1 = [p for p in final.picks if p.overall_pick <= n_teams]
    round1_positions = [p.position for p in round1]
    assert round1_positions.count("K") == 0
    assert round1_positions.count("DST") == 0
