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
import polars as pl
import pytest

from ffdraft.draft.rollout import (
    DraftState,
    Pick,
    pool_adp_lookup,
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
# Real ADP as the opponent model's signal, injectable so Task 6's backtest
# can supply a historical season's ADP instead of 2026's (see module
# docstring, "ADP for the opponent model").


def _pd_named(player_id: str, name: str, position: str, rank: int, value: float) -> PlayerDistribution:
    # `normalize_name` (used by the ADP join) strips digits, so a name like
    # "p1" collides with "p2" once normalized -- unlike `_pd`, this gives
    # each synthetic player a real, digit-free name distinct from its id.
    return PlayerDistribution(
        player_id=player_id, name=name, position=position, rank=rank, tier=1,
        distribution=ConstantDist(value),
    )


def _synthetic_adp_pool():
    return {
        "p1": _pd_named("p1", "Player One", "WR", rank=1, value=20.0),
        "p2": _pd_named("p2", "Player Two", "RB", rank=2, value=18.0),
        "p3": _pd_named("p3", "Player Three", "QB", rank=3, value=25.0),  # will not match adp_table
        "p4": _pd_named("p4", "Player Four", "WR", rank=4, value=15.0),  # will not match adp_table
        # DST's `name` (matched against `rankings`) is the full team name,
        # distinct from its `player_id` -- same convention as
        # `distribution.build_player_pool`.
        "DST::Houston Texans": PlayerDistribution(
            player_id="DST::Houston Texans", name="Houston Texans", position="DST",
            rank=5, tier=None, distribution=ConstantDist(8.0),
        ),
    }


def _synthetic_adp_table():
    return pl.DataFrame(
        {
            "name": ["Player One", "Player Two", "Houston Defense"],
            "position": ["WR", "RB", "DST"],
            "team": [None, None, "HOU"],
            "adp": [1.5, 3.5, 90.0],
        }
    )


def _synthetic_rankings():
    return pl.DataFrame(
        {
            "name": ["Player One", "Player Two", "Player Three", "Player Four", "Houston Texans"],
            "position": ["WR", "RB", "QB", "WR", "DST"],
            "team": [None, None, None, None, "HOU"],
        }
    )


def test_pool_adp_lookup_uses_real_adp_for_matched_players():
    pool = _synthetic_adp_pool()
    adp_by_id, report = pool_adp_lookup(pool, _synthetic_adp_table(), _synthetic_rankings())
    assert adp_by_id["p1"] == 1.5
    assert adp_by_id["p2"] == 3.5
    assert adp_by_id["DST::Houston Texans"] == 90.0
    assert report.matched == 3
    assert set(report.unmatched_ids) == {"p3", "p4"}


def test_pool_adp_lookup_extrapolates_unmatched_players_continuously():
    # A dedicated, DST-free fixture here: matched pairs are close to linear
    # (rank 1 -> adp 1.5, rank 2 -> adp 3.5, rank 5 -> adp 9.5), so the
    # extrapolated values for unmatched ranks 3 and 4 have a predictable,
    # tight expected range -- isolates the "is it continuous, not a flat
    # sentinel" question from any one outlier's influence on the fit
    # (the DST/skill scale difference is exercised separately in
    # `test_pool_adp_lookup_uses_real_adp_for_matched_players`).
    pool = {
        "p1": _pd_named("p1", "Player One", "WR", rank=1, value=20.0),
        "p2": _pd_named("p2", "Player Two", "RB", rank=2, value=18.0),
        "p3": _pd_named("p3", "Player Three", "QB", rank=3, value=25.0),  # unmatched
        "p4": _pd_named("p4", "Player Four", "WR", rank=4, value=15.0),  # unmatched
        "p5": _pd_named("p5", "Player Five", "TE", rank=5, value=12.0),
    }
    adp_table = pl.DataFrame(
        {
            "name": ["Player One", "Player Two", "Player Five"],
            "position": ["WR", "RB", "TE"],
            "team": [None, None, None],
            "adp": [1.5, 3.5, 9.5],
        }
    )
    rankings = pl.DataFrame(
        {"name": [p.name for p in pool.values()], "position": [p.position for p in pool.values()], "team": [None] * 5}
    )
    adp_by_id, report = pool_adp_lookup(pool, adp_table, rankings)

    assert set(report.unmatched_ids) == {"p3", "p4"}
    # Continuity: p3/p4's extrapolated adp should sit between their
    # matched neighbors (rank 2 -> 3.5, rank 5 -> 9.5), not a discontinuous
    # jump to some arbitrary large/small sentinel.
    assert 3.5 < adp_by_id["p3"] < adp_by_id["p4"] < 9.5


def test_pool_adp_lookup_falls_back_to_rank_when_nothing_matches():
    pool = {"p1": _pd("p1", "WR", rank=1, value=20.0), "p2": _pd("p2", "RB", rank=2, value=18.0)}
    empty_adp = pl.DataFrame({"name": [], "position": [], "team": [], "adp": []}, schema={
        "name": pl.String, "position": pl.String, "team": pl.String, "adp": pl.Float64,
    })
    rankings = pl.DataFrame({"name": ["p1", "p2"], "position": ["WR", "RB"], "team": [None, None]})
    adp_by_id, report = pool_adp_lookup(pool, empty_adp, rankings)
    assert adp_by_id["p1"] == 1.0
    assert adp_by_id["p2"] == 2.0
    assert report.matched == 0


def test_run_rollout_accepts_injectable_real_adp():
    pool = _big_synthetic_pool(200)
    # A real-ADP-shaped table that matches none of the synthetic pool names
    # -- exercises the "adp_table supplied but everything falls through to
    # extrapolation" path end to end inside a full rollout.
    adp_table = pl.DataFrame({"name": [], "position": [], "team": [], "adp": []}, schema={
        "name": pl.String, "position": pl.String, "team": pl.String, "adp": pl.Float64,
    })
    rankings = pl.DataFrame(
        {"name": list(pool.keys()), "position": [p.position for p in pool.values()], "team": [None] * len(pool)}
    )
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    rng = np.random.default_rng(8)
    final = run_rollout(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, rng, our_team=1,
        adp_table=adp_table, rankings=rankings,
    )
    assert final.is_complete


def test_run_rollout_adp_table_actually_changes_opponent_behavior():
    # Two players, otherwise identical rank -- real ADP that strongly
    # favors one of them should make opponents draft it first far more
    # often than the rank-proxy default (which, with equal rank, would
    # treat them near-identically).
    pool = {
        "favored": _pd("favored", "WR", rank=10, value=15.0),
        "disfavored": _pd("disfavored", "WR", rank=10, value=15.0),
    }
    for i in range(20):
        pool[f"filler{i}"] = _pd(f"filler{i}", "RB", rank=20 + i, value=10.0)

    adp_table = pl.DataFrame(
        {
            "name": ["favored", "disfavored"],
            "position": ["WR", "WR"],
            "team": [None, None],
            "adp": [1.0, 500.0],
        }
    )
    rankings = pl.DataFrame(
        {"name": list(pool.keys()), "position": [p.position for p in pool.values()], "team": [None] * len(pool)}
    )

    state = DraftState.from_picks([], n_teams=2, rounds=1)
    first_picks = []
    for seed in range(20):
        final = run_rollout(
            state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(seed), our_team=2,
            adp_table=adp_table, rankings=rankings,
        )
        first_picks.append(final.picks[0].player_id)
    assert first_picks.count("favored") > first_picks.count("disfavored")


def test_run_rollout_raises_if_adp_table_given_without_rankings():
    pool = _big_synthetic_pool(20)
    state = DraftState.from_picks([], n_teams=2, rounds=2)
    adp_table = pl.DataFrame({"name": [], "position": [], "team": [], "adp": []}, schema={
        "name": pl.String, "position": pl.String, "team": pl.String, "adp": pl.Float64,
    })
    with pytest.raises(ValueError):
        run_rollout(
            state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(9), our_team=1,
            adp_table=adp_table,
        )


def test_run_rollout_supports_historical_season_adp_source():
    # Simulates Task 6's use case: a caller supplies a specific historical
    # season's ADP (a slice of `adp_history`, shaped identically to
    # `adp_2026`) instead of the live 2026 table -- proving the ADP source
    # is a swappable parameter, not hardcoded to 2026.
    pool = _big_synthetic_pool(60)
    historical_adp = pl.DataFrame(
        {
            "name": [f"p{i}" for i in range(0, 60, 6)],
            "position": [pool[f"p{i}"].position for i in range(0, 60, 6)],
            "team": [None] * 10,
            "adp": [float(i + 1) for i in range(10)],
        }
    )
    rankings = pl.DataFrame(
        {"name": list(pool.keys()), "position": [p.position for p in pool.values()], "team": [None] * len(pool)}
    )
    state = DraftState.from_picks([], n_teams=4, rounds=6)
    final = run_rollout(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, np.random.default_rng(10), our_team=1,
        adp_table=historical_adp, rankings=rankings,
    )
    assert final.is_complete


# ---------------------------------------------------------------------------
# Real 2026 pool: timing and plausibility (see task report for narrative)


@pytest.fixture(scope="module")
def real_fixtures():
    from ffdraft import store
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
    adp_table = store.read("adp_2026")
    rankings = store.read("rankings_2026")
    return pool, replacement, model, N_TEAMS, DRAFT_SLOT, ROSTER_SIZE, adp_table, rankings


def test_real_rollout_from_pick_one_timing_and_shape(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures
    state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
    rng = np.random.default_rng(100)

    start = time.perf_counter()
    final = run_rollout(
        state, pool, model, replacement, rng, our_team=draft_slot,
        adp_table=adp_table, rankings=rankings,
    )
    elapsed = time.perf_counter() - start

    assert final.is_complete
    print(f"\nfull rollout from pick 1 (real ADP): {elapsed:.3f}s")
    # Generous ceiling: catches an accidental O(n^2)/refit-per-pick
    # regression, not meant to pin an exact number. In particular this
    # confirms the ADP match + extrapolation fit (run once, before the
    # per-pick loop) doesn't show up as an O(n) or O(n^2) cost per pick.
    assert elapsed < 15.0


def test_real_rollout_from_late_draft_state_timing(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures
    # Build a partial state covering all but the last round by running one
    # rollout and truncating its picks.
    seed_state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
    seeded = run_rollout(
        seed_state, pool, model, replacement, np.random.default_rng(101), our_team=draft_slot,
        adp_table=adp_table, rankings=rankings,
    )
    n_before_last_round = n_teams * (rounds - 1)
    late_state = DraftState.from_picks(
        seeded.picks[:n_before_last_round], n_teams=n_teams, rounds=rounds
    )

    start = time.perf_counter()
    final = run_rollout(
        late_state, pool, model, replacement, np.random.default_rng(102), our_team=draft_slot,
        adp_table=adp_table, rankings=rankings,
    )
    elapsed = time.perf_counter() - start

    assert final.is_complete
    print(f"late-draft-state rollout (last round only, real ADP): {elapsed:.3f}s")
    assert elapsed < 5.0


def test_real_rollout_adp_coverage_is_reported(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures
    adp_by_id, report = pool_adp_lookup(pool, adp_table, rankings)

    print(
        f"\nADP coverage: {report.matched}/{report.total} pool players matched real ADP "
        f"({report.unmatched} extrapolated)."
    )
    # Every pool player gets a usable, positive adp either way.
    assert len(adp_by_id) == report.total
    assert all(v > 0 for v in adp_by_id.values())
    # Real ADP (`data/adp_2026.parquet`) covers ~246 of ~510 pool players --
    # the rest (mostly deep bench, rounds 15+) are extrapolated, not
    # matched. Loose bounds so this doesn't pin an exact count.
    assert 150 < report.matched < 350


def test_real_rollout_first_two_rounds_are_reported(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures

    def _first_two_rounds(use_real_adp: bool, seed: int):
        kwargs = {"adp_table": adp_table, "rankings": rankings} if use_real_adp else {}
        rng = np.random.default_rng(seed)
        state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
        final = run_rollout(state, pool, model, replacement, rng, our_team=draft_slot, **kwargs)
        return [p for p in final.picks if p.overall_pick <= 2 * n_teams]

    id_to_name = {pid: pd.name for pid, pd in pool.items()}

    def _print_rounds(label, picks):
        print(f"\n{label}:")
        for p in picks:
            name = id_to_name.get(p.player_id, p.player_id)
            print(f"  pick {p.overall_pick:>3} (team {p.team:>2}): {name} ({p.position})")

    ecr_rounds = _first_two_rounds(use_real_adp=False, seed=200)
    real_adp_rounds = _first_two_rounds(use_real_adp=True, seed=200)
    _print_rounds("First two rounds under the ECR-rank proxy", ecr_rounds)
    _print_rounds("First two rounds under real 2026 ADP", real_adp_rounds)

    assert len(real_adp_rounds) == 2 * n_teams
    # Loose plausibility guard: round 1 should be dominated by RB/WR (this
    # league's top-of-draft consensus), not a flood of K/DST/QB.
    round1_positions = [p.position for p in real_adp_rounds if p.overall_pick <= n_teams]
    assert round1_positions.count("K") == 0
    assert round1_positions.count("DST") == 0
