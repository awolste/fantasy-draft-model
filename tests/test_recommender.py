"""Tests for `draft.recommender`.

Fast tests use a small, fully synthetic league (few teams, few rounds, a
hand-built pool, tiny (n_rollouts, n_sims_per_rollout)) so the suite stays
quick -- same style as `tests/test_rollout.py`/`tests/test_value.py`. A
final group runs against the real 2026 pool/replacement/opponent-model data
to produce the pick-8/pick-13 report the task asks for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest

from ffdraft.draft.recommender import (
    CandidateRecommendation,
    RecommendationResult,
    prune_candidates,
    recommend_pick,
)
from ffdraft.draft.rollout import DraftState, Pick, team_for_pick
from ffdraft.models.distribution import PlayerDistribution
from ffdraft.models.opponent import DEFAULT_ROSTER_DECAY, DEFAULT_TEMPERATURE, OpponentModel
from ffdraft.sim.lineup import RosterPlayer


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


def _synthetic_pool(n: int = 60) -> dict[str, PlayerDistribution]:
    positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    pool = {}
    for i in range(n):
        pos = positions[i % len(positions)]
        pool[f"p{i}"] = _pd(f"p{i}", pos, rank=i + 1, value=30.0 - i * 0.2)
    return pool


def _pool_with_dominant_player(n: int = 60) -> dict[str, PlayerDistribution]:
    pool = _synthetic_pool(n)
    # Make one RB wildly, unambiguously better than everything else.
    pool["dominant"] = _pd("dominant", "RB", rank=1, value=100.0)
    return pool


# ---------------------------------------------------------------------------
# prune_candidates: determinism + top-value ordering


def test_prune_candidates_is_deterministic_given_a_state():
    pool = _synthetic_pool(60)
    available = list(pool.keys())
    replacement_means = {"QB": 18.3, "RB": 9.6, "WR": 9.9, "TE": 8.1, "K": 7.9, "DST": 7.5}
    a = prune_candidates(available, pool, [], replacement_means, n_candidates=15)
    b = prune_candidates(available, pool, [], replacement_means, n_candidates=15)
    assert [c.player_id for c in a] == [c.player_id for c in b]


def test_prune_candidates_returns_at_most_n_candidates():
    pool = _synthetic_pool(60)
    available = list(pool.keys())
    replacement_means = {"QB": 18.3, "RB": 9.6, "WR": 9.9, "TE": 8.1, "K": 7.9, "DST": 7.5}
    result = prune_candidates(available, pool, [], replacement_means, n_candidates=15)
    assert len(result) <= 15


# ---------------------------------------------------------------------------
# recommend_pick: dominant player, roster need, reproducibility, ranges


def test_recommend_pick_ranks_a_clearly_dominant_player_first():
    pool = _pool_with_dominant_player(60)
    # An empty draft's next pick is always team 1 (round 1, straight
    # order) -- see `team_for_pick`.
    state = DraftState.from_picks([], n_teams=10, rounds=4)
    result = recommend_pick(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=1, our_team=1,
        n_candidates=15, n_rollouts=3, n_sims_per_rollout=50,
    )
    assert result.candidates[0].player_id == "dominant"


def test_recommend_pick_respects_roster_need_for_a_deep_position():
    # Our team already has 2 competitive QBs rostered (one already the
    # starter); a 3rd QB that is not a clear upgrade over the incumbent
    # starter only ever offers depth/FLEX-capacity-discounted bench value
    # (see draft.value's `_bench_value` -- QB has no FLEX recourse, so this
    # decays fast), while every other available player still has an empty
    # starting slot to fill -- so the 3rd QB should not top the list.
    # The gap is made large and deliberate so this holds regardless of
    # rollout noise: with such a low greedy value, "qb_star" should not
    # even survive candidate pruning (a rollout-randomness-free step),
    # let alone rank first.
    #
    # Team 1 gets a genuine back-to-back "turn" pair at picks 1 and 20 (the
    # round1/round2 boundary team, see `team_for_pick`) -- pick 1 is round
    # 1's first pick, and reversed round 2 puts team 1 last, at pick 20 --
    # so its next pick (21) is naturally on the clock right after both of
    # its QBs are already rostered.
    pool = _synthetic_pool(80)
    pool["qb_star"] = _pd("qb_star", "QB", rank=1, value=23.0)  # not an upgrade over q1

    picks = []
    overall = 1
    for team in range(1, 11):
        if team == 1:
            picks.append(Pick(overall_pick=overall, team=1, player_id="q1", position="QB"))
        else:
            picks.append(Pick(overall_pick=overall, team=team, player_id=f"filler_{team}_a", position="RB"))
        overall += 1
    # Round 2 (reversed order): team 1 picks last, at overall 20.
    for team in range(10, 0, -1):
        if team == 1:
            picks.append(Pick(overall_pick=overall, team=1, player_id="q2", position="QB"))
        else:
            picks.append(Pick(overall_pick=overall, team=team, player_id=f"filler_{team}_b", position="WR"))
        overall += 1

    pool["q1"] = _pd("q1", "QB", rank=2, value=24.0)
    pool["q2"] = _pd("q2", "QB", rank=3, value=22.0)
    for team in range(1, 11):
        pool[f"filler_{team}_a"] = _pd(f"filler_{team}_a", "RB", rank=100 + team, value=16.0)
        pool[f"filler_{team}_b"] = _pd(f"filler_{team}_b", "WR", rank=200 + team, value=14.0)

    state = DraftState.from_picks(picks, n_teams=10, rounds=6)
    assert state.next_overall_pick == 21
    assert state.team_on_clock == 1

    replacement_means = {"QB": 18.3, "RB": 9.6, "WR": 9.9, "TE": 8.1, "K": 7.9, "DST": 7.5}
    available = [pid for pid in pool if pid not in state.drafted_ids]
    our_roster = [
        RosterPlayer(player_id="q1", position="QB", score=24.0),
        RosterPlayer(player_id="q2", position="QB", score=22.0),
    ]
    candidates = prune_candidates(available, pool, our_roster, replacement_means, n_candidates=15)
    # Deterministic, rollout-noise-free check: qb_star's bench-only value is
    # far below every startable filler, so it should not even survive
    # pruning to become a candidate at all.
    assert "qb_star" not in [c.player_id for c in candidates]

    result = recommend_pick(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=2, our_team=1,
        n_candidates=15, n_rollouts=3, n_sims_per_rollout=50,
    )
    assert result.candidates[0].player_id != "qb_star"


def test_recommend_pick_is_reproducible_from_a_seed():
    pool = _pool_with_dominant_player(60)
    state = DraftState.from_picks([], n_teams=10, rounds=4)
    a = recommend_pick(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=7, our_team=1,
        n_candidates=10, n_rollouts=3, n_sims_per_rollout=50,
    )
    b = recommend_pick(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=7, our_team=1,
        n_candidates=10, n_rollouts=3, n_sims_per_rollout=50,
    )
    assert [c.player_id for c in a.candidates] == [c.player_id for c in b.candidates]
    assert [c.championship_probability for c in a.candidates] == [
        c.championship_probability for c in b.candidates
    ]


def test_recommend_pick_probabilities_are_in_unit_interval_and_not_all_identical():
    pool = _synthetic_pool(80)
    state = DraftState.from_picks([], n_teams=10, rounds=4)
    result = recommend_pick(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=3, our_team=1,
        n_candidates=15, n_rollouts=4, n_sims_per_rollout=80,
    )
    probs = [c.championship_probability for c in result.candidates]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert len(set(probs)) > 1


def test_recommend_pick_raises_when_not_our_turn():
    pool = _synthetic_pool(60)
    picks = [Pick(overall_pick=1, team=1, player_id="p0", position="RB")]
    state = DraftState.from_picks(picks, n_teams=10, rounds=4)
    with pytest.raises(ValueError):
        recommend_pick(
            state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=1, our_team=8,
            n_rollouts=2, n_sims_per_rollout=20,
        )


# ---------------------------------------------------------------------------
# Late-draft state: few picks remaining


def test_recommend_pick_produces_sensible_recommendation_late_in_draft():
    pool = _synthetic_pool(60)
    n_teams, rounds = 10, 4
    # Fill every roster to one slot short of full (rounds=4, so 3 rounds --
    # 30 picks -- already made) using the first 30 pool players in snake
    # order, and confirm a recommendation still comes back sane.
    player_ids = list(pool.keys())
    picks = [
        Pick(
            overall_pick=overall,
            team=team_for_pick(overall, n_teams),
            player_id=player_ids[overall - 1],
            position=pool[player_ids[overall - 1]].position,
        )
        for overall in range(1, n_teams * (rounds - 1) + 1)
    ]

    state = DraftState.from_picks(picks, n_teams=n_teams, rounds=rounds)
    assert not state.is_complete
    result = recommend_pick(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=5, our_team=state.team_on_clock,
        n_candidates=10, n_rollouts=3, n_sims_per_rollout=50,
    )
    assert len(result.candidates) > 0
    assert all(0.0 <= c.championship_probability <= 1.0 for c in result.candidates)


# ---------------------------------------------------------------------------
# Real 2026 data: pick 8, pick 13, QB placement, timing


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


def test_real_recommendation_at_pick_8_from_empty_draft(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures

    # "From an empty draft" means no picks have happened yet when we ask
    # the question -- but pick 8 (draft_slot's first real turn) still
    # needs picks 1-7 to exist for it to be team 8's turn at all. Those 7
    # opponent picks are generated by the same opponent model the rollout
    # itself would use (one real rollout, truncated), not hand-picked, so
    # this is "the honest pick 8" rather than an artificial setup.
    from ffdraft.draft.rollout import run_rollout

    seed_state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
    seeded = run_rollout(
        seed_state, pool, model, replacement, np.random.default_rng(999), our_team=draft_slot,
        adp_table=adp_table, rankings=rankings,
    )
    state = DraftState.from_picks(seeded.picks[:7], n_teams=n_teams, rounds=rounds)
    assert state.next_overall_pick == 8
    assert state.team_on_clock == draft_slot

    start = time.perf_counter()
    result = recommend_pick(
        state, pool, model, replacement, seed=42, our_team=draft_slot,
        adp_table=adp_table, rankings=rankings,
    )
    elapsed = time.perf_counter() - start

    print(f"\nPick 8 (empty draft) recommendation -- {elapsed:.2f}s, "
          f"n_rollouts={result.n_rollouts}, n_sims_per_rollout={result.n_sims_per_rollout}")
    for c in result.candidates:
        flag = " (indistinguishable from leader)" if c.indistinguishable_from_leader else ""
        print(
            f"  {c.name:<22} {c.position:<4} value={c.greedy_value:6.2f} "
            f"p_champ={c.championship_probability * 100:5.2f}% "
            f"se={c.standard_error * 100:4.2f}pp{flag}"
        )

    assert len(result.candidates) > 0
    assert 0.0 <= result.candidates[0].championship_probability <= 1.0


def test_real_pruning_to_top_15_does_not_discard_the_eventual_winner(real_fixtures):
    # Sanity check the plan doc explicitly asks for: pruning to ~15
    # candidates by greedy value should not discard a player who would
    # plausibly have won. Widen the candidate set to 30 (cheap rollouts/
    # sims here, since this test only cares about *which* candidate wins,
    # not a precise probability) and confirm the winner is still within
    # what a 15-candidate cutoff would have offered.
    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures
    state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)

    result = recommend_pick(
        state, pool, model, replacement, seed=11, our_team=1,
        n_candidates=30, n_rollouts=5, n_sims_per_rollout=200,
        adp_table=adp_table, rankings=rankings,
    )
    by_greedy_value = sorted(result.candidates, key=lambda c: -c.greedy_value)
    top15_by_greedy_value = {c.player_id for c in by_greedy_value[:15]}
    winner = max(result.candidates, key=lambda c: c.championship_probability)
    print(
        f"\n30-candidate winner: {winner.name} ({winner.position}), "
        f"greedy value rank among the 30 evaluated: "
        f"{[c.player_id for c in by_greedy_value].index(winner.player_id) + 1}"
    )
    assert winner.player_id in top15_by_greedy_value


def test_real_recommendation_at_pick_13_shows_roster_awareness(real_fixtures):
    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures

    # Build a plausible first 12 picks by running one real rollout and
    # truncating it -- same technique as
    # test_rollout.py::test_real_rollout_from_late_draft_state_timing.
    # This guarantees pick 8's player is whatever the engine's own greedy
    # policy actually chose at an empty draft, and every other pick is a
    # real (not hand-assembled) opponent draw.
    from ffdraft.draft.rollout import run_rollout

    seed_state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)
    seeded = run_rollout(
        seed_state, pool, model, replacement, np.random.default_rng(900), our_team=draft_slot,
        adp_table=adp_table, rankings=rankings,
    )
    state = DraftState.from_picks(seeded.picks[:12], n_teams=n_teams, rounds=rounds)
    assert state.next_overall_pick == 13
    assert state.team_on_clock == draft_slot

    pick8_id = next(p.player_id for p in state.picks if p.overall_pick == 8)
    pick8_name = pool[pick8_id].name if pick8_id in pool else pick8_id

    result = recommend_pick(
        state, pool, model, replacement, seed=43, our_team=draft_slot,
        adp_table=adp_table, rankings=rankings,
    )
    print(f"\nPick 13 recommendation (after {pick8_name} at pick 8):")
    for c in result.candidates:
        print(
            f"  {c.name:<22} {c.position:<4} value={c.greedy_value:6.2f} "
            f"p_champ={c.championship_probability * 100:5.2f}% se={c.standard_error * 100:4.2f}pp"
        )
    assert len(result.candidates) > 0


# ---------------------------------------------------------------------------
# Staleness guard: catch a future policy/model change silently invalidating
# MEASURED_BETWEEN_ROLLOUT_SD_PP / MEASURED_WITHIN_ROLLOUT_SD_AT_5000_PP
# (and therefore DEFAULT_N_ROLLOUTS/DEFAULT_N_SIMS_PER_ROLLOUT) the way the
# pre-fix greedy policy silently invalidated the original 2.00pp figure --
# see recommender.py's module docstring, "Measured variance decomposition".
#
# This is deliberately cheap (small n_rollouts/n_sims_per_rollout, one real
# pick), not a precise re-measurement -- it cannot pin down the constant to
# the precision the full re-measurement did, only catch an order-of-
# magnitude drift (the ~2.00pp -> ~3.5-6pp kind that actually happened
# here). A tight, expensive re-measurement belongs in a task report, not in
# every CI run.


def test_variance_decomposition_has_not_drifted(real_fixtures):
    from ffdraft.draft.recommender import (
        MEASURED_BETWEEN_ROLLOUT_SD_PP,
        MEASURED_WITHIN_ROLLOUT_SD_AT_5000_PP,
        measure_variance_decomposition,
    )

    pool, replacement, model, n_teams, draft_slot, rounds, adp_table, rankings = real_fixtures
    state = DraftState.from_picks([], n_teams=n_teams, rounds=rounds)

    decomp = measure_variance_decomposition(
        state, pool, model, replacement, seed=20260803, our_team=1,
        n_rollouts=8, n_sims_per_rollout=300,
        adp_table=adp_table, rankings=rankings,
    )
    print(
        f"\nStaleness guard (n_rollouts=8, n_sims_per_rollout=300): "
        f"between-rollout SD={decomp.between_rollout_sd_pp:.2f}pp "
        f"(recorded {MEASURED_BETWEEN_ROLLOUT_SD_PP:.2f}pp), "
        f"within-rollout SD={decomp.within_rollout_sd_pp:.2f}pp "
        f"(recorded {MEASURED_WITHIN_ROLLOUT_SD_AT_5000_PP:.2f}pp at S=5,000)"
    )

    # Between-rollout SD: a wide multiplicative band (0.4x-2.5x the
    # recorded figure) around MEASURED_BETWEEN_ROLLOUT_SD_PP -- loose
    # enough that n_rollouts=8's own sampling noise (SD-of-SD roughly
    # +-27% of the true value at this n) does not make this flaky on an
    # unchanged policy, but tight enough to catch a real multi-x drift like
    # the one this guard exists because of. If this legitimately fails
    # after an intentional policy/opponent-model change, re-run the full
    # re-measurement (see module docstring) and update
    # MEASURED_BETWEEN_ROLLOUT_SD_PP/MEASURED_WITHIN_ROLLOUT_SD_AT_5000_PP
    # and the DEFAULT_N_ROLLOUTS/DEFAULT_N_SIMS_PER_ROLLOUT allocation
    # derived from them -- do not just widen this band.
    assert 0.4 * MEASURED_BETWEEN_ROLLOUT_SD_PP <= decomp.between_rollout_sd_pp <= 2.5 * MEASURED_BETWEEN_ROLLOUT_SD_PP, (
        f"between-rollout SD {decomp.between_rollout_sd_pp:.2f}pp has drifted materially from the "
        f"recorded {MEASURED_BETWEEN_ROLLOUT_SD_PP:.2f}pp -- the DEFAULT_N_ROLLOUTS/"
        "DEFAULT_N_SIMS_PER_ROLLOUT allocation in recommender.py needs re-deriving, not this test loosened."
    )


# ---------------------------------------------------------------------------
# restrict_to_positions
#
# Added for the precomputed pick tree (`scripts/build_pick_tree.py`), which
# asks "what is the best RB or WR here". It cannot get that by filtering the
# default output: the value function's positional tilt (HANDOFF item 3) can
# fill the whole pruned list with QBs and tight ends, and a candidate that
# was never pruned in was never simulated.


def test_restrict_to_positions_returns_only_those_positions():
    pool = _pool_with_dominant_player(60)
    state = DraftState.from_picks([], n_teams=10, rounds=4)
    result = recommend_pick(
        state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=1, our_team=1,
        n_candidates=15, n_rollouts=3, n_sims_per_rollout=50,
        restrict_to_positions=frozenset({"WR"}),
    )
    assert result.candidates, "restriction must not silently empty the list"
    assert {c.position for c in result.candidates} == {"WR"}


def test_restriction_surfaces_a_candidate_the_default_pruning_drops():
    """The reason this parameter exists. `dominant` is an RB so good it wins
    every default candidate list; restricting to WR must still return real,
    simulated WR candidates rather than nothing."""
    pool = _pool_with_dominant_player(60)
    state = DraftState.from_picks([], n_teams=10, rounds=4)
    kw = dict(
        state=state, pool=pool, model=FLAT_MODEL,
        replacement_by_position=REPLACEMENT_BY_POSITION, seed=1, our_team=1,
        n_candidates=5, n_rollouts=3, n_sims_per_rollout=50,
    )
    default = recommend_pick(**kw)
    restricted = recommend_pick(**kw, restrict_to_positions=frozenset({"WR"}))

    assert default.candidates[0].player_id == "dominant"
    assert all(c.position == "WR" for c in restricted.candidates)
    assert all(c.championship_probability is not None for c in restricted.candidates)


def test_restricting_to_a_position_with_nobody_left_raises():
    """An empty candidate list would render as a finished recommendation
    with no picks -- loud failure instead."""
    pool = _synthetic_pool(60)
    state = DraftState.from_picks([], n_teams=10, rounds=4)
    with pytest.raises(ValueError, match="restrict_to_positions"):
        recommend_pick(
            state, pool, FLAT_MODEL, REPLACEMENT_BY_POSITION, seed=1, our_team=1,
            n_candidates=15, n_rollouts=3, n_sims_per_rollout=50,
            restrict_to_positions=frozenset({"LB"}),
        )


def test_unrestricted_behaviour_is_unchanged():
    """The parameter must be inert when omitted -- everything already
    measured with this function has to stay reproducible."""
    pool = _pool_with_dominant_player(60)
    state = DraftState.from_picks([], n_teams=10, rounds=4)
    kw = dict(
        state=state, pool=pool, model=FLAT_MODEL,
        replacement_by_position=REPLACEMENT_BY_POSITION, seed=3, our_team=1,
        n_candidates=8, n_rollouts=3, n_sims_per_rollout=50,
    )
    a = recommend_pick(**kw)
    b = recommend_pick(**kw, restrict_to_positions=None)
    assert [c.player_id for c in a.candidates] == [c.player_id for c in b.candidates]
    assert [c.championship_probability for c in a.candidates] == [
        c.championship_probability for c in b.candidates
    ]
