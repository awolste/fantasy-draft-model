"""Tests for `ffdraft.live.survival`.

Survival probability answers the question `recommend_pick` structurally
cannot: *will he still be there at my next pick?* See HANDOFF section 7b --
the recommender scores "take him now", and its rollouts use the greedy
policy for our own future picks, so the "skip him now, take him later"
counterfactual is systematically undervalued. That gap is what makes the
tool reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.survival import picks_until_our_next_turn, wait_cost_pp


class _FakeModel:
    """Stands in for OpponentModel. The sampler itself is monkeypatched in
    these tests, but `survival_probabilities` now builds an `OpponentBoard`
    first, and that reads `predicted_reach` -- flat here, so the board's
    ordering is ADP order and the fake sampler's intent is unchanged."""

    def predicted_reach(self, manager_id: str, position: str) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# How many picks we have to survive


def test_picks_until_next_turn_at_the_turn_is_short():
    """Slot 8 picks at 8 and 13 -- only 4 opponent picks in between, so
    almost anything survives the turn."""
    assert picks_until_our_next_turn(8, n_teams=N_TEAMS, our_team=DRAFT_SLOT) == 4


def test_picks_until_next_turn_across_the_wrap_is_long():
    """From 13 the next of ours is 28: 14 intervening picks, so survival is
    much less likely. The asymmetry is exactly why a single 'will he last?'
    intuition fails at slot 8."""
    assert picks_until_our_next_turn(13, n_teams=N_TEAMS, our_team=DRAFT_SLOT) == 14


def test_picks_until_next_turn_is_none_at_our_final_pick():
    """No next turn means waiting is not an option, and survival is
    undefined rather than 0 or 1."""
    assert picks_until_our_next_turn(173, n_teams=N_TEAMS, our_team=DRAFT_SLOT) is None


def test_picks_until_next_turn_raises_if_it_is_not_our_pick():
    with pytest.raises(ValueError, match="not ours"):
        picks_until_our_next_turn(9, n_teams=N_TEAMS, our_team=DRAFT_SLOT)


# ---------------------------------------------------------------------------
# Cost of waiting


def test_wait_cost_is_zero_when_survival_is_certain():
    """If he is certain to be there, passing costs nothing -- regardless of
    how good he is."""
    assert wait_cost_pp(champ=0.25, p_survive=1.0, champ_best_survivor=0.10) == 0.0


def test_wait_cost_is_the_full_gap_when_he_will_certainly_be_gone():
    """Certain to vanish: the cost is the entire gap to the fallback, in
    percentage points."""
    assert wait_cost_pp(champ=0.25, p_survive=0.0, champ_best_survivor=0.20) == pytest.approx(5.0)


def test_wait_cost_scales_with_the_chance_of_losing_him():
    assert wait_cost_pp(champ=0.20, p_survive=0.5, champ_best_survivor=0.10) == pytest.approx(5.0)


def test_wait_cost_is_never_negative():
    """A fallback that scores higher than the candidate means waiting is
    free, not beneficial -- clamping keeps the column readable as a cost."""
    assert wait_cost_pp(champ=0.10, p_survive=0.0, champ_best_survivor=0.15) == 0.0


def test_wait_cost_is_undefined_without_a_next_turn():
    """At our last pick there is nothing to wait for."""
    assert wait_cost_pp(champ=0.2, p_survive=None, champ_best_survivor=0.1) is None


# ---------------------------------------------------------------------------
# Survival simulation


def test_survival_probabilities_are_bounded_and_ordered_by_adp(monkeypatch):
    """Earlier-ADP players must not survive more often than later-ADP ones.
    This is the sanity check that the simulation is wired to the opponent
    model at all rather than returning noise."""
    from ffdraft.live import survival as mod

    pool = {f"p{i}": type("P", (), {"position": "RB", "rank": i})() for i in range(1, 41)}
    adp = {f"p{i}": float(i) for i in range(1, 41)}

    # Opponents take the lowest-ADP player available, deterministically.
    # The seam is now `sample_pick_on_board`, which takes a boolean mask
    # over the board and returns a board index rather than a player id.
    def fake_sample_pick(board, alive, roster_counts, rng, **kw):
        live = np.flatnonzero(alive)
        return int(min(live, key=lambda i: adp[board.player_ids[i]]))

    monkeypatch.setattr(mod, "sample_pick_on_board", fake_sample_pick)

    probs = mod.survival_probabilities(
        candidate_ids=["p1", "p20", "p40"],
        pool=pool,
        adp_by_player_id=adp,
        drafted_ids=set(),
        model=_FakeModel(),
        n_picks=5,
        n_sims=20,
        rng=np.random.default_rng(0),
    )
    assert all(0.0 <= v <= 1.0 for v in probs.values())
    # The 5 lowest-ADP players go; p1 cannot survive, p20 and p40 must.
    assert probs["p1"] == 0.0
    assert probs["p20"] == 1.0
    assert probs["p40"] == 1.0


def test_survival_is_one_for_everyone_when_no_picks_intervene(monkeypatch):
    from ffdraft.live import survival as mod

    pool = {"a": type("P", (), {"position": "RB", "rank": 1})()}
    probs = mod.survival_probabilities(
        candidate_ids=["a"], pool=pool, adp_by_player_id={"a": 1.0},
        drafted_ids=set(), model=_FakeModel(), n_picks=0, n_sims=5,
        rng=np.random.default_rng(0),
    )
    assert probs["a"] == 1.0
