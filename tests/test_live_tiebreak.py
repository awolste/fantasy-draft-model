"""Tests for `ffdraft.live.tiebreak`.

Both rules here exist because a full mock draft exposed concrete defects:

* The engine took a **second kicker and a third quarterback** — picks that
  can never start. Open item 3 in HANDOFF: QB=3 and K=2 are cap-bound in
  every rollout.
* Worse, the R9 pick was **Brandon Aubrey, a kicker, at ADP 131.5** — a
  43-pick reach. It was not the recommender's leader; it won a tie-break
  from 8 indistinguishable candidates because the old rule treated "we have
  no kicker" as a roster need. Aubrey was 100% likely to survive, so the
  survival signal said wait and the need rule overrode it.
"""

from __future__ import annotations

from ffdraft.live.tiebreak import (
    describe_tie,
    LATE_FILL_POSITIONS,
    POSITION_CAPS,
    choose_from_tied,
    filter_capped,
)


def _row(name, pos, champ, wait=0.0, surv=1.0, tied=True):
    return {
        "player_id": name,
        "name": name,
        "position": pos,
        "championship_probability": champ,
        "wait_cost_pp": wait,
        "p_survive": surv,
        "indistinguishable_from_leader": tied,
    }


# ---------------------------------------------------------------------------
# Caps


def test_third_quarterback_is_filtered_out():
    rows = [_row("QB3", "QB", 0.20), _row("WR", "WR", 0.19)]
    assert [r["name"] for r in filter_capped(rows, {"QB": 2})] == ["WR"]


def test_second_kicker_is_filtered_out():
    rows = [_row("K2", "K", 0.20), _row("RB", "RB", 0.19)]
    assert [r["name"] for r in filter_capped(rows, {"K": 1})] == ["RB"]


def test_a_first_kicker_and_second_quarterback_are_allowed():
    rows = [_row("K1", "K", 0.2), _row("QB2", "QB", 0.2)]
    assert len(filter_capped(rows, {"K": 0, "QB": 1})) == 2


def test_filtering_returns_empty_when_everything_is_capped():
    """This is the important one. An earlier version fell back to returning
    the original rows so the caller always had something -- which silently
    disabled the cap exactly when it mattered, because by the late rounds
    the recommender's entire candidate list is QBs and kickers. The mock
    draft ended with four quarterbacks as a result."""
    rows = [_row("QB3", "QB", 0.2), _row("QB4", "QB", 0.1)]
    assert filter_capped(rows, {"QB": 5}) == []


def test_best_uncapped_available_looks_past_the_candidate_list():
    from ffdraft.live.tiebreak import best_uncapped_available

    class P:
        def __init__(self, name, position, rank):
            self.name, self.position, self.rank = name, position, rank

    pool = {"qb": P("A QB", "QB", 1), "wr": P("A WR", "WR", 50)}
    got = best_uncapped_available(pool, drafted_ids=set(), counts={"QB": 2})
    assert got["name"] == "A WR"
    assert got["championship_probability"] is None, "unsimulated pick must not fake a number"
    assert got["uncapped_fallback"] is True


def test_best_uncapped_available_returns_none_when_nothing_is_left():
    from ffdraft.live.tiebreak import best_uncapped_available

    assert best_uncapped_available({}, set(), {}) is None


def test_caps_are_the_documented_values():
    assert POSITION_CAPS["QB"] == 2
    assert POSITION_CAPS["K"] == 1


# ---------------------------------------------------------------------------
# Tie-break


def test_tie_break_prefers_the_player_least_likely_to_survive():
    """Of two players worth the same, take the one who will not be there
    next time."""
    rows = [
        _row("stays", "WR", 0.20, wait=0.0, surv=0.95),
        _row("goes", "RB", 0.20, wait=3.1, surv=0.10),
    ]
    assert choose_from_tied(rows, counts={}, rounds_left=10)["name"] == "goes"


def test_kicker_does_not_win_a_tie_on_roster_need_in_early_rounds():
    """The Aubrey regression. Every tied candidate has wait_cost 0, we have
    no kicker, and the old rule ranked 'unfilled starter slot' first -- so a
    kicker with 100% survival beat a skill player. Kickers are always there;
    filling that slot early is a wasted pick."""
    rows = [
        _row("kicker", "K", 0.20, wait=0.0, surv=1.0),
        _row("receiver", "WR", 0.20, wait=0.0, surv=1.0),
    ]
    chosen = choose_from_tied(rows, counts={"WR": 5, "RB": 2}, rounds_left=10)
    assert chosen["name"] == "receiver"


def test_kicker_does_win_on_need_once_the_draft_is_nearly_over():
    """Late, the slot genuinely has to be filled."""
    rows = [
        _row("kicker", "K", 0.20, wait=0.0, surv=1.0),
        _row("receiver", "WR", 0.20, wait=0.0, surv=1.0),
    ]
    chosen = choose_from_tied(rows, counts={"WR": 5, "RB": 2}, rounds_left=1)
    assert chosen["name"] == "kicker"


def test_roster_need_still_breaks_ties_among_skill_positions():
    """Scarcity weighting must not disable roster construction entirely --
    an unfilled RB slot should still pull a tie."""
    rows = [
        _row("rb", "RB", 0.20, wait=0.0, surv=1.0),
        _row("wr", "WR", 0.20, wait=0.0, surv=1.0),
    ]
    chosen = choose_from_tied(rows, counts={"WR": 2, "RB": 0}, rounds_left=10)
    assert chosen["name"] == "rb"


def test_late_fill_positions_are_the_documented_ones():
    assert LATE_FILL_POSITIONS == frozenset({"K", "DST"})


def test_untied_leader_is_returned_unchanged():
    rows = [_row("leader", "RB", 0.30, tied=True), _row("other", "WR", 0.10, tied=False)]
    assert choose_from_tied(rows, counts={}, rounds_left=10)["name"] == "leader"


# ---------------------------------------------------------------------------
# Explaining a tie


def _tied_row(name, position="RB", wait=0.0, survive=1.0, champ=0.2):
    return {
        "player_id": name.lower(),
        "name": name,
        "position": position,
        "championship_probability": champ,
        "indistinguishable_from_leader": True,
        "wait_cost_pp": wait,
        "p_survive": survive,
    }


def test_describe_tie_says_why_rather_than_only_that():
    """The flag is routinely misread as 'the numbers differ, so why equal'.
    The explanation has to name the actual reason."""
    text = describe_tie([_tied_row("A"), _tied_row("B")])
    assert "uncertainty" in text
    assert "carries no information" in text


def test_describe_tie_orders_by_wait_cost_not_by_championship_probability():
    """Ordering a tie by the objective it is tied on would invite reading
    the tie as a ranking."""
    rows = [
        _tied_row("Safe", wait=0.0, survive=0.99, champ=0.30),
        _tied_row("Scarce", wait=0.94, survive=0.19, champ=0.21),
    ]
    text = describe_tie(rows)
    assert text.index("Scarce") < text.index("Safe")
    assert "**Scarce** costs the most to pass on" in text


def test_describe_tie_breaks_zero_wait_cost_ties_by_survival():
    """Wait cost is clamped to zero once a likely survivor scores as well,
    so a coin-flip player can read 0 and still deserve to come first."""
    rows = [
        _tied_row("Certain", wait=0.0, survive=0.99),
        _tied_row("CoinFlip", wait=0.0, survive=0.49),
    ]
    text = describe_tie(rows)
    assert text.index("CoinFlip") < text.index("Certain")


def test_describe_tie_says_no_urgency_when_everybody_survives():
    """'Nothing to hurry about' is a real answer; manufacturing a reason to
    pick one would be worse than saying so."""
    text = describe_tie([_tied_row("A", survive=0.95), _tied_row("B", survive=0.99)])
    assert "No urgency" in text
    assert "costs the most to pass on" not in text


def test_describe_tie_falls_back_to_judgement_at_our_last_pick():
    """No next turn means survival cannot inform the choice at all."""
    rows = [
        _tied_row("A", wait=None, survive=None),
        _tied_row("B", wait=None, survive=None),
    ]
    for r in rows:
        r["wait_cost_pp"] = None
        r["p_survive"] = None
    text = describe_tie(rows)
    assert "no later turn" in text
    assert "availability" not in text


def test_describe_tie_agrees_with_the_automatic_choice():
    """The callout and `choose_from_tied` must not name different players."""
    rows = [
        _tied_row("Safe", position="WR", wait=0.0, survive=0.99, champ=0.30),
        _tied_row("Scarce", position="RB", wait=0.94, survive=0.19, champ=0.21),
    ]
    chosen = choose_from_tied(rows, counts={}, rounds_left=12)
    assert chosen["name"] == "Scarce"
    assert f"**{chosen['name']}** costs the most to pass on" in describe_tie(rows)
