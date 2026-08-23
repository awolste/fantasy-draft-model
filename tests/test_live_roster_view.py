"""`live.roster_view` -- our own roster as a starting lineup.

The point of this view is to be right about the two things a counts dict
gets wrong: that a third running back is a starter (via FLEX) and that a
second tight end usually is not. So that is what these pin.
"""

from __future__ import annotations

import pytest

from ffdraft.live.roster_view import league_standing, team_view


class _Dist:
    def __init__(self, mean):
        self.mean = mean


class _Player:
    def __init__(self, name, mean):
        self.name = name
        self.distribution = _Dist(mean)


REPLACEMENT = {"QB": 12.0, "RB": 6.0, "WR": 7.0, "TE": 5.0, "K": 7.0, "DST": 6.0}


def _pool(spec):
    """`{player_id: (name, projected weekly mean)}` -> a pool-shaped dict."""
    return {pid: _Player(name, mean) for pid, (name, mean) in spec.items()}


def test_empty_roster_is_ten_empty_slots_scored_at_replacement():
    view = team_view([], _pool({}), REPLACEMENT)

    assert len(view.starters) == 10
    assert view.n_replacement == 10
    assert all(s.is_empty and s.player_id is None for s in view.starters)
    assert view.bench == ()
    # FLEX is filled at the best replacement level among RB/WR/TE, so the
    # total is not simply the sum of the dedicated positions' levels.
    expected = 12.0 + 2 * 6.0 + 2 * 7.0 + 5.0 + 2 * 7.0 + 7.0 + 6.0
    assert view.projected_points == pytest.approx(expected)


def test_slots_read_in_lineup_order_and_are_numbered_only_when_repeated():
    view = team_view([], _pool({}), REPLACEMENT)
    assert [s.label for s in view.starters] == [
        "QB", "RB1", "RB2", "WR1", "WR2", "TE", "FLEX1", "FLEX2", "K", "DST",
    ]


def test_third_running_back_starts_at_flex_and_the_best_backs_are_numbered_first():
    """The counts-dict view cannot say this; it is the whole reason the
    lineup is solved rather than grouped by position."""
    pool = _pool({
        "rb1": ("Best Back", 18.0),
        "rb2": ("Good Back", 14.0),
        "rb3": ("Third Back", 11.0),
    })
    view = team_view([("rb1", "RB"), ("rb2", "RB"), ("rb3", "RB")], pool, REPLACEMENT)

    by_label = {s.label: s for s in view.starters}
    assert by_label["RB1"].name == "Best Back"
    assert by_label["RB2"].name == "Good Back"
    assert by_label["FLEX1"].name == "Third Back"
    assert view.bench == ()
    assert view.n_replacement == 7


def test_the_two_flex_slots_go_to_the_best_spares_and_the_rest_sit():
    """Three players compete for two flex slots, so somebody sits -- and
    which somebody is a projection question, not a position question."""
    pool = _pool({
        "te1": ("Starter TE", 13.0),
        "te2": ("Backup TE", 9.0),
        "wr1": ("WR One", 16.0),
        "wr2": ("WR Two", 15.0),
        "wr3": ("WR Three", 10.5),
        "wr4": ("WR Four", 9.8),
    })
    roster = [("te1", "TE"), ("te2", "TE"),
              ("wr1", "WR"), ("wr2", "WR"), ("wr3", "WR"), ("wr4", "WR")]
    view = team_view(roster, pool, REPLACEMENT)

    by_label = {s.label: s for s in view.starters}
    assert by_label["TE"].name == "Starter TE"
    # 10.5 and 9.8 take the two flex slots, best first; the second tight
    # end -- a "starter" by any counts-based reading -- is the one benched.
    assert by_label["FLEX1"].name == "WR Three"
    assert by_label["FLEX2"].name == "WR Four"
    assert [b.name for b in view.bench] == ["Backup TE"]


def test_bench_is_ordered_by_projection_not_draft_order():
    pool = _pool({
        "qb1": ("QB One", 22.0),
        "qb2": ("QB Two", 19.0),
        "qb3": ("QB Three", 20.5),
    })
    view = team_view([("qb1", "QB"), ("qb2", "QB"), ("qb3", "QB")], pool, REPLACEMENT)

    assert [s for s in view.starters if s.label == "QB"][0].name == "QB One"
    # Quarterbacks are not FLEX-eligible, so the other two are bench, best
    # first regardless of when they were taken.
    assert [b.name for b in view.bench] == ["QB Three", "QB Two"]


def test_empty_slots_are_named_so_the_ui_can_say_what_is_missing():
    pool = _pool({"qb1": ("QB One", 22.0), "k1": ("Kicker", 8.0)})
    view = team_view([("qb1", "QB"), ("k1", "K")], pool, REPLACEMENT)

    assert view.empty_slots == ("RB1", "RB2", "WR1", "WR2", "TE", "FLEX1", "FLEX2", "DST")
    assert view.n_replacement == 8


def test_a_player_missing_from_the_pool_raises_rather_than_rendering_blank():
    """A nameless starter is the roster that looks fine and is not."""
    with pytest.raises(KeyError, match="absent from the pool"):
        team_view([("ghost", "RB")], _pool({}), REPLACEMENT)


def test_projected_points_matches_the_sum_of_the_slots_it_shows():
    """The headline number must be the table beneath it, or one of them is
    lying about the other."""
    pool = _pool({
        "rb1": ("Back One", 17.0), "wr1": ("Rec One", 15.5), "qb1": ("QB One", 21.0),
    })
    view = team_view([("rb1", "RB"), ("wr1", "WR"), ("qb1", "QB")], pool, REPLACEMENT)
    assert view.projected_points == pytest.approx(
        sum(s.projected_points for s in view.starters)
    )


# ---------------------------------------------------------------------------
# Where we sit against the room


class _Pick:
    def __init__(self, team, player_id, position):
        self.team = team
        self.player_id = player_id
        self.position = position


def test_league_standing_ranks_us_against_every_other_team():
    pool = _pool({
        "a": ("Stud", 25.0), "b": ("Good", 18.0), "c": ("Fine", 12.0),
    })
    picks = [
        _Pick(1, "a", "RB"),   # team 1 has the best back
        _Pick(2, "b", "RB"),
        _Pick(3, "c", "RB"),
    ]
    standing = league_standing(picks, pool, REPLACEMENT, n_teams=4, our_team=2)

    assert standing.n_teams == 4
    assert standing.rank == 2
    assert standing.points == standing.points_by_team[1]
    # Team 4 drafted nobody, so it sits at the all-replacement floor.
    assert standing.points_by_team[3] < standing.points_by_team[2]


def test_league_standing_is_all_replacement_and_all_tied_before_any_pick():
    """Before the draft every team is identical, so nobody is 'ahead' --
    the metric must not manufacture an ordering out of nothing."""
    standing = league_standing([], _pool({}), REPLACEMENT, n_teams=10, our_team=8)
    assert len(set(standing.points_by_team)) == 1
    assert standing.rank == 1


def test_league_standing_ties_share_the_better_rank():
    pool = _pool({"a": ("One", 20.0), "b": ("Two", 20.0)})
    picks = [_Pick(1, "a", "RB"), _Pick(2, "b", "RB")]
    standing = league_standing(picks, pool, REPLACEMENT, n_teams=3, our_team=2)
    assert standing.rank == 1


def test_league_standing_rejects_a_slot_outside_the_league():
    with pytest.raises(ValueError, match="outside 1..10"):
        league_standing([], _pool({}), REPLACEMENT, n_teams=10, our_team=11)
