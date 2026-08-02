from ffdraft.league import (
    SCORING, STARTERS, FLEX_ELIGIBLE, ROSTER_SIZE, BENCH_SIZE,
    N_TEAMS, DRAFT_SLOT, REGULAR_SEASON_WEEKS,
    PLAYOFF_TEAMS, PLAYOFF_ROUNDS, PLAYOFF_BYES, starting_slots_total,
)


def test_scoring_uses_six_point_passing_tds():
    assert SCORING.pass_td == 6.0


def test_scoring_matches_espn_ppr_defaults():
    assert SCORING.pass_yd == 0.04
    assert SCORING.rush_yd == 0.1
    assert SCORING.rec_yd == 0.1
    assert SCORING.rec == 1.0
    assert SCORING.interception == -2.0
    assert SCORING.fumble_lost == -2.0
    assert SCORING.two_pt == 2.0


def test_starters_sum_to_ten():
    assert starting_slots_total() == 10


def test_roster_shape():
    assert ROSTER_SIZE == 18
    assert BENCH_SIZE == 8
    assert ROSTER_SIZE == starting_slots_total() + BENCH_SIZE


def test_flex_excludes_qb_k_dst():
    assert FLEX_ELIGIBLE == frozenset({"RB", "WR", "TE"})


def test_league_and_playoff_shape():
    assert N_TEAMS == 10
    assert DRAFT_SLOT == 8
    assert REGULAR_SEASON_WEEKS == 14
    assert PLAYOFF_TEAMS == 6
    assert PLAYOFF_ROUNDS == 3
    assert PLAYOFF_BYES == 2
