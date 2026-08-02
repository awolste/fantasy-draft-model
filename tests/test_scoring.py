import polars as pl

from ffdraft.league import SCORING
from ffdraft.scoring import score_stat_line, add_fantasy_points

STAT_COLUMNS = [
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "fumbles_lost", "two_pt_conversions",
]


def blank_line(**overrides) -> dict:
    line = {col: 0.0 for col in STAT_COLUMNS}
    line.update(overrides)
    return line


def test_empty_stat_line_scores_zero():
    assert score_stat_line(blank_line(), SCORING) == 0.0


def test_passing_touchdown_is_worth_six():
    assert score_stat_line(blank_line(passing_tds=1), SCORING) == 6.0


def test_passing_yards_quarter_point_per_25():
    assert score_stat_line(blank_line(passing_yards=300), SCORING) == 12.0


def test_full_ppr_reception_worth_one():
    assert score_stat_line(blank_line(receptions=8), SCORING) == 8.0


def test_interceptions_and_fumbles_are_negative():
    line = blank_line(interceptions=2, fumbles_lost=1)
    assert score_stat_line(line, SCORING) == -6.0


def test_realistic_quarterback_line():
    # 320 pass yd, 3 pass TD, 1 INT, 30 rush yd, 1 rush TD
    line = blank_line(
        passing_yards=320, passing_tds=3, interceptions=1,
        rushing_yards=30, rushing_tds=1,
    )
    # 12.8 + 18 - 2 + 3 + 6 = 37.8
    assert score_stat_line(line, SCORING) == 37.8


def test_realistic_receiver_line():
    # 9 rec, 120 yd, 1 TD
    line = blank_line(receptions=9, receiving_yards=120, receiving_tds=1)
    assert score_stat_line(line, SCORING) == 27.0


def test_six_point_passing_td_beats_four_point_assumption():
    """Guards the single most important league-specific rule."""
    line = blank_line(passing_tds=4)
    assert score_stat_line(line, SCORING) == 24.0
    assert score_stat_line(line, SCORING) != 16.0


def test_add_fantasy_points_appends_column_to_frame():
    df = pl.DataFrame([
        blank_line(passing_tds=1) | {"player_id": "A"},
        blank_line(receptions=5, receiving_yards=50) | {"player_id": "B"},
    ])
    out = add_fantasy_points(df, SCORING)
    assert "fantasy_points" in out.columns
    assert out["fantasy_points"].to_list() == [6.0, 10.0]


def test_add_fantasy_points_treats_missing_stats_as_zero():
    """Kickers have no passing columns; nulls must not poison the sum."""
    df = pl.DataFrame({"player_id": ["K1"], "receptions": [None], "rushing_yards": [12.0]})
    out = add_fantasy_points(df, SCORING)
    assert out["fantasy_points"].to_list() == [1.2]
