from dataclasses import fields

import polars as pl
import pytest

from ffdraft.league import KICKING, KickingRules
from ffdraft.models.kicking import KICKING_STAT_TO_RULE, score_kicking_line, add_kicking_points, validate_stat_to_rule


def blank(**overrides) -> dict:
    line = {stat: 0.0 for stat in KICKING_STAT_TO_RULE}
    line.update(overrides)
    return line


def test_pat_worth_one():
    assert score_kicking_line(blank(pat_made=3)) == 3.0


def test_short_field_goals_worth_three():
    assert score_kicking_line(blank(fg_made_0_19=1, fg_made_20_29=1, fg_made_30_39=1)) == 9.0


def test_forty_yard_range_worth_four():
    assert score_kicking_line(blank(fg_made_40_49=2)) == 8.0


def test_fifty_and_sixty_both_worth_five():
    """Distance past 50 carries no extra value in this league."""
    assert score_kicking_line(blank(fg_made_50_59=1)) == 5.0
    assert score_kicking_line(blank(fg_made_60_=1)) == 5.0


def test_missed_field_goal_costs_one():
    assert score_kicking_line(blank(fg_missed=2)) == -2.0


def test_realistic_kicker_week():
    # 3 PAT, one 45-yarder, one 52-yarder, one miss
    line = blank(pat_made=3, fg_made_40_49=1, fg_made_50_59=1, fg_missed=1)
    assert score_kicking_line(line) == 11.0


def test_add_kicking_points_appends_column():
    df = pl.DataFrame([blank(pat_made=2) | {"player_id": "K1"}])
    out = add_kicking_points(df)
    assert out["fantasy_points"].to_list() == [2.0]


def test_missing_columns_treated_as_zero():
    df = pl.DataFrame({"player_id": ["K1"], "pat_made": [4.0]})
    assert add_kicking_points(df)["fantasy_points"].to_list() == [4.0]


def test_blocked_field_goal_costs_one():
    assert score_kicking_line(blank(fg_blocked=1)) == -1.0


def test_missed_and_blocked_both_penalize():
    assert score_kicking_line(blank(fg_missed=1, fg_blocked=1)) == -2.0


def test_validate_stat_to_rule_rejects_unknown_field():
    bad_mapping = {"pat_made": "not_a_real_field"}
    with pytest.raises(ValueError, match="not_a_real_field"):
        validate_stat_to_rule(bad_mapping)


def test_validate_stat_to_rule_accepts_known_fields():
    valid_fields = {f.name for f in fields(KickingRules)}
    good_mapping = {"pat_made": next(iter(valid_fields))}
    validate_stat_to_rule(good_mapping)  # should not raise
