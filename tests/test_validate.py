import polars as pl
import pytest

from ffdraft.league import N_TEAMS
from ffdraft.validate import (
    ValidationError,
    check_draft_completeness,
    check_id_match_rate,
    check_season_coverage,
    collision_review_groups,
    match_rate,
)

# --- check_id_match_rate / match_rate ---------------------------------------


def test_check_id_match_rate_passes_above_threshold():
    df = pl.DataFrame({
        "name": ["A", "B", "C", "D"],
        "gsis_id": ["1", None, None, None],
        "espn_id": [None, 2, None, None],
        "sleeper_id": [None, None, "3", None],
    })
    rate = check_id_match_rate(df, "test", threshold=0.5)
    assert rate == 0.75


def test_check_id_match_rate_raises_below_threshold():
    df = pl.DataFrame({
        "name": ["A", "B"],
        "gsis_id": [None, None],
        "espn_id": [None, None],
        "sleeper_id": [None, None],
    })
    with pytest.raises(ValidationError, match="test"):
        check_id_match_rate(df, "test", threshold=0.5)


def test_check_id_match_rate_error_message_includes_unmatched_sample():
    df = pl.DataFrame({
        "name": ["Nobody Special"],
        "gsis_id": [None],
        "espn_id": [None],
        "sleeper_id": [None],
    })
    with pytest.raises(ValidationError, match="Nobody Special"):
        check_id_match_rate(df, "test", threshold=0.5)


def test_check_id_match_rate_does_not_count_rookies_as_unmatched():
    """The single most important test in this file: rookies have a null
    gsis_id by construction (nflverse only assigns gsis_id on NFL roster
    appearance) but resolve fine via espn_id/sleeper_id once matched. They
    must count as matched, not unmatched -- gating on gsis_id alone would
    fail every incoming draft class."""
    df = pl.DataFrame({
        "name": ["Veteran", "Rookie A", "Rookie B"],
        "gsis_id": ["00-1234567", None, None],
        "espn_id": [1111, 2222, 3333],
        "sleeper_id": ["10", None, None],
    })
    rate = check_id_match_rate(df, "test", threshold=0.9)
    assert rate == 1.0
    assert match_rate(df) == 1.0


# --- check_draft_completeness -----------------------------------------------


def _season_df(season: int, n_rounds: int) -> pl.DataFrame:
    total = N_TEAMS * n_rounds
    overall = list(range(1, total + 1))
    rounds = [((p - 1) // N_TEAMS) + 1 for p in overall]
    round_picks = [((p - 1) % N_TEAMS) + 1 for p in overall]
    return pl.DataFrame({
        "season": [season] * total,
        "overall_pick": overall,
        "round": rounds,
        "round_pick": round_picks,
        "team_id": round_picks,
        "espn_player_id": list(range(total)),
    })


def test_check_draft_completeness_passes_for_170_pick_season():
    df = _season_df(2018, 17)  # this league's actual pre-2023 season length
    check_draft_completeness(df)  # must not raise


def test_check_draft_completeness_passes_for_180_pick_season():
    df = _season_df(2023, 18)  # this league's actual 2023+ season length
    check_draft_completeness(df)  # must not raise


def test_check_draft_completeness_passes_across_mixed_season_lengths():
    """Verified fact: this league ran 170-pick drafts 2018-2022 and 180-pick
    drafts 2023-2025. The check must not assume a constant picks-per-season."""
    df = pl.concat([_season_df(2018, 17), _season_df(2023, 18)])
    check_draft_completeness(df)  # must not raise


def test_check_draft_completeness_raises_on_duplicate_pick():
    df = _season_df(2018, 17)
    picks = df["overall_pick"].to_list()
    picks[-1] = picks[0]  # duplicate pick 1; pick 170 now missing
    df = df.with_columns(pl.Series("overall_pick", picks))
    with pytest.raises(ValidationError, match="2018"):
        check_draft_completeness(df)


def test_check_draft_completeness_raises_on_gap():
    df = _season_df(2018, 17)
    picks = df["overall_pick"].to_list()
    picks[-1] = 171  # unique, out of range -- gap at 170, no duplicate
    df = df.with_columns(pl.Series("overall_pick", picks))
    with pytest.raises(ValidationError, match="2018"):
        check_draft_completeness(df)


def test_check_draft_completeness_raises_on_wrong_total():
    df = _season_df(2018, 17).head(169)  # one pick short of N_TEAMS * max(round)
    with pytest.raises(ValidationError, match="2018"):
        check_draft_completeness(df)


def test_check_draft_completeness_names_the_failing_season():
    good = _season_df(2018, 17)
    bad = _season_df(2019, 17).head(169)
    df = pl.concat([good, bad])
    with pytest.raises(ValidationError, match="2019"):
        check_draft_completeness(df)


# --- check_season_coverage --------------------------------------------------


def test_check_season_coverage_passes_when_complete():
    df = pl.DataFrame({"season": [2018, 2019, 2020]})
    check_season_coverage(df, "test", (2018, 2019, 2020))  # must not raise


def test_check_season_coverage_raises_on_missing_season():
    df = pl.DataFrame({"season": [2018, 2019, 2021]})
    with pytest.raises(ValidationError, match="2020"):
        check_season_coverage(df, "test", (2018, 2019, 2020, 2021))


# --- collision_review_groups -------------------------------------------------


def _collisions_df() -> pl.DataFrame:
    return pl.DataFrame({
        "gsis_id": [None, "00-000001", None, "00-000002"],
        "espn_id": [None, 111, None, 222],
        "sleeper_id": [None, "1", None, "2"],
        "name": ["Player A Sr.", "Player A Jr.", "Player B Sr.", "Player B Jr."],
        "position": ["WR", "WR", "RB", "RB"],
        # group "player a": 1996 - 1990 = 6 years apart -> flag for review
        # group "player b": 2020 - 1980 = 40 years apart -> classic retired-parent case
        "draft_year": [1990, 1996, 1980, 2020],
        "name_key": ["player a", "player a", "player b", "player b"],
        "is_kept": [False, True, False, True],
    })


def test_collision_review_groups_flags_within_10_years_group():
    groups = collision_review_groups(_collisions_df())
    by_key = {g["name_key"]: g for g in groups}
    assert by_key["player a"]["needs_review"] is True
    assert by_key["player b"]["needs_review"] is False


def test_collision_review_groups_does_not_raise():
    """Loud but non-blocking: a review flag must never raise -- a hard
    failure here would halt all ingestion on what might be a false positive."""
    groups = collision_review_groups(_collisions_df())  # must not raise
    assert len(groups) == 2


def test_collision_review_groups_reports_winner_and_losers():
    groups = collision_review_groups(_collisions_df())
    by_key = {g["name_key"]: g for g in groups}
    group_a = by_key["player a"]
    assert group_a["winner"]["name"] == "Player A Jr."
    assert group_a["winner"]["draft_year"] == 1996
    assert [loser["name"] for loser in group_a["losers"]] == ["Player A Sr."]


def test_collision_review_groups_excludes_non_fantasy_positions():
    collisions = _collisions_df().with_columns(pl.lit("DST").alias("position"))
    groups = collision_review_groups(collisions)
    assert groups == []
