import numpy as np
import polars as pl
import pytest

from ffdraft.models.replacement import (
    POSITIONS,
    ReplacementLevelDistribution,
    deep_pool_average,
    drafted_player_ids,
    median_rostered_starter,
    replacement_by_position,
    weekly_replacement_values,
)
from ffdraft.store import exists

# ---------------------------------------------------------------------------
# drafted_player_ids: identity resolution + position labeling


def _crosswalk_row(espn_id, gsis_id):
    return {"gsis_id": gsis_id, "espn_id": espn_id, "sleeper_id": None, "name": "x", "position": "x"}


def _weekly_row(season, week, player_id, position, points):
    return {
        "season": season,
        "week": week,
        "player_id": player_id,
        "position": position,
        "fantasy_points": points,
    }


def test_drafted_player_ids_resolves_via_espn_id_and_weekly_position():
    league_drafts = pl.DataFrame(
        [
            {"season": 2022, "overall_pick": 1, "round": 1, "round_pick": 1, "team_id": 1, "espn_player_id": 100},
            # DST picks use ESPN's negative sentinel ids and must be dropped.
            {"season": 2022, "overall_pick": 2, "round": 1, "round_pick": 2, "team_id": 2, "espn_player_id": -16030},
        ]
    )
    crosswalk = pl.DataFrame([_crosswalk_row(100, "P1")])
    weekly = pl.DataFrame([_weekly_row(2022, 1, "P1", "RB", 12.0)])

    drafted = drafted_player_ids(league_drafts, crosswalk, weekly)
    assert drafted.to_dicts() == [{"season": 2022, "position": "RB", "player_id": "P1"}]


def test_drafted_player_ids_drops_unresolved_picks():
    league_drafts = pl.DataFrame(
        [{"season": 2022, "overall_pick": 1, "round": 1, "round_pick": 1, "team_id": 1, "espn_player_id": 999}]
    )
    crosswalk = pl.DataFrame(schema={"gsis_id": pl.String, "espn_id": pl.Int64, "sleeper_id": pl.String, "name": pl.String, "position": pl.String})
    weekly = pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "player_id": pl.String, "position": pl.String, "fantasy_points": pl.Float64})

    drafted = drafted_player_ids(league_drafts, crosswalk, weekly)
    assert drafted.height == 0


# ---------------------------------------------------------------------------
# weekly_replacement_values: the estimator itself


def _synthetic_season(position: str, season: int, n_weeks: int = 5):
    """Two rostered players (always 100 pts, obviously not "available") and
    five undrafted players whose scores are designed so the top-5-by-
    trailing-form shortlist is deterministic and checkable by hand."""
    drafted_ids = {"R1", "R2"}
    weekly_rows = []
    for week in range(1, n_weeks + 1):
        weekly_rows.append(_weekly_row(season, week, "R1", position, 100.0))
        weekly_rows.append(_weekly_row(season, week, "R2", position, 100.0))
        # Five undrafted candidates: A is a high, steady scorer (always the
        # top trailing-form candidate from week 2 on); B-E score lower.
        weekly_rows.append(_weekly_row(season, week, "A", position, 20.0))
        weekly_rows.append(_weekly_row(season, week, "B", position, 5.0))
        weekly_rows.append(_weekly_row(season, week, "C", position, 5.0))
        weekly_rows.append(_weekly_row(season, week, "D", position, 5.0))
        weekly_rows.append(_weekly_row(season, week, "E", position, 5.0))
    weekly = pl.DataFrame(weekly_rows)
    drafted = pl.DataFrame(
        [{"season": season, "position": position, "player_id": pid} for pid in drafted_ids]
    )
    return weekly, drafted


def test_replacement_value_excludes_rostered_players():
    """A rostered player scoring far more than anyone else must never leak
    into the "available" pool that replacement level is computed from."""
    weekly, drafted = _synthetic_season("RB", 2022)
    table = weekly_replacement_values(weekly, drafted, positions=("RB",), top_k=5)
    # Only 5 undrafted players exist, all scoring <= 20; the rostered
    # players' 100.0 must never appear in the fitted value.
    assert table["value"].max() < 100.0


def test_replacement_value_is_mean_of_top_k_by_trailing_form():
    """With exactly 5 undrafted candidates and top_k=5, the value is just
    the mean of all 5 candidates' current-week score -- a simple case that
    pins down the arithmetic."""
    weekly, drafted = _synthetic_season("RB", 2022, n_weeks=4)
    table = weekly_replacement_values(weekly, drafted, positions=("RB",), top_k=5, trailing_weeks=3)
    week3 = table.filter(pl.col("week") == 3).to_dicts()[0]
    # candidates' week-3 scores: 20, 5, 5, 5, 5 -> mean 8.0
    assert week3["value"] == pytest.approx(8.0)


def test_replacement_value_shortlist_uses_trailing_form_not_current_week():
    """top_k=1 with a candidate pool where the current-week ranking would
    differ from the trailing-form ranking proves the shortlist is chosen
    ex-ante (by past form), not by peeking at this week's own outcome."""
    weekly_rows = []
    season = 2023
    # Weeks 1-2: F is the best performer (sets up F's trailing form).
    for week in [1, 2]:
        weekly_rows.append(_weekly_row(season, week, "F", "WR", 30.0))
        weekly_rows.append(_weekly_row(season, week, "G", "WR", 1.0))
    # Week 3: G suddenly outscores F, but the shortlist should still be
    # F (highest trailing average heading into week 3), not G.
    weekly_rows.append(_weekly_row(season, 3, "F", "WR", 2.0))
    weekly_rows.append(_weekly_row(season, 3, "G", "WR", 50.0))
    weekly = pl.DataFrame(weekly_rows)
    drafted = pl.DataFrame(schema={"season": pl.Int64, "position": pl.String, "player_id": pl.String})

    table = weekly_replacement_values(weekly, drafted, positions=("WR",), top_k=1, trailing_weeks=2)
    week3 = table.filter(pl.col("week") == 3).to_dicts()[0]
    assert week3["value"] == pytest.approx(2.0)  # F's actual week-3 score, not G's


def test_week_one_excluded_no_trailing_history():
    weekly, drafted = _synthetic_season("RB", 2022, n_weeks=3)
    table = weekly_replacement_values(weekly, drafted, positions=("RB",))
    assert 1 not in table["week"].to_list()


# ---------------------------------------------------------------------------
# ReplacementLevelDistribution: WeeklyDistribution protocol conformance


def test_mean_property_matches_observed_values():
    dist = ReplacementLevelDistribution(position="RB", values=(4.0, 6.0, 8.0))
    assert dist.mean == pytest.approx(6.0)


def test_sample_returns_requested_size():
    dist = ReplacementLevelDistribution(position="RB", values=(4.0, 6.0, 8.0))
    out = dist.sample(np.random.default_rng(0), 500)
    assert out.shape == (500,)


def test_sampled_mean_converges_to_stated_mean():
    dist = ReplacementLevelDistribution(position="WR", values=tuple(np.linspace(3.0, 15.0, 50)))
    samples = dist.sample(np.random.default_rng(0), 200_000)
    assert abs(samples.mean() - dist.mean) < 0.1


def test_sampling_is_reproducible_from_a_seed():
    dist = ReplacementLevelDistribution(position="TE", values=(2.0, 4.0, 6.0, 8.0))
    a = dist.sample(np.random.default_rng(7), 1000)
    b = dist.sample(np.random.default_rng(7), 1000)
    assert np.array_equal(a, b)


def test_sampling_does_not_touch_global_numpy_state():
    dist = ReplacementLevelDistribution(position="TE", values=(2.0, 4.0, 6.0, 8.0))
    np.random.seed(12345)
    before = np.random.get_state()[1].copy()
    dist.sample(np.random.default_rng(0), 1000)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


def test_sample_varies_not_a_single_constant():
    """Regression guard for Step 2: replacement level must not degenerate
    into one fixed scalar."""
    dist = ReplacementLevelDistribution(position="RB", values=(3.0, 9.0, 15.0, 21.0))
    samples = dist.sample(np.random.default_rng(0), 1000)
    assert samples.std() > 0.5


def test_empty_values_raises():
    with pytest.raises(ValueError):
        ReplacementLevelDistribution(position="RB", values=())


# ---------------------------------------------------------------------------
# Real-data regression guards -- gated on the cached fit existing.

_HAS_REPLACEMENT_FIT = exists("replacement_weekly_values") or exists("weekly_stats")


def _get_replacement_and_baselines():
    from ffdraft.store import read

    weekly = read("weekly_stats")
    league_drafts = read("league_drafts")
    crosswalk = read("id_crosswalk")
    drafted = drafted_player_ids(league_drafts, crosswalk, weekly)

    replacement = replacement_by_position(weekly=weekly, league_drafts=league_drafts, crosswalk=crosswalk)
    starters = median_rostered_starter(weekly)
    deep = deep_pool_average(weekly, drafted)
    return replacement, starters, deep


@pytest.mark.skipif(not _HAS_REPLACEMENT_FIT, reason="requires weekly_stats/league_drafts/id_crosswalk data")
def test_replacement_level_is_positive_for_every_position():
    replacement, _, _ = _get_replacement_and_baselines()
    for position in POSITIONS:
        assert replacement[position].mean > 0, position


@pytest.mark.skipif(not _HAS_REPLACEMENT_FIT, reason="requires weekly_stats/league_drafts/id_crosswalk data")
def test_replacement_level_is_below_median_starter_and_above_deep_pool():
    """The central-trap regression guard: replacement level must sit
    strictly between the deep-pool average (the trap) and the median
    rostered starter (too generous) for every position."""
    replacement, starters, deep = _get_replacement_and_baselines()
    starters_by_pos = {r["position"]: r["median_starter"] for r in starters.to_dicts()}
    deep_by_pos = {r["position"]: r["deep_pool_avg"] for r in deep.to_dicts()}

    for position in POSITIONS:
        r = replacement[position].mean
        s = starters_by_pos[position]
        d = deep_by_pos[position]
        assert d < r < s, f"{position}: deep={d}, replacement={r}, starter_median={s}"


@pytest.mark.skipif(not _HAS_REPLACEMENT_FIT, reason="requires weekly_stats/league_drafts/id_crosswalk data")
def test_replacement_level_varies_by_week_not_a_single_constant():
    replacement, _, _ = _get_replacement_and_baselines()
    for position in POSITIONS:
        assert len(set(replacement[position].values)) > 5, position


@pytest.mark.skipif(not _HAS_REPLACEMENT_FIT, reason="requires weekly_stats/league_drafts/id_crosswalk data")
def test_qb_and_k_replacement_are_relatively_closer_to_starters_than_rb_and_wr():
    """The streamability finding: if QB/K replacement is relatively closer
    to its own starter median than RB/WR is, report it -- this is a
    regression guard on an observed finding, not an assumed one. See the
    Task 5 report for the actual ratios if this ever needs re-deriving."""
    replacement, starters, _ = _get_replacement_and_baselines()
    starters_by_pos = {r["position"]: r["median_starter"] for r in starters.to_dicts()}

    ratios = {
        pos: replacement[pos].mean / starters_by_pos[pos] for pos in POSITIONS
    }
    assert ratios["QB"] > ratios["RB"]
    assert ratios["QB"] > ratios["WR"]
    assert ratios["K"] > ratios["RB"]
    assert ratios["K"] > ratios["WR"]
