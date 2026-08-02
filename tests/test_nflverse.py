import polars as pl
import pytest

from ffdraft.sources.nflverse import CANONICAL_COLUMNS, normalize_weekly

FIXTURE = "tests/fixtures/nflverse_2024_sample.parquet"


@pytest.fixture
def raw():
    return pl.read_parquet(FIXTURE)


def test_normalize_produces_exactly_the_canonical_columns(raw):
    out = normalize_weekly(raw)
    assert out.columns == list(CANONICAL_COLUMNS)


def test_normalize_scores_fantasy_points(raw):
    out = normalize_weekly(raw)
    assert out["fantasy_points"].null_count() == 0
    assert out["fantasy_points"].max() > 0


def test_positions_restricted_to_fantasy_relevant(raw):
    out = normalize_weekly(raw)
    assert set(out["position"].unique()) <= {"QB", "RB", "WR", "TE", "K"}


def test_no_null_player_ids(raw):
    out = normalize_weekly(raw)
    assert out["player_id"].null_count() == 0


def test_week_and_season_are_integers(raw):
    out = normalize_weekly(raw)
    assert out.schema["week"] == pl.Int64
    assert out.schema["season"] == pl.Int64


def test_quarterback_points_reflect_six_point_tds(raw):
    """A QB week with passing TDs must score above the 4-point-TD equivalent."""
    out = normalize_weekly(raw)
    qbs = out.filter((pl.col("position") == "QB") & (pl.col("passing_tds") >= 2))
    assert qbs.height > 0
    row = qbs.row(0, named=True)
    four_pt_equivalent = row["fantasy_points"] - 2 * row["passing_tds"]
    assert row["fantasy_points"] > four_pt_equivalent


def test_alternate_aliases_resolve():
    """A frame using the non-primary candidate for every multi-candidate
    alias must still normalize correctly. The real fixture only exercises
    whichever candidate happened to be present in that one nflreadpy
    release, so this is the only place the other candidates are covered."""
    raw = pl.DataFrame({
        "gsis_id": ["00-1234567"],
        "player_name": ["Test Player"],
        "position": ["QB"],
        "recent_team": ["BUF"],
        "season": [2024],
        "week": [1],
        "passing_yards": [300],
        "passing_tds": [3],
        "passing_interceptions": [1],
        "rushing_yards": [20],
        "rushing_tds": [1],
        "receptions": [0],
        "targets": [0],
        "receiving_yards": [0],
        "receiving_tds": [0],
        "special_teams_tds": [0],
        "fumble_recovery_tds": [0],
    })
    out = normalize_weekly(raw)
    assert out.columns == list(CANONICAL_COLUMNS)
    row = out.row(0, named=True)
    assert row["player_id"] == "00-1234567"
    assert row["player_name"] == "Test Player"
    assert row["team"] == "BUF"
    assert row["interceptions"] == 1
    assert row["passing_tds"] == 3


def test_missing_optional_summed_parts_default_to_zero():
    """When none of FUMBLE_PARTS or TWO_PT_PARTS are present upstream, the
    summed columns must come back as 0.0 rather than erroring or null."""
    raw = pl.DataFrame({
        "player_id": ["00-1111111"],
        "player_name": ["Another Player"],
        "position": ["RB"],
        "team": ["SF"],
        "season": [2024],
        "week": [1],
        "passing_yards": [0],
        "passing_tds": [0],
        "interceptions": [0],
        "rushing_yards": [50],
        "rushing_tds": [1],
        "receptions": [2],
        "targets": [3],
        "receiving_yards": [15],
        "receiving_tds": [0],
        "special_teams_tds": [0],
        "fumble_recovery_tds": [0],
    })
    out = normalize_weekly(raw)
    row = out.row(0, named=True)
    assert row["fumbles_lost"] == 0.0
    assert row["two_pt_conversions"] == 0.0


def test_return_touchdown_flows_through_and_scores_six():
    """special_teams_tds (kickoff/punt return TDs) must reach the canonical
    frame and score 6 points, matching the league's KRTD/PRTD rule -- the
    original bug scored these as zero because normalize_weekly dropped the
    column entirely."""
    raw = pl.DataFrame({
        "player_id": ["00-2222222"],
        "player_name": ["Return Man"],
        "position": ["WR"],
        "team": ["KC"],
        "season": [2024],
        "week": [1],
        "passing_yards": [0],
        "passing_tds": [0],
        "interceptions": [0],
        "rushing_yards": [0],
        "rushing_tds": [0],
        "receptions": [0],
        "targets": [0],
        "receiving_yards": [0],
        "receiving_tds": [0],
        "special_teams_tds": [1],
        "fumble_recovery_tds": [0],
    })
    out = normalize_weekly(raw)
    row = out.row(0, named=True)
    assert row["special_teams_tds"] == 1
    assert row["fantasy_points"] == 6.0


def test_fumble_recovery_touchdown_flows_through_and_scores_six():
    """fumble_recovery_tds (FTD) must reach the canonical frame and score 6
    points, matching the league's FTD rule. Distinct from special_teams_tds
    -- nflverse tracks fumble-recovery TDs separately from punt/kickoff
    return TDs."""
    raw = pl.DataFrame({
        "player_id": ["00-3333333"],
        "player_name": ["Scoop Guy"],
        "position": ["RB"],
        "team": ["DAL"],
        "season": [2024],
        "week": [1],
        "passing_yards": [0],
        "passing_tds": [0],
        "interceptions": [0],
        "rushing_yards": [0],
        "rushing_tds": [0],
        "receptions": [0],
        "targets": [0],
        "receiving_yards": [0],
        "receiving_tds": [0],
        "special_teams_tds": [0],
        "fumble_recovery_tds": [1],
    })
    out = normalize_weekly(raw)
    row = out.row(0, named=True)
    assert row["fumble_recovery_tds"] == 1
    assert row["fantasy_points"] == 6.0


def test_missing_required_column_raises():
    """If no candidate for a required canonical column is present upstream,
    normalize_weekly must raise rather than silently producing an all-null
    column that add_fantasy_points would then treat as zero."""
    raw = pl.DataFrame({
        "player_id": ["00-1111111"],
        "player_name": ["Another Player"],
        "position": ["RB"],
        "team": ["SF"],
        "season": [2024],
        "week": [1],
        "passing_yards": [0],
        # passing_tds intentionally omitted under every alias
        "interceptions": [0],
        "rushing_yards": [50],
        "rushing_tds": [1],
        "receptions": [2],
        "targets": [3],
        "receiving_yards": [15],
        "receiving_tds": [0],
    })
    with pytest.raises(ValueError, match="passing_tds"):
        normalize_weekly(raw)
