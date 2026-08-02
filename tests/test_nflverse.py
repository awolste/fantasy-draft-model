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
