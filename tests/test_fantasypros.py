import polars as pl
import pytest

from ffdraft.sources.fantasypros import RANKING_COLUMNS, parse_ecr


@pytest.fixture
def html():
    with open("tests/fixtures/fantasypros_ecr.html") as fh:
        return fh.read()


def test_parse_ecr_returns_canonical_columns(html):
    out = parse_ecr(html)
    assert out.columns == list(RANKING_COLUMNS)


def test_parse_ecr_returns_a_full_player_pool(html):
    out = parse_ecr(html)
    assert out.height > 150


def test_ranks_start_at_one_and_are_unique(html):
    out = parse_ecr(html)
    assert out["rank"].min() == 1
    assert out["rank"].n_unique() == out.height


def test_positions_are_normalized_without_rank_digits(html):
    """FantasyPros writes 'WR1', 'RB12'; we want the bare position."""
    out = parse_ecr(html)
    assert set(out["position"].unique()) <= {"QB", "RB", "WR", "TE", "K", "DST"}


def test_every_row_has_a_name(html):
    out = parse_ecr(html)
    assert out["name"].null_count() == 0
    assert (out["name"].str.len_chars() > 0).all()
