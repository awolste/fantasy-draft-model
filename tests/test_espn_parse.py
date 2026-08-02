import json

import polars as pl
import pytest

from ffdraft.league import N_TEAMS
from ffdraft.sources.espn_parse import (
    DRAFT_COLUMNS, MANAGER_COLUMNS, RESULT_COLUMNS,
    parse_draft, parse_managers, parse_results,
)


@pytest.fixture
def payload():
    with open("tests/fixtures/espn_2024.json") as fh:
        return json.load(fh)


def test_parse_draft_returns_canonical_columns(payload):
    out = parse_draft(payload, season=2024)
    assert out.columns == list(DRAFT_COLUMNS)


def test_parse_draft_has_one_row_per_pick(payload):
    out = parse_draft(payload, season=2024)
    assert out.height == 180
    assert out["overall_pick"].min() == 1
    assert out["overall_pick"].max() == 180


def test_parse_draft_round_and_slot_are_consistent(payload):
    """overall_pick must equal (round - 1) * N_TEAMS + round_pick."""
    out = parse_draft(payload, season=2024)
    derived = (pl.col("round") - 1) * N_TEAMS + pl.col("round_pick")
    mismatches = out.filter(derived != pl.col("overall_pick"))
    assert mismatches.height == 0


def test_parse_managers_keys_on_swid_not_team_id(payload):
    out = parse_managers(payload, season=2024)
    assert out.columns == list(MANAGER_COLUMNS)
    assert out.height == 10
    assert out["manager_id"].null_count() == 0
    assert out["manager_id"].n_unique() == 10


def test_parse_results_identifies_exactly_one_champion(payload):
    out = parse_results(payload, season=2024)
    assert out.columns == list(RESULT_COLUMNS)
    champions = out.filter(pl.col("final_rank") == 1)
    assert champions.height == 1


def test_parse_results_playoff_seeds_cover_six_teams(payload):
    out = parse_results(payload, season=2024)
    seeded = out.filter(pl.col("playoff_seed") <= 6)
    assert seeded.height == 6
