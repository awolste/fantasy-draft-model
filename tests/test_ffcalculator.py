import json

import pytest

from ffdraft.sources.ffcalculator import ADP_COLUMNS, parse_adp


@pytest.fixture
def payload_2026():
    with open("tests/fixtures/ffc_adp_2026.json") as fh:
        return json.load(fh)


@pytest.fixture
def payload_2024():
    with open("tests/fixtures/ffc_adp_2024.json") as fh:
        return json.load(fh)


def test_parse_adp_returns_canonical_columns(payload_2026):
    out = parse_adp(payload_2026, 2026)
    assert out.columns == list(ADP_COLUMNS)


def test_parse_adp_returns_a_full_player_pool_sorted_by_adp(payload_2026):
    out = parse_adp(payload_2026, 2026)
    assert out.height > 150
    assert out["adp"][0] < 2.0


def test_season_column_matches_passed_season(payload_2026, payload_2024):
    out_2026 = parse_adp(payload_2026, 2026)
    assert set(out_2026["season"].unique().to_list()) == {2026}

    out_2024 = parse_adp(payload_2024, 2024)
    assert set(out_2024["season"].unique().to_list()) == {2024}


def test_positions_are_normalized(payload_2026):
    out = parse_adp(payload_2026, 2026)
    assert set(out["position"].unique()) <= {"QB", "RB", "WR", "TE", "K", "DST"}


def test_parse_adp_raises_on_error_status_naming_season():
    payload = {"status": "Error", "errors": "No ADP data found.", "players": []}
    with pytest.raises(ValueError, match="2025"):
        parse_adp(payload, 2025)


def test_parse_adp_raises_on_empty_players_even_with_success_status():
    payload = {"status": "Success", "players": []}
    with pytest.raises(ValueError, match="2026"):
        parse_adp(payload, 2026)
