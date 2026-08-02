import polars as pl
import pytest

from ffdraft.ids import CROSSWALK_COLUMNS, match_by_name, normalize_name

CROSSWALK = pl.DataFrame({
    "gsis_id": ["00-0034796", "00-0036322", "00-0033873"],
    "espn_id": [3139477, 4262921, 3116406],
    "sleeper_id": ["4034", "6794", "4035"],
    "name": ["Lamar Jackson", "Ja'Marr Chase", "Christian McCaffrey"],
    "position": ["QB", "WR", "RB"],
})


def test_normalize_name_strips_punctuation_and_case():
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"
    assert normalize_name("A.J. Brown") == "aj brown"
    assert normalize_name("  Amon-Ra  St. Brown ") == "amon ra st brown"


def test_normalize_name_strips_generational_suffixes():
    assert normalize_name("Odell Beckham Jr.") == "odell beckham"
    assert normalize_name("Michael Pittman Jr") == "michael pittman"
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"


def test_match_by_name_resolves_apostrophes():
    names = pl.DataFrame({"name": ["JaMarr Chase"], "position": ["WR"]})
    out = match_by_name(names, CROSSWALK)
    assert out["gsis_id"].to_list() == ["00-0036322"]


def test_match_by_name_requires_position_agreement():
    """Two players can share a name; position disambiguates."""
    names = pl.DataFrame({"name": ["Lamar Jackson"], "position": ["WR"]})
    out = match_by_name(names, CROSSWALK)
    assert out["gsis_id"].to_list() == [None]


def test_match_by_name_keeps_unmatched_rows_with_null_id():
    names = pl.DataFrame({"name": ["Nonexistent Player"], "position": ["RB"]})
    out = match_by_name(names, CROSSWALK)
    assert out.height == 1
    assert out["gsis_id"].to_list() == [None]


def test_crosswalk_columns_are_the_four_id_systems():
    assert set(CROSSWALK_COLUMNS) == {"gsis_id", "espn_id", "sleeper_id", "name", "position"}
