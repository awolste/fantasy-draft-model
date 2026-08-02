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


# --- Cardinality-preservation regression tests ---
# `normalize_name` strips generational suffixes, so father/son pairs like
# Marvin Harrison Sr./Jr. collide onto the same (name, position) key. If the
# crosswalk passed to `match_by_name` has more than one row per key, a plain
# left join multiplies output rows -- a silent, uncaught cardinality bug.

COLLISION_CROSSWALK = pl.DataFrame({
    "gsis_id": [None, "00-0039849"],
    "espn_id": [None, 4432708],
    "sleeper_id": [None, "9490"],
    "name": ["Marvin Harrison", "Marvin Harrison Jr."],
    "position": ["WR", "WR"],
})


def test_match_by_name_raises_on_duplicate_crosswalk_keys():
    """A crosswalk with an undeduped (name, position) collision must fail
    loudly rather than silently return extra rows."""
    names = pl.DataFrame({"name": ["Marvin Harrison Jr."], "position": ["WR"]})
    with pytest.raises(ValueError, match="duplicate"):
        match_by_name(names, COLLISION_CROSSWALK)


def test_match_by_name_resolves_collision_to_intended_player_when_deduped():
    """Once the crosswalk is deduped (as `load_crosswalk` does, preferring
    the row with a real gsis_id), the collision resolves to the active
    player, not the retired namesake."""
    deduped = COLLISION_CROSSWALK.filter(pl.col("gsis_id").is_not_null())
    names = pl.DataFrame({"name": ["Marvin Harrison Jr."], "position": ["WR"]})
    out = match_by_name(names, deduped)
    assert out.height == 1
    assert out["gsis_id"].to_list() == ["00-0039849"]
    assert out["espn_id"].to_list() == [4432708]


def test_match_by_name_output_height_matches_input_height():
    """General invariant: for a properly deduped crosswalk, one input row
    produces exactly one output row, whether matched, unmatched, or a name
    that would have collided before dedupe."""
    deduped = COLLISION_CROSSWALK.filter(pl.col("gsis_id").is_not_null())
    crosswalk = pl.concat([CROSSWALK, deduped])
    names = pl.DataFrame({
        "name": ["Lamar Jackson", "Nonexistent Player", "Marvin Harrison Jr."],
        "position": ["QB", "RB", "WR"],
    })
    out = match_by_name(names, crosswalk)
    assert out.height == names.height
