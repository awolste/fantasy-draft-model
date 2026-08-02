"""Convert raw NFL stat lines into fantasy points under this league's rules.

Never consume a precomputed fantasy point total from an external source.
Public totals assume 4-point passing touchdowns; this league uses 6.
"""

from collections.abc import Mapping
from dataclasses import fields

import polars as pl

from .league import SCORING, ScoringRules

# Maps a stat column name to the ScoringRules field that prices it.
STAT_TO_RULE: dict[str, str] = {
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "interceptions": "interception",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "receptions": "rec",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",
    "fumbles_lost": "fumble_lost",
    "two_pt_conversions": "two_pt",
}


def validate_stat_to_rule(mapping: Mapping[str, str]) -> None:
    """Raise if any mapped value is not a real ScoringRules field name."""
    valid_fields = {f.name for f in fields(ScoringRules)}
    invalid = {stat: rule for stat, rule in mapping.items() if rule not in valid_fields}
    if invalid:
        raise ValueError(
            f"STAT_TO_RULE references unknown ScoringRules field(s): {invalid}. "
            f"Valid fields are: {sorted(valid_fields)}"
        )


validate_stat_to_rule(STAT_TO_RULE)


def score_stat_line(line: Mapping[str, float], rules: ScoringRules = SCORING) -> float:
    """Score a single stat line. Missing or null stats count as zero."""
    total = 0.0
    for stat, rule_name in STAT_TO_RULE.items():
        value = line.get(stat)
        if value is None:
            continue
        total += float(value) * getattr(rules, rule_name)
    return round(total, 2)


def add_fantasy_points(df: pl.DataFrame, rules: ScoringRules = SCORING) -> pl.DataFrame:
    """Append a `fantasy_points` column. Stat columns absent from the frame
    contribute nothing, so the same call works for QBs and kickers alike."""
    terms = [
        (pl.col(stat).fill_null(0.0) * getattr(rules, rule_name))
        for stat, rule_name in STAT_TO_RULE.items()
        if stat in df.columns
    ]
    if not terms:
        return df.with_columns(pl.lit(0.0).alias("fantasy_points"))
    return df.with_columns(pl.sum_horizontal(terms).round(2).alias("fantasy_points"))
