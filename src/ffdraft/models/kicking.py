"""Convert raw kicker stat lines into fantasy points under this league's rules.

Models on `ffdraft.scoring` closely. Note FG 50-59 and 60+ are both worth 5
points in this league -- distance past 50 carries no extra value, which is
unusual and easy to get wrong.
"""

from collections.abc import Mapping
from dataclasses import fields

import polars as pl

from ..league import KICKING, KickingRules

# Maps a stat column name to the KickingRules field that prices it. Two
# many-to-one collapses live here, both intentional:
#   - the 0-19/20-29/30-39 distance buckets all collapse onto fg_0_39, since
#     this league prices the whole 0-39 range as a single 3-point tier.
#   - fg_blocked collapses onto fg_missed: upstream tracks blocked attempts
#     as a bucket separate from fg_missed (fg_att = fg_made + fg_missed +
#     fg_blocked), but a blocked kick is a failed attempt, matching this
#     league's "FG missed" rule and ESPN's own convention.
KICKING_STAT_TO_RULE: dict[str, str] = {
    "pat_made": "pat_made",
    "fg_made_0_19": "fg_0_39",
    "fg_made_20_29": "fg_0_39",
    "fg_made_30_39": "fg_0_39",
    "fg_made_40_49": "fg_40_49",
    "fg_made_50_59": "fg_50_59",
    "fg_made_60_": "fg_60_plus",
    "fg_missed": "fg_missed",
    "fg_blocked": "fg_missed",
}


def validate_stat_to_rule(mapping: Mapping[str, str]) -> None:
    """Raise if any mapped value is not a real KickingRules field name."""
    valid_fields = {f.name for f in fields(KickingRules)}
    invalid = {stat: rule for stat, rule in mapping.items() if rule not in valid_fields}
    if invalid:
        raise ValueError(
            f"KICKING_STAT_TO_RULE references unknown KickingRules field(s): {invalid}. "
            f"Valid fields are: {sorted(valid_fields)}"
        )


validate_stat_to_rule(KICKING_STAT_TO_RULE)


def score_kicking_line(line: Mapping[str, float], rules: KickingRules = KICKING) -> float:
    """Score a single kicker stat line. Missing or null stats count as zero."""
    total = 0.0
    for stat, rule_name in KICKING_STAT_TO_RULE.items():
        value = line.get(stat)
        if value is None:
            continue
        total += float(value) * getattr(rules, rule_name)
    return round(total, 2)


def add_kicking_points(df: pl.DataFrame, rules: KickingRules = KICKING) -> pl.DataFrame:
    """Append a `fantasy_points` column. Stat columns absent from the frame
    contribute nothing."""
    terms = [
        (pl.col(stat).fill_null(0.0) * getattr(rules, rule_name))
        for stat, rule_name in KICKING_STAT_TO_RULE.items()
        if stat in df.columns
    ]
    if not terms:
        return df.with_columns(pl.lit(0.0).alias("fantasy_points"))
    return df.with_columns(pl.sum_horizontal(terms).round(2).alias("fantasy_points"))
