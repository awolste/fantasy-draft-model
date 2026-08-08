"""The assembled model context for a **live 2026 draft**.

## Why this exists

Every model component already had a live path -- `build_player_pool()`
defaults to `rankings_2026`, `replacement_by_position()` reads the live
cache -- but they had never been assembled into one object. The only
assembled context in the project was `backtest.fit_holdout_context`, which
builds a **2024** board for backtesting.

Reaching for that on draft day would produce a tool that recommends 2024
players in 2026 and raises no error: confident, plausible, wrong. That is
the failure mode `docs/HANDOFF.md` section 8 identifies as this project's
dominant risk, so the live context is a named, tested thing rather than a
default someone has to remember not to override.

## Relationship to `HoldoutContext`

`DraftContext` is the shape `live.cache.recommend` consumes. It is not the
same shape as `HoldoutContext`, which names two of these fields differently
(`adp_holdout` / `rankings_holdout`). `from_holdout` adapts one to the
other so tests and measurement scripts can drive the live code paths with a
cheap historical context, without either type pretending to be the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..models.defense import dst_distribution
from ..models.distribution import PlayerDistribution, build_player_pool
from ..models.opponent import TRAINING_SEASONS, build_training_set, fit_opponent_model
from ..models.replacement import replacement_by_position as _replacement_by_position
from ..store import read

LIVE_SEASON = 2026


@dataclass(frozen=True)
class DraftContext:
    """Everything `recommend_pick` needs, under one set of names."""

    pool: Mapping[str, PlayerDistribution]
    opponent_model: Any
    replacement_by_position: Mapping[str, Any]
    adp_table: pl.DataFrame
    rankings: pl.DataFrame

    @classmethod
    def from_holdout(cls, holdout: Any) -> DraftContext:
        """Adapt a `backtest.HoldoutContext` so the live code paths can be
        exercised against a historical season -- used by tests and by
        `scripts/measure_live_budget.py`, never on draft day."""
        return cls(
            pool=holdout.pool,
            opponent_model=holdout.opponent_model,
            replacement_by_position=holdout.replacement_by_position,
            adp_table=holdout.adp_holdout,
            rankings=holdout.rankings_holdout,
        )


def live_context(season: int = LIVE_SEASON) -> DraftContext:
    """Assemble the real 2026 draft context.

    Fit on **all** available history (`TRAINING_SEASONS`) -- unlike the
    backtest, there is no holdout to respect here, because the season being
    predicted has not happened yet.
    """
    rankings = read(f"rankings_{season}")
    adp_table = read(f"adp_{season}")

    pool = build_player_pool(rankings=rankings)

    replacement = dict(_replacement_by_position())
    replacement["DST"] = dst_distribution()

    training, _report = build_training_set(
        read("league_drafts"),
        read("league_managers"),
        read("id_crosswalk"),
        read("adp_history"),
        seasons=TRAINING_SEASONS,
    )
    model = fit_opponent_model(training)

    return DraftContext(
        pool=pool,
        opponent_model=model,
        replacement_by_position=replacement,
        adp_table=adp_table,
        rankings=rankings,
    )
