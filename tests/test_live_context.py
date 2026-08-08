"""Tests for `ffdraft.live.context`.

The live context is the piece that was missing when Stage 4's plan was
written: every other Stage 4 component had been sketched against
`backtest.fit_holdout_context`, which builds a **2024** draft board. Using
it on draft day would recommend 2024 players for a 2026 draft and raise no
error -- the exact failure mode HANDOFF section 8 warns about.
"""

from __future__ import annotations

import polars as pl

from ffdraft.live.context import DraftContext, live_context


def test_draft_context_exposes_the_names_recommend_needs():
    """`live.cache.recommend` reads these five attributes. `HoldoutContext`
    names two of them differently (`adp_holdout`/`rankings_holdout`), so the
    shared shape has to be explicit rather than assumed."""
    ctx = DraftContext(
        pool={},
        opponent_model=object(),
        replacement_by_position={},
        adp_table=pl.DataFrame(),
        rankings=pl.DataFrame(),
    )
    for attr in ("pool", "opponent_model", "replacement_by_position", "adp_table", "rankings"):
        assert hasattr(ctx, attr), attr


def test_live_context_is_built_from_2026_data_not_a_holdout_season():
    ctx = live_context()

    # The 2026 ADP table is the live one, not a historical slice.
    assert "season" not in ctx.adp_table.columns or set(
        ctx.adp_table["season"].unique().to_list()
    ) == {2026}

    # A real pool, and a real replacement level for every position we start.
    assert len(ctx.pool) > 300
    for position in ("QB", "RB", "WR", "TE", "K", "DST"):
        assert position in ctx.replacement_by_position

    # Sanity: the pool is 2026 players. Rankings come from rankings_2026,
    # so pool size should track it rather than a holdout season's board.
    assert ctx.rankings.height > 400


def test_pool_rank_is_an_overall_rank_and_is_unique():
    """`PlayerDistribution.rank` is the **overall** board rank, not a
    within-position one (within-position ordering is derived separately in
    `_matched_skill_players`). `live.cache.state_key` relies on this: "top
    60 by rank" must mean the top 60 players on the board, not the top 60
    of each position. Uniqueness matters for the same reason -- ties would
    make the key's `sorted(...)[:60]` cut non-deterministic.
    """
    ctx = live_context()
    ranks = [p.rank for p in ctx.pool.values()]

    assert len(ranks) == len(set(ranks)), "overall ranks must be unique across the pool"
    assert min(ranks) >= 1

    # Overall, not per-position: only one position can own rank 1.
    positions_holding_rank_1 = {p.position for p in ctx.pool.values() if p.rank == 1}
    assert len(positions_holding_rank_1) == 1

    # All six startable positions are represented.
    assert {p.position for p in ctx.pool.values()} >= {"QB", "RB", "WR", "TE", "K", "DST"}
