"""Exercises the exact code path `live/app.py` runs at one of our picks.

The Streamlit page itself is verified by loading it in a browser, but that
only proves it renders. This pins the branch that produces the numbers: the
cache-miss -> live-recommend -> rows path, at a real draft state, on the
live 2026 pool.

Kept fast by using a tiny budget. The *shape* of the output is what matters
here; `LIVE_BUDGET` is timed separately in `scripts/measure_live_budget.py`.
"""

from __future__ import annotations

from ffdraft.league import DRAFT_SLOT
from ffdraft.live.budget import Budget
from ffdraft.live.cache import candidates_to_rows, recommend, state_key
from ffdraft.live.context import live_context
from ffdraft.live.state import DraftBoard

TINY = Budget(n_candidates=4, n_rollouts=2, n_sims_per_rollout=25, seed=3)


def _board_at_our_first_pick(ctx) -> DraftBoard:
    board = DraftBoard()
    for pid in sorted(ctx.pool, key=lambda p: ctx.pool[p].rank)[: DRAFT_SLOT - 1]:
        board.record(pid, ctx.pool[pid].position)
    return board


def test_app_recommendation_path_produces_usable_rows():
    ctx = live_context()
    board = _board_at_our_first_pick(ctx)
    assert board.is_our_turn, "board should be on our pick before recommending"

    rows = candidates_to_rows(recommend(board.to_draft_state(), ctx, TINY))

    assert rows, "recommendation returned no candidates"
    required = {
        "player_id", "name", "position", "championship_probability",
        "standard_error", "gap_from_leader_pp", "indistinguishable_from_leader",
    }
    for row in rows:
        assert required <= set(row), f"missing {required - set(row)}"

    # The leader must be first and have zero gap -- the UI relies on this
    # ordering rather than re-sorting.
    assert rows[0]["gap_from_leader_pp"] == 0
    assert all(r["gap_from_leader_pp"] >= 0 for r in rows)
    assert all(0.0 <= r["championship_probability"] <= 1.0 for r in rows)


def test_app_state_key_matches_what_the_board_reports():
    """The UI builds the cache key from board properties. If those drift
    from the DraftState the recommender sees, the cache would be keyed on
    one board and answered for another."""
    ctx = live_context()
    board = _board_at_our_first_pick(ctx)
    state = board.to_draft_state()

    key = state_key(
        board.next_overall_pick, board.our_roster_counts, board.drafted_ids, ctx.pool
    )
    assert key.overall_pick == state.next_overall_pick
    assert set(board.drafted_ids) == set(state.drafted_ids)
