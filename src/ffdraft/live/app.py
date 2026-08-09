"""Draft-day UI.

    .venv/bin/streamlit run src/ffdraft/live/app.py

**Live-only.** Precompute was built, measured, and dropped: sampling 300
drafts found 203/248/269 distinct states at our first three picks, so the
8 most common covered only 17.0%/11.3%/9.0% of drafts. Nearly every state
occurs exactly once -- the space is irreducibly large at the granularity
that changes a recommendation. ~78 minutes of precompute bought roughly one
pick in eight; a live recommendation costs 13-36s and covers all of them.
See HANDOFF section 11.

Three things this page must always show, because each is a way the tool
could otherwise mislead:

1. **Provenance** -- the actual rollouts x sims behind the number, the
   elapsed time, and whether it was reused from this session. A number
   whose origin you cannot see is a number you cannot calibrate.
2. **Ties** -- candidates statistically indistinguishable from the leader
   are called out, not silently ranked. Presenting a strict order over a
   tie is the single most misleading thing this tool could do.
3. **The standing caveat** -- measured edge +2.86pp against a
   between-season SE of 4.15pp, 3 of 5 holdout seasons. A decision aid, not
   an oracle.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`;
refers to teams by draft slot only.
"""

from __future__ import annotations

import time

import numpy as np

import streamlit as st

# Absolute imports, deliberately: Streamlit executes this file as
# `__main__` rather than importing it as `ffdraft.live.app`, so relative
# imports raise "attempted relative import with no known parent package".
from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET, LIVE_BUDGET
from ffdraft.live.cache import candidates_to_rows, recommend, state_key
from ffdraft.live.context import live_context
from ffdraft.live.state import DraftBoard
from ffdraft.live.tiebreak import best_uncapped_available, filter_capped
from ffdraft.live.survival import (
    DEFAULT_N_SIMS,
    annotate_rows,
    picks_until_our_next_turn,
    survival_probabilities,
)


@st.cache_resource
def _adp_lookup(_ctx):
    """pool id -> 2026 ADP, resolved once per session."""
    from ffdraft.draft.rollout import pool_adp_lookup

    return pool_adp_lookup(_ctx.pool, _ctx.adp_table, _ctx.rankings)[0]


@st.cache_resource
def _context():
    """Fitting the context takes a few seconds; do it once per session, not
    once per pick."""
    return live_context()


def _memoized_recommendation(key, board, ctx):
    """Compute once per draft state, per session.

    Streamlit re-executes this whole script on every interaction. Without a
    memo, clicking Undo or touching the selectbox after a recommendation
    appeared would re-run the simulation -- 35.7s at our first pick, on the
    clock. Keyed by `state_key`, so a genuinely new board still costs
    compute and a rerun at an unchanged board is instant.
    """
    memo = st.session_state.setdefault("rec_memo", {})
    if key in memo:
        return memo[key], True
    t0 = time.perf_counter()
    with st.spinner(f"Simulating ({LIVE_BUDGET.label})…"):
        rows = candidates_to_rows(recommend(board.to_draft_state(), ctx, LIVE_BUDGET))

        # Survival is a separate, much cheaper simulation: only the picks
        # between now and our next turn, with no season sims behind them.
        n_picks = picks_until_our_next_turn(
            board.next_overall_pick, board.n_teams, board.our_team, board.total_picks
        )
        if n_picks is None:
            rows = annotate_rows(rows, {}, has_next_turn=False)
        else:
            adp_lookup = _adp_lookup(ctx)
            survival = survival_probabilities(
                candidate_ids=[r["player_id"] for r in rows],
                pool=ctx.pool,
                adp_by_player_id=adp_lookup,
                drafted_ids=board.drafted_ids,
                model=ctx.opponent_model,
                n_picks=n_picks,
                n_sims=DEFAULT_N_SIMS,
                rng=np.random.default_rng(board.next_overall_pick),
                n_teams=board.n_teams,
                first_pick=board.next_overall_pick + 1,
            )
            rows = annotate_rows(rows, survival, has_next_turn=True)
    memo[key] = (rows, time.perf_counter() - t0)
    return memo[key], False


def main() -> None:
    st.set_page_config(page_title="Draft assistant", layout="wide")
    st.title(f"Draft assistant — slot {DRAFT_SLOT} of {N_TEAMS}")
    st.caption(
        "Measured edge over ADP-following: **+2.86pp**, between-season SE **4.15pp**, "
        "beating ADP in 3 of 5 holdout seasons. A decision aid, not an oracle. "
        "Draft structure was measured and does not matter — best available, not a script."
    )

    ctx = _context()

    if "board" not in st.session_state:
        st.session_state.board = DraftBoard()
    board: DraftBoard = st.session_state.board

    left, right = st.columns([1, 2])

    with left:
        st.subheader(
            f"Pick {board.next_overall_pick} · round {board.current_round} · "
            f"team {board.team_on_clock}"
        )
        undrafted = {
            pid: p for pid, p in ctx.pool.items() if pid not in board.drafted_ids
        }
        labels = {
            f"{p.name} ({p.position}) #{p.rank}": pid for pid, p in undrafted.items()
        }
        choice = st.selectbox(
            "Player taken",
            sorted(labels, key=lambda label: ctx.pool[labels[label]].rank),
            index=None,
            key=f"pick_{board.next_overall_pick}",
            disabled=board.is_complete,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Record", disabled=choice is None, width="stretch"):
                pid = labels[choice]
                board.record(pid, ctx.pool[pid].position)
                st.rerun()
        with c2:
            if st.button("Undo", disabled=not board.picks, width="stretch"):
                board.undo()
                st.rerun()

        if board.picks:
            st.caption("Last 5 picks")
            st.dataframe(
                [
                    {
                        "pick": p.overall_pick,
                        "team": p.team,
                        "player": ctx.pool[p.player_id].name,
                        "pos": p.position,
                    }
                    for p in board.picks[-5:][::-1]
                ],
                hide_index=True,
                width="stretch",
            )
        st.caption(f"Our roster: {board.our_roster_counts or '(empty)'}")

    with right:
        if board.is_complete:
            st.success("Draft complete.")
            return
        if not board.is_our_turn:
            remaining = next(
                (n for n in board.our_pick_numbers if n >= board.next_overall_pick),
                None,
            )
            st.info(
                f"Team {board.team_on_clock} is on the clock. "
                f"Our next pick: overall {remaining}."
            )
            return

        st.subheader("Recommendation")
        key = state_key(
            board.next_overall_pick, board.our_roster_counts, board.drafted_ids, ctx.pool
        )
        (rows, elapsed), from_memo = _memoized_recommendation(key, board, ctx)
        st.caption(
            f"**live** · budget {LIVE_BUDGET.label} · {elapsed:.1f}s"
            + (" · reused from this session (board unchanged)" if from_memo else "")
            + f" · full budget for reference is {FULL_BUDGET.label}"
        )

        capped_out = filter_capped(rows, board.our_roster_counts)
        if capped_out:
            rows = capped_out
        else:
            fb = best_uncapped_available(
                ctx.pool, board.drafted_ids, board.our_roster_counts, _adp_lookup(ctx)
            )
            st.warning(
                "Every simulated candidate is at a position you already have "
                "enough of (QB≤2, K≤1). Showing the best uncapped player by ADP "
                "instead — he was **not** simulated, so he has no title equity."
            )
            rows = [fb] if fb else rows
        tied = [r for r in rows if r["indistinguishable_from_leader"]]
        if len(tied) > 1:
            names = ", ".join(r["name"] for r in tied)
            st.info(
                f"**{len(tied)} candidates are statistically indistinguishable "
                f"from the leader**: {names}. Treat them as equivalent and use "
                "your own judgement — need, injury news, bye weeks."
            )

        st.dataframe(
            [
                {
                    "player": r["name"],
                    "pos": r["position"],
                    "title %": round(r["championship_probability"] * 100, 2),
                    "± SE": round(r["standard_error"] * 100, 2),
                    "gap (pp)": round(r["gap_from_leader_pp"], 2),
                    "survives %": (
                        None if r.get("p_survive") is None
                        else round(r["p_survive"] * 100)
                    ),
                    "wait cost (pp)": (
                        None if r.get("wait_cost_pp") is None
                        else round(r["wait_cost_pp"], 2)
                    ),
                    "tied w/ leader": r["indistinguishable_from_leader"],
                }
                for r in rows
            ],
            hide_index=True,
            width="stretch",
        )


main()
