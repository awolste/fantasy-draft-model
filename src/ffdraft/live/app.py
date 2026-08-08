"""Draft-day UI.

    .venv/bin/streamlit run src/ffdraft/live/app.py

Three things this page must always show, because each is a way the tool
could otherwise mislead:

1. **Provenance** -- cache (full budget) or live (reduced budget), and the
   actual rollouts x sims. A number whose origin you cannot see is a number
   you cannot calibrate.
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
from pathlib import Path

import streamlit as st

# Absolute imports, deliberately: Streamlit executes this file as
# `__main__` rather than importing it as `ffdraft.live.app`, so relative
# imports raise "attempted relative import with no known parent package".
from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET, LIVE_BUDGET
from ffdraft.live.cache import (
    RecommendationCache,
    candidates_to_rows,
    recommend,
    state_key,
)
from ffdraft.live.context import live_context
from ffdraft.live.state import DraftBoard
from ffdraft.store import CacheStaleError, fingerprint

CACHE_PATH = Path("data/recommendation_cache.json")


@st.cache_resource
def _context():
    """Fitting the context takes a few seconds; do it once per session, not
    once per pick."""
    return live_context()


@st.cache_resource
def _cache(_fp: str):
    """Returns (cache, status_message). A stale or missing cache is never
    fatal -- the live fallback exists precisely so draft day survives it."""
    if not CACHE_PATH.exists():
        return None, "No recommendation cache found — every pick runs live."
    try:
        return RecommendationCache.load(CACHE_PATH, fingerprint=_fp), None
    except CacheStaleError as exc:
        return None, f"Cache is stale and will not be used: {exc}"
    except Exception as exc:  # never let a malformed cache take the tool down
        return None, f"Cache unreadable, running live: {exc}"


def main() -> None:
    st.set_page_config(page_title="Draft assistant", layout="wide")
    st.title(f"Draft assistant — slot {DRAFT_SLOT} of {N_TEAMS}")
    st.caption(
        "Measured edge over ADP-following: **+2.86pp**, between-season SE **4.15pp**, "
        "beating ADP in 3 of 5 holdout seasons. A decision aid, not an oracle. "
        "Draft structure was measured and does not matter — best available, not a script."
    )

    ctx = _context()
    fp = fingerprint(ctx.rankings, ctx.adp_table)
    cache, cache_note = _cache(fp)
    if cache_note:
        st.warning(cache_note)

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
        rows = cache.get(key) if cache is not None else None

        if rows is not None:
            st.success(
                f"**cache hit** — full budget ({FULL_BUDGET.label}), "
                f"precomputed. Probabilities are ±~0.5pp for a board sharing this key."
            )
        else:
            t0 = time.perf_counter()
            with st.spinner(f"Cache miss — simulating live ({LIVE_BUDGET.label})…"):
                rows = candidates_to_rows(recommend(board.to_draft_state(), ctx, LIVE_BUDGET))
            st.warning(
                f"**live** — reduced budget ({LIVE_BUDGET.label}), "
                f"{time.perf_counter() - t0:.1f}s. Wider error bars than a cached pick."
            )

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
                    "tied w/ leader": r["indistinguishable_from_leader"],
                }
                for r in rows
            ],
            hide_index=True,
            width="stretch",
        )


main()
