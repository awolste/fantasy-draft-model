"""Measure which recommendation budget fits inside a ~60s live pick.

The Stage 4 spec makes this a validation-gate requirement: LIVE_BUDGET must
be measured on this machine, not guessed. Runs against the **live 2026
context**, because that is the pool the tool will actually face -- a
holdout season has a different pool size and would mis-time the result.

Prints the leader and its standard error at each budget, so the accuracy
cost of going cheap is visible rather than implied. If the leader changes
versus FULL_BUDGET, that is the finding, and it belongs in the comment
above LIVE_BUDGET.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`.
"""

from __future__ import annotations

import time

from ffdraft.draft.recommender import recommend_pick
from ffdraft.draft.rollout import DraftState, Pick, team_for_pick
from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET, Budget
from ffdraft.live.context import live_context

CANDIDATE_BUDGETS = [
    Budget(10, 8, 200, seed=1),
    Budget(10, 12, 250, seed=1),
    Budget(12, 16, 300, seed=1),
    Budget(15, 20, 400, seed=1),
    Budget(15, 30, 500, seed=1),
]


def _state_at_our_first_pick(ctx) -> DraftState:
    """Advance the board to overall pick 8 by giving teams 1-7 the top of
    the board. `recommend_pick` refuses to answer when it is not our turn
    (by design), so an empty board is not a valid input here -- an earlier
    version of this script passed one and got a loud, correct ValueError."""
    taken = sorted(ctx.pool, key=lambda pid: ctx.pool[pid].rank)[: DRAFT_SLOT - 1]
    picks = [
        Pick(
            overall_pick=i + 1,
            team=team_for_pick(i + 1, N_TEAMS),
            player_id=pid,
            position=ctx.pool[pid].position,
        )
        for i, pid in enumerate(taken)
    ]
    return DraftState.from_picks(picks, n_teams=N_TEAMS, rounds=DRAFT_ROUNDS)


def main() -> None:
    ctx = live_context()
    state = _state_at_our_first_pick(ctx)
    assert state.team_on_clock == DRAFT_SLOT, state.team_on_clock

    print(f"pool: {len(ctx.pool)} players | measuring at overall pick {state.next_overall_pick}\n")
    print(f"{'budget':<16}{'work':>12}{'elapsed':>10}{'leader SE':>12}  leader")
    for b in [*CANDIDATE_BUDGETS, FULL_BUDGET]:
        t0 = time.perf_counter()
        res = recommend_pick(
            state,
            ctx.pool,
            ctx.opponent_model,
            ctx.replacement_by_position,
            seed=b.seed,
            our_team=DRAFT_SLOT,
            n_candidates=b.n_candidates,
            n_rollouts=b.n_rollouts,
            n_sims_per_rollout=b.n_sims_per_rollout,
            adp_table=ctx.adp_table,
            rankings=ctx.rankings,
        )
        dt = time.perf_counter() - t0
        top = res.candidates[0]
        flag = "  <-- OVER 60s" if dt > 60 else ""
        print(
            f"{b.label:<16}{b.work:>12,}{dt:>9.1f}s{top.standard_error * 100:>11.2f}pp"
            f"  {top.name} ({top.position}){flag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
