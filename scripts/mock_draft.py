"""Play a full mock draft: we follow the engine, everyone else follows ADP.

Our pick at each of our 18 turns is the recommender's leader, with ties
broken deliberately rather than by list order:

1. Among candidates flagged `indistinguishable_from_leader` -- i.e. those
   the model genuinely cannot separate -- prefer the highest
   `wait_cost_pp`. That is the survival tie-break: of two players worth the
   same, take the one who will not be there next time.
2. Then prefer a position whose *dedicated starter slots* are still unfilled
   (roster construction), since a player who cannot crack the lineup is
   worth his bench value, not his headline value.

Opponents take the lowest available ADP, full stop. That is not a claim
about how the league drafts -- the fitted opponent model is more realistic
-- it is a clean, deterministic baseline so our own picks are the only
thing varying.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ffdraft.draft.rollout import pool_adp_lookup
from ffdraft.league import DRAFT_SLOT, FLEX_ELIGIBLE, N_TEAMS, STARTERS
from ffdraft.live.budget import LIVE_BUDGET
from ffdraft.live.cache import candidates_to_rows, recommend
from ffdraft.live.context import live_context
from ffdraft.live.state import DraftBoard
from ffdraft.live.survival import (
    DEFAULT_N_SIMS,
    annotate_rows,
    picks_until_our_next_turn,
    survival_probabilities,
)

NO_ADP = 10_000.0


def _needs_starter(position: str, counts: dict[str, int]) -> bool:
    """True if `position`'s own dedicated starter slots are not yet full."""
    return counts.get(position, 0) < STARTERS.get(position, 0)


def _pick_ours(rows: list[dict], counts: dict[str, int]) -> dict:
    """Leader, with ties broken on wait cost then roster need."""
    tied = [r for r in rows if r["indistinguishable_from_leader"]] or rows[:1]
    if len(tied) == 1:
        return tied[0]
    return sorted(
        tied,
        key=lambda r: (
            -(r.get("wait_cost_pp") or 0.0),
            not _needs_starter(r["position"], counts),
            -r["championship_probability"],
        ),
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    ctx = live_context()
    adp = pool_adp_lookup(ctx.pool, ctx.adp_table, ctx.rankings)[0]
    board = DraftBoard()
    log: list[dict] = []
    t_start = time.perf_counter()

    while not board.is_complete:
        if not board.is_our_turn:
            avail = [p for p in ctx.pool if p not in board.drafted_ids]
            pid = min(avail, key=lambda p: (adp.get(p, NO_ADP), ctx.pool[p].rank))
            board.record(pid, ctx.pool[pid].position)
            continue

        overall = board.next_overall_pick
        t0 = time.perf_counter()
        rows = candidates_to_rows(recommend(board.to_draft_state(), ctx, LIVE_BUDGET))
        n_picks = picks_until_our_next_turn(
            overall, board.n_teams, board.our_team, board.total_picks
        )
        if n_picks is None:
            rows = annotate_rows(rows, {}, has_next_turn=False)
        else:
            surv = survival_probabilities(
                candidate_ids=[r["player_id"] for r in rows],
                pool=ctx.pool,
                adp_by_player_id=adp,
                drafted_ids=board.drafted_ids,
                model=ctx.opponent_model,
                n_picks=n_picks,
                n_sims=DEFAULT_N_SIMS,
                rng=np.random.default_rng(args.seed + overall),
                n_teams=board.n_teams,
                first_pick=overall + 1,
            )
            rows = annotate_rows(rows, surv, has_next_turn=True)

        chosen = _pick_ours(rows, board.our_roster_counts)
        n_tied = sum(1 for r in rows if r["indistinguishable_from_leader"])
        log.append(
            {
                "round": board.current_round,
                "overall": overall,
                "name": chosen["name"],
                "pos": chosen["position"],
                "adp": adp.get(chosen["player_id"]),
                "champ": chosen["championship_probability"] * 100,
                "surv": chosen.get("p_survive"),
                "wait": chosen.get("wait_cost_pp"),
                "was_leader": chosen["player_id"] == rows[0]["player_id"],
                "n_tied": n_tied,
                "secs": time.perf_counter() - t0,
            }
        )
        print(
            f"  R{board.current_round:<2} pick {overall:<4}{chosen['name'][:22]:<24}"
            f"{chosen['position']:<4} adp {adp.get(chosen['player_id'], float('nan')):>6.1f} "
            f"champ {chosen['championship_probability'] * 100:5.2f}% "
            f"{'(tie-break from ' + str(n_tied) + ')' if not log[-1]['was_leader'] else ''}",
            flush=True,
        )
        board.record(chosen["player_id"], chosen["position"])

    print(f"\ndraft complete in {time.perf_counter() - t_start:.0f}s\n")
    print("=" * 78)
    print(f"OUR TEAM — slot {DRAFT_SLOT} of {N_TEAMS}")
    print("=" * 78)
    print(f"{'rd':>3}  {'pick':>4}  {'player':<24}{'pos':<5}{'adp':>7}{'proj ppg':>10}{'surv':>7}")
    for e in log:
        pid_name = e["name"][:23]
        proj = next(
            float(p.distribution.mean) for p in ctx.pool.values() if p.name == e["name"]
        )
        surv = "—" if e["surv"] is None else f"{e['surv'] * 100:.0f}%"
        adp_s = "—" if e["adp"] is None else f"{e['adp']:.1f}"
        print(f"{e['round']:>3}  {e['overall']:>4}  {pid_name:<24}{e['pos']:<5}{adp_s:>7}{proj:>10.2f}{surv:>7}")

    counts = board.our_roster_counts
    print(f"\nposition counts: {dict(sorted(counts.items()))}")
    print(f"starters required: {dict(sorted(STARTERS.items()))} (FLEX from {sorted(FLEX_ELIGIBLE)})")
    unfilled = [p for p in STARTERS if p != "FLEX" and counts.get(p, 0) < STARTERS[p]]
    print(f"unfilled dedicated starter slots: {unfilled or 'none'}")
    n_tb = sum(1 for e in log if not e["was_leader"])
    print(f"picks decided by tie-break rather than the outright leader: {n_tb}/{len(log)}")
    print(f"mean recommendation time: {sum(e['secs'] for e in log) / len(log):.1f}s")


if __name__ == "__main__":
    main()
