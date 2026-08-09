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

Opponents draft from the **fitted opponent model** by default (`--opponents
model`), which carries this league's measured reach/fall behaviour from
eight seasons of drafts. `--opponents adp` gives the old strict-ADP
baseline.

The difference matters and was found the hard way. Under strict ADP nobody
ever reaches, so every player falls to exactly his ADP slot -- which handed
us Justin Jefferson at pick 13 (ADP 13.3), a player who goes in round 1 of
real mocks. A strict-ADP board is systematically generous, and any roster
built against it looks better than one you could actually draft.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ffdraft.draft.rollout import pool_adp_lookup
from ffdraft.league import DRAFT_SLOT, FLEX_ELIGIBLE, N_TEAMS, STARTERS
from ffdraft.live.budget import LIVE_BUDGET, Budget
from ffdraft.live.cache import candidates_to_rows, recommend
from ffdraft.live.context import live_context
from ffdraft.live.state import DraftBoard
from ffdraft.models.opponent import AvailablePlayer, sample_pick
from ffdraft.live.tiebreak import best_uncapped_available, choose_from_tied, filter_capped
from ffdraft.live.survival import (
    DEFAULT_N_SIMS,
    annotate_rows,
    picks_until_our_next_turn,
    survival_probabilities,
)

NO_ADP = 10_000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--fast", action="store_true",
                        help="tiny budget: for wiring checks, not for results")
    parser.add_argument("--debug-caps", action="store_true")
    parser.add_argument(
        "--opponents", choices=("model", "adp"), default="model",
        help="'model' = fitted opponent model (realistic reach/fall); "
             "'adp' = strict lowest-ADP (generous, nobody reaches)",
    )
    args = parser.parse_args()

    budget = Budget(4, 2, 25, seed=args.seed) if args.fast else LIVE_BUDGET
    ctx = live_context()
    adp = pool_adp_lookup(ctx.pool, ctx.adp_table, ctx.rankings)[0]
    board = DraftBoard()
    log: list[dict] = []
    opp_rng = np.random.default_rng(args.seed)
    opp_counts: dict[int, dict[str, int]] = {}
    t_start = time.perf_counter()
    print(f"opponents: {args.opponents}\n", flush=True)

    while not board.is_complete:
        if not board.is_our_turn:
            avail = [p for p in ctx.pool if p not in board.drafted_ids]
            if args.opponents == "adp":
                pid = min(avail, key=lambda p: (adp.get(p, NO_ADP), ctx.pool[p].rank))
            else:
                team = board.team_on_clock
                counts = opp_counts.setdefault(team, {})
                pid = sample_pick(
                    ctx.opponent_model,
                    f"slot_{team}",
                    [
                        AvailablePlayer(
                            player_id=p,
                            position=ctx.pool[p].position,
                            adp=adp.get(p, float(ctx.pool[p].rank)),
                        )
                        for p in avail
                    ],
                    counts,
                    opp_rng,
                )
                counts[ctx.pool[pid].position] = counts.get(ctx.pool[pid].position, 0) + 1
            board.record(pid, ctx.pool[pid].position)
            continue

        overall = board.next_overall_pick
        t0 = time.perf_counter()
        rows = candidates_to_rows(recommend(board.to_draft_state(), ctx, budget))
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

        before = len(rows)
        rows = filter_capped(rows, board.our_roster_counts)
        if not rows:
            # Every recommended candidate is at a capped position -- common
            # late, since the value function's tilt fills the candidate list
            # with QBs and kickers. Look past it to the pool.
            fb = best_uncapped_available(
                ctx.pool, board.drafted_ids, board.our_roster_counts, adp
            )
            if fb is None:
                raise RuntimeError("no uncapped player available anywhere in the pool")
            rows = [fb]
        if args.debug_caps:
            print(f"    [caps] counts={board.our_roster_counts} "
                  f"rows {before}->{len(rows)} "
                  f"positions={sorted({r['position'] for r in rows})}", flush=True)
        chosen = choose_from_tied(
            rows,
            board.our_roster_counts,
            rounds_left=board.rounds - board.current_round + 1,
        )
        n_tied = sum(1 for r in rows if r["indistinguishable_from_leader"])
        log.append(
            {
                "round": board.current_round,
                "overall": overall,
                "name": chosen["name"],
                "pos": chosen["position"],
                "adp": adp.get(chosen["player_id"]),
                "champ": (
                    None if chosen["championship_probability"] is None
                    else chosen["championship_probability"] * 100
                ),
                "fallback": chosen.get("uncapped_fallback", False),
                "surv": chosen.get("p_survive"),
                "wait": chosen.get("wait_cost_pp"),
                "was_leader": chosen["player_id"] == rows[0]["player_id"],
                "n_tied": n_tied,
                "secs": time.perf_counter() - t0,
            }
        )
        champ_s = (
            "   n/a" if chosen["championship_probability"] is None
            else f"{chosen['championship_probability'] * 100:5.2f}%"
        )
        note = ""
        if chosen.get("uncapped_fallback"):
            note = "(CAPPED OUT — best uncapped by ADP, not simulated)"
        elif not log[-1]["was_leader"]:
            note = f"(tie-break from {n_tied})"
        print(
            f"  R{board.current_round:<2} pick {overall:<4}{chosen['name'][:22]:<24}"
            f"{chosen['position']:<4} adp {adp.get(chosen['player_id'], float('nan')):>6.1f} "
            f"champ {champ_s} {note}",
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
