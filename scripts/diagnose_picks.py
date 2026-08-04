"""Diagnostic: side-by-side of what the engine drafts vs. what ADP-following
drafts, at our own picks only, on the 2024 holdout.

Motivation (see docs/HANDOFF.md 7b): the engine loses the 2024 backtest to
ADP-following by 13.5pp, and three hypotheses have been eliminated without
denting the gap. Every aggregate has been examined; the picks themselves
have not. This prints them.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import statistics

from ffdraft.backtest import _weekly_lookup, fit_holdout_context, run_one_draft
from ffdraft.league import DRAFT_SLOT, N_TEAMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    ctx = fit_holdout_context()
    weekly = _weekly_lookup(ctx.weekly_holdout, max_week=17)

    def real_ppg(pid: str) -> float | None:
        weeks = weekly.get(pid)
        if not weeks:
            return None
        return statistics.fmean(weeks.values())

    for seed in args.seeds:
        states = {c: run_one_draft(seed, DRAFT_SLOT, c, ctx) for c in ("engine", "adp")}
        ours = {
            c: [p for p in s.picks if p.team == DRAFT_SLOT]
            for c, s in states.items()
        }
        print(f"\n{'=' * 100}\nseed {seed} -- our picks (slot {DRAFT_SLOT} of {N_TEAMS})\n{'=' * 100}")
        print(f"{'rd':>3}  {'ENGINE':<44}{'ADP':<44}")
        engine_ppg: list[float] = []
        adp_ppg: list[float] = []
        for rnd, (e, a) in enumerate(zip(ours["engine"], ours["adp"]), start=1):

            def fmt(pick, bucket: list[float]) -> str:
                p = ctx.pool[pick.player_id]
                ppg = real_ppg(pick.player_id)
                if ppg is not None:
                    bucket.append(ppg)
                adp = ctx.adp_by_player_id.get(pick.player_id, float("nan"))
                shown = f"{ppg:5.1f}" if ppg is not None else "  n/a"
                return f"{p.name[:20]:<21}{p.position:<4}adp{adp:6.1f} {shown}  "

            print(f"{rnd:>3}  {fmt(e, engine_ppg):<44}{fmt(a, adp_ppg):<44}")
        print(
            f"     mean real 2024 ppg over picks:  engine {statistics.fmean(engine_ppg):5.2f}"
            f"   adp {statistics.fmean(adp_ppg):5.2f}"
        )
        for label, picks in ours.items():
            counts: dict[str, int] = {}
            for p in picks:
                counts[p.position] = counts.get(p.position, 0) + 1
            print(f"     {label:<7} positions: {counts}")

    # Aggregate: how far ahead of ADP does each policy reach? A reach of +40
    # means the player would very likely still have been there 40 picks later,
    # so those 40 picks of value were burned.
    print(f"\n{'=' * 100}\nreach = (player ADP) - (overall pick used on him); positive = reached\n{'=' * 100}")
    reaches: dict[str, list[float]] = {"engine": [], "adp": []}
    by_pos: dict[str, dict[str, list[float]]] = {"engine": {}, "adp": {}}
    for seed in args.seeds:
        for contender in ("engine", "adp"):
            state = run_one_draft(seed, DRAFT_SLOT, contender, ctx)
            for p in state.picks:
                if p.team != DRAFT_SLOT:
                    continue
                adp = ctx.adp_by_player_id.get(p.player_id)
                if adp is None:
                    continue
                r = adp - p.overall_pick
                reaches[contender].append(r)
                by_pos[contender].setdefault(p.position, []).append(r)
    for contender in ("engine", "adp"):
        rs = reaches[contender]
        big = [r for r in rs if r >= 30]
        print(
            f"{contender:<7} mean reach {statistics.fmean(rs):+6.1f}  "
            f"median {statistics.median(rs):+6.1f}  picks reached >=30: {len(big)}/{len(rs)}"
        )
        for pos in sorted(by_pos[contender]):
            v = by_pos[contender][pos]
            print(f"          {pos:<4} n={len(v):<3} mean reach {statistics.fmean(v):+6.1f}")


if __name__ == "__main__":
    main()
