"""Where does the QB/TE tilt actually come from? (HANDOFF open item 3)

The tilt is measured (7c extension: 2RB_1WR - TE_early = +4.85pp, 3.6 sigma)
but never *localised*. `draft.value` is a composition of several pieces and
any of them could be responsible:

1. **Replacement levels** (`models/replacement.py`) -- TE's 8.2 vs WR's 9.9
   hands every TE a 1.7-point head start before anything else happens.
2. **Projected means** (`models/distribution.py`) -- elite TE/QB means could
   simply be too high relative to WR/RB.
3. **Mean-only valuation** -- `value.py` reads `.mean` and nothing else, so
   two players with equal means are equal to it even if one has twice the
   weekly upside. Championship probability is convex in points, so this
   would systematically favour high-floor QB/TE over high-ceiling WR.

These have different fixes and one of them (1) was already measured as *not*
helping when attacked bluntly (section 10c: adding a delta to QB replacement
lowers the championship rate monotonically). So localise before changing
anything.

Prints, for an empty roster on a given season's pool: the top 40 by
`draft.value`, the top 40 by ADP, the positional mix of each, and a
per-position decomposition of mean / replacement / VOR / spread.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from ffdraft.draft.value import value_available
from ffdraft.sim.lineup import build_replacement_means

N_TOP = 40


def _ctx(season: int | None):
    """Live 2026 context, or a holdout context fit through `season - 1`."""
    if season is None:
        from ffdraft.live.context import live_context

        c = live_context()
        return c.pool, c.replacement_by_position, c.adp_table, c.rankings, None
    from ffdraft.backtest import fit_holdout_context

    c = fit_holdout_context(fit_through_season=season - 1, holdout_season=season)
    return c.pool, c.replacement_by_position, c.adp_holdout, c.rankings_holdout, c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=None,
                    help="holdout season; omit for the live 2026 pool")
    args = ap.parse_args()

    pool, replacement_by_position, adp_table, rankings, _ = _ctx(args.season)
    from ffdraft.draft.rollout import pool_adp_lookup

    adp = pool_adp_lookup(pool, adp_table, rankings)[0]
    replacement_means = build_replacement_means(
        {k: v for k, v in replacement_by_position.items() if k != "DST"},
        replacement_by_position["DST"].mean,
    )

    print("\nreplacement means (ppg):")
    for pos, v in sorted(replacement_means.items(), key=lambda kv: -kv[1]):
        print(f"  {pos:<5}{v:7.2f}")

    values = value_available(list(pool), pool, [], replacement_means)
    by_value = sorted(values, key=lambda v: -v.value)

    print(f"\ntop {N_TOP} by draft.value, empty roster:")
    print(f"{'#':>3}  {'player':<24}{'pos':<5}{'mean':>7}{'repl':>7}{'value':>8}{'adp':>8}")
    for i, v in enumerate(by_value[:N_TOP], 1):
        print(
            f"{i:>3}  {pool[v.player_id].name[:23]:<24}{v.position:<5}"
            f"{v.mean:>7.2f}{replacement_means[v.position]:>7.2f}{v.value:>8.2f}"
            f"{adp.get(v.player_id, float('nan')):>8.1f}"
        )

    ranked_by_adp = sorted(
        (pid for pid in pool if pid in adp), key=lambda pid: adp[pid]
    )[:N_TOP]

    mix_value = Counter(v.position for v in by_value[:N_TOP])
    mix_adp = Counter(pool[pid].position for pid in ranked_by_adp)
    print(f"\npositional mix of the top {N_TOP}:")
    print(f"  draft.value : {dict(sorted(mix_value.items()))}")
    print(f"  ADP         : {dict(sorted(mix_adp.items()))}")

    # Per-position: what the top few look like on each input separately, so
    # a tilt in the means is distinguishable from a tilt in replacement.
    print("\nper-position, top 5 by mean (mean / VOR / weekly sd):")
    for pos in ("QB", "RB", "WR", "TE"):
        at = sorted(
            (p for p in pool.values() if p.position == pos),
            key=lambda p: -float(p.distribution.mean),
        )[:5]
        print(f"  {pos}")
        for p in at:
            sd = float(np.std(p.distribution.sample(np.random.default_rng(0), 4000)))
            vor = float(p.distribution.mean) - replacement_means[pos]
            print(
                f"    {p.name[:22]:<24}{float(p.distribution.mean):>7.2f}"
                f"{vor:>7.2f}{sd:>7.2f}"
            )


    dropoff_table(pool, adp, replacement_means)


def dropoff_table(pool, adp, replacement_means) -> None:
    """Best-available mean at each of our own picks, per position.

    This is the quantity `draft.value` does *not* use. Its baseline is the
    waiver replacement level -- what the slot is worth if never drafted at
    all. But we hold 18 picks and will certainly fill every slot, so the
    real opportunity cost of passing on a player is the best one still
    there at our **next** turn, not the wire.

    Where the two disagree most is exactly where a tilt would come from: a
    position whose top is flat loses little by waiting (small true cost)
    while still showing a large gap over a weak waiver level (large VOR).
    """
    from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
    from ffdraft.draft.rollout import team_for_pick

    our_picks = [
        p
        for p in range(1, DRAFT_ROUNDS * N_TEAMS + 1)
        if team_for_pick(p, N_TEAMS) == DRAFT_SLOT
    ]
    board = sorted((pid for pid in pool if pid in adp), key=lambda pid: adp[pid])

    print("\nbest-available projected mean at each of our picks (ADP board):")
    print(f"{'pick':>5}" + "".join(f"{pos:>8}" for pos in ("QB", "RB", "WR", "TE")))
    best_at: dict[int, dict[str, float]] = {}
    for pick in our_picks[:10]:
        gone = set(board[: pick - 1])
        row = {}
        for pos in ("QB", "RB", "WR", "TE"):
            left = [
                float(pool[pid].distribution.mean)
                for pid in board
                if pid not in gone and pool[pid].position == pos
            ]
            row[pos] = max(left) if left else float("nan")
        best_at[pick] = row
        print(f"{pick:>5}" + "".join(f"{row[pos]:>8.2f}" for pos in ("QB", "RB", "WR", "TE")))

    print("\nthe two competing notions of a pick's worth, at our first three picks:")
    print(f"{'pick':>5}  {'pos':<5}{'best':>8}{'VOR':>8}{'wait to next':>14}{'VOR-wait':>10}")
    for a, b in zip(our_picks[:3], our_picks[1:4]):
        for pos in ("QB", "RB", "WR", "TE"):
            best = best_at[a][pos]
            vor = best - replacement_means[pos]
            wait = best - best_at[b][pos]
            print(
                f"{a:>5}  {pos:<5}{best:>8.2f}{vor:>8.2f}{wait:>14.2f}{vor - wait:>10.2f}"
            )
        print()


if __name__ == "__main__":
    main()
