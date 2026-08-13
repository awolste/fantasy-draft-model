"""Why does the engine collapse in 2023 and 2024? (HANDOFF open item 1)

Championship rate falls from ~21% (2020-22) to ~7% (2023-24) under both
opponent models, so the cause is not the temperature schedule. This asks
the first question that splits the possibilities in half:

**Is it a drafting failure or a projection failure?**

The engine's whole claim is that its projections beat the market. It
drafts by value over replacement computed from `PlayerDistribution.mean`,
fit through the prior season; the ADP contender drafts by consensus. If
the model's projections predicted realized production *better* than ADP in
2020-22 and *worse* in 2023-24, the collapse needs no explanation in the
drafting logic at all -- the engine was simply trusting a worse signal, and
every rollout, tie-break and value tweak downstream is irrelevant to it.

So per holdout season this measures, over the same set of players:

* how well the model's projected mean ranks realized fantasy points
* how well ADP ranks the same realized points

using Spearman correlation, which asks only about ordering -- the thing a
draft actually consumes. `top-N hit rate` is reported beside it because
correlation over 500 players can be dominated by the deep tail nobody
drafts; the top 60 is roughly the part of the board that decides a season.

No rollouts and no season simulation: this is a property of the inputs,
not of the search, and it runs in about a minute.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from ffdraft.backtest import fit_holdout_context
from ffdraft.league import REGULAR_SEASON_WEEKS

USABLE_HOLDOUTS = [2020, 2021, 2022, 2023, 2024]
TOP_N = 60


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=USABLE_HOLDOUTS)
    args = ap.parse_args()

    print(
        "\nHow well did each signal order that season's REAL production?\n"
        "  model = PlayerDistribution.mean, fit through the prior season\n"
        "  adp   = that season's consensus ADP (negated, so higher is better)\n"
    )
    header = (
        f"{'season':<8}{'n':>6}{'model rho':>11}{'adp rho':>10}{'diff':>8}"
        f"{'model top60':>13}{'adp top60':>11}{'diff':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    within = []
    for holdout in args.seasons:
        ctx = fit_holdout_context(fit_through_season=holdout - 1, holdout_season=holdout)

        realized = (
            ctx.weekly_holdout.filter(pl.col("week") <= REGULAR_SEASON_WEEKS)
            .group_by("player_id")
            .agg(pl.col("fantasy_points").sum().alias("total"))
        )
        real = dict(realized.iter_rows())

        pids = [p for p in ctx.pool if p in real and p in ctx.adp_by_player_id]
        proj = np.array([float(ctx.pool[p].distribution.mean) for p in pids])
        adp = np.array([ctx.adp_by_player_id[p] for p in pids])
        got = np.array([real[p] for p in pids])

        rho_model = _spearman(proj, got)
        rho_adp = _spearman(-adp, got)

        # Of the players each signal ranks in its top 60, how many were
        # actually top 60 in realized points?
        best = set(np.argsort(-got)[:TOP_N].tolist())
        hit_model = len(best & set(np.argsort(-proj)[:TOP_N].tolist())) / TOP_N
        hit_adp = len(best & set(np.argsort(adp)[:TOP_N].tolist())) / TOP_N

        # Cross-position raw points are a trap: quarterbacks score the most
        # of them, so any signal that likes QBs looks prescient without
        # having found anything a fantasy roster can use. Redo it WITHIN
        # position, which is the comparison a draft actually makes.
        per_pos = {}
        for pos in ("QB", "RB", "WR", "TE"):
            idx = [i for i, pid in enumerate(pids) if ctx.pool[pid].position == pos]
            if len(idx) < 12:
                continue
            n_top = max(6, len(idx) // 5)
            g = got[idx]
            best_pos = set(np.argsort(-g)[:n_top].tolist())
            per_pos[pos] = (
                _spearman(proj[idx], g),
                _spearman(-adp[idx], g),
                len(best_pos & set(np.argsort(-proj[idx])[:n_top].tolist())) / n_top,
                len(best_pos & set(np.argsort(adp[idx])[:n_top].tolist())) / n_top,
            )
        within.append((holdout, per_pos))

        rows.append((holdout, rho_model, rho_adp, hit_model, hit_adp))
        print(
            f"{holdout:<8}{len(pids):>6}{rho_model:>11.3f}{rho_adp:>10.3f}"
            f"{rho_model - rho_adp:>+8.3f}{hit_model:>12.0%}{hit_adp:>11.0%}"
            f"{hit_model - hit_adp:>+8.0%}",
            flush=True,
        )

    print("\n\nWITHIN POSITION -- model rho minus adp rho (positive = model better):")
    poss = ("QB", "RB", "WR", "TE")
    print(f"{'season':<8}" + "".join(f"{p:>9}" for p in poss))
    for holdout, per_pos in within:
        print(f"{holdout:<8}" + "".join(
            f"{per_pos[p][0] - per_pos[p][1]:>+9.3f}" if p in per_pos else f"{'-':>9}"
            for p in poss))
    print(f"\n{'season':<8}" + "".join(f"{p:>9}" for p in poss)
          + "   (model minus adp, top-quintile hit rate)")
    for holdout, per_pos in within:
        print(f"{holdout:<8}" + "".join(
            f"{per_pos[p][2] - per_pos[p][3]:>+9.0%}" if p in per_pos else f"{'-':>9}"
            for p in poss))

    early = [r for r in rows if r[0] <= 2022]
    late = [r for r in rows if r[0] >= 2023]
    if early and late:
        print(f"\n{'':<8}{'model - adp rho':>18}{'model - adp top60':>20}")
        for label, group in (("2020-22", early), ("2023-24", late)):
            d_rho = np.mean([r[1] - r[2] for r in group])
            d_hit = np.mean([r[3] - r[4] for r in group])
            print(f"{label:<8}{d_rho:>+18.3f}{d_hit:>+20.1%}")
        print(
            "\nIf the model's advantage over ADP shrinks or reverses in 2023-24,\n"
            "the collapse is a PROJECTION failure and nothing in the draft logic\n"
            "needs changing. If it holds steady, the projections were fine and\n"
            "the fault is in how the engine turns them into picks."
        )


if __name__ == "__main__":
    main()
