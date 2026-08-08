"""Run the backtest across every usable holdout season, not just 2024.

docs/HANDOFF.md 7b, tests 7-8: the 2024 holdout is the most hostile of the
usable seasons for this project's central thesis (a QB premium from 6-point
passing TDs). QB replacement ran 2.15 ppg above its long-run mean that
season, which alone moves our top-40 QB share from 8 to 3. A one-season
gate cannot distinguish "the QB thesis is wrong" from "2024 was the worst
year for it" -- both predict what was observed. So the gate is a
multi-season average, and this script computes it.

Usable holdout range is **2020-2024**. FFC ADP covers 2018-2024, but 2019
cannot be a holdout: fitting through 2018 alone leaves only 17 matched
(ADP, performance) TE pairs and `fit_rank_curves` correctly raises rather
than fitting a curve on that.

This drives the production `run_backtest` -- since the ex-ante lineup fix,
`real_season_champion` is the trustworthy scorer, so there is no longer a
separate "ex ante" mode to compare against. Each season refits every model
component from scratch through the prior season.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback

from ffdraft.backtest import BacktestSummary, fit_holdout_context, run_backtest

REPORT = ("engine", "adp", "consensus", "random")
USABLE_HOLDOUTS = [2020, 2021, 2022, 2023, 2024]


def run_season(holdout: int, n: int, seed0: int) -> BacktestSummary | None:
    print(f"\n{'=' * 78}\nholdout {holdout}  (fit through {holdout - 1})\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    try:
        ctx = fit_holdout_context(fit_through_season=holdout - 1, holdout_season=holdout)
    except Exception:
        print(f"  FIT FAILED for holdout {holdout}:\n{traceback.format_exc()}", flush=True)
        return None
    print(
        f"  fit in {time.perf_counter() - t0:.0f}s | pool {len(ctx.pool)} | rounds {ctx.rounds}",
        flush=True,
    )

    summary, _ = run_backtest(ctx, n_realizations=n, seed0=seed0)
    for name in REPORT:
        c = summary.contenders[name]
        print(f"  {name:<10}{c.championship_rate * 100:6.2f}% (SE {c.se * 100:.2f})", flush=True)
    p = summary.paired_vs_engine["adp"]
    print(f"  engine - adp = {p.paired_diff_pp:+6.2f}pp (SE {p.se_pp:.2f}pp)", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=USABLE_HOLDOUTS)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed0", type=int, default=1)
    args = parser.parse_args()

    summaries: dict[int, BacktestSummary] = {}
    for holdout in args.seasons:
        s = run_season(holdout, args.n, args.seed0)
        if s is not None:
            summaries[holdout] = s

    print(f"\n\n{'=' * 78}\nSUMMARY by holdout season\n{'=' * 78}")
    header = f"{'season':<9}" + "".join(f"{c:>12}" for c in REPORT) + f"{'engine-adp':>16}"
    print(header)
    for holdout, s in sorted(summaries.items()):
        cells = "".join(f"{s.contenders[c].championship_rate * 100:>11.2f}%" for c in REPORT)
        p = s.paired_vs_engine["adp"]
        print(f"{holdout:<9}{cells}{p.paired_diff_pp:>+11.2f}pp")

    if len(summaries) > 1:
        per = [s.paired_vs_engine["adp"].paired_diff_pp for s in summaries.values()]
        mean = sum(per) / len(per)
        se = math.sqrt(sum((x - mean) ** 2 for x in per) / (len(per) - 1) / len(per))
        wins = sum(1 for x in per if x > 0)
        print(
            f"\npooled across {len(per)} seasons (equal weight per season):"
            f"\n  engine - adp = {mean:+.2f}pp  (between-season SE {se:.2f}pp)"
            f"\n  engine beats ADP in {wins} of {len(per)} seasons"
        )


if __name__ == "__main__":
    main()
