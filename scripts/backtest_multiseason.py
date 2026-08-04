"""Run the backtest across multiple holdout seasons, not just 2024.

docs/HANDOFF.md 7b, test 7: the 2024 holdout is the single most hostile of
the eight seasons on record for this project's central thesis (a QB premium
from 6-point passing TDs). QB replacement level ran 2.15 ppg above its
long-run mean that season, which alone moves our top-40 QB share from 8 to
3. A one-season gate cannot distinguish "the QB thesis is wrong" from "2024
was the worst year in eight for it" -- both predict what was observed.

This runs the same engine-vs-ADP comparison on every holdout season with
real ADP available (FFC covers 2018-2024; 2019 is the earliest usable
holdout since 2018 must be the fit window), under BOTH scoring rules:

  hindsight -- `real_season_champion`'s rule: lineups set from realized
               weekly scores. Known to pay FLEX depth a premium no manager
               could collect (test 6: worth 5.75pp to an ADP roster, ~0 to
               the engine's), so it is reported but not trusted.
  ex_ante   -- lineup chosen on projected means, scored on realized points.

Each season refits every model component from scratch through the prior
season, so runtime is dominated by the per-season fit.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback

import numpy as np

from diagnose_hindsight import champion  # noqa: E402
from diagnose_k_timing import run_draft  # noqa: E402
from ffdraft.backtest import fit_holdout_context
from ffdraft.league import DRAFT_SLOT

SCORE_SEED_OFFSET = 10_000_000
CONTENDERS = ("engine", "adp")
MODES = ("hindsight", "ex_ante")


def run_season(holdout: int, n: int, seed0: int) -> dict[tuple[str, str], list[int]] | None:
    print(f"\n{'=' * 78}\nholdout {holdout}  (fit through {holdout - 1})\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    try:
        ctx = fit_holdout_context(fit_through_season=holdout - 1, holdout_season=holdout)
    except Exception:
        print(f"  FIT FAILED for holdout {holdout}:\n{traceback.format_exc()}", flush=True)
        return None
    print(f"  fit in {time.perf_counter() - t0:.0f}s | pool {len(ctx.pool)} | rounds {ctx.rounds}", flush=True)
    print(f"  {ctx.report.describe()}", flush=True)

    projected_mean = {pid: float(p.distribution.mean) for pid, p in ctx.pool.items()}
    wins: dict[tuple[str, str], list[int]] = {(c, m): [] for c in CONTENDERS for m in MODES}

    t1 = time.perf_counter()
    for seed in range(seed0, seed0 + n):
        for c in CONTENDERS:
            state = run_draft(seed, c, ctx)
            for m in MODES:
                champ = champion(state, ctx, projected_mean, seed + SCORE_SEED_OFFSET, m)
                wins[(c, m)].append(1 if champ == DRAFT_SLOT else 0)
    print(f"  {n} realizations in {time.perf_counter() - t1:.0f}s", flush=True)

    for m in MODES:
        e = np.array(wins[("engine", m)], dtype=float)
        a = np.array(wins[("adp", m)], dtype=float)
        d = e - a
        print(
            f"  {m:<10} engine {e.mean() * 100:5.2f}%  adp {a.mean() * 100:5.2f}%  "
            f"engine-adp {d.mean() * 100:+6.2f}pp (SE {d.std(ddof=1) / math.sqrt(len(d)) * 100:.2f}pp)",
            flush=True,
        )
    return wins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=1)
    args = parser.parse_args()

    all_wins: dict[int, dict[tuple[str, str], list[int]]] = {}
    for holdout in args.seasons:
        w = run_season(holdout, args.n, args.seed0)
        if w is not None:
            all_wins[holdout] = w

    print(f"\n\n{'=' * 78}\nSUMMARY -- engine minus ADP, by holdout season\n{'=' * 78}")
    print(f"{'season':<9}{'hindsight':>26}{'ex_ante':>26}")
    for holdout, wins in sorted(all_wins.items()):
        cells = ""
        for m in MODES:
            e = np.array(wins[("engine", m)], dtype=float)
            a = np.array(wins[("adp", m)], dtype=float)
            d = e - a
            cells += f"{d.mean() * 100:+9.2f}pp (SE {d.std(ddof=1) / math.sqrt(len(d)) * 100:4.2f})"
        print(f"{holdout:<9}{cells}")

    print(f"\n{'season':<9}{'engine hind':>13}{'adp hind':>11}{'engine ex':>12}{'adp ex':>10}")
    for holdout, wins in sorted(all_wins.items()):
        vals = [np.array(wins[(c, m)], dtype=float).mean() * 100 for m in MODES for c in CONTENDERS]
        print(f"{holdout:<9}{vals[0]:>12.2f}%{vals[1]:>10.2f}%{vals[2]:>11.2f}%{vals[3]:>9.2f}%")

    if all_wins:
        print("\npooled across seasons (equal weight per season):")
        for m in MODES:
            per = [
                (np.array(w[("engine", m)], dtype=float) - np.array(w[("adp", m)], dtype=float)).mean()
                for w in all_wins.values()
            ]
            mean = sum(per) / len(per)
            se = (
                math.sqrt(sum((x - mean) ** 2 for x in per) / (len(per) - 1) / len(per))
                if len(per) > 1
                else float("nan")
            )
            print(f"  {m:<10} engine - adp = {mean * 100:+6.2f}pp  (between-season SE {se * 100:.2f}pp)")


if __name__ == "__main__":
    main()
