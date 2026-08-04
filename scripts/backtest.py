"""Stage 3 Task 6: run the 2024 holdout backtest and print the report.

Usage:
    .venv/bin/python scripts/backtest.py --n 100 --seed0 1

See `ffdraft/backtest.py` for the implementation and its module docstring
for the full leakage argument. This script only fits the holdout context
once, runs N draft realizations, and prints the headline numbers.

Personal-data rule: this script never reads `.env` or
`data/manager_labels.csv`, and only ever refers to teams/opponents by
index or draft slot, never by manager name.
"""

from __future__ import annotations

import argparse
import time

from ffdraft.backtest import fit_holdout_context, run_backtest
from ffdraft.league import DRAFT_SLOT, N_TEAMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of draft realizations")
    parser.add_argument("--seed0", type=int, default=1, help="first seed (seeds are seed0..seed0+n-1)")
    args = parser.parse_args()

    print("Fitting holdout context (through 2023 only; see report below for exact seasons)...")
    t0 = time.perf_counter()
    ctx = fit_holdout_context()
    print(f"Fit in {time.perf_counter() - t0:.1f}s\n")

    print("=== Component-by-component season coverage ===")
    print(ctx.report.describe())
    print(f"\nPool size: {len(ctx.pool)} players")
    print(f"2024 draft rounds: {ctx.rounds} ({ctx.rounds * N_TEAMS} total picks)")
    print(f"Our draft slot: {DRAFT_SLOT}\n")

    print(f"=== Running {args.n} draft realizations (seeds {args.seed0}..{args.seed0 + args.n - 1}) ===")
    summary, results = run_backtest(ctx, n_realizations=args.n, seed0=args.seed0)

    print(f"\nElapsed: {summary.elapsed_seconds:.1f}s ({summary.elapsed_seconds / max(args.n, 1):.2f}s/realization)")

    print("\n=== Championship rate by contender ===")
    print(f"{'contender':<12}{'rate':>10}{'se':>10}{'n':>8}")
    for name in ("engine", "adp", "consensus", "random"):
        c = summary.contenders[name]
        print(f"{name:<12}{c.championship_rate * 100:>9.2f}%{c.se * 100:>9.2f}%{c.n:>8}")

    print("\n=== Paired difference vs engine (engine - contender, percentage points) ===")
    for name in ("adp", "consensus", "random"):
        p = summary.paired_vs_engine[name]
        print(f"engine - {name:<10}: {p.paired_diff_pp:+6.2f}pp  (SE {p.se_pp:.2f}pp)")

    print("\n=== Engine vs ADP-baseline pick divergence at our own picks, by round bucket ===")
    for bucket in ("early", "middle", "late"):
        rb = summary.round_bucket_divergence.get(bucket)
        if rb is not None:
            print(f"{bucket:<8}: {rb.divergence_rate * 100:5.1f}% of our picks differ from ADP  (n={rb.n})")

    n_fallback_totals = {c: sum(r.n_fallback_by_contender[c] for r in results) for c in summary.contenders}
    print(f"\nTotal replacement-level scoring fallbacks (drafted-but-unmatched-in-2024) by contender: {n_fallback_totals}")


if __name__ == "__main__":
    main()
