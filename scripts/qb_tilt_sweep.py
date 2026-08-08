"""Experiment: should the QB bet be sized down?

docs/HANDOFF.md 7c ends on a judgement call -- that the early-QB decision
is the one that carries real weight, swinging from +15.75pp (2022) to
-12.17pp (2023), and that the bet should probably be sized down because
the model cannot forecast which season it will get. That was reasoning,
not measurement. This measures it.

The knob is deliberate and minimal: add `delta` to the QB replacement mean
**used by the drafting value function only**. Raising QB replacement lowers
every QB's value over replacement, so the engine takes fewer and later
quarterbacks. Scoring is untouched -- rosters are still scored on real
weekly results at the true replacement levels, so this changes the
decision, never the yardstick.

`delta = +2.15` is not arbitrary: it is exactly how much QB replacement was
underestimated in 2024 (17.96 fitted vs 20.11 realized, measured by
scripts/validate_replacement_2024.py). Sweeping past it tests whether the
engine is better off pricing QBs pessimistically in general.

What to look at: NOT just the pooled mean. The claim under test is about
*consistency*, so the between-season spread matters at least as much. A
delta that raises the mean while leaving a +/-15pp season swing has not
fixed the thing that motivated it.

This deliberately does NOT implement the "propagate replacement
uncertainty into value.py" idea, which HANDOFF 7b retracts: value.py's
lineup marginal is linear in R (averaging changes nothing) and its bench
term is convex in R via the zero floor (averaging raises marginal players'
value), so that change would if anything increase the tilt.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import numpy as np

from ffdraft.backtest import (
    _patched_engine_availability,
    adp_pick_policy,
    fit_holdout_context,
    real_season_champion,
)
from ffdraft.draft.rollout import DraftState, _choose_our_pick, run_rollout
from ffdraft.league import DRAFT_SLOT, N_TEAMS

SCORE_SEED_OFFSET = 10_000_000
USABLE_HOLDOUTS = [2020, 2021, 2022, 2023, 2024]
DEFAULT_DELTAS = [0.0, 1.0, 2.15, 3.0, 5.0]


def qb_adjusted_policy(delta: float, n_teams: int):
    """The engine's own greedy policy, with QB replacement raised by
    `delta` for valuation purposes only."""

    def _policy(available_ids, pool, roster_pairs, replacement_means, replacement_by_position):
        adjusted = dict(replacement_means)
        adjusted["QB"] = adjusted["QB"] + delta
        return _choose_our_pick(
            available_ids, pool, roster_pairs, adjusted, replacement_by_position,
            n_teams=n_teams,
        )

    return _policy


def run_draft(seed: int, contender: str, ctx, n_teams: int = N_TEAMS) -> DraftState:
    rng = np.random.default_rng(seed)
    state = DraftState.from_picks([], n_teams=n_teams, rounds=ctx.rounds)
    if contender == "adp":
        policy = adp_pick_policy(ctx.adp_by_player_id)
    else:
        policy = qb_adjusted_policy(float(contender), n_teams)

    def _run() -> DraftState:
        return run_rollout(
            state, ctx.pool, ctx.opponent_model, ctx.replacement_by_position, rng,
            our_team=DRAFT_SLOT, adp_table=ctx.adp_holdout, rankings=ctx.rankings_holdout,
            pick_policy=policy,
        )

    if contender == "adp":
        return _run()
    with _patched_engine_availability(ctx.availability_by_position):
        return _run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=USABLE_HOLDOUTS)
    parser.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed0", type=int, default=1)
    args = parser.parse_args()

    contenders = [str(d) for d in args.deltas] + ["adp"]
    rates: dict[int, dict[str, float]] = {}
    qb_counts: dict[str, list[int]] = {c: [] for c in contenders}

    for holdout in args.seasons:
        print(f"\n{'=' * 78}\nholdout {holdout}\n{'=' * 78}", flush=True)
        ctx = fit_holdout_context(fit_through_season=holdout - 1, holdout_season=holdout)
        projected_mean = {pid: float(p.distribution.mean) for pid, p in ctx.pool.items()}
        rates[holdout] = {}
        t0 = time.perf_counter()
        for c in contenders:
            wins = []
            for seed in range(args.seed0, args.seed0 + args.n):
                state = run_draft(seed, c, ctx)
                ours = [p for p in state.picks if p.team == DRAFT_SLOT]
                qb_counts[c].append(sum(1 for p in ours if p.position == "QB"))
                champ, _ = real_season_champion(
                    state, ctx.weekly_holdout, ctx.replacement_by_position,
                    ctx.replacement_means, seed=seed + SCORE_SEED_OFFSET,
                    projected_mean=projected_mean,
                )
                wins.append(1 if champ == DRAFT_SLOT else 0)
            rates[holdout][c] = float(np.mean(wins))
            print(f"  delta={c:<6}{rates[holdout][c] * 100:6.2f}%", flush=True)
        print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"\n\n{'=' * 78}\nCHAMPIONSHIP RATE BY QB-REPLACEMENT DELTA (slot {DRAFT_SLOT})\n{'=' * 78}")
    print(f"{'season':<9}" + "".join(f"{c:>10}" for c in contenders))
    for h in sorted(rates):
        print(f"{h:<9}" + "".join(f"{rates[h][c] * 100:>9.2f}%" for c in contenders))

    print(f"\n{'':<9}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}{'range':>10}{'mean QB':>10}")
    for c in contenders:
        per = [rates[h][c] * 100 for h in rates]
        m = statistics.fmean(per)
        sd = statistics.stdev(per) if len(per) > 1 else float("nan")
        print(
            f"delta={c:<4}{m:>10.2f}{sd:>10.2f}{min(per):>10.2f}{max(per):>10.2f}"
            f"{max(per) - min(per):>10.2f}{statistics.fmean(qb_counts[c]):>10.2f}"
        )

    print(
        "\nJudge on BOTH columns. The motivating claim (HANDOFF 7c) is that the QB"
        "\nbet is high-variance, so a delta that lifts the mean but leaves the range"
        "\nintact has not addressed it. With 5 seasons, sd is estimated from 5 points"
        "\nand is itself noisy -- treat ordering between adjacent deltas as weak."
    )


if __name__ == "__main__":
    main()
