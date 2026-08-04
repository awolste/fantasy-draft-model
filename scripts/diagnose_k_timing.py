"""Diagnostic: does the engine's *pick timing* on kickers explain the gap?

docs/HANDOFF.md 7b tested hypothesis A ("wasted roster spots") by capping
QB<=2 and K<=1 and measured only +0.67pp. But that capped *counts*, not
*timing*: `engine_capped` still spent a round-8 pick on a kicker whose ADP
is 141. scripts/diagnose_picks.py measures the engine's mean kicker reach
at +48.9 picks, against an ADP policy that never reaches >=30 on any pick.

This contender is the engine's own greedy policy, unchanged, except that K
is not draftable until the final round -- where it would go anyway. One
variable. Paired against ADP-following on the same seeds.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import math
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


def engine_defer_kicker_policy(pool, rounds: int, n_teams: int):
    """The engine's greedy policy with K removed from the candidate set
    until the final round."""

    def _policy(available_ids, pool_arg, roster_pairs, replacement_means, replacement_by_position):
        rnd = len(roster_pairs) + 1
        if rnd < rounds:
            candidates = [pid for pid in available_ids if pool_arg[pid].position != "K"]
        else:
            candidates = list(available_ids)
        return _choose_our_pick(
            candidates, pool_arg, roster_pairs, replacement_means, replacement_by_position,
            n_teams=n_teams,
        )

    return _policy


def vor_only_policy():
    """The ladder rung docs/HANDOFF.md 7b says is missing: static value over
    replacement, argmax(mean - replacement_mean[position]). Scarcity
    correction with *no* roster awareness, no lineup solve, no bench term,
    no FLEX reasoning. Separates "scarcity" from "roster construction"."""

    def _policy(available_ids, pool, roster_pairs, replacement_means, replacement_by_position):
        pid = max(
            available_ids,
            key=lambda p: float(pool[p].distribution.mean) - replacement_means[pool[p].position],
        )
        return pid, pool[pid].position

    return _policy


def run_draft(seed: int, contender: str, ctx, n_teams: int = N_TEAMS) -> DraftState:
    rng = np.random.default_rng(seed)
    state = DraftState.from_picks([], n_teams=n_teams, rounds=ctx.rounds)
    if contender == "adp":
        policy = adp_pick_policy(ctx.adp_by_player_id)
    elif contender == "engine_defer_k":
        policy = engine_defer_kicker_policy(ctx.pool, ctx.rounds, n_teams)
    elif contender == "vor_only":
        policy = vor_only_policy()
    elif contender == "engine":
        policy = None
    else:
        raise ValueError(contender)

    def _run() -> DraftState:
        return run_rollout(
            state, ctx.pool, ctx.opponent_model, ctx.replacement_by_position, rng,
            our_team=DRAFT_SLOT, adp_table=ctx.adp_holdout, rankings=ctx.rankings_holdout,
            pick_policy=policy,
        )

    if contender.startswith("engine"):
        with _patched_engine_availability(ctx.availability_by_position):
            return _run()
    return _run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed0", type=int, default=1)
    args = parser.parse_args()

    ctx = fit_holdout_context()
    contenders = ("engine", "engine_defer_k", "vor_only", "adp")
    wins: dict[str, list[int]] = {c: [] for c in contenders}

    t0 = time.perf_counter()
    for seed in range(args.seed0, args.seed0 + args.n):
        for c in contenders:
            state = run_draft(seed, c, ctx)
            champ, _ = real_season_champion(
                state, ctx.weekly_holdout, ctx.replacement_by_position, ctx.replacement_means,
                seed=seed + SCORE_SEED_OFFSET,
            )
            wins[c].append(1 if champ == DRAFT_SLOT else 0)
    elapsed = time.perf_counter() - t0

    print(f"\nN={args.n} realizations, {elapsed:.0f}s ({elapsed / args.n:.2f}s/realization)\n")
    print(f"{'contender':<18}{'rate':>9}{'se':>9}")
    for c in contenders:
        w = np.array(wins[c], dtype=float)
        rate, se = w.mean(), w.std(ddof=1) / math.sqrt(len(w))
        print(f"{c:<18}{rate * 100:>8.2f}%{se * 100:>8.2f}%")

    print("\npaired differences (percentage points):")
    for a, b in (
        ("engine_defer_k", "engine"),
        ("vor_only", "engine"),
        ("vor_only", "adp"),
        ("engine", "adp"),
    ):
        d = np.array(wins[a], dtype=float) - np.array(wins[b], dtype=float)
        print(f"  {a} - {b:<16}: {d.mean() * 100:+6.2f}pp  (SE {d.std(ddof=1) / math.sqrt(len(d)) * 100:.2f}pp)")


if __name__ == "__main__":
    main()
