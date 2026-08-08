"""Diagnostic: is the backtest's *scoring* biased toward roster depth?

`real_season_champion` sets every team's weekly lineup with `solve_lineup`
on that week's REAL scores -- perfect hindsight, every week. That is equal
across teams but NOT equal across roster *shapes*: a roster with many
FLEX-eligible players gets to pick the best 6 of N ex post, while a single
-slot position (QB, TE, K) gains nothing from depth. ADP-following drafts
~15 RB/WR and zero TE; the engine drafts ~12 plus TE and K. So hindsight
may be paying ADP a premium the engine cannot access.

This scores the SAME drafts two ways:
  hindsight  -- lineup chosen by realized weekly score (what the backtest does)
  ex_ante    -- lineup chosen by PROJECTED mean, then scored on realized points
               (what a real manager can actually do)

If ADP's edge over the engine collapses under `ex_ante`, the gap is an
artifact of the scoring rule rather than of the engine's picks.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence

import numpy as np

from ffdraft.backtest import (
    N_SCORE_WEEKS,
    PLAYOFF_BYES,
    REGULAR_SEASON_WEEKS,
    _kicker_gsis_by_name,
    _weekly_lookup,
    resolve_real_player_weeks,
)
from ffdraft.league import DRAFT_SLOT
from ffdraft.sim.lineup import RosterPlayer, solve_lineup
from ffdraft.sim.season import _run_playoffs, _seed_teams, build_regular_season_schedule

from diagnose_k_timing import run_draft  # noqa: E402

SCORE_SEED_OFFSET = 10_000_000


def _team_weekly_totals(
    picks: Sequence[tuple[str, str]],
    weekly_lookup,
    kicker_lookup,
    replacement_by_position,
    replacement_means: Mapping[str, float],
    projected_mean: Mapping[str, float],
    rng: np.random.Generator,
    mode: str,
    n_weeks: int = N_SCORE_WEEKS,
) -> np.ndarray:
    players = [
        resolve_real_player_weeks(
            pid, pos, weekly_lookup, kicker_lookup, replacement_by_position, rng, n_weeks
        )
        for pid, pos in picks
    ]
    totals = np.empty(n_weeks, dtype=float)
    for wk in range(n_weeks):
        if mode == "hindsight":
            roster = [
                RosterPlayer(p.player_id, p.position, p.scores[wk], p.available[wk]) for p in players
            ]
            totals[wk] = solve_lineup(roster, replacement_means).total_points
            continue

        # ex_ante: choose the lineup on projected means, score it on reality.
        realized = {p.player_id: p.scores[wk] for p in players}
        roster = [
            RosterPlayer(
                p.player_id,
                p.position,
                projected_mean.get(p.player_id, replacement_means[p.position]),
                p.available[wk],
            )
            for p in players
        ]
        result = solve_lineup(roster, replacement_means)
        total = 0.0
        for slot in result.slots:
            if slot.player_id is None:
                total += replacement_means[slot.slot if slot.slot != "FLEX" else "WR"]
            else:
                total += realized[slot.player_id]
        totals[wk] = total
    return totals


def champion(final_state, ctx, projected_mean, seed: int, mode: str) -> int:
    weekly_lookup = _weekly_lookup(ctx.weekly_holdout, N_SCORE_WEEKS)
    kicker_lookup = _kicker_gsis_by_name(ctx.weekly_holdout)
    rng = np.random.default_rng(seed)
    n_teams = final_state.n_teams
    team_totals = np.empty((n_teams, 1, N_SCORE_WEEKS), dtype=float)
    for team in range(1, n_teams + 1):
        team_totals[team - 1, 0, :] = _team_weekly_totals(
            list(final_state.rosters[team]),
            weekly_lookup,
            kicker_lookup,
            ctx.replacement_by_position,
            ctx.replacement_means,
            projected_mean,
            rng,
            mode,
        )
    team_totals = team_totals + rng.uniform(0, 1e-9, size=team_totals.shape)
    schedule = build_regular_season_schedule(n_teams, REGULAR_SEASON_WEEKS)
    wins = np.zeros((n_teams, 1), dtype=float)
    points_for = np.zeros((n_teams, 1), dtype=float)
    for week, pairs in enumerate(schedule):
        for a, b in pairs:
            sa, sb = team_totals[a, :, week], team_totals[b, :, week]
            wins[a] += sa > sb
            wins[b] += sb > sa
            points_for[a] += sa
            points_for[b] += sb
    seeds = _seed_teams(wins, points_for, rng)
    return int(_run_playoffs(team_totals, seeds, REGULAR_SEASON_WEEKS, PLAYOFF_BYES)[0]) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed0", type=int, default=1)
    args = parser.parse_args()

    from ffdraft.backtest import fit_holdout_context

    ctx = fit_holdout_context()
    projected_mean = {pid: float(p.distribution.mean) for pid, p in ctx.pool.items()}

    contenders = ("engine", "adp")
    modes = ("hindsight", "ex_ante")
    wins: dict[tuple[str, str], list[int]] = {(c, m): [] for c in contenders for m in modes}

    t0 = time.perf_counter()
    for seed in range(args.seed0, args.seed0 + args.n):
        for c in contenders:
            state = run_draft(seed, c, ctx)
            for m in modes:
                champ = champion(state, ctx, projected_mean, seed + SCORE_SEED_OFFSET, m)
                wins[(c, m)].append(1 if champ == DRAFT_SLOT else 0)
    print(f"\nN={args.n}, {time.perf_counter() - t0:.0f}s\n")

    print(f"{'contender':<10}{'scoring':<12}{'rate':>9}{'se':>9}")
    for m in modes:
        for c in contenders:
            w = np.array(wins[(c, m)], dtype=float)
            print(
                f"{c:<10}{m:<12}{w.mean() * 100:>8.2f}%"
                f"{w.std(ddof=1) / math.sqrt(len(w)) * 100:>8.2f}%"
            )

    print("\nadp - engine, by scoring rule:")
    for m in modes:
        d = np.array(wins[("adp", m)], dtype=float) - np.array(wins[("engine", m)], dtype=float)
        print(f"  {m:<12}: {d.mean() * 100:+6.2f}pp  (SE {d.std(ddof=1) / math.sqrt(len(d)) * 100:.2f}pp)")


if __name__ == "__main__":
    main()
