"""Stage 3 Task 7 -- the headline deliverable: does 0RB, Hero RB, or
2RB:1WR win more from draft slot 8?

Method. Force our first three picks (overall 8, 13, 28 in a 10-team snake)
into a given position pattern, taking the **highest-value available player
at the required position** using the engine's own value function, then let
the engine draft unconstrained from round 4. Score the finished roster on
that season's REAL weekly results via the production
`real_season_champion` (ex-ante lineups since the test-6 fix). Opponents
come from the through-prior-season opponent model and are paired across
structures by seed, so every structure faces the identical draft board.

Why real results across 2020-2024 rather than simulated 2026: a simulated
answer would only report what the model already believes about RB vs WR.
Scoring on real seasons asks what actually won. It also inherits the
multi-season discipline from HANDOFF 7b test 8 -- a single season is not
enough to separate a structural effect from that season's noise.

Read the error bars before the ranking. Stage 2 measured RB and WR starters
as worth nearly the same over replacement (6.4 vs 6.5 points/week), which
is a strong prior that these structures should land close together. **A
large gap is a reason to investigate, not to believe.**

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
USABLE_HOLDOUTS = [2020, 2021, 2022, 2023, 2024]

# The three structures the design doc names, plus an unconstrained engine
# reference and an ADP-following external reference.
STRUCTURES: dict[str, tuple[str, ...] | None] = {
    # The three the design doc named -- measured 2026-08-08, statistically
    # indistinguishable from each other (HANDOFF 7c).
    "0RB": ("WR", "WR", "WR"),
    "hero_RB": ("RB", "WR", "WR"),
    "2RB_1WR": ("RB", "RB", "WR"),
    # Added after the RB/WR answer came back null. The original study never
    # tested taking a QB or an elite TE early -- and that is where the
    # variance actually is: 7c measured constrained-vs-unconstrained
    # drafting (effectively "was an early QB taken?") swinging from +15.75pp
    # to -12.17pp, ten times the structural effect. The engine's QB/TE tilt
    # is either an edge or a leak, and nothing has measured which.
    "QB_early": ("QB", "RB", "WR"),
    "TE_early": ("TE", "RB", "WR"),
    "QB_and_TE": ("QB", "TE", "WR"),
    "engine_free": None,
}
CONTENDERS = (*STRUCTURES, "adp")


def structured_policy(pattern: tuple[str, ...], n_teams: int):
    """Force the first `len(pattern)` picks to the named positions, taking
    the best available at that position by the engine's own value
    function; unconstrained afterwards.

    Falls through to an unconstrained pick if no player of the required
    position is available -- which cannot happen this early in a real
    draft, but a silent IndexError here would be exactly the kind of
    plausible-looking corruption this project keeps getting bitten by.
    """

    def _policy(available_ids, pool, roster_pairs, replacement_means, replacement_by_position):
        rnd = len(roster_pairs) + 1
        candidates = available_ids
        if rnd <= len(pattern):
            required = pattern[rnd - 1]
            at_position = [pid for pid in available_ids if pool[pid].position == required]
            if not at_position:
                raise ValueError(
                    f"structure demanded a {required} in round {rnd} but none was available"
                )
            candidates = at_position
        return _choose_our_pick(
            candidates, pool, roster_pairs, replacement_means, replacement_by_position,
            n_teams=n_teams,
        )

    return _policy


def run_draft(seed: int, contender: str, ctx, n_teams: int = N_TEAMS) -> DraftState:
    rng = np.random.default_rng(seed)
    state = DraftState.from_picks([], n_teams=n_teams, rounds=ctx.rounds)
    if contender == "adp":
        policy = adp_pick_policy(ctx.adp_by_player_id)
    elif STRUCTURES[contender] is None:
        policy = None
    else:
        policy = structured_policy(STRUCTURES[contender], n_teams)

    def _run() -> DraftState:
        return run_rollout(
            state, ctx.pool, ctx.opponent_model, ctx.replacement_by_position, rng,
            our_team=DRAFT_SLOT, adp_table=ctx.adp_holdout, rankings=ctx.rankings_holdout,
            pick_policy=policy,
        )

    if contender == "adp":
        return _run()
    # Every engine-driven policy calls draft.value, so it needs the
    # through-prior-season availability patch (see backtest.py's docstring,
    # "Availability leakage inside the engine's own picks").
    with _patched_engine_availability(ctx.availability_by_position):
        return _run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=USABLE_HOLDOUTS)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed0", type=int, default=1)
    args = parser.parse_args()

    per_season: dict[int, dict[str, np.ndarray]] = {}

    for holdout in args.seasons:
        print(f"\n{'=' * 78}\nholdout {holdout} (fit through {holdout - 1})\n{'=' * 78}", flush=True)
        ctx = fit_holdout_context(fit_through_season=holdout - 1, holdout_season=holdout)
        projected_mean = {pid: float(p.distribution.mean) for pid, p in ctx.pool.items()}
        wins: dict[str, list[int]] = {c: [] for c in CONTENDERS}

        t0 = time.perf_counter()
        for seed in range(args.seed0, args.seed0 + args.n):
            for c in CONTENDERS:
                state = run_draft(seed, c, ctx)
                champ, _ = real_season_champion(
                    state, ctx.weekly_holdout, ctx.replacement_by_position,
                    ctx.replacement_means, seed=seed + SCORE_SEED_OFFSET,
                    projected_mean=projected_mean,
                )
                wins[c].append(1 if champ == DRAFT_SLOT else 0)
        print(f"  {args.n} realizations in {time.perf_counter() - t0:.0f}s", flush=True)

        per_season[holdout] = {c: np.array(v, dtype=float) for c, v in wins.items()}
        for c in CONTENDERS:
            w = per_season[holdout][c]
            print(
                f"  {c:<13}{w.mean() * 100:6.2f}% (SE {w.std(ddof=1) / math.sqrt(len(w)) * 100:.2f})",
                flush=True,
            )

    print(f"\n\n{'=' * 78}\nCHAMPIONSHIP RATE BY STRUCTURE AND SEASON (slot {DRAFT_SLOT})\n{'=' * 78}")
    print(f"{'season':<9}" + "".join(f"{c:>13}" for c in CONTENDERS))
    for holdout in sorted(per_season):
        print(
            f"{holdout:<9}"
            + "".join(f"{per_season[holdout][c].mean() * 100:>12.2f}%" for c in CONTENDERS)
        )

    print(f"\n{'=' * 78}\nPOOLED ACROSS SEASONS (equal weight per season)\n{'=' * 78}")
    pooled: dict[str, tuple[float, float]] = {}
    for c in CONTENDERS:
        per = [per_season[h][c].mean() for h in per_season]
        m = sum(per) / len(per)
        se = (
            math.sqrt(sum((x - m) ** 2 for x in per) / (len(per) - 1) / len(per))
            if len(per) > 1
            else float("nan")
        )
        pooled[c] = (m, se)
        print(f"  {c:<13}{m * 100:6.2f}%  (between-season SE {se * 100:.2f}pp)")

    print(f"\n{'=' * 78}\nPAIRWISE, PAIRED BY SEED WITHIN SEASON (row - column, pp)\n{'=' * 78}")
    names = list(STRUCTURES)
    print(f"{'':<13}" + "".join(f"{c:>15}" for c in names))
    for a in names:
        cells = ""
        for b in names:
            if a == b:
                cells += f"{'-':>15}"
                continue
            per = [(per_season[h][a] - per_season[h][b]).mean() for h in per_season]
            m = sum(per) / len(per)
            se = (
                math.sqrt(sum((x - m) ** 2 for x in per) / (len(per) - 1) / len(per))
                if len(per) > 1
                else float("nan")
            )
            cells += f"{m * 100:>+8.2f}+-{se * 100:>4.2f}"
        print(f"{a:<13}{cells}")

    print(
        "\nRead this with the Stage 2 prior in mind: RB and WR starters are worth"
        "\nnearly the same over replacement (6.4 vs 6.5 pts/week), so these"
        "\nstructures are expected to land close together. Treat any difference"
        "\nsmaller than about twice its SE as 'no measured difference'."
    )


if __name__ == "__main__":
    main()
