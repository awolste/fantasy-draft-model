"""Does fixing the QB/TE tilt actually win? ANSWERED: no. (HANDOFF item 3)

**Result, 2026-08-09, N=300/season, 2020-2024, paired by seed within
season. Neither fix is worth shipping and nothing in `value.py` changed.**

```
season      engine  baseline    guard     both      adp
2020        16.33%     4.00%   15.33%    4.00%    8.00%
2021        17.67%    14.67%   22.33%   17.67%   11.67%
2022        18.33%    14.67%    7.33%    5.67%    8.33%
2023         7.67%     7.33%   15.00%   13.00%   11.33%
2024         5.33%     2.67%    7.67%    3.67%   10.00%

paired vs engine:  baseline -4.40pp (SE 2.06, 2.1 sigma)   <- harmful
                   guard    +0.47pp (SE 3.18, 0.1 sigma)   <- nothing
                   both     -4.27pp (SE 3.55, 1.2 sigma)
```

`guard` swings from -11.00pp (2022) to +7.33pp (2023) and averages to
nothing. 2022 is the tell: it is the season with the *most* early QB/TE
picks (1.60) and the guard's *worst* result, because 2022 is when QB
replacement bottomed at 16.36 and the early quarterback actually paid
(HANDOFF 7c measured unconstrained drafting beating every forced structure
by 15.75pp that year). **A rounds-1-3 ban is a bet on how good streamable
QBs turn out to be, which is not knowable on draft day.** That is the same
wall 7c hit from the other direction, now confirmed against the engine's
own free choices rather than against forced structures.

Do not read `guard`'s +0.47pp as mild support. It is 0.1 sigma, and its
per-season spread is larger than any effect the structure study measured.

---

Original question below.

The tilt is real and localised (`scripts/diagnose_tilt.py`): `draft.value`
prices candidates against the **waiver** replacement level, which is what a
slot is worth if never drafted at all, and that overstates flat-topped
positions -- at pick 8 on an ADP board, passing on the best QB or TE costs
literally nothing (both survive to pick 33) while passing on the best RB
costs 0.95 immediately and 3.73 by the next turn.

Two candidate fixes, deliberately measured *before* either becomes a
default, because the one previous attempt in this area (HANDOFF 10c's QB
`delta` sweep) looked reasonable and measured worse:

**`baseline`** -- replace the waiver replacement level with the "last
starter drafted" level (`draft.baseline.starter_replacement_means`) in the
drafting value function only. Scoring keeps true waiver levels, so this
changes the decision, never the yardstick. Principled, but the diagnostic
already warns it is a small effect: it moves the top-40 mix only from
QB 6/TE 4 to QB 5/TE 4.

**`guard`** -- exclude QB and TE from our first three picks, a policy-level
constraint in the same family as the bench floor/ceiling guards already in
`_choose_our_pick` and the position caps in `live/tiebreak.py`. This is the
fix the evidence directly supports: HANDOFF 7c measured `2RB_1WR` beating
`TE_early` by 4.85pp (3.6 sigma, survives Bonferroni over 9 comparisons),
with all 9 skill-vs-QB/TE comparisons pointing the same way.

`both` tests them together, since the baseline shift is a level correction
and the guard is a timing constraint -- they are not redundant.

Read `engine` vs each arm **paired by seed within season**: every arm faces
an identical draft board, so the pairwise SE is far smaller than the
between-season spread, which is ~4pp and swamps everything (HANDOFF 7c).

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
from ffdraft.league import DRAFT_SLOT, FLEX_ELIGIBLE, N_TEAMS, STARTERS

SCORE_SEED_OFFSET = 10_000_000
USABLE_HOLDOUTS = [2020, 2021, 2022, 2023, 2024]

# Rounds 1-3 is the window the structure study actually measured; the guard
# is not evidence for anything outside it.
GUARD_ROUNDS = 3
GUARD_EXCLUDED = frozenset({"QB", "TE"})

ARMS = ("engine", "baseline", "guard", "both", "adp")


def starter_replacement_means(
    pool,
    waiver_means,
    starters=STARTERS,
    n_teams: int = N_TEAMS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
) -> dict[str, float]:
    """The `baseline` arm: replacement level as the best player at a
    position who does not start anywhere in the league ("last starter
    drafted"), floored at the waiver level.

    **This lives here, in the script that measured it, rather than in
    `src/` -- it lost in all five seasons (-4.40pp, 2.1 sigma) and is not
    production code.** It is kept only so the result is reproducible.

    Nothing here names a position: the `starters['FLEX'] * n_teams` flex
    slots are filled greedily by projected mean from every FLEX-eligible
    position's leftovers, mirroring `sim.lineup.solve_lineup` at league
    scale, and each position's cutoff is its dedicated count plus the flex
    share it actually won.
    """
    means_by_position: dict[str, list[float]] = {}
    for entry in pool.values():
        means_by_position.setdefault(entry.position, []).append(float(entry.distribution.mean))
    for values in means_by_position.values():
        values.sort(reverse=True)

    dedicated = {pos: starters[pos] * n_teams for pos in starters if pos != "FLEX"}

    leftovers: list[tuple[float, str]] = []
    for pos in flex_eligible:
        for mean in means_by_position.get(pos, [])[dedicated.get(pos, 0) :]:
            leftovers.append((mean, pos))
    leftovers.sort(reverse=True)

    flex_won: dict[str, int] = {}
    for _, pos in leftovers[: starters.get("FLEX", 0) * n_teams]:
        flex_won[pos] = flex_won.get(pos, 0) + 1

    out: dict[str, float] = {}
    for pos, waiver in waiver_means.items():
        cutoff = dedicated.get(pos, 0) + flex_won.get(pos, 0)
        values = means_by_position.get(pos, [])
        if cutoff == 0 or cutoff >= len(values):
            out[pos] = float(waiver)
            continue
        out[pos] = max(float(waiver), values[cutoff])
    return out


def tilt_policy(use_baseline: bool, use_guard: bool, ctx, n_teams: int):
    """`_choose_our_pick` with either fix applied.

    Both are expressed here rather than in `draft/rollout.py` so that
    measuring them cannot change production behaviour -- nothing is a
    default until it has won.
    """
    value_means = (
        starter_replacement_means(ctx.pool, ctx.replacement_means)
        if use_baseline
        else ctx.replacement_means
    )

    def _policy(available_ids, pool, roster_pairs, replacement_means, replacement_by_position):
        candidates = available_ids
        if use_guard and len(roster_pairs) < GUARD_ROUNDS:
            allowed = [pid for pid in available_ids if pool[pid].position not in GUARD_EXCLUDED]
            if not allowed:
                raise ValueError("no non-QB/TE player available in the first three rounds")
            candidates = allowed
        return _choose_our_pick(
            candidates, pool, roster_pairs, value_means, replacement_by_position,
            n_teams=n_teams,
        )

    return _policy


def run_draft(seed: int, arm: str, ctx, n_teams: int = N_TEAMS) -> DraftState:
    rng = np.random.default_rng(seed)
    state = DraftState.from_picks([], n_teams=n_teams, rounds=ctx.rounds)
    if arm == "adp":
        policy = adp_pick_policy(ctx.adp_by_player_id)
    elif arm == "engine":
        policy = None
    else:
        policy = tilt_policy(arm in ("baseline", "both"), arm in ("guard", "both"), ctx, n_teams)

    def _run() -> DraftState:
        return run_rollout(
            state, ctx.pool, ctx.opponent_model, ctx.replacement_by_position, rng,
            our_team=DRAFT_SLOT, adp_table=ctx.adp_holdout, rankings=ctx.rankings_holdout,
            pick_policy=policy,
        )

    if arm == "adp":
        return _run()
    with _patched_engine_availability(ctx.availability_by_position):
        return _run()


def _mean_se(per_season: list[float]) -> tuple[float, float]:
    m = sum(per_season) / len(per_season)
    if len(per_season) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in per_season) / (len(per_season) - 1)
    return m, math.sqrt(var / len(per_season))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=USABLE_HOLDOUTS)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=1)
    args = ap.parse_args()

    per_season: dict[int, dict[str, np.ndarray]] = {}
    qb_te_early: dict[int, dict[str, float]] = {}

    for holdout in args.seasons:
        print(f"\n{'=' * 78}\nholdout {holdout} (fit through {holdout - 1})\n{'=' * 78}", flush=True)
        ctx = fit_holdout_context(fit_through_season=holdout - 1, holdout_season=holdout)
        projected_mean = {pid: float(p.distribution.mean) for pid, p in ctx.pool.items()}
        wins: dict[str, list[int]] = {a: [] for a in ARMS}
        early: dict[str, list[int]] = {a: [] for a in ARMS}

        t0 = time.perf_counter()
        for seed in range(args.seed0, args.seed0 + args.n):
            for arm in ARMS:
                state = run_draft(seed, arm, ctx)
                champ, _ = real_season_champion(
                    state, ctx.weekly_holdout, ctx.replacement_by_position,
                    ctx.replacement_means, seed=seed + SCORE_SEED_OFFSET,
                    projected_mean=projected_mean,
                )
                wins[arm].append(1 if champ == DRAFT_SLOT else 0)
                ours = state.rosters[DRAFT_SLOT][:GUARD_ROUNDS]
                early[arm].append(sum(1 for _, pos in ours if pos in GUARD_EXCLUDED))
        print(f"  {args.n} realizations x {len(ARMS)} arms in {time.perf_counter() - t0:.0f}s", flush=True)

        per_season[holdout] = {a: np.array(v, dtype=float) for a, v in wins.items()}
        qb_te_early[holdout] = {a: float(np.mean(early[a])) for a in ARMS}
        for arm in ARMS:
            w = per_season[holdout][arm]
            print(
                f"  {arm:<10}{w.mean() * 100:6.2f}%  (SE {w.std(ddof=1) / math.sqrt(len(w)) * 100:.2f})"
                f"   QB/TE in R1-3: {qb_te_early[holdout][arm]:.2f}",
                flush=True,
            )

    print(f"\n\n{'=' * 78}\nCHAMPIONSHIP RATE BY ARM AND SEASON (slot {DRAFT_SLOT})\n{'=' * 78}")
    print(f"{'season':<9}" + "".join(f"{a:>12}" for a in ARMS))
    for holdout in sorted(per_season):
        print(f"{holdout:<9}" + "".join(f"{per_season[holdout][a].mean() * 100:>11.2f}%" for a in ARMS))

    print(f"\n{'=' * 78}\nPOOLED (equal weight per season)\n{'=' * 78}")
    for arm in ARMS:
        m, se = _mean_se([per_season[h][arm].mean() for h in per_season])
        mean_early = sum(qb_te_early[h][arm] for h in qb_te_early) / len(qb_te_early)
        print(f"  {arm:<10}{m * 100:6.2f}%  (between-season SE {se * 100:.2f}pp)   "
              f"mean QB/TE in R1-3: {mean_early:.2f}")

    print(f"\n{'=' * 78}\nPAIRED BY SEED, vs the current engine (pp)\n{'=' * 78}")
    print("Paired within season, so this SE is the one to read -- it removes the")
    print("~4pp between-season swing that swamps the pooled table above.\n")
    for arm in ARMS:
        if arm == "engine":
            continue
        m, se = _mean_se(
            [(per_season[h][arm] - per_season[h]["engine"]).mean() for h in per_season]
        )
        sigma = abs(m / se) if se and not math.isnan(se) and se > 0 else float("nan")
        print(f"  {arm:<10}{m * 100:>+7.2f}pp  (SE {se * 100:.2f})  {sigma:.1f} sigma")


if __name__ == "__main__":
    main()
