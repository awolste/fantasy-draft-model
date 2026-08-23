"""Does the lossy cache key (top-N availability) change the recommendation?

`live/cache.py` keys on availability within the top `TOP_N_FOR_KEY` only.
That is an approximation: two boards differing *solely* below the cutoff
share a key and would be served the same cached answer. This checks that
they deserve to be.

Method. Build a base board whose drafted set *includes some players from
below the cutoff*, then perturb it by swapping one of those deep players
for a different deep player. Top-N availability is untouched, so the key is
unchanged by construction. Run the full budget on each and compare.

Getting this construction right matters: an earlier version swapped a
player from *inside* the top N, which moved the key on every perturbation.
It then reported "OK -- 3/3 agreement" while its own warning said the key
had moved. It validated nothing and said the opposite. The run is now
treated as INVALID if any key moves.

If the leader disagrees, `TOP_N_FOR_KEY` is too small: raise it and re-run.
Do not accept a mismatch. It would mean the cache can serve a wrong answer
on draft day and never say so, which is precisely the failure mode
`docs/HANDOFF.md` section 8 identifies as this project's dominant risk.

Runs against the **live 2026 context** -- the pool the tool will actually
face. Full budget is ~200s per call (see `live/budget.py`), so this is
deliberately a small number of perturbations at a few picks, not a sweep.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ffdraft.draft.rollout import DraftState, Pick, team_for_pick
from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET
from ffdraft.live.cache import TOP_N_FOR_KEY, recommend, state_key
from ffdraft.live.context import live_context


def _board(ctx, drafted: list[str]) -> DraftState:
    picks = [
        Pick(
            overall_pick=i + 1,
            team=team_for_pick(i + 1, N_TEAMS),
            player_id=pid,
            position=ctx.pool[pid].position,
        )
        for i, pid in enumerate(drafted)
    ]
    return DraftState.from_picks(picks, n_teams=N_TEAMS, rounds=DRAFT_ROUNDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick", type=int, default=DRAFT_SLOT,
                        help="which overall pick to test at (must be one of ours)")
    parser.add_argument("--perturbations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-deep", type=int, default=3,
                        help="below-cutoff players to seed the base board with")
    args = parser.parse_args()

    ctx = live_context()
    ranked = sorted(ctx.pool, key=lambda pid: ctx.pool[pid].rank)
    deep = ranked[TOP_N_FOR_KEY:]
    rng = np.random.default_rng(args.seed)

    # Deliberately include `n_deep` below-cutoff players in the drafted set,
    # so there is something to perturb that does not move the key. A real
    # early board would not look like this, but the question being asked is
    # about the KEY's sensitivity, not about realistic boards.
    n_deep = min(args.n_deep, args.pick - 1)
    base_drafted = ranked[: args.pick - 1 - n_deep] + [
        deep[int(rng.integers(0, len(deep)))] for _ in range(n_deep)
    ]
    base = _board(ctx, base_drafted)
    if base.team_on_clock != DRAFT_SLOT:
        raise SystemExit(
            f"pick {args.pick} belongs to team {base.team_on_clock}, not ours "
            f"({DRAFT_SLOT}) -- recommend_pick only answers for our own turn."
        )

    counts: dict[str, int] = {}
    for p in base.picks:
        if p.team == DRAFT_SLOT:
            counts[p.position] = counts.get(p.position, 0) + 1
    base_key = state_key(args.pick, counts, set(base_drafted), ctx.pool)

    print(f"cutoff TOP_N_FOR_KEY={TOP_N_FOR_KEY} | pick {args.pick} | "
          f"full budget {FULL_BUDGET.label} (~8s per call since HANDOFF section 15)\n")

    t0 = time.perf_counter()
    base_rec = recommend(base, ctx, FULL_BUDGET)
    base_leader = base_rec.candidates[0]
    print(f"base leader: {base_leader.name} ({base_leader.position}) "
          f"p={base_leader.championship_probability * 100:.2f}% "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)

    agree = 0
    keys_held = 0
    for i in range(args.perturbations):
        swapped = list(base_drafted)
        # Swap one DEEP player for another DEEP player. Both are below the
        # cutoff, so top-N availability -- and therefore the key -- is
        # untouched. Swapping anything above the cutoff would move the key
        # and test a different question entirely.
        while True:
            replacement = deep[int(rng.integers(0, len(deep)))]
            if replacement not in swapped:
                break
        swapped[-1] = replacement

        c2: dict[str, int] = {}
        board = _board(ctx, swapped)
        for p in board.picks:
            if p.team == DRAFT_SLOT:
                c2[p.position] = c2.get(p.position, 0) + 1
        key = state_key(args.pick, c2, set(swapped), ctx.pool)

        rec = recommend(board, ctx, FULL_BUDGET)
        leader = rec.candidates[0]
        same_key = key == base_key
        same_leader = leader.player_id == base_leader.player_id
        agree += same_leader
        keys_held += same_key
        gap = abs(leader.championship_probability - base_leader.championship_probability) * 100
        print(
            f"  perturbation {i}: same_key={same_key} same_leader={same_leader} "
            f"leader={leader.name} ({leader.position}) delta_p={gap:.2f}pp",
            flush=True,
        )
        if not same_key:
            print("    WARNING: key moved -- the perturbation was not below the cutoff.")

    print(f"\nleader agreement: {agree}/{args.perturbations} "
          f"| keys held: {keys_held}/{args.perturbations}")
    if keys_held < args.perturbations:
        raise SystemExit(
            "INVALID RUN -- the key moved, so boards sharing a key were never "
            "compared. This tests nothing about the cutoff. Fix the perturbation "
            "construction (it must swap only players below the cutoff) and re-run."
        )
    if agree < args.perturbations:
        print(
            "MISMATCH -- boards sharing a key produced different leaders.\n"
            f"Raise TOP_N_FOR_KEY above {TOP_N_FOR_KEY} and re-run. Do not ship "
            "this cutoff: the cache would serve a wrong answer silently."
        )
    else:
        print("OK -- the cutoff does not change the recommendation at this pick.")


if __name__ == "__main__":
    main()
