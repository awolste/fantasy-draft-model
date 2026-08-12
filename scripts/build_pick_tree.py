"""Precompute a decision tree over our first six picks, for draft day.

Each node is **one of our picks**. Its children are the picks worth making
at our *next* turn given that choice. Six levels deep, so a path through
the tree is a complete plan for rounds 1-6 at draft slot 8 (overall picks
8, 13, 28, 33, 48, 53).

## What branches, and what does not

**Only our own choices branch.** Opponent behaviour between our picks does
not: branching on it too would multiply the tree by the size of the draft
pool at every level and make it unreadable, and it is not a decision we
control anyway.

Instead every node carries `p_available` -- the probability that player is
still on the board when we get there, computed by simulating the
intervening opponent picks `SURVIVAL_SIMS` times with the fitted opponent
model (including its per-round temperature, HANDOFF 7d). **This is the
number that says whether a branch is real.** A node at 0.15 is a
contingency to recognise, not a plan; a node at 0.95 is close to a
decision you will actually face.

The *board* each node is evaluated on is one canonical opponent
realisation, seeded deterministically from the path so the tree is
reproducible. That is a genuine limitation, not a hidden one: a different
realisation would put slightly different players in the candidate lists.
`p_available` is the honest correction to it, and is computed over many
realisations rather than the one.

## Roster shape

The owner's constraint: aim for 3 RB and 3 WR after six rounds, allowing
4/2 or 2/4 when one position clearly warrants it. Since 3+3 = 6, that
makes all six picks RB or WR -- which is also what the structure study
concluded independently for rounds 1-3 (HANDOFF 7c: do not spend an early
pick on a QB or an elite TE).

`_reachable` enforces it by feasibility rather than by a hardcoded
pattern: a position is offered only if some legal final split (RB in
2..4) is still reachable from the resulting counts.

## Cost

One `recommend_pick` per internal node, at the same `LIVE_BUDGET` the live
app uses on draft day -- so the tree says what the app would say, not
something cheaper. With the default branching that is 94 calls.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from ffdraft.draft.rollout import DraftState, Pick, pool_adp_lookup, team_for_pick
from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import LIVE_BUDGET, Budget
from ffdraft.live.cache import candidates_to_rows, recommend
from ffdraft.live.context import live_context
from ffdraft.live.survival import survival_probabilities
from ffdraft.models.opponent import AvailablePlayer, sample_pick

N_ROUNDS = 6
POSITIONS = ("RB", "WR")
MIN_AT_POSITION = 2
MAX_AT_POSITION = 4

# Children per node, by round. Wider at the top because that is where the
# real choice is and where the owner wants to see alternatives; narrower
# later, or the tree is unreadable without adding information.
BRANCHING = (3, 2, 2, 2, 2, 2)

SURVIVAL_SIMS = 200


def _reachable(rb: int, wr: int) -> bool:
    """Can `(rb, wr)` still finish at a legal split (RB in 2..4, RB+WR=6)?"""
    if rb + wr > N_ROUNDS:
        return False
    lo = max(rb, MIN_AT_POSITION)
    hi = min(MAX_AT_POSITION, N_ROUNDS - wr)
    return lo <= hi


def _path_seed(path: tuple[str, ...], base: int) -> int:
    """Deterministic per-path opponent seed, so the tree is reproducible."""
    h = base
    for pid in path:
        h = (h * 1_000_003 + hash(pid)) % (2**31 - 1)
    return h


def _fill_opponents(picks, drafted, ctx, adp, upto_exclusive, rng):
    """Sample opponent picks up to (not including) `upto_exclusive`."""
    while len(picks) + 1 < upto_exclusive:
        overall = len(picks) + 1
        team = team_for_pick(overall, N_TEAMS)
        counts: dict[str, int] = {}
        for p in picks:
            if p.team == team:
                counts[p.position] = counts.get(p.position, 0) + 1
        available = [
            AvailablePlayer(
                player_id=pid,
                position=ctx.pool[pid].position,
                adp=adp.get(pid, float(ctx.pool[pid].rank)),
            )
            for pid in ctx.pool
            if pid not in drafted
        ]
        taken = sample_pick(
            ctx.opponent_model, f"slot_{team}", available, counts, rng,
            round_=(overall - 1) // N_TEAMS + 1,
        )
        picks.append(
            Pick(overall_pick=overall, team=team,
                 player_id=taken, position=ctx.pool[taken].position)
        )
        drafted.add(taken)
    return picks, drafted


def build(args) -> dict:
    ctx = live_context()
    adp = pool_adp_lookup(ctx.pool, ctx.adp_table, ctx.rankings)[0]
    budget = Budget(4, 4, 60, seed=args.seed) if args.fast else LIVE_BUDGET
    # This script has a `main()` guard, so it can spread each node's
    # candidates over the cores -- see
    # `draft.recommender._evaluate_all_candidates` for why that is opt-in
    # and why the Streamlit app must not do it.
    budget = dataclasses.replace(budget, n_workers=args.workers)

    our_picks = [
        p for p in range(1, DRAFT_ROUNDS * N_TEAMS + 1)
        if team_for_pick(p, N_TEAMS) == DRAFT_SLOT
    ][:N_ROUNDS]
    print(f"our first {N_ROUNDS} picks: {our_picks}\n", flush=True)

    n_calls = 0
    t0 = time.perf_counter()

    def expand(prior_picks, drafted, path, counts, depth):
        """Return the list of child nodes for our pick at index `depth`."""
        nonlocal n_calls
        our_pick = our_picks[depth]

        # Canonical opponent fill up to our pick, seeded from the path.
        rng = np.random.default_rng(_path_seed(tuple(path), args.seed))
        picks = list(prior_picks)
        drafted_now = set(drafted)
        drafted_before_interval = set(drafted)
        picks, drafted_now = _fill_opponents(
            picks, drafted_now, ctx, adp, our_pick, rng
        )

        state = DraftState.from_picks(picks, n_teams=N_TEAMS, rounds=DRAFT_ROUNDS)
        # Restrict pruning to the positions still legal for the target
        # roster shape. Without this the candidate list is sometimes
        # entirely QB/TE by round 3 (HANDOFF item 3) and there is no RB or
        # WR left to branch on -- filtering the default output cannot
        # recover a candidate that was never evaluated.
        allowed = frozenset(
            pos for pos in POSITIONS
            if _reachable(
                counts.get("RB", 0) + (pos == "RB"),
                counts.get("WR", 0) + (pos == "WR"),
            )
        )
        rows = candidates_to_rows(
            recommend(state, ctx, budget, restrict_to_positions=allowed)
        )
        n_calls += 1

        if not rows:
            raise RuntimeError(
                f"no candidate at pick {our_pick}, counts={counts}, allowed={sorted(allowed)}"
            )
        chosen = rows[: BRANCHING[depth] if depth < len(BRANCHING) else 1]

        # How likely each of these is actually still there, over many
        # opponent realisations rather than the one canonical fill.
        n_interval = our_pick - (our_picks[depth - 1] if depth else 0) - 1
        surv = survival_probabilities(
            candidate_ids=[r["player_id"] for r in chosen],
            pool=ctx.pool,
            adp_by_player_id=adp,
            drafted_ids=drafted_before_interval,
            model=ctx.opponent_model,
            n_picks=max(0, n_interval),
            n_sims=args.survival_sims,
            rng=np.random.default_rng(_path_seed(tuple(path), args.seed) + 1),
            n_teams=N_TEAMS,
            first_pick=(our_picks[depth - 1] + 1) if depth else 1,
        )

        print(
            f"  [{n_calls:>3}] {time.perf_counter() - t0:6.0f}s  R{depth + 1} pick {our_pick}"
            f"  path={'/'.join(ctx.pool[p].name.split()[-1] for p in path) or '(root)'}"
            f"  -> {', '.join(ctx.pool[r['player_id']].name for r in chosen)}",
            flush=True,
        )

        nodes = []
        for r in chosen:
            pid = r["player_id"]
            entry = ctx.pool[pid]
            child_counts = dict(counts)
            child_counts[r["position"]] = child_counts.get(r["position"], 0) + 1

            node = {
                "player_id": pid,
                "name": entry.name,
                "position": r["position"],
                "round": depth + 1,
                "overall_pick": our_pick,
                "adp": adp.get(pid),
                "proj_ppg": float(entry.distribution.mean),
                "championship_probability": r["championship_probability"],
                "standard_error": r["standard_error"],
                "gap_from_leader_pp": r["gap_from_leader_pp"],
                "indistinguishable_from_leader": r["indistinguishable_from_leader"],
                "p_available": surv.get(pid),
                "counts": child_counts,
                "children": [],
            }

            if depth + 1 < N_ROUNDS:
                child_picks = list(picks) + [
                    Pick(overall_pick=our_pick, team=DRAFT_SLOT,
                         player_id=pid, position=r["position"])
                ]
                node["children"] = expand(
                    child_picks, drafted_now | {pid},
                    path + [pid], child_counts, depth + 1,
                )
            nodes.append(node)
        return nodes

    roots = expand([], set(), [], {}, 0)
    elapsed = time.perf_counter() - t0
    print(f"\n{n_calls} recommend_pick calls in {elapsed:.0f}s", flush=True)

    return {
        "generated": time.strftime("%Y-%m-%d"),
        "draft_slot": DRAFT_SLOT,
        "n_teams": N_TEAMS,
        "our_picks": our_picks,
        "budget": budget.label,
        "survival_sims": args.survival_sims,
        "branching": list(BRANCHING),
        "n_recommend_calls": n_calls,
        "seconds": round(elapsed),
        "roots": roots,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = one per core (minus two); 1 = serial")
    ap.add_argument("--fast", action="store_true",
                    help="tiny budget: wiring check only, NOT for draft-day use")
    ap.add_argument("--survival-sims", type=int, default=SURVIVAL_SIMS)
    ap.add_argument("--out", type=Path, default=Path("data/pick_tree.json"))
    args = ap.parse_args()

    tree = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tree, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
