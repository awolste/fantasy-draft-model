"""Precompute full-budget recommendations for the states we are most likely
to face, and report the hit rate that results.

States are **sampled from the opponent model**, not guessed: simulate many
drafts forward, record the state key at each of our picks, and precompute
the most frequent keys. That also yields a measured hit rate per round --
the number that says whether precompute is doing anything.

Expect good coverage in rounds 1-3 and thin coverage later; the state space
branches faster than any cache can follow. That is what the live fallback
is for, and it is not a defect. **Do not inflate --top-k until the number
looks good**: a cache full of states we will never see is not coverage.

## The trap this avoids

The naive version records keys but not the boards that produced them. The
key is lossy, so a board cannot be reconstructed from it -- and the script
then writes an empty cache while printing encouraging coverage percentages.
That looks exactly like success. So sampling stores a representative
`DraftState` per key, and the script refuses to save a cache with fewer
entries than it claims to have precomputed.

Run this AFTER re-running the ingest near draft day. A cache built against
a stale pool will refuse to load (fingerprint guard in `live/cache.py`).

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ffdraft.draft.rollout import DraftState, run_rollout
from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET
from ffdraft.live.cache import (
    RecommendationCache,
    candidates_to_rows,
    recommend,
    state_key,
)
from ffdraft.live.context import live_context
from ffdraft.store import fingerprint

CACHE_PATH = Path("data/recommendation_cache.json")


def _our_roster_counts(picks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in picks:
        if p.team == DRAFT_SLOT:
            counts[p.position] = counts.get(p.position, 0) + 1
    return counts


def sample_states(ctx, n_sims: int, our_rounds: list[int]):
    """Simulate `n_sims` drafts and record, per round, how often each state
    key occurs and one representative board that produced it."""
    counts: dict[int, Counter] = {r: Counter() for r in our_rounds}
    representative: dict[int, dict] = {r: {} for r in our_rounds}

    for seed in range(n_sims):
        rng = np.random.default_rng(seed)
        empty = DraftState.from_picks([], n_teams=N_TEAMS, rounds=DRAFT_ROUNDS)
        final = run_rollout(
            empty, ctx.pool, ctx.opponent_model, ctx.replacement_by_position, rng,
            our_team=DRAFT_SLOT, adp_table=ctx.adp_table, rankings=ctx.rankings,
        )
        our_picks = [p.overall_pick for p in final.picks if p.team == DRAFT_SLOT]
        for r in our_rounds:
            target = our_picks[r - 1]
            before = [p for p in final.picks if p.overall_pick < target]
            key = state_key(
                target, _our_roster_counts(before), {p.player_id for p in before}, ctx.pool
            )
            counts[r][key] += 1
            # Store the board itself, not just the key: the key is lossy and
            # a board cannot be rebuilt from it.
            representative[r].setdefault(
                key, DraftState.from_picks(before, n_teams=N_TEAMS, rounds=DRAFT_ROUNDS)
            )
    return counts, representative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=2000,
                        help="draft simulations to sample states from")
    parser.add_argument("--top-k", type=int, default=40,
                        help="most-frequent states to precompute per round")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3],
                        help="which of OUR rounds to precompute")
    parser.add_argument("--out", type=Path, default=CACHE_PATH)
    args = parser.parse_args()

    ctx = live_context()
    fp = fingerprint(ctx.rankings, ctx.adp_table)

    print(f"sampling {args.n_sims} drafts to find reachable states...", flush=True)
    t0 = time.perf_counter()
    counts, representative = sample_states(ctx, args.n_sims, args.rounds)
    print(f"  sampled in {time.perf_counter() - t0:.0f}s\n", flush=True)

    cache = RecommendationCache(path=args.out, fingerprint=fp)
    expected = 0

    for r in args.rounds:
        total = sum(counts[r].values())
        top = counts[r].most_common(args.top_k)
        covered = sum(c for _, c in top)
        print(
            f"round {r}: {len(counts[r]):>6} distinct states | "
            f"top {args.top_k} cover {100 * covered / total:5.1f}% of simulated drafts",
            flush=True,
        )
        for key, _count in top:
            board = representative[r][key]
            if board.team_on_clock != DRAFT_SLOT:
                raise RuntimeError(
                    f"sampled board for round {r} has team {board.team_on_clock} on "
                    f"the clock, not ours ({DRAFT_SLOT}) -- sampling is wrong."
                )
            cache.put(key, candidates_to_rows(recommend(board, ctx, FULL_BUDGET)))
            expected += 1
        print(f"  precomputed {expected} entries so far "
              f"({time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    if len(cache) != expected:
        raise RuntimeError(
            f"cache holds {len(cache)} entries but {expected} were precomputed -- "
            "keys are colliding. Refusing to write a cache that silently lost entries."
        )
    if expected == 0:
        raise RuntimeError("precomputed nothing; refusing to write an empty cache")

    cache.save()
    print(f"\nwrote {args.out}: {len(cache)} entries, fingerprint {fp[:12]}...")


if __name__ == "__main__":
    main()
