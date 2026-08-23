"""Will he still be there at my next pick?

## Why this exists

`draft.recommender.recommend_pick` answers "which available player
maximises title probability if I take him **now**". It does not answer
whether waiting is free. Its rollouts model opponents faithfully, but our
*own* future picks use the greedy `draft.value` policy, which has no notion
of who will still be available -- so the "skip him now, take him next
round" counterfactual is systematically undervalued and the tool reaches.
This is recorded in `docs/HANDOFF.md` section 7b as a known structural gap.

## Why survival is NOT blended into the recommendation

Championship probability is meant to *be* the whole objective. If the
rollouts modelled our future picks properly, survival would already be
inside it. Survival is therefore a **correction for a defect**, not a
second objective to trade against title equity -- and a weighted blend of
the two (`0.7*champ + 0.3*survival`) would be arbitrary: they share no
units and no principled weight, and it would bury the defect behind a knob.

What *is* principled is the decision-relevant quantity, in real units of
championship percentage points:

    wait_cost = (1 - P(survives)) * (champ_him - champ_best_likely_survivor)

"The chance he is gone, times how much worse my fallback is." The ranking
stays on championship probability -- unchanged and still the model's own
objective -- and `wait_cost_pp` is surfaced beside it so a high-equity
candidate who is 90% likely to last can be deferred deliberately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..draft.rollout import team_for_pick
from ..league import DRAFT_ROUNDS
from ..models.opponent import build_opponent_board, sample_pick_on_board

# RAISED 300 -> 3000 on 2026-08-22, because `wait_cost_pp` became the
# **primary tiebreaker** the app shows (`tiebreak.describe_tie` orders a
# tie by it), so this simulation's noise now propagates straight into a
# decision rather than sitting beside one.
#
# Measured at our pick 13 with 14 intervening picks -- the long wait, and
# so the worst case -- as the SD of `p_survive` across 8 independent seeds:
#
#   n_sims    cost   SD of p_survive (mid-range players)
#      300    0.28s   2.8pp        <- old: "49%" and "55%" were one number
#     1000    0.88s   1.4pp
#     3000    1.44s   0.9pp        <- now
#
# ~1.2s on a ~17s pick (HANDOFF section 15) against a 45s clock, to cut the
# noise in the number the owner is told to decide on by roughly 3x. Cost is
# sublinear in `n_sims` because building the opponent board is fixed cost.
DEFAULT_N_SIMS = 3000
# A candidate at or above this is worth deferring, and is eligible to be the
# fallback in `wait_cost_pp`. Judgement, not a fitted value -- it only sets
# which candidate the cost is measured against.
LIKELY_SURVIVOR = 0.5


def picks_until_our_next_turn(
    overall_pick: int,
    n_teams: int,
    our_team: int,
    total_picks: int | None = None,
) -> int | None:
    """How many opponent picks fall between this pick of ours and the next.

    Returns `None` at our final pick, where waiting is not an option and
    survival is undefined rather than 0 or 1. Raises if `overall_pick` is
    not ours -- asking "how long until my next turn" from someone else's
    pick is a caller bug.

    At slot 8 this is deeply asymmetric: 4 picks from 8 to 13 (the turn),
    but 14 from 13 to 28. A single intuition about "will he last" is wrong
    half the time because of it.
    """
    if team_for_pick(overall_pick, n_teams) != our_team:
        raise ValueError(
            f"overall pick {overall_pick} belongs to team "
            f"{team_for_pick(overall_pick, n_teams)}, not ours ({our_team})"
        )
    if total_picks is None:
        total_picks = DRAFT_ROUNDS * n_teams
    for nxt in range(overall_pick + 1, overall_pick + 2 * n_teams + 1):
        if nxt > total_picks:
            return None  # the draft ends first; there is no next turn
        if team_for_pick(nxt, n_teams) == our_team:
            return nxt - overall_pick - 1
    return None


def survival_probabilities(
    candidate_ids: Sequence[str],
    pool: Mapping[str, Any],
    adp_by_player_id: Mapping[str, float],
    drafted_ids: set[str],
    model: Any,
    n_picks: int,
    n_sims: int,
    rng: np.random.Generator,
    n_teams: int = 10,
    first_pick: int = 1,
) -> dict[str, float]:
    """Fraction of `n_sims` simulations in which each candidate is still
    available after `n_picks` opponent picks.

    Cheap by design: this simulates only the intervening picks, with no
    season simulation behind them, so it costs a fraction of one
    `recommend_pick` rollout.
    """
    if n_picks <= 0:
        return dict.fromkeys(candidate_ids, 1.0)

    # Same board the rollouts use (see `models.opponent.OpponentBoard`).
    # The `rank` fallback is applied here rather than inside the board,
    # which insists on full ADP coverage, so this keeps the tolerant
    # behaviour its own callers rely on without loosening the board's.
    board = build_opponent_board(
        model,
        pool,
        {pid: adp_by_player_id.get(pid, float(pool[pid].rank)) for pid in pool},
    )
    start_alive = np.fromiter(
        (pid not in drafted_ids for pid in board.player_ids), dtype=bool, count=board.size
    )
    candidate_idx = [(pid, board.index_of[pid]) for pid in candidate_ids]
    survived = dict.fromkeys(candidate_ids, 0)

    for _ in range(n_sims):
        alive = start_alive.copy()
        roster_counts: dict[int, dict[str, int]] = {}
        for offset in range(n_picks):
            if not alive.any():
                break
            team = team_for_pick(first_pick + offset, n_teams)
            counts = roster_counts.setdefault(team, {})
            taken = sample_pick_on_board(
                board, alive, counts, rng,
                # Survival must use the same per-round temperature the
                # rollouts do, or "will he last?" would be answered by a
                # differently-behaved league than the one being simulated.
                round_=(first_pick + offset - 1) // n_teams + 1,
            )
            alive[taken] = False
            pos = pool[board.player_ids[taken]].position
            counts[pos] = counts.get(pos, 0) + 1
        for pid, idx in candidate_idx:
            # `not start_alive[idx]` preserves an edge of the list-based
            # version this replaced: a candidate already drafted before the
            # call could never appear in the "taken" set, so it counted as
            # surviving every simulation. No caller passes one -- candidates
            # come from a recommendation over available players -- but a
            # rewrite billed as exactly output-preserving has to include the
            # corners nobody reaches.
            if alive[idx] or not start_alive[idx]:
                survived[pid] += 1

    return {pid: survived[pid] / n_sims for pid in candidate_ids}


def wait_cost_pp(
    champ: float,
    p_survive: float | None,
    champ_best_survivor: float,
) -> float | None:
    """Expected championship percentage points lost by passing on him now.

    `(1 - p_survive) * (champ - champ_best_survivor)`, clamped at zero and
    returned in percentage points. `None` when there is no next turn.

    Clamped because a fallback that scores *higher* means waiting is free,
    not beneficial -- a negative "cost" would read as a reason to wait when
    it is really a reason to take the fallback instead.
    """
    if p_survive is None:
        return None
    return max(0.0, (1.0 - p_survive) * (champ - champ_best_survivor)) * 100.0


def annotate_rows(
    rows: list[dict],
    survival: Mapping[str, float],
    has_next_turn: bool,
) -> list[dict]:
    """Attach `p_survive` and `wait_cost_pp` to recommendation rows.

    Ordering is left untouched: rows stay sorted by championship
    probability, which is the model's own objective. This adds information
    for the human, it does not silently re-rank on an unvalidated score.
    """
    if not has_next_turn:
        return [{**r, "p_survive": None, "wait_cost_pp": None} for r in rows]

    survivors = [
        r["championship_probability"]
        for r in rows
        if survival.get(r["player_id"], 0.0) >= LIKELY_SURVIVOR
    ]
    best_survivor = max(survivors) if survivors else 0.0

    return [
        {
            **r,
            "p_survive": survival.get(r["player_id"]),
            "wait_cost_pp": wait_cost_pp(
                r["championship_probability"],
                survival.get(r["player_id"]),
                best_survivor,
            ),
        }
        for r in rows
    ]
