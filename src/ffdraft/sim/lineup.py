"""Weekly lineup optimizer: given a roster's already-realized weekly scores,
fill the starting lineup (`league.STARTERS`) to maximize total points.

This module does not sample anything -- Task 4 (availability) and Task 3/
distribution.py already decided who played and what they scored; this module
just solves the resulting (small) assignment problem exactly.

## Why greedy-by-position is not enough, and what this does instead

The naive greedy -- "fill each position's dedicated slots with its own top
scorers, one position at a time, sending each position's leftovers to FLEX
as you go" -- is not optimal, because it can lock a mediocre leftover into a
FLEX slot before a later position's much better leftover is even
considered. See `tests/test_lineup.py::test_naive_sequential_greedy_loses_to_optimal_solver`
for a concrete numeric case where this costs real points.

This module instead uses a two-phase algorithm that is provably exact for
this specific slot shape (each dedicated position has a lower-bound
requirement only, FLEX has no position restriction beyond
`league.FLEX_ELIGIBLE`, and a player can fill at most one slot):

**Phase 1 (reserve).** For every position with a dedicated slot count,
reserve its own top-`count` available players for those dedicated slots.
Any position players beyond that reserved count, for positions that are
FLEX-eligible, go into a shared "leftover" pool.

**Phase 2 (flex).** Sort the leftover pool (pooled across *every*
FLEX-eligible position, not filled position-by-position) by score and take
the top `FLEX` count for the FLEX slots.

**Why this is exactly optimal:**

1. *Reserving the top-`count` of a position for that position's own
   dedicated slots is lossless.* For any optimal lineup, look at whichever
   `count` players of position P it actually started (dedicated or FLEX
   combined -- for a fixed *number* of position-P players started, the
   *identity* of which specific dedicated slot label they occupy cannot
   change the total). If that started set is not exactly the top-`count`
   by score, swapping in a higher-scoring available P player for a
   lower-scoring started one (both position P) cannot decrease the total,
   and strictly increases it if scores differ. So WLOG every optimal
   lineup starts each position's own top-`count` players via *some*
   assignment to that position's dedicated slots -- which is exactly what
   phase 1 does, unconditionally.

2. *Given phase 1's reservation, allocating FLEX slots by pooling every
   remaining player and taking the overall top-`FLEX` is optimal.* Each
   FLEX-eligible position's own remaining players are already sorted
   (phase 1 removed exactly its best `count`, so what's left is that
   position's own descending tail). Choosing `FLEX` additional players from
   several already-sorted, mutually exclusive sequences to maximize the sum
   is solved by merging the sequences and taking the global top-`FLEX` --
   equivalent to a `FLEX`-way merge / exchange argument: if some chosen
   leftover player scores lower than some un-chosen leftover player (any
   position), swapping them can only help. No position-level upper bound
   exists on FLEX (a FLEX slot can go to any FLEX-eligible position), so
   there is no constraint stopping this merge-and-take-top from being
   reached.

This is verified two ways in `tests/test_lineup.py`: an exhaustive brute
force comparison on many small randomized rosters, and an independent
cross-check against `scipy.optimize.linear_sum_assignment` (a generic,
well-established exact solver for weighted bipartite matching) used purely
as a test oracle.

## Unavailable players and empty slots

A player with `available=False` is dropped before phase 1 even runs -- he
is not a body this method or any other can start, however high his score.

**Replacement level applies only when there is no real, available,
eligible body left for a slot -- a quantity trigger, not a quality one.**
If a position has fewer available players than its dedicated-slot count,
the unfilled dedicated slots use that position's replacement mean
(`replacement_means[position]`). If FLEX has fewer leftover candidates than
its slot count, the unfilled FLEX slots use the *best* of the FLEX-eligible
positions' replacement means (`max(replacement_means[p] for p in
flex_eligible)`) -- a rational manager streaming a FLEX spot picks whichever
position's replacement pool is currently most productive, not a fixed one.

This deliberately does *not* let replacement level outbid a real,
available player who simply had a bad week: a rostered, healthy player who
scored below replacement level is still started if he is the only body for
that slot, because he *is* a body -- "the roster cannot fill a slot" (this
module's trigger condition, per the Task 6 spec) is a statement about
headcount, not about that week's realized quality. A model that let a mean
replacement value outbid an actual, already-realized (hindsight-known)
score would be comparing an average expectation to a perfect-hindsight
number, which is not the well-defined question this module answers.

## Interface / performance

`solve_lineup` is pure Python over a roster of ~18 players -- cheap per
call (sorting and slicing small lists), but Task 7 needs ~140,000 calls
per full run. See the module's `__main__`-adjacent benchmark note in the
Task 6 report for actual timings; if pure-Python-per-call throughput ever
becomes the bottleneck, the fix is to vectorize this same two-phase
algorithm across many (team, week, sim) scenarios at once with numpy
(argsort per axis, no scipy/LP call), not to swap in a cheaper but
inexact heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..league import FLEX_ELIGIBLE, STARTERS


@dataclass(frozen=True)
class RosterPlayer:
    """One roster player's already-realized score for the week being
    optimized. `position` must be one of the keys of `starters` (excluding
    `"FLEX"`, which is never a real player's position)."""

    player_id: str
    position: str
    score: float
    available: bool = True


@dataclass(frozen=True)
class SlotResult:
    """One filled (or replacement-filled) lineup slot.

    `slot` is the dedicated position label (e.g. "RB") or "FLEX". When
    `is_replacement` is True, `player_id` is None and `points` is the
    replacement-level value used to score that empty slot -- never zero,
    never a crash.
    """

    slot: str
    player_id: str | None
    points: float
    is_replacement: bool


@dataclass(frozen=True)
class LineupResult:
    slots: tuple[SlotResult, ...]
    total_points: float


def _validate_replacement_means(
    replacement_means: Mapping[str, float],
    starters: Mapping[str, int],
    flex_eligible: frozenset[str],
) -> None:
    needed = {pos for pos, count in starters.items() if pos != "FLEX" and count > 0}
    needed |= set(flex_eligible)
    missing = needed - set(replacement_means)
    if missing:
        raise ValueError(
            f"replacement_means is missing entries for {sorted(missing)} -- "
            "every dedicated position (and every FLEX-eligible position) needs "
            "a replacement-level value so empty slots never fall back to zero."
        )


def _validate_roster_positions(
    roster: Sequence[RosterPlayer], starters: Mapping[str, int], flex_eligible: frozenset[str]
) -> None:
    known = {pos for pos in starters if pos != "FLEX"} | set(flex_eligible)
    unknown = {p.position for p in roster} - known
    if unknown:
        raise ValueError(
            f"roster contains player(s) at unrecognized position(s) {sorted(unknown)} -- "
            f"expected one of {sorted(known)} (from league.STARTERS / league.FLEX_ELIGIBLE)."
        )


def solve_lineup(
    roster: Sequence[RosterPlayer],
    replacement_means: Mapping[str, float],
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
) -> LineupResult:
    """Fill the starting lineup (shape read from `starters`/`flex_eligible`,
    defaulting to `league.STARTERS`/`league.FLEX_ELIGIBLE`) to maximize total
    points, using only `roster` players with `available=True`.

    See the module docstring for the two-phase algorithm and why it is
    exactly optimal, and for how unavailable players and thin rosters are
    handled.

    `replacement_means` must have an entry for every dedicated position with
    a nonzero slot count and every FLEX-eligible position -- typically
    `{"QB": ..., "RB": ..., "WR": ..., "TE": ..., "K": ..., "DST": ...}`,
    e.g. from `replacement.replacement_by_position()[pos].mean` plus
    `defense.dst_distribution().mean`. Callers should compute this dict once
    outside any per-lineup hot loop.
    """
    _validate_replacement_means(replacement_means, starters, flex_eligible)
    _validate_roster_positions(roster, starters, flex_eligible)

    available = [p for p in roster if p.available]

    # Union, not just starters' own keys: a FLEX-eligible position with no
    # dedicated slot at all (count 0, via starters.get below) still needs to
    # contribute its players to the leftover pool for FLEX.
    dedicated_positions = sorted({pos for pos in starters if pos != "FLEX"} | set(flex_eligible))

    slot_results: list[SlotResult] = []
    leftover: list[RosterPlayer] = []

    for pos in dedicated_positions:
        count = starters.get(pos, 0)
        pool = sorted((p for p in available if p.position == pos), key=lambda p: p.score, reverse=True)
        chosen, rest = pool[:count], pool[count:]
        if pos in flex_eligible:
            leftover.extend(rest)
        for i in range(count):
            if i < len(chosen):
                slot_results.append(SlotResult(pos, chosen[i].player_id, chosen[i].score, False))
            else:
                slot_results.append(SlotResult(pos, None, replacement_means[pos], True))

    flex_count = starters.get("FLEX", 0)
    if flex_count:
        leftover_sorted = sorted(leftover, key=lambda p: p.score, reverse=True)
        flex_fill = leftover_sorted[:flex_count]
        flex_replacement = max(replacement_means[p] for p in flex_eligible)
        for i in range(flex_count):
            if i < len(flex_fill):
                slot_results.append(SlotResult("FLEX", flex_fill[i].player_id, flex_fill[i].score, False))
            else:
                slot_results.append(SlotResult("FLEX", None, flex_replacement, True))

    total = sum(s.points for s in slot_results)
    return LineupResult(slots=tuple(slot_results), total_points=total)


def build_replacement_means(
    replacement_by_position: Mapping[str, object],
    dst_mean: float,
) -> dict[str, float]:
    """Convenience for callers (Task 7): flatten `replacement.
    replacement_by_position()` (position -> object with a `.mean` property,
    i.e. `ReplacementLevelDistribution`) plus the shared DST mean (`defense.
    dst_distribution().mean`) into the flat `{position: float}` dict
    `solve_lineup` expects. Call this once per simulation run, not per
    lineup solve -- `replacement_by_position()` itself hits cached storage
    and is not meant to run 140,000 times."""
    means = {pos: float(dist.mean) for pos, dist in replacement_by_position.items()}
    means["DST"] = float(dst_mean)
    return means
