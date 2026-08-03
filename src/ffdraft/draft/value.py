"""Fast, roster-aware marginal value for draft candidates (Stage 3 Task 3).

## What this is for

Two callers, both performance-critical, and neither wants the season
simulator:

1. **Pruning** ~500 available players down to ~15 candidates worth a full
   Monte Carlo evaluation (`draft/recommender.py`, a later task).
2. **Choosing our own future picks inside a rollout** (`draft/rollout.py`).
   A rollout plays the draft forward many times; a nested Monte Carlo at
   each of our own future picks would be far too expensive, so this greedy,
   deterministic function stands in.

Both callers need this to be cheap and roughly right, not slow and exact --
the season simulator remains the source of truth for anything that matters
enough to spend a Monte Carlo budget on.

## The formulation

A player's value is their **marginal contribution to the projected starting
lineup**, over replacement, given the roster drafted so far:

    value(p) = solve_lineup(roster + p) - solve_lineup(roster)

using each player's *mean* weekly points (the `.mean` of their fitted
distribution) as a stand-in for a realized score -- deterministic, no
sampling, matching every other "fast path" convention in this project's
draft-time code.

`solve_lineup` (`sim/lineup.py`) is already the exact optimizer for this
league's lineup shape (`league.STARTERS`/`FLEX_ELIGIBLE`), including FLEX,
so it is reused rather than re-derived here. This is also why FLEX capacity
falls out correctly for free: `solve_lineup` pools every FLEX-eligible
leftover and takes the global top-N, so a third good RB competes for a FLEX
slot on equal footing with leftover WRs/TEs, while a fourth QB has nowhere
to go at all (QB is not FLEX-eligible in this league). This function does
not need its own separate "how many of this position can start" accounting;
it inherits the optimizer's, which is provably exact (see `sim/lineup.py`'s
module docstring).

`solve_lineup` is monotonic in its roster argument in the *usual* case:
adding a candidate to the pool of choices can only match or beat the prior
optimum. There is one documented exception (see `sim/lineup.py`'s module
docstring, "Unavailable players and empty slots"): a real leftover player
always fills a FLEX slot ahead of replacement level, even if that player's
score is below the replacement mean it would otherwise borrow -- correct
for `solve_lineup`'s actual job (a specific week's already-realized score,
where a rostered body is never worse than "didn't play"), but it means the
*raw* lineup marginal computed here can occasionally go slightly negative
for a genuinely below-replacement mean player. The `max()` with the bench
term below neutralizes this automatically: such a player also has
`over_replacement <= 0`, so `bench_value` is `0` too, and the reported
`value` floors at `0` rather than going negative -- see
`test_value_is_never_negative`.

## Bench value is not zero

A player who cannot presently crack the starting lineup (a 4th RB when both
RB slots and both FLEX slots are already better-occupied, a 2nd QB in this
one-QB league) still has *some* value: insurance against a starter missing
time. Stage 2's availability fit (`models/availability.py`) measured RB
per-week availability at 80.4% (a starter misses about 1 week in 5) with
75.5% persistence once out (absences cluster, they are not one-off). A
purely additive lineup-marginal calculation prices that at exactly zero,
which is wrong in the same direction the plan doc warns about generically
("plausible-looking numbers that raise no errors").

This module prices bench value crudely and says so: `BENCH_DISCOUNT = 0.20`
applied to the player's points-over-replacement, representing "this player
is worth roughly a fifth of a full starter's edge, because he only actually
plays when the starter(s) ahead of him miss a week" -- 0.20 is chosen to
sit at the RB miss rate (19.6%) established in Stage 2, used here as the
single representative rate across positions rather than fitting one rate
per position from the same fairly thin availability data a second time
(false precision the plan doc explicitly warns against). A single flat
constant, applied only when it exceeds the (non-negative) lineup marginal,
is preferred to a fitted-looking number this task was never asked to
derive.

    value(p) = max(lineup_marginal(p), BENCH_DISCOUNT * over_replacement(p))

Taking the max (not a sum) avoids double-counting: whenever a player
*would* start, `lineup_marginal` already captures that value in full, and
it is always at least as large as the discounted bench figure for a
meaningfully-above-replacement player (0.20 of a quantity is smaller than
the whole quantity). The bench term only binds when the player cannot
presently start at all.

## Determinism

Nothing here samples. Every input is a fixed scalar (`.mean` of a fitted
distribution, or a replacement-level mean); `solve_lineup` itself is a pure
sort-and-slice. Two calls with the same arguments return bit-identical
results.

## Performance

One call is two `solve_lineup` invocations (baseline once per batch, one
more per candidate) over a roster of at most 18 players -- sorting/slicing
short lists, no parquet reads, no RNG. See
`tests/test_value.py::test_value_available_is_fast_for_a_500_player_pool`
for a synthetic-pool timing regression guard; real 2026-pool numbers are
reported in the task writeup.

## Missing players

`available_ids`/`player_id` must already exist in `pool` -- a lookup miss
raises `KeyError` rather than silently skipping or zero-filling the player,
per this project's "loud failures over silent defaults" principle. Callers
that source `available_ids` from the same pool they pass in (the normal
case) cannot hit this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..league import FLEX_ELIGIBLE, STARTERS
from ..models.distribution import PlayerDistribution
from ..sim.lineup import RosterPlayer, solve_lineup

# See module docstring, "Bench value is not zero": a flat fraction of a
# benched player's points-over-replacement, representative of Stage 2's
# fitted RB per-week miss rate (19.6%, i.e. p_available=0.804), used as one
# crude constant rather than a per-position refit of the same thin data.
BENCH_DISCOUNT = 0.20


@dataclass(frozen=True)
class PlayerValue:
    """One available player's roster-aware marginal value.

    `mean` is carried through for callers that want to display or sanity
    check the raw per-week projection alongside the derived `value`.
    """

    player_id: str
    position: str
    value: float
    mean: float


def _to_roster_player(player_id: str, entry: PlayerDistribution) -> RosterPlayer:
    return RosterPlayer(player_id=player_id, position=entry.position, score=float(entry.distribution.mean))


def player_value(
    player_id: str,
    pool: Mapping[str, PlayerDistribution],
    roster: Sequence[RosterPlayer],
    replacement_means: Mapping[str, float],
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
    bench_discount: float = BENCH_DISCOUNT,
) -> PlayerValue:
    """Value a single candidate against a roster already drafted so far.

    `roster` is the drafted-so-far roster as `sim.lineup.RosterPlayer`
    (typically built by scoring each `models.roster.build_roster` result's
    players at their distribution `.mean`). `replacement_means` is the flat
    `{position: float}` dict `solve_lineup` expects (see
    `sim.lineup.build_replacement_means`).

    Prefer `value_available` when scoring many candidates against the same
    roster -- it computes the shared baseline lineup once instead of once
    per call.
    """
    return value_available(
        [player_id],
        pool,
        roster,
        replacement_means,
        starters=starters,
        flex_eligible=flex_eligible,
        bench_discount=bench_discount,
    )[0]


def value_available(
    available_ids: Sequence[str],
    pool: Mapping[str, PlayerDistribution],
    roster: Sequence[RosterPlayer],
    replacement_means: Mapping[str, float],
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
    bench_discount: float = BENCH_DISCOUNT,
) -> list[PlayerValue]:
    """Value every player in `available_ids` against the same roster.

    This is the hot-loop entry point: pruning ~500 available players calls
    this once, not 500 times with a re-derived baseline. The baseline
    lineup (`roster` alone) is solved exactly once; each candidate then
    costs one additional `solve_lineup` call over `roster + [candidate]`.

    See the module docstring for the value formula (lineup marginal, floored
    against a discounted bench value) and why it is exact for FLEX without
    any position-count bookkeeping of its own.
    """
    baseline = solve_lineup(roster, replacement_means, starters, flex_eligible).total_points

    results: list[PlayerValue] = []
    for player_id in available_ids:
        entry = pool[player_id]
        candidate = _to_roster_player(player_id, entry)
        with_player = solve_lineup(
            list(roster) + [candidate], replacement_means, starters, flex_eligible
        ).total_points
        lineup_marginal = with_player - baseline

        over_replacement = float(entry.distribution.mean) - replacement_means[entry.position]
        bench_value = bench_discount * max(over_replacement, 0.0)

        value = max(lineup_marginal, bench_value)
        results.append(
            PlayerValue(
                player_id=player_id,
                position=entry.position,
                value=value,
                mean=float(entry.distribution.mean),
            )
        )
    return results
