"""Fast, roster-aware marginal value for draft candidates (Stage 3 Task 3).

## What this is for

Two callers, both performance-critical, and neither wants the season
simulator:

1. **Pruning** ~500 available players down to ~15 candidates worth a full
   Monte Carlo evaluation (`draft/recommender.py`).
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

    lineup_marginal(p) = solve_lineup(roster + p) - solve_lineup(roster)

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
to go at all (QB is not FLEX-eligible in this league).

`solve_lineup` is monotonic in its roster argument in the *usual* case:
adding a candidate to the pool of choices can only match or beat the prior
optimum. There is one documented exception (see `sim/lineup.py`'s module
docstring, "Unavailable players and empty slots"): a real leftover player
always fills a FLEX slot ahead of replacement level, even if that player's
score is below the replacement mean it would otherwise borrow -- correct
for `solve_lineup`'s actual job (a specific week's already-realized score),
but it means the *raw* `lineup_marginal` computed here can occasionally go
slightly negative for a genuinely below-replacement mean player. The
`max()` with the bench term below neutralizes this (see
`test_value_is_never_negative`).

## Bench value: v2 -- reachability, not a flat discount

**This module previously priced every non-starting player's bench value at
a flat `0.20 x over_replacement`, regardless of how many players were
already ahead of them.** That was a real, shipped bug, caught only once a
full greedy rollout was built on top of this function and produced rosters
with six quarterbacks and six tight ends (see the task report for the
incident). The flat constant does not decay with depth at all -- a would-be
6th QB priced the same as a 2nd -- so nothing in the greedy policy ever
stopped stacking a position once its dedicated/FLEX slots were full,
because doing so was never actually worse than taking a useful player
elsewhere (a 4th-string QB's high *raw* mean, discounted by the same flat
0.20 applied to every position, could still outrank a real 5th WR's smaller
`over_replacement`).

**The fix models what bench value actually is: the probability this
specific player, at this specific depth, ever gets a real start**, times
what he's worth when he does:

    bench_value(p) = P(reaches lineup | depth) * over_replacement(p)

`P(reaches lineup | depth)` mirrors `solve_lineup`'s own two-phase
structure exactly (see `sim/lineup.py`'s module docstring) -- not a
separate hardcoded position-count notion:

- **Phase 1 (own dedicated slot).** Count how many currently-rostered
  players at `p`'s own position already have a mean at or above `p`'s
  (`same_pos_ahead`). If that count is below `starters[p.position]`, `p`
  can already start in his own dedicated slot outright -- `lineup_marginal`
  already prices this in full, so returning the undiscounted
  `over_replacement` here is harmless (the `max()` picks whichever is
  larger; there is no double count).
- **Phase 2 (FLEX), only for a FLEX-eligible position whose own dedicated
  slots are already saturated.** `solve_lineup` pools every FLEX-eligible
  position's *leftover* (the players beyond each position's own dedicated
  count) and takes the pool's top `starters['FLEX']`. This module computes
  that exact same leftover pool from `roster` (`_leftover_flex_pool`) and
  counts how many of *those* leftover players have a mean at or above
  `p`'s (`count_ahead`). Capacity here is `starters['FLEX']`. This is why a
  4th RB, sitting behind 2 RBs + 2 WRs + a TE that are all still within
  their own dedicated counts (leftover pool empty), reaches the FLEX pool
  essentially uncontested, while a QB with the same raw depth has no phase
  2 at all (QB is not FLEX-eligible) and is capped by phase 1 alone.
- **A non-FLEX-eligible position whose phase-1 capacity is saturated**
  (QB, K, DST) has no phase 2 -- `count_ahead`/capacity stay exactly
  `same_pos_ahead`/`starters[position]`.

`gap = count_ahead - capacity + 1` (whichever phase applies) is how many of
those `count_ahead` players ahead of `p` would need to be unavailable,
simultaneously, in a given week, for `p` to start. `gap <= 0` means `p`
reaches outright at that phase (same treatment as phase 1's outright case
above).

**The probability model.** Stage 2's availability fit
(`models/availability.py`) gives each position a per-week
`p_available` (e.g. QB 0.842, RB 0.804, WR 0.842, TE 0.797, K 0.934). Each
of the `count_ahead` players-ahead is treated as an *independent* weekly
Bernoulli(unavailable) draw at their own position's rate (their average,
when the group is mixed-position, i.e. RB/WR/TE) -- an approximation, not
a refit: it uses each week on its own terms (the fitted `p_available` is
already the marginal, stationary per-week rate, which is the right
quantity for "a given week's" reachability question), and it treats the
`count_ahead` outages as independent rather than modeling the Markov
persistence `models/availability.py` fits for a single player's own
time-series -- correlation *across* different players' injuries is a much
smaller, noisier effect than the marginal miss rate itself, and this
function's job is "roughly right and cheap," not a second full
availability-and-persistence refit. `P(reaches) = P(X >= gap)` for
`X ~ Binomial(count_ahead, q)`, computed in closed form (`_prob_at_least`,
plain `math.comb` -- `count_ahead` is at most a roster's worth of players,
so this is a handful of terms, not a scipy call in a multi-million-call hot
loop).

This reproduces every example in the incident report without any
position-specific code: a QB2 (`same_pos_ahead=1`, `capacity=1`, `gap=1`,
`q~0.158`) gets roughly `P(X>=1)=0.158` of his over-replacement value; a
QB3 (`same_pos_ahead=2`, `gap=2`) gets roughly `q^2~0.025`; a QB6 gets a
`gap` binomial tail so small it rounds to nothing. A 4th RB behind exactly
2 RBs + 2 WRs + 1 TE (every position still within its own dedicated count,
so the phase-2 leftover pool is empty) has `gap<=0` in phase 2 and starts
outright -- not discounted at all -- because FLEX's shared capacity is
still uncontested; only once the FLEX-eligible group's *leftover* pool
itself saturates does an extra RB/WR/TE start decaying the same way a QB
does.

A player never priced above their outright-starting value: `bench_value`
is a probability (<=1) times `over_replacement`, and `over_replacement`
itself floors at 0 for a below-replacement mean, so a below-replacement
player's `bench_value` is 0 regardless of depth.

**A second bug, caught only by testing the greedy policy end to end (not
`draft.value` in isolation), in the same fix.** `over_replacement`'s
baseline is *not* always `p`'s own position's replacement mean: a player
reached via phase 2 (FLEX) is displacing whatever an empty FLEX slot would
otherwise use, which is the *best* of every FLEX-eligible position's
replacement mean (matching `solve_lineup`'s own "an empty FLEX slot uses
the best of the group" rule) -- not `p`'s own position's, which can be
considerably lower. Using `p`'s own (lower) replacement mean here
overpriced any FLEX-only reach: concretely, TE's own replacement (8.1) is
well below the FLEX fallback (WR's 9.9, in this league), so a second or
third TE reachable only through FLEX was overvalued by that whole gap.
This alone was enough, in a real rollout built on the very first version of
this fix, to make an extra elite-mean TE outrank a comparable-or-better WR
and produce a real position-stacking regression -- caught by
`tests/test_rollout.py::test_real_greedy_rollout_produces_a_sane_position_mix`,
not by any unit test of `draft.value` alone. See `_bench_value`'s
docstring for exactly which baseline applies to which phase.

## Determinism

Nothing here samples. Every input is a fixed scalar or a closed-form
binomial tail; `solve_lineup` itself is a pure sort-and-slice. Two calls
with the same arguments return bit-identical results.

## Performance

One call is two `solve_lineup` invocations (baseline once per batch, one
more per candidate) plus one small closed-form binomial-tail sum, over a
roster of at most 18 players -- no parquet reads, no RNG, in the hot loop
itself. `availability_by_position` defaults to `models.availability.
availability_by_position()` (Stage 2's cached fit), loaded and memoized
**once per process** via `functools.lru_cache` -- the first call pays a
cache read, every subsequent call (this runs inside every rollout, at
every one of our own picks, across every candidate and every rollout) is a
plain dict reference, not a re-fit or a re-read. See
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
from functools import lru_cache
from math import comb
from typing import Mapping, Sequence

from ..league import FLEX_ELIGIBLE, STARTERS
from ..models.availability import PlayerAvailability
from ..models.availability import availability_by_position as _fit_availability_by_position
from ..models.distribution import PlayerDistribution
from ..sim.lineup import RosterPlayer, solve_lineup


@lru_cache(maxsize=1)
def _default_availability_by_position() -> dict[str, PlayerAvailability]:
    """Memoized so the (cached-on-disk, but still a real read) fit is paid
    once per process, not once per candidate -- see module docstring,
    "Performance"."""
    return _fit_availability_by_position()


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


def _leftover_flex_pool(
    roster: Sequence[RosterPlayer], starters: Mapping[str, int], flex_eligible: frozenset[str]
) -> list[RosterPlayer]:
    """The players `solve_lineup`'s phase 2 would actually see: for each
    FLEX-eligible position, everyone beyond that position's own top-
    `starters[position]` (by score), pooled across the whole group. Mirrors
    `sim.lineup.solve_lineup`'s reserve-then-pool phases exactly (see that
    module's docstring) so this module's notion of "how deep is `p` really"
    matches what the optimizer itself would do, not an approximation of it.
    """
    leftover: list[RosterPlayer] = []
    for pos in flex_eligible:
        same_pos = sorted((p for p in roster if p.position == pos), key=lambda p: p.score, reverse=True)
        count = starters.get(pos, 0)
        leftover.extend(same_pos[count:])
    return leftover


def _prob_at_least(n: int, k: int, q: float) -> float:
    """`P(X >= k)` for `X ~ Binomial(n, q)`, in closed form via `math.comb`.

    `n` here is at most a roster's worth of players (well under a hundred),
    so this direct sum is fast enough for a hot loop called millions of
    times across a recommender's rollouts -- see module docstring,
    "Performance", for why this avoids a `scipy.stats` call per candidate.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    return sum(comb(n, i) * (q**i) * ((1.0 - q) ** (n - i)) for i in range(k, n + 1))


def _weighted_miss_rate(
    ahead: Sequence[RosterPlayer], availability_by_position: Mapping[str, PlayerAvailability]
) -> float:
    """Average per-week unavailability rate across `ahead`'s own positions.
    A position missing from `availability_by_position` (DST has none -- see
    `models/roster.py`'s docstring, "DST has no separate replacement
    level": a team is modeled as never without a defense) contributes 0
    (never unavailable), not a `KeyError` -- "no availability model" means
    exactly that everywhere else in this project."""
    return sum(
        1.0 - availability_by_position[p.position].p_available
        for p in ahead
        if p.position in availability_by_position
    ) / len(ahead)


def _bench_value(
    position: str,
    mean: float,
    roster: Sequence[RosterPlayer],
    replacement_means: Mapping[str, float],
    availability_by_position: Mapping[str, PlayerAvailability],
    starters: Mapping[str, int],
    flex_eligible: frozenset[str],
) -> float:
    """`P(reaches lineup | depth) * over_replacement` -- see module
    docstring, "Bench value: v2", for the full two-phase derivation.

    **The baseline `over_replacement` is subtracted from depends on which
    phase actually reaches `p`.** A slot filled via `p`'s own dedicated
    count uses `p`'s own position's replacement mean (an empty dedicated
    slot always falls back to exactly that). A slot filled via FLEX uses
    the FLEX fallback instead -- `max` of every FLEX-eligible position's
    replacement mean, matching `solve_lineup`'s own "an empty FLEX slot
    uses the best of the group" rule exactly (see that module's docstring).
    Using `p`'s own (lower) replacement mean for a FLEX-phase reach was a
    real, caught bug: TE's own replacement level (8.1) is well below the
    FLEX fallback (WR's 9.9 here), so a second/third TE reaching only
    through FLEX was overpriced by that gap -- enough, in a real rollout,
    to make an extra elite-mean TE outrank a comparable-or-better WR and
    drive a real position-stacking regression (see
    `tests/test_rollout.py::test_real_greedy_rollout_produces_a_sane_position_mix`).
    """
    # Phase 1: p's own dedicated slots.
    own_capacity = starters.get(position, 0)
    same_pos_ahead = [p for p in roster if p.position == position and p.score >= mean]
    if len(same_pos_ahead) < own_capacity:
        return max(mean - replacement_means[position], 0.0)  # reaches his own slot outright

    if position in flex_eligible:
        # Phase 2: the pooled FLEX leftover, exactly as solve_lineup builds
        # it -- see `_leftover_flex_pool`. What an unfilled FLEX slot would
        # otherwise use is the best of the group's replacement means, not
        # p's own -- see docstring above.
        leftover = _leftover_flex_pool(roster, starters, flex_eligible)
        ahead = [p for p in leftover if p.score >= mean]
        capacity = starters.get("FLEX", 0)
        baseline = max(replacement_means[p] for p in flex_eligible)
    else:
        # No FLEX recourse for a non-eligible position (QB/K/DST): phase 1
        # is the whole story, and the baseline stays p's own position.
        ahead = same_pos_ahead
        capacity = own_capacity
        baseline = replacement_means[position]

    over_replacement = mean - baseline
    if over_replacement <= 0:
        return 0.0

    gap = len(ahead) - capacity + 1
    if gap <= 0:
        # Reaches outright at this phase.
        return over_replacement
    if not ahead:
        # gap > 0 with nobody actually ahead only happens when capacity
        # itself is 0 (e.g. a caller's STARTERS config with FLEX=0) --
        # there is truly no path to the lineup, so this is 0, not a
        # division by an empty list.
        return 0.0

    q = _weighted_miss_rate(ahead, availability_by_position)
    p_reaches = _prob_at_least(len(ahead), gap, q)
    return p_reaches * over_replacement


def player_value(
    player_id: str,
    pool: Mapping[str, PlayerDistribution],
    roster: Sequence[RosterPlayer],
    replacement_means: Mapping[str, float],
    availability_by_position: Mapping[str, PlayerAvailability] | None = None,
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
) -> PlayerValue:
    """Value a single candidate against a roster already drafted so far.

    `roster` is the drafted-so-far roster as `sim.lineup.RosterPlayer`
    (typically built by scoring each `models.roster.build_roster` result's
    players at their distribution `.mean`). `replacement_means` is the flat
    `{position: float}` dict `solve_lineup` expects (see
    `sim.lineup.build_replacement_means`). `availability_by_position`
    defaults to Stage 2's fitted, memoized rates (see module docstring,
    "Performance") -- pass an explicit dict (as tests do) for a fully
    hermetic, deterministic call independent of real cached data.

    Prefer `value_available` when scoring many candidates against the same
    roster -- it computes the shared baseline lineup once instead of once
    per call.
    """
    return value_available(
        [player_id],
        pool,
        roster,
        replacement_means,
        availability_by_position=availability_by_position,
        starters=starters,
        flex_eligible=flex_eligible,
    )[0]


def value_available(
    available_ids: Sequence[str],
    pool: Mapping[str, PlayerDistribution],
    roster: Sequence[RosterPlayer],
    replacement_means: Mapping[str, float],
    availability_by_position: Mapping[str, PlayerAvailability] | None = None,
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
) -> list[PlayerValue]:
    """Value every player in `available_ids` against the same roster.

    This is the hot-loop entry point: pruning ~500 available players calls
    this once, not 500 times with a re-derived baseline. The baseline
    lineup (`roster` alone) is solved exactly once; each candidate then
    costs one additional `solve_lineup` call over `roster + [candidate]`
    plus one closed-form bench-reachability calculation.

    See the module docstring for the value formula (lineup marginal,
    floored against a depth/FLEX-capacity-aware bench value) and why it is
    exact for FLEX without any position-count bookkeeping of its own.
    """
    if availability_by_position is None:
        availability_by_position = _default_availability_by_position()

    baseline = solve_lineup(roster, replacement_means, starters, flex_eligible).total_points

    results: list[PlayerValue] = []
    for player_id in available_ids:
        entry = pool[player_id]
        candidate = _to_roster_player(player_id, entry)
        with_player = solve_lineup(
            list(roster) + [candidate], replacement_means, starters, flex_eligible
        ).total_points
        lineup_marginal = with_player - baseline

        mean = float(entry.distribution.mean)
        bench_value = _bench_value(
            entry.position, mean, roster, replacement_means, availability_by_position, starters, flex_eligible
        )

        value = max(lineup_marginal, bench_value)
        results.append(
            PlayerValue(player_id=player_id, position=entry.position, value=value, mean=mean)
        )
    return results
