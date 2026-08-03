"""Turn a list of draft picks into simulator-ready roster players.

Stage 2 built this operation twice, informally, in two scripts:
`scripts/season_report.py`'s `build_2024_rosters` (real 2024 ESPN draft
picks, resolved via `weekly_stats`/crosswalk) and `scripts/
correlation_impact.py`'s `_make_roster_players` (synthetic picks built
directly from the 2026 pool). Neither was tested, and Stage 3's Monte Carlo
rollout performs this same operation thousands of times per candidate pick
-- so it is promoted here as one tested, fast, library entry point.

## What a "pick" is

A pick is `(player_id, position)`: the id used to look the player up in the
pool, plus the position the caller believes that player occupies. Position
must travel with the id because the pool cannot always supply it -- when a
pick is *not* in the pool (see below), there is nothing left to ask for a
position, and the fallback needs one to pick the right replacement
distribution. Real callers already have this pair before they get here: a
draft pick resolves to a specific position (via `weekly_stats`, ESPN's own
labeling, or the pool itself) well before roster construction runs.

## The raise-vs-fallback line

A pick absent from `pool` is the *normal* case, not an error: seven
skill players are excluded from the 2026 pool outright (unmatched names,
see `distribution.build_player_pool`), historical drafts contain players
from outside the current pool entirely, and -- the case Stage 3 cares about
most -- a rollout evaluates rosters with most slots still empty. All of
these fall back to that position's replacement-level distribution, and the
fallback is counted and returned to the caller (see `RosterBuildResult`),
never printed or swallowed.

What raises instead is input that cannot describe a real pick no matter
what the pool contains:

- `player_id` that is not a non-empty string.
- `position` that is not one of this league's six roster positions
  (`QB`, `RB`, `WR`, `TE`, `K`, `DST` -- `league.FLEX_ELIGIBLE` plus the
  three non-FLEX-eligible starting positions).
- a `position` for which `replacement_by_position` has no entry at all --
  this can only happen when the *caller* passed an incomplete mapping (a
  configuration bug), since every valid position has a defined replacement
  level in this project (see below for DST's).

Note what deliberately does *not* raise: a pick whose pool entry's own
`position` differs from the caller-supplied `position`. Real historical
data is allowed to disagree with itself at the margins (a hybrid usage
case, a source labeling quirk) without that becoming a hard crash here;
when the id is found in the pool, the pool's own `PlayerDistribution.
position` is authoritative for the resulting `SeasonRosterPlayer`, and the
caller-supplied position only matters for choosing a replacement/
availability distribution on the fallback path. This is a judgment call,
not a given -- documented so a future maintainer can revisit it if it turns
out to hide something worth catching.

## Replacement level is already a healthy player

A fallback player never gets an availability model applied on top of its
replacement distribution -- at *any* position, not just DST. This was not
the original design: an earlier version of this module applied
`availability_by_position` uniformly to found and fallback players alike,
which shifted `scripts/season_report.py`'s 2024 calibration numbers (see
that script's git history / the task report for the before/after). On
review, the earlier design was wrong, not the numbers.

`replacement.weekly_replacement_values` (see `models/replacement.py`)
defines replacement level as the mean *actual, realized* score of the top
few available players that week, ranked ex-ante by trailing form --
`current = available.filter(pl.col("week") == week)` reads directly from
`weekly_stats`, which only has a row for a week a player actually
appeared in. A player who was hurt or inactive that week simply is not a
candidate in that week's shortlist; the estimator is already, by
construction, "what a manager gets by streaming a player who is available
and playing." Layering `PlayerAvailability`'s Markov chain on top of that
would model that *same* replacement slot going unavailable a second time --
double-counting unavailability that has already been priced out of the
replacement number. A real manager whose streamed pickup got scratched
simply streams a different healthy player instead of leaving the slot
empty; the replacement-level number already reflects that behavior, so the
roster construction step must not re-apply availability to it.

## DST has no separate "replacement level"

Every DST in this league shares one distribution (`defense.
dst_distribution()`, per the Task 2 owner decision) -- there is no
meaningful concept of "replacement level defense" distinct from "any
defense." `replacement.replacement_by_position()` therefore has no `"DST"`
key at all (its `POSITIONS` tuple is QB/RB/WR/TE/K only). Callers of this
module must add one themselves, pointed at the shared distribution, e.g.:

    replacement = replacement_by_position()
    replacement = {**replacement, "DST": dst_distribution()}

This mirrors `sim.lineup.build_replacement_means`'s existing convention of
adding a `"DST"` entry to its own flat means dict for the identical reason.
A DST pick absent from the pool then falls back to that same shared
distribution -- functionally identical to being *found* in the pool, since
every pool DST entry already points at the one shared object.

## Performance

One call is a fixed number of dict lookups and dataclass constructions per
pick -- no parquet reads, no refitting, no network. See
`tests/test_roster.py::test_full_roster_build_is_fast` for a measured
per-call timing on a full 18-man roster.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

from ..league import FLEX_ELIGIBLE
from ..sim.season import SeasonRosterPlayer
from .availability import PlayerAvailability
from .base import WeeklyDistribution
from .distribution import PlayerDistribution

# One draft pick: the id to look up in the pool, plus the position the
# caller believes it occupies. See module docstring, "What a 'pick' is."
DraftPick = Tuple[str, str]

# This league's six roster positions -- FLEX is a slot, not a position, so
# it is deliberately excluded (see `league.STARTERS` vs `league.
# FLEX_ELIGIBLE`).
VALID_POSITIONS: frozenset[str] = frozenset(FLEX_ELIGIBLE) | {"QB", "K", "DST"}

_EMPTY_AVAILABILITY: Mapping[str, PlayerAvailability] = MappingProxyType({})


@dataclass(frozen=True)
class RosterBuildResult:
    """`players` is simulator-ready (`sim.season.SeasonRosterPlayer`, one per
    input pick, same order). `n_fallback` is how many of those picks were
    not found in `pool` and were built from that position's replacement
    distribution instead -- reported so a caller can see it (e.g. "13 of
    180 picks fell back to replacement"), never just printed or dropped.
    `fallback_player_ids` names exactly which picks fell back, for callers
    that want to inspect rather than just count.
    """

    players: list[SeasonRosterPlayer]
    n_fallback: int
    fallback_player_ids: tuple[str, ...] = ()


def build_roster(
    picks: Sequence[DraftPick],
    pool: Mapping[str, PlayerDistribution],
    replacement_by_position: Mapping[str, WeeklyDistribution],
    availability_by_position: Mapping[str, PlayerAvailability] = _EMPTY_AVAILABILITY,
) -> RosterBuildResult:
    """Build simulator-ready roster players from a list of draft picks.

    `picks` may be empty, partial (a handful of picks mid-draft -- the
    normal case for Stage 3's rollout), or a full 18-man roster; every
    length is handled identically, one pick at a time, with no assumption
    that every slot is filled (`sim.lineup.solve_lineup` already tolerates
    an incomplete roster by falling back to replacement level itself).

    `pool` is typically `distribution.build_player_pool()`'s return value.
    `replacement_by_position` is typically `replacement.
    replacement_by_position()` plus a `"DST"` entry the caller adds (see
    module docstring). `availability_by_position` is typically
    `availability.availability_by_position()`; it is only ever consulted for
    a player *found* in the pool, and even then not for DST (every team is
    assumed to always have a defense to start, matching `SeasonRosterPlayer`
    's own convention). A fallback (replacement-level) player never gets an
    availability model, at any position -- see module docstring,
    "Replacement level is already a healthy player."

    Raises `ValueError` for a malformed pick (bad `player_id`, unknown
    `position`, or a valid `position` missing from `replacement_by_position`
    entirely) -- see module docstring, "The raise-vs-fallback line", for
    why these are programming errors rather than fallback candidates.
    """
    players: list[SeasonRosterPlayer] = []
    fallback_ids: list[str] = []

    for pick in picks:
        player_id, position = pick

        if not isinstance(player_id, str) or not player_id.strip():
            raise ValueError(f"malformed pick: player_id must be a non-empty string, got {player_id!r}")
        if not isinstance(position, str) or position not in VALID_POSITIONS:
            raise ValueError(
                f"malformed pick for player_id={player_id!r}: position {position!r} is not one "
                f"of this league's roster positions ({sorted(VALID_POSITIONS)})"
            )

        entry = pool.get(player_id)
        if entry is not None:
            resolved_position = entry.position
            distribution: WeeklyDistribution = entry.distribution
            availability = None if resolved_position == "DST" else availability_by_position.get(resolved_position)
        else:
            if position not in replacement_by_position:
                raise ValueError(
                    f"no replacement-level distribution for position {position!r} -- "
                    "replacement_by_position is missing this position entirely, which is a "
                    "configuration bug (every valid position needs one; see module docstring "
                    "for DST's convention), not an expected fallback."
                )
            resolved_position = position
            distribution = replacement_by_position[position]
            # No availability model on the fallback path, for any position
            # -- see module docstring, "Replacement level is already a
            # healthy player." Applying one here would double-count
            # unavailability that `replacement.py`'s estimator has already
            # conditioned away.
            availability = None
            fallback_ids.append(player_id)

        players.append(SeasonRosterPlayer(player_id, resolved_position, distribution, availability))

    return RosterBuildResult(
        players=players,
        n_fallback=len(fallback_ids),
        fallback_player_ids=tuple(fallback_ids),
    )
