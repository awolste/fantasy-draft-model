"""Our own roster, as a starting lineup rather than a list of names.

## Why a lineup and not a list

`DraftBoard.our_roster_counts` already answers "how many running backs do
I have". It does not answer the question actually being asked on the
clock, which is **"what does my starting ten look like, and what is still
empty"** -- and those differ, because two of the ten slots are FLEX and a
third running back is a starter while a third tight end is not.

So this fills the real lineup with `sim.lineup.solve_lineup`, the same
exactly-optimal solver the season simulator uses, scoring each player by
his projected weekly mean. Reusing it rather than sorting by position here
means the view cannot disagree with the engine about who starts.

## What the numbers mean, and what they do not

Every point figure is **projected points per week** (`PlayerDistribution.
distribution.mean`), not a season total and not a realized score.

`projected_points` sums the filled slots *and* the empty ones, because
that is what `solve_lineup` scores -- an empty slot is worth replacement
level, never zero. Early in a draft most of the total is therefore
replacement level and says almost nothing about the team. `n_replacement`
is returned beside it so a caller can say so out loud instead of
presenting a confident-looking number built mostly from filler; the app
does exactly that.

There is no uncertainty attached to any of this. It is a projection
ordering, useful for seeing shape and gaps, and it is not the
championship-probability estimate the recommendation itself is built on.

**A team's projection can go *down* when it drafts.** `solve_lineup` uses
replacement level as a quantity trigger, not a quality one (see its module
docstring): once a real body occupies a slot it is started, even if he
projects below the replacement level that empty slot was scored at. So
taking a deep bench receiver before you have two starters lowers this
number. That is the engine's own convention, applied identically to all
ten teams, and it is inherited here rather than second-guessed -- but it
does mean an early pick that drops your projection is a display of the
convention, not a bug.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..league import FLEX_ELIGIBLE, STARTERS
from ..sim.lineup import RosterPlayer, solve_lineup

# The order a fantasy roster is conventionally read in. `solve_lineup`
# returns dedicated slots in sorted-position order (DST, K, QB, RB, ...),
# which is correct but not how anyone reads a lineup card.
DISPLAY_ORDER: tuple[str, ...] = ("QB", "RB", "WR", "TE", "FLEX", "K", "DST")


@dataclass(frozen=True)
class Slot:
    """One starting slot. `player_id is None` means nobody fills it yet."""

    label: str
    """Display label, numbered where a position has several slots: "RB1"."""
    position: str
    player_id: str | None
    name: str | None
    projected_points: float
    """The player's projected weekly mean, or replacement level if empty."""
    is_empty: bool


@dataclass(frozen=True)
class BenchPlayer:
    player_id: str
    name: str
    position: str
    projected_points: float


@dataclass(frozen=True)
class TeamView:
    starters: tuple[Slot, ...]
    bench: tuple[BenchPlayer, ...]
    projected_points: float
    """Sum over `starters`, empty slots included at replacement level."""
    n_replacement: int
    """How many of the ten slots are still filler. Report this whenever
    `projected_points` is shown."""

    @property
    def empty_slots(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.starters if s.is_empty)


def _label_slots(labels: Sequence[str]) -> list[str]:
    """Number only the positions that have more than one slot: one tight
    end is "TE", two flex slots are "FLEX1" and "FLEX2"."""
    total: dict[str, int] = {}
    for label in labels:
        total[label] = total.get(label, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        if total[label] == 1:
            out.append(label)
        else:
            seen[label] = seen.get(label, 0) + 1
            out.append(f"{label}{seen[label]}")
    return out


def team_view(
    roster_pairs: Sequence[tuple[str, str]],
    pool: Mapping[str, Any],
    replacement_means: Mapping[str, float],
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
) -> TeamView:
    """Our drafted `(player_id, position)` pairs as a starting lineup plus
    a bench.

    `position` comes from the pick, not from `pool`, matching how the rest
    of the live path records a pick -- but the *name* comes from `pool`, so
    a player_id absent from `pool` raises rather than being rendered as a
    blank row. On draft day a silently nameless starter is worse than a
    crash: it is the roster that looks fine and is not.
    """
    missing = [pid for pid, _ in roster_pairs if pid not in pool]
    if missing:
        raise KeyError(
            f"roster references {len(missing)} player(s) absent from the pool "
            f"(e.g. {missing[:3]}) -- the board and the pool have diverged."
        )

    players = [
        RosterPlayer(
            player_id=pid,
            position=position,
            score=float(pool[pid].distribution.mean),
        )
        for pid, position in roster_pairs
    ]

    result = solve_lineup(players, replacement_means, starters, flex_eligible)

    # `solve_lineup` returns dedicated slots in sorted-position order, then
    # FLEX. Re-order for reading, keeping each position's own slots in the
    # solver's order (best first) so "RB1" really is the better back.
    by_position: dict[str, list] = {}
    for slot in result.slots:
        by_position.setdefault(slot.slot, []).append(slot)
    ordered = [s for pos in DISPLAY_ORDER for s in by_position.get(pos, [])]
    # Anything the league adds that DISPLAY_ORDER has not heard of still
    # appears, at the end, rather than vanishing from the lineup.
    ordered += [s for s in result.slots if s.slot not in DISPLAY_ORDER]

    labels = _label_slots([s.slot for s in ordered])
    slots = tuple(
        Slot(
            label=label,
            position=s.slot,
            player_id=s.player_id,
            name=None if s.player_id is None else pool[s.player_id].name,
            projected_points=float(s.points),
            is_empty=s.is_replacement,
        )
        for label, s in zip(labels, ordered)
    )

    starting_ids = {s.player_id for s in slots if s.player_id is not None}
    bench = tuple(
        sorted(
            (
                BenchPlayer(
                    player_id=pid,
                    name=pool[pid].name,
                    position=position,
                    projected_points=float(pool[pid].distribution.mean),
                )
                for pid, position in roster_pairs
                if pid not in starting_ids
            ),
            key=lambda b: (-b.projected_points, b.player_id),
        )
    )

    return TeamView(
        starters=slots,
        bench=bench,
        projected_points=float(result.total_points),
        n_replacement=sum(1 for s in slots if s.is_empty),
    )


@dataclass(frozen=True)
class LeagueStanding:
    """Where our projected starting lineup sits against the other nine.

    Free to compute -- the board already records every team's picks, so
    this needs no simulation at all, only the same lineup solve run ten
    times.

    **It compresses early and that matters.** With most slots empty, every
    team is scored mostly on the same replacement levels, so small gaps in
    `points` are small differences in real roster quality; `rank` can swing
    on one pick. It is a sanity check against the room, not a standings
    table, and it says nothing about championship probability -- the two
    disagree routinely, because a lineup that scores well on average can
    still lose a 14-week season to variance.
    """

    points: float
    """Our own projected starting points per week."""
    rank: int
    """1 = highest projection in the league. Ties share the better rank."""
    n_teams: int
    points_by_team: tuple[float, ...]
    """Index `t - 1` holds team `t`'s projection, so a caller can show the
    spread rather than only our place in it."""


def league_standing(
    picks: Sequence[Any],
    pool: Mapping[str, Any],
    replacement_means: Mapping[str, float],
    n_teams: int,
    our_team: int,
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
) -> LeagueStanding:
    """Rank every team's projected starting lineup from the board's picks.

    `picks` is the whole board (anything with `.team`, `.player_id` and
    `.position`), not just ours.
    """
    if not 1 <= our_team <= n_teams:
        raise ValueError(f"our_team={our_team} is outside 1..{n_teams}")

    rosters: dict[int, list[tuple[str, str]]] = {t: [] for t in range(1, n_teams + 1)}
    for pick in picks:
        rosters.setdefault(pick.team, []).append((pick.player_id, pick.position))

    points = tuple(
        team_view(rosters[t], pool, replacement_means, starters, flex_eligible).projected_points
        for t in range(1, n_teams + 1)
    )
    ours = points[our_team - 1]
    return LeagueStanding(
        points=ours,
        rank=1 + sum(1 for p in points if p > ours),
        n_teams=n_teams,
        points_by_team=points,
    )
