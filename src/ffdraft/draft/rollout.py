"""Play the remaining draft forward from any state (Stage 3 Task 4).

Opponents are sampled from Task 2's league-wide opponent model
(`models/opponent.py`); our own future picks are chosen greedily by Task 3's
roster-aware value function (`draft/value.py`) -- a nested Monte Carlo at
each of our own future picks would be far too expensive for the recommender
(Task 5), which needs many rollouts per candidate.

## Draft state

`DraftState` is built from a (possibly empty, possibly partial) list of
`Pick`s, never grown incrementally from empty only -- this is the object
Task 5's recommender and, later, a live draft assistant both construct from
whatever has actually happened in the real draft so far. A `Pick` carries
its own `position` explicitly (matching `models.roster.build_roster`'s
`DraftPick` convention) rather than looking it up in a player pool, because
a historical replay's players may not exist in whatever pool the caller
supplies (see `models/roster.py`'s docstring on this exact point) and the
opponent model's roster-need term (`roster_decay`) needs a position to
count regardless.

## Snake order

Derived from `n_teams` alone (see `team_for_pick`), never hardcoded: round
`r` (1-indexed) goes 1->n_teams if `r` is odd, n_teams->1 if `r` is even.
`slot_pick_numbers` is the direct, tested answer to "which overall picks
belong to draft slot S" -- see `tests/test_rollout.py` for the slot-8
verification the task explicitly asks for (an off-by-one in snake direction
is easy to write and invisible in aggregate results, so it is checked
against the literal expected sequence 8, 13, 28, 33, ...).

## Historical replay / configurable round count

`DraftState.from_picks` and `run_rollout` both take `rounds` as an explicit
parameter rather than assuming `league.ROSTER_SIZE` -- this league's draft
was 17 rounds in 2018-2022 and is 18 rounds from 2023 on, and Task 6's
backtest replays real historical drafts of the size that was actually used
that season.

## Our own picks: greedy, not jointly optimized for back-to-back pairs

From slot 8, our own picks arrive in back-to-back pairs at the turn (8 and
13, 28 and 33, ...). A purely greedy pick-by-pick policy cannot see that
the second pick of a pair is coming and could, in principle, take two
players of the same position when one of each would serve the roster
better.

**Decision: accept the greedy approximation, not joint pair optimization.**
Reasons:

1. The second pick of a pair is evaluated *after* the first pick's result
   is already folded into the roster (`value.value_available` is
   roster-aware), so it already sees the updated positional need and will
   not blindly duplicate a now-well-filled position -- most of the benefit
   a joint search would capture is already recovered this way.
2. A true joint search would require evaluating pairs of candidates
   together (each pair requiring its own `solve_lineup` call), squaring the
   per-turn cost right at the point in the draft (rounds 1-2) where this
   function is called the most across a recommender's ~15 candidates x N
   rollouts.
3. Task 5's recommender is the caller that actually cares about pick 8 (the
   first of the pair); pick 13 inside a rollout is already an approximation
   of what *some* future version of ourselves will do once pick 13 is
   itself the decision being recommended, so over-investing in its
   precision here buys little.

This is a known limitation, not an oversight: a rollout can occasionally
produce a same-position pair at the turn where a human would visibly
diversify. See `tests/test_rollout.py` for the plausibility check on a
sampled rollout's first two rounds.

## ADP for the opponent model, in a season with no ADP yet

`models.opponent.pick_probabilities` needs each available player's `adp`.
For a live 2026 rollout there is no 2026 `adp_history` (`opponent.py`'s
`TRAINING_SEASONS` stops at 2024, and season-long ADP for a season that
hasn't happened yet does not exist by definition) -- `data/adp_2026.parquet`
exists but only covers the top ~246 players (the same deep-bench censoring
`opponent.py`'s `JoinReport` documents for historical seasons), not the
full ~500-player pool a rollout draws from.

**Decision: use each `PlayerDistribution.rank` as the `adp` proxy for every
player, uniformly, rather than blending real 2026 ADP for the top players
with a separately-scaled fallback for the rest.** `rank` is a full-coverage,
monotonic, overall (cross-position) consensus ordering already computed by
`distribution.build_player_pool` for every pool player -- exactly the
"expected order of selection" signal ADP is a proxy for. `pick_probabilities`
only ever consumes `adp` through a rank-space softmax (see its docstring:
"Rank space ... is used for the softmax"), so what matters is getting the
*relative order* of available players right, which `rank` already does by
construction; blending in real ADP for a censored top slice while falling
back to a different scale for the rest would risk exactly the kind of
one-sided discontinuity `JoinReport` warns is a real bias, not a knob to
casually reach for. This also avoids duplicating `opponent.py`'s
name/team-matching join logic a second time in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
from ..models.base import WeeklyDistribution
from ..models.distribution import PlayerDistribution
from ..models.opponent import (
    DEFAULT_ROSTER_DECAY,
    DEFAULT_TEMPERATURE,
    AvailablePlayer,
    OpponentModel,
    sample_pick,
)
from ..models.roster import build_roster
from ..sim.lineup import RosterPlayer
from .value import value_available

# ---------------------------------------------------------------------------
# Snake order


def team_for_pick(overall_pick: int, n_teams: int) -> int:
    """The 1-indexed draft slot on the clock at `overall_pick`, in a
    standard snake draft: round 1 goes 1->n_teams, round 2 goes n_teams->1,
    round 3 goes 1->n_teams again, and so on.

    `overall_pick` is 1-indexed. Raises `ValueError` for a non-positive
    pick number or a non-positive `n_teams` -- a loud failure rather than a
    silently wrong team.
    """
    if overall_pick < 1:
        raise ValueError(f"overall_pick must be >= 1, got {overall_pick}")
    if n_teams < 1:
        raise ValueError(f"n_teams must be >= 1, got {n_teams}")

    zero_based = overall_pick - 1
    round_index = zero_based // n_teams  # 0-indexed round
    position_in_round = zero_based % n_teams  # 0-indexed position within round

    if round_index % 2 == 0:
        return position_in_round + 1
    return n_teams - position_in_round


def slot_pick_numbers(draft_slot: int, n_teams: int, rounds: int) -> tuple[int, ...]:
    """Every overall pick number belonging to `draft_slot` across `rounds`
    rounds of an `n_teams`-team snake draft. Direct, testable answer to
    "which picks are ours" -- see the module docstring for why this is
    verified against the literal expected sequence rather than trusted by
    construction alone."""
    if not (1 <= draft_slot <= n_teams):
        raise ValueError(f"draft_slot must be in [1, {n_teams}], got {draft_slot}")
    return tuple(
        pick
        for pick in range(1, n_teams * rounds + 1)
        if team_for_pick(pick, n_teams) == draft_slot
    )


# ---------------------------------------------------------------------------
# Draft state


@dataclass(frozen=True)
class Pick:
    """One completed draft pick. `position` travels with the pick (rather
    than being looked up in a pool) because a historical replay's player may
    not exist in whatever pool a caller supplies -- see module docstring."""

    overall_pick: int
    team: int
    player_id: str
    position: str


@dataclass(frozen=True)
class DraftState:
    """The whole state of a draft in progress: every pick made so far, and
    each team's roster derived from it. Built from a (possibly empty,
    possibly partial) list of `Pick`s via `from_picks` -- never only grown
    from empty -- so this same type serves a fresh draft, a rollout
    mid-draft, and a historical replay alike.
    """

    n_teams: int
    rounds: int
    picks: tuple[Pick, ...]
    rosters: Mapping[int, tuple[tuple[str, str], ...]]
    drafted_ids: frozenset[str]

    @property
    def total_picks(self) -> int:
        return self.n_teams * self.rounds

    @property
    def next_overall_pick(self) -> int:
        return len(self.picks) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.picks) >= self.total_picks

    @property
    def team_on_clock(self) -> int:
        if self.is_complete:
            raise ValueError("draft is complete -- there is no team on the clock")
        return team_for_pick(self.next_overall_pick, self.n_teams)

    @classmethod
    def from_picks(
        cls, picks: Sequence[Pick], n_teams: int = N_TEAMS, rounds: int = DRAFT_ROUNDS
    ) -> "DraftState":
        """Build a `DraftState` from a list of picks, in overall-pick order.

        Validates, rather than silently tolerating:
        - `picks` covers overall picks 1..len(picks) with no gaps or
          repeats, in order.
        - each pick's `team` matches what `team_for_pick` says should be on
          the clock at that `overall_pick` (a wrong team recorded here would
          silently corrupt every downstream roster-need computation).
        - no `player_id` is drafted twice.
        - the draft is not longer than `n_teams * rounds`.
        """
        if n_teams < 1:
            raise ValueError(f"n_teams must be >= 1, got {n_teams}")
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")

        total_picks = n_teams * rounds
        if len(picks) > total_picks:
            raise ValueError(
                f"got {len(picks)} picks but this draft only has {total_picks} slots "
                f"({n_teams} teams x {rounds} rounds)"
            )

        rosters: dict[int, list[tuple[str, str]]] = {team: [] for team in range(1, n_teams + 1)}
        drafted_ids: set[str] = set()

        for expected_overall, pick in enumerate(picks, start=1):
            if pick.overall_pick != expected_overall:
                raise ValueError(
                    f"picks must cover overall picks 1..{len(picks)} in order with no gaps -- "
                    f"expected overall_pick {expected_overall}, got {pick.overall_pick}"
                )
            expected_team = team_for_pick(pick.overall_pick, n_teams)
            if pick.team != expected_team:
                raise ValueError(
                    f"pick {pick.overall_pick}: recorded team {pick.team} but the snake order "
                    f"for {n_teams} teams says team {expected_team} should be on the clock"
                )
            if pick.player_id in drafted_ids:
                raise ValueError(
                    f"player_id {pick.player_id!r} is drafted more than once -- pick "
                    f"{pick.overall_pick}"
                )
            drafted_ids.add(pick.player_id)
            rosters[pick.team].append((pick.player_id, pick.position))

        return cls(
            n_teams=n_teams,
            rounds=rounds,
            picks=tuple(picks),
            rosters={team: tuple(roster) for team, roster in rosters.items()},
            drafted_ids=frozenset(drafted_ids),
        )


# ---------------------------------------------------------------------------
# Our own picks: greedy by value.py


def _our_roster_players(
    roster_pairs: Sequence[tuple[str, str]],
    pool: Mapping[str, PlayerDistribution],
    replacement_by_position: Mapping[str, WeeklyDistribution],
) -> list[RosterPlayer]:
    """Turn our own drafted-so-far (player_id, position) pairs into
    `sim.lineup.RosterPlayer`s for `value.value_available`, via
    `models.roster.build_roster` so a historical pick absent from `pool`
    still falls back to that position's replacement level rather than
    crashing (see `models/roster.py`'s docstring)."""
    result = build_roster(list(roster_pairs), pool, replacement_by_position)
    return [
        RosterPlayer(player_id=p.player_id, position=p.position, score=float(p.distribution.mean))
        for p in result.players
    ]


def _choose_our_pick(
    available_ids: Sequence[str],
    pool: Mapping[str, PlayerDistribution],
    roster_pairs: Sequence[tuple[str, str]],
    replacement_means: Mapping[str, float],
    replacement_by_position: Mapping[str, WeeklyDistribution],
) -> tuple[str, str]:
    """Greedily pick the highest-`value.py`-value available player for our
    own roster so far. See module docstring for why pairs at the turn are
    not jointly optimized."""
    roster = _our_roster_players(roster_pairs, pool, replacement_by_position)
    values = value_available(list(available_ids), pool, roster, replacement_means)
    best = max(values, key=lambda v: v.value)
    return best.player_id, best.position


# ---------------------------------------------------------------------------
# The rollout itself


def run_rollout(
    state: DraftState,
    pool: Mapping[str, PlayerDistribution],
    model: OpponentModel,
    replacement_by_position: Mapping[str, WeeklyDistribution],
    rng: np.random.Generator,
    our_team: int = DRAFT_SLOT,
    temperature: float = DEFAULT_TEMPERATURE,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
) -> DraftState:
    """Play `state` forward, pick by pick, until every roster is full.

    Every pick already in `state` is left untouched. For each remaining
    pick: if it is `our_team`'s turn, the next player is chosen greedily by
    `draft.value` against our own roster so far; otherwise a player is
    sampled from `models.opponent.sample_pick` using that team's roster
    counts. `rng` is the only source of randomness (an explicit
    `np.random.Generator`, never global numpy state -- see
    `tests/test_rollout.py::test_rollout_does_not_touch_global_numpy_state`),
    so identical seeds reproduce identical drafts.

    `pool` must contain every player that could possibly be drafted --
    `available` at every step is exactly `pool.keys() - already_drafted`.
    See the module docstring, "ADP for the opponent model", for how each
    player's `adp` (needed by `sample_pick`) is derived from `pool` alone
    (via `PlayerDistribution.rank`) rather than a separate ADP source.

    Raises `ValueError` (via `sample_pick`/`value_available`) if the pool
    runs out of players before every roster is full -- a real configuration
    error (pool too small for `n_teams * rounds`), not a case to paper over.
    """
    replacement_means = {pos: float(dist.mean) for pos, dist in replacement_by_position.items()}

    picks: list[Pick] = list(state.picks)
    rosters: dict[int, list[tuple[str, str]]] = {
        team: list(roster) for team, roster in state.rosters.items()
    }
    roster_counts: dict[int, dict[str, int]] = {team: {} for team in range(1, state.n_teams + 1)}
    for team, roster in rosters.items():
        for _, position in roster:
            roster_counts[team][position] = roster_counts[team].get(position, 0) + 1

    drafted_ids: set[str] = set(state.drafted_ids)
    available_ids: list[str] = [pid for pid in pool if pid not in drafted_ids]

    total_picks = state.total_picks
    next_overall = state.next_overall_pick

    while len(picks) < total_picks:
        team = team_for_pick(next_overall, state.n_teams)

        if team == our_team:
            player_id, position = _choose_our_pick(
                available_ids, pool, rosters[team], replacement_means, replacement_by_position
            )
        else:
            candidates = [
                AvailablePlayer(player_id=pid, position=pool[pid].position, adp=float(pool[pid].rank))
                for pid in available_ids
            ]
            player_id = sample_pick(
                model,
                manager_id=f"slot_{team}",
                available=candidates,
                roster_counts=roster_counts[team],
                rng=rng,
                temperature=temperature,
                roster_decay=roster_decay,
            )
            position = pool[player_id].position

        picks.append(Pick(overall_pick=next_overall, team=team, player_id=player_id, position=position))
        rosters[team].append((player_id, position))
        roster_counts[team][position] = roster_counts[team].get(position, 0) + 1
        drafted_ids.add(player_id)
        available_ids = [pid for pid in available_ids if pid != player_id]
        next_overall += 1

    return DraftState.from_picks(picks, n_teams=state.n_teams, rounds=state.rounds)
