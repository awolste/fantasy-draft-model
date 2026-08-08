"""The draft board as entered by hand on draft day.

Deliberately a thin wrapper over `draft.rollout.DraftState` rather than a
parallel model of the draft. `to_draft_state` is the only bridge, so the
engine can never be handed a board shape this module invented -- and
`tests/test_live_state.py` pins that the two agree about whose turn it is
at every pick, because a drift there would quietly ask the recommender for
the wrong team's pick.

Pick entry is manual (owner decision; see the Stage 4 spec). No ESPN
polling means this cannot silently desync from the real draft. The cost is
that a mis-entry is possible, which is why `undo` exists and why
double-drafting and over-drafting both raise rather than being absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..draft.rollout import DraftState, Pick, team_for_pick
from ..league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS


@dataclass
class DraftBoard:
    rounds: int = DRAFT_ROUNDS
    n_teams: int = N_TEAMS
    our_team: int = DRAFT_SLOT
    picks: list[Pick] = field(default_factory=list)

    @property
    def next_overall_pick(self) -> int:
        return len(self.picks) + 1

    @property
    def total_picks(self) -> int:
        return self.rounds * self.n_teams

    @property
    def is_complete(self) -> bool:
        return len(self.picks) >= self.total_picks

    @property
    def team_on_clock(self) -> int:
        return team_for_pick(self.next_overall_pick, self.n_teams)

    @property
    def is_our_turn(self) -> bool:
        return (not self.is_complete) and self.team_on_clock == self.our_team

    @property
    def current_round(self) -> int:
        return (self.next_overall_pick - 1) // self.n_teams + 1

    @property
    def drafted_ids(self) -> set[str]:
        return {p.player_id for p in self.picks}

    @property
    def our_pick_numbers(self) -> list[int]:
        return [
            n
            for n in range(1, self.total_picks + 1)
            if team_for_pick(n, self.n_teams) == self.our_team
        ]

    @property
    def our_roster_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.picks:
            if p.team == self.our_team:
                counts[p.position] = counts.get(p.position, 0) + 1
        return counts

    def record(self, player_id: str, position: str) -> None:
        if self.is_complete:
            raise ValueError(
                f"draft is complete ({self.total_picks} picks); cannot record another"
            )
        if player_id in self.drafted_ids:
            raise ValueError(f"{player_id} was already drafted")
        self.picks.append(
            Pick(
                overall_pick=self.next_overall_pick,
                team=self.team_on_clock,
                player_id=player_id,
                position=position,
            )
        )

    def undo(self) -> Pick:
        """Remove and return the last pick.

        Raises `IndexError` on an empty board rather than no-opping: a
        silent no-op during a live draft leaves this board out of step with
        the real room and says nothing about it.
        """
        return self.picks.pop()

    def to_draft_state(self) -> DraftState:
        return DraftState.from_picks(
            list(self.picks), n_teams=self.n_teams, rounds=self.rounds
        )
