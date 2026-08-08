"""Tests for `ffdraft.live.state`."""

from __future__ import annotations

import pytest

from ffdraft.league import DRAFT_ROUNDS, DRAFT_SLOT, N_TEAMS
from ffdraft.live.state import DraftBoard


def test_board_starts_empty_and_knows_who_is_on_the_clock():
    b = DraftBoard()
    assert b.next_overall_pick == 1
    assert b.team_on_clock == 1
    assert b.is_our_turn is False


def test_our_picks_are_the_slot_8_snake_sequence():
    """Slot 8 in a 10-team snake picks at 8 and 13 -- back to back at the
    turn -- then 28 and 33, and so on."""
    assert DraftBoard().our_pick_numbers[:6] == [8, 13, 28, 33, 48, 53]
    assert len(DraftBoard().our_pick_numbers) == DRAFT_ROUNDS


def test_recording_picks_advances_the_snake_and_flags_our_turn():
    b = DraftBoard()
    for i in range(7):
        b.record(f"p{i}", "RB")
    assert b.next_overall_pick == 8
    assert b.team_on_clock == DRAFT_SLOT
    assert b.is_our_turn is True


def test_undo_reverts_exactly_one_pick():
    b = DraftBoard()
    b.record("a", "RB")
    b.record("b", "WR")
    undone = b.undo()
    assert undone.player_id == "b"
    assert b.next_overall_pick == 2
    assert b.drafted_ids == {"a"}


def test_undo_on_empty_board_raises_rather_than_silently_doing_nothing():
    """A silent no-op mid-draft would leave the board out of step with the
    real room, and nothing would say so."""
    with pytest.raises(IndexError):
        DraftBoard().undo()


def test_drafting_the_same_player_twice_raises():
    b = DraftBoard()
    b.record("a", "RB")
    with pytest.raises(ValueError, match="already drafted"):
        b.record("a", "RB")


def test_recording_past_the_end_of_the_draft_raises():
    b = DraftBoard(rounds=1)
    for i in range(N_TEAMS):
        b.record(f"p{i}", "RB")
    assert b.is_complete
    with pytest.raises(ValueError, match="complete"):
        b.record("extra", "RB")


def test_to_draft_state_round_trips_into_the_engine_type():
    b = DraftBoard()
    b.record("a", "RB")
    st = b.to_draft_state()
    assert st.next_overall_pick == 2
    assert st.drafted_ids == {"a"}


def test_board_and_engine_agree_on_who_is_on_the_clock_every_pick():
    """The board must never disagree with `DraftState` about turn order --
    a drift here would silently ask the recommender for the wrong team's
    pick, and `recommend_pick` would raise only when it happened to be
    someone else's turn."""
    b = DraftBoard(rounds=3)
    for i in range(N_TEAMS * 3):
        assert b.team_on_clock == b.to_draft_state().team_on_clock, f"pick {i + 1}"
        b.record(f"p{i}", "RB")
