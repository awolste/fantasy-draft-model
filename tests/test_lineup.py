"""Tests for `sim.lineup`.

Two independent oracles back the correctness claims here, on top of the
hand-derived exchange-argument proof in the module docstring:

- `_hungarian_oracle_total`: an independent generic exact solver
  (`scipy.optimize.linear_sum_assignment`, a different algorithm/library
  from anything in `sim/lineup.py`) over the *same* problem (players x
  slots, with one very-low-value placeholder row per slot standing in for
  "leave this slot empty and use replacement level" -- see its docstring).
  Used for randomized cross-checks at realistic roster sizes (up to 18
  players).
- `_brute_force_total`: literal exhaustive enumeration of every valid
  slot-by-slot assignment (including "leave empty"), for tiny cases where
  that is actually tractable. Slower, but makes zero assumptions shared
  with either `solve_lineup` or the Hungarian oracle.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from ffdraft.league import FLEX_ELIGIBLE, STARTERS
from ffdraft.sim.lineup import (
    LineupResult,
    RosterPlayer,
    SlotResult,
    build_replacement_means,
    solve_lineup,
)

REPLACEMENT = {"QB": 10.0, "RB": 4.0, "WR": 4.5, "TE": 3.0, "K": 5.0, "DST": 7.5}


def rp(player_id, position, score, available=True):
    return RosterPlayer(player_id=player_id, position=position, score=score, available=available)


# ---------------------------------------------------------------------------
# Independent oracles


def _hungarian_oracle_total(roster, replacement_means, starters=STARTERS, flex_eligible=FLEX_ELIGIBLE):
    """Ground truth via a generic exact assignment solver, independent of
    `solve_lineup`'s own algorithm.

    One row per real available player, plus one placeholder row per slot
    representing "leave this slot empty, score it at replacement level"
    (cost set far worse than any real assignment, so a placeholder is only
    ever chosen when no real eligible player remains for that slot --
    matching `solve_lineup`'s quantity-only replacement trigger). Ineligible
    real-player/slot pairs get an even larger cost so they are never chosen
    while any placeholder remains available (and placeholders make every
    slot always feasible, so the matrix is never infeasible).

    A placeholder's cost is `OFFSET - replacement_for(slot)`, not a flat
    constant: when multiple slots could be the one left empty (a genuine
    tie on how many bodies are available), a flat placeholder cost would
    make the solver indifferent between leaving a low-replacement slot
    empty and a high-replacement one, even though the optimal choice
    clearly maximizes total points by leaving the highest-replacement slot
    empty. `OFFSET` is large enough that a placeholder is still always
    worse than any real, eligible assignment regardless of that player's
    score.
    """
    slots = [pos for pos, count in starters.items() for _ in range(count)]
    available = [p for p in roster if p.available]
    n_slots = len(slots)

    OFFSET = 1e6  # dominates any real cost, however negative a real score is
    INELIGIBLE_COST = 1e12  # cost of a disallowed pairing (never chosen)

    def replacement_for(slot_pos):
        if slot_pos == "FLEX":
            return max(replacement_means[p] for p in flex_eligible)
        return replacement_means[slot_pos]

    n_rows = len(available) + n_slots
    cost = np.full((n_rows, n_slots), INELIGIBLE_COST)

    for ri, p in enumerate(available):
        for si, slot_pos in enumerate(slots):
            eligible = p.position == slot_pos or (slot_pos == "FLEX" and p.position in flex_eligible)
            if eligible:
                cost[ri, si] = -p.score

    for si, slot_pos in enumerate(slots):
        cost[len(available) + si, si] = OFFSET - replacement_for(slot_pos)

    row_ind, col_ind = linear_sum_assignment(cost)

    total = 0.0
    for ri, si in zip(row_ind, col_ind):
        if ri < len(available):
            total += available[ri].score
        else:
            total += replacement_for(slots[si])
    return total


def _brute_force_total(roster, replacement_means, starters, flex_eligible=FLEX_ELIGIBLE):
    """Literal exhaustive search: try every valid way to fill each slot in
    turn (a specific available, not-yet-used, eligible player, or leave it
    empty for replacement), and return the best total. Only tractable for a
    handful of players/slots -- used for small randomized cases.

    Ranks candidate assignments lexicographically by `(-empty_slot_count,
    real_point_total)`: fewest empty slots first (an available, eligible
    body is always used ahead of replacement, however low its score --
    `solve_lineup`'s quantity-only trigger), and only among assignments
    tied on that does it maximize actual points. A plain score-sum
    comparison would let replacement values (which are typically positive)
    outbid a real player having a bad week, which is not the policy being
    tested.
    """
    slots = [pos for pos, count in starters.items() for _ in range(count)]
    available = [p for p in roster if p.available]
    n = len(available)

    def replacement_for(slot_pos):
        if slot_pos == "FLEX":
            return max(replacement_means[p] for p in flex_eligible)
        return replacement_means[slot_pos]

    best_key = None

    def rec(i, used, empty_count, real_total):
        nonlocal best_key
        if i == len(slots):
            key = (-empty_count, real_total)
            if best_key is None or key > best_key:
                best_key = key
            return
        slot_pos = slots[i]
        rec(i + 1, used, empty_count + 1, real_total + replacement_for(slot_pos))
        for j in range(n):
            if not (used >> j) & 1:
                p = available[j]
                eligible = p.position == slot_pos or (slot_pos == "FLEX" and p.position in flex_eligible)
                if eligible:
                    rec(i + 1, used | (1 << j), empty_count, real_total + p.score)

    rec(0, 0, 0, 0.0)
    return best_key[1]


# ---------------------------------------------------------------------------
# The naive, broken greedy the task warns about: fill each position's
# dedicated slots, then IMMEDIATELY spend any open FLEX slots on that same
# position's own leftovers, before ever looking at later positions. This is
# used only to demonstrate the failure mode `solve_lineup` avoids.


def _naive_sequential_greedy_total(roster, replacement_means, starters=STARTERS, flex_eligible=FLEX_ELIGIBLE):
    available = [p for p in roster if p.available]
    dedicated_positions = [pos for pos in starters if pos != "FLEX"]
    flex_remaining = starters.get("FLEX", 0)
    total = 0.0
    for pos in dedicated_positions:
        count = starters[pos]
        pool = sorted((p for p in available if p.position == pos), key=lambda p: p.score, reverse=True)
        chosen, rest = pool[:count], pool[count:]
        for i in range(count):
            total += chosen[i].score if i < len(chosen) else replacement_means[pos]
        if pos in flex_eligible and flex_remaining > 0:
            take = rest[:flex_remaining]
            total += sum(p.score for p in take)
            flex_remaining -= len(take)
    if flex_remaining > 0:
        total += flex_remaining * max(replacement_means[p] for p in flex_eligible)
    return total


# ---------------------------------------------------------------------------
# Standard case: every slot filled correctly


def test_standard_case_fills_every_slot():
    roster = [
        rp("qb1", "QB", 20.0),
        rp("rb1", "RB", 18.0),
        rp("rb2", "RB", 15.0),
        rp("rb3", "RB", 9.0),
        rp("wr1", "WR", 17.0),
        rp("wr2", "WR", 12.0),
        rp("wr3", "WR", 8.0),
        rp("te1", "TE", 10.0),
        rp("te2", "TE", 4.0),
        rp("k1", "K", 7.0),
        rp("dst1", "DST", 6.0),
    ]
    result = solve_lineup(roster, REPLACEMENT)
    assert len(result.slots) == sum(STARTERS.values()) == 10
    assert all(not s.is_replacement for s in result.slots)
    started_ids = {s.player_id for s in result.slots}
    # Best total: QB 20 + RB 18+15 + WR 17+12 + TE 10 + FLEX(best 2 leftovers:
    # rb3=9, wr3=8) + K 7 + DST 6
    assert result.total_points == pytest.approx(20 + 18 + 15 + 17 + 12 + 10 + 9 + 8 + 7 + 6)
    assert "rb3" in started_ids and "wr3" in started_ids
    assert "te2" not in started_ids  # benched: worse than rb3/wr3 for FLEX


# ---------------------------------------------------------------------------
# The adversarial case: greedy-by-position (even "then flex with whatever's
# left over from that position") loses when it spends FLEX before it has
# seen every position's leftovers.


def _adversarial_roster():
    return [
        rp("qb1", "QB", 15.0),
        rp("rb1", "RB", 10.0),
        rp("rb2", "RB", 9.0),
        rp("rb3", "RB", 8.0),
        rp("rb4", "RB", 7.0),
        rp("wr1", "WR", 100.0),
        rp("wr2", "WR", 90.0),
        rp("wr3", "WR", 30.0),
        rp("te1", "TE", 5.0),
        rp("k1", "K", 6.0),
        rp("dst1", "DST", 8.0),
    ]


def test_adversarial_case_greedy_by_position_loses():
    roster = _adversarial_roster()

    naive_total = _naive_sequential_greedy_total(roster, REPLACEMENT)
    optimal_total = solve_lineup(roster, REPLACEMENT).total_points

    # Hand-computed: naive spends both FLEX slots on RB leftovers (8, 7)
    # right after processing RB, before it ever sees WR's leftover (30).
    # QB 15 + RB 10+9 + WR 100+90 + TE 5 + FLEX(8+7) + K 6 + DST 8 = 258
    assert naive_total == pytest.approx(15 + 10 + 9 + 100 + 90 + 5 + 8 + 7 + 6 + 8)
    # Optimal pools ALL leftovers (RB 8,7 and WR 30) before choosing FLEX,
    # and picks the true top 2: 30 and 8.
    # QB 15 + RB 10+9 + WR 100+90 + TE 5 + FLEX(30+8) + K 6 + DST 8 = 281
    assert optimal_total == pytest.approx(15 + 10 + 9 + 100 + 90 + 5 + 30 + 8 + 6 + 8)

    assert optimal_total > naive_total
    assert optimal_total - naive_total == pytest.approx(23.0)

    # And confirm the optimal total really is optimal, independently.
    assert optimal_total == pytest.approx(_hungarian_oracle_total(roster, REPLACEMENT))
    assert optimal_total == pytest.approx(
        _brute_force_total(roster, REPLACEMENT, dict(STARTERS))
    )


# ---------------------------------------------------------------------------
# Unavailable players excluded even if the highest scorer


def test_unavailable_highest_scorer_excluded():
    roster = [
        rp("qb1", "QB", 20.0),
        rp("rb1", "RB", 99.0, available=False),  # would dominate if allowed
        rp("rb2", "RB", 12.0),
        rp("rb3", "RB", 10.0),
        rp("wr1", "WR", 14.0),
        rp("wr2", "WR", 11.0),
        rp("te1", "TE", 9.0),
        rp("k1", "K", 7.0),
        rp("dst1", "DST", 6.0),
    ]
    result = solve_lineup(roster, REPLACEMENT)
    started_ids = {s.player_id for s in result.slots}
    assert "rb1" not in started_ids
    # rb2, rb3 both used (dedicated RB + FLEX, only 2 RBs available)
    assert "rb2" in started_ids and "rb3" in started_ids


# ---------------------------------------------------------------------------
# Thin roster: not enough bodies to fill every slot -> replacement, no crash


def test_thin_roster_uses_replacement_for_unfilled_slots():
    roster = [
        rp("qb1", "QB", 20.0),
        rp("rb1", "RB", 12.0),  # only 1 RB: dedicated RB slot #2 empty
        rp("wr1", "WR", 14.0),
        rp("wr2", "WR", 11.0),
        # no TE at all
        rp("k1", "K", 7.0),
        rp("dst1", "DST", 6.0),
    ]
    result = solve_lineup(roster, REPLACEMENT)
    assert len(result.slots) == 10

    rb_slots = [s for s in result.slots if s.slot == "RB"]
    te_slots = [s for s in result.slots if s.slot == "TE"]
    flex_slots = [s for s in result.slots if s.slot == "FLEX"]

    assert sum(1 for s in rb_slots if s.is_replacement) == 1
    assert rb_slots[0].points == REPLACEMENT["RB"] or rb_slots[1].points == REPLACEMENT["RB"]
    assert all(s.is_replacement for s in te_slots)
    assert te_slots[0].points == REPLACEMENT["TE"]
    # No RB/WR/TE leftover exists, so both FLEX slots fall back to
    # replacement too, at the best of RB/WR/TE replacement means.
    assert all(s.is_replacement for s in flex_slots)
    assert all(s.points == max(REPLACEMENT["RB"], REPLACEMENT["WR"], REPLACEMENT["TE"]) for s in flex_slots)

    expected = (
        20.0  # QB
        + 12.0
        + REPLACEMENT["RB"]  # RB
        + 14.0
        + 11.0  # WR
        + REPLACEMENT["TE"]  # TE
        + 2 * max(REPLACEMENT["RB"], REPLACEMENT["WR"], REPLACEMENT["TE"])  # FLEX
        + 7.0  # K
        + 6.0  # DST
    )
    assert result.total_points == pytest.approx(expected)


def test_completely_empty_roster_uses_replacement_everywhere_no_crash():
    result = solve_lineup([], REPLACEMENT)
    assert len(result.slots) == 10
    assert all(s.is_replacement for s in result.slots)
    assert all(s.player_id is None for s in result.slots)


# ---------------------------------------------------------------------------
# FLEX never takes QB, K, or DST


def test_flex_never_takes_qb_k_or_dst():
    roster = [
        rp("qb1", "QB", 50.0),
        rp("qb2", "QB", 45.0),  # would love to be FLEX if it were legal
        rp("k1", "K", 40.0),
        rp("dst1", "DST", 35.0),
        rp("rb1", "RB", 5.0),
        rp("rb2", "RB", 4.0),
        rp("wr1", "WR", 3.0),
        rp("wr2", "WR", 2.0),
        rp("te1", "TE", 1.0),
    ]
    result = solve_lineup(roster, REPLACEMENT)
    flex_slots = [s for s in result.slots if s.slot == "FLEX"]
    for s in flex_slots:
        assert s.player_id != "qb2"
        if s.player_id is not None:
            assert s.player_id not in {"qb1", "qb2", "k1", "dst1"}
    # qb2 sits on the bench even though it outscores every FLEX-eligible
    # player -- it is simply not an eligible body for FLEX.
    started_ids = {s.player_id for s in result.slots}
    assert "qb2" not in started_ids


# ---------------------------------------------------------------------------
# Lineup shape read from league.STARTERS, not hardcoded


def test_lineup_shape_follows_starters_config_not_hardcoded():
    custom_starters = {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 1, "K": 1, "DST": 1}
    custom_flex_eligible = frozenset({"RB", "WR"})
    replacement = {"QB": 10.0, "RB": 4.0, "WR": 4.5, "K": 5.0, "DST": 7.5}

    roster = [
        rp("qb1", "QB", 20.0),
        rp("rb1", "RB", 12.0),
        rp("rb2", "RB", 11.0),
        rp("wr1", "WR", 9.0),
        rp("k1", "K", 7.0),
        rp("dst1", "DST", 6.0),
    ]
    result = solve_lineup(roster, replacement, starters=custom_starters, flex_eligible=custom_flex_eligible)
    assert len(result.slots) == sum(custom_starters.values()) == 6
    assert not any(s.slot == "TE" for s in result.slots)
    # rb1(12) dedicated RB, rb2(11) and wr1(9) compete for FLEX -> rb2 wins
    flex = [s for s in result.slots if s.slot == "FLEX"][0]
    assert flex.player_id == "rb2"

    # And the real default config really does have TE + a different FLEX
    # pool -- this test isn't accidentally passing because TE is absent
    # from `league.STARTERS` too.
    assert STARTERS.get("TE") == 1
    assert "TE" in FLEX_ELIGIBLE


def test_default_starters_argument_is_leagues_starters():
    # Guards against the default silently drifting from league.py.
    import inspect

    sig = inspect.signature(solve_lineup)
    assert sig.parameters["starters"].default is STARTERS
    assert sig.parameters["flex_eligible"].default is FLEX_ELIGIBLE


# ---------------------------------------------------------------------------
# Validation: loud failures over silent defaults


def test_missing_replacement_mean_raises():
    incomplete = {k: v for k, v in REPLACEMENT.items() if k != "TE"}
    roster = [rp("qb1", "QB", 10.0)]
    with pytest.raises(ValueError, match="TE"):
        solve_lineup(roster, incomplete)


def test_unknown_position_raises():
    roster = [rp("x1", "PUNTER", 10.0)]
    with pytest.raises(ValueError, match="PUNTER"):
        solve_lineup(roster, REPLACEMENT)


# ---------------------------------------------------------------------------
# Exhaustive agreement with brute force / Hungarian oracle on randomized
# small rosters


def _random_roster(rng: random.Random, positions=("QB", "RB", "WR", "TE", "K", "DST"), max_per_pos=4):
    roster = []
    for pos in positions:
        n = rng.randint(0, max_per_pos)
        for i in range(n):
            score = round(rng.uniform(-5, 40), 2)
            available = rng.random() > 0.15
            roster.append(rp(f"{pos}{i}", pos, score, available=available))
    return roster


def test_exhaustive_agreement_with_hungarian_oracle_on_randomized_rosters():
    rng = random.Random(12345)
    for _ in range(200):
        roster = _random_roster(rng)
        got = solve_lineup(roster, REPLACEMENT).total_points
        want = _hungarian_oracle_total(roster, REPLACEMENT)
        assert got == pytest.approx(want), roster


def test_exhaustive_agreement_with_brute_force_on_tiny_randomized_rosters():
    # Small custom shape (2 RB + 2 WR + 1 FLEX = 5 slots) so literal
    # exhaustive enumeration stays tractable.
    starters = {"RB": 2, "WR": 2, "FLEX": 1}
    replacement = {"RB": 4.0, "WR": 4.5}
    rng = random.Random(999)
    for _ in range(100):
        roster = []
        for pos in ("RB", "WR"):
            n = rng.randint(0, 3)
            for i in range(n):
                score = round(rng.uniform(-5, 20), 2)
                available = rng.random() > 0.2
                roster.append(rp(f"{pos}{i}", pos, score, available=available))
        flex_eligible = frozenset({"RB", "WR"})
        got = solve_lineup(roster, replacement, starters=starters, flex_eligible=flex_eligible).total_points
        want = _brute_force_total(roster, replacement, starters, flex_eligible=flex_eligible)
        assert got == pytest.approx(want), roster


# ---------------------------------------------------------------------------
# build_replacement_means helper


class _FakeDist:
    def __init__(self, mean):
        self.mean = mean


def test_build_replacement_means_flattens_and_adds_dst():
    by_pos = {"QB": _FakeDist(10.0), "RB": _FakeDist(4.0), "WR": _FakeDist(4.5), "TE": _FakeDist(3.0), "K": _FakeDist(5.0)}
    means = build_replacement_means(by_pos, dst_mean=7.5)
    assert means == {"QB": 10.0, "RB": 4.0, "WR": 4.5, "TE": 3.0, "K": 5.0, "DST": 7.5}
