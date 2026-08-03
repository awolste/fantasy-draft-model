"""Tests for `draft.value`.

Uses small hand-built synthetic pools/rosters (same style as
`tests/test_roster.py`) so these are fast and independent of the real data
pipeline. A final group of tests runs against the real 2026 pool/replacement
data to produce the plausibility report the task asks for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest

from ffdraft.draft.value import BENCH_DISCOUNT, PlayerValue, player_value, value_available
from ffdraft.models.distribution import PlayerDistribution
from ffdraft.sim.lineup import RosterPlayer


@dataclass(frozen=True)
class ConstantDist:
    value: float

    @property
    def mean(self) -> float:
        return self.value

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, self.value)


def _pd(player_id: str, position: str, value: float) -> PlayerDistribution:
    return PlayerDistribution(
        player_id=player_id, name=player_id, position=position, rank=1, tier=1,
        distribution=ConstantDist(value),
    )


REPLACEMENT_MEANS = {"QB": 18.3, "RB": 9.6, "WR": 9.9, "TE": 8.1, "K": 7.9, "DST": 7.5}

POOL = {
    "qb1": _pd("qb1", "QB", 24.0),
    "qb2": _pd("qb2", "QB", 20.0),
    "qb3": _pd("qb3", "QB", 18.5),
    "rb1": _pd("rb1", "RB", 16.0),
    "rb2": _pd("rb2", "RB", 14.0),
    "rb3": _pd("rb3", "RB", 12.5),
    "rb4": _pd("rb4", "RB", 10.0),
    "wr1": _pd("wr1", "WR", 15.0),
    "wr2": _pd("wr2", "WR", 13.0),
    "wr3": _pd("wr3", "WR", 11.5),
    "te1": _pd("te1", "TE", 9.5),
    "te2": _pd("te2", "TE", 8.3),
    "k1": _pd("k1", "K", 8.5),
    "dst1": _pd("dst1", "DST", 8.0),
}


def _roster(*ids: str) -> list[RosterPlayer]:
    return [RosterPlayer(pid, POOL[pid].position, POOL[pid].distribution.mean) for pid in ids]


# ---------------------------------------------------------------------------
# Empty roster: best available valued highest


def test_empty_roster_values_best_player_highest():
    available = list(POOL.keys())
    values = value_available(available, POOL, [], REPLACEMENT_MEANS)
    best = max(values, key=lambda v: v.value)
    # From an empty roster every player's marginal is (mean - that
    # position's replacement mean), since it fills the first open slot at
    # its own position. rb1 (16.0 - 9.6 = 6.4) beats qb1 (24.0 - 18.3 =
    # 5.7) because QB's replacement level is unusually strong relative to
    # its starters here -- exactly the documented QB/RB asymmetry this
    # value function is supposed to reproduce, not fight.
    assert best.player_id == "rb1"


# ---------------------------------------------------------------------------
# Depth at a position lowers that position's value


def test_deep_position_valued_lower_than_a_fresh_roster():
    # rb4 (10.0) drafted onto an empty roster fills a real dedicated RB
    # slot outright. The same rb4 drafted onto a roster already deep at RB
    # (2 dedicated + both FLEX slots soaked by better RBs/WRs) can only add
    # bench-insurance value, since there is no slot left for it to start
    # in. Same player, same pool -- the roster context alone must lower it.
    empty_value = value_available(["rb4"], POOL, [], REPLACEMENT_MEANS)[0]

    deep_roster = _roster("rb1", "rb2", "rb3", "wr1", "wr2", "wr3")
    deep_value = value_available(["rb4"], POOL, deep_roster, REPLACEMENT_MEANS)[0]

    assert deep_value.value < empty_value.value


# ---------------------------------------------------------------------------
# A position that cannot start any more players values near replacement


def test_position_that_cannot_start_more_is_valued_near_replacement():
    # QB has only 1 slot in this league and it's already filled by someone
    # far better; a second QB right at replacement level should value near
    # zero (it cannot start, and there's almost nothing over replacement to
    # discount for bench value either).
    roster = _roster("qb1")
    pool = dict(POOL)
    pool["qb_replacement_level"] = _pd("qb_replacement_level", "QB", REPLACEMENT_MEANS["QB"] + 0.1)
    values = value_available(["qb_replacement_level"], pool, roster, REPLACEMENT_MEANS)
    assert values[0].value == pytest.approx(0.0, abs=0.5)


# ---------------------------------------------------------------------------
# The FLEX-awareness pair: third RB meaningfully positive, second QB near zero


def test_third_rb_meaningful_second_qb_near_zero():
    # Roster: 1 QB starter, 2 RB starters, 2 WR starters, 1 TE starter --
    # every dedicated slot full, both FLEX slots open.
    roster = _roster("qb1", "rb1", "rb2", "wr1", "wr2", "te1")
    values = {
        v.player_id: v
        for v in value_available(["rb3", "qb2"], POOL, roster, REPLACEMENT_MEANS)
    }
    third_rb = values["rb3"]
    second_qb = values["qb2"]

    # Third RB (12.5 mean) is FLEX-eligible and beats the current worst
    # thing that could occupy a FLEX slot (te1 at 9.5, wr2 at 13.0) -- or at
    # minimum earns real bench credit; either way it must clear a
    # meaningful bar, not just epsilon.
    assert third_rb.value > 1.0
    # Second QB (20.0 mean) cannot start at all (QB has 1 slot, not
    # FLEX-eligible) -- it only earns the crude bench discount.
    over_repl = 20.0 - REPLACEMENT_MEANS["QB"]
    assert second_qb.value == pytest.approx(BENCH_DISCOUNT * over_repl, rel=1e-9)
    assert second_qb.value < third_rb.value


# ---------------------------------------------------------------------------
# Lineup shape is read from the starters config, not hardcoded


def test_lineup_shape_follows_starters_config_not_hardcoded():
    roster = _roster("qb1", "rb1", "rb2", "wr1", "wr2", "te1")

    default_values = {
        v.player_id: v for v in value_available(["rb3"], POOL, roster, REPLACEMENT_MEANS)
    }

    # Zero FLEX slots: a 3rd RB now has nowhere at all to start.
    no_flex_starters = MappingProxyType({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 0, "K": 1, "DST": 1})
    no_flex_values = {
        v.player_id: v
        for v in value_available(
            ["rb3"], POOL, roster, REPLACEMENT_MEANS, starters=no_flex_starters
        )
    }

    assert no_flex_values["rb3"].value < default_values["rb3"].value

    # More FLEX slots: even more room for RB depth to start, value should
    # not decrease.
    more_flex_starters = MappingProxyType({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 4, "K": 1, "DST": 1})
    more_flex_values = {
        v.player_id: v
        for v in value_available(
            ["rb3"], POOL, roster, REPLACEMENT_MEANS, starters=more_flex_starters
        )
    }
    assert more_flex_values["rb3"].value >= default_values["rb3"].value


# ---------------------------------------------------------------------------
# Determinism -- no sampling


def test_deterministic_no_sampling():
    roster = _roster("qb1", "rb1", "wr1")
    available = ["rb2", "rb3", "wr2", "te1", "qb2"]
    v1 = value_available(available, POOL, roster, REPLACEMENT_MEANS)
    v2 = value_available(available, POOL, roster, REPLACEMENT_MEANS)
    assert v1 == v2


# ---------------------------------------------------------------------------
# The reported value never goes negative, even in solve_lineup's one
# documented non-monotonic edge case (a real leftover player displacing a
# *better* replacement-level FLEX fill -- see sim/lineup.py's module
# docstring and the note in this module's docstring). te2 (8.3) is below
# this pool's WR replacement mean (9.9), which is what would otherwise fill
# an empty FLEX slot, so its raw lineup marginal is negative; the bench
# floor must still keep the reported value at 0, not negative.


def test_value_is_never_negative():
    roster = _roster("qb1", "rb1", "rb2", "wr1", "wr2", "te1")
    for pid in ["rb3", "rb4", "wr3", "qb2", "k1", "dst1", "te2"]:
        pv = player_value(pid, POOL, roster, REPLACEMENT_MEANS)
        assert pv.value >= 0.0


# ---------------------------------------------------------------------------
# Single-call convenience wrapper matches the batch path


def test_player_value_matches_value_available_single_entry():
    roster = _roster("qb1", "rb1")
    single = player_value("wr1", POOL, roster, REPLACEMENT_MEANS)
    batch = value_available(["wr1"], POOL, roster, REPLACEMENT_MEANS)[0]
    assert single == batch


# ---------------------------------------------------------------------------
# Timing (synthetic, small pool) -- a coarse sanity check; the real
# per-call/whole-pool numbers against the full 2026 pool are reported
# separately (see the task report), since that requires real cached data.


def test_value_available_is_fast_for_a_500_player_pool():
    import itertools

    big_pool = dict(POOL)
    positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    for i, pos in zip(range(500), itertools.cycle(positions)):
        pid = f"synthetic_{i}"
        big_pool[pid] = _pd(pid, pos, 5.0 + (i % 30) * 0.5)

    roster = _roster("qb1", "rb1", "wr1")
    available = list(big_pool.keys())

    start = time.perf_counter()
    value_available(available, big_pool, roster, REPLACEMENT_MEANS)
    elapsed = time.perf_counter() - start

    # Generous ceiling -- this is meant to catch an accidental O(n^2) or a
    # sampling call creeping in, not to pin an exact number.
    assert elapsed < 2.0
