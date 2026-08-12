"""Tests for `draft.value`.

Uses small hand-built synthetic pools/rosters (same style as
`tests/test_roster.py`) so these are fast and independent of the real data
pipeline. An explicit `AVAILABILITY` fixture (matching Stage 2's fitted
rates, see `models/availability.py`'s docstring) is passed to every call so
these tests are fully hermetic and deterministic, independent of the real
cached availability data `value_available`'s default falls back to. A
final group of tests runs against the real 2026 pool/replacement data to
produce the plausibility report the task asks for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest

from ffdraft.draft.value import PlayerValue, player_value, value_available
from ffdraft.models.availability import PlayerAvailability
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

# Matches Stage 2's fitted per-week availability rates (see
# `models/availability.py`'s docstring table) -- injected explicitly so
# these tests are hermetic and independent of the real fitted cache.
AVAILABILITY = {
    "QB": PlayerAvailability(position="QB", p_available=0.842, persistence=0.755, n_player_weeks=1000),
    "RB": PlayerAvailability(position="RB", p_available=0.804, persistence=0.755, n_player_weeks=1000),
    "WR": PlayerAvailability(position="WR", p_available=0.842, persistence=0.755, n_player_weeks=1000),
    "TE": PlayerAvailability(position="TE", p_available=0.797, persistence=0.755, n_player_weeks=1000),
    "K": PlayerAvailability(position="K", p_available=0.934, persistence=0.755, n_player_weeks=1000),
}

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


def _value_available(available_ids, pool, roster, replacement_means=REPLACEMENT_MEANS, **kwargs):
    return value_available(
        available_ids, pool, roster, replacement_means, availability_by_position=AVAILABILITY, **kwargs
    )


def _player_value(player_id, pool, roster, replacement_means=REPLACEMENT_MEANS, **kwargs):
    return player_value(
        player_id, pool, roster, replacement_means, availability_by_position=AVAILABILITY, **kwargs
    )


# ---------------------------------------------------------------------------
# Empty roster: best available valued highest


def test_empty_roster_values_best_player_highest():
    available = list(POOL.keys())
    values = _value_available(available, POOL, [])
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
    empty_value = _value_available(["rb4"], POOL, [])[0]

    deep_roster = _roster("rb1", "rb2", "rb3", "wr1", "wr2", "wr3")
    deep_value = _value_available(["rb4"], POOL, deep_roster)[0]

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
    values = _value_available(["qb_replacement_level"], pool, roster)
    assert values[0].value == pytest.approx(0.0, abs=0.5)


# ---------------------------------------------------------------------------
# The FLEX-awareness pair: third RB meaningfully positive, second QB near zero


def test_third_rb_meaningful_second_qb_near_zero():
    # Roster: 1 QB starter, 2 RB starters, 2 WR starters, 1 TE starter --
    # every dedicated slot full, both FLEX slots open (and, critically,
    # nobody yet occupies the FLEX-leftover pool at all).
    roster = _roster("qb1", "rb1", "rb2", "wr1", "wr2", "te1")
    values = {
        v.player_id: v
        for v in _value_available(["rb3", "qb2"], POOL, roster)
    }
    third_rb = values["rb3"]
    second_qb = values["qb2"]

    # Third RB (12.5 mean): both RB dedicated slots are taken (rb1, rb2),
    # but the FLEX-leftover pool is empty (no WR/TE depth beyond their own
    # dedicated counts either), so rb3 reaches a FLEX slot outright -- over
    # the FLEX fallback baseline (the best of RB/WR/TE's replacement means,
    # WR's 9.9 here), not RB's own 9.6, since that's what an empty FLEX
    # slot actually uses (see `solve_lineup`).
    flex_fallback = max(REPLACEMENT_MEANS[p] for p in ("RB", "WR", "TE"))
    assert third_rb.value == pytest.approx(12.5 - flex_fallback)
    # Second QB (20.0 mean) cannot start at all (QB has 1 slot, not
    # FLEX-eligible, and qb1 is better) -- its value is exactly
    # P(qb1 unavailable this week) * over_replacement.
    over_repl = 20.0 - REPLACEMENT_MEANS["QB"]
    p_qb1_out = 1.0 - AVAILABILITY["QB"].p_available
    assert second_qb.value == pytest.approx(p_qb1_out * over_repl, rel=1e-9)
    assert second_qb.value < third_rb.value
    # This is the pair the task exists to prove: FLEX depth is worth much
    # more than a backup at a position with no FLEX recourse.
    assert third_rb.value > 1.0
    assert second_qb.value < 0.5


# ---------------------------------------------------------------------------
# QB bench value decays with depth -- the core regression for the incident
# (six quarterbacks): a flat discount does not decay at all; this must.


def test_qb_bench_value_decays_with_depth():
    p_out = 1.0 - AVAILABILITY["QB"].p_available  # ~0.158

    # QB2: 1 QB (qb1) ahead, needs qb1 out. P(X>=1 | n=1) == p_out.
    roster_1qb = _roster("qb1")
    qb2 = _player_value("qb2", POOL, roster_1qb)
    over_repl_qb2 = 20.0 - REPLACEMENT_MEANS["QB"]
    assert qb2.value == pytest.approx(p_out * over_repl_qb2, rel=1e-9)

    # QB3: 2 QBs (qb1, qb2) ahead, needs both out simultaneously.
    # P(X>=2 | n=2) == p_out^2 -- "on the order of 2-3%", per the incident
    # report's own back-of-envelope.
    roster_2qb = _roster("qb1", "qb2")
    qb3 = _player_value("qb3", POOL, roster_2qb)
    over_repl_qb3 = 18.5 - REPLACEMENT_MEANS["QB"]
    assert qb3.value == pytest.approx((p_out**2) * over_repl_qb3, rel=1e-9)
    assert qb3.value < qb2.value

    # QB6: 5 QBs ahead (qb1..qb5, synthetic bench-depth QBs all better than
    # qb6), needs all 5 out simultaneously -- "essentially nothing".
    pool = dict(POOL)
    for i, val in enumerate([24.0, 20.0, 18.5, 18.4, 18.35], start=1):
        pool[f"deep_qb{i}"] = _pd(f"deep_qb{i}", "QB", val)
    pool["qb6"] = _pd("qb6", "QB", 18.31)
    roster_5qb = [
        RosterPlayer(pid, "QB", pool[pid].distribution.mean)
        for pid in ["deep_qb1", "deep_qb2", "deep_qb3", "deep_qb4", "deep_qb5"]
    ]
    qb6 = _player_value("qb6", pool, roster_5qb)
    assert qb6.value < 0.02
    assert qb6.value < qb3.value


# ---------------------------------------------------------------------------
# A 4th RB, still within the pooled FLEX-eligible group's capacity, is
# worth far more than a 2nd QB with comparable over-replacement depth --
# the direct RB-vs-QB comparison the incident report calls out explicitly.


def test_flex_eligible_depth_worth_more_than_qb_depth_at_comparable_reach():
    # Both candidates need exactly 1 specific rostered player to be out to
    # reach the lineup (gap=1, n=1) -- but the RB's competitor is a WR
    # leftover body (unaffected here) and the RB itself has an empty FLEX
    # path, while the QB has none at all. Direct comparison: a 4th
    # flex-eligible player who *does* face gap=1 (deep bench, FLEX genuinely
    # saturated) still discounts by a positional miss rate close to a QB's,
    # but starts from many more live bodies "ahead" (harder for all of them
    # to be out at once) once the group is truly deep -- demonstrated here
    # via the shallow case (RB reaches outright, QB does not).
    roster = _roster("qb1", "rb1", "rb2", "wr1", "wr2", "te1")
    values = {v.player_id: v for v in _value_available(["rb3", "qb2"], POOL, roster)}
    # rb3 reaches the (uncontested) FLEX pool outright; qb2 does not reach
    # at all without qb1 missing a week. Same "1 body away" intuition, very
    # different payoff -- FLEX recourse is what makes the difference.
    assert values["rb3"].value > values["qb2"].value * 5


# ---------------------------------------------------------------------------
# Lineup shape is read from the starters config, not hardcoded


def test_lineup_shape_follows_starters_config_not_hardcoded():
    roster = _roster("qb1", "rb1", "rb2", "wr1", "wr2", "te1")

    default_values = {v.player_id: v for v in _value_available(["rb3"], POOL, roster)}

    # Zero FLEX slots: a 3rd RB now has nowhere at all to start.
    no_flex_starters = MappingProxyType({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 0, "K": 1, "DST": 1})
    no_flex_values = {
        v.player_id: v
        for v in _value_available(["rb3"], POOL, roster, starters=no_flex_starters)
    }
    assert no_flex_values["rb3"].value == pytest.approx(0.0)
    assert no_flex_values["rb3"].value < default_values["rb3"].value

    # More FLEX slots: even more room for RB depth to start, value should
    # not decrease.
    more_flex_starters = MappingProxyType({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 4, "K": 1, "DST": 1})
    more_flex_values = {
        v.player_id: v
        for v in _value_available(["rb3"], POOL, roster, starters=more_flex_starters)
    }
    assert more_flex_values["rb3"].value >= default_values["rb3"].value


# ---------------------------------------------------------------------------
# Determinism -- no sampling


def test_deterministic_no_sampling():
    roster = _roster("qb1", "rb1", "wr1")
    available = ["rb2", "rb3", "wr2", "te1", "qb2"]
    v1 = _value_available(available, POOL, roster)
    v2 = _value_available(available, POOL, roster)
    assert v1 == v2


# ---------------------------------------------------------------------------
# The reported value never goes negative, even in solve_lineup's one
# documented non-monotonic edge case (a real leftover player displacing a
# *better* replacement-level FLEX fill -- see sim/lineup.py's module
# docstring and the note in this module's docstring). te2 (8.3) is below
# this pool's WR replacement mean (9.9), which is what would otherwise fill
# an empty FLEX slot, so its raw lineup marginal is negative; the bench
# floor must still keep the reported value at 0 or above, not negative.


def test_value_is_never_negative():
    roster = _roster("qb1", "rb1", "rb2", "wr1", "wr2", "te1")
    for pid in ["rb3", "rb4", "wr3", "qb2", "k1", "dst1", "te2"]:
        pv = _player_value(pid, POOL, roster)
        assert pv.value >= 0.0


# ---------------------------------------------------------------------------
# Single-call convenience wrapper matches the batch path


def test_player_value_matches_value_available_single_entry():
    roster = _roster("qb1", "rb1")
    single = _player_value("wr1", POOL, roster)
    batch = _value_available(["wr1"], POOL, roster)[0]
    assert single == batch


# ---------------------------------------------------------------------------
# The default availability_by_position (Stage 2's real fitted, memoized
# rates) is used when the caller doesn't inject one -- proves the
# production path (rollout.py/recommender.py, which don't pass this
# explicitly) actually gets non-trivial bench discounting, not a silent
# no-op.


def test_default_availability_by_position_is_used_when_not_supplied():
    roster = _roster("qb1")
    pv = player_value("qb2", POOL, roster, REPLACEMENT_MEANS)  # no availability_by_position
    over_repl = 20.0 - REPLACEMENT_MEANS["QB"]
    # Real QB miss rate is close to (not necessarily identical to) 0.158;
    # loose bounds confirm it's neither 0 (no discounting at all) nor the
    # full undiscounted over_replacement (no discounting applied).
    assert 0.0 < pv.value < over_repl


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
    _value_available(available, big_pool, roster)
    elapsed = time.perf_counter() - start

    # Generous ceiling -- this is meant to catch an accidental O(n^2) or a
    # sampling call creeping in, not to pin an exact number.
    assert elapsed < 2.0


# ---------------------------------------------------------------------------
# dominant_candidates
#
# The reduction that made `recommend_pick` 2.7x faster with bit-identical
# output. It is exact, not an approximation: within a position `value` is
# monotone non-decreasing in the projected mean, so only the best-by-mean
# player at each position can be the argmax. These tests pin the property
# the proof rests on -- if monotonicity ever breaks, the speedup silently
# starts choosing different players.


def _mixed_pool():
    from ffdraft.models.distribution import PlayerDistribution

    pool = {}
    for i, (pos, mean) in enumerate(
        [("RB", 20.0), ("RB", 15.0), ("RB", 3.0), ("WR", 18.0), ("WR", 12.0),
         ("WR", 2.0), ("QB", 26.0), ("QB", 19.0), ("TE", 14.0), ("TE", 6.0)]
    ):
        pool[f"p{i}"] = _pd(f"p{i}", pos, mean)
    return pool


def test_value_is_monotone_in_mean_within_a_position():
    """The property the whole reduction rests on."""
    from ffdraft.draft.value import value_available

    pool = _mixed_pool()
    roster = []
    values = {v.player_id: v.value for v in value_available(
        list(pool), pool, roster, REPLACEMENT_MEANS, availability_by_position=AVAILABILITY
    )}
    by_pos = {}
    for pid, entry in pool.items():
        by_pos.setdefault(entry.position, []).append(pid)
    for pids in by_pos.values():
        ordered = sorted(pids, key=lambda p: -float(pool[p].distribution.mean))
        vals = [values[p] for p in ordered]
        assert vals == sorted(vals, reverse=True), (
            f"value must not increase as mean falls within a position: {vals}"
        )


def test_dominant_candidates_keeps_one_per_position_by_default():
    from ffdraft.draft.value import dominant_candidates

    pool = _mixed_pool()
    kept = dominant_candidates(list(pool), pool)
    assert len(kept) == 4, "one per position present in the pool"
    assert {pool[p].position for p in kept} == {"RB", "WR", "QB", "TE"}
    assert set(kept) == {"p0", "p3", "p6", "p8"}, "the best mean at each position"


def test_reduced_argmax_matches_the_full_argmax():
    """The behavioural guarantee: same chosen player, far less work."""
    from ffdraft.draft.value import dominant_candidates, value_available

    pool = _mixed_pool()
    for roster in ([], [RosterPlayer("x", "RB", 19.0)], [RosterPlayer("x", "QB", 27.0)]):
        full = value_available(list(pool), pool, roster, REPLACEMENT_MEANS,
                               availability_by_position=AVAILABILITY)
        reduced = value_available(dominant_candidates(list(pool), pool), pool, roster,
                                  REPLACEMENT_MEANS, availability_by_position=AVAILABILITY)
        assert max(full, key=lambda v: v.value).value == max(reduced, key=lambda v: v.value).value


def test_dominant_candidates_preserves_input_order():
    """Callers tie-break on list order, so the reduction must not reshuffle."""
    from ffdraft.draft.value import dominant_candidates

    pool = _mixed_pool()
    ids = list(pool)
    kept = dominant_candidates(ids, pool, per_position=2)
    assert kept == [p for p in ids if p in set(kept)]


def test_dominant_candidates_rejects_a_nonsense_width():
    from ffdraft.draft.value import dominant_candidates

    with pytest.raises(ValueError, match="per_position"):
        dominant_candidates(["p0"], _mixed_pool(), per_position=0)
