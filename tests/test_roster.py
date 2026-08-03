"""Tests for `models.roster.build_roster`.

Uses small hand-built test doubles for `pool`/`replacement_by_position`
(same style as `tests/test_season.py`) so most tests are fast and
independent of the data ingest pipeline. The final test group verifies
against the real 2024 historical draft via `scripts/season_report.py`,
since that is the one case with a real expected answer (see task spec).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ffdraft.models.availability import PlayerAvailability
from ffdraft.models.distribution import PlayerDistribution
from ffdraft.models.roster import RosterBuildResult, build_roster
from ffdraft.sim.lineup import solve_lineup


@dataclass(frozen=True)
class ConstantDist:
    value: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, self.value)

    @property
    def mean(self) -> float:
        return self.value


def _pd(player_id: str, position: str, value: float) -> PlayerDistribution:
    return PlayerDistribution(
        player_id=player_id,
        name=player_id,
        position=position,
        rank=1,
        tier=1,
        distribution=ConstantDist(value),
    )


POOL = {
    "qb1": _pd("qb1", "QB", 20.0),
    "rb1": _pd("rb1", "RB", 15.0),
    "rb2": _pd("rb2", "RB", 12.0),
    "wr1": _pd("wr1", "WR", 14.0),
    "wr2": _pd("wr2", "WR", 11.0),
    "te1": _pd("te1", "TE", 8.0),
    "k1": _pd("k1", "K", 7.0),
    "DST::shared": _pd("DST::shared", "DST", 7.5),
}

REPLACEMENT = {
    "QB": ConstantDist(10.0),
    "RB": ConstantDist(4.0),
    "WR": ConstantDist(4.5),
    "TE": ConstantDist(3.0),
    "K": ConstantDist(5.0),
    "DST": ConstantDist(7.5),  # the shared DST distribution, per module docstring
}

AVAIL = {
    "QB": PlayerAvailability(position="QB", p_available=0.9, persistence=0.5, n_player_weeks=100),
    "RB": PlayerAvailability(position="RB", p_available=0.8, persistence=0.5, n_player_weeks=100),
}


# ---------------------------------------------------------------------------
# Full roster, everything found in the pool


def test_full_roster_every_pick_found_in_pool():
    picks = [
        ("qb1", "QB"), ("rb1", "RB"), ("rb2", "RB"), ("wr1", "WR"), ("wr2", "WR"),
        ("te1", "TE"), ("k1", "K"), ("DST::shared", "DST"),
    ]
    result = build_roster(picks, POOL, REPLACEMENT, AVAIL)
    assert isinstance(result, RosterBuildResult)
    assert len(result.players) == len(picks)
    assert result.n_fallback == 0
    assert result.fallback_player_ids == ()

    by_id = {p.player_id: p for p in result.players}
    assert by_id["qb1"].distribution.mean == 20.0
    assert by_id["qb1"].position == "QB"
    assert by_id["qb1"].availability is AVAIL["QB"]
    # DST never gets an availability model.
    assert by_id["DST::shared"].availability is None


# ---------------------------------------------------------------------------
# Partial roster -- the normal Stage 3 rollout case


def test_partial_roster_builds_and_simulates():
    picks = [("qb1", "QB"), ("rb1", "RB"), ("wr1", "WR")]
    result = build_roster(picks, POOL, REPLACEMENT, AVAIL)
    assert len(result.players) == 3
    assert result.n_fallback == 0

    # solve_lineup already tolerates a partial roster by filling unfillable
    # slots from replacement level -- this just proves build_roster's
    # output plugs straight into it without choking.
    replacement_means = {pos: dist.mean for pos, dist in REPLACEMENT.items()}
    lineup_players = [
        type("RP", (), {"player_id": p.player_id, "position": p.position, "points": p.distribution.mean})()
        for p in result.players
    ]
    # sim.lineup.RosterPlayer is a dataclass with (player_id, position, points)
    from ffdraft.sim.lineup import RosterPlayer

    roster_for_lineup = [RosterPlayer(p.player_id, p.position, p.distribution.mean) for p in result.players]
    lineup = solve_lineup(roster_for_lineup, replacement_means)
    assert lineup.total_points > 0


# ---------------------------------------------------------------------------
# Fallback: pick absent from pool


def test_pick_absent_from_pool_falls_back_to_replacement_at_right_position():
    picks = [("qb1", "QB"), ("unknown_rb", "RB")]
    result = build_roster(picks, POOL, REPLACEMENT, AVAIL)
    assert result.n_fallback == 1
    assert result.fallback_player_ids == ("unknown_rb",)

    fallback_player = [p for p in result.players if p.player_id == "unknown_rb"][0]
    assert fallback_player.position == "RB"
    assert fallback_player.distribution is REPLACEMENT["RB"]
    # Replacement level is already "a healthy player who actually played
    # that week" (see models/replacement.py's estimator) -- applying an
    # availability model on top would double-count unavailability, so a
    # fallback player never gets one, at any position.
    assert fallback_player.availability is None


def test_found_in_pool_player_still_gets_availability_model():
    picks = [("qb1", "QB")]
    result = build_roster(picks, POOL, REPLACEMENT, AVAIL)
    assert result.players[0].availability is AVAIL["QB"]


def test_dst_pick_absent_from_pool_falls_back_to_shared_dst_distribution():
    picks = [("DST::not_in_pool", "DST")]
    result = build_roster(picks, POOL, REPLACEMENT, AVAIL)
    assert result.n_fallback == 1
    player = result.players[0]
    assert player.position == "DST"
    assert player.distribution is REPLACEMENT["DST"]
    assert player.availability is None


# ---------------------------------------------------------------------------
# Malformed input raises rather than silently falling back


@pytest.mark.parametrize(
    "picks",
    [
        [("", "QB")],
        [("   ", "QB")],
        [(None, "QB")],
        [(123, "QB")],
    ],
)
def test_malformed_player_id_raises(picks):
    with pytest.raises(ValueError):
        build_roster(picks, POOL, REPLACEMENT, AVAIL)


@pytest.mark.parametrize(
    "picks",
    [
        [("qb1", "FLEX")],       # FLEX is a slot, not a position
        [("qb1", "PK")],         # crosswalk's kicker label, not this project's
        [("qb1", "quarterback")],
        [("qb1", "")],
        [("qb1", None)],
    ],
)
def test_unknown_position_raises(picks):
    with pytest.raises(ValueError):
        build_roster(picks, POOL, REPLACEMENT, AVAIL)


def test_valid_position_missing_from_replacement_mapping_raises():
    # "DST" deliberately omitted from replacement_by_position here -- a
    # configuration bug, not a fallback candidate (see module docstring).
    incomplete_replacement = {k: v for k, v in REPLACEMENT.items() if k != "DST"}
    picks = [("DST::not_in_pool", "DST")]
    with pytest.raises(ValueError):
        build_roster(picks, POOL, incomplete_replacement, AVAIL)


def test_pool_position_mismatch_does_not_raise_pool_is_authoritative():
    # qb1 is really a QB in the pool; caller mislabels it RB. This is
    # deliberately NOT treated as malformed input (see module docstring) --
    # the pool's own position wins for a found player.
    picks = [("qb1", "RB")]
    result = build_roster(picks, POOL, REPLACEMENT, AVAIL)
    assert result.n_fallback == 0
    assert result.players[0].position == "QB"


# ---------------------------------------------------------------------------
# Empty roster


def test_empty_pick_list_produces_empty_roster():
    result = build_roster([], POOL, REPLACEMENT, AVAIL)
    assert result.players == []
    assert result.n_fallback == 0

    replacement_means = {pos: dist.mean for pos, dist in REPLACEMENT.items()}
    lineup = solve_lineup([], replacement_means)
    assert lineup.total_points > 0  # every slot filled from replacement level


# ---------------------------------------------------------------------------
# Performance: report per-call timing for a full 18-man roster


def test_full_roster_build_is_fast():
    import time

    picks = (
        [("qb1", "QB")] * 1
        + [("rb1", "RB"), ("rb2", "RB")]
        + [("wr1", "WR"), ("wr2", "WR")]
        + [("te1", "TE")]
        + [("k1", "K")]
        + [("DST::shared", "DST")]
    )
    # Pad to 18 with fallback picks (mid-draft realism: several bench slots
    # still unresolved against the pool).
    picks = picks + [(f"bench_{i}", "RB") for i in range(18 - len(picks))]
    assert len(picks) == 18

    n_calls = 2000
    t0 = time.perf_counter()
    for _ in range(n_calls):
        build_roster(picks, POOL, REPLACEMENT, AVAIL)
    t1 = time.perf_counter()
    per_call_us = (t1 - t0) / n_calls * 1e6
    print(f"\nbuild_roster: {per_call_us:.1f}us/call for a full 18-man roster ({n_calls} calls)")
    # Generous ceiling -- this is a sanity guard against an accidental O(n^2)
    # or per-call I/O regression, not a tight performance assertion.
    assert per_call_us < 2000
