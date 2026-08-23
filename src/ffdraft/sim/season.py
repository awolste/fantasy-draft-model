"""14-week regular season + 6-team/3-round playoff bracket -> champion.

Task 7, the capstone of Stage 2: given ten rosters (each a list of players
with a sampleable weekly-points distribution and, for skill positions and
kickers, an availability model), simulate the season `n_sims` times and
report each team's championship count.

## Why this is vectorized, not a Python loop per (team, week, sim)

`sim/lineup.py`'s `solve_lineup` is exact but pure Python -- ~21us/call, and
a naive "call it once per (team, week, sim)" implementation would need
`n_teams * total_weeks * n_sims` calls (1,000 sims => 170,000 calls => ~3.5s
on lineup solving *alone*, before any sampling). This module instead
re-derives the same two-phase exact algorithm (see `sim/lineup.py`'s
docstring for the proof) as pure numpy array ops that operate on every
(sim, week) pair *simultaneously*, one call per team. That drops lineup
solving to `n_teams` vectorized calls total, independent of `n_sims`.
`test_lineup_matches_solve_lineup_on_randomized_rosters` in
`tests/test_season.py` proves this vectorized path agrees with
`solve_lineup` exactly (not approximately) on many randomized cases --
required per the Task 7 spec, since a fast-but-subtly-wrong lineup solver
would understate every roster slightly and invisibly.

## Reproducibility

Every simulation takes an explicit integer `seed`. A single
`np.random.Generator` is created from that seed and threaded through every
sampling call in a fixed order (availability batch, then each player's
distribution in roster order, then the tiny tie-break noise draw) -- so the
same seed always produces bit-identical results, and no call anywhere
touches global numpy random state.

## The schedule

10 teams over 14 weeks is not a clean double round-robin (18 weeks) or even
a clean single round-robin (9 weeks, one week left over 5 times). This
module builds one **fixed** schedule via the standard "circle method" single
round-robin (9 weeks, every team plays every other team exactly once), then
replays its first `14 - 9 = 5` weeks a second time to fill out the season
(`build_regular_season_schedule`). Two things follow from that:

- **Fixed, not resampled per simulation.** A real ESPN league publishes one
  schedule before Week 1 and does not reshuffle it simulation-by-simulation
  -- this module simulates *a* season under *a* schedule, not "schedule
  luck averaged away." Every simulation run (and, in Stage 3, every
  candidate-pick comparison) uses the identical schedule, so differences
  between runs are attributable to roster/player variance, not schedule
  variance. This is a deliberate variance-reduction choice: resampling the
  schedule per simulation would add an extra, real but here-irrelevant
  source of noise to a comparison whose whole point is isolating
  roster-driven differences.
- **The first 5 pairings play twice, not once.** This mirrors how real
  10-team, 14-week fantasy schedules are commonly built (there is no way to
  give everyone an equal number of games against every opponent in 14 weeks
  with 10 teams), and is reported here rather than silently chosen.

## Hindsight bias (see the Task 7 report for the quantified finding)

`solve_lineup` -- and this module's vectorized equivalent -- picks a
lineup from each week's *realized* score, which is perfect hindsight no
real manager has. `simulate_season(..., hindsight=False)` runs the
identical season (same sampled scores, same schedule, same playoff
bracket) but selects each week's starters by each player's *projected
mean* (`distribution.mean`) instead, among that week's actually-available
players -- i.e. "a manager who reads the projections and starts his best
projected players, then the games happen." Comparing `hindsight=True` vs
`hindsight=False` on the same roster set isolates how much of a team's
title equity comes purely from lineup-selection hindsight rather than
roster quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..league import (
    FLEX_ELIGIBLE,
    N_TEAMS,
    PLAYOFF_BYES,
    PLAYOFF_ROUNDS,
    PLAYOFF_TEAMS,
    REGULAR_SEASON_WEEKS,
    STARTERS,
)
from ..models.availability import PlayerAvailability, sample_availability_batch
from ..models.base import WeeklyDistribution
from .lineup import _validate_replacement_means

# ---------------------------------------------------------------------------
# Roster representation


@dataclass(frozen=True)
class SeasonRosterPlayer:
    """One roster player for the season simulator.

    `distribution` is anything satisfying `models.base.WeeklyDistribution`
    (a fitted `PlayerDistribution.distribution`, the shared
    `defense.dst_distribution()`, a kicker distribution, or a test double).
    `availability` is `None` for positions that are always available in this
    model (DST -- a team is never without a defense) or when a caller wants
    to model a player as never missing a game; otherwise it is the
    position's fitted `PlayerAvailability` (from
    `availability.availability_by_position()`), which every player at that
    position may share -- each still samples an *independent* Markov chain.
    """

    player_id: str
    position: str
    distribution: WeeklyDistribution
    availability: PlayerAvailability | None = None


TeamRoster = Sequence[SeasonRosterPlayer]


@dataclass(frozen=True)
class SeasonSimResult:
    """`champion_counts[i]` is how many of `n_sims` simulations team `i`
    (index into the `rosters` sequence passed to `simulate_season`) won."""

    champion_counts: tuple[int, ...]
    n_sims: int
    champion_seed_counts: tuple[int, ...] = ()
    """`champion_seed_counts[k]` = number of simulations where the champion
    entered the playoffs as seed `k + 1` (0 = seed 1, 1 = seed 2, ...).
    Length equals `playoff_teams`; seeds beyond that never appear (they
    don't make the playoffs) and so are absent, not zero-padded to
    `n_teams`."""

    def __post_init__(self) -> None:
        if sum(self.champion_counts) != self.n_sims:
            raise ValueError(
                f"champion_counts sums to {sum(self.champion_counts)}, expected "
                f"n_sims={self.n_sims} -- every simulation must produce exactly one champion."
            )

    @property
    def championship_probabilities(self) -> tuple[float, ...]:
        return tuple(c / self.n_sims for c in self.champion_counts)


# ---------------------------------------------------------------------------
# Schedule


def round_robin_pairs(n_teams: int) -> list[list[tuple[int, int]]]:
    """Single round-robin via the standard "circle method": `n_teams - 1`
    weeks (requires `n_teams` even), each team playing every other team
    exactly once across the whole cycle."""
    if n_teams % 2 != 0:
        raise ValueError(f"round_robin_pairs requires an even team count, got {n_teams}")
    if n_teams < 2:
        raise ValueError(f"round_robin_pairs requires at least 2 teams, got {n_teams}")

    teams = list(range(n_teams))
    weeks: list[list[tuple[int, int]]] = []
    for _ in range(n_teams - 1):
        half = n_teams // 2
        pairs = [(teams[i], teams[n_teams - 1 - i]) for i in range(half)]
        weeks.append(pairs)
        # Rotate every seat except the first (fixed) one.
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return weeks


def build_regular_season_schedule(
    n_teams: int = N_TEAMS, n_weeks: int = REGULAR_SEASON_WEEKS
) -> list[list[tuple[int, int]]]:
    """Fixed `n_weeks`-week schedule. See module docstring's "The schedule"
    section for what this does and why it is fixed rather than resampled
    per simulation."""
    base = round_robin_pairs(n_teams)
    if n_weeks <= len(base):
        return list(base[:n_weeks])
    extra = n_weeks - len(base)
    return list(base) + list(base[:extra])


# ---------------------------------------------------------------------------
# Vectorized lineup solve: the numpy re-derivation of `sim/lineup.py`'s
# exact two-phase algorithm, applied to every (sim, week) pair at once.


def _dedicated_positions(starters: Mapping[str, int], flex_eligible: frozenset[str]) -> list[str]:
    return sorted({pos for pos in starters if pos != "FLEX"} | set(flex_eligible))


def solve_team_lineups_vectorized(
    value_by_pos: Mapping[str, np.ndarray],
    available_by_pos: Mapping[str, np.ndarray],
    replacement_means: Mapping[str, float],
    n_sims: int,
    n_weeks: int,
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
    key_by_pos: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """One team's optimal lineup total for every (sim, week) pair at once.

    `value_by_pos[pos]` and `available_by_pos[pos]` are `(n_players_pos,
    n_sims, n_weeks)` arrays (points and availability for that position's
    roster players; a position absent from the mapping, or present with
    `n_players_pos == 0`, means the team has no player at that position).

    `key_by_pos`, if given, is used only to *rank* players within a
    position/FLEX pool (same shape as `value_by_pos[pos]`, or broadcastable
    to it) -- the *points* actually summed always come from `value_by_pos`.
    This is what lets `hindsight=False` in `simulate_season` rank by
    projected mean while still scoring the realized result. When omitted,
    `key_by_pos = value_by_pos` (ranking by realized score -- hindsight
    lineup selection, matching `solve_lineup`'s behavior exactly).

    Returns a `(n_sims, n_weeks)` array of total lineup points.

    See `sim/lineup.py`'s module docstring for the exchange-argument proof
    that phase-1-reserve-then-phase-2-pool-FLEX is exactly optimal; this
    function is that same algorithm, per (sim, week) slice, done with numpy
    sorting instead of Python `sorted()`.
    """
    if key_by_pos is None:
        key_by_pos = value_by_pos

    shape = (n_sims, n_weeks)
    dedicated = _dedicated_positions(starters, flex_eligible)

    total = np.zeros(shape, dtype=float)
    leftover_rows: list[np.ndarray] = []

    for pos in dedicated:
        count = starters.get(pos, 0)
        values = value_by_pos.get(pos)
        avail = available_by_pos.get(pos)
        keys = key_by_pos.get(pos)
        n_p = 0 if values is None else values.shape[0]

        if n_p > 0:
            keys_arr = np.broadcast_to(keys, values.shape)
            masked_key = np.where(avail, keys_arr, -np.inf)
            if keys is values:
                # Ranking *by* the same array we are ranking, which is what
                # `hindsight=True` means -- so the sorted values are just the
                # sorted keys, and the argsort plus the gather it feeds are
                # both avoidable. The rows this leaves as `-inf` (the
                # unavailable players) are exactly the rows both consumers
                # below discard on `avail_count`, so nothing that is read
                # changes. Measured 2.36x on this block.
                sorted_values = -np.sort(-masked_key, axis=0)
            else:
                order = np.argsort(-masked_key, axis=0)
                sorted_values = np.take_along_axis(values, order, axis=0)
            avail_count = np.asarray(avail, dtype=bool).sum(axis=0)
        else:
            sorted_values = None
            avail_count = np.zeros(shape, dtype=int)

        if count > 0:
            reserved = np.zeros(shape, dtype=float)
            for i in range(count):
                if sorted_values is not None and i < n_p:
                    use_real = i < avail_count
                    reserved = reserved + np.where(use_real, sorted_values[i], replacement_means[pos])
                else:
                    reserved = reserved + replacement_means[pos]
            total = total + reserved

        if pos in flex_eligible and sorted_values is not None:
            for j in range(count, n_p):
                is_real = j < avail_count
                leftover_rows.append(np.where(is_real, sorted_values[j], -np.inf))

    flex_count = starters.get("FLEX", 0)
    if flex_count > 0:
        flex_replacement = max(replacement_means[p] for p in flex_eligible)
        if leftover_rows:
            pool = np.stack(leftover_rows, axis=0)
            n_l = pool.shape[0]
            sorted_pool = -np.sort(-pool, axis=0)
        else:
            n_l = 0
            sorted_pool = None

        flex_total = np.zeros(shape, dtype=float)
        for i in range(flex_count):
            if sorted_pool is not None and i < n_l:
                row = sorted_pool[i]
                is_real = np.isfinite(row)
                flex_total = flex_total + np.where(is_real, row, flex_replacement)
            else:
                flex_total = flex_total + flex_replacement
        total = total + flex_total

    return total


# ---------------------------------------------------------------------------
# Sampling


def _group_by_position(
    positions: Sequence[str], arrays: Sequence[np.ndarray]
) -> dict[str, np.ndarray]:
    """Stack a team's per-player `(n_sims, n_weeks)` arrays into
    `{position: (n_players_pos, n_sims, n_weeks)}`."""
    by_pos: dict[str, list[np.ndarray]] = {}
    for pos, arr in zip(positions, arrays):
        by_pos.setdefault(pos, []).append(arr)
    return {pos: np.stack(arrs, axis=0) for pos, arrs in by_pos.items()}


def _sample_all(
    rosters: Sequence[TeamRoster],
    n_sims: int,
    n_weeks: int,
    rng: np.random.Generator,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Sample every player's weekly scores and availability.

    Returns `(scores, available)`, each `list[team][player_index]` of
    `(n_sims, n_weeks)` arrays. Availability is sampled in one batched call
    across every player with a `PlayerAvailability` (any order, but a fixed
    one -- flat roster order -- so results are reproducible from `rng`'s
    state); players with `availability=None` are always available. Each
    player's score distribution is then sampled individually (cheap --
    see `models/base.py`'s performance note) in the same fixed order.
    """
    flat: list[tuple[int, int, SeasonRosterPlayer]] = [
        (t, i, player) for t, roster in enumerate(rosters) for i, player in enumerate(roster)
    ]

    avail_flat_idx = [k for k, (_, _, p) in enumerate(flat) if p.availability is not None]
    if avail_flat_idx:
        p_avail = np.array([flat[k][2].availability.p_available for k in avail_flat_idx])
        persistence = np.array([flat[k][2].availability.persistence for k in avail_flat_idx])
        avail_batch = sample_availability_batch(p_avail, persistence, rng, n_sims, n_weeks)
    else:
        avail_batch = np.empty((0, n_sims, n_weeks), dtype=bool)

    scores: list[list[np.ndarray | None]] = [[None] * len(roster) for roster in rosters]
    available: list[list[np.ndarray | None]] = [[None] * len(roster) for roster in rosters]

    avail_cursor = 0
    for k, (t, i, player) in enumerate(flat):
        scores[t][i] = player.distribution.sample(rng, n_sims * n_weeks).reshape(n_sims, n_weeks)
        if player.availability is not None:
            available[t][i] = avail_batch[avail_cursor]
            avail_cursor += 1
        else:
            available[t][i] = np.ones((n_sims, n_weeks), dtype=bool)

    return scores, available  # type: ignore[return-value]


def _team_weekly_totals(
    rosters: Sequence[TeamRoster],
    scores: list[list[np.ndarray]],
    available: list[list[np.ndarray]],
    replacement_means: Mapping[str, float],
    n_sims: int,
    n_weeks: int,
    starters: Mapping[str, int],
    flex_eligible: frozenset[str],
    hindsight: bool,
) -> np.ndarray:
    """`(n_teams, n_sims, n_weeks)` array of each team's optimal starting
    lineup total, for every simulated week (regular season and playoff
    weeks alike -- the algorithm is identical for both)."""
    _validate_replacement_means(replacement_means, starters, flex_eligible)

    out = np.empty((len(rosters), n_sims, n_weeks), dtype=float)
    for t, roster in enumerate(rosters):
        positions = [p.position for p in roster]
        value_by_pos = _group_by_position(positions, scores[t])
        avail_by_pos = _group_by_position(positions, available[t])
        key_by_pos = None
        if not hindsight:
            means = [p.distribution.mean for p in roster]
            mean_arrays = [
                np.full((n_sims, n_weeks), m, dtype=float) for m in means
            ]
            key_by_pos = _group_by_position(positions, mean_arrays)
        out[t] = solve_team_lineups_vectorized(
            value_by_pos,
            avail_by_pos,
            replacement_means,
            n_sims,
            n_weeks,
            starters=starters,
            flex_eligible=flex_eligible,
            key_by_pos=key_by_pos,
        )
    return out


# ---------------------------------------------------------------------------
# Playoffs


def _validate_bracket_shape(playoff_teams: int, playoff_rounds: int, playoff_byes: int, n_teams: int) -> None:
    """This module's bracket builder supports exactly one shape: byes go
    straight to round 2, round 1's winners exactly fill the other round-2
    slots, and round 2's two winners meet in a single round-3 final. That is
    this league's actual shape (`PLAYOFF_TEAMS=6, PLAYOFF_ROUNDS=3,
    PLAYOFF_BYES=2`) read from `league.py`, not hardcoded here -- but a
    different shape (e.g. 4 byes, or 4 rounds) needs `_run_playoffs`
    extended, not silently mis-bracketed, hence the loud failure."""
    if playoff_teams > n_teams:
        raise ValueError(f"playoff_teams ({playoff_teams}) exceeds n_teams ({n_teams})")
    if playoff_rounds != 3 or playoff_byes != 2:
        raise NotImplementedError(
            "season.py's bracket builder only supports a 3-round bracket with exactly "
            f"2 byes; got playoff_rounds={playoff_rounds}, playoff_byes={playoff_byes}. "
            "Extend _run_playoffs before using a different playoff shape."
        )
    round1_teams = playoff_teams - playoff_byes
    if round1_teams <= 0 or round1_teams % 2 != 0:
        raise ValueError(
            f"playoff_teams - playoff_byes must be a positive even number, got {round1_teams}"
        )
    if round1_teams // 2 != playoff_byes:
        raise ValueError(
            f"round-1 winners ({round1_teams // 2}) must equal the bye count ({playoff_byes}) "
            "for round 2 to pair every bye against a round-1 winner and round 3 to be a single final."
        )


def _seed_teams(
    wins: np.ndarray, points_for: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """`(n_teams, n_sims)` win/points arrays -> `(n_teams, n_sims)` array of
    team indices ordered best (row 0, seed 1) to worst, per simulation.
    Standard tiebreak: more wins, then more points-for; a further exact tie
    (possible with fixed replacement-level fillers) is broken by an
    infinitesimal per-(team, sim) random jitter so no simulation is left
    with an undefined seed order.
    """
    n_teams, n_sims = wins.shape
    jitter = rng.uniform(0, 1e-9, size=(n_teams, n_sims))
    key = wins.astype(float) * 1.0e7 + points_for + jitter
    return np.argsort(-key, axis=0)


def _pick_winners(
    team_totals: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, week: int
) -> np.ndarray:
    """`team_totals` is `(n_teams, n_sims, n_weeks)`; `idx_a`/`idx_b` are
    `(n_sims,)` team indices facing off that `week`, per simulation. Returns
    the `(n_sims,)` winning team index per simulation (ties cannot occur --
    `team_totals` already had jitter-free but continuous scores; see
    `simulate_season` for the shared tie-break noise added once up front)."""
    scores_a = np.take_along_axis(team_totals[:, :, week], idx_a[None, :], axis=0)[0]
    scores_b = np.take_along_axis(team_totals[:, :, week], idx_b[None, :], axis=0)[0]
    return np.where(scores_a >= scores_b, idx_a, idx_b)


def _run_playoffs(
    team_totals: np.ndarray,
    seeds: np.ndarray,
    regular_season_weeks: int,
    playoff_byes: int,
) -> np.ndarray:
    """Returns `(n_sims,)` champion team index per simulation."""
    byes = seeds[:playoff_byes]  # (playoff_byes, n_sims)
    field = seeds[playoff_byes:]  # (round1_teams, n_sims)
    round1_teams = field.shape[0]

    round1_winners = []
    for i in range(round1_teams // 2):
        a, b = field[i], field[round1_teams - 1 - i]
        round1_winners.append(_pick_winners(team_totals, a, b, regular_season_weeks))

    round2_winners = []
    for i in range(playoff_byes):
        a = byes[i]
        b = round1_winners[round1_teams // 2 - 1 - i]
        round2_winners.append(_pick_winners(team_totals, a, b, regular_season_weeks + 1))

    champion = _pick_winners(
        team_totals, round2_winners[0], round2_winners[1], regular_season_weeks + 2
    )
    return champion


# ---------------------------------------------------------------------------
# Public entry point


def simulate_season(
    rosters: Sequence[TeamRoster],
    n_sims: int,
    seed: int,
    replacement_means: Mapping[str, float],
    *,
    hindsight: bool = True,
    schedule: Sequence[Sequence[tuple[int, int]]] | None = None,
    starters: Mapping[str, int] = STARTERS,
    flex_eligible: frozenset[str] = FLEX_ELIGIBLE,
    n_teams: int = N_TEAMS,
    regular_season_weeks: int = REGULAR_SEASON_WEEKS,
    playoff_teams: int = PLAYOFF_TEAMS,
    playoff_rounds: int = PLAYOFF_ROUNDS,
    playoff_byes: int = PLAYOFF_BYES,
) -> SeasonSimResult:
    """Simulate the season `n_sims` times from `n_teams` rosters and return
    each team's championship count. See the module docstring for the
    schedule, hindsight, and reproducibility decisions.

    `replacement_means` is typically `lineup.build_replacement_means(
    replacement.replacement_by_position(), defense.dst_distribution().mean)`
    -- computed once by the caller, not per simulation.
    """
    if len(rosters) != n_teams:
        raise ValueError(f"expected {n_teams} rosters, got {len(rosters)}")
    _validate_bracket_shape(playoff_teams, playoff_rounds, playoff_byes, n_teams)
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")

    if schedule is None:
        schedule = build_regular_season_schedule(n_teams, regular_season_weeks)
    if len(schedule) != regular_season_weeks:
        raise ValueError(
            f"schedule has {len(schedule)} weeks, expected regular_season_weeks={regular_season_weeks}"
        )

    total_weeks = regular_season_weeks + playoff_rounds
    rng = np.random.default_rng(seed)

    scores, available = _sample_all(rosters, n_sims, total_weeks, rng)
    team_totals = _team_weekly_totals(
        rosters, scores, available, replacement_means, n_sims, total_weeks,
        starters, flex_eligible, hindsight,
    )

    # A shared, tiny tie-break jitter so exact-tie weeks (plausible only via
    # coincident replacement-level fillers) resolve deterministically from
    # `rng` rather than via numpy's arbitrary argsort/`>=` tie handling.
    team_totals = team_totals + rng.uniform(0, 1e-9, size=team_totals.shape)

    wins = np.zeros((n_teams, n_sims), dtype=float)
    points_for = np.zeros((n_teams, n_sims), dtype=float)
    for week, pairs in enumerate(schedule):
        for a, b in pairs:
            sa, sb = team_totals[a, :, week], team_totals[b, :, week]
            wins[a] += sa > sb
            wins[b] += sb > sa
            points_for[a] += sa
            points_for[b] += sb

    seeds = _seed_teams(wins, points_for, rng)
    champion = _run_playoffs(team_totals, seeds, regular_season_weeks, playoff_byes)

    champion_counts = np.bincount(champion, minlength=n_teams)
    champion_seed = np.argmax(seeds == champion[None, :], axis=0)
    champion_seed_counts = np.bincount(champion_seed, minlength=playoff_teams)[:playoff_teams]

    return SeasonSimResult(
        champion_counts=tuple(int(c) for c in champion_counts),
        n_sims=n_sims,
        champion_seed_counts=tuple(int(c) for c in champion_seed_counts),
    )
