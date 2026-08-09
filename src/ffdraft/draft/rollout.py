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
diversify (see `tests/test_rollout.py` for the plausibility check on a
sampled rollout's first two rounds).

## Greedy value alone is not enough: the RB2/QB5 incident

A second, larger incident (after `draft/value.py`'s own bench-value fixes)
surfaced only once real rollouts were run across many seeds: our own
greedy-value-driven roster converged, *every single seed*, to exactly 2
RBs and 4-5 QBs, while the opponent model (fit on real historical drafts)
averaged ~4.8 RBs and ~1.6 QBs. RB is this league's least-available
position (80.4% weekly, and Stage 2's own finding); ending every draft
with zero RB bench is a real handicap, not a quirk of one bad seed.

**Why this is not fixable by a better *value* formula alone -- proven, not
assumed.** The natural instinct is "price bench value by expected starts
across the whole 14-week season, not a single week's snapshot." This
provably cannot change anything: `WeeklyDistribution`/`PlayerAvailability`
seed every player's week-0 state in the chain's *stationary* distribution
(see `models/availability.py`), so by linearity of expectation, the
expected number of qualifying weeks over a 14-week season is exactly `14 x
P(a single week qualifies)` -- a constant rescale of what `draft.value`
already computes, regardless of persistence. Persistence changes how
outages *cluster in time* (variance/timing), not the *expected total*
insurance value. A "season-aware expected value" reformulation is
therefore mathematically rank-invariant to the per-week reachability
`draft.value` already uses. (A tempting alternative -- weight bench value
by each position's *expected outage duration*, `1/(1-persistence)` -- was
checked directly against the real fitted rates and makes the problem
*worse*, not better: QB's fitted persistence (0.833, ~6-week average
absences) is higher than RB's (0.739, ~3.8 weeks) or WR's (0.733, ~3.75
weeks) in this data, so duration-weighting would inflate QB bench value
relative to RB, the opposite of what's needed.)

The real cause is not a pricing error but a genuine blind spot of any
mean-only proxy: championship probability is a *nonlinear* function of a
roster's points (a persistent, multi-week absence with no bench cover is
disproportionately costly if it lands during a playoff push), and a
"roughly right, fast, no-sampling" greedy value function is asked to
optimize a *linear* (expected-points) proxy for that nonlinear objective
by design (see `draft/value.py`'s own module docstring, "the season
simulator remains the source of truth for anything that matters enough to
spend a Monte Carlo budget on"). There is also a real, non-buggy mechanical
contributor once a position's own dedicated slots are full: `solve_lineup`
prices an empty FLEX slot at the *best* of RB/WR/TE's replacement means
(WR's, in this league), so every FLEX-eligible position's *bench* value is
judged against that same, WR-favoring bar -- correct for what `solve_lineup`
actually does, but it structurally advantages the position whose own
replacement level happens to be highest, independent of real season-long
risk.

**Decision: a policy-level guard on `_choose_our_pick`, not a further
change to `draft.value`'s pricing.** Since the value function's own math is
provably already doing the best a mean-only, no-sampling proxy can do, the
fix has to live in the *policy* that consumes it, mirroring how the
opponent model (fit on real human/ADP behavior) already encodes roster-need
behavior it did not have to derive from first principles either
(`models.opponent`'s `roster_decay`). Two guards, both derived from
`league.STARTERS`/`FLEX_ELIGIBLE` (never a hardcoded target roster shape),
apply only once our own starting lineup is otherwise fillable
(`len(roster_pairs) >= starting_slots_total()`) -- see
`_positions_below_bench_floor`/`_positions_at_or_above_bench_ceiling`:

- **A minimum bench floor** (`MIN_BENCH_PER_FLEX_POSITION`, currently 1):
  every FLEX-eligible position must reach at least one bench body beyond
  its own dedicated starter count before candidates outside the
  under-floor position(s) are even considered. This is the "fill starters,
  then buy insurance" behavior the coordinator's own diagnosis names, made
  literal and general -- it does not say *how much* insurance beyond the
  floor, which stays fully greedy.
- **A bench ceiling** (`MAX_EXTRA_BENCH_NO_FLEX`, currently `{"QB": 2, "K":
  1}`) on non-FLEX-eligible, non-DST positions only: once every
  FLEX-eligible position clears its floor, a QB beyond 3 total or a K
  beyond 2 total is excluded from consideration. This is not a new
  assumption -- `draft.value`'s own proven decay curve
  (`tests/test_value.py::test_qb_bench_value_decays_with_depth`) already
  shows a 4th/5th QB's value is near zero, and K's fitted availability
  (93.4%, the highest of any position) makes even a single backup nearly
  valueless; the ceiling just removes the noise/near-ties that otherwise
  let that near-zero value edge out a genuinely useful FLEX-eligible bench
  pick.
- **DST is deliberately untouched by either guard.** Every DST shares the
  exact same distribution as DST's own replacement level (see
  `models/roster.py`'s docstring, "DST has no separate replacement
  level"), so a drafted DST is worth precisely zero marginal value -- the
  value function's conclusion here is correct, not a blind spot, and a
  floor/ceiling would fight it rather than compensate for it.

## Why the floor must depend on what was drafted, not just STARTERS

**A fixed floor of `starters[pos] + 1` for every FLEX-eligible position,
identical regardless of what was actually drafted, was tried first and
failed the structure-differentiation check.** `starters['RB']` is `2`
whatever a rollout's opening looked like, so a fixed floor of `3` total RB
converges *any* structure to exactly 3 RBs once the guard activates,
whether it opened 0RB (0 real RBs by round 3, the floor mechanism supplies
exactly one late) or 2RB (2 elite RBs by round 2, the floor supplies
exactly one more) -- checked directly: forcing WR/WR/WR vs. RB/RB/WR into
picks 8/13/28 and running the rest of an 18-round rollout from each
produced the *identical* final position mix, every seed
(`{RB:3, QB:3, WR:8, TE:2, K:2}`) -- the exact failure mode the structure
study cannot tolerate, because a floor keyed only on the league-wide
`STARTERS` constant cannot, by construction, know or reflect which
structure produced the roster in front of it.

**The fix: the floor must scale with the *quality* of what's already
rostered, not just a position-count constant.** `ELITE_STARTER_BENCH_BONUS`
adds one extra required bench slot to a FLEX-eligible position once it
holds at least one *elite* dedicated starter -- ranked inside the league's
total dedicated-slot count for that position (`starters[pos] * n_teams`,
i.e. good enough to start on every team in a league this size). A 2RB
opening's early picks are, definitionally, elite RBs (that is what "2RB"
means as a structure), so its RB floor becomes 4, not 3; a 0RB opening's
eventual floor-triggered RB pickup is a mid/late-round player who does not
clear the elite-rank bar, so its floor stays at 3. This is derived
entirely from real `PlayerDistribution.rank` and `league.STARTERS`/
`N_TEAMS` -- never a target roster shape or a position-specific "want more
RB" rule -- and it re-derives a different number for every structure
*because* it is a function of what that structure actually drafted, not a
constant applied identically to all of them.

**Why this still does not predetermine Task 7's structure study.** The
mechanism is symmetric across every FLEX-eligible position (a 2RB
opening's *WR* floor is unaffected, since it drafted no elite WRs early;
a 0RB opening's *WR* floor gets the same elite bonus RB's floor would have
gotten, symmetrically, since 0RB's early picks are the elite WRs). It does
not say "RB insurance matters more than WR insurance" -- it says "insurance
matters more for whichever position you already invested elite capital in,
whichever position that turns out to be." See
`tests/test_rollout.py::test_structure_smoke_0rb_vs_2rb_produce_different_final_shapes`
for the check that forcing 0RB vs. 2RB into the first three picks now
produces genuinely different final rosters, not a shared attractor.

## ADP for the opponent model

`models.opponent.pick_probabilities` was calibrated against real ADP
(`reach = log(adp) - log(overall_pick)` -- see `OpponentModel`'s
docstring), so real ADP is the right signal to feed it whenever it exists.
An earlier version of this module used `PlayerDistribution.rank`
(FantasyPros-style consensus rank) as a stand-in for ADP everywhere,
reasoning that a rollout needs full coverage (~500+ players) and
`data/adp_2026.parquet` only has ~246 rows. Measured directly against real
2026 ADP, that proxy is close but not interchangeable: Spearman correlation
0.933 (strong, not 1.0), mean absolute rank disagreement ~24 places, and
individual misses that are not small (e.g. one player ranked 456th by
consensus but 144th by real ADP). Worse, the two scales have different
*density* -- ECR spans the whole ~510-player pool, ADP only ~246 -- which
distorts reach for exactly the players past ADP's range, on the one
quantity the opponent model was fit against.

**Decision: use real ADP (`pool_adp_lookup`, via `models.opponent.
build_pool_adp_lookup`) as the primary signal for every pool player that
matches, and fill in the rest with a linear extrapolation of ADP on
`PlayerDistribution.rank` fit from the matched pairs** -- continuous by
construction (the same fitted line that would interpolate a gap in the
middle of ADP's coverage also extrapolates past its last real rank, so
there is no seam at the boundary the way a flat fallback constant or an
unrelated second scale would create). This is deliberately simple (a single
global linear fit, not a per-position or piecewise one): the goal is a
sensible, continuous placement for players who would almost never
realistically be drafted (past ADP's ~246-deep coverage in a 180-pick
draft) or who merely failed today's name-matching join, not a precise
model of their true ADP.

The ADP source is a parameter (`adp_table`, plus `rankings` for D/ST's
team-abbreviation lookup -- see `build_pool_adp_lookup`'s docstring),
never hardcoded to 2026: a live rollout passes `data/adp_2026.parquet` and
`rankings_2026`; Task 6's historical backtest passes `adp_history` filtered
to the season being replayed, alongside whatever rankings-shaped table
matches that season's pool. Leaving both `None` (the default) skips real-ADP
matching entirely and falls back to the plain rank-proxy for every player --
useful for fast synthetic tests of the draft mechanics that have no
matching ADP data to inject at all.

Whichever branch is used, the match (a `polars` join) and the fallback fit
happen exactly **once per rollout call**, before the pick-by-pick loop
starts, not once per pick -- the loop only ever does an O(1) dict lookup
per available player.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import polars as pl

from ..league import DRAFT_ROUNDS, DRAFT_SLOT, FLEX_ELIGIBLE, N_TEAMS, STARTERS, starting_slots_total
from ..models.base import WeeklyDistribution
from ..models.distribution import PlayerDistribution
from ..models.opponent import (
    DEFAULT_ROSTER_DECAY,
    DEFAULT_TEMPERATURE,
    AdpMatchReport,
    AvailablePlayer,
    OpponentModel,
    build_pool_adp_lookup,
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
# Real ADP for a player pool -- see module docstring, "ADP for the opponent
# model".


def _pool_rows_for_adp_match(
    pool: Mapping[str, PlayerDistribution], rankings: pl.DataFrame
) -> pl.DataFrame:
    """One row per pool player (`player_id`, `name`, `position`, `team`),
    the shape `models.opponent.build_pool_adp_lookup` needs. `team` is only
    populated for D/ST rows -- `PlayerDistribution` itself carries no team,
    so it is recovered from `rankings` (the same rankings table the pool
    was built from), keyed by the D/ST's full team name."""
    dst_team_by_name = dict(
        rankings.filter(pl.col("position") == "DST").select(["name", "team"]).iter_rows()
    )
    return pl.DataFrame(
        [
            {
                "player_id": player_id,
                "name": p.name,
                "position": p.position,
                "team": dst_team_by_name.get(p.name) if p.position == "DST" else None,
            }
            for player_id, p in pool.items()
        ],
        schema={"player_id": pl.String, "name": pl.String, "position": pl.String, "team": pl.String},
    )


def _extrapolate_unmatched_adp(
    pool: Mapping[str, PlayerDistribution],
    matched: Mapping[str, float],
    unmatched_ids: Sequence[str],
) -> dict[str, float]:
    """Fill in `adp` for every pool player `build_pool_adp_lookup` could not
    match, via a linear fit of `adp` on `PlayerDistribution.rank` estimated
    from the matched pairs -- see module docstring for why this (not a flat
    fallback constant) is the chosen way to stay continuous across the
    boundary of real ADP's coverage.

    If fewer than two matched players have distinct ranks, there is not
    enough signal to fit a line at all (this only happens in degenerate/
    test inputs -- the real 2026 match has 218+ points); every unmatched
    player then falls back to its own `rank` directly.
    """
    full = dict(matched)
    if not unmatched_ids:
        return full

    ranks = np.array([pool[pid].rank for pid in matched], dtype=float)
    adps = np.array([matched[pid] for pid in matched], dtype=float)

    if len(set(ranks.tolist())) < 2:
        for pid in unmatched_ids:
            full[pid] = float(pool[pid].rank)
        return full

    slope, intercept = np.polyfit(ranks, adps, 1)
    for pid in unmatched_ids:
        predicted = slope * pool[pid].rank + intercept
        # adp must stay strictly positive (it is used inside a log below).
        full[pid] = max(predicted, 1.0)
    return full


def pool_adp_lookup(
    pool: Mapping[str, PlayerDistribution],
    adp_table: pl.DataFrame,
    rankings: pl.DataFrame,
) -> tuple[dict[str, float], AdpMatchReport]:
    """Real ADP for every player in `pool`, matched via `models.opponent.
    build_pool_adp_lookup` and continuously extrapolated for anyone that
    doesn't match (see `_extrapolate_unmatched_adp`). `rankings` is the
    table the pool itself was built from (`rankings_2026` for a live 2026
    pool), used only to recover D/ST team abbreviations -- see
    `_pool_rows_for_adp_match`.

    Returns `(adp_by_player_id, report)` with an entry for *every* pool
    player (unlike `build_pool_adp_lookup`, which only returns matches) --
    `report` still names which ones were extrapolated, via
    `report.unmatched_ids`.
    """
    pool_rows = _pool_rows_for_adp_match(pool, rankings)
    matched, report = build_pool_adp_lookup(pool_rows, adp_table)
    full = _extrapolate_unmatched_adp(pool, matched, report.unmatched_ids)
    return full, report


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


# ---------------------------------------------------------------------------
# Greedy-policy myopia guard -- see module docstring, "Greedy value alone is
# not enough: the RB2/QB5 incident".

MIN_BENCH_PER_FLEX_POSITION = 1
"""Every FLEX-eligible position (RB/WR/TE) gets at least this many real
bench bodies beyond its own dedicated starter count, once our own starting
lineup is otherwise fillable. See module docstring."""

MAX_EXTRA_BENCH_NO_FLEX: Mapping[str, int] = {"QB": 2, "K": 1}
"""Non-FLEX-eligible, non-DST positions (QB, K) are capped at their own
dedicated starter count plus this many bench bodies -- i.e. QB tops out at
3 total (1 starter + 2 bench), K at 2 total (1 starter + 1 bench). Not a
new assumption -- these enforce what `draft.value`'s own proven decay curve
already says these slots are worth once the count is this deep (see
`tests/test_value.py::test_qb_bench_value_decays_with_depth`), removing
noise/near-ties that otherwise let a near-zero-value pick edge out a
genuinely useful FLEX-eligible bench player. K's multiple is tighter than
QB's because K has this league's *highest* fitted availability (93.4%,
`models/availability.py`) -- a backup kicker has essentially no bye-week or
injury-insurance case at all, unlike a backup QB's small-but-real one (see
`tests/test_value.py`'s QB2/QB3 values)."""


ELITE_STARTER_BENCH_BONUS = 1
"""Extra bench-floor slots (on top of `MIN_BENCH_PER_FLEX_POSITION`) for a
FLEX-eligible position once it holds at least one *elite* dedicated
starter -- a rostered player whose within-position rank is inside the
league's total dedicated-slot count for that position
(`starters[pos] * n_teams`, i.e. he would start on every team in a league
this size, not just a marginally-startable one). This is the piece that
makes the floor track *what was actually drafted*, not a fixed constant
identical for every structure -- see module docstring, "Why the floor must
depend on what was drafted, not just STARTERS"."""


def _elite_starter_bonus(
    position: str,
    roster_pairs: Sequence[tuple[str, str]],
    pool: Mapping[str, PlayerDistribution],
    n_teams: int,
) -> int:
    if position not in FLEX_ELIGIBLE:
        return 0
    starter_slots = STARTERS.get(position, 0) * n_teams
    for player_id, pos in roster_pairs:
        if pos != position:
            continue
        entry = pool.get(player_id)
        if entry is not None and entry.rank <= starter_slots:
            return ELITE_STARTER_BENCH_BONUS
    return 0


def _bench_floor(
    position: str,
    roster_pairs: Sequence[tuple[str, str]],
    pool: Mapping[str, PlayerDistribution],
    n_teams: int,
) -> int:
    return (
        STARTERS.get(position, 0)
        + MIN_BENCH_PER_FLEX_POSITION
        + _elite_starter_bonus(position, roster_pairs, pool, n_teams)
    )


def _positions_below_bench_floor(
    roster_counts: Mapping[str, int],
    roster_pairs: Sequence[tuple[str, str]],
    pool: Mapping[str, PlayerDistribution],
    n_teams: int,
) -> list[str]:
    return [
        pos
        for pos in sorted(FLEX_ELIGIBLE)
        if roster_counts.get(pos, 0) < _bench_floor(pos, roster_pairs, pool, n_teams)
    ]


def _positions_at_or_above_bench_ceiling(roster_counts: Mapping[str, int]) -> set[str]:
    # DST is deliberately excluded from the ceiling (there is nothing to
    # cap -- see module docstring) and FLEX-eligible positions are excluded
    # (they have no ceiling here at all; only the floor above applies to
    # them).
    return {
        pos
        for pos, extra in MAX_EXTRA_BENCH_NO_FLEX.items()
        if roster_counts.get(pos, 0) >= STARTERS.get(pos, 0) + extra
    }


def _choose_our_pick(
    available_ids: Sequence[str],
    pool: Mapping[str, PlayerDistribution],
    roster_pairs: Sequence[tuple[str, str]],
    replacement_means: Mapping[str, float],
    replacement_by_position: Mapping[str, WeeklyDistribution],
    n_teams: int = N_TEAMS,
) -> tuple[str, str]:
    """Greedily pick the highest-`value.py`-value available player for our
    own roster so far, after the bench floor/ceiling guard (module
    docstring) narrows the candidate set. See module docstring for why
    pairs at the turn are not jointly optimized."""
    roster = _our_roster_players(roster_pairs, pool, replacement_by_position)

    candidate_ids = list(available_ids)
    if len(roster_pairs) >= starting_slots_total():
        # Only once our own starting lineup is otherwise fillable -- early
        # picks are already governed correctly by `draft.value`'s own
        # (large, unambiguous) starter-slot marginal values.
        roster_counts: dict[str, int] = {}
        for _, position in roster_pairs:
            roster_counts[position] = roster_counts.get(position, 0) + 1

        below_floor = _positions_below_bench_floor(roster_counts, roster_pairs, pool, n_teams)
        if below_floor:
            floor_ids = [pid for pid in candidate_ids if pool[pid].position in below_floor]
            if floor_ids:
                candidate_ids = floor_ids
        else:
            at_ceiling = _positions_at_or_above_bench_ceiling(roster_counts)
            if at_ceiling:
                remaining_ids = [pid for pid in candidate_ids if pool[pid].position not in at_ceiling]
                if remaining_ids:
                    candidate_ids = remaining_ids

    values = value_available(candidate_ids, pool, roster, replacement_means)
    best = max(values, key=lambda v: v.value)
    return best.player_id, best.position


# ---------------------------------------------------------------------------
# The rollout itself


PickPolicy = Callable[
    [Sequence[str], Mapping[str, PlayerDistribution], Sequence[tuple[str, str]],
     Mapping[str, float], Mapping[str, WeeklyDistribution]],
    tuple[str, str],
]
"""A callable with the same signature as `_choose_our_pick` (positionally:
available_ids, pool, roster_pairs, replacement_means,
replacement_by_position) -> (player_id, position). See `run_rollout`'s
`pick_policy` parameter."""


def run_rollout(
    state: DraftState,
    pool: Mapping[str, PlayerDistribution],
    model: OpponentModel,
    replacement_by_position: Mapping[str, WeeklyDistribution],
    rng: np.random.Generator,
    our_team: int = DRAFT_SLOT,
    temperature: float | None = None,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
    adp_table: pl.DataFrame | None = None,
    rankings: pl.DataFrame | None = None,
    pick_policy: PickPolicy | None = None,
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

    `adp_table`/`rankings` supply real ADP for `sample_pick`'s `adp` --
    see the module docstring, "ADP for the opponent model". Both `None`
    (the default) skips real-ADP matching and uses `PlayerDistribution.rank`
    directly for every player -- fine for tests of the draft mechanics that
    have no real ADP data to inject. Supplying `adp_table` without
    `rankings` is a configuration error (the D/ST team-abbreviation lookup
    has nothing to read) and raises `ValueError` rather than silently
    matching skill players only. The match + fallback fit runs exactly once
    here, before the per-pick loop, not once per pick.

    Raises `ValueError` (via `sample_pick`/`value_available`) if the pool
    runs out of players before every roster is full -- a real configuration
    error (pool too small for `n_teams * rounds`), not a case to paper over.

    `pick_policy`, if given, replaces `_choose_our_pick` (this module's own
    greedy `draft.value` policy) for `our_team`'s picks only -- opponents
    are always sampled from `model` regardless. Signature: `(available_ids,
    pool, roster_pairs, replacement_means, replacement_by_position) ->
    (player_id, position)`, matching `_choose_our_pick`'s own positional
    arguments (minus `n_teams`, which a caller closing over a fixed
    `state.n_teams` normally does not need). This is what lets Stage 3's
    Task 6 backtest run the *same* opponent draws against four different
    "who occupies our slot" policies -- our own greedy engine, best-
    available-by-ADP, best-available-by-consensus-rank, and random-but-
    legal -- for a paired comparison, without duplicating this whole
    pick-by-pick loop four times.
    """
    replacement_means = {pos: float(dist.mean) for pos, dist in replacement_by_position.items()}

    if adp_table is not None:
        if rankings is None:
            raise ValueError(
                "rankings must be supplied whenever adp_table is given -- it is needed to "
                "resolve D/ST team abbreviations for the ADP join (see pool_adp_lookup)."
            )
        adp_lookup, _adp_report = pool_adp_lookup(pool, adp_table, rankings)
    else:
        adp_lookup = {pid: float(p.rank) for pid, p in pool.items()}

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
            if pick_policy is not None:
                player_id, position = pick_policy(
                    available_ids, pool, rosters[team], replacement_means, replacement_by_position
                )
            else:
                player_id, position = _choose_our_pick(
                    available_ids, pool, rosters[team], replacement_means, replacement_by_position,
                    n_teams=state.n_teams,
                )
        else:
            candidates = [
                AvailablePlayer(player_id=pid, position=pool[pid].position, adp=adp_lookup[pid])
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
                # 1-indexed round of the pick being sampled. `temperature`
                # is normally None, so this is what selects the fitted
                # temperature -- see models.opponent.TEMPERATURE_BY_ROUND.
                round_=(next_overall - 1) // state.n_teams + 1,
            )
            position = pool[player_id].position

        picks.append(Pick(overall_pick=next_overall, team=team, player_id=player_id, position=position))
        rosters[team].append((player_id, position))
        roster_counts[team][position] = roster_counts[team].get(position, 0) + 1
        drafted_ids.add(player_id)
        available_ids = [pid for pid in available_ids if pid != player_id]
        next_overall += 1

    return DraftState.from_picks(picks, n_teams=state.n_teams, rounds=state.rounds)
