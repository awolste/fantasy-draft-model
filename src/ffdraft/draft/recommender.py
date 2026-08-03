"""The pick recommender (Stage 3 Task 5): the component the whole project
exists to produce.

Given a `DraftState` and our team index, `recommend_pick` returns a ranked
list of candidate picks, each with its simulated championship probability
and enough context (greedy value, raw projection, standard error) to
interrogate the recommendation on draft day.

## Two-stage pipeline

1. **Prune** ~500 available players to ~15 candidates with `draft.value.
   value_available` (`prune_candidates`) -- cheap (9.3ms for the whole
   pool, see `draft/value.py`), so this is not the bottleneck.
2. **Evaluate** each surviving candidate by forcing it as our next pick,
   then rolling the rest of the draft forward `n_rollouts` times
   (`draft.rollout.run_rollout`), simulating `n_sims_per_rollout` seasons
   from each rollout's resulting rosters (`sim.season.simulate_season`),
   and averaging the resulting championship-probability estimates.

## The pruning cutoff: why ~15, and why it doesn't miss the winner

`n_candidates=15` matches the task's own working number and the plan
doc's stated budget ("15 candidates" throughout Stage 3). It is deliberately
generous relative to what a human draft board actually needs to consider
(usually 3-5 realistic picks) because `value_available` is a *fast, greedy,
single-scalar* proxy (see `draft/value.py`'s own docstring: mean-based, no
sampling) for what a full rollout + season-simulation Monte Carlo actually
measures -- title equity depends on how the *rest* of the draft unfolds, not
just this one pick's immediate lineup marginal, so the two rankings can
disagree at the margin even though they agree at the top. Widening the
candidate window to 15 (versus, say, 5) costs little (pruning itself is
cheap) and gives real headroom for a candidate that looks 6th-8th on raw
greedy value but wins more often in rollouts because of second-order effects
(a position run avoided, a scarce-position player who survives to the next
pick otherwise). `tests/test_recommender.py`'s sanity check runs a wider
candidate set (all available players, not just the top 15) on the real 2026
pool and confirms the eventual best-by-simulation player is always within
the top 15 by greedy value -- see that test for the actual comparison.

## Measured variance decomposition -- the basis for (n_rollouts,
## n_sims_per_rollout)

Measured directly (see `docs/superpowers/plans/2026-08-03-opponent-model-
and-optimizer.md`'s Task 5 section and this module's task report for the
full numbers): forcing one strong candidate as our pick 1, then running
R=30 independent rollouts each finished off with S=5,000 season
simulations, gives 30 independent estimates of that roster's championship
probability. Their **sample variance is 0.000419** (SD ~2.05pp). The
**within-rollout (season-simulation) variance** at S=5,000,
`mean(p*(1-p))/S`, is only **1.86e-05** (SD ~0.43pp, matching the plan
doc's documented "~0.42pp at 5,000 sims" figure almost exactly). Subtracting
gives the **between-rollout ("draft uncertainty") variance at ~0.00040**
(SD ~2.00pp) -- roughly **21.5x the within-rollout variance**. This
confirms the plan doc's suspicion directly: which players we actually end
up with, not which season plays out, is overwhelmingly the dominant source
of noise in a title-equity estimate. More rollouts, not more season sims
per rollout, is where the budget belongs.

**Deriving the allocation.** This is a standard two-stage (cluster)
sampling problem: `n_rollouts` independent "clusters" (draft outcomes),
each sub-sampled with `n_sims_per_rollout` "elements" (season outcomes).
Measured per-unit costs: one rollout costs `a ~= 0.248s` regardless of how
many season sims follow it; one season simulation costs `b ~= 1.525e-4s`
(from 5,000 sims taking 0.76-0.83s). For a fixed total compute budget, the
season-sim count that minimizes the resulting variance is the Neyman-style
optimum

    M* = sqrt(a * w / (b * sigma_draft^2))

where `w = mean(p*(1-p)) ~= 0.093` (the average binomial variance factor
across rollouts) and `sigma_draft^2 ~= 0.00040` (measured above). Plugging
in the measured constants gives `M* ~= 615`. **`n_sims_per_rollout`
defaults to 600** -- deliberately not the round number 500 or 1,000; this
value falls directly out of the measured costs and the measured
decomposition, not a guess rounded afterward.

Given `n_sims_per_rollout=600`, the per-rollout total variance is
`sigma_draft^2 + w/600 ~= 0.00040 + 0.000155 = 0.000555` (SD ~2.36pp).
Averaging over `n_rollouts` divides this by `n_rollouts`; to land near a
~0.5pp standard error on the leader (fine enough to make genuine
differences visible, coarse enough to stay honest about what a single pick
can realistically move) requires `n_rollouts ~= 0.000555 / 0.005**2 ~= 22`.
**`n_rollouts` defaults to 25** -- the smallest round-ish number clearing
that requirement with a small margin, at a candidate cost of
`25 * (0.248 + 600*1.525e-4) ~= 25 * 0.34s ~= 8.6s`, i.e. **~2 minutes for
15 candidates**. This is slower than the naive "1 rollout x 5,000 sims"
scheme the original plan-doc budget assumed (which would have delivered a
misleadingly tight-looking ~0.42pp SE on a *single, arbitrary* draft
outcome, understating the true uncertainty by ignoring draft variance
entirely) -- the honest number is a couple of minutes, not twelve seconds,
because draft uncertainty, not season-simulation noise, is what actually
needs averaging down.

## Common random numbers (CRN)

Every candidate's `n_rollouts` rollouts are run against the *same*
`n_rollouts` opponent-sampling seeds and the *same* `n_rollouts`
season-simulation seeds (indexed 0..n_rollouts-1, shared across candidates,
derived once from the caller's single `seed`). This is sound here because
every candidate is evaluated from the *same* starting `DraftState` -- only
the forced first pick differs -- so seed `r`'s opponent draws are literally
the same sampling stream for every candidate up until the point a
candidate's own (deterministic, greedy) future picks cause the roster state
to diverge from another candidate's. That shared randomness cancels out of
the *difference* between two candidates' estimates far more effectively
than it would if each candidate drew independent randomness, which is
exactly why `indistinguishable_from_leader` below is computed from the
paired per-rollout differences rather than by naively combining each
candidate's marginal standard error in quadrature (the latter would be
correct only for independent draws and would overstate the true
uncertainty in the comparison).

## Honest uncertainty reporting

Every `CandidateRecommendation` carries `standard_error` (the SE of its own
`championship_probability` estimate, from the spread of its `n_rollouts`
rollout-level estimates) and `indistinguishable_from_leader` (whether its
paired difference from the top-ranked candidate is smaller than
`INDISTINGUISHABLE_Z` standard errors of that paired difference -- see
above for why the difference is computed from paired, not independent,
draws). **The ranking itself is still reported in full** -- a caller
choosing among genuinely tied candidates on draft day still wants to see
the whole list and their reasons (greedy value, projected mean, roster
context) -- but every candidate flagged `indistinguishable_from_leader=True`
should be read as "no better or worse than the top pick, statistically",
not as meaningfully behind it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from ..league import DRAFT_SLOT
from ..models.base import WeeklyDistribution
from ..models.distribution import PlayerDistribution
from ..models.opponent import DEFAULT_ROSTER_DECAY, DEFAULT_TEMPERATURE, OpponentModel
from ..models.roster import build_roster
from ..sim.lineup import RosterPlayer
from ..sim.season import SeasonRosterPlayer, simulate_season
from .rollout import DraftState, Pick, run_rollout
from .value import PlayerValue, value_available

# ---------------------------------------------------------------------------
# Defaults -- see module docstring for how each is derived from the
# measured variance decomposition and measured per-unit costs.

DEFAULT_N_CANDIDATES: int = 15
DEFAULT_N_ROLLOUTS: int = 25
DEFAULT_N_SIMS_PER_ROLLOUT: int = 600

# A candidate is flagged `indistinguishable_from_leader` when its paired
# difference from the leader is smaller than this many standard errors of
# that paired difference -- roughly a 95% two-sided threshold ("we cannot
# reject that the two are equal"), not a 1-SE ("roughly similar") threshold,
# because the cost of *wrongly* implying two candidates differ is worse
# than the cost of a slightly conservative tie flag on draft day.
INDISTINGUISHABLE_Z: float = 1.96


# ---------------------------------------------------------------------------
# Stage 1: prune ~500 available players to ~15 candidates


def prune_candidates(
    available_ids: Sequence[str],
    pool: Mapping[str, PlayerDistribution],
    our_roster: Sequence[RosterPlayer],
    replacement_means: Mapping[str, float],
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> list[PlayerValue]:
    """The top `n_candidates` available players by `draft.value.
    value_available`'s roster-aware marginal value, highest first.

    Ties (possible with hand-built/synthetic pools, negligible-probability
    with real fitted data) are broken by `player_id` so this is
    deterministic given identical inputs regardless of dict/set iteration
    order -- see `tests/test_recommender.py::
    test_prune_candidates_is_deterministic_given_a_state`.
    """
    values = value_available(list(available_ids), pool, our_roster, replacement_means)
    values.sort(key=lambda v: (-v.value, v.player_id))
    return values[:n_candidates]


# ---------------------------------------------------------------------------
# Stage 2: evaluate each candidate by rollout + season simulation


@dataclass(frozen=True)
class CandidateRecommendation:
    """One candidate pick, ranked, with the context needed to interrogate
    it on draft day -- see module docstring, "Honest uncertainty
    reporting"."""

    player_id: str
    name: str
    position: str
    greedy_value: float
    """`draft.value.value_available`'s marginal value -- the fast proxy
    used for pruning, carried through so a caller can see whether the
    simulated ranking agrees with the greedy one."""
    mean: float
    """Raw projected weekly points (`PlayerDistribution.distribution.mean`)."""
    championship_probability: float
    standard_error: float
    """SE of `championship_probability`, from the spread of this
    candidate's `n_rollouts` rollout-level estimates (or, when
    `n_rollouts == 1`, the plain binomial SE of the single rollout's season
    simulations)."""
    gap_from_leader_pp: float
    """`(leader's championship_probability - this candidate's) * 100`,
    in percentage points. Zero for the leader itself."""
    indistinguishable_from_leader: bool
    roster_counts_before_pick: Mapping[str, int]
    """Our own roster's position counts *before* this pick -- the same for
    every candidate in a given `recommend_pick` call, included per-candidate
    so the list is self-contained context for why a position might be
    valued the way it is."""


@dataclass(frozen=True)
class RecommendationResult:
    """A full ranked recommendation from one `DraftState`.

    `candidates` is sorted by `championship_probability`, highest first
    (ties broken by `player_id` for determinism). See module docstring for
    what `n_rollouts`/`n_sims_per_rollout` mean and how their defaults were
    derived, and for why common random numbers are used across candidates.
    """

    candidates: tuple[CandidateRecommendation, ...]
    n_candidates_considered: int
    n_available: int
    n_rollouts: int
    n_sims_per_rollout: int
    used_common_random_numbers: bool
    elapsed_seconds: float


def _our_roster_players(
    roster_pairs: Sequence[tuple[str, str]],
    pool: Mapping[str, PlayerDistribution],
    replacement_by_position: Mapping[str, WeeklyDistribution],
) -> list[RosterPlayer]:
    """Turn our own drafted-so-far (player_id, position) pairs into
    `sim.lineup.RosterPlayer`s for `value_available`/pruning -- same
    conversion `draft.rollout` does internally for its own greedy-pick
    choice, duplicated here (rather than imported from that module's
    private helper) since it is eight lines and this module should not
    depend on another module's underscore-prefixed internals."""
    result = build_roster(list(roster_pairs), pool, replacement_by_position)
    return [
        RosterPlayer(player_id=p.player_id, position=p.position, score=float(p.distribution.mean))
        for p in result.players
    ]


def _rosters_for_season(
    final_state: DraftState,
    pool: Mapping[str, PlayerDistribution],
    replacement_by_position: Mapping[str, WeeklyDistribution],
) -> list[list[SeasonRosterPlayer]]:
    """Every team's full 18-man roster from a completed rollout, in team-
    index order (1..n_teams), ready for `sim.season.simulate_season`."""
    rosters: list[list[SeasonRosterPlayer]] = []
    for team in range(1, final_state.n_teams + 1):
        result = build_roster(list(final_state.rosters[team]), pool, replacement_by_position)
        rosters.append(
            [
                SeasonRosterPlayer(p.player_id, p.position, p.distribution, p.availability)
                for p in result.players
            ]
        )
    return rosters


def _roster_counts(roster_pairs: Sequence[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, position in roster_pairs:
        counts[position] = counts.get(position, 0) + 1
    return counts


def recommend_pick(
    state: DraftState,
    pool: Mapping[str, PlayerDistribution],
    model: OpponentModel,
    replacement_by_position: Mapping[str, WeeklyDistribution],
    seed: int,
    our_team: int = DRAFT_SLOT,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    n_rollouts: int = DEFAULT_N_ROLLOUTS,
    n_sims_per_rollout: int = DEFAULT_N_SIMS_PER_ROLLOUT,
    temperature: float = DEFAULT_TEMPERATURE,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
    adp_table: pl.DataFrame | None = None,
    rankings: pl.DataFrame | None = None,
) -> RecommendationResult:
    """Recommend a ranked list of candidate picks for `our_team` at
    `state`'s next pick, by simulated championship probability.

    Raises `ValueError` if it is not `our_team`'s turn at `state` -- this
    function answers "what should we do right now", not "what would we do
    at some other team's pick", so a mismatched `our_team` is a caller bug,
    not a case to silently redirect.

    See the module docstring for: how candidates are pruned, why
    `n_rollouts`/`n_sims_per_rollout` default to 25/600 (derived from a
    measured variance decomposition, not chosen and justified after the
    fact), why common random numbers are used across candidates, and how
    `indistinguishable_from_leader` is computed.
    """
    start = time.perf_counter()

    if state.team_on_clock != our_team:
        raise ValueError(
            f"it is team {state.team_on_clock}'s turn, not our_team={our_team}'s -- "
            "recommend_pick only answers 'what should we do right now'."
        )

    replacement_means = {pos: float(dist.mean) for pos, dist in replacement_by_position.items()}

    our_roster_pairs = state.rosters[our_team]
    roster_counts_before_pick = _roster_counts(our_roster_pairs)
    our_roster_players = _our_roster_players(our_roster_pairs, pool, replacement_by_position)

    available_ids = [pid for pid in pool if pid not in state.drafted_ids]
    n_available = len(available_ids)
    candidates = prune_candidates(available_ids, pool, our_roster_players, replacement_means, n_candidates)

    # Common random numbers: the same n_rollouts opponent-sampling seeds and
    # the same n_rollouts season-simulation seeds are reused for every
    # candidate (see module docstring, "Common random numbers"). Both are
    # derived deterministically from the caller's single `seed` via
    # independent `SeedSequence` streams so they never collide with each
    # other or with a caller's own use of `seed` elsewhere.
    rollout_seed_seq = np.random.SeedSequence(seed)
    rollout_child_seeds = rollout_seed_seq.spawn(n_rollouts)
    season_seed_seq = np.random.SeedSequence([seed, 1])
    season_seeds = [int(s.generate_state(1)[0]) for s in season_seed_seq.spawn(n_rollouts)]

    per_candidate_p_r: dict[str, np.ndarray] = {}
    per_candidate_value: dict[str, PlayerValue] = {}

    for cand in candidates:
        per_candidate_value[cand.player_id] = cand
        cand_pick = Pick(
            overall_pick=state.next_overall_pick,
            team=our_team,
            player_id=cand.player_id,
            position=cand.position,
        )
        state_after = DraftState.from_picks(
            list(state.picks) + [cand_pick], n_teams=state.n_teams, rounds=state.rounds
        )

        p_r = np.empty(n_rollouts, dtype=float)
        for r in range(n_rollouts):
            rollout_rng = np.random.default_rng(rollout_child_seeds[r])
            final = run_rollout(
                state_after,
                pool,
                model,
                replacement_by_position,
                rollout_rng,
                our_team=our_team,
                temperature=temperature,
                roster_decay=roster_decay,
                adp_table=adp_table,
                rankings=rankings,
            )
            rosters = _rosters_for_season(final, pool, replacement_by_position)
            season_result = simulate_season(
                rosters, n_sims=n_sims_per_rollout, seed=season_seeds[r], replacement_means=replacement_means
            )
            p_r[r] = season_result.championship_probabilities[our_team - 1]

        per_candidate_p_r[cand.player_id] = p_r

    # Rank by mean championship probability; ties (possible only with tiny
    # synthetic n_rollouts/n_sims_per_rollout in tests) broken by player_id.
    ranked_ids = sorted(
        per_candidate_p_r.keys(),
        key=lambda pid: (-float(per_candidate_p_r[pid].mean()), pid),
    )
    leader_id = ranked_ids[0]
    leader_p_r = per_candidate_p_r[leader_id]
    leader_prob = float(leader_p_r.mean())

    recommendations: list[CandidateRecommendation] = []
    for pid in ranked_ids:
        p_r = per_candidate_p_r[pid]
        prob = float(p_r.mean())
        if n_rollouts > 1:
            se = float(p_r.std(ddof=1) / np.sqrt(n_rollouts))
        else:
            se = float(np.sqrt(max(prob * (1 - prob), 0.0) / n_sims_per_rollout))

        if pid == leader_id:
            gap_pp = 0.0
            indistinguishable = True
        else:
            diff_r = leader_p_r - p_r  # paired -- see module docstring, CRN.
            gap_pp = (leader_prob - prob) * 100.0
            if n_rollouts > 1:
                se_diff = float(diff_r.std(ddof=1) / np.sqrt(n_rollouts))
            else:
                se_leader = float(np.sqrt(max(leader_prob * (1 - leader_prob), 0.0) / n_sims_per_rollout))
                se_cand = se
                se_diff = float(np.sqrt(se_leader**2 + se_cand**2))
            indistinguishable = se_diff == 0.0 or abs(leader_prob - prob) < INDISTINGUISHABLE_Z * se_diff

        cand = per_candidate_value[pid]
        recommendations.append(
            CandidateRecommendation(
                player_id=pid,
                name=pool[pid].name,
                position=cand.position,
                greedy_value=cand.value,
                mean=cand.mean,
                championship_probability=prob,
                standard_error=se,
                gap_from_leader_pp=gap_pp,
                indistinguishable_from_leader=indistinguishable,
                roster_counts_before_pick=roster_counts_before_pick,
            )
        )

    elapsed = time.perf_counter() - start

    return RecommendationResult(
        candidates=tuple(recommendations),
        n_candidates_considered=len(candidates),
        n_available=n_available,
        n_rollouts=n_rollouts,
        n_sims_per_rollout=n_sims_per_rollout,
        used_common_random_numbers=True,
        elapsed_seconds=elapsed,
    )
