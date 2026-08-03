"""The opponent draft model: P(manager m drafts player p at pick n | who is
still available).

This is the piece that makes the optimizer specific to *this* league rather
than generic best-player-available advice. It is fit on seven seasons
(2018-2024) of this league's real snake drafts, joined against that
season's consensus ADP so a manager's behaviour can be measured *relative
to* what was "expected" at the time, not relative to hindsight. (The
league has an eighth season of draft history, 2025, but it cannot be used
for fitting -- see below.)

## Why 2025 is excluded from fitting

`adp_history` covers 2018-2024 only -- there is no 2025 consensus ADP from
any source in this project. Since every quantity here (`reach`, positional
bias) is defined as a deviation from that season's ADP, a 2025 pick simply
has nothing to be measured against and cannot contribute to the fit.
`TRAINING_SEASONS` is therefore 2018-2024; 2025's 180 picks are silently
excluded by that season filter, not dropped by a join failure (see
`build_training_set`).

## The identity problem this module has to solve before it can fit anything

`league_drafts.espn_player_id` is ESPN's own player id. Every other model in
this project resolves ESPN identity by joining `id_crosswalk` on `espn_id`
(see `models/distribution.py`) -- but `id_crosswalk` is sourced from
nflverse's *player* id table, which has zero rows for team defenses. A D/ST
pick's `espn_player_id` therefore cannot go through that path at all: it is
always negative, and ESPN's fantasy API encodes it as
`espn_player_id = -16000 - pro_team_id`, where `pro_team_id` is ESPN's own
(non-alphabetical, historically ordered) team numbering -- confirmed against
every negative id actually present in this league's `league_drafts`
(`-16021` -> pro_team_id 21 -> PHI, `-16033` -> 33 -> BAL, etc., both of
which are plausible early-DST picks). `PRO_TEAM_ABBREV` below is that
lookup table, built from the standard ESPN fantasy numbering and verified to
map every negative id in this league's real draft history to a team
abbreviation `adp_history` actually uses (after aliasing "WSH" to "WAS",
the one spelling mismatch between the two).

`id_crosswalk` also labels kickers "PK" where every other source in this
project (including `adp_history`) uses "K" -- see `models/distribution.py`'s
identical note. `_POSITION_ALIASES` fixes that up for this join too, so
kickers are not silently excluded from training the way `distribution.py`
deliberately excludes them from crosswalk-based ID matching (kickers still
get *drafted* in this league, so the opponent model needs them).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from .. import store
from ..ids import normalize_name

_EMPTY_ROSTER_COUNTS: Mapping[str, int] = MappingProxyType({})

# 2018-2024. See module docstring for why 2025 is excluded.
TRAINING_SEASONS: tuple[int, ...] = tuple(range(2018, 2025))

# ESPN's proTeamId -> team abbreviation (as `adp_history` spells it). Used
# only to identify D/ST picks -- see module docstring.
PRO_TEAM_ABBREV: Mapping[int, str] = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA",
    27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

# `id_crosswalk` calls kickers "PK"; every other source here (including
# `adp_history`) uses "K".
_POSITION_ALIASES: Mapping[str, str] = {"PK": "K"}

VALID_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE", "K", "DST"})


# ---------------------------------------------------------------------------
# Step 1: the training set


@dataclass(frozen=True)
class JoinReport:
    """How much of `league_drafts` 2018-2024 survived the join to `adp_history`.

    Reported rather than silently dropped, per this project's stated failure
    mode (plausible-looking numbers from a silently-lossy join). Broken down
    by position and by manager so a caller can check the loss is not
    concentrated in a way that would bias the fit -- e.g. if D/ST picks
    failed to join at a much higher rate than skill positions, the model
    would systematically under-see D/ST behaviour and Step 3's validation
    would be untrustworthy even if it happened to pass.

    On real data, the loss *is* concentrated exactly this way, and in a
    direction worth naming explicitly: D/ST (~19%), K (~16%), and TE
    (~15%) fail to join at 2-6x the rate of QB/RB (~3%) -- because
    `adp_history` is a top-N consensus export that simply does not go deep
    enough to cover every deep-bench D/ST, K, or backup TE this league's
    8-man benches allow managers to stash. Checked directly: the unmatched
    rows for these three positions cluster in the latest rounds of the
    draft (round 13+ for D/ST and K, spread from round 10 for TE), later
    than the average *usable* pick at that position. This means the
    surviving (usable) sample for these three positions systematically
    excludes their most extreme late picks -- a one-sided censoring, not a
    random gap -- which biases `pos_effect` for D/ST, K, and TE upward
    (toward "earlier than ADP") relative to what the true, uncensored
    positional bias would be. It does not change the *sign* of the
    already-expected D/ST result, but it means the K and TE positive
    effects (see `OpponentModel`) should be read with this caveat rather
    than taken as clean evidence of early-K/TE-drafting behaviour. QB/RB/WR
    have no comparable issue -- their unusable rate is both much lower and
    not detectably concentrated at the tail.
    """

    total_picks: int
    usable_picks: int
    unusable_picks: int
    unusable_rate: float
    picks_by_position: Mapping[str, int]
    unusable_by_position: Mapping[str, int]
    unusable_rate_by_position: Mapping[str, float]
    usable_by_manager: Mapping[str, int]


def _resolve_picks(
    league_drafts: pl.DataFrame,
    league_managers: pl.DataFrame,
    crosswalk: pl.DataFrame,
    adp_history: pl.DataFrame,
) -> pl.DataFrame:
    """Join `league_drafts` -> `league_managers` -> (crosswalk|team map) ->
    `adp_history`, with no season filter. One row per input pick:
    `season`, `overall_pick`, `round`, `manager_id`, `position`, `player_key`
    (a stable identity string -- normalized name for skill/K positions,
    team abbreviation for D/ST -- always populated since it only depends on
    `league_drafts`/`crosswalk`, never on the ADP join succeeding), and
    `adp` (nullable: null exactly when this pick could not be matched to
    that season's ADP).

    Shared by `build_training_set` (which restricts to `TRAINING_SEASONS`
    and drops the unusable rows after reporting them) and `backtest_
    holdout_season` (which needs every pick, including unusable ones, to
    correctly track roster composition and available-player removal even
    when a specific pick can't be scored).
    """
    drafts = league_drafts.join(
        league_managers.select(["season", "team_id", "manager_id"]),
        on=["season", "team_id"],
        how="left",
    )
    if drafts.filter(pl.col("manager_id").is_null()).height:
        missing = drafts.filter(pl.col("manager_id").is_null()).select(["season", "team_id"]).unique()
        raise ValueError(
            f"{missing.height} (season, team_id) pairs in league_drafts have no "
            f"matching row in league_managers -- this should be impossible for a "
            f"consistent ingest; investigate rather than silently dropping these "
            f"picks. First few: {missing.head(5).to_dicts()}"
        )

    is_dst = pl.col("espn_player_id") < 0

    dst = drafts.filter(is_dst).with_columns(
        (-16000 - pl.col("espn_player_id"))
        .replace_strict(dict(PRO_TEAM_ABBREV), default=None)
        .alias("team"),
        pl.lit("DST").alias("position"),
    )
    dst = dst.with_columns(pl.col("team").alias("player_key"))
    dst = dst.join(
        adp_history.select(["team", "position", "season", "adp"]),
        on=["team", "position", "season"],
        how="left",
    )

    non_dst = drafts.filter(~is_dst).join(
        crosswalk.select(["espn_id", "name", "position"]),
        left_on="espn_player_id",
        right_on="espn_id",
        how="left",
    )
    non_dst = non_dst.with_columns(
        pl.col("position").replace(_POSITION_ALIASES),
        pl.col("name").map_elements(
            lambda n: normalize_name(n) if n is not None else None,
            return_dtype=pl.String,
        ).alias("_key"),
    )
    non_dst = non_dst.with_columns(pl.col("_key").alias("player_key"))
    adp_keyed = adp_history.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("_key")
    )
    non_dst = non_dst.join(
        adp_keyed.select(["_key", "position", "season", "adp"]),
        on=["_key", "position", "season"],
        how="left",
    )

    keep_cols = ["season", "overall_pick", "round", "manager_id", "position", "player_key", "adp"]
    return pl.concat([dst.select(keep_cols), non_dst.select(keep_cols)])


def build_training_set(
    league_drafts: pl.DataFrame | None = None,
    league_managers: pl.DataFrame | None = None,
    crosswalk: pl.DataFrame | None = None,
    adp_history: pl.DataFrame | None = None,
    seasons: Sequence[int] = TRAINING_SEASONS,
) -> tuple[pl.DataFrame, JoinReport]:
    """Join `league_drafts` -> `league_managers` -> (crosswalk|team map) ->
    `adp_history`, restricted to `seasons` (default `TRAINING_SEASONS`,
    2018-2024).

    Returns `(training, report)`. `training` has one row per *usable* pick
    (i.e. one that found an ADP for that player/team in that season):
    `season`, `overall_pick`, `round`, `manager_id`, `position`, `adp`,
    `reach` (`log(adp) - log(overall_pick)` -- see `OpponentModel`'s
    docstring). Unusable rows are not included in `training` at all -- they
    contribute nothing to the fit -- but are counted in `report` so the
    loss is visible.

    `seasons` lets `backtest_holdout_season` fit on everything *except* the
    held-out season without duplicating this join.

    All four data inputs default to this project's persisted datasets so
    this can be called with no arguments in normal use; tests inject small
    synthetic frames instead.
    """
    if league_drafts is None:
        league_drafts = store.read("league_drafts")
    if league_managers is None:
        league_managers = store.read("league_managers")
    if crosswalk is None:
        crosswalk = store.read("id_crosswalk")
    if adp_history is None:
        adp_history = store.read("adp_history")

    drafts = league_drafts.filter(pl.col("season").is_in(seasons))
    total_picks = drafts.height

    combined = _resolve_picks(drafts, league_managers, crosswalk, adp_history)

    unusable = combined.filter(pl.col("adp").is_null())
    usable = combined.filter(pl.col("adp").is_not_null())

    picks_by_position = dict(
        combined.group_by("position").len().sort("position").iter_rows()
    )
    unusable_by_position = dict(
        unusable.group_by("position").len().sort("position").iter_rows()
    )
    unusable_rate_by_position = {
        pos: unusable_by_position.get(pos, 0) / n for pos, n in picks_by_position.items()
    }
    usable_by_manager = dict(
        usable.group_by("manager_id").len().sort("manager_id").iter_rows()
    )

    report = JoinReport(
        total_picks=total_picks,
        usable_picks=usable.height,
        unusable_picks=unusable.height,
        unusable_rate=unusable.height / total_picks if total_picks else 0.0,
        picks_by_position=picks_by_position,
        unusable_by_position=unusable_by_position,
        unusable_rate_by_position=unusable_rate_by_position,
        usable_by_manager=usable_by_manager,
    )

    training = usable.with_columns(
        (pl.col("adp").log() - pl.col("overall_pick").log()).alias("reach")
    )

    bad_positions = set(training["position"].unique().to_list()) - VALID_POSITIONS
    if bad_positions:
        raise ValueError(
            f"training set has usable rows with unrecognized position(s) "
            f"{bad_positions} -- adp_history should only ever have "
            f"{sorted(VALID_POSITIONS)}; investigate the join rather than "
            f"silently including them."
        )

    return training, report


# ---------------------------------------------------------------------------
# Step 2: reach tendency with shrinkage


def _one_way_shrinkage(
    values: np.ndarray, groups: np.ndarray
) -> tuple[dict, dict, dict, float, float]:
    """Empirical-Bayes (Efron-Morris / James-Stein style) shrinkage of
    per-group means toward the grand mean (here, always 0 -- callers pass
    already-demeaned residuals), for an unbalanced one-way random-effects
    model.

    Returns `(shrunk_mean_by_group, weight_by_group, n_by_group, sigma2, tau2)`.

    `sigma2` is the pooled within-group variance (residual noise common to
    every group). `tau2` is the between-group variance -- how much groups
    genuinely differ from each other, over and above sampling noise -- and
    is the classic method-of-moments (unbiased one-way ANOVA) estimator:

        MSW = pooled within-group sum of squares / (N - G)
        MSB = between-group sum of squares / (G - 1)
        n0  = (N - sum(n_g^2)/N) / (G - 1)
        tau2 = max(0, (MSB - MSW) / n0)

    `n0` (not a simple average of `n_g`) is the standard correction for
    unbalanced groups (Searle, Casella & McCulloch, *Variance Components*,
    ch.3) -- this project's managers are wildly unbalanced (2 to 8 seasons
    of usable history), so using a naive average `n_g` here would bias
    `tau2`.

    Each group's shrinkage weight is `w_g = tau2 / (tau2 + sigma2/n_g)` --
    the fraction of that group's own sample mean trusted, versus the
    remaining `1 - w_g` pulled to 0 (the grand mean of the residuals passed
    in). This is monotonically increasing in `n_g`: a group with few
    observations gets a small `w_g` and comes out close to 0; a group with
    many gets `w_g` close to 1 and comes out close to its own raw mean --
    exactly the behaviour required ("a 2-season manager must come out near
    league-average; an 8-season manager mostly themselves").

    If `tau2` estimates to 0 (groups are statistically indistinguishable
    from each other given the noise), every weight is 0 and every group is
    shrunk all the way to the grand mean -- the honest conclusion when
    there is no detectable group-level signal, not a bug to work around.
    """
    unique_groups = sorted(set(groups.tolist()))
    n_by_group: dict = {}
    mean_by_group: dict = {}
    for g in unique_groups:
        mask = groups == g
        n_by_group[g] = int(mask.sum())
        mean_by_group[g] = float(values[mask].mean())

    n_total = len(values)
    n_groups = len(unique_groups)

    ssw = 0.0
    for g in unique_groups:
        mask = groups == g
        ssw += float(((values[mask] - mean_by_group[g]) ** 2).sum())
    dof_w = n_total - n_groups
    sigma2 = ssw / dof_w if dof_w > 0 else float(values.var())

    grand_mean = float(values.mean())
    ssb = sum(
        n_by_group[g] * (mean_by_group[g] - grand_mean) ** 2 for g in unique_groups
    )
    msb = ssb / (n_groups - 1) if n_groups > 1 else 0.0

    if n_groups > 1 and n_total > 0:
        sum_n_sq = sum(n**2 for n in n_by_group.values())
        n0 = (n_total - sum_n_sq / n_total) / (n_groups - 1)
    else:
        n0 = 1.0
    tau2 = max(0.0, (msb - sigma2) / n0) if n0 > 0 else 0.0

    weight_by_group: dict = {}
    shrunk_by_group: dict = {}
    for g in unique_groups:
        n_g = n_by_group[g]
        denom = tau2 + sigma2 / n_g
        w_g = tau2 / denom if denom > 0 else 0.0
        weight_by_group[g] = w_g
        shrunk_by_group[g] = w_g * mean_by_group[g]

    return shrunk_by_group, weight_by_group, n_by_group, sigma2, tau2


@dataclass(frozen=True)
class OpponentModel:
    """A fitted, additive, hierarchical model of draft reach.

    `reach` (see `build_training_set`) for pick `i` by manager `m` at
    position `pos` is modeled as:

        reach_i = league_mu
                + pos_effect[pos]
                + manager_effect[m]
                + manager_pos_effect[(m, pos)]
                + noise

    `pos_effect` is the league-wide positional bias -- fit directly from
    all ~1,100 usable observations with no shrinkage, since it is only 6
    categories and every one of them (even the thinnest, D/ST) has several
    dozen observations. This is the term Step 3 validates against the known
    D/ST-early tendency.

    `manager_effect` is each manager's overall reach tendency (after
    removing the league-wide positional pattern), shrunk toward 0 (the
    league prior) in proportion to that manager's observation count via
    `_one_way_shrinkage`.

    `manager_pos_effect` is a manager's *own* positional bias beyond both
    the league-wide positional pattern and their own overall tendency --
    e.g. "manager X reaches early in general, AND especially for RBs". Cell
    counts here are thin (a manager's-worth of picks split six ways across
    up to 7 seasons), so this term is shrunk independently and, honestly,
    is expected to shrink close to 0 for most cells -- see the report's
    "per-manager modelling" discussion.
    """

    league_mu: float
    pos_effect: Mapping[str, float]
    manager_effect: Mapping[str, float]
    manager_pos_effect: Mapping[tuple[str, str], float]
    manager_weight: Mapping[str, float]
    manager_pos_weight: Mapping[tuple[str, str], float]
    manager_n: Mapping[str, int]
    sigma2: float
    tau2_manager: float
    tau2_manager_pos: float

    def predicted_reach(self, manager_id: str, position: str) -> float:
        """The model's point estimate of this manager's log-scale reach at
        this position -- `league_mu + pos_effect[position] + manager_effect
        [manager_id] + manager_pos_effect[(manager_id, position)]`, with 0
        substituted for any term the model has never seen (an unknown
        manager or position falls back to the league-wide position effect,
        or all the way to `league_mu` if the position is also unseen)."""
        return (
            self.league_mu
            + self.pos_effect.get(position, 0.0)
            + self.manager_effect.get(manager_id, 0.0)
            + self.manager_pos_effect.get((manager_id, position), 0.0)
        )


def fit_opponent_model(training: pl.DataFrame) -> OpponentModel:
    """Fit `OpponentModel` from `build_training_set`'s output.

    See `OpponentModel`'s docstring for the model form and `_one_way_
    shrinkage` for the shrinkage estimator.
    """
    reach = training["reach"].to_numpy()
    positions = training["position"].to_numpy()
    managers = training["manager_id"].to_numpy()

    league_mu = float(reach.mean())
    resid0 = reach - league_mu

    pos_effect: dict = {}
    for pos in sorted(set(positions.tolist())):
        pos_effect[pos] = float(resid0[positions == pos].mean())

    resid1 = resid0 - np.array([pos_effect[p] for p in positions])

    manager_effect, manager_weight, manager_n, sigma2, tau2_m = _one_way_shrinkage(
        resid1, managers
    )

    resid2 = resid1 - np.array([manager_effect[m] for m in managers])

    manager_pos_groups = np.array(
        [f"{m}\x1f{p}" for m, p in zip(managers.tolist(), positions.tolist())]
    )
    mp_shrunk, mp_weight, mp_n, _, tau2_mp = _one_way_shrinkage(resid2, manager_pos_groups)

    def _split(key: str) -> tuple[str, str]:
        m, p = key.split("\x1f")
        return m, p

    manager_pos_effect = {_split(k): v for k, v in mp_shrunk.items()}
    manager_pos_weight = {_split(k): v for k, v in mp_weight.items()}

    return OpponentModel(
        league_mu=league_mu,
        pos_effect=pos_effect,
        manager_effect=manager_effect,
        manager_pos_effect=manager_pos_effect,
        manager_weight=manager_weight,
        manager_pos_weight=manager_pos_weight,
        manager_n=manager_n,
        sigma2=sigma2,
        tau2_manager=tau2_m,
        tau2_manager_pos=tau2_mp,
    )


# ---------------------------------------------------------------------------
# Step 4: the sampling distribution


@dataclass(frozen=True)
class AvailablePlayer:
    """One still-available player, as the sampler needs to see it."""

    player_id: str
    position: str
    adp: float


# Softmax temperature in *rank* space (not raw score space) -- chosen so
# that reach/positional effects are compared in the same round-comparable
# units `reach` itself uses (see module docstring), then converted once to
# a plain, interpretable "how many rank-positions apart" scale. A
# temperature of 3.0 means two players three ranks apart on the manager's
# adjusted board are about e:1 (~2.7x) more/less likely -- concentrated
# enough that the top few adjusted-ranked players dominate (matching how
# real drafts are heavily front-loaded on consensus value) while still
# leaving room for the model to be wrong. Chosen by a small grid search
# against the 2024 holdout backtest (see `models/opponent.py`'s report /
# `scripts/backtest_opponent.py`); not fit jointly with the reach model
# itself, to keep the two fits independent and inspectable.
DEFAULT_TEMPERATURE: float = 3.0

# Each already-rostered player at a position multiplies that position's
# sampling weight by this factor for the *next* pick at that position --
# i.e. a manager with 3 QBs already rostered weighs a 4th QB at roughly
# `ROSTER_DECAY**3` of an otherwise-identical player at a position they
# have none of. Deliberately uniform across positions rather than fit with
# a per-position soft cap: real roster-construction counts vary a lot by
# position already (this league's teams draft a mean of ~6.2 WR and ~5.3 RB
# but only ~1.9 QB and ~1.05 D/ST per season -- see the Step 4 report), and
# fitting six more free parameters from ~1,100 thinly-sliced observations
# risks fitting noise dressed up as position-specific need. A single global
# decay is the "at least crude" version the task explicitly permits: it
# reliably makes an already-deep position much less likely without
# asserting a precise per-position ceiling the data cannot actually
# support.
DEFAULT_ROSTER_DECAY: float = 0.55

# Never exactly zero: a hard zero for the manager's `n`th QB would make the
# model refuse to ever finish a roster in a rollout that has run out of
# every other position, which is a modeling artifact, not a real
# constraint -- managers occasionally do draft a 4th QB. This is the floor
# `roster_multiplier` cannot go below regardless of how deep a position
# already is.
_MIN_ROSTER_MULTIPLIER: float = 1e-6


def pick_probabilities(
    model: OpponentModel,
    manager_id: str,
    available: Sequence[AvailablePlayer],
    roster_counts: Mapping[str, int] = _EMPTY_ROSTER_COUNTS,
    temperature: float = DEFAULT_TEMPERATURE,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
) -> dict[str, float]:
    """`P(manager_id drafts p)` for each `p` in `available`, given the
    manager's current `roster_counts` (`{position: count}`).

    Every player in `available` gets a nonzero probability (see
    `_MIN_ROSTER_MULTIPLIER`); probabilities are restricted to exactly the
    players passed in (nothing outside `available` can be returned) and sum
    to 1.

    Form: each player's `adp` is adjusted by the manager's fitted reach at
    that position (`OpponentModel.predicted_reach`) to get a predicted
    log-scale pick position -- "when this manager would typically take a
    player at this ADP". Available players are ranked by that adjusted
    value (rank 1 = the manager's most-wanted player still on the board)
    and a softmax over `-rank/temperature` converts ranks to probabilities.
    Rank space (not raw adjusted-value space) is used for the softmax so
    `temperature` means the same thing regardless of what round of the
    draft this is -- see `DEFAULT_TEMPERATURE`.

    That rank-based weight is then multiplied by a per-position roster-need
    penalty, `roster_decay ** roster_counts.get(position, 0)` (floored at
    `_MIN_ROSTER_MULTIPLIER`), before renormalizing -- see `DEFAULT_
    ROSTER_DECAY` for why this is deliberately position-uniform rather than
    fit per position.

    Pure function of its inputs: no RNG, no global state. `sample_pick`
    below turns this into an actual draw.
    """
    if not available:
        return {}

    n = len(available)
    adp = np.empty(n)
    positions: list[str] = []
    player_ids: list[str] = []
    for i, p in enumerate(available):
        adp[i] = p.adp
        positions.append(p.position)
        player_ids.append(p.player_id)

    reach_adjust = np.array(
        [model.predicted_reach(manager_id, pos) for pos in positions]
    )
    adjusted_log_pick = np.log(adp) - reach_adjust

    # rank 0 = smallest adjusted_log_pick = the manager's top choice
    order = np.argsort(adjusted_log_pick, kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(n)

    base_weight = np.exp(-rank / temperature)

    roster_multiplier = np.array(
        [
            max(roster_decay ** roster_counts.get(pos, 0), _MIN_ROSTER_MULTIPLIER)
            for pos in positions
        ]
    )

    weight = base_weight * roster_multiplier
    total = weight.sum()
    if total <= 0:
        # Every weight underflowed to 0 (pathological input); fall back to
        # uniform over `available` rather than dividing by zero or
        # returning an all-zero, non-summing-to-1 distribution.
        probs = np.full(n, 1.0 / n)
    else:
        probs = weight / total

    return dict(zip(player_ids, probs.tolist()))


def sample_pick(
    model: OpponentModel,
    manager_id: str,
    available: Sequence[AvailablePlayer],
    roster_counts: Mapping[str, int],
    rng: np.random.Generator,
    temperature: float = DEFAULT_TEMPERATURE,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
) -> str:
    """Draw one `player_id` from `pick_probabilities`'s distribution using
    `rng` (an explicit `np.random.Generator` -- never global numpy state,
    so draws are reproducible from a seed and safe to call from parallel
    rollouts). Raises `ValueError` if `available` is empty."""
    if not available:
        raise ValueError("sample_pick called with no available players")

    probs_by_id = pick_probabilities(
        model, manager_id, available, roster_counts, temperature, roster_decay
    )
    player_ids = list(probs_by_id.keys())
    probs = np.array([probs_by_id[pid] for pid in player_ids])
    # Guard against floating-point drift so np.random.Generator.choice's
    # internal sum-to-1 check never spuriously fails.
    probs = probs / probs.sum()
    choice_idx = rng.choice(len(player_ids), p=probs)
    return player_ids[choice_idx]


# ---------------------------------------------------------------------------
# Step 5: the backtest -- does this add anything over plain ADP?

# A pick's round bucketed into thirds of this league's *deepest* draft
# (18 rounds, 2023-2025) -- "early/middle/late" per the task spec, so the
# 2024 holdout (an 18-round draft) splits evenly. `round` itself (not
# `overall_pick`) is the bucketing key so this is stable across the 17- vs
# 18-round eras rather than needing a season-specific overall-pick cutoff.
_ROUND_BUCKETS: tuple[tuple[str, range], ...] = (
    ("early", range(1, 4)),
    ("middle", range(4, 11)),
    ("late", range(11, 19)),
)


def _round_bucket(round_: int) -> str:
    for name, r in _ROUND_BUCKETS:
        if round_ in r:
            return name
    return "late"


@dataclass(frozen=True)
class RoundBucketAccuracy:
    n: int
    top1_accuracy: float
    top5_accuracy: float
    baseline_top1_accuracy: float
    baseline_top5_accuracy: float


@dataclass(frozen=True)
class BacktestResult:
    """Top-1/top-5 accuracy of the fitted model's `pick_probabilities`
    against what a manager actually drafted in a held-out season, versus
    the "take the highest-ranked available player by ADP" baseline.

    `n_scored` is the number of the held-out season's picks that could be
    scored at all (the actual pick had a resolvable ADP, i.e. was itself a
    member of the candidate pool); `n_excluded_unmatched` is how many could
    not be (same join-failure phenomenon `JoinReport` measures, applied to
    the held-out season alone) -- these can never be predicted correctly by
    either the model or the baseline, since they are not offered as
    candidates, so they are excluded from both accuracy figures rather than
    silently counted as misses (which would penalize both approaches
    identically and just add noise) or silently counted as hits (which
    would inflate both).
    """

    n_scored: int
    n_excluded_unmatched: int
    top1_accuracy: float
    top5_accuracy: float
    baseline_top1_accuracy: float
    baseline_top5_accuracy: float
    by_round_bucket: Mapping[str, RoundBucketAccuracy]


def backtest_holdout_season(
    holdout_season: int,
    league_drafts: pl.DataFrame | None = None,
    league_managers: pl.DataFrame | None = None,
    crosswalk: pl.DataFrame | None = None,
    adp_history: pl.DataFrame | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
) -> BacktestResult:
    """Fit `OpponentModel` on every `TRAINING_SEASONS` season *except*
    `holdout_season`, then replay `holdout_season`'s real draft pick by
    pick: at each pick, ask the model for its top-1/top-5 most likely
    players among who is actually still available (tracked by replaying
    the real picks in order, not re-simulated), and check against what was
    actually taken.

    The "available" candidate pool for a pick is every `holdout_season`
    `adp_history` entry not yet picked earlier in the real draft -- i.e.
    the model and the ADP baseline are given the exact same candidate list
    at every pick, so neither has an information advantage the other
    lacks. `roster_counts` (for the model's roster-need term) is updated
    from the real picks regardless of whether a given pick was itself
    ADP-matched, since roster composition is observable from `league_
    drafts`/the crosswalk alone (see `_resolve_picks`), independent of the
    ADP join succeeding.

    Baseline: "take the highest-ranked available player by ADP" -- top-1 is
    the single lowest-ADP player still available; top-5 is the five
    lowest-ADP players still available.
    """
    if league_drafts is None:
        league_drafts = store.read("league_drafts")
    if league_managers is None:
        league_managers = store.read("league_managers")
    if crosswalk is None:
        crosswalk = store.read("id_crosswalk")
    if adp_history is None:
        adp_history = store.read("adp_history")

    fit_seasons = tuple(s for s in TRAINING_SEASONS if s != holdout_season)
    training, _ = build_training_set(
        league_drafts, league_managers, crosswalk, adp_history, seasons=fit_seasons
    )
    model = fit_opponent_model(training)

    season_drafts = league_drafts.filter(pl.col("season") == holdout_season).sort("overall_pick")
    resolved = _resolve_picks(season_drafts, league_managers, crosswalk, adp_history).sort(
        "overall_pick"
    )

    # The full candidate pool for this season: every ADP entry, keyed the
    # same way `_resolve_picks` keys real picks (`player_key` = normalized
    # name for skill/K, team abbreviation for D/ST) so a real pick's
    # `player_key` can be looked up in it directly.
    season_adp = adp_history.filter(pl.col("season") == holdout_season)
    pool: dict[str, AvailablePlayer] = {}
    for row in season_adp.iter_rows(named=True):
        if row["position"] == "DST":
            key = row["team"]
        else:
            position = _POSITION_ALIASES.get(row["position"], row["position"])
            key = normalize_name(row["name"])
            row = {**row, "position": position}
        pool[key] = AvailablePlayer(player_id=key, position=row["position"], adp=row["adp"])

    roster_counts: dict[str, dict[str, int]] = {}
    n_scored = 0
    n_excluded = 0
    top1_hits = 0
    top5_hits = 0
    baseline_top1_hits = 0
    baseline_top5_hits = 0
    bucket_stats: dict[str, dict[str, int]] = {
        name: {"n": 0, "top1": 0, "top5": 0, "b_top1": 0, "b_top5": 0}
        for name, _ in _ROUND_BUCKETS
    }

    for row in resolved.iter_rows(named=True):
        manager_id = row["manager_id"]
        player_key = row["player_key"]
        counts = roster_counts.setdefault(manager_id, {})

        available = list(pool.values())
        actual_in_pool = player_key in pool

        if available and actual_in_pool:
            probs = pick_probabilities(
                model, manager_id, available, counts, temperature, roster_decay
            )
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            top1_ids = {ranked[0][0]} if ranked else set()
            top5_ids = {pid for pid, _ in ranked[:5]}

            by_adp = sorted(available, key=lambda p: p.adp)
            baseline_top1_ids = {by_adp[0].player_id} if by_adp else set()
            baseline_top5_ids = {p.player_id for p in by_adp[:5]}

            is_top1 = player_key in top1_ids
            is_top5 = player_key in top5_ids
            is_b_top1 = player_key in baseline_top1_ids
            is_b_top5 = player_key in baseline_top5_ids

            n_scored += 1
            top1_hits += int(is_top1)
            top5_hits += int(is_top5)
            baseline_top1_hits += int(is_b_top1)
            baseline_top5_hits += int(is_b_top5)

            bucket = _round_bucket(row["round"])
            bucket_stats[bucket]["n"] += 1
            bucket_stats[bucket]["top1"] += int(is_top1)
            bucket_stats[bucket]["top5"] += int(is_top5)
            bucket_stats[bucket]["b_top1"] += int(is_b_top1)
            bucket_stats[bucket]["b_top5"] += int(is_b_top5)
        else:
            n_excluded += 1

        # Remove the actual pick from the pool (if it was ever in it) and
        # update roster counts, regardless of whether this pick was
        # scoreable -- both must reflect the real draft state for every
        # subsequent pick.
        pool.pop(player_key, None)
        counts[row["position"]] = counts.get(row["position"], 0) + 1

    def _rate(hits: int, n: int) -> float:
        return hits / n if n else 0.0

    by_bucket = {
        name: RoundBucketAccuracy(
            n=stats["n"],
            top1_accuracy=_rate(stats["top1"], stats["n"]),
            top5_accuracy=_rate(stats["top5"], stats["n"]),
            baseline_top1_accuracy=_rate(stats["b_top1"], stats["n"]),
            baseline_top5_accuracy=_rate(stats["b_top5"], stats["n"]),
        )
        for name, stats in bucket_stats.items()
    }

    return BacktestResult(
        n_scored=n_scored,
        n_excluded_unmatched=n_excluded,
        top1_accuracy=_rate(top1_hits, n_scored),
        top5_accuracy=_rate(top5_hits, n_scored),
        baseline_top1_accuracy=_rate(baseline_top1_hits, n_scored),
        baseline_top5_accuracy=_rate(baseline_top5_hits, n_scored),
        by_round_bucket=by_bucket,
    )
