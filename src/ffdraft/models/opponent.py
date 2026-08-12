"""The opponent draft model: P(player p is drafted at pick n | who is still
available), league-wide.

This is the piece that makes the optimizer specific to *this* league rather
than generic best-player-available advice. It is fit on seven seasons
(2018-2024) of this league's real snake drafts, joined against that
season's consensus ADP so draft behaviour can be measured *relative to*
what was "expected" at the time, not relative to hindsight. (The league
has an eighth season of draft history, 2025, but it cannot be used for
fitting -- see below.)

## Why there is no per-manager personalization

An earlier version of this module fit a three-level hierarchical model
(league + manager + manager*position, with empirical-Bayes shrinkage on
the manager terms -- see git history prior to the commit that removed it).
On the real 2018-2023 fit, between-manager variance came out at
tau2 ~= 0.0017 against a within-manager noise floor of sigma2 ~= 0.071 --
individual fitted manager effects landed mostly within +-0.06 of zero, and
the manager*position interaction shrank to exactly 0 for every cell
(tau2_manager_pos == 0.0). Head-to-head on the 2024 holdout backtest, the
per-manager model's predictions were *identical* to a league-wide-only
model at every one of the 165 scored picks -- not just similar accuracy,
but the same top-1 player predicted every single time (see
`scripts/compare_league_wide.py`, run once to produce this evidence, kept
for reproducibility). The manager terms were real but too small, given
this league's thin per-manager sample (2-8 seasons each), to ever change a
single rank-1 decision. The owner decided to drop personalization rather
than carry machinery that provably buys nothing on this data; a future
season with a larger per-manager sample could justify reintroducing it,
which is why `manager_id` is still threaded through this module's public
functions even though it no longer affects their output (see
`pick_probabilities`).

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

# `rankings_2026` (this project's live consensus-rank source, see
# `models/distribution.py`) labels Jacksonville "JAC"; every ADP source used
# here (`adp_history`, `adp_2026`) uses "JAX". Same phenomenon as the
# "WSH" -> "WAS" alias already baked into `PRO_TEAM_ABBREV` above, just
# surfacing from a different upstream source -- fixed the same way, by name,
# rather than silently mismatching every Jacksonville D/ST lookup.
_TEAM_ALIASES: Mapping[str, str] = {"JAC": "JAX"}


# ---------------------------------------------------------------------------
# Shared name/position join helper -- used by `_resolve_picks` (historical
# picks -> that season's ADP) and `build_pool_adp_lookup` (a live player pool
# -> real ADP for `draft.rollout`), so this exact join convention lives in
# one place rather than being copied.


def match_by_normalized_name(
    df: pl.DataFrame,
    adp_table: pl.DataFrame,
    name_col: str = "name",
    position_col: str = "position",
    season_col: str | None = None,
) -> pl.DataFrame:
    """Left-join `df` to `adp_table` on (normalized name, position [,
    season]), adding a nullable `adp` column to `df`.

    Both sides are normalized with `ids.normalize_name` before joining, so
    punctuation/case/suffix differences (e.g. "A.J. Brown" vs "AJ Brown")
    don't cause a spurious miss -- the same normalization `_resolve_picks`
    has always used. `position_col` is matched exactly (no aliasing here;
    callers that need `_POSITION_ALIASES`-style position renaming, e.g. "PK"
    -> "K", must apply it before calling). `season_col`, if given, must name
    a column present in both `df` and `adp_table` and is included in the
    join key -- needed when `adp_table` (like `adp_history`) spans multiple
    seasons and a name could otherwise match the wrong season's row.
    """
    adp_keyed = adp_table.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("_key")
    )
    keyed = df.with_columns(
        pl.col(name_col)
        .map_elements(lambda n: normalize_name(n) if n is not None else None, return_dtype=pl.String)
        .alias("_key")
    )

    join_keys_left = ["_key", position_col] + ([season_col] if season_col else [])
    join_keys_right = ["_key", "position"] + ([season_col] if season_col else [])
    select_cols = ["_key", "position", "adp"] + ([season_col] if season_col else [])

    return keyed.join(
        adp_keyed.select(select_cols),
        left_on=join_keys_left,
        right_on=join_keys_right,
        how="left",
    )


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
    non_dst = non_dst.with_columns(pl.col("position").replace(_POSITION_ALIASES))
    non_dst = match_by_normalized_name(non_dst, adp_history, season_col="season")
    non_dst = non_dst.with_columns(pl.col("_key").alias("player_key"))

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
# Step 2: league-wide reach tendency


@dataclass(frozen=True)
class OpponentModel:
    """A fitted, additive, league-wide model of draft reach.

    `reach` (see `build_training_set`) for pick `i` at position `pos` is
    modeled as:

        reach_i = league_mu + pos_effect[pos] + noise

    `pos_effect` is the league-wide positional bias -- fit directly from
    all ~1,100 usable observations, since it is only 6 categories and every
    one of them (even the thinnest, D/ST) has several dozen observations.
    This is the term Step 3 validates against the known D/ST-early
    tendency, and it also carries the QB-drafted-later-than-consensus
    finding (this league drafts off generic rankings that don't credit its
    6-point passing TDs -- a real, documented league edge, not noise).

    There is deliberately no manager-level term -- see the module
    docstring's "Why there is no per-manager personalization" section for
    the evidence (a fitted per-manager model was head-to-head identical to
    this one on the 2024 holdout backtest).

    `sigma2` is the residual variance left in `reach` after removing
    `league_mu` and `pos_effect` -- i.e. how much pick-to-pick noise this
    simple model doesn't explain. Reported for context, not consumed by
    fitting (there is no shrinkage left to inform).
    """

    league_mu: float
    pos_effect: Mapping[str, float]
    sigma2: float

    def predicted_reach(self, manager_id: str, position: str) -> float:
        """The model's point estimate of log-scale reach at this position --
        `league_mu + pos_effect[position]`, with 0 substituted if the
        position has never been seen.

        `manager_id` is accepted but unused: there is no per-manager term
        (see the module docstring). It stays in the signature so callers
        that already track a manager identity -- `pick_probabilities`,
        `sample_pick`, the backtest replay -- don't need a signature change
        if a future season's larger sample justifies reintroducing
        personalization."""
        return self.league_mu + self.pos_effect.get(position, 0.0)


def fit_opponent_model(training: pl.DataFrame) -> OpponentModel:
    """Fit `OpponentModel` from `build_training_set`'s output: the grand
    mean of `reach` plus each position's mean residual from it. See
    `OpponentModel`'s docstring for the model form.
    """
    reach = training["reach"].to_numpy()
    positions = training["position"].to_numpy()

    league_mu = float(reach.mean())
    resid0 = reach - league_mu

    pos_effect: dict = {}
    for pos in sorted(set(positions.tolist())):
        pos_effect[pos] = float(resid0[positions == pos].mean())

    resid1 = resid0 - np.array([pos_effect[p] for p in positions])
    sigma2 = float(resid1.var())

    return OpponentModel(league_mu=league_mu, pos_effect=pos_effect, sigma2=sigma2)


# ---------------------------------------------------------------------------
# Step 4: the sampling distribution


@dataclass(frozen=True)
class AvailablePlayer:
    """One still-available player, as the sampler needs to see it."""

    player_id: str
    position: str
    adp: float


# ---------------------------------------------------------------------------
# Step 4b: real ADP for a live player pool (`draft.rollout`)
#
# `AvailablePlayer.adp` should be real consensus ADP whenever it exists --
# `pick_probabilities` was calibrated against real ADP (see `OpponentModel`'s
# docstring), and a same-season consensus rank (e.g. FantasyPros ECR) is
# only a proxy for it: checked directly against `data/adp_2026.parquet` for
# this league's real 2026 pool, ECR rank and real ADP rank correlate at
# Spearman 0.933 (strong but not interchangeable) with a mean absolute rank
# disagreement of ~24 places, and ECR covers ~2x as many players as ADP
# (~510 vs ~246) -- a different density on exactly the quantity
# `pick_probabilities` fits (`log(adp) - log(overall_pick)`), which would
# distort reach for anyone past ADP's coverage if a same-scale-looking but
# differently-dense rank were fed in directly.
#
# `build_pool_adp_lookup` below matches a player pool to a real ADP table by
# name (skill/K, via `match_by_normalized_name`) or by team abbreviation
# (D/ST, since ADP exports label a defense "Houston Defense" while this
# project's own rankings call the same team "Houston Texans" -- team
# abbreviation, not name, is the column both sides share reliably). The ADP
# table itself is always a parameter, never hardcoded to 2026: a live
# rollout passes `data/adp_2026.parquet`, Task 6's historical backtest
# passes `adp_history` filtered to the season being replayed.


@dataclass(frozen=True)
class AdpMatchReport:
    """How many of a player pool's entries matched a real ADP row.

    Every pool player is accounted for in exactly one of `matched`/
    `unmatched` -- an unmatched player is never silently dropped from the
    pool and never silently defaulted to some plausible-looking adp value
    here; `unmatched_ids` names them so the caller (`draft.rollout`) can
    decide what to do (see its module docstring for the chosen fallback).
    """

    total: int
    matched: int
    unmatched: int
    unmatched_ids: tuple[str, ...]


def build_pool_adp_lookup(
    pool_rows: pl.DataFrame,
    adp_table: pl.DataFrame,
) -> tuple[dict[str, float], AdpMatchReport]:
    """Match a player pool to real ADP.

    `pool_rows` needs one row per pool player: `player_id`, `name`,
    `position`, and `team` (only consulted for D/ST rows -- the team
    abbreviation used to look up that team's defense in `adp_table`; may be
    null for every non-D/ST row). `adp_table` needs `name`, `position`,
    `team`, `adp` -- the shape both `data/adp_2026.parquet` and
    `adp_history` already have.

    Skill/K positions match by normalized name + position
    (`match_by_normalized_name`); D/ST matches by team abbreviation, with
    `_TEAM_ALIASES` fixing the one known "JAC" vs "JAX" mismatch between
    this project's rankings source and every ADP table.

    Returns `(adp_by_player_id, report)`. `adp_by_player_id` has an entry
    only for players that matched; see `AdpMatchReport` for how the rest are
    reported, never dropped or defaulted here.
    """
    is_dst = pl.col("position") == "DST"
    skill_rows = pool_rows.filter(~is_dst)
    # Cast explicitly: an all-null `team` column (every non-D/ST pool row
    # has no team) otherwise infers a `Null` dtype, which `.replace()`
    # below cannot cast a string mapping onto -- even on the D/ST-only
    # slice, the column's schema dtype is fixed by the full frame it came
    # from until cast.
    dst_rows = pool_rows.filter(is_dst).with_columns(pl.col("team").cast(pl.String))

    skill_matched = match_by_normalized_name(skill_rows, adp_table)

    dst_adp = adp_table.filter(pl.col("position") == "DST").with_columns(
        pl.col("team").cast(pl.String).replace(_TEAM_ALIASES).alias("_team")
    )
    dst_matched = dst_rows.with_columns(pl.col("team").replace(_TEAM_ALIASES).alias("_team")).join(
        dst_adp.select(["_team", "adp"]), on="_team", how="left"
    )

    combined = pl.concat(
        [skill_matched.select(["player_id", "adp"]), dst_matched.select(["player_id", "adp"])]
    )

    adp_by_id = {
        row["player_id"]: row["adp"]
        for row in combined.iter_rows(named=True)
        if row["adp"] is not None
    }
    all_ids = pool_rows["player_id"].to_list()
    unmatched_ids = tuple(pid for pid in all_ids if pid not in adp_by_id)

    report = AdpMatchReport(
        total=len(all_ids),
        matched=len(adp_by_id),
        unmatched=len(unmatched_ids),
        unmatched_ids=unmatched_ids,
    )
    return adp_by_id, report


# Softmax temperature in *rank* space (not raw score space) -- chosen so
# that reach/positional effects are compared in the same round-comparable
# units `reach` itself uses (see module docstring), then converted once to
# a plain, interpretable "how many rank-positions apart" scale. A
# temperature of 3.0 means two players three ranks apart on the manager's
# adjusted board are about e:1 (~2.7x) more/less likely.
#
# **Superseded 2026-08-09 by `TEMPERATURE_BY_ROUND`; retained only so
# `scripts/compare_league_wide.py` can still reproduce the pre-schedule
# numbers it was run to produce.** Do not use it for new work: a single
# temperature is wrong in both directions at once (see below).
LEGACY_FLAT_TEMPERATURE: float = 3.0
DEFAULT_TEMPERATURE: float = LEGACY_FLAT_TEMPERATURE  # backwards-compatible alias

# Temperature rises steeply through the draft: round 1 is near-consensus
# (everyone agrees who the best player is) while round 15 is idiosyncratic.
# One flat number could not represent both, and was measurably wrong at
# each end (`scripts/diagnose_temperature.py`, against 7 real drafts):
#
#     best-available ADP rank   pick 8:  real  5.1  vs flat-3.0  4.2
#                               pick 48: real 32.1  vs flat-3.0 41.6
#
# i.e. too random early (elite players surviving to our pick that never
# survived in reality) *and* too rigid late (burning through good players
# faster than real managers do). Lowering the flat value fixes the first
# and worsens the second.
#
# Fit by maximum likelihood per round over all 1,125 usable picks in
# `TRAINING_SEASONS` (`scripts/fit_temperature.py`), holding the reach
# model and `roster_decay` fixed so this cannot absorb a misfit from
# either. Validated leave-one-season-out: held-out log-likelihood improves
# by +1.72/pick and is better in **7 of 7 seasons**.
#
# Two honest caveats, both measured:
#
# * Nearly all of that +1.72 comes from late rounds. Restricted to rounds
#   1-3 -- the only picks this tool actually advises on -- the gain is
#   +0.065/pick (SE 0.022), better in 6 of 7 seasons. Real, but small.
# * Top-1 accuracy is slightly *worse* (-0.45pp), which is the criterion
#   the flat 3.0 was originally tuned on. That criterion is the wrong one
#   here: rollouts **sample** from this distribution rather than taking its
#   argmax, so calibration is what matters and point accuracy is not.
#
# A linear form (T = 1.95 + 1.16*round) fits the late rounds equally well
# and was rejected: its intercept lands at T(1) = 3.11, outside round 1's
# own 95% confidence interval of [1.85, 2.90], so it is significantly
# wrong exactly at our first pick. The per-round table generalised just as
# well out of sample (+1.72 vs +1.73/pick), so nothing was bought by the
# smoother form.
#
# Values are the raw per-round MLE and are **not** perfectly monotone
# (rounds 4-5 and 14-16 dip slightly). Theory says temperature should rise
# monotonically; those dips are sampling noise at ~70 picks per round.
# They are left unsmoothed because smoothing them changed nothing
# measurable, and an unsmoothed fit is easier to re-derive and check.
TEMPERATURE_BY_ROUND: Mapping[int, float] = MappingProxyType({
    1: 2.30, 2: 4.20, 3: 4.90, 4: 6.95, 5: 6.90, 6: 12.00,
    7: 11.00, 8: 12.00, 9: 11.50, 10: 12.50, 11: 12.50, 12: 15.00,
    13: 17.50, 14: 20.50, 15: 19.50, 16: 17.50, 17: 24.00,
})

_DEEPEST_FITTED_ROUND: int = max(TEMPERATURE_BY_ROUND)


def temperature_for_round(round_: int) -> float:
    """The fitted softmax temperature for a 1-indexed draft round.

    Rounds past the deepest fitted one reuse that round's value rather than
    extrapolating: round 18 exists only in this league's 2023+ drafts and
    has too few training picks to fit its own temperature, and by then the
    board is flat enough that the distinction is immaterial. Extrapolating
    a trend off the end of the data would be inventing a number.
    """
    if round_ < 1:
        raise ValueError(f"round_ must be 1-indexed and positive, got {round_}")
    return TEMPERATURE_BY_ROUND[min(round_, _DEEPEST_FITTED_ROUND)]


def _resolve_temperature(temperature: float | None, round_: int | None) -> float:
    """Either an explicit temperature or the fitted one for `round_`.

    Raises when given neither, rather than quietly falling back to the
    superseded flat 3.0 -- a caller that forgot to thread the round through
    would otherwise get plausible-looking output from the very model this
    schedule was written to replace.
    """
    if temperature is not None:
        return temperature
    if round_ is None:
        raise ValueError(
            "pass either an explicit `temperature` or the `round_` to look one "
            "up for -- see TEMPERATURE_BY_ROUND for why there is no flat default"
        )
    return temperature_for_round(round_)

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
    temperature: float | None = None,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
    round_: int | None = None,
) -> dict[str, float]:
    """`P(manager_id drafts p)` for each `p` in `available`, given the
    manager's current `roster_counts` (`{position: count}`).

    `manager_id` no longer changes the reach term (`OpponentModel.
    predicted_reach` is league-wide only -- see that docstring); it is kept
    as a parameter only because `roster_counts` is genuinely per-manager
    state the caller already tracks, and because a future personalized
    model would want it back at this call site without a signature change.

    Every player in `available` gets a nonzero probability (see
    `_MIN_ROSTER_MULTIPLIER`); probabilities are restricted to exactly the
    players passed in (nothing outside `available` can be returned) and sum
    to 1.

    Form: each player's `adp` is adjusted by the league-wide fitted reach at
    that position (`OpponentModel.predicted_reach`) to get a predicted
    log-scale pick position -- "when a typical manager would take a player
    at this ADP". Available players are ranked by that adjusted value (rank
    1 = the most-wanted player still on the board) and a softmax over
    `-rank/temperature` converts ranks to probabilities.
    Rank space (not raw adjusted-value space) is used for the softmax so
    `temperature` means the same thing regardless of the size of the
    remaining board -- see `TEMPERATURE_BY_ROUND`.

    **Temperature comes from the fitted per-round schedule unless one is
    passed explicitly.** Give `round_` (1-indexed) and leave `temperature`
    as `None` for normal use; pass an explicit `temperature` only to pin a
    value (tests, and the sweeps that measured this). Passing neither
    raises rather than silently reinstating the superseded flat 3.0.

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

    temperature = _resolve_temperature(temperature, round_)
    n = len(available)
    adp = np.empty(n)
    positions: list[str] = []
    player_ids: list[str] = []
    for i, p in enumerate(available):
        adp[i] = p.adp
        positions.append(p.position)
        player_ids.append(p.player_id)

    # Both the reach adjustment and the roster-need penalty depend only on
    # a player's *position*, of which there are six -- but the obvious
    # per-player comprehension called `predicted_reach` 12.4 million times
    # in a single `recommend_pick`. Look each up once per position.
    unique_positions = set(positions)
    reach_by_position = {
        pos: model.predicted_reach(manager_id, pos) for pos in unique_positions
    }
    multiplier_by_position = {
        pos: max(roster_decay ** roster_counts.get(pos, 0), _MIN_ROSTER_MULTIPLIER)
        for pos in unique_positions
    }

    reach_adjust = np.fromiter(
        (reach_by_position[pos] for pos in positions), dtype=float, count=n
    )
    adjusted_log_pick = np.log(adp) - reach_adjust

    # rank 0 = smallest adjusted_log_pick = the manager's top choice
    order = np.argsort(adjusted_log_pick, kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(n)

    base_weight = np.exp(-rank / temperature)

    roster_multiplier = np.fromiter(
        (multiplier_by_position[pos] for pos in positions), dtype=float, count=n
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
    temperature: float | None = None,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
    round_: int | None = None,
) -> str:
    """Draw one `player_id` from `pick_probabilities`'s distribution using
    `rng` (an explicit `np.random.Generator` -- never global numpy state,
    so draws are reproducible from a seed and safe to call from parallel
    rollouts). Raises `ValueError` if `available` is empty."""
    if not available:
        raise ValueError("sample_pick called with no available players")

    probs_by_id = pick_probabilities(
        model, manager_id, available, roster_counts, temperature, roster_decay, round_
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
    temperature: float | None = None,
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
                model, manager_id, available, counts, temperature, roster_decay,
                round_=row["round"],
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
