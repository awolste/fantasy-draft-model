"""Tests for `ffdraft.backtest` (Stage 3 Task 6).

Most tests use small hand-built test doubles (same style as
`tests/test_roster.py`/`tests/test_season.py`) so they are fast and
independent of the data ingest pipeline. A final, slower group exercises
`fit_holdout_context()` against real data to verify the leakage cuts
actually land where the module docstring claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from ffdraft.backtest import (
    CONTENDERS,
    RealizationResult,
    _deep_rankings_holdout,
    adp_pick_policy,
    consensus_rank_pick_policy,
    random_legal_pick_policy,
    real_season_champion,
    real_team_weekly_totals,
    resolve_real_player_weeks,
    summarize,
)
from ffdraft.league import N_TEAMS
from ffdraft.models.distribution import PlayerDistribution
from ffdraft.models.replacement import ReplacementLevelDistribution


@dataclass(frozen=True)
class _ConstDist:
    value: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, self.value)

    @property
    def mean(self) -> float:
        return self.value


def _pd(player_id: str, position: str, rank: int) -> PlayerDistribution:
    return PlayerDistribution(
        player_id=player_id, name=player_id, position=position, rank=rank, tier=1,
        distribution=_ConstDist(10.0),
    )


# ---------------------------------------------------------------------------
# Pick policies


def test_adp_pick_policy_takes_lowest_adp_available():
    pool = {"a": _pd("a", "RB", 3), "b": _pd("b", "WR", 1), "c": _pd("c", "QB", 2)}
    adp_by_id = {"a": 30.0, "b": 5.0, "c": 15.0}
    policy = adp_pick_policy(adp_by_id)
    pid, pos = policy(["a", "b", "c"], pool, [], {}, {})
    assert (pid, pos) == ("b", "WR")


def test_adp_pick_policy_missing_adp_sorts_last():
    pool = {"a": _pd("a", "RB", 1), "b": _pd("b", "WR", 2)}
    adp_by_id = {"a": 10.0}  # "b" has no entry
    policy = adp_pick_policy(adp_by_id)
    pid, _ = policy(["a", "b"], pool, [], {}, {})
    assert pid == "a"


def test_consensus_rank_pick_policy_takes_lowest_rank():
    pool = {"a": _pd("a", "RB", 30), "b": _pd("b", "WR", 5), "c": _pd("c", "QB", 15)}
    policy = consensus_rank_pick_policy()
    pid, pos = policy(["a", "b", "c"], pool, [], {}, {})
    assert (pid, pos) == ("b", "WR")


def test_random_legal_pick_policy_is_reproducible_from_seed():
    pool = {f"p{i}": _pd(f"p{i}", "RB", i) for i in range(20)}
    available = list(pool.keys())
    picks_a = [
        random_legal_pick_policy(np.random.default_rng(7))(available, pool, [], {}, {}) for _ in range(5)
    ]
    picks_b = [
        random_legal_pick_policy(np.random.default_rng(7))(available, pool, [], {}, {}) for _ in range(5)
    ]
    assert picks_a == picks_b


def test_random_legal_pick_policy_only_returns_available_ids():
    pool = {f"p{i}": _pd(f"p{i}", "WR", i) for i in range(5)}
    rng = np.random.default_rng(1)
    policy = random_legal_pick_policy(rng)
    for _ in range(20):
        pid, pos = policy(list(pool.keys()), pool, [], {}, {})
        assert pid in pool
        assert pos == pool[pid].position


# ---------------------------------------------------------------------------
# Real-results player resolution


REPLACEMENT = {
    "RB": ReplacementLevelDistribution(position="RB", values=(5.0, 6.0, 7.0)),
    "WR": ReplacementLevelDistribution(position="WR", values=(4.0, 5.0, 6.0)),
    "K": ReplacementLevelDistribution(position="K", values=(3.0, 4.0)),
    "DST": _ConstDist(7.5),
}


def test_resolve_real_player_uses_real_weekly_scores_when_matched():
    weekly_lookup = {"p1": {1: 20.0, 2: 15.0}}  # no row for week 3 -> unavailable
    result = resolve_real_player_weeks(
        "p1", "RB", weekly_lookup, {}, REPLACEMENT, np.random.default_rng(1), n_weeks=3
    )
    assert result.scores == (20.0, 15.0, 0.0)
    assert result.available == (True, True, False)
    assert result.used_replacement_fallback is False


def test_resolve_real_player_falls_back_to_replacement_when_never_played():
    result = resolve_real_player_weeks(
        "ghost", "RB", {}, {}, REPLACEMENT, np.random.default_rng(1), n_weeks=5
    )
    assert result.used_replacement_fallback is True
    assert all(result.available)
    assert all(s in REPLACEMENT["RB"].values for s in result.scores)


def test_resolve_real_player_dst_always_uses_shared_distribution():
    # Even if a DST-shaped pick id happened to collide with a weekly_lookup
    # key, D/ST must never resolve to real weekly_stats rows -- this
    # project never ingested per-team defensive game stats (see
    # models/defense.py's docstring).
    weekly_lookup = {"DST::Made Up Defense": {1: 999.0}}
    result = resolve_real_player_weeks(
        "DST::Made Up Defense", "DST", weekly_lookup, {}, REPLACEMENT, np.random.default_rng(1), n_weeks=2
    )
    assert 999.0 not in result.scores
    assert all(result.available)


def test_resolve_real_player_kicker_resolves_by_normalized_name():
    weekly_lookup = {"gsis_k1": {1: 8.0, 2: 9.0}}
    kicker_lookup = {"justin tucker": "gsis_k1"}
    result = resolve_real_player_weeks(
        "K::Justin Tucker", "K", weekly_lookup, kicker_lookup, REPLACEMENT,
        np.random.default_rng(1), n_weeks=2,
    )
    assert result.scores == (8.0, 9.0)
    assert result.used_replacement_fallback is False


# ---------------------------------------------------------------------------
# Team/season scoring


def test_real_team_weekly_totals_falls_back_to_replacement_for_unfilled_slot():
    # A one-player "roster" leaves every other dedicated slot empty --
    # solve_lineup must fill them from replacement_means, never crash.
    replacement_means = {"QB": 1.0, "RB": 2.0, "WR": 3.0, "TE": 4.0, "K": 5.0, "DST": 6.0}
    weekly_lookup = {"only_rb": {1: 20.0}}
    totals, n_fb = real_team_weekly_totals(
        [("only_rb", "RB")], weekly_lookup, {}, REPLACEMENT, replacement_means,
        np.random.default_rng(1), {"only_rb": 15.0}, n_weeks=1,
    )
    assert n_fb == 0
    assert totals.shape == (1,)
    assert totals[0] > 0  # real RB score plus a pile of replacement-level slots


def test_lineup_is_chosen_ex_ante_but_scored_on_realized_points():
    """The lineup must be picked on *projected* means and scored on
    *realized* points. Choosing it on realized points is hindsight, and it
    silently favours depth at FLEX-eligible positions -- worth +5.75pp to a
    depth-stacked roster (docs/HANDOFF.md 7b, test 6). See
    `real_team_weekly_totals`.

    The league starts 2 RB + 2 FLEX, so five RBs are needed to force a real
    bench decision. Four are well projected and all bust (1.0 each); the
    fifth is projected at replacement level and explodes for 99.0. An
    ex-ante manager starts the four he projected and scores 4.0 from RBs.
    Hindsight would start the 99.
    """
    replacement_means = {"RB": 2.0, "WR": 3.0, "TE": 4.0, "QB": 1.0, "K": 5.0, "DST": 6.0}
    picks = [(f"rb{i}", "RB") for i in range(5)]
    weekly_lookup = {f"rb{i}": {1: 1.0} for i in range(4)} | {"rb4": {1: 99.0}}
    projected = {"rb0": 20.0, "rb1": 19.0, "rb2": 18.0, "rb3": 17.0, "rb4": 2.0}
    totals, _ = real_team_weekly_totals(
        picks, weekly_lookup, {}, REPLACEMENT, replacement_means,
        np.random.default_rng(1), projected, n_weeks=1,
    )
    # 4 started RBs x 1.0 realized, plus every other slot at replacement:
    # QB 1.0 + WR 2x3.0 + TE 4.0 + K 5.0 + DST 6.0 = 22.0
    assert totals[0] == pytest.approx(4.0 + 22.0), (
        "lineup was chosen on realized scores (hindsight), not projected means"
    )


def test_a_player_with_no_projection_falls_back_to_his_replacement_mean():
    """A pick with no entry in `projected_mean` must not crash and must not
    be treated as projecting zero -- replacement level is the right prior."""
    replacement_means = {"RB": 2.0, "WR": 3.0, "TE": 4.0, "QB": 1.0, "K": 5.0, "DST": 6.0}
    totals, _ = real_team_weekly_totals(
        [("unknown_rb", "RB")], {"unknown_rb": {1: 12.0}}, {}, REPLACEMENT,
        replacement_means, np.random.default_rng(1), {}, n_weeks=1,
    )
    assert totals[0] > 0


def test_real_season_champion_returns_a_valid_team_and_is_deterministic():
    from ffdraft.draft.rollout import DraftState, Pick

    replacement_means = {"QB": 1.0, "RB": 2.0, "WR": 3.0, "TE": 4.0, "K": 5.0, "DST": 6.0}
    picks = []
    overall = 1
    for rnd in range(1, 3):
        order = range(1, N_TEAMS + 1) if rnd % 2 == 1 else range(N_TEAMS, 0, -1)
        for team in order:
            picks.append(Pick(overall_pick=overall, team=team, player_id=f"t{team}_r{rnd}", position="RB"))
            overall += 1
    state = DraftState.from_picks(picks, n_teams=N_TEAMS, rounds=2)
    weekly_holdout = pl.DataFrame(
        {
            "player_id": [f"t{t}_r{r}" for t in range(1, N_TEAMS + 1) for r in range(1, 3)],
            "player_name": ["x"] * (N_TEAMS * 2),
            "position": ["RB"] * (N_TEAMS * 2),
            "week": [1] * (N_TEAMS * 2),
            "fantasy_points": [float(t) for t in range(1, N_TEAMS + 1) for _ in range(2)],
        }
    )
    champ_a, _ = real_season_champion(state, weekly_holdout, REPLACEMENT, replacement_means, seed=42, projected_mean={})
    champ_b, _ = real_season_champion(state, weekly_holdout, REPLACEMENT, replacement_means, seed=42, projected_mean={})
    assert champ_a == champ_b
    assert 1 <= champ_a <= N_TEAMS


# ---------------------------------------------------------------------------
# Summarize / paired comparison


def _realization(seed, wins: dict[str, bool]) -> RealizationResult:
    return RealizationResult(
        seed=seed,
        champion_by_contender={},
        our_win=wins,
        n_fallback_by_contender={c: 0 for c in CONTENDERS},
        round_divergence_engine_vs_adp={1: True, 5: False, 15: True},
    )


def test_summarize_computes_rates_and_paired_diff():
    results = [
        _realization(1, {"engine": True, "adp": False, "consensus": False, "random": False}),
        _realization(2, {"engine": False, "adp": True, "consensus": True, "random": False}),
        _realization(3, {"engine": True, "adp": True, "consensus": True, "random": False}),
        _realization(4, {"engine": False, "adp": False, "consensus": False, "random": False}),
    ]
    summary = summarize(results, elapsed_seconds=1.0)
    assert summary.contenders["engine"].championship_rate == pytest.approx(0.5)
    assert summary.contenders["adp"].championship_rate == pytest.approx(0.5)
    assert summary.contenders["random"].championship_rate == pytest.approx(0.0)
    # engine wins realizations 1,3; adp wins 2,3 -> paired diff per realization:
    # [1-0, 0-1, 1-1, 0-0] = [1,-1,0,0] -> mean 0
    assert summary.paired_vs_engine["adp"].paired_diff_pp == pytest.approx(0.0)


def test_summarize_round_bucket_divergence_aggregates_across_realizations():
    results = [_realization(1, {c: False for c in CONTENDERS}) for _ in range(3)]
    summary = summarize(results, elapsed_seconds=1.0)
    # round 1 -> early, round 5 -> middle, round 15 -> late (per opponent._round_bucket)
    assert summary.round_bucket_divergence["early"].n == 3
    assert summary.round_bucket_divergence["early"].divergence_rate == pytest.approx(1.0)
    assert summary.round_bucket_divergence["middle"].divergence_rate == pytest.approx(0.0)
    assert summary.round_bucket_divergence["late"].n == 3


# ---------------------------------------------------------------------------
# Deep rankings extension (pool-depth fix)


def test_deep_rankings_holdout_extends_past_adp_coverage_without_duplicating():
    adp_holdout = pl.DataFrame(
        {
            "name": ["Star RB", "Star WR"],
            "position": ["RB", "WR"],
            "team": [None, None],
            "adp": [1.0, 2.0],
            "season": [2024, 2024],
        }
    )
    adp_history_full = pl.DataFrame(
        {
            "name": ["Deep RB"],
            "position": ["RB"],
            "team": [None],
            "adp": [50.0],
            "season": [2022],
        }
    )
    weekly_holdout = pl.DataFrame(
        {
            "player_name": ["Star RB", "Star WR", "Deep RB", "Unranked WR"],
            "position": ["RB", "WR", "RB", "WR"],
        }
    )
    out = _deep_rankings_holdout(adp_holdout, adp_history_full, weekly_holdout, fit_through_season=2023)
    names = out["name"].to_list()
    assert names.count("Star RB") == 1
    assert names.count("Star WR") == 1
    assert "Deep RB" in names
    assert "Unranked WR" in names
    # base ADP rows keep the lowest ranks; extras come after, and the extra
    # with real historical ADP (Deep RB) sorts ahead of the one with none.
    ranked = out.sort("rank")["name"].to_list()
    assert ranked.index("Deep RB") < ranked.index("Unranked WR")
    assert ranked.index("Star RB") < ranked.index("Deep RB")


def test_deep_rankings_holdout_is_a_noop_when_nothing_new_appeared():
    adp_holdout = pl.DataFrame(
        {"name": ["Only Player"], "position": ["QB"], "team": [None], "adp": [1.0], "season": [2024]}
    )
    weekly_holdout = pl.DataFrame({"player_name": ["Only Player"], "position": ["QB"]})
    out = _deep_rankings_holdout(adp_holdout, pl.DataFrame(
        {"name": [], "position": [], "team": [], "adp": [], "season": []},
        schema={"name": pl.String, "position": pl.String, "team": pl.String, "adp": pl.Float64, "season": pl.Int64},
    ), weekly_holdout, fit_through_season=2023)
    assert out.height == 1


# ---------------------------------------------------------------------------
# Slower, real-data integration checks: the leakage cuts actually land where
# the module docstring claims.


@pytest.fixture(scope="module")
def holdout_ctx():
    from ffdraft.backtest import fit_holdout_context

    return fit_holdout_context()


def test_fit_report_seasons_never_include_the_holdout_season(holdout_ctx):
    report = holdout_ctx.report
    assert report.tier_shape_seasons[1] <= 2023
    assert report.rank_curve_seasons[1] <= 2023
    assert max(report.opponent_model_seasons) <= 2023
    assert report.replacement_seasons[1] <= 2023
    assert max(report.availability_seasons) <= 2023
    assert report.holdout_season == 2024


def test_holdout_pool_is_deep_enough_for_a_full_draft(holdout_ctx):
    from ffdraft.league import N_TEAMS

    assert len(holdout_ctx.pool) >= N_TEAMS * holdout_ctx.rounds


def test_holdout_context_does_not_write_to_the_live_unnamespaced_caches(holdout_ctx, monkeypatch):
    # Regression guard for the exact scenario the module docstring
    # describes: calling the *cached* wrapper with through-2023 data while
    # the live (full-history) cache already exists under the same
    # unnamespaced name must raise CacheStaleError, not silently succeed --
    # this is why fit_holdout_context uses the uncached fit_* functions
    # directly for tier shapes/replacement/availability instead.
    import polars as pl

    from ffdraft import store
    from ffdraft.models.tier_shape import load_or_fit_tier_shapes

    if not store.exists("distribution_tier_shapes"):
        pytest.skip("no live tier-shape cache present to collide with")

    weekly_full = store.read("weekly_stats")
    weekly_through_2023 = weekly_full.filter(pl.col("season") <= 2023)
    if weekly_through_2023.height == weekly_full.height:
        pytest.skip("live weekly_stats has no post-2023 rows to differ on")

    with pytest.raises(store.CacheStaleError):
        load_or_fit_tier_shapes(weekly_through_2023)
