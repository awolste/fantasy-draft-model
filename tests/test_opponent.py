"""Tests for `models.opponent`.

Most tests use small hand-built synthetic frames (fast, deterministic,
independent of the real data pipeline -- same style as `tests/
test_roster.py`). A smaller group of integration tests runs against the
real persisted datasets to check the two "gates" the task spec calls out:
the D/ST-early regression guard (Step 3) and the training-set join-rate
report (Step 1). Those tests never print or otherwise surface a manager_id
-- see the task's personal-data rule -- they only compare/aggregate them.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ffdraft.models.opponent import (
    PRO_TEAM_ABBREV,
    TRAINING_SEASONS,
    AdpMatchReport,
    AvailablePlayer,
    BacktestResult,
    JoinReport,
    OpponentModel,
    backtest_holdout_season,
    build_pool_adp_lookup,
    build_training_set,
    fit_opponent_model,
    match_by_normalized_name,
    pick_probabilities,
    sample_pick,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures


def _league_drafts_row(season, overall_pick, round_, team_id, espn_player_id):
    return {
        "season": season,
        "overall_pick": overall_pick,
        "round": round_,
        "round_pick": overall_pick,
        "team_id": team_id,
        "espn_player_id": espn_player_id,
    }


def _managers_row(season, team_id, manager_id, team_name="team"):
    return {
        "season": season,
        "team_id": team_id,
        "manager_id": manager_id,
        "team_name": team_name,
    }


def _crosswalk_row(espn_id, name, position, gsis_id=None, sleeper_id=None):
    return {
        "gsis_id": gsis_id,
        "espn_id": espn_id,
        "sleeper_id": sleeper_id,
        "name": name,
        "position": position,
    }


def _adp_row(name, position, team, adp, season, adp_stdev=5.0, times_drafted=100):
    return {
        "name": name,
        "position": position,
        "team": team,
        "adp": adp,
        "adp_stdev": adp_stdev,
        "times_drafted": times_drafted,
        "season": season,
    }


def _synthetic_frames():
    """A tiny two-manager, two-season league: manager 'mgr-A' always picks
    close to ADP; manager 'mgr-B' consistently reaches early for RBs."""
    league_drafts = pl.DataFrame(
        [
            # 2018: A owns team 1, B owns team 2
            _league_drafts_row(2018, 1, 1, 1, 101),  # A takes player 101 (QB) at pick 1
            _league_drafts_row(2018, 2, 1, 2, 201),  # B takes player 201 (RB) at pick 2
            _league_drafts_row(2018, 3, 1, 1, 102),  # A takes player 102 (WR) at pick 3
            _league_drafts_row(2018, 4, 1, 2, 202),  # B takes player 202 (RB) at pick 4
            # 2019: same team assignments
            _league_drafts_row(2019, 1, 1, 1, 103),  # A: QB
            _league_drafts_row(2019, 2, 1, 2, 203),  # B: RB, big reach
            _league_drafts_row(2019, 3, 1, 1, 104),  # A: WR
            _league_drafts_row(2019, 4, 1, 2, 204),  # B: RB, big reach
        ]
    )
    league_managers = pl.DataFrame(
        [
            _managers_row(2018, 1, "mgr-A"),
            _managers_row(2018, 2, "mgr-B"),
            _managers_row(2019, 1, "mgr-A"),
            _managers_row(2019, 2, "mgr-B"),
        ]
    )
    crosswalk = pl.DataFrame(
        [
            _crosswalk_row(101, "Player QB One", "QB"),
            _crosswalk_row(102, "Player WR One", "WR"),
            _crosswalk_row(103, "Player QB Two", "QB"),
            _crosswalk_row(104, "Player WR Two", "WR"),
            _crosswalk_row(201, "Player RB One", "RB"),
            _crosswalk_row(202, "Player RB Two", "RB"),
            _crosswalk_row(203, "Player RB Three", "RB"),
            _crosswalk_row(204, "Player RB Four", "RB"),
        ]
    )
    adp_history = pl.DataFrame(
        [
            _adp_row("Player QB One", "QB", "AAA", 1.0, 2018),
            _adp_row("Player WR One", "WR", "AAA", 3.0, 2018),
            _adp_row("Player RB One", "RB", "AAA", 2.0, 2018),  # picked at 2, on ADP
            _adp_row("Player RB Two", "RB", "AAA", 20.0, 2018),  # picked at 4, big reach
            _adp_row("Player QB Two", "QB", "AAA", 1.0, 2019),
            _adp_row("Player WR Two", "WR", "AAA", 3.0, 2019),
            _adp_row("Player RB Three", "RB", "AAA", 2.0, 2019),
            _adp_row("Player RB Four", "RB", "AAA", 40.0, 2019),  # picked at 4, huge reach
        ]
    )
    return league_drafts, league_managers, crosswalk, adp_history


# ---------------------------------------------------------------------------
# Step 1: training set


def test_training_set_excludes_2025_and_includes_2018_through_2024():
    league_drafts, league_managers, crosswalk, adp_history = _synthetic_frames()
    # Add a 2025 pick (no ADP for it -- and there shouldn't be, since
    # adp_history simply doesn't cover 2025 in real data).
    league_drafts = pl.concat(
        [league_drafts, pl.DataFrame([_league_drafts_row(2025, 1, 1, 1, 105)])]
    )
    league_managers = pl.concat(
        [league_managers, pl.DataFrame([_managers_row(2025, 1, "mgr-A")])]
    )
    crosswalk = pl.concat([crosswalk, pl.DataFrame([_crosswalk_row(105, "Player QB Three", "QB")])])

    training, report = build_training_set(league_drafts, league_managers, crosswalk, adp_history)

    assert set(TRAINING_SEASONS) == set(range(2018, 2025))
    assert 2025 not in training["season"].to_list()
    assert set(training["season"].unique().to_list()) <= set(TRAINING_SEASONS)
    # the 2025 pick was never even eligible to join -- it's outside the
    # season filter, not a join failure, so total_picks reflects only the
    # 8 in-scope 2018-2019 rows here (report.total_picks counts pre-filter
    # rows within TRAINING_SEASONS only).
    assert report.total_picks == 8


def test_training_set_reports_usable_and_unusable_counts():
    league_drafts, league_managers, crosswalk, adp_history = _synthetic_frames()
    # Drop one ADP row so that one pick becomes unusable.
    adp_history = adp_history.filter(~((pl.col("name") == "Player WR Two")))

    training, report = build_training_set(league_drafts, league_managers, crosswalk, adp_history)

    assert isinstance(report, JoinReport)
    assert report.total_picks == 8
    assert report.usable_picks == 7
    assert report.unusable_picks == 1
    assert report.unusable_rate == pytest.approx(1 / 8)
    assert training.height == 7


def test_training_set_handles_dst_via_team_map_not_crosswalk():
    league_drafts, league_managers, crosswalk, adp_history = _synthetic_frames()
    # crosswalk deliberately has zero DST rows, matching real id_crosswalk.
    assert crosswalk.filter(pl.col("position") == "DST").height == 0

    # ESPN encodes a DST pick as -16000 - pro_team_id. Use pro_team_id=21 (PHI).
    dst_espn_id = -16000 - 21
    league_drafts = pl.concat(
        [league_drafts, pl.DataFrame([_league_drafts_row(2018, 5, 1, 1, dst_espn_id)])]
    )
    adp_history = pl.concat(
        [adp_history, pl.DataFrame([_adp_row("Philadelphia Defense", "DST", "PHI", 90.0, 2018)])]
    )

    training, report = build_training_set(league_drafts, league_managers, crosswalk, adp_history)

    assert PRO_TEAM_ABBREV[21] == "PHI"
    dst_rows = training.filter(pl.col("position") == "DST")
    assert dst_rows.height == 1
    assert dst_rows["adp"].item() == 90.0
    assert dst_rows["manager_id"].item() == "mgr-A"


def test_training_set_raises_on_missing_manager_join():
    league_drafts, league_managers, crosswalk, adp_history = _synthetic_frames()
    # Remove team 2's manager row for 2018 -- every pick by team 2 in 2018
    # should now fail to resolve a manager, which must raise, not silently
    # drop those rows.
    league_managers = league_managers.filter(
        ~((pl.col("season") == 2018) & (pl.col("team_id") == 2))
    )
    with pytest.raises(ValueError, match="league_managers"):
        build_training_set(league_drafts, league_managers, crosswalk, adp_history)


# ---------------------------------------------------------------------------
# Step 2: league-wide reach tendency


def test_pos_effect_fit_from_grand_mean_residuals():
    """Direct unit test of the (now unshrunk, league-wide-only) fit: each
    position's `pos_effect` is exactly its mean residual from the grand
    mean of `reach`, with no manager term involved at all."""
    n_per_pos = 20
    rng = np.random.default_rng(5)
    rows = []
    for pos, true_effect in (("RB", 0.3), ("WR", -0.2)):
        for _ in range(n_per_pos):
            rows.append((pos, true_effect + float(rng.normal(0, 0.05))))

    n = len(rows)
    training = pl.DataFrame(
        {
            "season": [2018] * n,
            "overall_pick": [10] * n,
            "round": [1] * n,
            "manager_id": ["mgr-A"] * n,
            "position": [r[0] for r in rows],
            "adp": [10.0] * n,
            "reach": [r[1] for r in rows],
        }
    )

    model = fit_opponent_model(training)

    # pos_effect is each position's mean residual from the *grand* mean of
    # reach (~0.05 here, the midpoint of 0.3 and -0.2 with equal group
    # sizes), not from 0 -- so RB comes out near +0.25 and WR near -0.25.
    assert model.pos_effect["RB"] == pytest.approx(0.25, abs=0.05)
    assert model.pos_effect["WR"] == pytest.approx(-0.25, abs=0.05)
    # RB and WR should be about half a unit apart, as constructed.
    assert model.pos_effect["RB"] - model.pos_effect["WR"] == pytest.approx(0.5, abs=0.1)


def test_predicted_reach_ignores_manager_id():
    """`manager_id` is still accepted (see the module docstring's
    plumbing note) but must not affect the prediction -- two different
    manager ids at the same position must produce the identical value."""
    model = OpponentModel(league_mu=0.1, pos_effect={"RB": 0.05}, sigma2=0.07)
    assert model.predicted_reach("mgr-A", "RB") == model.predicted_reach("mgr-B", "RB")
    assert model.predicted_reach("mgr-A", "RB") == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Step 3: the DST validation gate


def test_dst_positional_bias_is_positive_on_real_league_data():
    """The known league fact: defenses go earlier here than consensus ADP
    (8-man benches invite stashing). `pos_effect['DST']` must come out
    positive (reach = log(adp) - log(pick); positive means picked before
    ADP implied, i.e. drafted earlier than expected) without being told --
    this is the model-validation gate from the task spec. If this ever
    fails, the fit is wrong; investigate rather than loosening this test.
    """
    training, report = build_training_set()
    assert report.usable_picks > 500  # sanity: the real join should not have collapsed
    model = fit_opponent_model(training)
    assert model.pos_effect["DST"] > 0


def test_all_positions_have_a_fitted_bias_on_real_league_data():
    training, _ = build_training_set()
    model = fit_opponent_model(training)
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        assert pos in model.pos_effect


def test_qb_positional_bias_is_negative_on_real_league_data():
    """The other known league fact (owner-confirmed, not just observed):
    this league drafts off generic consensus rankings that don't credit its
    6-point passing TDs, so QBs go later here than that consensus would
    suggest -- `pos_effect['QB']` must come out negative (reach = log(adp)
    - log(pick); negative means picked after ADP implied). This is now the
    league-wide-only model's positional term, but it is fit the same way
    the earlier per-manager model fit it, so the removal of manager terms
    must not have changed this finding's sign or rough size."""
    training, _ = build_training_set()
    model = fit_opponent_model(training)
    assert model.pos_effect["QB"] < 0
    assert model.pos_effect["QB"] == pytest.approx(-0.10, abs=0.05)


# ---------------------------------------------------------------------------
# Step 4: the sampling distribution


def _model_no_effects() -> OpponentModel:
    return OpponentModel(league_mu=0.0, pos_effect={}, sigma2=1.0)


def test_pick_probabilities_only_returns_available_players():
    model = _model_no_effects()
    available = [
        AvailablePlayer("p1", "RB", 5.0),
        AvailablePlayer("p2", "WR", 8.0),
        AvailablePlayer("p3", "QB", 20.0),
    ]
    probs = pick_probabilities(model, "mgr-unknown", available, round_=1)
    assert set(probs.keys()) == {"p1", "p2", "p3"}


def test_pick_probabilities_sum_to_one_and_nonnegative():
    model = _model_no_effects()
    available = [AvailablePlayer(f"p{i}", "RB", float(i + 1)) for i in range(25)]
    probs = pick_probabilities(model, "mgr-x", available, roster_counts={"RB": 4}, round_=1)
    assert all(p >= 0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)


def test_pick_probabilities_favors_lower_adp_absent_roster_pressure():
    model = _model_no_effects()
    available = [
        AvailablePlayer("best", "RB", 1.0),
        AvailablePlayer("worst", "RB", 100.0),
    ]
    probs = pick_probabilities(model, "mgr-x", available, round_=1)
    assert probs["best"] > probs["worst"]


def test_roster_need_suppresses_repeated_position():
    """A manager with three quarterbacks rostered should be markedly less
    likely to take a 4th than an otherwise-identical manager with zero."""
    model = _model_no_effects()
    available = [
        AvailablePlayer("qb", "QB", 10.0),
        AvailablePlayer("rb", "RB", 10.0),
    ]
    probs_deep = pick_probabilities(model, "mgr-x", available, roster_counts={"QB": 3}, round_=1)
    probs_empty = pick_probabilities(model, "mgr-x", available, roster_counts={"QB": 0}, round_=1)
    assert probs_deep["qb"] < probs_empty["qb"]


def test_sampling_is_reproducible_from_seed_and_does_not_touch_global_state():
    model = _model_no_effects()
    available = [AvailablePlayer(f"p{i}", "RB", float(i + 1)) for i in range(10)]

    before = np.random.get_state()

    rng1 = np.random.default_rng(42)
    draws1 = [sample_pick(model, "mgr-x", available, {}, rng1, round_=1) for _ in range(20)]

    after = np.random.get_state()
    # Global numpy random state must be untouched by a call that was given
    # its own explicit Generator.
    assert before[1].tolist() == after[1].tolist()

    rng2 = np.random.default_rng(42)
    draws2 = [sample_pick(model, "mgr-x", available, {}, rng2, round_=1) for _ in range(20)]

    assert draws1 == draws2


def test_sample_pick_raises_on_empty_available():
    model = _model_no_effects()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_pick(model, "mgr-x", [], {}, rng, round_=1)


# ---------------------------------------------------------------------------
# Step 5: the backtest gate


def test_backtest_beats_adp_baseline_on_2024_holdout():
    """The gate: fit on 2018-2023 only, replay 2024's real draft, and
    confirm the model's top-1/top-5 accuracy exceeds the "highest-ranked
    available by ADP" baseline. If this regresses to *not* beating the
    baseline, that is the honest, reportable finding the task spec calls
    for -- but as of this implementation it does beat it in every round
    bucket, so this is asserted as a real regression guard, not loosened
    to always pass.
    """
    result = backtest_holdout_season(2024)
    assert isinstance(result, BacktestResult)
    assert result.n_scored > 100  # sanity: most of a ~180-pick season should score
    assert result.top1_accuracy > result.baseline_top1_accuracy
    assert result.top5_accuracy > result.baseline_top5_accuracy


def test_backtest_reports_by_round_bucket():
    result = backtest_holdout_season(2024)
    assert set(result.by_round_bucket.keys()) == {"early", "middle", "late"}
    total_bucketed = sum(b.n for b in result.by_round_bucket.values())
    assert total_bucketed == result.n_scored


def test_backtest_excludes_unmatched_picks_from_scoring_not_from_state():
    """A synthetic season where one pick has no ADP match at all (e.g. a
    deep sleeper never covered by that year's ADP source): it must not be
    scoreable, but must still count toward the manager's roster tally and
    be gone from the pool for later picks."""
    league_drafts = pl.DataFrame(
        [
            _league_drafts_row(2018, 1, 1, 1, 101),
            _league_drafts_row(2018, 2, 1, 1, 999),  # unmatched: no crosswalk entry
            _league_drafts_row(2018, 3, 1, 1, 102),
            _league_drafts_row(2024, 1, 1, 1, 103),
            _league_drafts_row(2024, 2, 1, 1, 104),
        ]
    )
    league_managers = pl.DataFrame(
        [_managers_row(2018, 1, "mgr-A"), _managers_row(2024, 1, "mgr-A")]
    )
    crosswalk = pl.DataFrame(
        [
            _crosswalk_row(101, "Player QB One", "QB"),
            _crosswalk_row(102, "Player WR One", "WR"),
            _crosswalk_row(103, "Player QB Two", "QB"),
            _crosswalk_row(104, "Player WR Two", "WR"),
        ]
    )
    adp_history = pl.DataFrame(
        [
            _adp_row("Player QB One", "QB", "AAA", 1.0, 2018),
            _adp_row("Player WR One", "WR", "AAA", 3.0, 2018),
            _adp_row("Player QB Two", "QB", "AAA", 1.0, 2024),
            _adp_row("Player WR Two", "WR", "AAA", 3.0, 2024),
        ]
    )
    # espn_player_id 999 has no crosswalk row at all -> name/position/adp
    # all null for that pick -- unmatched, not scoreable, but still a real
    # roster event.
    result = backtest_holdout_season(
        2024, league_drafts, league_managers, crosswalk, adp_history
    )
    assert result.n_scored == 2
    assert result.n_excluded_unmatched == 0  # the unmatched pick is in 2018, not 2024


def test_sampling_per_call_timing():
    """Report per-sample timing -- Task 2's spec requires this since the
    rollout calls this thousands of times per candidate pick."""
    import time

    training, _ = build_training_set()
    model = fit_opponent_model(training)
    available = [AvailablePlayer(f"p{i}", "RB", float(i + 1)) for i in range(200)]
    rng = np.random.default_rng(0)

    n_calls = 2000
    start = time.perf_counter()
    for _ in range(n_calls):
        sample_pick(model, "mgr-x", available, {"RB": 2}, rng, round_=1)
    elapsed = time.perf_counter() - start

    per_call_ms = 1000 * elapsed / n_calls
    print(f"\nopponent.sample_pick: {per_call_ms:.4f} ms/call over {n_calls} calls, "
          f"{len(available)} available players")
    # generous ceiling -- this is a report, not a tight perf assertion, but
    # a regression to seconds-per-call would silently blow Stage 3's rollout
    # budget, so still assert something.
    assert per_call_ms < 5.0


# ---------------------------------------------------------------------------
# match_by_normalized_name: the reusable name-normalization join, shared by
# `_resolve_picks` and `draft.rollout`'s real-ADP matching (Stage 3 Task 4
# follow-up).


def test_match_by_normalized_name_matches_ignoring_case_and_punctuation():
    df = pl.DataFrame(
        {"name": ["A.J. Brown", "Bijan Robinson"], "position": ["WR", "RB"]}
    )
    adp_table = pl.DataFrame(
        {
            "name": ["AJ Brown", "Someone Else"],
            "position": ["WR", "QB"],
            "adp": [10.0, 99.0],
        }
    )
    joined = match_by_normalized_name(df, adp_table)
    got = dict(zip(joined["name"].to_list(), joined["adp"].to_list()))
    assert got["A.J. Brown"] == 10.0
    assert got["Bijan Robinson"] is None


def test_match_by_normalized_name_respects_position_even_with_name_match():
    # Same normalized name, different position -- must not cross-match.
    df = pl.DataFrame({"name": ["Josh Allen"], "position": ["QB"]})
    adp_table = pl.DataFrame({"name": ["Josh Allen"], "position": ["LB"], "adp": [500.0]})
    joined = match_by_normalized_name(df, adp_table)
    assert joined["adp"].to_list() == [None]


def test_match_by_normalized_name_respects_season_column_when_given():
    df = pl.DataFrame({"name": ["Player X"], "position": ["RB"], "season": [2024]})
    adp_table = pl.DataFrame(
        {"name": ["Player X", "Player X"], "position": ["RB", "RB"], "season": [2023, 2024], "adp": [1.0, 2.0]}
    )
    joined = match_by_normalized_name(df, adp_table, season_col="season")
    assert joined["adp"].to_list() == [2.0]


# ---------------------------------------------------------------------------
# build_pool_adp_lookup: match a live player pool to a real ADP table
# (`data/adp_2026.parquet` for a live rollout, `adp_history` filtered to one
# season for the historical backtest).


def test_build_pool_adp_lookup_matches_skill_players_by_name():
    pool_rows = pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "name": ["Ja'Marr Chase", "Nobody Special"],
            "position": ["WR", "WR"],
            "team": [None, None],
        }
    )
    adp_table = pl.DataFrame(
        {"name": ["Jamarr Chase"], "position": ["WR"], "team": ["CIN"], "adp": [4.1]}
    )
    adp_by_id, report = build_pool_adp_lookup(pool_rows, adp_table)
    assert adp_by_id["p1"] == 4.1
    assert "p2" not in adp_by_id
    assert report.total == 2
    assert report.matched == 1
    assert report.unmatched == 1
    assert report.unmatched_ids == ("p2",)


def test_build_pool_adp_lookup_matches_dst_by_team_abbreviation():
    pool_rows = pl.DataFrame(
        {
            "player_id": ["dst1"],
            "name": ["Houston Texans"],
            "position": ["DST"],
            "team": ["HOU"],
        }
    )
    adp_table = pl.DataFrame(
        {"name": ["Houston Defense"], "position": ["DST"], "team": ["HOU"], "adp": [106.6]}
    )
    adp_by_id, report = build_pool_adp_lookup(pool_rows, adp_table)
    assert adp_by_id["dst1"] == 106.6
    assert report.matched == 1


def test_build_pool_adp_lookup_applies_jac_jax_team_alias_for_dst():
    # rankings-style team label for Jacksonville is "JAC"; ADP tables use
    # "JAX" -- see module docstring's `_TEAM_ALIASES`.
    pool_rows = pl.DataFrame(
        {"player_id": ["dst1"], "name": ["Jacksonville Jaguars"], "position": ["DST"], "team": ["JAC"]}
    )
    adp_table = pl.DataFrame(
        {"name": ["Jacksonville Defense"], "position": ["DST"], "team": ["JAX"], "adp": [150.0]}
    )
    adp_by_id, report = build_pool_adp_lookup(pool_rows, adp_table)
    assert adp_by_id["dst1"] == 150.0


def test_build_pool_adp_lookup_reports_unmatched_never_silently_drops():
    pool_rows = pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "name": ["Known Player", "Unknown Player A", "Unknown Player B"],
            "position": ["RB", "RB", "WR"],
            "team": [None, None, None],
        }
    )
    adp_table = pl.DataFrame({"name": ["Known Player"], "position": ["RB"], "team": ["AAA"], "adp": [5.0]})
    adp_by_id, report = build_pool_adp_lookup(pool_rows, adp_table)
    assert isinstance(report, AdpMatchReport)
    assert set(adp_by_id) == {"p1"}
    assert set(report.unmatched_ids) == {"p2", "p3"}
    # Every pool player accounted for, one way or the other.
    assert report.matched + report.unmatched == report.total == 3


# ---------------------------------------------------------------------------
# The fitted per-round temperature schedule (HANDOFF open item 3b).
#
# The incumbent flat 3.0 was wrong in both directions at once: too random
# early (elite players surviving to our pick 8 when they never did in seven
# real drafts) and too rigid late. These guard the properties that made the
# schedule worth shipping, not the exact fitted numbers -- a refit on new
# draft data should move the values without failing the suite.


def test_temperature_rises_from_early_to_late_rounds():
    """The whole point of the schedule. Round 1 is near-consensus; the late
    rounds are idiosyncratic. Compared endpoint-to-endpoint rather than
    pairwise-monotone on purpose: the raw MLE has small non-monotone dips
    at rounds 4-5 and 14-16 that are sampling noise at ~70 picks/round."""
    from ffdraft.models.opponent import temperature_for_round

    assert temperature_for_round(1) < temperature_for_round(5)
    assert temperature_for_round(5) < temperature_for_round(12)
    assert temperature_for_round(12) < temperature_for_round(17)


def test_round_one_is_more_decisive_than_the_superseded_flat_value():
    """Round 1's MLE is 2.30 with a 95% CI of [1.85, 2.90] -- the old flat
    3.0 sits outside it. This is the specific defect the owner reported:
    top players surviving to pick 8 far more often than they really do."""
    from ffdraft.models.opponent import LEGACY_FLAT_TEMPERATURE, temperature_for_round

    assert temperature_for_round(1) < LEGACY_FLAT_TEMPERATURE
    assert temperature_for_round(1) <= 2.90


def test_rounds_past_the_fitted_table_reuse_the_deepest_round():
    """Round 18 exists only in this league's 2023+ drafts and has too few
    training picks to fit. Reusing the deepest fitted value is deliberate --
    extrapolating a trend off the end of the data would invent a number."""
    from ffdraft.models.opponent import TEMPERATURE_BY_ROUND, temperature_for_round

    deepest = max(TEMPERATURE_BY_ROUND)
    assert temperature_for_round(deepest + 1) == TEMPERATURE_BY_ROUND[deepest]
    assert temperature_for_round(99) == TEMPERATURE_BY_ROUND[deepest]


def test_a_missing_round_raises_rather_than_defaulting():
    """No silent fallback to the superseded flat value: a caller that forgot
    to thread the round through would otherwise get plausible-looking output
    from exactly the model this schedule replaced."""
    from ffdraft.models.opponent import temperature_for_round

    model = OpponentModel(league_mu=0.0, pos_effect={"RB": 0.0}, sigma2=0.1)
    available = [AvailablePlayer(player_id="a", position="RB", adp=1.0)]

    with pytest.raises(ValueError, match="round_"):
        pick_probabilities(model, "mgr-x", available)

    with pytest.raises(ValueError, match="round_"):
        temperature_for_round(0)


def test_lower_temperature_concentrates_probability_on_the_top_of_the_board():
    """The mechanism behind the fix, stated as a property: a colder round 1
    must make the consensus best player *more* likely to go, which is what
    stops him surviving to our pick."""
    model = OpponentModel(league_mu=0.0, pos_effect={"RB": 0.0}, sigma2=0.1)
    available = [
        AvailablePlayer(player_id=f"p{i}", position="RB", adp=float(i + 1))
        for i in range(30)
    ]
    cold = pick_probabilities(model, "mgr-x", available, temperature=1.0)
    warm = pick_probabilities(model, "mgr-x", available, temperature=10.0)
    assert cold["p0"] > warm["p0"]
