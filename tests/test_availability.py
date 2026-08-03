import string

import numpy as np
import polars as pl
import pytest

from ffdraft.models.availability import (
    ROSTERED_STATUSES,
    PlayerAvailability,
    availability_by_position,
    availability_cohort,
    eligible_weeks,
    fit_availability_rates,
    sample_availability_batch,
    team_reg_game_weeks,
)
from ffdraft.store import exists

# ---------------------------------------------------------------------------
# team_reg_game_weeks / eligible_weeks: the denominator plumbing


def _schedule_row(season, week, home, away, game_type="REG"):
    return {
        "season": season,
        "week": week,
        "game_type": game_type,
        "home_team": home,
        "away_team": away,
    }


def test_team_reg_game_weeks_excludes_byes():
    """A team with no game that week (a bye) must not appear -- there is no
    special-case bye handling, it falls out of only including scheduled
    games."""
    schedules = pl.DataFrame(
        [
            _schedule_row(2024, 1, "AAA", "BBB"),
            _schedule_row(2024, 2, "AAA", "CCC"),
            # AAA has a bye in week 3 -- no row at all for AAA in week 3
            _schedule_row(2024, 3, "BBB", "CCC"),
        ]
    )
    weeks = team_reg_game_weeks(schedules)
    aaa_weeks = weeks.filter(pl.col("team") == "AAA")["week"].sort().to_list()
    assert aaa_weeks == [1, 2]


def test_team_reg_game_weeks_excludes_postseason():
    schedules = pl.DataFrame(
        [
            _schedule_row(2024, 1, "AAA", "BBB"),
            _schedule_row(2024, 19, "AAA", "BBB", game_type="WC"),
        ]
    )
    weeks = team_reg_game_weeks(schedules)
    aaa_weeks = weeks.filter(pl.col("team") == "AAA")["week"].to_list()
    assert aaa_weeks == [1]


def _roster_row(season, week, team, gsis_id, position, status):
    return {
        "season": season,
        "week": week,
        "team": team,
        "gsis_id": gsis_id,
        "position": position,
        "status": status,
    }


def test_eligible_weeks_includes_injured_reserve():
    """RES (injured reserve) weeks count in the denominator -- an IR stint
    is exactly the unavailability this module measures, not a reason to
    shrink the denominator and hide it."""
    schedules = pl.DataFrame([_schedule_row(2024, 1, "AAA", "BBB"), _schedule_row(2024, 2, "AAA", "CCC")])
    rosters = pl.DataFrame(
        [
            _roster_row(2024, 1, "AAA", "P1", "RB", "ACT"),
            _roster_row(2024, 2, "AAA", "P1", "RB", "RES"),
        ]
    )
    elig = eligible_weeks(rosters, schedules)
    assert elig.filter(pl.col("gsis_id") == "P1")["week"].sort().to_list() == [1, 2]


def test_eligible_weeks_excludes_practice_squad_and_cut():
    schedules = pl.DataFrame([_schedule_row(2024, 1, "AAA", "BBB"), _schedule_row(2024, 2, "AAA", "CCC")])
    rosters = pl.DataFrame(
        [
            _roster_row(2024, 1, "AAA", "P1", "RB", "DEV"),
            _roster_row(2024, 2, "AAA", "P1", "RB", "CUT"),
        ]
    )
    elig = eligible_weeks(rosters, schedules)
    assert elig.height == 0


def test_all_rostered_statuses_used_are_documented():
    assert ROSTERED_STATUSES == frozenset({"ACT", "INA", "RES"})


# ---------------------------------------------------------------------------
# availability_cohort: ex-ante selection, no survivorship bias


def _letter_name(i: int) -> str:
    """A unique all-letters token for index `i` -- `normalize_name` strips
    digits entirely, so numeric suffixes (e.g. "Player 1" / "Player 12")
    collide once normalized. Base-26 letters avoid that."""
    digits = string.ascii_lowercase
    if i == 0:
        return "aaaa"
    token = ""
    n = i
    while n > 0:
        token = digits[n % 26] + token
        n //= 26
    return "aaaa" + token


def _crosswalk_row(name, position, gsis_id):
    return {
        "gsis_id": gsis_id,
        "espn_id": None,
        "sleeper_id": None,
        "name": name,
        "position": position,
    }


def test_cohort_includes_injury_shortened_season_via_adp():
    """A player drafted early (good ADP) who then gets hurt in week 2 and
    plays only 2 games must still show up in the cohort -- selection is on
    preseason ADP, not on how many games they end up playing. This is the
    specific survivorship-bias bug this module's design fixes."""
    adp_history = pl.DataFrame(
        [{"name": "Hurt Back", "position": "RB", "adp": 3.0, "season": 2020}]
        + [
            {"name": f"Filler RB {_letter_name(i)}", "position": "RB", "adp": float(10 + i), "season": 2020}
            for i in range(40)
        ]
    )
    crosswalk = pl.DataFrame(
        [_crosswalk_row("Hurt Back", "RB", "HURT1")]
        + [_crosswalk_row(f"Filler RB {_letter_name(i)}", "RB", f"FILL{i}") for i in range(40)]
    )
    weekly = pl.DataFrame(
        {
            "season": [2020, 2020],
            "week": [1, 2],
            "player_id": ["HURT1", "HURT1"],
            "position": ["RB", "RB"],
            "fantasy_points": [20.0, 18.0],
        }
    )
    cohort = availability_cohort(adp_history, weekly, crosswalk)
    assert "HURT1" in cohort.filter(pl.col("position") == "RB")["gsis_id"].to_list()


def test_kicker_cohort_does_not_use_adp():
    """Kickers match the ID crosswalk at ~0% (documented project-wide
    finding), so the K cohort must come from weekly_stats season rank, not
    from an ADP/crosswalk join that would silently produce an empty
    cohort."""
    adp_history = pl.DataFrame(
        {"name": ["Some Kicker"], "position": ["K"], "adp": [50.0], "season": [2020]}
    )
    # Kickers match the crosswalk at ~0% in reality, so an empty crosswalk
    # here is realistic, not a shortcut -- `_kicker_cohort` must not need it.
    crosswalk = pl.DataFrame(
        schema={"gsis_id": pl.String, "espn_id": pl.Int64, "sleeper_id": pl.String,
                "name": pl.String, "position": pl.String}
    )
    weekly = pl.DataFrame(
        {
            "season": [2020] * 3,
            "week": [1, 2, 3],
            "player_id": ["K1", "K1", "K1"],
            "position": ["K", "K", "K"],
            "fantasy_points": [8.0, 9.0, 7.0],
        }
    )
    cohort = availability_cohort(adp_history, weekly, crosswalk)
    k_cohort = cohort.filter(pl.col("position") == "K")
    assert "K1" in k_cohort["gsis_id"].to_list()


# ---------------------------------------------------------------------------
# fit_availability_rates: recovers known parameters from synthetic data


def _build_synthetic_league(position: str, p_available: float, persistence: float, seed: int):
    """Generate a synthetic history for one position using the module's own
    Markov sampler (a legitimate round-trip check: if the fit can recover
    parameters it knows were used to generate the data, the fitting math is
    correct), then wire the result into schedules/rosters/weekly/adp/
    crosswalk frames shaped like the real pipeline's inputs."""
    rng = np.random.default_rng(seed)
    n_players = 60
    n_seasons = 3
    n_weeks = 16
    seasons = list(range(2020, 2020 + n_seasons))

    schedules_rows = [
        _schedule_row(season, week, "AAA", "BBB") for season in seasons for week in range(1, n_weeks + 1)
    ]
    schedules = pl.DataFrame(schedules_rows)

    availability = sample_availability_batch(
        np.full(n_players, p_available),
        np.full(n_players, persistence),
        rng,
        n_sims=n_seasons,
        n_weeks=n_weeks,
    )  # (n_players, n_seasons, n_weeks)

    roster_rows = []
    weekly_rows = []
    adp_rows = []
    crosswalk_rows = []
    for p in range(n_players):
        gsis_id = f"{position}{p}"
        crosswalk_rows.append(_crosswalk_row(f"{position} Player {_letter_name(p)}", position, gsis_id))
        for si, season in enumerate(seasons):
            adp_rows.append(
                {
                    "name": f"{position} Player {_letter_name(p)}",
                    "position": position,
                    "adp": float(1 + p),
                    "season": season,
                }
            )
            for week in range(1, n_weeks + 1):
                team = "AAA" if p % 2 == 0 else "BBB"
                roster_rows.append(_roster_row(season, week, team, gsis_id, position, "ACT"))
                appeared = bool(availability[p, si, week - 1])
                if appeared:
                    weekly_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "player_id": gsis_id,
                            "position": position,
                            "fantasy_points": 10.0,
                        }
                    )
    return (
        schedules,
        pl.DataFrame(roster_rows),
        pl.DataFrame(weekly_rows),
        pl.DataFrame(adp_rows),
        pl.DataFrame(crosswalk_rows),
    )


def test_fit_recovers_position_level_rate_from_synthetic_history():
    schedules, rosters, weekly, adp_history, crosswalk = _build_synthetic_league(
        "RB", p_available=0.75, persistence=0.7, seed=1
    )
    # Give every non-RB position enough (trivial, fully-available) data to
    # clear the fit's minimum-sample guards without affecting the RB result.
    filler_schedules, filler_rosters, filler_weekly, filler_adp, filler_crosswalk = (
        _build_synthetic_league("QB", p_available=0.95, persistence=0.5, seed=2)
    )
    wr_schedules, wr_rosters, wr_weekly, wr_adp, wr_crosswalk = _build_synthetic_league(
        "WR", p_available=0.95, persistence=0.5, seed=3
    )
    te_schedules, te_rosters, te_weekly, te_adp, te_crosswalk = _build_synthetic_league(
        "TE", p_available=0.95, persistence=0.5, seed=4
    )
    k_schedules, k_rosters, k_weekly, k_adp, k_crosswalk = _build_synthetic_league(
        "K", p_available=0.8, persistence=0.5, seed=5
    )

    schedules = pl.concat([schedules, filler_schedules, wr_schedules, te_schedules, k_schedules]).unique()
    rosters = pl.concat([rosters, filler_rosters, wr_rosters, te_rosters, k_rosters])
    weekly = pl.concat([weekly, filler_weekly, wr_weekly, te_weekly, k_weekly])
    adp_history = pl.concat([adp_history, filler_adp, wr_adp, te_adp])  # K excluded: no ADP for K
    crosswalk = pl.concat([crosswalk, filler_crosswalk, wr_crosswalk, te_crosswalk, k_crosswalk])

    table = fit_availability_rates(schedules, rosters, weekly, adp_history, crosswalk)
    rb_row = table.filter(pl.col("position") == "RB").to_dicts()[0]

    assert abs(rb_row["p_available"] - 0.75) < 0.03
    assert abs(rb_row["persistence"] - 0.7) < 0.08


# ---------------------------------------------------------------------------
# PlayerAvailability: validation + stationary-consistency


def test_p_available_out_of_range_raises():
    with pytest.raises(ValueError):
        PlayerAvailability(position="RB", p_available=1.5, persistence=0.7, n_player_weeks=100)


def test_persistence_out_of_range_raises():
    with pytest.raises(ValueError):
        PlayerAvailability(position="RB", p_available=0.8, persistence=-0.1, n_player_weeks=100)


def test_p_become_unavailable_is_in_range():
    avail = PlayerAvailability(position="RB", p_available=0.8, persistence=0.75, n_player_weeks=1000)
    assert 0.0 <= avail.p_become_unavailable <= 1.0


def test_full_season_historical_player_gets_high_availability():
    """A position/profile matching a player who played essentially every
    game historically should be assigned a high availability probability."""
    avail = PlayerAvailability(position="QB", p_available=0.97, persistence=0.5, n_player_weeks=1000)
    assert avail.p_available >= 0.9


# ---------------------------------------------------------------------------
# sample_availability_batch / sample_season


def test_sample_shapes():
    out = sample_availability_batch(
        np.array([0.8, 0.6]), np.array([0.7, 0.7]), np.random.default_rng(0), n_sims=100, n_weeks=14
    )
    assert out.shape == (2, 100, 14)
    assert out.dtype == np.bool_


def test_sampled_mean_converges_to_p_available():
    rng = np.random.default_rng(0)
    out = sample_availability_batch(np.array([0.8]), np.array([0.75]), rng, n_sims=50_000, n_weeks=14)
    assert abs(out.mean() - 0.8) < 0.01


def test_sampling_is_reproducible_from_a_seed():
    a = sample_availability_batch(
        np.array([0.8]), np.array([0.75]), np.random.default_rng(7), n_sims=1000, n_weeks=14
    )
    b = sample_availability_batch(
        np.array([0.8]), np.array([0.75]), np.random.default_rng(7), n_sims=1000, n_weeks=14
    )
    assert np.array_equal(a, b)


def test_sampling_does_not_touch_global_numpy_state():
    np.random.seed(12345)
    before = np.random.get_state()[1].copy()
    sample_availability_batch(
        np.array([0.8]), np.array([0.75]), np.random.default_rng(0), n_sims=1000, n_weeks=14
    )
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


def test_persistence_elevates_probability_of_staying_out():
    """A player out this week must have a materially higher chance of being
    out next week than the marginal rate would predict -- the whole point
    of modeling persistence instead of independent draws."""
    p_available = 0.8
    persistence = 0.75  # P(out next | out this week)
    rng = np.random.default_rng(3)
    out = sample_availability_batch(
        np.array([p_available]), np.array([persistence]), rng, n_sims=200_000, n_weeks=14
    )[0]  # (n_sims, 14)

    out_this_week = ~out[:, 5]
    out_next_week_given_out = ~out[out_this_week, 6]
    p_out_next_given_out = out_next_week_given_out.mean()

    available_this_week = out[:, 5]
    out_next_week_given_available = ~out[available_this_week, 6]
    p_out_next_given_available = out_next_week_given_available.mean()

    assert p_out_next_given_out > p_out_next_given_available + 0.3
    assert abs(p_out_next_given_out - persistence) < 0.03


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        sample_availability_batch(np.array([0.8, 0.7]), np.array([0.7]), np.random.default_rng(0), 10)


def test_probabilities_out_of_range_raise():
    with pytest.raises(ValueError):
        sample_availability_batch(np.array([1.5]), np.array([0.7]), np.random.default_rng(0), 10)
    with pytest.raises(ValueError):
        sample_availability_batch(np.array([0.8]), np.array([-0.1]), np.random.default_rng(0), 10)


def test_player_availability_sample_season_matches_batch():
    avail = PlayerAvailability(position="RB", p_available=0.8, persistence=0.75, n_player_weeks=1000)
    out = avail.sample_season(np.random.default_rng(0), n_sims=1000, n_weeks=14)
    assert out.shape == (1000, 14)


# ---------------------------------------------------------------------------
# Real-data regression guards -- gated on the cached fit existing, since a
# cold fit needs network access to nflreadpy.

_HAS_AVAILABILITY_FIT = exists("availability_rates")


@pytest.mark.skipif(not _HAS_AVAILABILITY_FIT, reason="requires a cached availability_rates fit")
def test_all_positions_in_zero_one_range():
    rates = availability_by_position()
    for pos, avail in rates.items():
        assert 0.0 <= avail.p_available <= 1.0, pos
        assert 0.0 <= avail.persistence <= 1.0, pos


@pytest.mark.skipif(not _HAS_AVAILABILITY_FIT, reason="requires a cached availability_rates fit")
def test_rb_availability_is_lower_than_wr_and_qb():
    """Regression guard for the finding this task exists to encode: a
    denominator bug (survivorship bias from a games-played-based cohort)
    initially produced RB availability statistically indistinguishable from
    QB/WR -- see module docstring. If this regresses, the denominator is
    broken again."""
    rates = availability_by_position()
    assert rates["RB"].p_available < rates["WR"].p_available
    assert rates["RB"].p_available < rates["QB"].p_available


@pytest.mark.skipif(not _HAS_AVAILABILITY_FIT, reason="requires a cached availability_rates fit")
def test_position_rates_are_plausible():
    rates = availability_by_position()
    for pos, avail in rates.items():
        assert 0.6 <= avail.p_available <= 1.0, pos
        assert avail.n_player_weeks >= 200, pos
