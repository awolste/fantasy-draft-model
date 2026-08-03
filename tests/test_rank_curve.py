import numpy as np
import polars as pl
import pytest

from ffdraft.models.rank_curve import (
    RankCurve,
    fit_rank_curves,
    load_or_fit_rank_curves,
)
from ffdraft.store import CacheStaleError

# ---------------------------------------------------------------------------
# Synthetic fixtures shared by fit_rank_curves / load_or_fit_rank_curves tests


def _synthetic_weekly_stats(rng: np.random.Generator, n_players: int = 45) -> pl.DataFrame:
    """Small but valid `weekly_stats`-shaped frame for one position (WR),
    two rank tiers, enough seasons/players/weeks to clear the fit
    functions' internal minimum-sample thresholds.

    Deliberately covers more players (default 45) than
    `_synthetic_adp_history` does (30) -- mirroring the real project, where
    plenty of ranked/rostered players never appear in historical ADP data
    at all, which is exactly the range `fit_rank_curves`'s deep-tail piece
    has to cover using finish rank instead.
    """
    rows = []
    for season in range(2020, 2024):
        for player_idx in range(n_players):
            player_id = f"WR{player_idx}"
            base_mean = 20.0 / (1.0 + player_idx * 0.15)
            zero_prob = 0.0 if player_idx < 12 else 0.25
            for week in range(1, 15):
                if rng.random() < zero_prob:
                    pts = 0.0
                else:
                    pts = float(rng.gamma(3.0, base_mean / 3.0))
                rows.append(
                    {
                        "season": season,
                        "position": "WR",
                        "player_id": player_id,
                        "player_name": player_id,
                        "week": week,
                        "fantasy_points": round(pts, 2),
                    }
                )
    return pl.DataFrame(rows)


def _synthetic_adp_history(rng: np.random.Generator, n_players: int = 30) -> pl.DataFrame:
    rows = []
    for season in range(2020, 2024):
        for player_idx in range(n_players):
            rows.append(
                {
                    "season": season,
                    "position": "WR",
                    "name": f"WR{player_idx}",
                    "adp": float(player_idx + 1 + rng.normal(0, 0.5)),
                }
            )
    return pl.DataFrame(rows)


def test_fit_rank_curves_is_decreasing_in_rank(monkeypatch):
    import ffdraft.models.rank_curve as rank_curve_mod

    monkeypatch.setattr(rank_curve_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(2)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)
    adp_table, deep_table = fit_rank_curves(adp_history, weekly)
    row = adp_table.filter(pl.col("position") == "WR").to_dicts()[0]
    knots = deep_table.filter(pl.col("position") == "WR").sort("knot_rank")
    curve = RankCurve(
        position="WR",
        a=row["a"],
        b=row["b"],
        c=row["c"],
        n_players=row["n_players"],
        boundary_rank=row["boundary_rank"],
        deep_knot_ranks=tuple(knots["knot_rank"].to_list()),
        deep_knot_ppg=tuple(knots["knot_ppg"].to_list()),
        max_supported_rank=row["max_supported_rank"],
    )
    # decreasing within the ADP-fit range (rank 1-30) ...
    assert curve.mean_for_rank(1) > curve.mean_for_rank(12) > curve.mean_for_rank(30)
    # ... and continues decreasing (or holding) into the deep tail (31-45),
    # which is built from finish rank rather than the ADP power law.
    assert curve.boundary_rank == 30
    assert curve.max_supported_rank >= 45
    assert curve.mean_for_rank(30) >= curve.mean_for_rank(45)


def test_deep_tail_is_continuous_at_the_adp_boundary(monkeypatch):
    """The whole point of rescaling the deep piece: no jump at the seam."""
    import ffdraft.models.rank_curve as rank_curve_mod

    monkeypatch.setattr(rank_curve_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(2)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)
    adp_table, deep_table = fit_rank_curves(adp_history, weekly)
    row = adp_table.filter(pl.col("position") == "WR").to_dicts()[0]
    knots = deep_table.filter(pl.col("position") == "WR").sort("knot_rank")
    curve = RankCurve(
        position="WR",
        a=row["a"],
        b=row["b"],
        c=row["c"],
        n_players=row["n_players"],
        boundary_rank=row["boundary_rank"],
        deep_knot_ranks=tuple(knots["knot_rank"].to_list()),
        deep_knot_ppg=tuple(knots["knot_ppg"].to_list()),
        max_supported_rank=row["max_supported_rank"],
    )
    just_inside = curve.mean_for_rank(curve.boundary_rank)
    just_outside = curve.mean_for_rank(curve.boundary_rank + 1)
    assert abs(just_inside - just_outside) / just_inside < 0.15


def test_rank_curve_raises_beyond_max_supported_rank():
    curve = RankCurve(
        position="WR",
        a=20.0,
        b=0.3,
        c=1.0,
        n_players=100,
        boundary_rank=30,
        deep_knot_ranks=(35.0, 45.0),
        deep_knot_ppg=(5.0, 2.0),
        max_supported_rank=45,
    )
    curve.mean_for_rank(45)  # does not raise
    with pytest.raises(ValueError, match="exceeds this curve's fitted range"):
        curve.mean_for_rank(46)


def test_rank_curve_never_projects_below_the_floor():
    """Even where the deep-tail data trends toward zero, a real player
    doesn't project to exactly 0 ppg."""
    curve = RankCurve(
        position="WR",
        a=20.0,
        b=0.3,
        c=1.0,
        n_players=100,
        boundary_rank=30,
        deep_knot_ranks=(35.0, 45.0),
        deep_knot_ppg=(0.2, 0.0),
        max_supported_rank=45,
    )
    assert curve.mean_for_rank(45) >= 0.5


# ---------------------------------------------------------------------------
# load_or_fit_rank_curves: cache staleness (Stage 3 Task 1 Fix 1). Caches
# fitted parameters to disk and must not serve them back once the input
# data (`adp_history`) has changed out from under an existing cache file.


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    import ffdraft.store as store_mod

    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    return tmp_path


def test_load_or_fit_rank_curves_hits_cache_when_input_unchanged(tmp_data_dir, monkeypatch):
    import ffdraft.models.rank_curve as rank_curve_mod

    monkeypatch.setattr(rank_curve_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(13)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)
    first = load_or_fit_rank_curves(adp_history, weekly)

    def _boom(*args, **kwargs):
        raise AssertionError("fit_rank_curves should not be called on a cache hit")

    monkeypatch.setattr("ffdraft.models.rank_curve.fit_rank_curves", _boom)
    second = load_or_fit_rank_curves(adp_history, weekly)
    assert set(second) == set(first)


def test_load_or_fit_rank_curves_raises_on_changed_input(tmp_data_dir, monkeypatch):
    import ffdraft.models.rank_curve as rank_curve_mod

    monkeypatch.setattr(rank_curve_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(14)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)
    load_or_fit_rank_curves(adp_history, weekly)

    changed_adp = _synthetic_adp_history(np.random.default_rng(997))
    with pytest.raises(CacheStaleError, match="force_refit"):
        load_or_fit_rank_curves(changed_adp, weekly)


# ---------------------------------------------------------------------------
# Cache namespacing (Stage 3 Task 1 Fix 2): two distinct `rankings` contexts
# (e.g. the live 2026 pool vs. a historical backtest's ADP-anchored
# rankings) must not evict each other's cache, but a genuine re-fit of the
# SAME context's inputs must still raise.


def test_two_rankings_contexts_both_stay_cached_without_evicting_each_other(tmp_data_dir, monkeypatch):
    import ffdraft.models.rank_curve as rank_curve_mod

    monkeypatch.setattr(rank_curve_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(21)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)

    rankings_a = pl.DataFrame({"rank": [1, 2, 3], "name": ["WR0", "WR1", "WR2"], "position": ["WR"] * 3})
    rankings_b = pl.DataFrame(
        {"rank": [1, 2, 3, 4, 5], "name": ["WR0", "WR1", "WR2", "WR3", "WR4"], "position": ["WR"] * 5}
    )

    curves_a = load_or_fit_rank_curves(adp_history, weekly, rankings=rankings_a)
    curves_b = load_or_fit_rank_curves(adp_history, weekly, rankings=rankings_b)

    # Fitting context B must not have evicted context A's cache -- re-reading
    # A afterwards must be a cache hit (fit_rank_curves not called again),
    # not a stale-cache raise and not a silent refit.
    def _boom(*args, **kwargs):
        raise AssertionError("fit_rank_curves should not be called on a cache hit for context A")

    monkeypatch.setattr(rank_curve_mod, "fit_rank_curves", _boom)
    curves_a_again = load_or_fit_rank_curves(adp_history, weekly, rankings=rankings_a)
    assert set(curves_a_again) == set(curves_a) == set(curves_b)


def test_changed_input_within_one_rankings_context_still_raises(tmp_data_dir, monkeypatch):
    import ffdraft.models.rank_curve as rank_curve_mod

    monkeypatch.setattr(rank_curve_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(22)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)
    rankings = pl.DataFrame({"rank": [1, 2, 3], "name": ["WR0", "WR1", "WR2"], "position": ["WR"] * 3})

    load_or_fit_rank_curves(adp_history, weekly, rankings=rankings)

    changed_adp = _synthetic_adp_history(np.random.default_rng(998))
    with pytest.raises(CacheStaleError, match="force_refit"):
        load_or_fit_rank_curves(changed_adp, weekly, rankings=rankings)
