import numpy as np
import polars as pl
import pytest

from ffdraft.models.tier_shape import (
    fit_tier_shapes,
    load_or_fit_tier_shapes,
    n_tiers,
    tier_for_rank,
    tier_shape_from_row,
)
from ffdraft.store import CacheStaleError

# ---------------------------------------------------------------------------
# tier_for_rank / n_tiers


def test_tier_for_rank_wr_boundaries():
    assert tier_for_rank("WR", 1) == 1
    assert tier_for_rank("WR", 12) == 1
    assert tier_for_rank("WR", 13) == 2
    assert tier_for_rank("WR", 24) == 2
    assert tier_for_rank("WR", 25) == 3
    assert tier_for_rank("WR", 36) == 3
    assert tier_for_rank("WR", 37) == 4
    assert tier_for_rank("WR", 500) == 4


def test_n_tiers_matches_breaks_plus_one():
    assert n_tiers("QB") == 3
    assert n_tiers("RB") == 4
    assert n_tiers("K") == 2


# ---------------------------------------------------------------------------
# fit_tier_shapes on synthetic data


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
        # A smoothly decreasing true mean by rank (like the real
        # rank->points relationship), so the curve fit has an actual decay
        # to recover rather than a discontinuous step. Tier 1 (rank 1-12)
        # is also less zero-inflated than tier 2, to exercise the
        # tier-shape fit's p_zero split.
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


def test_fit_tier_shapes_recovers_lower_zero_rate_in_top_tier(monkeypatch):
    import ffdraft.models.tier_shape as tier_shape_mod

    monkeypatch.setattr(tier_shape_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(1)
    weekly = _synthetic_weekly_stats(rng)
    table = fit_tier_shapes(weekly)
    rows = {r["tier"]: r for r in table.to_dicts()}
    assert rows[1]["p_zero"] < rows[2]["p_zero"]
    assert rows[1]["n_weeks"] > 0 and rows[2]["n_weeks"] > 0


# ---------------------------------------------------------------------------
# load_or_fit_tier_shapes: cache staleness (Stage 3 Task 1 Fix 1). Caches
# fitted parameters to disk and must not serve them back once the input
# data (`weekly`) has changed out from under an existing cache file.


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    import ffdraft.store as store_mod

    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    return tmp_path


def test_load_or_fit_tier_shapes_hits_cache_when_input_unchanged(tmp_data_dir, monkeypatch):
    import ffdraft.models.tier_shape as tier_shape_mod

    monkeypatch.setattr(tier_shape_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(10)
    weekly = _synthetic_weekly_stats(rng)
    first = load_or_fit_tier_shapes(weekly)

    def _boom(*args, **kwargs):
        raise AssertionError("fit_tier_shapes should not be called on a cache hit")

    monkeypatch.setattr("ffdraft.models.tier_shape.fit_tier_shapes", _boom)
    second = load_or_fit_tier_shapes(weekly)
    assert second == first


def test_load_or_fit_tier_shapes_raises_on_changed_input(tmp_data_dir, monkeypatch):
    import ffdraft.models.tier_shape as tier_shape_mod

    monkeypatch.setattr(tier_shape_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(11)
    weekly = _synthetic_weekly_stats(rng)
    load_or_fit_tier_shapes(weekly)

    changed_weekly = _synthetic_weekly_stats(np.random.default_rng(999))
    with pytest.raises(CacheStaleError, match="force_refit"):
        load_or_fit_tier_shapes(changed_weekly)


def test_load_or_fit_tier_shapes_force_refit_overrides_stale_cache(tmp_data_dir, monkeypatch):
    import ffdraft.models.tier_shape as tier_shape_mod

    monkeypatch.setattr(tier_shape_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(12)
    weekly = _synthetic_weekly_stats(rng)
    load_or_fit_tier_shapes(weekly)

    changed_weekly = _synthetic_weekly_stats(np.random.default_rng(998))
    # force_refit=True bypasses the staleness check (and rewrites the cache
    # + fingerprint to match the new input).
    refit = load_or_fit_tier_shapes(changed_weekly, force_refit=True)
    expected = {
        (r["position"], r["tier"]): tier_shape_from_row(r)
        for r in fit_tier_shapes(changed_weekly).to_dicts()
    }
    assert refit == expected
    # subsequent unchanged calls now hit the new cache without raising.
    load_or_fit_tier_shapes(changed_weekly)
