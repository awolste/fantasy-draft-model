import numpy as np
import polars as pl
import pytest

from ffdraft.models.distribution import (
    TierShape,
    ZeroInflatedGammaDistribution,
    anchor_tier_shape,
    build_player_pool,
    excluded_players,
    fit_rank_curves,
    fit_tier_shapes,
    n_tiers,
    tier_for_rank,
)

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
# ZeroInflatedGammaDistribution + anchor_tier_shape


def _shape(position="WR", tier=1, p_negative=0.0, p_zero=0.1, gamma_shape=3.0) -> TierShape:
    return TierShape(
        position=position,
        tier=tier,
        p_negative=p_negative,
        p_zero=p_zero,
        gamma_shape=gamma_shape,
        neg_shape=1.0,
        neg_scale=1.0,
        n_weeks=1000,
    )


def test_anchored_distribution_mean_matches_target():
    dist = anchor_tier_shape(_shape(), target_mean=15.0)
    assert dist.mean == 15.0


def test_sampled_mean_converges_to_stated_mean():
    dist = anchor_tier_shape(_shape(p_negative=0.05), target_mean=14.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(rng, 500_000)
    assert abs(samples.mean() - dist.mean) < 0.15


def test_samples_never_negative_when_tier_has_no_negative_mass():
    """A tier fitted with p_negative == 0 (e.g. a dominant WR1 tier, which
    in the real data essentially never has a sub-zero week) must never
    sample below zero."""
    dist = anchor_tier_shape(_shape(p_negative=0.0), target_mean=20.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(rng, 200_000)
    assert (samples >= 0).all()


def test_samples_can_be_negative_when_tier_has_negative_mass():
    """A QB tier (interceptions and fumbles can put a QB below zero) must
    actually produce negative samples, not just permit them in theory."""
    dist = anchor_tier_shape(_shape(position="QB", p_negative=0.08), target_mean=18.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(rng, 200_000)
    assert (samples < 0).any()
    # roughly matches the fitted negative probability
    assert abs((samples < 0).mean() - 0.08) < 0.01


def test_right_skewed_mean_greater_than_median_for_skill_position():
    dist = anchor_tier_shape(_shape(p_zero=0.2, gamma_shape=2.0), target_mean=12.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(rng, 200_000)
    assert samples.mean() > np.median(samples)


def test_zero_inflation_produces_zeros_at_a_plausible_rate():
    dist = anchor_tier_shape(_shape(p_zero=0.25), target_mean=10.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(rng, 200_000)
    assert abs((samples == 0.0).mean() - 0.25) < 0.01


def test_sampling_is_reproducible_from_a_seed():
    dist = anchor_tier_shape(_shape(), target_mean=10.0)
    a = dist.sample(np.random.default_rng(7), 1000)
    b = dist.sample(np.random.default_rng(7), 1000)
    assert np.array_equal(a, b)


def test_sampling_does_not_touch_global_numpy_state():
    dist = anchor_tier_shape(_shape(), target_mean=10.0)
    np.random.seed(42)
    before = np.random.get_state()[1].copy()
    dist.sample(np.random.default_rng(0), 10_000)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


def test_identical_means_different_tiers_produce_different_tails():
    """The test that proves this model does something a point projection
    cannot: a WR1-tier shape and a deep-bench WR shape, anchored to the
    *same* mean, must still show different upper-tail behavior because
    their fitted `gamma_shape` (relative skew/tail weight) differs."""
    stud_shape = _shape(tier=1, p_negative=0.0, p_zero=0.0, gamma_shape=6.0)
    boom_bust_shape = _shape(tier=4, p_negative=0.02, p_zero=0.3, gamma_shape=1.2)

    target_mean = 10.0
    stud = anchor_tier_shape(stud_shape, target_mean)
    bust = anchor_tier_shape(boom_bust_shape, target_mean)
    assert stud.mean == bust.mean == target_mean

    rng = np.random.default_rng(0)
    stud_samples = stud.sample(rng, 300_000)
    bust_samples = bust.sample(rng, 300_000)

    stud_p90 = np.quantile(stud_samples, 0.9)
    bust_p90 = np.quantile(bust_samples, 0.9)
    # same mean, but the low-gamma-shape/high-zero-inflation tier must have
    # a meaningfully higher ceiling relative to its own median.
    assert abs(stud_p90 - bust_p90) > 3.0


def test_anchor_raises_when_tier_has_no_positive_mass():
    with pytest.raises(ValueError, match="no positive-week mass"):
        anchor_tier_shape(_shape(p_negative=0.6, p_zero=0.5), target_mean=10.0)


# ---------------------------------------------------------------------------
# fit_tier_shapes / fit_rank_curves on synthetic data


def _synthetic_weekly_stats(rng: np.random.Generator) -> pl.DataFrame:
    """Small but valid `weekly_stats`-shaped frame for one position (WR),
    two rank tiers, enough seasons/players/weeks to clear the fit
    functions' internal minimum-sample thresholds."""
    rows = []
    for season in range(2020, 2024):
        # 30 players with a smoothly decreasing true mean by rank (like the
        # real rank->points relationship), so the curve fit has an actual
        # decay to recover rather than a discontinuous step. Tier 1
        # (rank 1-12) is also less zero-inflated than tier 2 (rank 13-30),
        # to exercise the tier-shape fit's p_zero split.
        for player_idx in range(30):
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


def _synthetic_adp_history(rng: np.random.Generator) -> pl.DataFrame:
    rows = []
    for season in range(2020, 2024):
        for player_idx in range(30):
            rows.append(
                {
                    "season": season,
                    "position": "WR",
                    "name": f"WR{player_idx}",
                    "adp": float(player_idx + 1 + rng.normal(0, 0.5)),
                }
            )
    return pl.DataFrame(rows)


def test_fit_tier_shapes_recovers_lower_zero_rate_in_top_tier(monkeypatch):
    import ffdraft.models.distribution as dist_mod

    monkeypatch.setattr(dist_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(1)
    weekly = _synthetic_weekly_stats(rng)
    table = fit_tier_shapes(weekly)
    rows = {r["tier"]: r for r in table.to_dicts()}
    assert rows[1]["p_zero"] < rows[2]["p_zero"]
    assert rows[1]["n_weeks"] > 0 and rows[2]["n_weeks"] > 0


def test_fit_rank_curves_is_decreasing_in_rank(monkeypatch):
    import ffdraft.models.distribution as dist_mod

    monkeypatch.setattr(dist_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(2)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)
    curves = fit_rank_curves(adp_history, weekly)
    row = curves.filter(pl.col("position") == "WR").to_dicts()[0]
    from ffdraft.models.distribution import rank_curve_from_row

    curve = rank_curve_from_row(row)
    assert curve.mean_for_rank(1) > curve.mean_for_rank(12) > curve.mean_for_rank(30)


# ---------------------------------------------------------------------------
# build_player_pool: exclusion handling


def _tiny_crosswalk() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000002", None],
            "espn_id": [1, 2, None],
            "sleeper_id": ["1", "2", None],
            "name": ["Known Player", "Another Known Player", "No Id Player"],
            "position": ["WR", "WR", "WR"],
        }
    )


def test_unmatched_skill_player_is_excluded_not_defaulted(monkeypatch, tmp_path):
    import ffdraft.store as store_mod
    import ffdraft.models.distribution as dist_mod

    # build_player_pool(force_refit=True) persists fitted params via
    # ffdraft.store.write -- redirect DATA_DIR so this test can never touch
    # the real, gitignored data/distribution_*.parquet files.
    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dist_mod, "TIER_BREAKS", {"WR": (12,)})
    rng = np.random.default_rng(3)
    weekly = _synthetic_weekly_stats(rng)
    adp_history = _synthetic_adp_history(rng)
    rankings = pl.DataFrame(
        {
            "rank": [1, 2, 3],
            "name": ["Known Player", "Totally Unknown Player", "No Id Player"],
            "position": ["WR", "WR", "WR"],
        }
    )
    pool = build_player_pool(
        rankings=rankings,
        weekly=weekly,
        adp_history=adp_history,
        crosswalk=_tiny_crosswalk(),
        force_refit=True,
    )
    names_in_pool = {p.name for p in pool.values()}
    assert "Known Player" in names_in_pool
    assert "Totally Unknown Player" not in names_in_pool
    assert "No Id Player" not in names_in_pool  # matched by name, but no usable ID

    excluded = excluded_players(rankings, crosswalk=_tiny_crosswalk())
    excluded_names = {e["name"] for e in excluded}
    assert excluded_names == {"Totally Unknown Player", "No Id Player"}


def test_kicker_bypasses_crosswalk_entirely(monkeypatch, tmp_path):
    """Kickers must be assignable even when the crosswalk has zero coverage
    for them (matching this project's real-world PK-vs-K mismatch)."""
    import ffdraft.store as store_mod
    import ffdraft.models.distribution as dist_mod

    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dist_mod, "TIER_BREAKS", {"K": (12,)})
    rng = np.random.default_rng(4)
    weekly = pl.DataFrame(
        [
            {
                "season": 2022,
                "position": "K",
                "player_id": f"K{i}",
                "player_name": f"K{i}",
                "week": w,
                "fantasy_points": round(float(rng.gamma(3.0, 2.5)), 2),
            }
            for i in range(20)
            for w in range(1, 15)
        ]
    )
    adp_history = pl.DataFrame(
        [
            {"season": 2022, "position": "K", "name": f"K{i}", "adp": float(150 + i)}
            for i in range(20)
        ]
    )
    rankings = pl.DataFrame({"rank": [1, 2], "name": ["Kicker A", "Kicker B"], "position": ["K", "K"]})
    empty_crosswalk = pl.DataFrame(
        {"gsis_id": [], "espn_id": [], "sleeper_id": [], "name": [], "position": []},
        schema={"gsis_id": pl.String, "espn_id": pl.Int64, "sleeper_id": pl.String,
                "name": pl.String, "position": pl.String},
    )
    pool = build_player_pool(
        rankings=rankings,
        weekly=weekly,
        adp_history=adp_history,
        crosswalk=empty_crosswalk,
        force_refit=True,
    )
    assert len(pool) == 2
    assert all(p.position == "K" for p in pool.values())
