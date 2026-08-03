import pytest

from ffdraft.models.calibration import (
    STARTER_COHORT_TOP_N,
    modeled_starter_quantile_ratios,
    observed_starter_quantiles,
)
from ffdraft.models.distribution import build_player_pool
from ffdraft.store import exists, read

# ---------------------------------------------------------------------------
# Calibration: modeled tail behavior vs the real starter cohort.
#
# This is the acceptance criterion the plan actually specified ("judged
# primarily on upper-tail fit") and it was previously never enforced by a
# test -- a hand-run diagnostic reported a badly-inflated ratio (2.44
# modeled vs 1.54 observed for QB) that no test caught. This test would
# have failed loudly on that version: the gap was +58%, several times over
# even the loosest tolerance below.
#
# Which quantile is the fitted target vs an independent check matters here:
# `_fit_gamma_shape` solves the Gamma shape to match the empirical
# **p90**/median ratio directly (see its docstring). So a p90 assertion
# alone is close to circular -- it mostly re-confirms that rank-anchoring,
# zero-inflation, and tiering compose without distorting the ratio the fit
# already targeted, which is real coverage but can't catch a wrong tail
# *family*. p95 and p99 are NOT fit targets: nothing in the fitting
# procedure sees them, so if Gamma were the wrong family, or the tail
# diverged past what a single quantile match can fix, these would drift
# while p90 stayed locked -- and only these two would catch it.
#
# Uses real `data/weekly_stats.parquet`/`rankings_2026`/`adp_history` (not
# synthetic fixtures) because the whole point is checking the fit against
# actual NFL history -- skipped rather than failed when that data hasn't
# been ingested, consistent with this project's data/ being gitignored.
#
# Tolerances are chosen from the *observed* side's sampling noise, which is
# the binding constraint (the modeled side is drawn from a closed-form
# distribution at 500k samples/position and its own MC noise is
# negligible by comparison). A 2000-resample bootstrap of
# `observed_starter_quantiles`'s ratio90/95/99 over the real starter-cohort
# data gave these standard errors (as a fraction of the ratio):
#   p90: QB/RB/WR ~2%, TE/K ~3%   -> tightest, least noisy
#   p95: ~2-3% for QB/RB/WR, ~3%  for TE/K
#   p99: QB ~2.9%, K ~3.3%, WR ~3.1%, RB ~4.6%, TE ~5.5% (worst -- see below)
# p99 is noisiest because it's estimated from very few order statistics:
# with only 900-1900 starter-weeks per position, the top 1% is ~9-19 weeks
# (TE has the fewest, 930 weeks -> ~9.3 informing p99, hence its 5.5% CV).
#
# Tolerance = roughly 6x the worst observed-side CV in that quantile's
# group, which is comfortably tight (an actual family misspecification
# should produce an error many times larger than sampling noise -- the
# pre-fix bug was ~20-30x a typical CV) while not flagging normal
# small-sample variation in a legitimately correct model:
CALIBRATION_TOLERANCE = {
    "ratio90": 0.15,  # ~2-3% observed CV -> ~6x headroom; also the fitted target
    "ratio95": 0.20,  # ~2-3% observed CV -> ~7-10x headroom; independent check
    "ratio99": 0.35,  # up to 5.5% observed CV (TE) -> ~6x headroom; independent check
}

_HAS_REAL_DATA = exists("weekly_stats") and exists("rankings_2026") and exists("adp_history")


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="requires ingested data/ (run scripts/ingest_all.py)")
def test_modeled_tail_matches_observed_starter_cohort(tmp_path, monkeypatch):
    import ffdraft.store as store_mod

    # Read real, already-ingested data first (this test's whole point is
    # checking the fit against actual NFL history), then redirect DATA_DIR
    # before calling build_player_pool -- it force-fits (no cache exists
    # for these two datasets under a fresh tmp_path) and persists via
    # `store.write`, which must never land in the real, gitignored
    # data/distribution_*.parquet files.
    weekly = read("weekly_stats")
    rankings = read("rankings_2026")
    adp_history = read("adp_history")
    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)

    observed = observed_starter_quantiles(weekly)
    pool = build_player_pool(rankings=rankings, weekly=weekly, adp_history=adp_history)
    modeled = modeled_starter_quantile_ratios(pool)

    observed_by_pos = {r["position"]: r for r in observed.to_dicts()}
    modeled_by_pos = {r["position"]: r for r in modeled.to_dicts()}

    failures = []
    for position, top_n in STARTER_COHORT_TOP_N.items():
        obs = observed_by_pos[position]
        mod = modeled_by_pos[position]
        for ratio_key, tolerance in CALIBRATION_TOLERANCE.items():
            rel_err = abs(mod[ratio_key] - obs[ratio_key]) / obs[ratio_key]
            if rel_err >= tolerance:
                failures.append(
                    f"{position} {ratio_key}: modeled={mod[ratio_key]:.3f} vs observed "
                    f"{obs[ratio_key]:.3f} (top-{top_n} starter cohort, "
                    f"n_weeks={obs['n_weeks']}) -- {rel_err:.1%} off, "
                    f"exceeds {tolerance:.0%} tolerance"
                )
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Guard against the deep-tail extrapolation bug: a player at the bottom of
# the ranked pool must project well below a mid-tier starter at the same
# position, not close to it (which is what an unconstrained power-law
# extrapolation of the ADP curve used to produce -- e.g. QB50 at 16.5 ppg,
# barely below QB12's 19.9).
#
# Thresholds are position-specific, not one blanket number, because real
# positional value curves have genuinely different depth/decay -- kickers
# in particular are much flatter than skill positions. Each threshold below
# is set with headroom above the *actual* deepest/mid-tier ratio measured
# from the current fitted pool (reported in the task writeup), so this
# fails loudly on a real regression (e.g. the pool flattening back out)
# without being brittle to small day-to-day refits of the ADP/finish data:
#   QB: actual ~0.20  -> threshold 0.5   (matches the coordinator's example)
#   RB: actual ~0.05  -> threshold 0.3
#   WR: actual ~0.10  -> threshold 0.3
#   TE: actual ~0.23  -> threshold 0.5
#   K:  actual ~0.73  -> threshold 0.85  (kickers are a genuinely flat position)
DEEP_VS_MID_TIER_MAX_RATIO = {
    "QB": 0.5,
    "RB": 0.3,
    "WR": 0.3,
    "TE": 0.5,
    "K": 0.85,
}


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="requires ingested data/ (run scripts/ingest_all.py)")
def test_deep_pool_players_project_well_below_mid_tier_starters(tmp_path, monkeypatch):
    import ffdraft.store as store_mod

    # Same hermeticity concern as test_modeled_tail_matches_observed_starter_cohort
    # above: read real data first, then redirect DATA_DIR so build_player_pool's
    # force-fit-and-persist never writes to the real data/ directory.
    rankings = read("rankings_2026")
    weekly = read("weekly_stats")
    adp_history = read("adp_history")
    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    pool = build_player_pool(rankings=rankings, weekly=weekly, adp_history=adp_history)

    by_pos: dict[str, list] = {}
    for p in pool.values():
        by_pos.setdefault(p.position, []).append(p)

    failures = []
    for position, mid_tier_rank in STARTER_COHORT_TOP_N.items():
        players = sorted(by_pos[position], key=lambda p: p.rank)
        mid_tier = players[mid_tier_rank - 1]
        deepest = players[-1]
        ratio = deepest.distribution.mean / mid_tier.distribution.mean
        threshold = DEEP_VS_MID_TIER_MAX_RATIO[position]
        if ratio >= threshold:
            failures.append(
                f"{position}: pool-depth player ({deepest.name}, mean="
                f"{deepest.distribution.mean:.2f}) is {ratio:.2f}x the mid-tier "
                f"starter ({mid_tier.name}, mean={mid_tier.distribution.mean:.2f}) "
                f"-- exceeds the {threshold} max ratio"
            )
    assert not failures, "\n".join(failures)
