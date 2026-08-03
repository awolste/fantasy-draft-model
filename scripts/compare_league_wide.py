"""One-off comparison: current per-manager OpponentModel vs a league-wide-
only variant (manager_effect and manager_pos_effect forced to 0), replayed
against the same 2024 holdout backtest. Not part of the persisted test
suite -- a throwaway script to produce the evidence for the removal
decision. Safe to delete once the decision is recorded.
"""
from __future__ import annotations

import math

from ffdraft import store
from ffdraft.models.opponent import (
    TRAINING_SEASONS,
    AvailablePlayer,
    _round_bucket,
    build_training_set,
    fit_opponent_model,
    pick_probabilities,
    DEFAULT_TEMPERATURE,
    DEFAULT_ROSTER_DECAY,
)
from ffdraft.ids import normalize_name
import polars as pl


def backtest(holdout_season, strip_manager_terms):
    league_drafts = store.read("league_drafts")
    league_managers = store.read("league_managers")
    crosswalk = store.read("id_crosswalk")
    adp_history = store.read("adp_history")

    fit_seasons = tuple(s for s in TRAINING_SEASONS if s != holdout_season)
    training, _ = build_training_set(
        league_drafts, league_managers, crosswalk, adp_history, seasons=fit_seasons
    )
    model = fit_opponent_model(training)
    if strip_manager_terms:
        model = model.__class__(
            league_mu=model.league_mu,
            pos_effect=model.pos_effect,
            manager_effect={},
            manager_pos_effect={},
            manager_weight={},
            manager_pos_weight={},
            manager_n={},
            sigma2=model.sigma2,
            tau2_manager=model.tau2_manager,
            tau2_manager_pos=model.tau2_manager_pos,
        )

    from ffdraft.models.opponent import _resolve_picks, _POSITION_ALIASES

    season_drafts = league_drafts.filter(pl.col("season") == holdout_season).sort("overall_pick")
    resolved = _resolve_picks(season_drafts, league_managers, crosswalk, adp_history).sort(
        "overall_pick"
    )

    season_adp = adp_history.filter(pl.col("season") == holdout_season)
    pool = {}
    for row in season_adp.iter_rows(named=True):
        if row["position"] == "DST":
            key = row["team"]
        else:
            position = _POSITION_ALIASES.get(row["position"], row["position"])
            key = normalize_name(row["name"])
            row = {**row, "position": position}
        pool[key] = AvailablePlayer(player_id=key, position=row["position"], adp=row["adp"])

    roster_counts = {}
    n_scored = 0
    top1_hits = 0
    top5_hits = 0
    baseline_top1_hits = 0
    baseline_top5_hits = 0
    bucket_stats = {name: {"n": 0, "top1": 0, "top5": 0} for name in ("early", "middle", "late")}
    per_pick_correct_top1 = []  # for paired comparison

    for row in resolved.iter_rows(named=True):
        manager_id = row["manager_id"]
        player_key = row["player_key"]
        counts = roster_counts.setdefault(manager_id, {})

        available = list(pool.values())
        actual_in_pool = player_key in pool

        if available and actual_in_pool:
            probs = pick_probabilities(
                model, manager_id, available, counts, DEFAULT_TEMPERATURE, DEFAULT_ROSTER_DECAY
            )
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            top1_ids = {ranked[0][0]} if ranked else set()
            top5_ids = {pid for pid, _ in ranked[:5]}

            is_top1 = player_key in top1_ids
            is_top5 = player_key in top5_ids

            n_scored += 1
            top1_hits += int(is_top1)
            top5_hits += int(is_top5)
            per_pick_correct_top1.append((row["overall_pick"], is_top1))

            bucket = _round_bucket(row["round"])
            bucket_stats[bucket]["n"] += 1
            bucket_stats[bucket]["top1"] += int(is_top1)
            bucket_stats[bucket]["top5"] += int(is_top5)

        pool.pop(player_key, None)
        counts[row["position"]] = counts.get(row["position"], 0) + 1

    def rate(h, n):
        return h / n if n else 0.0

    return {
        "n_scored": n_scored,
        "top1": rate(top1_hits, n_scored),
        "top5": rate(top5_hits, n_scored),
        "by_bucket": {
            name: {"n": s["n"], "top1": rate(s["top1"], s["n"]), "top5": rate(s["top5"], s["n"])}
            for name, s in bucket_stats.items()
        },
        "per_pick_top1": per_pick_correct_top1,
    }


def se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


if __name__ == "__main__":
    per_manager = backtest(2024, strip_manager_terms=False)
    league_wide = backtest(2024, strip_manager_terms=True)

    print(f"n_scored: per-manager={per_manager['n_scored']}  league-wide={league_wide['n_scored']}")
    print()
    print(f"{'metric':<20}{'per-manager':>14}{'SE':>10}{'league-wide':>14}{'SE':>10}{'diff':>10}")
    n = per_manager["n_scored"]
    for label, key in (("overall top1", "top1"), ("overall top5", "top5")):
        pm, lw = per_manager[key], league_wide[key]
        print(f"{label:<20}{pm:>14.3%}{se(pm,n):>10.3%}{lw:>14.3%}{se(lw,n):>10.3%}{pm-lw:>+10.3%}")

    for bucket in ("early", "middle", "late"):
        for metric in ("top1", "top5"):
            pm_b = per_manager["by_bucket"][bucket]
            lw_b = league_wide["by_bucket"][bucket]
            pm, lw, nb = pm_b[metric], lw_b[metric], pm_b["n"]
            label = f"{bucket} {metric} (n={nb})"
            print(f"{label:<20}{pm:>14.3%}{se(pm,nb):>10.3%}{lw:>14.3%}{se(lw,nb):>10.3%}{pm-lw:>+10.3%}")

    # paired McNemar-style check on top1 disagreements
    pm_map = dict(per_manager["per_pick_top1"])
    lw_map = dict(league_wide["per_pick_top1"])
    both_right = sum(1 for k in pm_map if pm_map[k] and lw_map.get(k))
    pm_only = sum(1 for k in pm_map if pm_map[k] and not lw_map.get(k))
    lw_only = sum(1 for k in pm_map if not pm_map[k] and lw_map.get(k))
    both_wrong = sum(1 for k in pm_map if not pm_map[k] and not lw_map.get(k))
    print()
    print(f"paired top1 disagreements: per-manager-only-right={pm_only}  league-wide-only-right={lw_only}  both-right={both_right}  both-wrong={both_wrong}")
