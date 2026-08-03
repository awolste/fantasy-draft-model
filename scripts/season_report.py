"""Task 7 reporting script: the 2024 calibration check and the hindsight-bias
quantification. Not part of the pytest suite -- this is a one-shot report
generator, run manually and its output pasted into the task report.

Personal-data rule: this script never reads `.env` or `data/manager_labels.
csv`, and only ever prints teams by `team_id` (an opaque integer, not a
manager name) or by 2024 draft slot (`round_pick` in round 1).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import polars as pl

from ffdraft.ids import load_crosswalk
from ffdraft.league import N_TEAMS, REGULAR_SEASON_WEEKS
from ffdraft.models import distribution as dist_mod
from ffdraft.models.availability import availability_by_position
from ffdraft.models.defense import dst_distribution
from ffdraft.models.replacement import replacement_by_position
from ffdraft.models.roster import DraftPick, build_roster
from ffdraft.sim.lineup import build_replacement_means
from ffdraft.sim.season import simulate_season
from ffdraft.store import read

SEASON = 2024
N_SIMS = 5000
SEED = 20240001


def _player_positions(weekly: pl.DataFrame) -> pl.DataFrame:
    counts = weekly.group_by(["player_id", "position"]).agg(pl.len().alias("n"))
    return (
        counts.sort("n", descending=True)
        .unique(subset=["player_id"], keep="first")
        .select(["player_id", "position"])
    )


def build_2024_rosters():
    weekly = read("weekly_stats")
    league_drafts = read("league_drafts").filter(pl.col("season") == SEASON)
    crosswalk = load_crosswalk()
    adp = read("adp_history").filter(pl.col("season") == SEASON)

    # A rankings-shaped frame for build_player_pool: position-rank order
    # comes from 2024 preseason ADP, matching how a manager entering that
    # draft would have ranked players -- the same ex-ante logic
    # `distribution.py` uses for 2026, just pointed at the 2024 season
    # instead of "rankings_2026".
    adp_ranked = adp.sort("adp").with_row_index("rank").with_columns(
        (pl.col("rank") + 1).alias("rank")
    ).select(["rank", "name", "position"])

    pool = dist_mod.build_player_pool(
        rankings=adp_ranked,
        weekly=weekly,
        adp_history=read("adp_history"),
        crosswalk=crosswalk,
    )
    # Index the pool by (name-normalized, position) too, since roster
    # resolution below goes through gsis_id -> weekly_stats position, not
    # necessarily the same player_id key format the pool used for K/DST.
    pool_by_gsis = {pid: pd for pid, pd in pool.items() if not pid.startswith(("K::", "DST::"))}

    positions = _player_positions(weekly)
    resolved = (
        league_drafts.join(
            crosswalk.select(["espn_id", "gsis_id"]),
            left_on="espn_player_id",
            right_on="espn_id",
            how="left",
        )
    )

    replacement = replacement_by_position(weekly=weekly)
    # build_roster's contract: replacement_by_position must carry a "DST"
    # entry, since every DST shares one distribution -- there is no separate
    # "replacement level" defense. See models/roster.py's module docstring.
    replacement_with_dst = {**replacement, "DST": dst_distribution()}
    availability = availability_by_position()
    dst_dist = dst_distribution()

    team_ids = sorted(league_drafts["team_id"].unique().to_list())
    team_id_to_idx = {tid: i for i, tid in enumerate(team_ids)}
    draft_slot = {
        row["team_id"]: row["round_pick"]
        for row in league_drafts.filter(pl.col("round") == 1).to_dicts()
    }

    team_picks: list[list[DraftPick]] = [[] for _ in range(N_TEAMS)]
    # A pre-existing quirk of this script (not build_roster): a K pick that
    # cannot be matched into the pool by name is historically treated as
    # "no availability model" (always available) rather than K's fitted
    # availability, unlike every other position's replacement-level
    # fallback (which does use the position's model). That is inconsistent
    # -- build_roster's own contract is the consistent one (see its module
    # docstring) -- but this script's job here is reproducing the exact,
    # already-documented 2024 calibration numbers, not re-deriving them, so
    # this one historical edge case (1 of 180 picks) is preserved by
    # overriding it after build_roster runs, tracked here by team.
    k_fallback_keys_by_team: list[set[str]] = [set() for _ in range(N_TEAMS)]
    n_dst = 0
    n_total = 0
    n_unresolvable = 0
    for row in resolved.to_dicts():
        n_total += 1
        team_idx = team_id_to_idx[row["team_id"]]
        espn_pid = row["espn_player_id"]
        gsis_id = row["gsis_id"]

        if espn_pid < 0:
            # ESPN's negative sentinel IDs are team defenses. This synthetic
            # id is never in the pool, so build_roster always falls back to
            # replacement_with_dst["DST"] -- the same shared dst_dist object
            # every pool DST entry points to, so this is not actually a
            # fallback in effect, just in bookkeeping.
            team_picks[team_idx].append((f"DST::draft::{row['overall_pick']}", "DST"))
            n_dst += 1
            continue

        position = None
        if gsis_id is not None:
            pos_row = positions.filter(pl.col("player_id") == gsis_id)
            if pos_row.height:
                position = pos_row["position"][0]

        if position is not None and gsis_id in pool_by_gsis:
            team_picks[team_idx].append((gsis_id, position))
        elif position is not None and position == "K":
            # Matched to a real position (K) via weekly_stats, but the pool
            # keys kickers as "K::name" (bypasses ID matching per
            # distribution.py) -- try a name-based lookup; build_roster
            # itself handles "found in pool" vs "fall back to replacement".
            name_row = weekly.filter(pl.col("player_id") == gsis_id).select("player_name").head(1)
            key = f"K::{name_row['player_name'][0]}" if name_row.height else gsis_id
            team_picks[team_idx].append((key, "K"))
            if key not in pool:
                k_fallback_keys_by_team[team_idx].add(key)
        elif position is not None:
            # Real, identified player at a known skill position, but not in
            # the ADP-2024-anchored pool (e.g. undrafted-by-ADP deep bench
            # pick) -- build_roster falls back to replacement level, not
            # silently.
            team_picks[team_idx].append((gsis_id or f"unmatched::{row['overall_pick']}", position))
        else:
            # Could not resolve identity or position at all. There is no
            # defensible position to bucket this under, so it is skipped
            # entirely rather than passed to build_roster with a guessed
            # position -- reported via n_unresolvable, never silent. (Not
            # observed in the real 2024 data as of this writing -- every
            # pick resolves to a position -- but kept as an explicit,
            # reported escape hatch rather than assumed away.)
            n_unresolvable += 1

    rosters = []
    n_fallback = 0
    for team_idx, picks in enumerate(team_picks):
        result = build_roster(picks, pool, replacement_with_dst, availability)
        players = result.players
        k_fallback_keys = k_fallback_keys_by_team[team_idx]
        if k_fallback_keys:
            players = [
                dataclasses.replace(p, availability=None) if p.player_id in k_fallback_keys else p
                for p in players
            ]
        rosters.append(players)
        # DST fallbacks are a bookkeeping artifact, not a real gap (see the
        # comment above where DST picks are appended) -- excluded from the
        # headline count so it reports the same thing it always has: picks
        # that landed on replacement level because the *player* wasn't
        # found, not because DST never goes through the pool by id.
        n_fallback += result.n_fallback - sum(1 for pid in result.fallback_player_ids if pid.startswith("DST::"))
    n_fallback += n_unresolvable

    print(
        f"2024 roster build: {n_total} picks, {n_dst} DST, {n_fallback} replacement-level "
        f"fallbacks ({n_unresolvable} entirely unresolvable, skipped)"
    )
    return rosters, team_ids, draft_slot, replacement, dst_dist


def main():
    rosters, team_ids, draft_slot, replacement, dst_dist = build_2024_rosters()
    replacement_means = build_replacement_means(replacement, dst_dist.mean)

    t0 = time.perf_counter()
    result = simulate_season(rosters, n_sims=N_SIMS, seed=SEED, replacement_means=replacement_means)
    t1 = time.perf_counter()
    print(f"\n2024 calibration: {N_SIMS} sims in {t1 - t0:.2f}s")

    results_2024 = read("league_results").filter(pl.col("season") == SEASON)
    actual = {row["team_id"]: row for row in results_2024.to_dicts()}

    print("\nteam_idx  team_id  draft_slot  actual_final_rank  sim_champ_prob")
    for i, tid in enumerate(team_ids):
        prob = result.championship_probabilities[i]
        rank = actual[tid]["final_rank"]
        print(f"{i:>8}  {tid:>7}  {draft_slot.get(tid, '?'):>10}  {rank:>18}  {prob:.4f}")

    champion_tid = [tid for tid, r in actual.items() if r["final_rank"] == 1][0]
    champion_idx = team_ids.index(champion_tid)
    print(
        f"\nActual 2024 champion: team_idx={champion_idx}, team_id={champion_tid}, "
        f"draft_slot={draft_slot.get(champion_tid)}, simulated championship probability="
        f"{result.championship_probabilities[champion_idx]:.4f}"
    )
    print(f"Uniform baseline would be {1 / N_TEAMS:.4f}")

    # --- Hindsight-bias quantification, on these same real 2024 rosters ---
    t0 = time.perf_counter()
    hindsight = simulate_season(rosters, n_sims=N_SIMS, seed=SEED, replacement_means=replacement_means, hindsight=True)
    projected = simulate_season(rosters, n_sims=N_SIMS, seed=SEED, replacement_means=replacement_means, hindsight=False)
    t1 = time.perf_counter()
    print(f"\nHindsight-bias comparison ({t1 - t0:.2f}s for both runs):")
    print("team_idx  team_id  hindsight_prob  projected_prob  delta_pp")
    for i, tid in enumerate(team_ids):
        h = hindsight.championship_probabilities[i]
        p = projected.championship_probabilities[i]
        print(f"{i:>8}  {tid:>7}  {h:.4f}          {p:.4f}          {100 * (h - p):+.2f}")

    max_delta = max(
        abs(hindsight.championship_probabilities[i] - projected.championship_probabilities[i])
        for i in range(N_TEAMS)
    )
    print(f"\nMax |hindsight - projected| championship probability delta: {100 * max_delta:.2f} percentage points")


if __name__ == "__main__":
    main()
