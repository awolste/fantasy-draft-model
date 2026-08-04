"""Validate the fitted replacement levels against real 2024 waiver reality.

docs/HANDOFF.md 7b: every losing draft policy shares one positional tilt
(more QB/TE, fewer RB/WR than ADP), and that tilt traces to the replacement
means -- QB replacement is fitted at 17.96 with QB30 projected at 18.01, so
the QB curve is nearly flat and elite-QB VOR prices Josh Allen 5th overall.
If QB replacement is too LOW, every QB is overpriced and the tilt follows.

This runs the *same* estimator (`weekly_replacement_values`) on 2024 alone
-- genuinely out of sample for a context fitted through 2023 -- and reports:
  1. fitted (<=2023) vs realized (2024) replacement, per position
  2. sensitivity to TOP_K, since a manager needing ONE starter may be better
     described by top-1/top-3 than by the top-5 mean
  3. how many players are actually available at each position, which is the
     mechanism: a 10-team league rosters ~18 QBs out of ~32 NFL starters

Personal-data rule: reads `league_drafts` only for its pick->player
resolution; never reads `.env` or `data/manager_labels.csv`, and never
prints a manager identity.
"""

from __future__ import annotations

import polars as pl

from ffdraft import store
from ffdraft.models.replacement import POSITIONS, drafted_player_ids, weekly_replacement_values

FIT_THROUGH = 2023
HOLDOUT = 2024


def main() -> None:
    weekly = store.read("weekly_stats")
    league_drafts = store.read("league_drafts")
    crosswalk = store.read("id_crosswalk")
    drafted = drafted_player_ids(league_drafts, crosswalk, weekly)

    print(f"=== rostered counts in this league, {HOLDOUT} draft ===")
    counts = (
        drafted.filter(pl.col("season") == HOLDOUT)
        .group_by("position")
        .agg(pl.len().alias("rostered"))
        .sort("position")
    )
    print(counts)

    for top_k in (1, 3, 5, 8):
        fitted = weekly_replacement_values(
            weekly.filter(pl.col("season") <= FIT_THROUGH),
            drafted.filter(pl.col("season") <= FIT_THROUGH),
            top_k=top_k,
        )
        realized = weekly_replacement_values(
            weekly.filter(pl.col("season") == HOLDOUT),
            drafted.filter(pl.col("season") == HOLDOUT),
            top_k=top_k,
        )
        f = fitted.group_by("position").agg(
            pl.col("value").mean().alias("fitted"),
            pl.col("n_candidates").mean().alias("n_avail"),
        )
        r = realized.group_by("position").agg(
            pl.col("value").mean().alias("realized_2024"),
            pl.col("n_candidates").mean().alias("n_avail_2024"),
        )
        joined = f.join(r, on="position", how="inner").sort("position")
        print(f"\n=== TOP_K = {top_k} ===")
        print(f"{'pos':<5}{'fitted<=2023':>14}{'realized 2024':>15}{'diff':>9}{'n_avail':>10}")
        for row in joined.iter_rows(named=True):
            print(
                f"{row['position']:<5}{row['fitted']:>14.2f}{row['realized_2024']:>15.2f}"
                f"{row['realized_2024'] - row['fitted']:>+9.2f}{row['n_avail_2024']:>10.0f}"
            )

    # What an elite QB is worth over replacement, under each TOP_K -- the
    # quantity that actually drives the draft board.
    print("\n=== what this does to elite-QB VOR (projected Josh Allen 27.55) ===")
    for top_k in (1, 3, 5, 8):
        realized = weekly_replacement_values(
            weekly.filter(pl.col("season") == HOLDOUT),
            drafted.filter(pl.col("season") == HOLDOUT),
            top_k=top_k,
        )
        by_pos = {
            row["position"]: row["value"]
            for row in realized.group_by("position")
            .agg(pl.col("value").mean().alias("value"))
            .iter_rows(named=True)
        }
        qb, rb, wr = by_pos["QB"], by_pos["RB"], by_pos["WR"]
        print(
            f"  top_k={top_k}: QB repl {qb:5.2f} -> Allen VOR {27.55 - qb:5.2f} | "
            f"RB repl {rb:5.2f} -> McCaffrey VOR {24.72 - rb:5.2f} | "
            f"WR repl {wr:5.2f} -> Hill VOR {23.83 - wr:5.2f}"
        )


if __name__ == "__main__":
    main()
