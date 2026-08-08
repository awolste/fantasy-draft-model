"""Is the QB replacement-level miss a trend or one-season noise?

`validate_replacement_2024.py` shows QB replacement fitted at 17.96 on
2018-2023 but realized at 20.11 in 2024 (+2.15), while RB and WR are fitted
~1 ppg too HIGH. That compounds to a ~3.3 ppg error in the QB-vs-WR
comparison, in exactly the direction that overprices quarterbacks.

A pooled multi-season average is only the right estimator if the series is
stationary. This prints the per-season series so that can be checked rather
than assumed. All values are under THIS league's scoring (6-pt passing TDs).

Personal-data rule: never reads `.env` or `data/manager_labels.csv`.
"""

from __future__ import annotations

import polars as pl

from ffdraft import store
from ffdraft.models.replacement import TOP_K, drafted_player_ids, weekly_replacement_values


def main() -> None:
    weekly = store.read("weekly_stats")
    drafted = drafted_player_ids(
        store.read("league_drafts"), store.read("id_crosswalk"), weekly
    )
    values = weekly_replacement_values(weekly, drafted, top_k=TOP_K)

    by_season = (
        values.group_by(["season", "position"])
        .agg(pl.col("value").mean().alias("value"))
        .sort(["position", "season"])
    )
    seasons = sorted(by_season["season"].unique().to_list())

    print(f"replacement level by season (top_k={TOP_K}, this league's scoring)\n")
    print("pos  " + "".join(f"{s:>8}" for s in seasons))
    for pos in ("QB", "RB", "WR", "TE", "K"):
        row = {
            r["season"]: r["value"]
            for r in by_season.filter(pl.col("position") == pos).iter_rows(named=True)
        }
        cells = "".join(f"{row[s]:>8.2f}" if s in row else f"{'-':>8}" for s in seasons)
        print(f"{pos:<5}{cells}")

    print("\nfit-window comparison for each position:")
    print(f"{'pos':<5}{'mean 2018-2023':>16}{'mean 2022-2023':>16}{'2024 actual':>14}")
    for pos in ("QB", "RB", "WR", "TE", "K"):
        sub = by_season.filter(pl.col("position") == pos)
        d = {r["season"]: r["value"] for r in sub.iter_rows(named=True)}
        old = [d[s] for s in seasons if s <= 2023 and s in d]
        recent = [d[s] for s in (2022, 2023) if s in d]
        actual = d.get(2024, float("nan"))
        print(
            f"{pos:<5}{sum(old) / len(old):>16.2f}"
            f"{sum(recent) / len(recent):>16.2f}{actual:>14.2f}"
        )


if __name__ == "__main__":
    main()
