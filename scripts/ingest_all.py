"""Run every ingest in dependency order, then validate.

Usage: .venv/bin/python scripts/ingest_all.py
"""

from ffdraft import ids, validate
from ffdraft.sources import espn_parse, fantasypros, ffcalculator, nflverse


def main() -> None:
    print("--- nflverse: weekly_stats ---")
    nflverse.ingest()

    print("--- ids: id_crosswalk, crosswalk_collisions ---")
    ids.ingest()

    print("--- espn: league_drafts, league_managers, league_results ---")
    espn_parse.ingest()

    print("--- fantasypros: rankings_2026 ---")
    fantasypros.ingest()

    print("--- ffcalculator: adp_history, adp_2026 ---")
    ffcalculator.ingest()

    print("--- validate: cross-source report ---")
    validate.report()


if __name__ == "__main__":
    main()
