"""Cross-source validation report -- the project's safety net.

Every failure mode this module checks is silent by nature: nothing crashes
when an ID match rate drops to 40%, a draft season is missing 10 picks, or a
name-collision dedupe quietly discards the wrong player. Each of those just
yields a smaller player pool or confidently wrong recommendations downstream.
This module makes those failures loud.

Personal-data note: `league_managers` and `league_results` carry real ESPN
account GUIDs (`manager_id`), real names, and real team names. Nothing in
this module prints, logs, or persists any of that -- only SWID-keyed counts
and tenure distributions. Player names (from nflverse/FantasyPros, not ESPN
league data) are not personal data in that sense and are reported freely,
e.g. in the collision report.
"""

from collections import Counter

import polars as pl

from .ids import _load_prepared_ids, find_collision_groups, load_crosswalk, match_by_name
from .league import LEAGUE_SEASONS, N_TEAMS
from .store import read

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K"}


class ValidationError(Exception):
    """Raised when a check fails a hard gate. Not raised for diagnostics."""


# --- Requirement 1: ID match rate -------------------------------------------


def _is_matched_expr() -> pl.Expr:
    return (
        pl.col("gsis_id").is_not_null()
        | pl.col("espn_id").is_not_null()
        | pl.col("sleeper_id").is_not_null()
    )


def match_rate(df: pl.DataFrame) -> float:
    """Fraction of rows that resolved to ANY crosswalk id (gsis/espn/sleeper).

    Deliberately not `gsis_id` alone: nflverse only assigns `gsis_id` to
    rostered players, so every incoming rookie has a null `gsis_id` by
    construction. Gating on `gsis_id` would count every rookie as unmatched
    even when the row correctly resolved via `espn_id` or `sleeper_id`.

    Plain diagnostic, no threshold -- pairs with `check_id_match_rate` for
    reporting numbers that shouldn't gate (match rate incl. DST, kicker rate).

    `df` must already carry `gsis_id`, `espn_id`, `sleeper_id` columns --
    i.e. it is the output of `ids.match_by_name`, not the raw input frame.
    """
    total = df.height
    if total == 0:
        return 1.0
    return df.filter(_is_matched_expr()).height / total


def check_id_match_rate(df: pl.DataFrame, label: str, threshold: float = 0.7) -> float:
    """Same as `match_rate`, but raises `ValidationError` below `threshold`."""
    total = df.height
    matched = df.filter(_is_matched_expr()).height
    rate = matched / total if total else 1.0
    if rate < threshold:
        sample = df.filter(~_is_matched_expr())["name"].to_list()[:10]
        raise ValidationError(
            f"{label}: id match rate {rate:.1%} ({matched}/{total}) is below "
            f"threshold {threshold:.1%}. Sample unmatched names: {sample}"
        )
    return rate


def id_coverage(df: pl.DataFrame) -> dict[str, float]:
    """Per-id-system coverage, as diagnostics -- not a pass/fail gate.

    `df` must already carry `gsis_id`, `espn_id`, `sleeper_id` (post-match).
    """
    total = df.height
    if total == 0:
        return {"gsis_id": 1.0, "espn_id": 1.0, "sleeper_id": 1.0}
    return {
        col: df.filter(pl.col(col).is_not_null()).height / total
        for col in ("gsis_id", "espn_id", "sleeper_id")
    }


# --- Requirement 2: draft completeness --------------------------------------


def check_draft_completeness(df: pl.DataFrame, label: str = "league_drafts") -> None:
    """Per season, `overall_pick` must run 1..N with no gaps/duplicates, and
    N must equal N_TEAMS * max(round) for that season. Does NOT hardcode a
    picks-per-season constant -- this league ran 170-pick drafts 2018-2022
    and 180-pick drafts 2023-2025.
    """
    for season in sorted(df["season"].unique().to_list()):
        sub = df.filter(pl.col("season") == season)
        picks = sub["overall_pick"].to_list()
        n = len(picks)
        max_round = sub["round"].max()
        expected_n = N_TEAMS * max_round
        if n != expected_n:
            raise ValidationError(
                f"{label} season {season}: {n} picks, expected "
                f"N_TEAMS({N_TEAMS}) * max(round)({max_round}) = {expected_n}"
            )
        seen = set(picks)
        duplicates = sorted({p for p in picks if picks.count(p) > 1})
        if duplicates:
            raise ValidationError(
                f"{label} season {season}: duplicate overall_pick values {duplicates}"
            )
        missing = sorted(set(range(1, n + 1)) - seen)
        if missing:
            raise ValidationError(
                f"{label} season {season}: overall_pick has gaps at {missing} "
                f"(expected a contiguous 1..{n})"
            )


# --- Requirement 5: season coverage -----------------------------------------


def check_season_coverage(df: pl.DataFrame, label: str, expected: tuple[int, ...]) -> None:
    present = set(df["season"].unique().to_list())
    missing = sorted(set(expected) - present)
    if missing:
        raise ValidationError(f"{label}: missing season(s) {missing}")


# --- Requirement 6: exactly one champion per season -------------------------


def check_exactly_one_champion(df: pl.DataFrame, seasons: tuple[int, ...]) -> None:
    """Every season must have exactly one team with `final_rank == 1`.
    Raises `ValidationError` naming the offending season and how many
    champions it actually has (0 or >1)."""
    champ_counts = (
        df.filter(pl.col("final_rank") == 1)
        .group_by("season")
        .agg(pl.len().alias("n_champions"))
    )
    for season in seasons:
        row = champ_counts.filter(pl.col("season") == season)
        n = row["n_champions"][0] if row.height else 0
        if n != 1:
            raise ValidationError(
                f"league_results season {season}: expected exactly 1 champion "
                f"(final_rank == 1), found {n}"
            )


# --- Requirement 4: collision report ----------------------------------------


def collision_review_groups(
    collisions: pl.DataFrame, positions: set[str] = FANTASY_POSITIONS
) -> list[dict]:
    """Summarize every collision group at a fantasy-relevant position.

    Each returned dict has: name_key, position, winner (dict), losers (list
    of dicts), and needs_review (bool) -- flagged True when the group's two
    highest non-null draft_year values are within 10 years of each other,
    i.e. the "retired parent vs active child" story may not hold.
    """
    fantasy = collisions.filter(pl.col("position").is_in(list(positions)))
    groups: list[dict] = []
    for (name_key, position), sub in fantasy.group_by(
        ["name_key", "position"], maintain_order=True
    ):
        rows = sub.to_dicts()
        winner = next(r for r in rows if r["is_kept"])
        losers = [r for r in rows if not r["is_kept"]]

        draft_years = sorted(
            (r["draft_year"] for r in rows if r["draft_year"] is not None),
            reverse=True,
        )
        needs_review = (
            len(draft_years) >= 2 and (draft_years[0] - draft_years[1]) <= 10
        )

        groups.append({
            "name_key": name_key,
            "position": position,
            "winner": {
                "name": winner["name"],
                "draft_year": winner["draft_year"],
                "gsis_id": winner["gsis_id"],
                "espn_id": winner["espn_id"],
                "sleeper_id": winner["sleeper_id"],
            },
            "losers": [
                {
                    "name": r["name"],
                    "draft_year": r["draft_year"],
                    "gsis_id": r["gsis_id"],
                    "espn_id": r["espn_id"],
                    "sleeper_id": r["sleeper_id"],
                }
                for r in losers
            ],
            "needs_review": needs_review,
        })
    return groups


def _print_collision_report(groups: list[dict]) -> None:
    print(f"  {len(groups)} fantasy-relevant collision groups")
    flagged = [g for g in groups if g["needs_review"]]
    print(f"  {len(flagged)} flagged for manual review (top-2 draft_year within 10 years)")
    for g in groups:
        tag = " [NEEDS REVIEW]" if g["needs_review"] else ""
        w = g["winner"]
        print(f"    - {g['position']} {w['name']!r} (kept, draft_year={w['draft_year']}){tag}")
        for loser in g["losers"]:
            print(
                f"        discarded: {loser['name']!r} (draft_year={loser['draft_year']}, "
                f"gsis_id={loser['gsis_id']}, espn_id={loser['espn_id']}, "
                f"sleeper_id={loser['sleeper_id']})"
            )


# --- report() ----------------------------------------------------------------


def report() -> None:
    """Run every check and print a health summary. Raises ValidationError on
    the first hard failure. Loud collision review is printed but never raises
    (see collision_review_groups) -- a false positive there should not halt
    ingestion.
    """
    print("=== ffdraft cross-source validation report ===")

    # weekly_stats shape and season range
    weekly = read("weekly_stats")
    seasons = weekly["season"]
    print(
        f"\nweekly_stats: {weekly.height} rows x {weekly.width} cols, "
        f"seasons {seasons.min()}-{seasons.max()}"
    )

    # league_drafts: season coverage + completeness
    drafts = read("league_drafts")
    check_season_coverage(drafts, "league_drafts", LEAGUE_SEASONS)
    check_draft_completeness(drafts, "league_drafts")
    print(
        f"league_drafts: {drafts.height} rows, seasons "
        f"{sorted(drafts['season'].unique().to_list())} all complete"
    )

    # manager tenure distribution -- counts only, never names or GUIDs
    managers = read("league_managers")
    tenure = (
        managers.group_by("manager_id")
        .agg(pl.col("season").n_unique().alias("n_seasons"))
    )
    tenure_hist = Counter(tenure["n_seasons"].to_list())
    print(f"league_managers: {tenure.height} distinct manager SWIDs")
    print("  tenure distribution (seasons -> number of managers):")
    for n_seasons in sorted(tenure_hist, reverse=True):
        print(f"    {n_seasons} season(s): {tenure_hist[n_seasons]} manager(s)")

    # exactly one champion per season
    results = read("league_results")
    check_exactly_one_champion(results, LEAGUE_SEASONS)
    print(f"league_results: exactly one champion in each of {len(LEAGUE_SEASONS)} seasons")

    # rankings and ADP match rates
    crosswalk = load_crosswalk()
    print("\nid match rates (matched = resolved to any of gsis_id/espn_id/sleeper_id):")
    for name in ("rankings_2026", "adp_2026", "adp_history"):
        df = read(name)
        # Requirement 3: DST has no nflverse id by construction, so it is
        # excluded before the gating check. The incl.-DST number is reported
        # as a diagnostic only, via plain `match_rate` (no threshold).
        with_dst_matched = match_by_name(df, crosswalk)
        rate_with_dst = match_rate(with_dst_matched)

        no_dst = df.filter(pl.col("position") != "DST")
        no_dst_matched = match_by_name(no_dst, crosswalk)
        rate_no_dst = check_id_match_rate(no_dst_matched, f"{name} (excl. DST)", threshold=0.7)

        # Kickers match poorly (thin nflverse id coverage) -- diagnostic only,
        # never gates the check.
        kickers = df.filter(pl.col("position") == "K")
        if kickers.height:
            kicker_matched = match_by_name(kickers, crosswalk)
            kicker_rate = match_rate(kicker_matched)
        else:
            kicker_rate = None

        print(
            f"  {name}: {rate_with_dst:.1%} incl. DST, {rate_no_dst:.1%} excl. DST"
            + (f", kickers {kicker_rate:.1%}" if kicker_rate is not None else "")
        )
        cov = id_coverage(no_dst_matched)
        print(
            f"    per-id coverage (excl. DST): gsis_id={cov['gsis_id']:.1%} "
            f"espn_id={cov['espn_id']:.1%} sleeper_id={cov['sleeper_id']:.1%}"
        )

    # collision report -- loud, non-blocking
    print("\ncrosswalk name/position collisions (silently discarded rows, preserved):")
    prepared = _load_prepared_ids()
    collisions = find_collision_groups(prepared)
    groups = collision_review_groups(collisions)
    _print_collision_report(groups)

    print("\n=== validation passed (see collision review above for non-blocking flags) ===")


if __name__ == "__main__":
    report()
