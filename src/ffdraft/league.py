"""Single source of truth for league settings.

Every scoring, roster, and playoff constant lives here. Nothing downstream
should hardcode a league rule -- if a number about this league appears in
another module, it is a bug.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringRules:
    """Points per unit. ESPN PPR defaults except pass_td, which is 6 here."""

    pass_yd: float = 0.04       # 1 point per 25 yards
    pass_td: float = 6.0        # league-specific: ESPN default is 4
    interception: float = -2.0
    rush_yd: float = 0.1
    rush_td: float = 6.0
    rec: float = 1.0            # full PPR
    rec_yd: float = 0.1
    rec_td: float = 6.0
    fumble_lost: float = -2.0
    two_pt: float = 2.0


SCORING = ScoringRules()

# Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DST = 10
STARTERS: dict[str, int] = {
    "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DST": 1,
}
FLEX_ELIGIBLE = frozenset({"RB", "WR", "TE"})

ROSTER_SIZE = 18
BENCH_SIZE = 8
IR_SLOTS = 1

N_TEAMS = 10
DRAFT_SLOT = 8
DRAFT_ROUNDS = ROSTER_SIZE

REGULAR_SEASON_WEEKS = 14
PLAYOFF_TEAMS = 6
PLAYOFF_ROUNDS = 3
PLAYOFF_BYES = 2

# Seasons of this league's history available on ESPN.
LEAGUE_SEASONS = tuple(range(2018, 2026))
# Seasons of NFL stats used to fit variance and role models.
STATS_SEASONS = tuple(range(2015, 2026))


def starting_slots_total() -> int:
    return sum(STARTERS.values())
