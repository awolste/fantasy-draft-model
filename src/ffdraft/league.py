"""Single source of truth for league settings.

Every scoring, roster, and playoff constant lives here. Nothing downstream
should hardcode a league rule -- if a number about this league appears in
another module, it is a bug.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


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
    return_td: float = 6.0      # kickoff/punt return TD (KRTD/PRTD)
    fumble_recovery_td: float = 6.0  # fumble recovered for TD (FTD)
    fumble_lost: float = -2.0
    two_pt: float = 2.0


SCORING: Final = ScoringRules()


@dataclass(frozen=True)
class KickingRules:
    """League kicking scoring. FG 50-59 and 60+ are both 5 -- distance past
    50 carries no extra value here, which is unusual and easy to get wrong."""

    pat_made: float = 1.0
    fg_0_39: float = 3.0
    fg_40_49: float = 4.0
    fg_50_59: float = 5.0
    fg_60_plus: float = 5.0
    fg_missed: float = -1.0


KICKING: Final = KickingRules()

# Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DST = 10
STARTERS: Final = MappingProxyType({
    "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DST": 1,
})
FLEX_ELIGIBLE: Final = frozenset({"RB", "WR", "TE"})

ROSTER_SIZE: Final = 18
BENCH_SIZE: Final = 8
IR_SLOTS: Final = 1

N_TEAMS: Final = 10
DRAFT_SLOT: Final = 8
DRAFT_ROUNDS: Final = ROSTER_SIZE

REGULAR_SEASON_WEEKS: Final = 14
PLAYOFF_TEAMS: Final = 6
PLAYOFF_ROUNDS: Final = 3
PLAYOFF_BYES: Final = 2

# Seasons of this league's history available on ESPN.
LEAGUE_SEASONS: Final = tuple(range(2018, 2026))
# Seasons of NFL stats used to fit variance and role models.
STATS_SEASONS: Final = tuple(range(2015, 2026))


def starting_slots_total() -> int:
    return sum(STARTERS.values())
