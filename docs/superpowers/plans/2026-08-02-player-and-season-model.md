# Player and Season Model Implementation Plan (Stage 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given ten rosters, simulate a full season thousands of times and return who won the championship — with weekly scoring distributions that reflect real variance rather than point estimates.

**Architecture:** Two layers. `models/` converts historical data into per-player weekly points *distributions* (not season projections), plus availability and waiver-replacement models. `sim/` consumes those to run a 14-week regular season with optimal weekly lineups, then the 6-team/3-round playoff bracket with two byes. Nothing in this stage knows about drafting; Stage 3 supplies rosters and asks who won.

**Tech Stack:** Python 3.11, polars, numpy (vectorized simulation), pytest. No new external dependencies expected.

---

## Read before starting

- `docs/superpowers/specs/2026-08-02-ff-draft-2026-design.md` — the design, the league configuration, and the K/DST scoring settings resolved on 2026-08-02.
- `docs/superpowers/plans/README.md` — the running list of non-obvious constraints. Several were discovered the hard way.
- Stage 1's modules: `league.py`, `scoring.py`, `store.py`, `ids.py`, `validate.py`, `sources/`.

## Design notes

**Why distributions and not projections.** The objective is championship probability, and 60% of this league makes the playoffs. Regular-season wins are therefore cheap, and the scarce thing is a deep playoff run — which rewards ceiling. A model built on expected points systematically undervalues the boom players who win brackets. Weekly fantasy scoring is right-skewed with fat tails; modeling it as normal around a mean throws away exactly the signal that matters here.

**Why K and DST are now modeled properly.** The original spec deferred them to "replacement level," but the owner supplied exact scoring rules, and a league that *starts* both cannot leave those lineup slots empty in simulation. They remain out of scope as *draft decisions* — the optimizer should still take them last — but their weekly scores must be real numbers with real variance.

**The DST data problem.** Points-allowed and yards-allowed bands are properties of a team's game, not of any player row. They cannot be derived from `weekly_stats`, which is player-level. Stage 2 needs a separate team-level ingest. Budget for this; it is the single largest unknown in the stage.

**Correlation is deliberately out of scope for now.** A QB and his WR1 score together; a shootout lifts both teams' skill players. Modeling that properly is a large job, and getting it wrong is worse than omitting it. Task 9 measures how much it matters before anyone builds it.

## File structure

```
src/ffdraft/
  models/
    kicking.py        FG/PAT scoring rules + kicker weekly distributions
    defense.py        team-defense scoring rules + DST weekly distributions
    distribution.py   per-player weekly points distributions
    availability.py   games-missed / injury model
    replacement.py    weekly waiver-wire replacement level by position
  sim/
    lineup.py         optimal weekly lineup from a roster
    season.py         14-week regular season + 6-team bracket -> champion
  sources/
    nflverse_team.py  team-game-level defensive stats (new ingest)
tests/
  test_kicking.py test_defense.py test_distribution.py test_availability.py
  test_replacement.py test_lineup.py test_season.py test_nflverse_team.py
```

---

### Task 1: Kicking scoring rules

**Files:**
- Create: `src/ffdraft/models/__init__.py`
- Create: `src/ffdraft/models/kicking.py`
- Modify: `src/ffdraft/league.py`
- Test: `tests/test_kicking.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kicking.py`:

```python
import polars as pl
import pytest

from ffdraft.league import KICKING
from ffdraft.models.kicking import KICKING_STAT_TO_RULE, score_kicking_line, add_kicking_points


def blank(**overrides) -> dict:
    line = {stat: 0.0 for stat in KICKING_STAT_TO_RULE}
    line.update(overrides)
    return line


def test_pat_worth_one():
    assert score_kicking_line(blank(pat_made=3)) == 3.0


def test_short_field_goals_worth_three():
    assert score_kicking_line(blank(fg_made_0_19=1, fg_made_20_29=1, fg_made_30_39=1)) == 9.0


def test_forty_yard_range_worth_four():
    assert score_kicking_line(blank(fg_made_40_49=2)) == 8.0


def test_fifty_and_sixty_both_worth_five():
    """Distance past 50 carries no extra value in this league."""
    assert score_kicking_line(blank(fg_made_50_59=1)) == 5.0
    assert score_kicking_line(blank(fg_made_60_=1)) == 5.0


def test_missed_field_goal_costs_one():
    assert score_kicking_line(blank(fg_missed=2)) == -2.0


def test_realistic_kicker_week():
    # 3 PAT, one 45-yarder, one 52-yarder, one miss
    line = blank(pat_made=3, fg_made_40_49=1, fg_made_50_59=1, fg_missed=1)
    assert score_kicking_line(line) == 11.0


def test_add_kicking_points_appends_column():
    df = pl.DataFrame([blank(pat_made=2) | {"player_id": "K1"}])
    out = add_kicking_points(df)
    assert out["fantasy_points"].to_list() == [2.0]


def test_missing_columns_treated_as_zero():
    df = pl.DataFrame({"player_id": ["K1"], "pat_made": [4.0]})
    assert add_kicking_points(df)["fantasy_points"].to_list() == [4.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_kicking.py -v`
Expected: FAIL with `ImportError: cannot import name 'KICKING'`

- [ ] **Step 3: Add the rules to `league.py`**

Append to `src/ffdraft/league.py`:

```python
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
```

- [ ] **Step 4: Write `src/ffdraft/models/kicking.py`**

Create `src/ffdraft/models/__init__.py` (empty) and:

```python
"""Kicker scoring under this league's rules.

nflverse reports made field goals bucketed by distance, which lines up
directly with the league's distance tiers -- except that 0-19, 20-29, and
30-39 all price at 3, and 50-59 and 60+ both price at 5.
"""

from collections.abc import Mapping

import polars as pl

from ..league import KICKING, KickingRules

KICKING_STAT_TO_RULE: dict[str, str] = {
    "pat_made": "pat_made",
    "fg_made_0_19": "fg_0_39",
    "fg_made_20_29": "fg_0_39",
    "fg_made_30_39": "fg_0_39",
    "fg_made_40_49": "fg_40_49",
    "fg_made_50_59": "fg_50_59",
    "fg_made_60_": "fg_60_plus",
    "fg_missed": "fg_missed",
}


def _validate(mapping: dict[str, str]) -> None:
    from dataclasses import fields

    valid = {f.name for f in fields(KickingRules)}
    unknown = sorted(set(mapping.values()) - valid)
    if unknown:
        raise ValueError(f"KICKING_STAT_TO_RULE references unknown rules: {unknown}")


_validate(KICKING_STAT_TO_RULE)


def score_kicking_line(line: Mapping[str, float], rules: KickingRules = KICKING) -> float:
    total = 0.0
    for stat, rule in KICKING_STAT_TO_RULE.items():
        value = line.get(stat)
        if value is None:
            continue
        total += float(value) * getattr(rules, rule)
    return round(total, 2)


def add_kicking_points(df: pl.DataFrame, rules: KickingRules = KICKING) -> pl.DataFrame:
    terms = [
        pl.col(stat).fill_null(0.0) * getattr(rules, rule)
        for stat, rule in KICKING_STAT_TO_RULE.items()
        if stat in df.columns
    ]
    if not terms:
        return df.with_columns(pl.lit(0.0).alias("fantasy_points"))
    return df.with_columns(pl.sum_horizontal(terms).round(2).alias("fantasy_points"))
```

- [ ] **Step 5: Verify the upstream column names**

The nflverse column names above (`fg_made_0_19`, `fg_made_60_`, `fg_missed`, `pat_made`) were observed in the Stage 1 schema dump but never exercised. Confirm each exists:

```bash
.venv/bin/python -c "
import nflreadpy as nfl, polars as pl
df = nfl.load_player_stats(seasons=[2024])
df = df if isinstance(df, pl.DataFrame) else df.to_polars()
want = ['pat_made','fg_made_0_19','fg_made_20_29','fg_made_30_39','fg_made_40_49','fg_made_50_59','fg_made_60_','fg_missed']
print({w: (w in df.columns) for w in want})
"
```

Any `False` means the name differs — correct `KICKING_STAT_TO_RULE`, not the test.

- [ ] **Step 6: Run tests and sanity-check against reality**

Run: `.venv/bin/pytest tests/test_kicking.py -v` → 8 passed.

Then score all 2024 kicker weeks and report the top 5 weeks and the season-long per-game mean. A top kicker week should land around 15–20 points and the positional mean around 8. If kickers are averaging 2 or 25, something is mismapped.

- [ ] **Step 7: Commit**

```bash
git add src/ffdraft/models/ src/ffdraft/league.py tests/test_kicking.py
git commit -m "feat: kicker scoring under league rules"
```

---

### Task 2: Team-defense stat ingest

**Files:**
- Create: `src/ffdraft/sources/nflverse_team.py`
- Test: `tests/test_nflverse_team.py`

This is the stage's largest unknown. DST scoring needs points allowed, yards allowed, sacks, interceptions, fumble recoveries, safeties, blocks, and defensive/return touchdowns — **per team per game**. None of that is in the player-level `weekly_stats`.

- [ ] **Step 1: Determine what nflverse actually offers**

Do this before writing any code, and report findings:

```bash
.venv/bin/python -c "
import nflreadpy as nfl, polars as pl
for fn in ['load_team_stats','load_schedules','load_pbp']:
    print('---', fn, hasattr(nfl, fn))
t = nfl.load_team_stats(seasons=[2024])
t = t if isinstance(t, pl.DataFrame) else t.to_polars()
print('team_stats shape:', t.shape)
print(sorted(t.columns))
"
```

Identify which of the required quantities are directly available and which must be derived. Points allowed is likely obtainable from `load_schedules` (final scores by game). Yards allowed is the opponent's total yards, so it may require joining a team's row to its opponent's row in the same game. Sacks/INTs/fumble recoveries are likely present as team defensive stats.

**If any required quantity cannot be obtained from `load_team_stats` + `load_schedules`, stop and report before falling back to play-by-play** — `load_pbp` is enormous and should be a deliberate decision, not a drift.

- [ ] **Step 2: Write the canonical schema and its test**

Target schema, one row per team per game:

```
season, week, team, opponent,
points_allowed, yards_allowed,
sacks, interceptions, fumble_recoveries, safeties, blocks,
defensive_tds, return_tds, two_pt_returns, one_pt_safeties
```

Write `tests/test_nflverse_team.py` against a saved fixture, asserting: canonical columns exactly; 32 teams × 17 games ≈ 544 rows for a full season; `points_allowed` for a team equals the opponent's points scored in that game; no nulls in `points_allowed` or `yards_allowed`; and that a known game matches reality (pick any 2024 game and verify the score against the schedule).

The opponent-consistency assertion is the important one — it is the check that catches a bad join, which is the likeliest failure here.

- [ ] **Step 3-5: Implement, verify, commit**

Follow the same adapter pattern as `sources/nflverse.py`: raise rather than defaulting when an expected column is absent, and use `_nflverse_compat.load_from_nflverse` for the package fallback. Write the dataset as `team_defense_stats`.

Commit: `feat: team-game defensive stat ingest`

---

### Task 3: Team-defense scoring

**Files:**
- Create: `src/ffdraft/models/defense.py`
- Modify: `src/ffdraft/league.py`
- Test: `tests/test_defense.py`

- [ ] **Step 1: Write the failing test**

The band boundaries are where bugs live. Test every boundary explicitly, including the two neutral bands:

```python
import pytest

from ffdraft.models.defense import points_allowed_points, yards_allowed_points, score_defense_line


@pytest.mark.parametrize("pa,expected", [
    (0, 5.0), (1, 4.0), (6, 4.0), (7, 3.0), (13, 3.0), (14, 1.0), (17, 1.0),
    (18, 0.0), (27, 0.0),           # neutral band -- explicitly zero, not missing
    (28, -1.0), (34, -1.0), (35, -3.0), (45, -3.0), (46, -5.0), (70, -5.0),
])
def test_points_allowed_bands(pa, expected):
    assert points_allowed_points(pa) == expected


@pytest.mark.parametrize("ya,expected", [
    (0, 5.0), (99, 5.0), (100, 3.0), (199, 3.0), (200, 2.0), (299, 2.0),
    (300, 0.0), (349, 0.0),         # neutral band
    (350, -1.0), (399, -1.0), (400, -3.0), (449, -3.0), (450, -5.0), (499, -5.0),
    (500, -6.0), (549, -6.0), (550, -7.0), (700, -7.0),
])
def test_yards_allowed_bands(ya, expected):
    assert yards_allowed_points(ya) == expected


def test_event_scoring():
    line = {
        "points_allowed": 10, "yards_allowed": 250,
        "sacks": 4, "interceptions": 2, "fumble_recoveries": 1,
        "safeties": 0, "blocks": 0, "defensive_tds": 1, "return_tds": 0,
        "two_pt_returns": 0, "one_pt_safeties": 0,
    }
    # 3 (PA) + 2 (YA) + 4 (sacks) + 4 (INT) + 2 (FR) + 6 (TD) = 21
    assert score_defense_line(line) == 21.0


def test_negative_defense_week():
    line = {
        "points_allowed": 48, "yards_allowed": 560,
        "sacks": 1, "interceptions": 0, "fumble_recoveries": 0,
        "safeties": 0, "blocks": 0, "defensive_tds": 0, "return_tds": 0,
        "two_pt_returns": 0, "one_pt_safeties": 0,
    }
    # -5 (PA) + -7 (YA) + 1 (sack) = -11
    assert score_defense_line(line) == -11.0
```

- [ ] **Step 2-4: Implement bands as explicit ordered tuples**

Encode both band tables in `league.py` as ordered `(upper_bound_inclusive, points)` tuples with a final catch-all, so no input can fall through without a score. Include the neutral bands explicitly. A lookup that returns `None` for an in-range value must raise, not default to zero — silently scoring a defense at zero would be indistinguishable from a legitimately neutral week.

- [ ] **Step 5: Sanity-check against real data**

Score all 2024 team-weeks and report: the highest-scoring DST week, the lowest, and the league-wide mean. Expect a mean around 6–8, a top week in the high 20s or 30s, and genuinely negative weeks to exist. If nothing is ever negative, the yards-allowed band is probably not wired up.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: team defense scoring with explicit neutral bands"
```

---

### Task 4: Weekly points distributions

**Files:**
- Create: `src/ffdraft/models/distribution.py`
- Test: `tests/test_distribution.py`

The core modeling task. Each 2026 player needs a sampleable weekly distribution whose **mean is anchored to consensus projections** and whose **shape is fitted from comparable historical players**.

- [ ] **Step 1: Choose the distributional form empirically, and document the choice**

Before implementing, examine real weekly scoring by position and role tier. Fit candidates — lognormal, gamma, and a zero-inflated variant — and report goodness of fit, especially in the upper tail, which is what the championship objective is sensitive to.

Report the comparison. Do not pick a form because it is convenient; pick it because the tail fits, and say how you judged that.

**Zero inflation is not optional.** A player who is inactive, injured mid-game, or simply never targeted produces a genuine zero, and those are common. A pure continuous fit will misplace them.

- [ ] **Step 2: Define the interface, and test against it**

```python
class WeeklyDistribution(Protocol):
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray: ...
    @property
    def mean(self) -> float: ...
```

Tests must assert: sampled mean converges to the stated mean within tolerance over many draws; samples are never negative for positions that cannot score negative; the distribution is right-skewed (mean > median) for skill positions; and that two players with identical means but different tiers produce different tail behavior.

That last test is the one that proves the model does something a point projection does not.

- [ ] **Step 3: Anchor means to 2026 consensus**

Join `rankings_2026` to the crosswalk via `ids.match_by_name`. **Rankings are ordinal, not point projections** — a rank must be converted to an expected weekly mean. Fit that mapping from history: for each past season, regress actual per-game fantasy points on preseason rank within position, and use the fitted curve.

Report the fitted rank→points curves per position. They should be steeply decreasing early and flat in the tail. If a curve is flat everywhere, the join is probably wrong.

- [ ] **Step 4: Handle the unmatched tail explicitly**

Roughly 8% of ranked players do not match the crosswalk, and **kickers match at 0%**. Kickers must be assigned distributions by rank directly from Task 1's scoring applied to historical kicker weeks, bypassing ID matching entirely. Defenses likewise, from Task 3.

Any skill player who cannot be matched and cannot be assigned a distribution must be **excluded from the draftable pool and reported**, never silently given a default. Print the count and a sample at build time.

- [ ] **Step 5-6: Verify and commit**

Report the top 20 players by projected weekly mean and confirm they look like a plausible 2026 top 20. Report the ratio of 90th-percentile to median weekly score by position — this quantifies how much ceiling each position offers and will be directly visible in Stage 3's structure results.

---

### Task 5: Availability model

**Files:**
- Create: `src/ffdraft/models/availability.py`
- Test: `tests/test_availability.py`

Games missed is among the largest real sources of season variance and is routinely ignored. Model per-player weekly availability as a probability, fitted from historical games-played rates by position and age where available.

Tests: availability is always in [0, 1]; a player with a full historical record gets a high value; the position-level means match the historical rates they were fitted from; and sampling over a 14-week season produces a games-played distribution matching history.

Report historical games-played rates by position. RB availability should be visibly worse than WR or QB — if it is not, the fit is wrong.

---

### Task 6: Replacement level

**Files:**
- Create: `src/ffdraft/models/replacement.py`
- Test: `tests/test_replacement.py`

What is actually startable off waivers in a 10-team, 18-roster league — 180 players rostered, so replacement level is *ordinary*, not rich. Derive weekly replacement-level scoring by position from historical data at that roster depth.

Tests: replacement level is positive for every position; it is below the median rostered starter at that position; and it varies by week rather than being a single constant.

Report replacement level by position. This number drives whether late-round RB strategies work, so it will be scrutinized in Stage 3.

---

### Task 7: Weekly lineup optimizer

**Files:**
- Create: `src/ffdraft/sim/__init__.py`, `src/ffdraft/sim/lineup.py`
- Test: `tests/test_lineup.py`

Given a roster and each player's realized weekly score, fill 1QB/2RB/2WR/1TE/2FLEX/1K/1DST to maximize total points. FLEX takes RB/WR/TE.

This is a small assignment problem; a greedy fill is **not** always optimal once FLEX is involved. Implement it correctly and prove it.

Tests must include: the standard case; a case where greedy-by-position gives the wrong answer and the optimizer beats it; an injured/unavailable player being excluded; a roster too thin to fill every slot (must score the empty slot as zero, not crash); and that FLEX never takes a QB, K, or DST.

Performance matters — this runs inside the innermost simulation loop. Report the per-call timing and confirm it is fast enough for the budget in Task 8.

---

### Task 8: Season simulator

**Files:**
- Create: `src/ffdraft/sim/season.py`
- Test: `tests/test_season.py`

Take ten rosters, simulate 14 regular-season weeks with a real schedule, apply waiver replacement for unavailable starters, seed the playoffs, run the 6-team/3-round bracket with byes for the top two, return the champion.

- [ ] Use `numpy` and vectorize across simulations. Sampling per player per week per simulation in Python loops will be far too slow for Stage 3, which needs thousands of season sims per candidate pick.
- [ ] Seed every RNG explicitly and make simulations reproducible from a seed. Non-reproducible results cannot be debugged.
- [ ] The bracket must match this league: 6 teams, 3 rounds, top 2 seeded on bye.

Tests: champion is always exactly one of the ten teams; over many simulations every team wins at least once given non-degenerate rosters; a roster of clearly superior players wins substantially more than 10%; the top-2 seeds win more often than seeds 3–6, and seeds 7–10 never win; and identical seeds produce identical results.

**Report the calibration check:** simulate the 2024 season using actual 2024 rosters from `league_drafts` and compare simulated championship probabilities against what actually happened. This will not match exactly — one season is one sample — but if the actual champion had a simulated probability near zero, the model is wrong somewhere.

Report the timing for 1,000 season simulations. Stage 3 needs roughly 15 candidate picks × a few thousand sims within a few seconds; if this is orders of magnitude off, say so now rather than discovering it in Stage 3.

---

### Task 9: Correlation impact measurement

**Files:**
- Create: `tests/test_correlation_impact.py` or a short script under `scripts/`

Player scores are correlated — a QB and his top receiver rise together, and a shootout lifts both teams. Task 8 treats players as independent, which understates the variance of a stacked roster.

Do **not** implement correlation. Measure whether it matters:

- [ ] Estimate historical within-team weekly score correlation between QB and his WR1/TE1, and between opposing skill players in the same game.
- [ ] Simulate a deliberately stacked roster and an equivalent unstacked roster under the independent model, and report the difference in championship probability.
- [ ] Report whether the omission is material at the level of precision Stage 3 needs (a few percentage points of title equity).

If it is material, that becomes a task in Stage 3's plan with real evidence behind it. If not, it stays out and we have a documented reason.

---

## Done when

- `.venv/bin/pytest` passes.
- A season can be simulated end to end from ten rosters and returns a champion, reproducibly from a seed.
- 1,000 season simulations complete within a documented time budget.
- Kickers and defenses produce real, non-zero, appropriately variable weekly scores.
- The 2024 calibration check has been run and reported.
- Task 9's correlation finding is recorded, with a recommendation for Stage 3.
