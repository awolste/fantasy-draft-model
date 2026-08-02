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

**DST is a single shared distribution, not a per-team model.** Owner decision, 2026-08-02: every team's defense draws from the *same* weekly distribution. Defensive scoring is largely luck, there is little to control at the draft, and it is not what this project is trying to answer. Since all ten teams get the same DST distribution, it contributes equal expected points to every roster and cancels out of head-to-head comparisons.

This removes what would have been the stage's largest unknown — a team-game-level ingest for points-allowed and yards-allowed, which are not derivable from the player-level `weekly_stats` table. That work is now unnecessary.

**Keep the weekly variance, though.** "Same for everyone" should mean *same distribution*, not *same constant*. A fixed number would strip a genuine source of week-to-week variance out of every roster, and this project's objective is sensitive to variance — a 60% playoff rate means ceiling matters more than consistency. Draw each team's DST score independently from a shared distribution each week, so it still swings individual matchups without advantaging anyone systematically.

Fitting that distribution needs only a mean and spread for a typical starting defense, which can be taken from published DST scoring history rather than a full ingest. If even that proves awkward, a normal distribution with a hand-set mean and standard deviation is acceptable here — this is deliberately the least-precise part of the model.

**Kickers stay individually modeled** (Task 1). Unlike DST, the required stats are already sitting in `weekly_stats` as distance-bucketed columns, so modeling them properly is nearly free. If it later proves noisy, the same shared-distribution treatment applies.

**Correlation is deliberately out of scope for now.** A QB and his WR1 score together; a shootout lifts both teams' skill players. Modeling that properly is a large job, and getting it wrong is worse than omitting it. Task 8 measures how much it matters before anyone builds it.

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

### Task 2: Shared DST distribution

**Files:**
- Create: `src/ffdraft/models/defense.py`
- Test: `tests/test_defense.py`

Per the owner decision above, every team's defense draws from one shared weekly distribution. No team-level stat ingest, no per-team defensive model, no scoring-rule implementation for points-allowed or yards-allowed bands.

- [ ] **Step 1: Establish the distribution parameters**

Determine a mean and spread for a typical *starting* fantasy defense in a 10-team league — meaning roughly the top 10 defenses, not the league-wide average across all 32, since only 10 are rostered as starters. Document the source of the numbers in a comment, whether fitted from data or set by hand.

As a sanity anchor: a starting DST in a league scoring like this one typically averages somewhere around 6–9 points per week, with genuinely negative weeks possible when a defense is blown out. If your parameters make negative weeks impossible, the spread is too narrow.

- [ ] **Step 2: Write the failing test**

```python
import numpy as np
import pytest

from ffdraft.models.defense import DstDistribution, dst_distribution


def test_mean_is_in_a_plausible_starting_dst_range():
    d = dst_distribution()
    assert 4.0 <= d.mean <= 12.0


def test_sampled_mean_converges_to_stated_mean():
    d = dst_distribution()
    rng = np.random.default_rng(0)
    samples = d.sample(rng, 100_000)
    assert abs(samples.mean() - d.mean) < 0.15


def test_negative_weeks_are_possible():
    """A defense that gets blown out can score below zero in this league."""
    d = dst_distribution()
    rng = np.random.default_rng(0)
    assert (d.sample(rng, 100_000) < 0).any()


def test_sampling_is_reproducible_from_a_seed():
    d = dst_distribution()
    a = d.sample(np.random.default_rng(7), 1000)
    b = d.sample(np.random.default_rng(7), 1000)
    assert np.array_equal(a, b)


def test_every_team_shares_one_distribution():
    """The whole point: no team has a DST edge over another."""
    assert dst_distribution() is dst_distribution() or (
        dst_distribution().mean == dst_distribution().mean
    )
```

- [ ] **Step 3: Implement**

Provide `dst_distribution()` returning an object satisfying the same `WeeklyDistribution` protocol used in Task 3, so `sim/` can treat DST identically to every other roster slot. A normal distribution is acceptable here — this is deliberately the least-precise part of the model, and the design note above explains why that is fine.

Do **not** add a per-team parameter, even "for future flexibility." The shared distribution is the decision; a per-team hook would invite someone to fill it with noise later and reintroduce a difference that the owner explicitly judged uninteresting.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/pytest tests/test_defense.py -v` → 5 passed.

```bash
git add src/ffdraft/models/defense.py tests/test_defense.py
git commit -m "feat: shared DST weekly distribution"
```

**Note for whoever reads this later:** the league's full DST scoring rules — the points-allowed and yards-allowed bands, sacks, turnovers, return touchdowns — are recorded in the spec. They are deliberately *not* implemented. If a future stage ever wants per-team defensive modeling, the rules are written down and the missing piece is a team-game-level ingest, which is the work this decision avoided.

---

### Task 3: Weekly points distributions

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

### Task 4: Availability model

**Files:**
- Create: `src/ffdraft/models/availability.py`
- Test: `tests/test_availability.py`

Games missed is among the largest real sources of season variance and is routinely ignored. Model per-player weekly availability as a probability, fitted from historical games-played rates by position and age where available.

Tests: availability is always in [0, 1]; a player with a full historical record gets a high value; the position-level means match the historical rates they were fitted from; and sampling over a 14-week season produces a games-played distribution matching history.

Report historical games-played rates by position. RB availability should be visibly worse than WR or QB — if it is not, the fit is wrong.

---

### Task 5: Replacement level

**Files:**
- Create: `src/ffdraft/models/replacement.py`
- Test: `tests/test_replacement.py`

What is actually startable off waivers in a 10-team, 18-roster league — 180 players rostered, so replacement level is *ordinary*, not rich. Derive weekly replacement-level scoring by position from historical data at that roster depth.

Tests: replacement level is positive for every position; it is below the median rostered starter at that position; and it varies by week rather than being a single constant.

Report replacement level by position. This number drives whether late-round RB strategies work, so it will be scrutinized in Stage 3.

---

### Task 6: Weekly lineup optimizer

**Files:**
- Create: `src/ffdraft/sim/__init__.py`, `src/ffdraft/sim/lineup.py`
- Test: `tests/test_lineup.py`

Given a roster and each player's realized weekly score, fill 1QB/2RB/2WR/1TE/2FLEX/1K/1DST to maximize total points. FLEX takes RB/WR/TE.

This is a small assignment problem; a greedy fill is **not** always optimal once FLEX is involved. Implement it correctly and prove it.

Tests must include: the standard case; a case where greedy-by-position gives the wrong answer and the optimizer beats it; an injured/unavailable player being excluded; a roster too thin to fill every slot (must score the empty slot as zero, not crash); and that FLEX never takes a QB, K, or DST.

Performance matters — this runs inside the innermost simulation loop. Report the per-call timing and confirm it is fast enough for the budget in Task 7.

---

### Task 7: Season simulator

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

### Task 8: Correlation impact measurement

**Files:**
- Create: `tests/test_correlation_impact.py` or a short script under `scripts/`

Player scores are correlated — a QB and his top receiver rise together, and a shootout lifts both teams. Task 7 treats players as independent, which understates the variance of a stacked roster.

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
- Task 8's correlation finding is recorded, with a recommendation for Stage 3.
