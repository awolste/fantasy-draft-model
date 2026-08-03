# Opponent Model and Optimizer Implementation Plan (Stage 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recommend the best available pick at any point in the draft by simulated championship probability — and, as a byproduct, answer empirically whether 0RB, Hero RB, or 2RB:1WR wins most often from slot 8.

**Architecture:** An opponent model predicts what each leaguemate does with each pick, fitted on eight seasons of this league's real drafts. A rollout plays the remaining draft forward, sampling opponents from that model and choosing our own future picks with a fast greedy policy. Each rollout terminates in Stage 2's season simulator, which returns a champion. The recommender ranks candidate picks by resulting title equity. A backtest on held-out 2024 decides whether any of it beats following ADP.

**Tech Stack:** Python 3.11, polars, numpy, pytest. No new external dependencies expected.

---

## Read before starting

- `docs/superpowers/specs/2026-08-02-ff-draft-2026-design.md` — the design and the league configuration
- `docs/superpowers/plans/README.md` — the running list of hard-won constraints, including the Stage 2 findings section
- Stage 2's modules, especially `sim/season.py` and `models/distribution.py`

## Facts established by Stages 1–2 — rely on these, do not re-derive

- **Realistic edge is small.** In the 2024 calibration the best roster had ~17% championship probability against a 10% baseline; the actual champion drew 9.1%. A backtest edge of a few percentage points is success. **A large edge is evidence of overfitting, not skill.**
- **Correlation is immaterial** (−0.42pp on a stacked-vs-diversified gap, 0.22pp SE). The simulator stays on independent sampling. Do not implement correlation.
- **Replacement level** (points/week): QB 18.3, RB 9.6, WR 9.9, TE 8.1, K 7.9. A rostered RB starter is worth ~6.4/week over waivers and a WR ~6.5 — **nearly identical**, which is the single most relevant prior for the 0RB question.
- **ADP history covers 2018–2024 and 2026. There is no 2025 ADP from any source.** The opponent model learns behaviour *relative to that year's ADP*, so 2025 cannot contribute to reach-tendency fitting, and the backtest holds out **2024**.
- **Draft size changed**: 17 rounds (170 picks) 2018–2022, 18 rounds (180 picks) 2023–2025.
- **Of the ten managers in the 2025 league**, tenures are `[8,8,8,8,8,7,6,4,3,2]` — seven have ≥6 seasons, one has 2. Per-manager estimates must shrink toward the league prior in proportion to seasons on record.
- **Defenses are drafted earlier here than consensus ADP suggests**, because 8-man benches allow stashing. The opponent model should recover this from data without being told; if it does not, the fit is suspect.
- **Slot 8 in a 10-team snake** means picks arrive in back-to-back pairs (8 and 13, 28 and 33, …). Pairs are better planned together than sequentially.
- **Budget:** 5,000 season simulations take 0.83s. A 15-candidate evaluation at 5,000 sims each is ~12s. That is the working budget.

---

## Task 1: Prerequisite cleanup

Two independent code reviews of Stage 2 identified work that must land before Stage 3 builds on it. Do this first; every later task depends on it.

**Files:**
- Create: `src/ffdraft/models/roster.py`, `tests/test_roster.py`
- Modify: `src/ffdraft/models/distribution.py` (split), `src/ffdraft/sim/lineup.py`, `src/ffdraft/sim/season.py`, `tests/test_distribution.py`

- [ ] **Step 1: Fix the cache-staleness risk — highest priority**

`load_or_fit_tier_shapes` and `load_or_fit_rank_curves` key their cache on **file existence alone**. If `weekly_stats` or `adp_history` is re-ingested while a cache exists, every subsequent `build_player_pool()` silently serves stale fitted parameters. No warning, no error. This is the project's stated top failure mode living in the cache layer, and the underlying data has already been re-ingested several times during development.

Add a fingerprint of the input data to each cached artifact — row counts plus a content hash, or the source files' modification times and shapes; choose one and justify it. On load, compare; on mismatch, **either refit automatically or raise with a clear message**. Do not warn and continue. Add a test that a changed input invalidates the cache.

- [ ] **Step 2: Fix test hermeticity**

`test_modeled_tail_matches_observed_starter_cohort` and `test_deep_pool_players_project_well_below_mid_tier_starters` call `build_player_pool()` against real data without redirecting `store.DATA_DIR`. On a machine without existing caches, running the suite writes real `data/distribution_*.parquet` artifacts as a side effect. Redirect `DATA_DIR` as the other tests already do.

- [ ] **Step 3: Promote roster construction into the library**

This is the structural blocker. "Turn a list of draft picks into simulator-ready roster players" exists only as ad hoc, untested logic in `scripts/season_report.py` (`build_2024_rosters`) and `scripts/correlation_impact.py` (`_make_roster_players`). Stage 3's rollout performs this operation thousands of times.

Create `src/ffdraft/models/roster.py` with a tested function along the lines of:

```python
def build_roster(
    picks: Sequence[str],
    pool: dict[str, PlayerDistribution],
    replacement_by_position: dict[str, WeeklyDistribution],
) -> list[SeasonRosterPlayer]:
```

It must: look up each pick in the pool; fall back to that position's replacement distribution when a pick is absent; and handle a **partial** roster, since mid-draft rosters are the normal case in a rollout. Migrate both scripts to use it and delete their local copies.

Note the simulation core already tolerates partial rosters — `solve_lineup` fills unfillable slots from replacement level — so only this construction step is missing.

- [ ] **Step 4: Split `distribution.py`**

It is 892 lines doing five jobs. A reviewer verified this split is cycle-free (nothing in `models/` or `sim/` currently imports it):

- `models/tier_shape.py` — `TIER_BREAKS`, `tier_for_rank`, `n_tiers`, `TierShape`, `_fit_gamma_shape`, `_demeaned_positive_ratios`, `_fit_negative_gamma`, `_season_position_rank`, `fit_tier_shapes`, `load_or_fit_tier_shapes`, `tier_shape_from_row`
- `models/rank_curve.py` — `_decay_curve`, `RankCurve`, `fit_rank_curves`, `load_or_fit_rank_curves`, `DEEP_TAIL_N_BINS`, `MIN_PROJECTED_MEAN`
- `models/distribution.py` (slimmed, ~250 lines) — `ZeroInflatedGammaDistribution`, `anchor_tier_shape`, `PlayerDistribution`, `build_player_pool`, `excluded_players`. **This is the module Stage 3 imports.**
- `models/calibration.py` — `observed_starter_quantiles`, `modeled_starter_quantile_ratios`, `STARTER_COHORT_*`

Also make `load_or_fit_tier_shapes` return hydrated `TierShape` objects rather than a raw DataFrame, matching `load_or_fit_rank_curves`. The asymmetry is accidental.

Split the test file along the same lines.

- [ ] **Step 5: Deduplicate `_validate_replacement_means`**

Near-identical in `sim/lineup.py:145` and `sim/season.py:288`. One implementation, imported by the other.

- [ ] **Step 6: Verify and commit**

Full suite must stay green. Commit each step separately.

---

## Task 2: Opponent draft model

**Files:** `src/ffdraft/models/opponent.py`, `tests/test_opponent.py`

Predict `P(manager m drafts player p with pick n | players still available)`. This is where eight seasons of league history earns its place.

- [ ] **Step 1: Build the training set**

Join `league_drafts` → `league_managers` (by season and team id, to get manager SWID) → `id_crosswalk` → `adp_history` for that season. Each row becomes: manager, season, overall pick number, the player taken, and that player's ADP that year.

**Seasons 2018–2024 only** — 2025 has no ADP, so it cannot contribute to ADP-relative fitting. Report how many usable pick observations result, and per manager.

- [ ] **Step 2: Fit reach tendency with shrinkage**

For each manager, estimate how far from ADP they pick — the mean and spread of (ADP rank − pick number), plus positional bias (do they take RBs earlier than ADP implies? QBs? defenses?).

**Shrink each manager's estimate toward the league-wide prior in proportion to their number of observed drafts.** A manager with 2 seasons must come out close to league-average; one with 8 mostly themselves. Use a standard shrinkage estimator and state the form.

- [ ] **Step 3: Validate against the known tendency**

The league is known to draft **defenses earlier than consensus ADP**. The model must recover this from data without being told. Report the fitted positional bias for DST. If it does not show defenses going early, the fit is wrong — investigate rather than proceeding.

- [ ] **Step 4: Convert to a sampling distribution**

Given a pick number, a manager, and the set of available players, produce a probability distribution over who is taken. A softmax over (ADP rank adjusted by that manager's reach and positional bias), restricted to available players, is a reasonable form — justify whatever you choose.

It must also respect roster need at least crudely: a manager with three quarterbacks rostered is unlikely to take a fourth. Decide how much roster-awareness to include and justify it; too much invites overfitting on thin per-manager data.

- [ ] **Step 5: Backtest the opponent model on its own terms**

Hold out one season. For each pick in that season, ask the model to predict, and report **top-1 and top-5 accuracy** against what was actually taken, versus a baseline of "take the highest available by ADP."

**If the model does not beat the ADP baseline, it adds nothing** and the rollout should use plain ADP with noise. Report this honestly — a negative result here is important and must not be buried.

---

## Task 3: Roster-aware value and candidate generation

**Files:** `src/ffdraft/draft/value.py`, `tests/test_value.py`

A fast, greedy value function. Two uses: pruning ~500 available players to ~15 candidates worth full evaluation, and choosing our own future picks inside a rollout, where a nested Monte Carlo would be far too expensive.

- [ ] Compute each available player's marginal contribution to the projected starting lineup given the roster so far, over that position's replacement level.
- [ ] Account for the FLEX slots — a third good RB has real value here, and a fourth quarterback has almost none.
- [ ] Tests: an empty roster values the best available highest; a roster already deep at a position values that position lower; positions that cannot start more players are valued at roughly replacement.
- [ ] Report the top 15 candidates from an empty roster at pick 8 and confirm they are plausible.

**This function is called inside every rollout, so it must be fast.** Report per-call timing.

---

## Task 4: Draft rollout

**Files:** `src/ffdraft/draft/rollout.py`, `tests/test_rollout.py`

Play the remaining draft forward from any state: opponents sampled from Task 2's model, our own picks chosen by Task 3's greedy value, until all rosters are full.

- [ ] Represent draft state explicitly: picks made, whose turn, rosters by team. Snake order from `league.py`, never hardcoded.
- [ ] Handle the **17 vs 18 round** difference when replaying historical drafts.
- [ ] Tests: a completed rollout fills every roster to the correct size; no player is drafted twice; snake order is correct including the turn; identical seeds reproduce identical drafts; and from slot 8 our picks land at the expected overall numbers.
- [ ] Report timing for one full rollout, and confirm the total for 15 candidates × N rollouts fits the budget.

---

## Task 5: The recommender

**Files:** `src/ffdraft/draft/recommender.py`, `tests/test_recommender.py`

For each candidate pick: roll the draft forward, build the ten rosters, simulate seasons, and record our championship probability. Recommend the argmax.

- [ ] Prune to ~15 candidates by Task 3's value function before rolling out.
- [ ] Return a **ranked list with reasons**, not a bare answer — each candidate's title equity and how it compares to the next best. A recommendation you cannot interrogate on draft day is much less useful.
- [ ] **Report the uncertainty.** With 5,000 sims, the standard error on a ~10% probability is about 0.42pp, so candidates within roughly 1pp are statistically indistinguishable. Say so in the output rather than implying false precision between near-ties.
- [ ] Tests: a clearly dominant player is recommended first; recommendations respect roster need (an eighth running back does not top the list); output is reproducible from a seed.
- [ ] Report end-to-end timing for a single recommendation at pick 8.

---

## Task 6: The backtest — the gate

**Files:** `scripts/backtest.py`

**This decides whether the project works.** Hold out 2024 entirely: fit the player, variance, and opponent models on data through 2023 only.

- [ ] Run the engine from slot 8 against 2024's actual ADP and the real opponents, producing a drafted roster.
- [ ] Score that roster on **real 2024 weekly results**, not simulated ones.
- [ ] Compare against three baselines: best-available-by-ADP, best-available-by-consensus-ranking, and random-but-positionally-legal.
- [ ] **Report results broken down by round**, not only in aggregate. Expect rounds 1–2 to largely reproduce consensus; the edge should concentrate in the round 1–3 structural choice and in rounds 3–8 where survival probability and scarcity diverge most from ADP.
- [ ] Repeat across many simulated 2024 seasons so the comparison is not one draw.

**If the engine does not beat ADP-following, it does not work.** Report that plainly if it happens. Expect a few percentage points of championship probability at most; a large edge means data leaked or the model is overfit — check that the 2024 holdout is genuinely clean, since an earlier calibration was found to be slightly in-sample.

---

## Task 7: The structure study and cheat sheet

**Files:** `scripts/structure_study.py`

The headline deliverable.

- [ ] Force the first three picks into each structure — 0RB (WR/WR/WR-ish), Hero RB (RB then WR/WR), 2RB:1WR, and any other worth testing — then let the engine draft normally from round 4.
- [ ] Simulate each to championship probability, from **slot 8 specifically**.
- [ ] **Report with error bars.** Given the ~0.42pp standard error, structures within about 1pp are indistinguishable, and saying so is more useful than declaring a winner that is noise.
- [ ] Produce a tiered draft board and if-then contingencies for the situations most likely to arise at picks 8 and 13.

Bear in mind the strong prior from Stage 2: RB and WR starters are worth nearly the same over replacement (6.4 vs 6.5 points/week). **If the study reports a large structural edge, be suspicious and investigate** before believing it.

---

## Done when

- `.venv/bin/pytest` passes.
- A recommendation can be produced for any draft state within the time budget, with title equity and uncertainty per candidate.
- The backtest has been run and its result reported honestly, whether or not it beats ADP.
- The structure study answers the 0RB / Hero RB / 2RB question from slot 8, with error bars.
