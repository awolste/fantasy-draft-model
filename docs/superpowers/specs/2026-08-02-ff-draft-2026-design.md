# 2026 Fantasy Football Draft Optimizer — Design

**Date:** 2026-08-02
**Status:** Approved, pending implementation plan

## Problem

Recommend the best available pick at every point in a snake draft, conditioned on the roster already built, for one specific league and one specific draft slot. The recommendation should maximize probability of winning the league championship — not projected points, and not regular-season wins.

A secondary goal, answered as a byproduct rather than assumed as an input: which early-round roster construction (0RB, Hero RB, 2RB:1WR, or other) actually wins most often from this slot in this format.

## League configuration

All modeling is specific to these settings. They are not defaults and should never be assumed elsewhere.

| Setting | Value |
|---|---|
| Teams | 10 |
| Scoring | Full PPR, 6-point passing TDs |
| Starters (10) | 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DST |
| Roster size | 18 (10 starters, 8 bench, 1 of which is IR) |
| Draft | Snake, no keepers, **slot 8** |
| Regular season | 14 weeks |
| Playoffs | 6 teams, 3 rounds, top 2 seeds get a bye |
| Platform | ESPN |

### Consequences of these settings

Three properties of this league drive design decisions and are easy to get wrong:

**6-point passing TDs raise QB value** above what published rankings assume. Nearly all consensus rankings and ADP are built for 4-point passing TDs. We therefore compute all fantasy points ourselves from raw stat lines rather than consuming any site's precomputed fantasy totals.

**60% of teams make the playoffs**, so making the playoffs is close to the default outcome and marginal regular-season wins are worth little — *except* for reaching the top-2 bye, which reduces the required playoff wins from three to two. The objective function should therefore reward ceiling and seed-chasing over week-to-week consistency. Boom/bust players are worth more in this league than in a typical one, and any model that optimizes expected points will systematically undervalue them.

**The waiver wire is ordinary, not rich.** 10 teams × 18 spots = 180 rostered players, the same depth as a 12-team league with 15-man rosters. An earlier assumption that a 10-team league implies a shallow pool and strong waivers was wrong and is explicitly rejected here. This weakens the a-priori case for late-round-RB strategies; the simulation will decide the question on evidence.

## Approach

For each candidate pick, simulate the remainder of the draft and then the full season thousands of times, and recommend the player with the highest resulting championship probability.

This was chosen over two simpler alternatives:

- **Structure-first** (decide a positional sequence offline, then take the best player at the prescribed position) is interpretable but brittle — it cannot recognize that the plan should be abandoned when a top-5 player falls unexpectedly.
- **Roster-aware value over replacement** is the standard good-tool approach and handles positional scarcity well, but it is greedy: it optimizes the current pick with no view of the roster's shape in round 12, which is how teams end up with excellent receivers and nothing startable at running back.

Monte Carlo rollout subsumes both. Critically, it makes roster construction strategy an **output** rather than an input: we never encode a belief about 0RB versus Hero RB, we read the answer off the simulation and the draft-day recommendations are automatically consistent with it. It also natively handles the back-to-back picks at slot 8 (picks 8 and 13 each round), which are better planned as a pair than sequentially.

The cost is that the answer is only as good as the opponent model and the variance model. A sophisticated optimizer over bad inputs produces confidently wrong recommendations. Validation (below) exists to catch this.

## Architecture

Five layers with strictly downward dependencies. `sim/` does not know where data came from; `draft/` does not know how a season is simulated. The opponent model and season simulator are the components most likely to be rewritten, so they sit behind stable interfaces.

```
data/     pull and normalize raw inputs
models/   turn raw data into probability distributions
sim/      draft rollout + season simulation
draft/    the optimizer that consumes sim output
ui/       cheat sheet (stage 1), live assistant (stage 2)
```

**Stack:** Python, numpy + polars, parquet files for storage, pytest, Streamlit for the live UI. Everything runs locally. No server, no database, no hosting, nothing to maintain between seasons.

### data/

| Module | Source | Provides |
|---|---|---|
| `espn_history.py` | ESPN v3 API, `leagueHistory` + `mDraftDetail` | 8 seasons of this league (2018–2025): every draft pick, weekly rosters and lineups, final standings, champions. Requires `espn_s2` and `SWID` cookies. |
| `nfl_stats.py` | nflverse via `nfl_data_py` | Weekly raw stat lines, 2015–2025 |
| `rankings.py` | FantasyPros ECR + consensus ADP (scrape), ESPN projections (API) | 2026 consensus rankings, projections, and ADP |
| `ids.py` | nflverse ID table, Sleeper players API | Crosswalk between nflverse `gsis_id`, ESPN `playerId`, Sleeper ids, and FantasyPros names |

Sleeper has no official public ADP endpoint; it is used for player ID crosswalking only. FantasyPros consensus ADP already aggregates multiple sites, so it serves as the ADP source.

The ID crosswalk is called out as its own component because a mismatch there fails silently — it drops players from the pool or attaches projections to the wrong person, and every downstream number remains plausible while being wrong. It gets explicit tests and a match-rate threshold that raises rather than warns.

Historical stats come from nflverse rather than a fantasy site because it supplies raw stat lines, letting us apply this league's exact scoring. It is also maintained and requires no scraping.

### models/

**`player_model.py`** — a *weekly points distribution* per player, not a season projection. Mean anchored to consensus; distribution shape fitted from historical players in comparable roles. Weekly scoring is right-skewed with fat tails, and modeling it as normal around a mean understates precisely the boom outcomes that win playoff brackets. Includes an availability model, since games missed is among the largest real sources of variance and is commonly ignored.

**`opponent_model.py`** — P(manager *m* drafts player *p* at pick *n*), fitted on eight years (2018–2025) of this league's actual picks against that year's ADP. Yields a per-manager reach tendency and positional bias over a global ADP-noise term. This is where the league history earns its place — considerably more than the champion rosters do.

One league tendency is known in advance and serves as a model check: **defenses are drafted earlier here than consensus ADP suggests**, because 8-man benches leave room to stash them. The model should recover this from the data without being told. If it does not, the fit is suspect. The practical consequence is that skill-position players last longer in this league than generic ADP implies, so correct play is more patient than a stock cheat sheet would advise.

**`replacement_model.py`** — realistic weekly waiver availability at this league's roster depth, derived from historical weekly data.

### sim/

**`draft_sim.py`** rolls the remaining draft forward, sampling opponents from the opponent model and choosing our own future picks with a fast greedy value function (nested Monte Carlo is too expensive inside a rollout).

**`season_sim.py`** simulates 14 regular-season weeks with optimal weekly lineups and waiver activity for all ten teams, then the 6-team, 3-round bracket with top-2 byes. Returns the champion.

**`evaluate.py`** converts a candidate pick into a championship probability.

### draft/

**`recommender.py`** prunes to roughly the top 15 candidates by roster-aware value, rolls out each, and returns them ranked by title equity with an explanation. Target budget is a few seconds per pick, which is well within the time available on the clock.

## Deliverables

**Stage 1 — structure study and cheat sheet.** The empirical answer to 0RB vs. Hero RB vs. 2RB:1WR from slot 8, a tiered draft board, and a set of if-then contingencies for the situations most likely to arise at picks 8 and 13.

**Stage 2 — live assistant.** Streamlit app with type-ahead pick entry, recommending after every pick.

Manual pick entry is the **primary** path for the live assistant. ESPN's live draft room runs over an undocumented websocket that changes between seasons, and polling the draft endpoint during a live draft can return stale state. Auto-sync will be attempted in August when the 2026 endpoints are live and testable, as a bonus. Building on the assumption that auto-sync works is the likeliest way to arrive at draft day with nothing.

## Validation

**Backtest against 2025.** Hold out the 2025 season entirely. Every model is fitted on data through 2024 only: the player and variance models on nflverse seasons 2015–2024, the opponent model on this league's 2018–2024 drafts (seven of the eight available seasons, since 2025 is the holdout). Run the engine against 2025's actual ADP from slot 8, then score the resulting roster against real 2025 weekly results. Compare to three baselines: best-available-by-ADP, best-available-by-consensus-ranking, and random-but-positionally-legal.

**Report results broken down by round**, not only in aggregate. The expectation is that rounds 1–2 largely reproduce consensus — at pick 8 there are only a few defensible choices and the tool should agree with them — while the measurable edge concentrates in the round 1–3 structural decision and in rounds 3–8, where survival probability and positional scarcity diverge most from raw ADP. If the aggregate edge turns out to come from somewhere else entirely, that is worth understanding before trusting it.

**If the engine does not beat ADP-following, it does not work.** This gate exists so that failure surfaces in July rather than December. The expected honest result is a modest edge of a few percentage points of championship probability. A large apparent edge should be treated as evidence of overfitting, not success.

## Out of scope

Trade evaluation, in-season roster management, auction drafts, dynasty/keeper formats, building projections from scratch, and any modeling of K or DST beyond replacement level.

## Open questions

None blocking. ESPN cookie extraction and the 2026 live-draft endpoint behavior are both resolved during implementation.
