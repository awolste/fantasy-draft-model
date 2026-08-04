# Handoff — 2026 Fantasy Draft Optimizer

**Last updated:** 2026-08-04
**Repo:** https://github.com/awolste/fantasy-draft-model (public)
**Read this first**, then `docs/superpowers/specs/2026-08-02-ff-draft-2026-design.md` for the design and `docs/superpowers/plans/README.md` for the stage roadmap and the running constraints list.

---

## 1. What this is

A draft optimizer for **one specific ESPN league**, for the 2026 season. It recommends the best available pick at any point in a snake draft by **simulated championship probability**, and as a byproduct answers empirically whether 0RB, Hero RB, or 2RB:1WR wins most often from draft slot 8.

It is not a general fantasy tool. Almost every number in it is specific to this league's settings, and several would be actively wrong elsewhere.

## 2. League configuration — never assume defaults

| Setting | Value |
|---|---|
| Teams | 10 |
| Scoring | Full PPR, **6-point passing TDs** |
| Starters (10) | 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DST |
| Roster | 18 (8 bench, 1 IR) |
| Regular season | 14 weeks |
| Playoffs | 6 teams, 3 rounds, top 2 seeds get byes |
| Draft | Snake, no keepers, **slot 8** |
| Platform | ESPN, private league |

Full K and DST scoring rules are in the spec. Two bands are **neutral, not missing**: points allowed 18–27 and yards allowed 300–349 both score 0. FG 50–59 and 60+ both score 5.

## 3. Current status

| Stage | Scope | State |
|---|---|---|
| 1 | Data foundation | **Complete**, merged to `main` |
| 2 | Player + season model | **Complete**, merged to `main` |
| 3 | Opponent model + optimizer | **5 of 7 tasks**, on branch `stage-3-optimizer` |
| 4 | Live draft assistant | Not started, no plan written |

**Branch `stage-3-optimizer` is ahead of `main` and unmerged. 360 tests pass** (342 + 18 new backtest tests). Suite takes ~11-12 minutes.

Stage 3 remaining: **Task 7 (the structure study — the headline deliverable)**. Task 6 (the backtest) is done, and it failed the gate: the engine loses to naive ADP-following on real 2024 results by 13.5pp ± 1.65pp — see item 1 below. **Task 7 should not proceed on the assumption the engine works** until this is investigated.

## 4. Architecture

```
src/ffdraft/
  league.py            single source of truth for every league rule
  scoring.py           raw stat line -> fantasy points (6-pt pass TD)
  store.py             parquet persistence + cache fingerprinting
  ids.py               player ID crosswalk (nflverse/ESPN/Sleeper/FantasyPros)
  validate.py          cross-source validation, raises on failure
  sources/             nflverse, espn, espn_parse, fantasypros, ffcalculator
  models/
    tier_shape.py      weekly distribution shape by position/tier
    rank_curve.py      rank -> projected weekly mean
    distribution.py    per-player weekly distributions, player pool  <- Stage 3 imports this
    calibration.py     tail-fit diagnostics
    availability.py    two-state Markov availability
    replacement.py     waiver replacement level by position
    kicking.py         kicker scoring
    defense.py         one shared DST distribution
    roster.py          build_roster: pick list -> simulator-ready roster
    opponent.py        league-wide opponent draft model
  sim/
    lineup.py          exact optimal weekly lineup (proven, reference impl)
    season.py          14 weeks + 6-team bracket -> champion (vectorized)
  draft/
    value.py           roster-aware marginal value (greedy, fast)
    rollout.py         play the remaining draft forward
    recommender.py     rank candidates by championship probability
scripts/               ingest_all, sanitize_espn_fixture, season_report,
                       correlation_impact, compare_league_wide, backtest (TODO)
```

Dependencies point downward. One known inversion: `sources/nflverse.py` imports `models/kicking.py` (flagged, not fixed — see open items).

## 5. Decisions made, and why

These were deliberate. Do not silently reverse them.

**DST is one shared distribution for all teams** (owner decision). Defensive scoring is mostly luck with little draft control. Consequence: the engine **never drafts a DST**, because a drafted DST is worth exactly what the replacement DST is worth. That is correct, not a bug. The full DST scoring rules are recorded in the spec but deliberately unimplemented — implementing them would need a team-game-level ingest that this decision avoids.

**Per-manager opponent modelling was removed.** Head-to-head on the 2024 holdout: the per-manager and league-wide models predicted the **identical top-1 player on all 165 picks**, zero disagreements. Between-manager variance (τ²≈0.0016) was swamped by within-manager noise (σ²≈0.069). The league-wide model is 13–15% faster and simpler. `manager_id` plumbing was kept in the API for a future season with more data.

**Correlation is not modelled.** Measured, not assumed: QB1–WR1 same team r=0.372, QB1–TE1 0.281, opposing-team totals 0.187, **WR1–WR2 same team only 0.034** (shared offense cancelled by target competition). Injecting a calibrated copula moved the stacked-vs-diversified championship gap by **−0.42pp** against a 0.22pp SE. Immaterial. Note the sign — correlation *reduces* a stacked roster's edge, because clustered bad weeks hurt more than clustered good weeks when 60% of the league makes the playoffs.

**The backtest holds out 2024, not 2025.** No source has 2025 ADP: Fantasy Football Calculator covers 2018–2024 and 2026 but not 2025; ESPN has genuine ADP only for 2023, 2024, and 2026 (every other season returns sentinel padding — `170.0` for every player in 2018 and 2025, `0` in 2019).

**ADP comes from Fantasy Football Calculator, not FantasyPros.** FantasyPros fences ADP behind registration (5 rows logged out). Its ECR rankings are still used and work fine. FFC's `teams` parameter does **not** filter — `teams=10` and `teams=12` return identical data, so no 10-team-specific ADP exists anywhere.

**Lineups are set from realized scores** (perfect hindsight no manager has). Quantified: max 2.5pp swing on real 2024 rosters. Modest, but those rosters were not variance-stacked, so it does not bound the effect on boom/bust builds.

## 6. Key measured findings

**The league-specific edge — owner confirmed.** This league drafts QBs **later** than consensus ADP (positional bias −0.100, strongest negative of any position) while using 6-point passing TDs, which make QBs worth *more* here than the generic rankings leaguemates read. The owner confirmed the cause: the league drafts off generic ESPN rankings and is not especially sophisticated. This is a standing, exploitable inefficiency. It surfaces on its own — Stage 2's distributions anchor QB means to actual league scoring, and the opponent model says QBs survive longer than that warrants. **If the recommender stops showing QBs early, treat it as a bug signal.** Josh Allen currently ranks 5th from an empty roster at pick 8.

**Realistic edge is small.** 2024 calibration: best real roster ~16% championship probability against a 10% baseline; the actual champion drew 10.3%. A 60% playoff rate plus a 3-round bracket washes out most roster edge.

**Replacement level** (points/week, and it is an *order statistic*, not a mean — what you get is the best available each week): QB 18.3, RB 9.6, WR 9.9, TE 8.1, K 7.9. Starter medians: 24.3 / 15.9 / 16.4 / 12.4 / 9.0. **A rostered RB starter is worth ~6.4/week over waivers and a WR ~6.5 — nearly identical**, the single most relevant prior for the 0RB question.

**Availability is persistent.** P(out next week | out this week) = **75.5%**, versus 7.1% if available. Two-state Markov chain. Rates: K 93.4%, QB/WR 84.2%, RB 80.4%, TE 79.7%.

**Draft uncertainty dominates season uncertainty.** Between-rollout SD 3.50pp against within-rollout 0.60pp — a factor of ~34. This inverted the cost model: a rollout costs more than 1,000 simulated seasons.

**Rank curves are only valid inside their fitted ADP range.** `RankCurve.mean_for_rank` raises past `max_supported_rank` rather than extrapolating. An earlier unconstrained power law put a 50th-ranked QB at 16.5 ppg against ~3.3 observed.

## 7. Open items

**Blocking the headline deliverable:**

1. **Task 6 — the backtest has been run, and the engine loses.** `scripts/backtest.py` / `src/ffdraft/backtest.py`. Full leakage audit passed (every fitted component checked against seasons through 2023 only; see `tests/test_backtest.py::test_fit_report_seasons_never_include_the_holdout_season`). N=600 draft realizations, same-seed paired comparison: **engine 3.00% ± 0.70% SE vs. ADP-baseline 16.50% ± 1.52% SE — engine minus ADP = −13.50pp ± 1.65pp SE**, an ~8-sigma deficit, not noise. Random-legal scored 0.00%, so the engine barely clears a policy with no skill at all. The gap concentrates in rounds 3–6 (mean real-2024 points-per-game of our picks there trails ADP's picks at the same slot by 2-3 ppg; picks 1-2 are roughly at parity or slightly ahead). **Do not treat the engine as validated.** Diagnosis is under way — see §7b below for what has already been ruled out.
1b. **The next diagnostic step is specified and not yet run:** put the ADP policy through the engine's own code path and confirm it still scores ~16%. See the end of §7b. This is cheap and decisive about whether the harness or the engine is at fault, and everything else downstream depends on the answer.

2. **Task 7 — the structure study has not been run**, and **should not run until item 1 is resolved.** It would play out each structure using a policy known to lose to ADP by 14pp, so any answer would be an artifact of a broken drafter rather than a fact about roster construction. 0RB vs Hero RB vs 2RB:1WR from slot 8, with error bars. Structures currently differentiate only modestly in position counts (0RB ends RB3/WR8, 2RB ends RB4/WR7) but differ substantially in player quality, which is what the comparison measures. Given RB≈WR over replacement, **be suspicious of any large structural edge** and investigate before believing it.

**Known defects and unresolved questions:**

3. **QB=3 and K=2 are cap-bound in every rollout.** The value function still ranks a third QB and second kicker above better alternatives; a policy guard (`MAX_EXTRA_BENCH_NO_FLEX`) is the only thing stopping it. That is masking, not fixing, and costs ~2 roster spots versus opponents' 1.7 QB / 1.2 K. `value.py` has been through three fix rounds; a fourth patch is probably the wrong move — consider whether greedy marginal value is the right policy at all.
4. **Between-rollout variance appears state-dependent and is unresolved.** Measured 3.50pp at pick 8 (n=50) but 5.97pp in an independent 10-seed check at a different state. The staleness guard catches drift over time but will not tell you the allocation is too tight at some other draft state.
5. **A recommendation honestly costs ~5.6 minutes** for 15 candidates (N=65 rollouts × M=500 sims). **This is too slow for a live draft**, where you have ~90 seconds. Stage 4 needs a decision: pre-compute likely draft states, cut rollouts and accept wider error bars, or restrict to a pre-selected shortlist. This is a usage decision, not an engineering one.
6. **Test suite takes ~12 minutes**, roughly quadrupled because the pick-8/pick-13 real-data tests use full production budgets. Consider pinning a smaller explicit budget in those two tests.
7. **`sources/nflverse.py` imports `models/kicking.py`** — a Stage 1 → Stage 2 layering inversion. Cheap to fix by moving `kicking.py` beside `scoring.py`.
8. **Python runs under Rosetta x86-64 emulation**, not native ARM. Polars warns it may crash, and one full-suite run segfaulted when two heavy Polars processes ran concurrently. Switching to a native ARM Python is worth doing before more heavy simulation work.
9. **`data/manager_labels.csv` is filled in** with real manager names (gitignored). Never read, print, or commit it.

## 7b. Diagnosing the backtest failure — state as of 2026-08-04

Three hypotheses were posed for the 13.5pp deficit:

- **A — wasted roster spots.** Every rollout ends QB=3, K=2, both exactly at the `MAX_EXTRA_BENCH_NO_FLEX` caps, versus opponents' ~1.7 QB and ~1.2 K. Roughly 3 picks on near-worthless players.
- **B — stale information.** Live ADP prices in current-season knowledge (camp battles, trades, coaching changes) that a fit through 2023 cannot see.
- **C — the value function's cross-positional reasoning is mis-specified**, independent of A and B.

### Test 1 — RULED OUT hypothesis B (complete)

Spearman correlation against **actual 2024 points per game**, using `fit_holdout_context(fit_through_season=2023, holdout_season=2024)`, players with ≥4 games (n=473):

| Position | our projected mean | ADP rank | n |
|---|---|---|---|
| QB | 0.737 | 0.735 | 59 |
| RB | 0.777 | 0.774 | 115 |
| WR | 0.652 | 0.656 | 197 |
| TE | 0.465 | 0.458 | 102 |
| **cross-position** | **0.674** | 0.650 | 473 |

**Within position, our projections and ADP are the same predictor to three decimals.** This is structural, not coincidental: `rankings_holdout` is built by sorting 2024 ADP, so `pool[pid].rank` *is* ADP rank, and the rank curve is monotone decreasing in it — we reproduce ADP's within-position ordering by construction. Cross-position we are slightly *better*, as expected from mapping ranks onto a common points scale.

**Consequence:** the deficit cannot originate in the projections. We carry the market's information faithfully and add a little. It must originate in the **decision layer** — how `value.py` turns projections into picks. That leaves A and C.

A corollary worth knowing: because there is no within-position disagreement between us and ADP, **every disagreement is cross-positional**. Any analysis looking for "players we overrate" within a position will find nothing.

### Test 2 — the ladder (complete). Ruled out hypotheses A and C.

Six contenders, N=600 realizations, common random numbers, same opponent seeds. Runtime 740s (1.23s/realization).

| Contender | Championship rate |
|---|---|
| `adp` | **16.17% ± 1.50%** |
| `consensus` | 16.17% ± 1.50% (identical — same underlying signal, no independent 2024 consensus source exists in this project) |
| `engine_capped` (QB≤2, K≤1) | 2.83% ± 0.68% |
| `engine` | 2.17% ± 0.59% |
| `proj_mean` (raw projected points, no VOR) | **0.00% ± 0.00%** |
| `random` | 0.00% ± 0.00% |

Paired adjacent differences:

```
proj_mean - adp        : -16.17pp  (SE 1.50)
engine - proj_mean     :  +2.17pp  (SE 0.59)
engine_capped - engine :  +0.67pp  (SE 0.75)   <- not significant
```

**Hypothesis A (wasted roster spots) is small.** Capping QB and K buys +0.67pp against an SE of 0.75 — indistinguishable from zero. The three wasted spots were real but cost almost nothing.

**Hypothesis C (VOR is harmful) is refuted, and in the direction opposite to what was predicted.** VOR and roster-awareness *help*: the engine beats raw-projection drafting by +2.17pp. The coordinator argued in advance that VOR was noise layered on a correct signal. That was wrong.

### Important caveat: `proj_mean` was a badly chosen rung

`proj_mean` scoring **exactly 0.00%** — 0 wins in 600, identical to random — is almost certainly degenerate rather than merely weak. Ranking by raw projected *points* ignores position entirely, and QBs carry the highest absolute means in this league (Josh Allen 27.45 vs Bijan Robinson 24.39), so it very likely drafts quarterback after quarterback. That is precisely the failure VOR exists to prevent, which is why VOR looks good by comparison.

So the honest reading of the ladder is **not** "VOR helps." It is: **VOR rescues a degenerate baseline and still lands 14pp short of ADP.**

The rung that would actually have been informative — and does not yet exist — is our projections converted to VOR but *without* the roster-awareness and bench-depth machinery. That would separate "scarcity correction" from "roster construction." Whoever picks this up should build that rung before drawing further conclusions.

### Where the diagnosis stands

Eliminated: stale projections (Test 1), wasted roster spots (rung 4), VOR-as-harmful (rung 3).

**The gap is essentially undiminished and unexplained.** The engine carries the market's information faithfully (Test 1), applies a scarcity correction that helps (+2.17pp), and still loses to reading ADP off a page by 14pp.

**Next check to run, and the cheapest decisive one:** run the ADP policy *through the engine's own code path* — same rollout, same roster construction, same scoring — and confirm it still scores ~16%. If it drops, the harness is disadvantaging our slot rather than the engine's picks being bad, and the leakage audit would not have caught that. If it holds at 16%, the engine's picks genuinely are that much worse and the approach itself is the problem.

Until that check is done, **do not conclude the approach is unsalvageable, and equally do not conclude the harness is at fault.** Both remain live.

### Minor issue noticed in passing

The holdout pool excludes 15 unmatched players, including **Hollywood Brown at rank 86** — genuinely draftable in 2024, and the same name-normalization gap as Stage 1 (he is "Marquise Brown" in the crosswalk). Too small to explain 13.5pp, and it handicaps every contender equally rather than biasing between them, but worth fixing.

## 8. The recurring failure pattern — read this before trusting any number

**Every genuine defect in this project produced plausible-looking output and raised no errors.** None were found by reading code. All were found by checking a reported number against raw data or against the artifact that actually mattered.

A partial list:

- Dedup on a null key silently deleted **all 2,045 rookies** from the player pool
- Name normalization collided Marvin Harrison Jr. with his father, making a join return more rows than it was given (~8% inflation)
- Rank curves extrapolated past their fitted range claimed a 50th-ranked QB scores 16.5/week (actual ~3.3)
- Weekly distributions were over-dispersed ~58% at QB because the fit maximized likelihood instead of matching the tail
- Replacement level was computed wrong twice — once as a mean of deep players, once with look-ahead bias
- A survivorship-bias trap made running backs look as durable as quarterbacks
- Return TDs scored 0 instead of 6 for 151 real rows
- The draft policy drafted **six quarterbacks and two running backs**, giving us 5.68% title odds against a 10% baseline — while the pick-8 recommendation list it produced looked entirely reasonable

- The engine itself lost to naive ADP-following by 13.5pp on real 2024 results — while producing recommendation lists that read as entirely sensible, and while every component beneath it had been individually verified

**The lesson that generalizes:** verify at the level of the artifact that matters, not the level of the component. A unit test on the value function passed while the policy built an absurd roster. The check that caught it was inspecting the *rosters the policy produces*. Similarly, the calibration test that mattered compared modelled quantiles against observed data, not against themselves.

Three separate privacy leaks in ESPN fixtures followed the same shape — each scrub covered what someone thought to list, and the next review found another field (GUIDs and names, then team abbreviations and league name, then the league ID). Run `scripts/sanitize_espn_fixture.py` and read its output rather than assuming coverage.

## 9. How to run things

```bash
cd /Users/andrewinvsys/Documents/Code/ff-draft-2026
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # then fill FF_LEAGUE_ID, ESPN_S2, ESPN_SWID
```

ESPN cookies come from a logged-in browser: DevTools → Application → Cookies → espn.com. They expire; a 401 during ingest means recapture. **Your `SWID` cookie is also your ESPN member GUID and appears throughout league data** — it leaks as ordinary content, not just in headers.

```bash
.venv/bin/python scripts/ingest_all.py    # pull + validate all nine datasets
.venv/bin/pytest                          # ~12 minutes
```

Caches are fingerprinted against their inputs and **raise `CacheStaleError`** if the underlying data changed — deliberately loud rather than silently refitting, because a silent refit inside a rollout loop would be an invisible performance cliff. Rank-curve caches are namespaced by rankings context so the 2026 pool and a historical backtest pool coexist without evicting each other.

## 10. Working agreement that has been in force

- Subagent-driven execution: fresh Sonnet subagent per task, then spec and quality review. Small, well-specified modules get a combined review; risky ones get separate passes.
- Reviews are dispatched in parallel (read-only); implementers never run concurrently (git collisions).
- Findings that change a documented number get written back into `docs/superpowers/plans/README.md` so they survive the session.
- Numbers reported by subagents are spot-checked against raw data before being believed. This has caught real errors more than once, including one where the coordinator's own first check used the wrong baseline and nearly flagged a correct model as broken.
