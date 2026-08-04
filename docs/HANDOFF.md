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

**Branch `stage-3-optimizer` is ahead of `main` and unmerged. 342 tests pass.** Suite takes ~12 minutes.

Stage 3 remaining: **Task 6 (the backtest — the gate)** and **Task 7 (the structure study — the headline deliverable)**.

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

1. **Task 6 — the backtest has not been run.** This is the gate that decides whether the engine beats ADP-following. Hold out 2024, fit through 2023, draft from slot 8, score on **real** 2024 weekly results, compare to three baselines, report by round. Expect a few percentage points at most; a large edge means leakage. Check the holdout is genuinely clean — an earlier calibration was found to be slightly in-sample.
2. **Task 7 — the structure study has not been run.** 0RB vs Hero RB vs 2RB:1WR from slot 8, with error bars. Structures currently differentiate only modestly in position counts (0RB ends RB3/WR8, 2RB ends RB4/WR7) but differ substantially in player quality, which is what the comparison measures. Given RB≈WR over replacement, **be suspicious of any large structural edge** and investigate before believing it.

**Known defects and unresolved questions:**

3. **QB=3 and K=2 are cap-bound in every rollout.** The value function still ranks a third QB and second kicker above better alternatives; a policy guard (`MAX_EXTRA_BENCH_NO_FLEX`) is the only thing stopping it. That is masking, not fixing, and costs ~2 roster spots versus opponents' 1.7 QB / 1.2 K. `value.py` has been through three fix rounds; a fourth patch is probably the wrong move — consider whether greedy marginal value is the right policy at all.
4. **Between-rollout variance appears state-dependent and is unresolved.** Measured 3.50pp at pick 8 (n=50) but 5.97pp in an independent 10-seed check at a different state. The staleness guard catches drift over time but will not tell you the allocation is too tight at some other draft state.
5. **A recommendation honestly costs ~5.6 minutes** for 15 candidates (N=65 rollouts × M=500 sims). **This is too slow for a live draft**, where you have ~90 seconds. Stage 4 needs a decision: pre-compute likely draft states, cut rollouts and accept wider error bars, or restrict to a pre-selected shortlist. This is a usage decision, not an engineering one.
6. **Test suite takes ~12 minutes**, roughly quadrupled because the pick-8/pick-13 real-data tests use full production budgets. Consider pinning a smaller explicit budget in those two tests.
7. **`sources/nflverse.py` imports `models/kicking.py`** — a Stage 1 → Stage 2 layering inversion. Cheap to fix by moving `kicking.py` beside `scoring.py`.
8. **Python runs under Rosetta x86-64 emulation**, not native ARM. Polars warns it may crash, and one full-suite run segfaulted when two heavy Polars processes ran concurrently. Switching to a native ARM Python is worth doing before more heavy simulation work.
9. **`data/manager_labels.csv` is filled in** with real manager names (gitignored). Never read, print, or commit it.

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
