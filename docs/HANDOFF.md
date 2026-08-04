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

Diagnosis as of 2026-08-04 (§7b, tests 1-6): **~6pp of that deficit is an artifact of the scoring rule** (hindsight lineup-setting pays FLEX depth a premium no manager could collect). The corrected, ex-ante deficit is **9.25pp ± 1.76pp** — smaller, but still a failed gate: the engine scores 2.75% against a 10% baseline. Cause not yet identified; current evidence points at the replacement means rather than at any policy or value-function layer.

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

**Lineups are set from realized scores** (perfect hindsight no manager has). Originally quantified at max 2.5pp swing on real 2024 rosters, with a caveat that those rosters were not variance-stacked. **That caveat proved to be the important part — see §7b Test 6.** For a *depth*-stacked roster (15 FLEX-eligible players, which is what ADP-following produces), hindsight is worth **5.75pp**, and it is worth nothing to a roster that spends picks on single-slot positions. The 2.5pp figure does not bound this, and `real_season_champion` should be treated as **defective for comparing rosters of different shapes**.

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
1b. **Diagnosis has advanced substantially (tests 3-6, §7b).** ~6pp of the deficit is now shown to be a *scoring-rule artifact*: `real_season_champion` sets lineups with perfect hindsight, which pays FLEX depth a premium no manager could collect. `real_season_champion` should be treated as defective for comparing rosters of different shapes, and fixing it is a well-specified task. The remaining ~9.25pp is real and unexplained; the evidence now points at the **replacement means**, QB first. See the end of §7b.

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

### Test 3 — harness fault is dead by construction (no run needed)

The check specified above — "run the ADP policy through the engine's own code path" — turned out to be **a no-op, answerable by reading `run_one_draft`**. Every contender already goes through `run_rollout` with the same opponent model, the same `adp_table`, the same `rankings`, and the same `real_season_champion` scoring. The *only* thing that varies is `pick_policy`. Neither the engine's `_choose_our_pick` nor `adp_pick_policy` consumes `rng`, so opponent picks are drawn from an identical RNG stream in both — the contenders are perfectly paired, pick for pick.

So ADP's 16.17% **is** ADP through the engine's code path. **The harness does not disadvantage our slot. The engine's picks genuinely are that much worse.** Do not re-run this.

(One latent flaw noticed while checking: `random_legal_pick_policy` *does* draw from the shared `rng`, so the random contender's opponents run on a shifted stream. It does not bias the distribution and random scores 0% regardless, but it breaks CRN pairing for that one contender.)

### Test 4 — found it: the engine reaches, and the reaching is concentrated in K and TE

`scripts/diagnose_picks.py` prints our 18 picks side by side, engine vs ADP, with each player's real 2024 ppg. **Nobody had ever looked at the picks themselves** — only at aggregates. The defect is visible immediately.

Define **reach = (player's ADP) − (overall pick spent on him)**. Positive means we took him earlier than the market would have, i.e. we burned that many picks of value. Over 8 seeds × 18 picks = 144 picks each:

| | mean reach | median | picks reached ≥30 |
|---|---|---|---|
| engine | −4.4 | −7.4 | **8 / 144** |
| adp | −13.2 | −9.5 | **0 / 144** |

By position, the engine:

```
K    n=10  mean reach  +48.9      <- the defect
TE   n=16  mean reach  +12.5
QB   n=24  mean reach   -2.8
WR   n=48  mean reach  -11.4
RB   n=46  mean reach  -15.4
```

ADP's worst position sits at −11.1 and it never reaches ≥30 on any pick. **The engine drafts a kicker roughly 49 picks before the market does** — concretely, Justin Tucker (ADP 141) taken in *round 8*, in both sampled seeds. It also takes Luke Musgrave (ADP 152) in round 13. RB and WR timing is essentially the same as ADP's, so the skill positions are not where this is lost.

**This also shows hypothesis A was tested the wrong way.** `engine_capped` capped QB and K *counts* (QB≤2, K≤1) and bought +0.67pp. It never touched *timing* — `engine_capped` still spends a round-8 pick on that one permitted kicker. The count was never the waste. The timing is.

**Diagnosis:** the greedy value policy has no notion of **who will still be available at our next pick**. It takes the highest marginal value on the board right now, so it happily spends round 8 on the best kicker — who would still be there in round 17. ADP-following gets this right for free, because ADP order *is* survival order. This is the classic "don't draft a kicker in round 8" rule, and the value function does not encode it.

### Test 4b — the kicker is NOT the disease

`scripts/diagnose_k_timing.py`. The engine's own greedy policy, unchanged, except K is undraftable until the final round. One variable. N=300, paired.

```
engine           2.00% ± 0.81%
engine_defer_k   3.33% ± 1.04%
vor_only         0.00% ± 0.00%
adp             15.33% ± 2.08%

engine_defer_k - engine :  +1.33pp  (SE 1.15pp)   <- not significant
engine         - adp    : -13.33pp  (SE 2.23pp)
```

Fixing the single worst reach in the draft buys **+1.33pp ± 1.15** and closes a tenth of the gap. Same shape as hypothesis A. **Reaching is a symptom, not the disease.** This run also independently replicates the headline deficit (−13.33pp here vs −13.50pp originally, different N).

### Test 5 — `vor_only` is 0.00%, but its draft board is fine

The missing rung now exists: `vor_only` = static `argmax(mean − replacement_mean[pos])`, no lineup solve, no bench term, no FLEX logic. It scored **0.00%**, matching `proj_mean` and `random`.

The obvious inference — "our VOR ordering is garbage" — is **wrong**, and `scripts/diagnose_vor_scale.py` shows why. The top of the static VOR board is a perfectly reasonable draft board:

```
 1 McCaffrey  RB 24.72 vor 14.85 (adp 1.4)     6 Kelce   TE 17.25 vor 9.05 (adp 24.4)
 2 Tyreek Hill WR 23.83 vor 13.78 (adp 2.6)    7 St.Brown WR 18.86 vor 8.81 (adp 6.2)
 3 Breece Hall RB 20.68 vor 10.81 (adp 2.2)    8 Bijan    RB 18.63 vor 8.76 (adp 4.9)
 4 CeeDee Lamb WR 20.56 vor 10.51 (adp 3.7)    9 Chase    WR 17.74 vor 7.69 (adp 7.2)
 5 Josh Allen  QB 27.55 vor  9.58 (adp 25.1)  10 Gibbs    RB 17.30 vor 7.43 (adp 9.4)
```

Replacement means (pts/wk): QB 17.96, WR 10.05, RB 9.88, TE 8.20, K 7.94, DST 7.50.

So the projections and the scarcity correction are both sane, and a policy built on them still wins zero times in 300. **A sane ranking is not sufficient.** That is the single most important fact established so far, and it points away from the value function entirely.

### Test 6 — the live hypothesis: the *scoring* rewards depth, via hindsight

Where the position mixes actually land, per 18-round draft:

```
adp     RB 7  WR 8  QB 3           -> 15 FLEX-eligible, zero TE
engine  RB 7  WR 5  QB 3  TE 2  K 1 -> 12 FLEX-eligible
```

ADP-following drafts **zero tight ends**, fills its TE slot at replacement all season, and still beats us by 13pp. Meanwhile `real_season_champion` calls `solve_lineup` on **that week's realized scores** — perfect hindsight, every week, every team.

Hindsight is equal across teams but **not across roster shapes**. A roster with 15 FLEX-eligible players picks the best 6 of 15 ex post; a single-slot position (QB, TE, K) gains nothing from depth. So the scoring rule may be paying ADP-following a depth premium that no real manager could collect, and that the engine — which spends picks on TE/K — cannot access.

This reframes the whole diagnosis: **the fault may be in the harness after all, but in the *scoring* step, not the drafting.** Test 3 ruled out the drafting path; it says nothing about scoring.

`scripts/diagnose_hindsight.py` scores the *same* drafts two ways — `hindsight` (what the backtest does) and `ex_ante` (lineup chosen on projected means, then scored on realized points, which is what a manager can actually do). **N=400:**

| contender | scoring | rate |
|---|---|---|
| engine | hindsight | 2.25% ± 0.74% |
| adp | hindsight | **17.75% ± 1.91%** |
| engine | ex_ante | 2.75% ± 0.82% |
| adp | ex_ante | **12.00% ± 1.63%** |

```
adp - engine, hindsight : +15.50pp  (SE 2.07pp)
adp - engine, ex_ante   :  +9.25pp  (SE 1.76pp)
```

**Confirmed, and the asymmetry is exactly as predicted.** Removing hindsight costs the ADP roster **−5.75pp** (17.75 → 12.00) and costs the engine roster **nothing** (2.25 → 2.75, i.e. it gains slightly). Depth in FLEX-eligible positions is only worth that much when you can pick your starters after the fact. **About 6pp of the ~15pp deficit is an artifact of the scoring rule, not of the engine's picks.**

An independent corroboration that the hindsight number is the broken one: Stage 2 measured the *best-drafted* 2024 roster at ~16% championship probability. Under hindsight scoring, plain ADP-following scores **17.75% — above that ceiling**. Under ex-ante scoring it scores 12.00%, comfortably inside it. The hindsight figure is not credible on its own terms.

Note the Stage 2 caveat that applies directly here: hindsight bias was measured at "max 2.5pp swing" on *real* 2024 rosters, with the explicit warning that those rosters were not deliberately variance-stacked. They were not deliberately **depth**-stacked either, so that 2.5pp never bounded this effect — and the real figure for a depth-stacked roster is 5.75pp.

### Where the diagnosis stands after tests 3-6

**Resolved:** ~6pp of the deficit is a scoring-rule artifact. The backtest as written overstates the engine's deficit by that much, and `real_season_champion` should be considered defective for comparing rosters of *different shapes*. Fixing it is a real, well-specified task.

**Still unexplained: the remaining ~9.25pp.** Ex ante, the engine scores **2.75% against a 10% baseline** — it is still much worse than drafting at random *among the ten teams*, and far worse than reading ADP off a page. That is the live problem.

What is now known about it, and constrains any future hypothesis:
- It is not the projections (Test 1), not stale information (Test 1), not wasted roster spots (rung 4), not reach timing (Test 4b), and not the roster-construction machinery on top of VOR — because **static VOR with none of that machinery also scores 0.00%** (Test 5).
- The draft board produced by our projections + scarcity correction is *sane* (Test 5). Whatever is wrong survives having a correct-looking ranking.
- Both the engine and `vor_only` lose to ADP, and both differ from ADP in the same direction: **more QB/TE, fewer RB/WR**. ADP's top-40 is RB 17 / WR 18 / QB 3 / TE 2; our VOR top-40 is RB 12 / WR 16 / QB 8 / TE 4. That positional tilt is the one thing every losing policy shares and ADP does not.
- The tilt comes from the replacement means: **QB replacement is 17.96 with QB30 at 18.01** — the QB curve is nearly flat past the top few, so elite-QB VOR (Josh Allen +9.58) prices him near the top of the board. If QB replacement is too *low* — e.g. if what you can actually stream weekly is better than 17.96 — every QB is overpriced and the tilt follows mechanically.

**The next check, and the one the evidence now points at:** validate the replacement means as an order statistic against real 2024 waiver-wire availability, QB first. A single miscalibrated replacement level would explain the shared positional tilt across every losing policy, which no policy-level fix has touched.

### Also important: the backtest never tested the recommender

`backtest.py`'s module docstring is explicit and honest about this, but it is easy to miss and it changes what the gate means. `contender="engine"` uses `run_rollout`'s **greedy `draft.value` policy**, not `draft.recommender.recommend_pick`, because the real recommender costs ~5.6 min/pick (N×18×5.6min is infeasible).

The docstring argues the Monte-Carlo layer "does not change how any later pick is chosen" — true — but it *does* choose the immediate pick, at every one of our 18 picks. So **the gate has never been run against the actual recommender.** Note the caveat before getting hopeful: `recommend_pick`'s rollouts use this same greedy policy for all of our *future* picks, so it inherits the same survival blind spot in its continuations — a candidate evaluated as "don't take the kicker now" will often just take the kicker one round later in the rollout, muting the comparison.

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
