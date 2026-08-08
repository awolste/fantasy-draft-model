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
| 3 | Opponent model + optimizer | **Complete (7 of 7)**, merged to `main` |
| 4 | Live draft assistant | Not started, no plan written |

**Stage 3 is merged to `main` (commit `c48ff46`). 362 tests pass**, suite ~11.5 minutes. Note `main` is ahead of `origin/main` and has not been pushed. See open item 8 before trusting a single green suite run.

Stage 3 is **complete**. Task 7, the structure study, is **answered: draft structure does not matter** — 0RB, Hero RB and 2RB:1WR are statistically indistinguishable across 2020–2024. See §7c for the numbers and for the finding that does matter (the early-QB decision).

Task 6 (the backtest) is done. Its original verdict was: *the engine loses to naive ADP-following on real 2024 results by 13.5pp ± 1.65pp*. **That verdict has since been overturned — see §7b tests 6-9.** Two things were wrong with it. (a) ~6pp of the deficit was an artifact of the scoring rule: `real_season_champion` set lineups with perfect hindsight, which pays FLEX depth a premium no real manager could collect. That is now fixed. (b) 2024 is the worst of the five usable holdout seasons for this engine. Re-run across 2020–2024 through the fixed production path, **the engine beats ADP in three of five seasons and averages +2.45pp against a between-season SE of 4.27pp** — not broken, but not demonstrated either.

**Quote ex-ante, multi-season numbers. Do not quote the single-season 2024 figure**, and do not repeat "the engine loses to ADP by 13.5pp" — that number is now known to be mostly artifact.

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

**Backtest lineups are set ex ante — this was changed, and the old behaviour was a real defect.** `real_season_champion` originally chose each week's lineup from *realized* scores. Quantified early on at "max 2.5pp swing" on real 2024 rosters, with a caveat that those rosters were not variance-stacked. **That caveat proved to be the important part — see §7b test 6.** For a *depth*-stacked roster (15 FLEX-eligible players, which is exactly what ADP-following produces) hindsight is worth **5.75pp**, and worth nothing to a roster spending picks on single-slot positions — so it silently decided the backtest.

Since commit `2b845cc`, lineups are chosen on **projected means** and scored on **realized points**; weekly availability stays real, because a manager does know who is inactive, it is the scores they cannot see. Note this applies to the *backtest*. The forward-looking season simulator (`sim/season.py`) samples scores it then optimises against, which is a different and legitimate use — there is no hindsight when the scores are drawn from the model's own distributions.

## 6. Key measured findings

**The league-specific edge — owner confirmed.** This league drafts QBs **later** than consensus ADP (positional bias −0.100, strongest negative of any position) while using 6-point passing TDs, which make QBs worth *more* here than the generic rankings leaguemates read. The owner confirmed the cause: the league drafts off generic ESPN rankings and is not especially sophisticated. This is a standing, exploitable inefficiency. It surfaces on its own — Stage 2's distributions anchor QB means to actual league scoring, and the opponent model says QBs survive longer than that warrants. **If the recommender stops showing QBs early, treat it as a bug signal.** Josh Allen currently ranks 5th from an empty roster at pick 8.

**Realistic edge is small.** 2024 calibration: best real roster ~16% championship probability against a 10% baseline; the actual champion drew 10.3%. A 60% playoff rate plus a 3-round bracket washes out most roster edge.

**Replacement level** (points/week, and it is an *order statistic*, not a mean — what you get is the best available each week): QB 18.3, RB 9.6, WR 9.9, TE 8.1, K 7.9. Starter medians: 24.3 / 15.9 / 16.4 / 12.4 / 9.0. **A rostered RB starter is worth ~6.4/week over waivers and a WR ~6.5 — nearly identical**, the single most relevant prior for the 0RB question.

**Availability is persistent.** P(out next week | out this week) = **75.5%**, versus 7.1% if available. Two-state Markov chain. Rates: K 93.4%, QB/WR 84.2%, RB 80.4%, TE 79.7%.

**Draft uncertainty dominates season uncertainty.** Between-rollout SD 3.50pp against within-rollout 0.60pp — a factor of ~34. This inverted the cost model: a rollout costs more than 1,000 simulated seasons.

**Rank curves are only valid inside their fitted ADP range.** `RankCurve.mean_for_rank` raises past `max_supported_rank` rather than extrapolating. An earlier unconstrained power law put a 50th-ranked QB at 16.5 ppg against ~3.3 observed.

## 7. Open items

**Blocking the headline deliverable:**

1. **Task 6 — the backtest has been run. Original verdict ("the engine loses by 13.5pp") is OVERTURNED; see §7b tests 6-8 and the note in §3.** Multi-season ex-ante result: engine beats ADP in 3 of 5 holdouts, pooled +2.20pp ± 4.06pp. The text below is kept for the leakage audit and the original measurement, but its conclusion is superseded.** `scripts/backtest.py` / `src/ffdraft/backtest.py`. Full leakage audit passed (every fitted component checked against seasons through 2023 only; see `tests/test_backtest.py::test_fit_report_seasons_never_include_the_holdout_season`). N=600 draft realizations, same-seed paired comparison: **engine 3.00% ± 0.70% SE vs. ADP-baseline 16.50% ± 1.52% SE — engine minus ADP = −13.50pp ± 1.65pp SE**, an ~8-sigma deficit, not noise. Random-legal scored 0.00%, so the engine barely clears a policy with no skill at all. The gap concentrates in rounds 3–6 (mean real-2024 points-per-game of our picks there trails ADP's picks at the same slot by 2-3 ppg; picks 1-2 are roughly at parity or slightly ahead). **Do not treat the engine as validated.** Diagnosis is under way — see §7b below for what has already been ruled out.
1b. **Diagnosis has advanced substantially (tests 3-6, §7b).** ~6pp of the deficit is now shown to be a *scoring-rule artifact*: `real_season_champion` sets lineups with perfect hindsight, which pays FLEX depth a premium no manager could collect. `real_season_champion` should be treated as defective for comparing rosters of different shapes, and fixing it is a well-specified task. The remaining ~9.25pp is real and unexplained; the evidence now points at the **replacement means**, QB first. See the end of §7b.

2. **Task 7 — the structure study has not been run.** It is no longer blocked on "the engine is broken" (§7b test 8 overturned that), but it should still wait for the two fixes listed at the end of §7b — ex-ante lineup scoring and replacement-level uncertainty — because its answer is conditional on how the QB tilt is sized. It would play out each structure using a policy known to lose to ADP by 14pp, so any answer would be an artifact of a broken drafter rather than a fact about roster construction. 0RB vs Hero RB vs 2RB:1WR from slot 8, with error bars. Structures currently differentiate only modestly in position counts (0RB ends RB3/WR8, 2RB ends RB4/WR7) but differ substantially in player quality, which is what the comparison measures. Given RB≈WR over replacement, **be suspicious of any large structural edge** and investigate before believing it.

**Known defects and unresolved questions:**

3. **QB=3 and K=2 are cap-bound in every rollout.** The value function still ranks a third QB and second kicker above better alternatives; a policy guard (`MAX_EXTRA_BENCH_NO_FLEX`) is the only thing stopping it. That is masking, not fixing, and costs ~2 roster spots versus opponents' 1.7 QB / 1.2 K. `value.py` has been through three fix rounds; a fourth patch is probably the wrong move — consider whether greedy marginal value is the right policy at all.
4. **Between-rollout variance appears state-dependent and is unresolved.** Measured 3.50pp at pick 8 (n=50) but 5.97pp in an independent 10-seed check at a different state. The staleness guard catches drift over time but will not tell you the allocation is too tight at some other draft state.
5. **A recommendation honestly costs ~5.6 minutes** for 15 candidates (N=65 rollouts × M=500 sims). **This is too slow for a live draft**, where you have ~90 seconds. Stage 4 needs a decision: pre-compute likely draft states, cut rollouts and accept wider error bars, or restrict to a pre-selected shortlist. This is a usage decision, not an engineering one.
6. **Test suite takes ~12 minutes**, roughly quadrupled because the pick-8/pick-13 real-data tests use full production budgets. Consider pinning a smaller explicit budget in those two tests.
7. **`sources/nflverse.py` imports `models/kicking.py`** — a Stage 1 → Stage 2 layering inversion. Cheap to fix by moving `kicking.py` beside `scoring.py`.
8. **Python runs under Rosetta x86-64 emulation**, not native ARM, and this still causes intermittent native crashes. `polars[rtcompat]` (§9) reduced but did **not eliminate** them: after installing it, a full `pytest` run died with a native fault dump, and two subsequent identical runs passed 362/362. `pytest-randomly` is not installed, so this is not test-order dependence — it is plain intermittency, roughly 1 in 3 observed. **Switching to a native ARM Python is now the top infrastructure priority**, not a nice-to-have: it removes Rosetta from the picture instead of patching around it. Until then, treat a single green suite as weak evidence and re-run before trusting a release. Never set `POLARS_SKIP_CPU_CHECK=1`.

8b. **Beware `cmd | tail` when checking exit codes.** A crashed `pytest` piped through `tail` reported **exit 0**, because a pipeline's status is the last command's. This nearly caused a segfaulting suite to be merged to `main` on a false green. Redirect to a file and check `$?` directly. Also note backgrounded commands start from the session cwd, so a `cd` in a *previous* tool call does not apply — one such run failed with exit 127 while still printing a success-looking line.
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

### Test 7 — replacement levels are NOT miscalibrated. The tilt is one season's variance.

`scripts/validate_replacement_2024.py` and `validate_replacement_trend.py`. The same estimator (`weekly_replacement_values`) run on 2024 alone — out of sample for a through-2023 fit.

```
pos   fitted <=2023   realized 2024     diff
QB           17.96           20.11    +2.15
RB            9.88            8.83    -1.05
WR           10.05            8.88    -1.19
TE            8.20            7.67    -0.53
K             7.97            7.22    -0.75
```

The QB miss is in the predicted direction and the RB/WR misses compound it: QB replacement too low **and** RB/WR replacement too high both overprice quarterbacks, ~3.3 ppg on the QB-vs-WR comparison.

**But it is not a calibration bug.** The full per-season series (this league's scoring):

```
       2018   2019   2020   2021   2022   2023   2024   2025
QB    18.57  18.34  19.31  16.83  16.36  18.37  20.11  18.20
RB    12.69   7.52  11.91  10.09   7.70   9.36   8.83   8.26
WR    10.40  11.08  10.03   8.85   8.86  11.06   8.88   9.55
```

The QB series is **stationary and noisy, not trending**: 8-season mean 18.26, sd 1.21, and 2025 comes back to 18.20 — essentially the fitted 17.96. **2024 is the high outlier of the series.** The fitted value is a good estimate of the long-run mean. The hypothesis is dead.

*(A ratio worth not over-reading: sd/spread across P10–P30 is QB 0.49, RB 0.57, WR 0.31, TE 0.52. RB is the noisiest by that measure, so "QB is uniquely unidentifiable" is **not** supported.)*

**What this does explain, completely, is the positional tilt.** Re-ranking the same pool with 2024's realized replacement instead of the fit (deliberately leaky — diagnostic only, see `diagnose_vor_scale.with_alternate_replacement`):

```
top 40   VOR, fitted <=2023 : QB 8  RB 12  TE 4  WR 16
         VOR, realized 2024 : QB 3  RB 13  TE 4  WR 20
         ADP                : QB 3  RB 17  TE 2  WR 18
```

QB share goes **8 → 3, exactly matching ADP.** The one feature every losing policy shared, and ADP did not, is entirely an artifact of 2024's replacement levels differing from their long-run means.

### What this means for the gate — read carefully

**It does not exonerate the engine.** Ex ante the engine still scores **2.75% against a 10% baseline**. Nothing here explains being that far *below* average, and the QB tilt alone should not do that.

**It does invalidate the gate as currently designed.** The 2024 backtest is a single sample, and — for a strategy whose designed edge is a QB premium (§6, the 6-point-passing-TD inefficiency) — it is the **most hostile of the eight seasons on record**. A one-season backtest cannot distinguish "the QB thesis is wrong" from "2024 was the worst year in eight for it." Both predict exactly what was observed.

**The next thing to do, and it is now well-specified:** run the backtest across **multiple holdout seasons**, not just 2024. FFC ADP covers 2018–2024, so 2019–2023 are all available as additional holdouts at the cost of refitting per season. Two things fall out:
- If the engine loses in every season, the approach is wrong and the QB story was a red herring.
- If it loses badly in 2024 and is competitive in seasons where QB replacement ran near its mean, the gate was measuring season variance and needs to be re-specified as a multi-season average.

Do this **before** any further work on `value.py` or the policy layer. Four separate policy-level fixes have now been measured and the largest bought 1.33pp; the evidence says the remaining problem is not in that layer.

### Test 8 — the multi-season sweep. 2024 was not representative, and the engine is not broken.

`scripts/backtest_multiseason.py`, N=400 per season, every component refit from scratch through the prior season, both scoring rules.

**engine − ADP, percentage points:**

| holdout | hindsight | ex ante | QB replacement that season |
|---|---|---|---|
| 2020 | +2.75 ± 2.16 | **+5.00 ± 1.96** | 19.31 |
| 2021 | +3.50 ± 2.31 | **+9.00 ± 2.25** | 16.83 |
| 2022 | +4.25 ± 2.30 | **+11.50 ± 2.30** | 16.36 |
| 2023 | −15.50 ± 2.24 | −5.00 ± 1.72 | 18.37 |
| 2024 | −15.00 ± 2.05 | −9.50 ± 1.78 | 20.11 |
| **pooled** | **−4.00 (between-season SE 4.60)** | **+2.20 (between-season SE 4.06)** | |

**The engine beats ADP in three of five seasons, by as much as +11.50pp.** The 2024 gate result was the worst of the five and is not representative. *Task 6's headline conclusion — "the engine loses, do not treat it as validated" — was an artifact of holding out a single, unusually hostile season.*

**Equally: the engine is not proven better.** Pooled ex ante is **+2.20pp against a between-season SE of 4.06pp** — indistinguishable from zero. The honest summary is *"not broken, not yet demonstrated."* Do not upgrade this to "validated."

**Hindsight bias replicates across seasons.** Pooled hindsight −4.00pp vs ex ante +2.20pp: the scoring rule costs the engine ~6pp in every season, matching test 6's 5.75pp measured independently. `real_season_champion` should be fixed, and until it is, **quote ex-ante numbers**.

**The engine's edge is a leveraged, unhedged bet on QB replacement level.** Across the five seasons, engine edge vs. that season's QB replacement: **Pearson r = −0.810, slope −4.61pp per +1 ppg** (ex ante; hindsight r = −0.624). The engine wins big when QB replacement runs low (2022: 16.36 → +11.50pp) and loses when it runs high (2024: 20.11 → −9.50pp).

Two caveats that matter, and neither is small:
- **n = 5 seasons.** r = −0.810 at n=5 is *not* significant at conventional levels (p ≈ 0.10). Treat it as a strong hint with a mechanism, not an established relationship.
- It was, however, **predicted in advance** — test 7 recorded the prediction ("if it loses badly in 2024 and is competitive in seasons where QB replacement ran near its mean") and committed it *before* this sweep ran. That is genuine out-of-sample confirmation of the direction, which is worth more than the correlation coefficient alone.

The mechanism is concrete: `value.py` consumes replacement level as a **point estimate**, so the QB tilt is sized as if the long-run mean were known exactly. The season-to-season sd is 1.21 ppg, which at −4.61pp per ppg is a ±5.6pp swing in championship probability. **The single highest-value improvement available is to propagate replacement-level uncertainty into the value function rather than treating it as known** — that shrinks the QB tilt toward the market and should convert a high-variance coin-flip into a smaller, more reliable edge.

**2019 is unusable as a holdout** and this is correct behavior, not a bug: fitting through 2018 alone leaves only 17 matched (ADP, performance) TE pairs, and `fit_rank_curves` raises rather than fitting a curve on that. The usable holdout range is **2020–2024**.

### Test 9 — the gate, re-run through the fixed production path

The ex-ante lineup fix is now **landed in `real_season_champion` itself** (not just in a diagnostic), so `run_backtest` is the trustworthy scorer. `scripts/backtest_multiseason.py` drives it directly. N=400 per season:

| holdout | engine | adp | consensus | random | engine − adp |
|---|---|---|---|---|---|
| 2020 | 12.00% | 7.00% | 7.00% | 0.00% | **+5.00pp** |
| 2021 | 18.75% | 9.75% | 9.75% | 0.00% | **+9.00pp** |
| 2022 | 21.50% | 8.50% | 8.50% | 0.00% | **+13.00pp** |
| 2023 | 4.00% | 9.00% | 9.00% | 0.00% | −5.00pp |
| 2024 | 2.50% | 12.25% | 12.25% | 0.00% | −9.75pp |

**Pooled: +2.45pp, between-season SE 4.27pp. Engine beats ADP in 3 of 5 seasons.**

This replicates the standalone diagnostic (+2.20pp ± 4.06pp) through completely separate scoring code, which is the cross-check that the fix landed correctly rather than merely changing the number.

### Revised status of the gate

The Stage 3 Task 6 gate, as specified ("beat ADP on held-out 2024"), **failed for reasons that were mostly not the engine's fault**: ~6pp of scoring-rule artifact plus a season whose QB replacement sat at the top of its eight-year range. Re-specified as a multi-season ex-ante average, the engine is at +2.20pp ± 4.06pp — **passing in sign, not in significance.**

Recommended next steps, in order:
1. ~~Fix `real_season_champion` to set lineups ex ante.~~ **DONE** — see test 9 and commit `2b845cc`.
2. Task 7, the structure study. Unblocked and built (`scripts/structure_study.py`).
3. Replacement-level uncertainty — **but the original rationale for it was wrong, see below.**

#### Correction: "propagate replacement uncertainty to shrink the QB tilt" is not sound as stated

An earlier version of this document recommended propagating replacement-level uncertainty into `value.py` on the grounds that it would shrink the QB tilt toward the market. **Working through `value.py`, it does the opposite or nothing:**

- For a clear starter, `lineup_marginal ≈ mean − R` is **linear** in R, so `E_R[value] = value(E[R])` exactly. Averaging over uncertainty changes nothing.
- For a marginal player, `_bench_value` is floored at zero (`if over_replacement <= 0: return 0.0`), making it **convex** in R. By Jensen, `E_R[value] > value(E[R])` — the value goes *up*.

Mid-tier QBs sit precisely in that marginal zone (QB10 at 20.47 against R ≈ 18.26 ± 1.21), so the proposed change would if anything **increase** the QB tilt. And §6 already records that variance has *positive* value in this league (60% playoff rate, 3-round bracket), so there is no general argument that hedging helps here either.

The experiment is still worth running — as a **hierarchical bootstrap** (draw a season-level replacement offset per simulated season from the between-season distribution, sample weeks around it) — but it should be run **with no predicted direction** and judged on measured season-to-season swing. Do not repeat the "it will shrink the tilt" claim; it was an assertion, not a derivation.

### Also important: the backtest never tested the recommender

`backtest.py`'s module docstring is explicit and honest about this, but it is easy to miss and it changes what the gate means. `contender="engine"` uses `run_rollout`'s **greedy `draft.value` policy**, not `draft.recommender.recommend_pick`, because the real recommender costs ~5.6 min/pick (N×18×5.6min is infeasible).

The docstring argues the Monte-Carlo layer "does not change how any later pick is chosen" — true — but it *does* choose the immediate pick, at every one of our 18 picks. So **the gate has never been run against the actual recommender.** Note the caveat before getting hopeful: `recommend_pick`'s rollouts use this same greedy policy for all of our *future* picks, so it inherits the same survival blind spot in its continuations — a candidate evaluated as "don't take the kicker now" will often just take the kicker one round later in the rollout, muting the comparison.

### Minor issue noticed in passing

The holdout pool excludes 15 unmatched players, including **Hollywood Brown at rank 86** — genuinely draftable in 2024, and the same name-normalization gap as Stage 1 (he is "Marquise Brown" in the crosswalk). Too small to explain 13.5pp, and it handicaps every contender equally rather than biasing between them, but worth fixing.

## 7c. Task 7 — the structure study. ANSWERED: structure does not matter.

`scripts/structure_study.py`. First three picks (overall 8/13/28) forced into each pattern, taking the best available at the required position by the engine's own value function, unconstrained from round 4. Scored on **real** weekly results across 2020–2024, N=400 per season, opponents paired by seed so every structure faces an identical board. Run on the **fixed Polars runtime** (see §9).

**Championship rate from slot 8, by season:**

| season | 0RB | hero_RB | 2RB_1WR | engine_free | adp |
|---|---|---|---|---|---|
| 2020 | 16.00% | 12.50% | 17.00% | 12.75% | 6.75% |
| 2021 | 21.75% | 19.00% | 17.75% | 19.25% | 9.50% |
| 2022 | 5.75% | 5.75% | 6.50% | 21.75% | 8.50% |
| 2023 | 21.00% | 18.25% | 9.25% | 4.00% | 9.00% |
| 2024 | 3.75% | 7.75% | 9.00% | 2.25% | 12.00% |

**Pooled (equal weight per season):**

```
0RB          13.65%  (between-season SE 3.78pp)
hero_RB      12.65%  (between-season SE 2.68pp)
2RB_1WR      11.90%  (between-season SE 2.29pp)
engine_free  12.00%  (between-season SE 3.92pp)
adp           9.15%  (between-season SE 0.85pp)
```

**Pairwise, paired by seed within season (row − column, pp):**

```
                0RB          hero_RB      2RB_1WR      engine_free
0RB             -            +1.00+-1.39  +1.75+-2.90  +1.65+-5.25
hero_RB         -1.00+-1.39  -            +0.75+-2.26  +0.65+-4.94
2RB_1WR         -1.75+-2.90  -0.75+-2.26  -            -0.10+-4.04
```

### The answer

**There is no measured difference between 0RB, Hero RB, and 2RB:1WR.** Every pairwise gap is ≤1.75pp against SEs of 1.4–2.9pp — all well inside one standard error, let alone two. The ordering (0RB > hero_RB > 2RB_1WR) is not meaningful and should not be reported as a ranking.

This is exactly what Stage 2 predicted and is the main reason to believe it: **a rostered RB starter is worth ~6.4 points/week over replacement and a WR ~6.5** (§6). If RB and WR are worth the same at the margin, then how you order them across three picks cannot matter much. Two independent measurements now agree.

**Do not let anyone re-litigate this by pointing at a single season.** The season-to-season swing dwarfs the structural effect by an order of magnitude:

```
range across seasons (max - min):  0RB 18.00pp   hero_RB 13.25pp
                                   2RB_1WR 11.25pp   engine_free 19.50pp   adp 5.25pp
```

0RB "wins" 2023 by 11.75pp over 2RB:1WR and *loses* 2024 by 5.25pp. Picking a structure off one season is picking noise.

### The finding that actually matters, and its caveat

The largest effects in this table are not between structures at all — they are between **constrained and unconstrained** drafting, and they run in opposite directions in different seasons:

```
season  QB replacement  engine_free  mean(structures)  free - structures
2020         19.31         12.75%         15.17%          -2.42pp
2021         16.83         19.25%         19.50%          -0.25pp
2022         16.36         21.75%          6.00%         +15.75pp
2023         18.37          4.00%         16.17%         -12.17pp
2024         20.11          2.25%          6.83%          -4.58pp
```

Forcing RB/WR in the first three rounds means **not taking the early QB**. In 2022, when QB replacement was at its lowest (16.36) and elite QBs were most valuable, the unconstrained engine beat every structure by 15.75pp. In 2023 and 2024 the same freedom cost it 12.17pp and 4.58pp.

**Caveat, and it is a real one:** r = −0.645 at n=5 is not significant, and 2021 breaks the pattern (second-lowest QB replacement, yet unconstrained drafting bought nothing). This is *consistent with* §7b test 8's r = −0.810 on a different quantity, but it is not independent confirmation — both are five points from the same five seasons. **Treat the QB-timing effect as the live hypothesis it is, not as an established result.**

### Practical guidance for the 2026 draft

1. **Do not pre-commit to a draft structure.** No structural rule was worth more than ~1.75pp, and none of that survives its error bar. Take the best available player.
2. **The decision that carries real weight at picks 8 and 13 is whether to spend one on a quarterback**, and its payoff swings ±15pp depending on something unknowable in advance (how good streamable QBs turn out to be that year).
3. ~~Size the QB bet down.~~ **SUPERSEDED — measured and contradicted, see §10.** Sweeping a QB-replacement penalty across 2020-2024 shows the mean championship rate *falls* as the tilt shrinks (11.47% at delta=0 down to 9.27% at delta=5). It buys lower season-to-season variance, which is worthless for a draft you run once. Leave the value function alone and take the best available player.

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

## 9. Environment: Polars was running unsound, and this is now fixed

**Found 2026-08-04, and it invalidates nothing measurably but could have.**

The venv Python is **x86_64 running under Rosetta on an Apple M1 Pro**. Polars' x86_64 build requires AVX/AVX2/FMA/BMI1/BMI2/LZCNT/MOVBE, none of which Rosetta emulates. Polars detects this at import and warns that continuing "will likely result in a crash." That warning was correct.

**The symptom.** A multi-season run died in `ids._load_prepared_ids` with `ShapeError: Series length 12269 doesn't match the DataFrame height of 12470`. That diagnosis is worth keeping:

- The failing call is a **pure column select/cast**, which cannot change a frame's length.
- So the *input* frame was internally ragged — one column at 12269 while the frame claimed 12470. Polars' invariants make that impossible in sound operation. It is a memory-safety symptom, not a data problem.
- It is **intermittent**: the identical fit succeeded minutes earlier, and `load_crosswalk()` returns a stable `(11897, 5)` six times out of six in isolation. Neither number in the error matches the correct height.
- It ran with **nothing else competing**, so it is not the concurrency segfault in open item 8 — same root cause, separate instance.

**A process failure worth not repeating:** `POLARS_SKIP_CPU_CHECK=1` was being set to quiet the warning in command output. That flag bypasses a correctness-and-stability guard, and it was used for log tidiness. In a project whose stated primary risk is silent corruption (§8), suppressing that warning was exactly the wrong instinct. **Do not set it.**

**The fix, applied:** `pip install 'polars[rtcompat]==1.43.2'`. Same Polars version, purely additive — it installs `polars-runtime-compat` beside the existing runtime, no downgrade, no removal, reversible with `pip uninstall polars-runtime-compat`. Verified: imports clean under `warnings.simplefilter('error')` and returns the correct crosswalk shape.

**How much to distrust earlier numbers.** The failure mode here is a loud crash rather than quiet bad arithmetic, and the headline results replicated independently — the multi-season gate came out at +2.20pp through one scoring implementation and +2.45pp through a separate one, which corrupted inputs would not reliably produce. Direct evidence: the structure study's 2020 row from the *unsound* run (0RB 15.50 / hero 12.00 / 2RB 15.75 / free 12.25 / adp 7.00) versus the sound re-run (16.00 / 12.50 / 17.00 / 12.75 / 6.75) — differences of 0.25–1.00pp, comfortably inside Monte-Carlo noise at N=400. So the earlier conclusions look sound; they simply could not be *demonstrated* sound, which is why §7c was re-run before being reported.

**Still worth doing:** switch to a native ARM Python (open item 8). `rtcompat` makes the current setup correct; a native build would make it correct *and* substantially faster, which matters given the suite already takes ~12 minutes.

## 10. The runtime can produce silently WRONG NUMBERS, not just crashes

**Found 2026-08-04, immediately after §9. This is the most serious open issue in the project and it outranks every modelling question below it.**

§9 recorded that Rosetta-hosted Polars crashes intermittently, and that `polars[rtcompat]` was installed as a mitigation. It then emerged that (a) rtcompat does **not** eliminate the crashes, and (b) worse, the same instability can produce **plausible, non-crashing, wrong results**.

### The evidence

`scripts/qb_tilt_sweep.py` was run over five holdout seasons in one process. Its `delta=0` contender is by construction identical to the plain engine — it adds 0.0 to QB replacement — so it must reproduce `engine_free`. Across seasons it did not:

| season | engine (backtest) | engine_free (study) | qb_tilt delta=0, 5-season run |
|---|---|---|---|
| 2020 | 12.00% | 12.75% | 11.00% ✓ |
| 2021 | 18.75% | 19.25% | **5.33%** ✗ |
| 2022 | 21.50% | 21.75% | **8.33%** ✗ |
| 2023 | 4.00% | 4.00% | 2.33% ✓ |
| 2024 | 2.50% | 2.25% | 2.67% ✓ |

Seasons 2 and 3 were wrong by 14pp and 13pp — roughly 6 sigma. The code was then cleared of blame, in this order:

1. **The drafts are byte-identical.** `delta=0` vs `engine_free`, 2021, seeds 1 and 2: all 180 picks identical, ours and every opponent's. The policy path is provably equivalent.
2. **The rate is identical in a clean process.** Both score **16.67%** on 2021 at N=150, in the same interpreter.
3. **A fresh 2-season re-run is correct.** Same script, same seeds: 2020 → 12.00%, 2021 → 16.67%.

So the identical code produced 5.33% once and ~17% every other time. **It is nondeterministic, and it is not test-order or script logic.**

A corroborating detail worth keeping: in the bad run, mean QB count per roster for `delta=0` was **3.95**; in the good run it is **3.00**. The *drafts themselves* differed, which means the corruption hit the fitted inputs (rank curves / replacement / tier shapes — the Polars-heavy fitting step), not the scoring. That is consistent with data-level corruption during model fitting.

### What this means

This is precisely the failure mode §8 says is the project's main risk — confident, plausible, wrong output with no error raised — except the source is the *runtime*, not the code, so no amount of code review or validation logic will catch it.

**Switching to a native ARM Python is no longer a performance nice-to-have. It is a correctness requirement and the top priority.** `rtcompat` is a partial mitigation only.

### How much to distrust existing results

Do not panic-discard everything; do check before quoting. The load-bearing results were each measured **more than once through independent code paths**, which is what makes them believable:

- **Multi-season gate:** +2.20pp via the standalone ex-ante diagnostic and +2.45pp via production `run_backtest` — separate scoring implementations, separate runs.
- **Structure study:** the 2020 row matched to within 0.25–1.00pp between an unsound run and a sound re-run.
- **Hindsight effect (5.75pp):** measured at N=400 and consistent with the independent per-season pattern in test 8.

**Rule going forward: no single-run number is trustworthy on this hardware.** Anything that matters must be replicated in a separate process, ideally by a different code path, and a disagreement between two runs of identical code should be treated as a runtime fault until proven otherwise — not explained away.

### RESOLVED 2026-08-04: native ARM Python installed, corruption confirmed and fixed

`uv python install cpython-3.11-macos-aarch64-none` (no sudo needed). The x86_64 venv has been **deleted**; **`.venv` is now native arm64**, so every existing `.venv/bin/...` reference still works. It was rebuilt from scratch rather than renamed — renaming a venv leaves stale absolute paths in `pyvenv.cfg` and every console-script shebang. **`.gitignore` now uses `.venv*/`, not `.venv/`**: a venv created as `.venv-arm` did not match the old pattern and 6,523 files (587MB) were committed by a `git add -A` before being caught and amended out pre-push. Prefer explicit `git add <paths>`. Note two dead ends: `/opt/homebrew` exists but has no `brew` binary, and `uv` is itself an x86_64 binary so `uv python list` only advertises x86_64 — pass the explicit `-macos-aarch64-none` key.

Verified native: `platform.machine() == 'arm64'`, Polars 1.43.2 imports clean under `warnings.simplefilter('error')`, and only the standard `polars-runtime-32` is installed — the `rtcompat` shim is gone, because there is nothing left to compensate for.

**Suite: 362 passed in 7:37**, versus 11:30 under Rosetta — a 34% speedup that also addresses open item 6.

**The corruption is confirmed as a runtime fault, and it is fixed.** Re-running the identical sweep on ARM, the `delta=0` control arm — which is identical by construction to `engine_free` — now reproduces it in every season:

| season | delta=0 on ARM | engine (x86 backtest) | engine_free (x86 study) | the corrupted run |
|---|---|---|---|---|
| 2020 | 12.00% | 12.00% | 12.75% | 11.00% |
| 2021 | **17.67%** | 18.75% | 19.25% | **5.33%** |
| 2022 | **20.33%** | 21.50% | 21.75% | **8.33%** |
| 2023 | 4.00% | 4.00% | 4.00% | 2.33% |
| 2024 | 3.33% | 2.50% | 2.25% | 2.67% |

Mean QB per roster is back to **3.00** (the corrupted run said 3.95). Every season now agrees within Monte-Carlo noise.

**The multi-season gate also replicates on the sound runtime: engine − adp = +2.86pp (between-season SE 4.15pp), beating ADP in 3 of 5 seasons**, against +2.45pp ± 4.27pp measured on x86_64. The Stage 3 conclusions stand.

### The QB-tilt experiment — ANSWERED, and it contradicts §7c's recommendation

Now trustworthy (control arm verified above). N=300/season, 2020–2024. `delta` is added to QB replacement **in the drafting value function only**; scoring always uses true replacement levels.

| delta | mean | sd | min | max | range | mean QB drafted |
|---|---|---|---|---|---|---|
| 0.0 | **11.47%** | 7.73 | 3.33 | 20.33 | 17.00 | 3.00 |
| 1.0 | 11.00% | 7.07 | 3.00 | 19.67 | 16.67 | 2.66 |
| 2.15 | 9.73% | 7.79 | 2.00 | 22.33 | 20.33 | 1.38 |
| 3.0 | 9.67% | **4.90** | 4.00 | 17.00 | 13.00 | 1.03 |
| 5.0 | 9.27% | **4.07** | 3.00 | 14.33 | 11.33 | 0.48 |
| adp | 8.62% | — | 6.67 | 12.00 | 5.33 | 1.97 |

**Sizing the QB bet down does not help, and §7c's recommendation to do so is not supported.** The mean *falls* monotonically as the tilt is reduced (11.47 → 9.27), while the season-to-season spread also falls (sd 7.73 → 4.07). It is an ordinary risk/return trade, with no delta that improves both.

**The decisive argument is that the variance reduction is not worth buying.** You draft *once*. For a single season, the objective is expected championship probability, and `delta=0` maximises it. Between-season spread here is not diversifiable risk — it is uncertainty about which kind of season you will get, and shrinking it lowers the expectation without protecting anything you can bank.

**Honest caveat: none of the mean differences are significant.** With 5 seasons the between-season SE is ~3.5pp, so 11.47 vs 9.67 is well inside noise. The *variance* ordering is the more credible signal, and it is the one that does not matter for a one-shot draft. The defensible reading is **"no evidence that shrinking the QB tilt helps; mild evidence it costs"** — so leave the value function alone.

This supersedes §7c's closing recommendation, which was reasoning rather than measurement. The reasoning was plausible and the measurement disagrees; the measurement wins.

`scripts/qb_tilt_sweep.py` is committed and its mechanism is verified (raising QB replacement does reduce QB count: 3.00 → 1.75 at delta=2.15 on 2024). **Its five-season results are void** and are deliberately not recorded here. Re-run it on a native ARM Python before drawing any conclusion about whether to size the QB bet down. The question from §7c remains open.
