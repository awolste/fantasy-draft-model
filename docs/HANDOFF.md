# Handoff — 2026 Fantasy Draft Optimizer

**Last updated:** 2026-08-22
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
| 4 | Live draft assistant | **Complete** (live-only; + survival columns, position caps, scarcity tie-break) |
| — | Draft-day aids | **Complete** — precomputed 6-round pick tree + interactive explorer (§14) |

**All four stages are merged and pushed to `origin/main`. 437 tests pass in ~3:20** on the native arm64 venv (both numbers improved 2026-08-11: `recommend_pick` is 10.7x faster with bit-identical output, §15).

> **Read §7e before trusting any performance claim on this page.** The engine wins 2020-2022 and *loses* 2023 and 2024, and the pooled edge over ADP averages those two opposite regimes together. The two seasons most like 2026 are the two it loses. This is open item 1 and it is unresolved.

See §9/§10 for why the runtime was replaced (open item 8 closed) and §11 for the live assistant.

Stage 3 is **complete**. Task 7, the structure study, is **answered, in two parts**: among RB/WR orderings draft structure does *not* matter (0RB, Hero RB and 2RB:1WR are statistically indistinguishable), but **taking a QB or elite TE in rounds 1-3 is measurably worse** — 9 of 9 comparisons favour a skill-only start, one at 3.6 sigma. See §7c.

Task 6 (the backtest) is done. Its original verdict was: *the engine loses to naive ADP-following on real 2024 results by 13.5pp ± 1.65pp*. **That verdict has since been overturned — see §7b tests 6-9.** Two things were wrong with it. (a) ~6pp of the deficit was an artifact of the scoring rule: `real_season_champion` set lineups with perfect hindsight, which pays FLEX depth a premium no real manager could collect. That is now fixed. (b) 2024 is the worst of the five usable holdout seasons for this engine. Re-run across 2020–2024 through the fixed production path, **the engine beats ADP in three of five seasons and averages +2.45pp against a between-season SE of 4.27pp** — not broken, but not demonstrated either. **Superseded 2026-08-12, §7e: that pooled average conceals a sign flip.** The engine wins 2020-2022 and loses 2023 (−8.67pp) and 2024 (−13.67pp), and the ADP baseline it is measured against never drafts a quarterback.

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

**Nothing is blocking a deliverable. Stages 1-4 are complete, merged and pushed.**

1. ~~Task 6: the engine loses the backtest by 13.5pp.~~ **RESOLVED — the verdict was wrong.** ~6pp was a hindsight-scoring artifact (now fixed) and 2024 was the worst of five usable holdouts. Multi-season, ex-ante, through the production path: **engine − ADP = +2.86pp, between-season SE 4.15pp, beating ADP in 3 of 5 seasons.** §7b tests 6-9, §10. **Do not quote that figure as a current result — see §7e**: it averages two opposite regimes, and the engine loses both of the most recent seasons.

2. ~~Task 7: the structure study has not been run.~~ **RESOLVED, in two parts.** Among RB/WR orderings, structure does not matter (all pairwise gaps ≤1.75pp against SEs of 1.4-2.9). But **taking a QB or elite TE in rounds 1-3 is measurably worse** — 9 of 9 comparisons favour a skill-only start, strongest at 3.6 sigma. §7c and its 2026-08-09 extension.

**Known defects and unresolved questions:**

3b. **The opponent model is too random at the top of the board.** `DEFAULT_TEMPERATURE = 3.0` flattens the pick distribution enough that it thinks consensus top-5 players routinely fall to our pick 8. Measured, 1000 sims of the first 7 picks against the live 2026 board:

```
Jahmyr Gibbs        adp  1.6    survives to 8:  9.2%
Bijan Robinson      adp  2.0                   14.0%
Puka Nacua          adp  2.9                   17.5%
Ja'Marr Chase       adp  3.9                   25.5%
Christian McCaffrey adp  5.3                   31.1%
Jonathan Taylor     adp  7.5                   54.5%
```

In real drafts those top-4 numbers are close to zero — nobody passes on the consensus #1 seven times. **The owner spotted this from a mock-draft roster before it was measured.**

**Why it happens, and why the temperature is not simply "wrong":** 3.0 was chosen to reproduce *aggregate* behaviour across all 180 picks, where reaching and falling genuinely do happen, and it does that well enough that the engine beats ADP in 3 of 5 backtested seasons (§10). But one temperature cannot fit the whole draft: the first five picks are near-deterministic while round 12 is nearly a free-for-all, so a single fitted value necessarily over-randomises the part of the draft with the least real randomness.

**What it does and does not affect.** It does *not* corrupt live recommendations much: on draft day the board is entered from reality, so the opponent model only drives the rollouts *after* our pick, plus the survival column (which will read optimistically early). It **does** make `scripts/mock_draft.py` systematically generous — treat mock rosters as informative about *shape*, never about which specific players you can expect to get.

**The fix is a pick-dependent temperature** — tight early, loose late — refitted against the eight seasons in `league_drafts` and **validated against the backtest**, not tuned until survival percentages look plausible. That is the same class of work as item 3 (a real refit, not a patch), and the same caution applies: this project's history says a change that makes numbers *look* better is exactly the kind that needs a measured gate.

3. **QB=3 and K=2 are cap-bound in every rollout.** The value function still ranks a third QB and second kicker above better alternatives; a policy guard (`MAX_EXTRA_BENCH_NO_FLEX`) is the only thing stopping it. That is masking, not fixing, and costs ~2 roster spots versus opponents' 1.7 QB / 1.2 K. `value.py` has been through three fix rounds; a fourth patch is probably the wrong move — consider whether greedy marginal value is the right policy at all.
4. **Between-rollout variance appears state-dependent and is unresolved.** Measured 3.50pp at pick 8 (n=50) but 5.97pp in an independent 10-seed check at a different state. The staleness guard catches drift over time but will not tell you the allocation is too tight at some other draft state.
5. ~~A recommendation costs ~5.6 minutes, too slow for a live draft.~~ **RESOLVED in Stage 4 (§11).** Full budget is now ~197s on native ARM, and the live tool uses a measured reduced budget (12×16×300) costing 35.7s at our worst pick and 13.1s at our last. Precompute was built and dropped after measuring a 9–17% hit rate.
6. **Test suite takes ~3:20** for 437 tests (was ~7:40 before the 2026-08-11 speedup, §15; ~12 min under Rosetta). Still dominated by the pick-8/pick-13 real-data tests running full production budgets; pinning a smaller explicit budget there is still worth doing.
7. **`sources/nflverse.py` imports `models/kicking.py`** — a Stage 1 → Stage 2 layering inversion. Cheap to fix by moving `kicking.py` beside `scoring.py`.
8. ~~Python runs under Rosetta x86-64 emulation.~~ **RESOLVED 2026-08-08.** Native arm64 CPython 3.11.15 installed via `uv python install cpython-3.11-macos-aarch64-none`; `.venv` is now native and the x86_64 one is deleted. Suite went 11:30 → ~7:40, and the intermittent native crashes and silent numeric corruption are gone (§10b, §10c). Never set `POLARS_SKIP_CPU_CHECK=1`.

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

### Extension (2026-08-09): QB-early and TE-early structures — the tilt is a LEAK

The original study only compared RB/WR orderings, which is why it came back null: Stage 2 had already measured RB and WR starters as worth nearly the same over replacement. It never tested the structures the engine actually *wants* to draft. Three were added and run through the identical harness (N=400/season, 2020–2024, opponents paired by seed):

```
pooled          rate    between-season SE
0RB            15.45%         3.89
2RB_1WR        13.55%         1.76
engine_free    12.10%         2.94
hero_RB        11.70%         2.91
adp            10.10%         0.76
QB_early        9.60%         1.96      <- QB/RB/WR
QB_and_TE       8.85%         3.54      <- QB/TE/WR
TE_early        8.70%         1.41      <- TE/RB/WR
```

**All 9 skill-vs-QB/TE pairwise comparisons favour the skill-only start.** The strongest:

```
2RB_1WR - TE_early   +4.85pp  (SE 1.34)   3.6 sigma   <- survives Bonferroni (9 tests, needs ~2.8)
0RB     - TE_early   +6.75pp  (SE 3.15)   2.1 sigma   <- does NOT survive; discount it
group mean: skill-only - QB/TE = +4.52pp (SE 3.54)    <- not significant on its own
```

**Read this honestly.** With 9 comparisons, ~0.5 false positives at 2 sigma are expected, so only the 3.6-sigma result should be leaned on. The group mean is directionally clear but not significant. What raises confidence beyond the arithmetic is that **9 of 9 comparisons point the same way**, and that a **known bias was working in QB_early's favour**: open item 3b means the opponent model lets top-5 players fall to pick 8 far too often, so forcing a QB at the top hands us elite RB/WR afterwards more than reality would. QB_early lost with the scales tipped its way.

**Conclusion: the engine's QB/TE positional tilt is a leak in rounds 1-3, not an edge.** This is the first *direct* measurement of it — every prior signal (§7b's reaching, the late-round candidate lists full of capped positions, the two-TE mock roster) was indirect. Note `engine_free` at 12.10% lands *below* both 0RB and 2RB_1WR precisely because, left alone, the engine sometimes takes the QB or TE it should not.

**Practical guidance, superseding the "no structural rule" reading of §7c for the QB/TE case specifically:** spend the first three picks on RB and WR in whatever order the board offers. Do not take a quarterback or an elite tight end in rounds 1-3. Among RB/WR orderings there is still no measurable difference.

**Caveat that has not gone away:** n = 5 seasons. Per-season swings (0RB ranges 5.25%-25.50%) remain far larger than any between-structure effect, so this is guidance, not a law.

### Resolution (2026-08-09): the tilt is real, localised — and neither fix wins

`scripts/diagnose_tilt.py` **localised the mechanism**, which nothing before it had done. `draft.value` prices candidates against the **waiver** replacement level — what a slot is worth if you never draft anyone there at all. But we hold 18 picks and always fill every slot, so the real fallback for passing on a player is *the best one still on the board when we get round to it*. That gap is not uniform across positions, which is why it shows up as a positional tilt rather than a harmless constant:

```
pick  pos     VOR    true cost of waiting
   8  QB     9.19            0.00     <- Allen is still there at pick 33
   8  TE     8.35            0.00     <- McBride too
   8  RB     7.60            0.95
  13  RB     6.65            3.73
```

Top 40 by `draft.value`: **QB 6, TE 4**. By ADP: **QB 1, TE 1**. The engine takes a mean of **1.32 QB/TE in rounds 1-3**.

`scripts/measure_tilt_fix.py` then tested two fixes, N=300/season, 2020-2024, paired by seed. **Both failed and `value.py` was left unchanged:**

```
paired vs engine:  baseline -4.40pp (SE 2.06, 2.1 sigma)   <- harmful, lost all 5 seasons
                   guard    +0.47pp (SE 3.18, 0.1 sigma)   <- nothing
```

- **`baseline`** replaced the waiver level with the "last starter drafted" level. Principled, and it *lost every season*. (It also raises the share of players priced at exactly 0.00 from 68% to 83%, so late rounds decay toward arbitrary tie-breaks.) Not in `src/` — it lives in the script that measured it.
- **`guard`** banned QB/TE from rounds 1-3, the fix the 3.6σ structure result appears to support. It swings **−11.00pp (2022) to +7.33pp (2023)** and averages to nothing.

**2022 is the tell.** It has the *most* early QB/TE picks (1.60) and the guard's *worst* season, because 2022 is when QB replacement bottomed at 16.36 and the early quarterback genuinely paid. **A rounds-1-3 ban is a bet on how good streamable QBs turn out to be that year — not knowable on draft day.** This is §7c's own wall, now confirmed against the engine's free choices rather than against forced structures.

**Do not re-read the 4.85pp headline as available upside.** It is measured against a *forced* TE-first structure. Unforced, `engine_free` − `2RB_1WR` is ~1.45pp and inside noise. The ceiling was always small.

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

## 7d. Opponent-model temperature — FIXED 2026-08-09 (item 3b)

The flat `DEFAULT_TEMPERATURE = 3.0` is replaced by `TEMPERATURE_BY_ROUND`, a per-round table fit by maximum likelihood.

**The defect was in both directions at once, which is why no single number could fix it.** Measured against seven real drafts (`scripts/diagnose_temperature.py`), best-available ADP rank at our own picks:

```
 pick        REAL     flat 3.0    schedule
    8         5.1         4.4         5.2     <- was too RANDOM early
   13         7.4         8.1         8.8
   28        20.0        21.8        18.5
   33        25.3        26.6        22.1
   48        32.1        41.4        31.7     <- was too RIGID late
```

Flat 3.0 left the consensus ADP-1 player on the board at pick 8 in **9%** of simulations; that never once happened in seven real drafts. Lowering the flat value fixes pick 8 and makes pick 48 worse.

**The fit** (`scripts/fit_temperature.py`): maximum likelihood per round over all 1,125 usable picks, with the reach model and `roster_decay` held fixed so temperature cannot absorb a misfit from either. T climbs 2.30 (round 1) to ~24 (round 17).

**Validation, leave-one-season-out:** held-out log-likelihood **+1.72/pick, better in 7 of 7 seasons**. Independently, on a statistic it was *not* fit against, mean absolute error versus the real draft board halves (2.76 → 1.32 ranks).

**Why not the simpler linear form.** `T = 1.95 + 1.16*round` matches on aggregate held-out LL (+1.73 vs +1.72) with 2 parameters instead of 17 — but its intercept is `T(1) = 3.11`, outside round 1's own 95% CI of **[1.85, 2.90]**. Aggregate LL is dominated by the ~80% of picks in rounds 6-17 where no decision is at stake; round 1 is our pick 8. The per-round table generalised just as well, so the smoother form bought nothing where it mattered.

**Two caveats, both measured and both against the change:**
- Nearly all of the +1.72 is late-round. Restricted to **rounds 1-3** — the only picks this tool advises on — the gain is **+0.065/pick (SE 0.022)**, better in 6 of 7. Real, ~3σ, small. This is a better-calibrated opponent model, not a transformation of draft-day advice.
- **Top-1 accuracy falls 0.45pp**, which is exactly the criterion the flat 3.0 was originally tuned on. Judged the wrong criterion — rollouts *sample* from this distribution rather than taking its argmax — but recorded rather than buried.

### The gate did NOT improve, and there is a leak in the re-run

```
season         engine         adp    engine-adp
2020           20.67%       9.67%      +11.00pp
2021           20.00%      10.00%      +10.00pp
2022           22.67%      15.00%       +7.67pp
2023            7.33%      16.00%       -8.67pp
2024            7.00%      20.67%      -13.67pp

engine - adp = +1.27pp (between-season SE 5.17pp), 3 of 5 seasons
   was:        +2.86pp (between-season SE 4.15pp), 3 of 5 seasons
```

**Statistically unchanged** (both well inside a ~5pp SE), and nominally slightly worse. Do not claim this fix improved the engine's edge; it did not. What it improved is the fidelity of the simulated league, which is its own job.

**Leak, named because it is mine:** `backtest.fit_holdout_context` correctly refits the *reach* model through the prior season only, but `TEMPERATURE_BY_ROUND` is a module constant fit on all of 2018-2024 — **including every holdout season**. The gate numbers above are therefore leaky. Two things bound the damage: it does not touch draft-day use (2026 is outside the fitting range), and it cannot have flattered us, since both contenders face the same simulated opponents and it was **ADP** that gained most (2024: 10.00% → 20.67%). Fixing it properly means threading a per-holdout schedule through `run_rollout` and moving the likelihood replay out of `scripts/` into `src/` — recorded as an open item, not half-done.

**Also worth noticing in that table:** ADP's own championship rate rises monotonically across holdouts, 9.67% → 20.67%. *(Followed up 2026-08-12, §7e: that trend is caused by this very schedule — ADP is flat under the old flat temperature. The guess here that it "predates this change" was wrong.)*

## 7e. The engine collapses in the two most recent seasons — 2026-08-12

**This displaces every other open item.** It is the only finding on this page that bears directly on whether to trust the tool on draft day, and it survives both opponent models.

```
engine championship rate      2020-22   ->   2023-24
  flat temperature (old)       17.44%        6.50%
  fitted schedule (new)        21.11%        7.17%
```

Per season, with the fitted schedule:

```
season   engine      adp    engine-adp
2020     20.67%    9.67%      +11.00pp
2021     20.00%   10.00%      +10.00pp
2022     22.67%   15.00%       +7.67pp
2023      7.33%   16.00%       -8.67pp
2024      7.00%   20.67%      -13.67pp
```

**The headline "+1.27pp edge over ADP" is an average across two opposite regimes.** The engine wins 2020-2022 and loses 2023 and 2024 — and those two are the seasons most like 2026. Averaging them produces a small positive number that conceals a sign flip. Do not quote the pooled figure without this table beside it.

Nothing yet explains the collapse. It is **not** the temperature schedule (it predates it, at a similar magnitude) and **not** ADP coverage (see below). Candidates worth testing, in no particular order: something in the 2023+ player pools that the value function handles badly; the rank-curve/replacement fits degrading as they take on more seasons; or 2023-24 simply being "chalk" years where consensus was accurate and a contrarian engine had nothing to find.

### The "ADP rises monotonically" trend was mine, not the data's

Recorded here because it was briefly promoted to a top open item and is now retired.

```
                    2020   2021   2022   2023   2024   corr w/ season
adp, flat temp      8.00  11.67   8.33  11.33  10.00      +0.345
adp, fitted sched.  9.67  10.00  15.00  16.00  20.67      +0.967
```

ADP-following was flat under the old flat temperature. The **per-round temperature schedule (§7d) is what made it trend** — possibly compounded by that schedule's known leak (fit on 2018-2024, including every holdout; open item 2a). This is not evidence about the seasons; it is evidence about a change I made.

### ADP coverage is incomplete in every season, and badly so in 2022 — upstream

- Each season has only **~200 real ADP rows against a ~637-player pool**. Everyone else gets an ADP *extrapolated* from a linear fit of ADP on board rank (`rollout._extrapolate_unmatched_adp`). This is documented and reasonable, but it means most of the board's ADP is synthetic in every backtest season.
- **2022 is truncated to 157 rows** — fewer than the 180 picks a draft needs — with K at 5 and D/ST at 6 against ~14 apiece elsewhere. That season's late-round ADP is almost entirely extrapolated.
- **This is upstream, not an ingest bug.** Queried live 2026-08-12: `fantasyfootballcalculator.com/api/v1/adp/ppr?year=2022` returns exactly 157 players (PK 5, DEF 6), matching our stored data row for row. 2021 returns 211 and 2023 returns 202. The same API already returns nothing at all for 2025 (see `sources/ffcalculator.py`), so upstream gaps are its established behaviour rather than a surprise.
- Coverage does **not** trend across seasons, so it cannot explain the ADP trend above, and 2022 sits mid-sequence.

### The `adp` and `consensus` contenders are the same policy

They return byte-identical rates in all five seasons. `rankings_holdout` is built by sorting that season's ADP (`backtest._adp_ranked_frame`), so "best available by consensus rank" and "best available by ADP" are the same ordering. The summary table presents them as two baselines; they are one. Not a bug, but do not read their agreement as corroboration.

### Dig into 7e, round 1 — 2026-08-12. Two structural findings, one dead end.

`scripts/diagnose_season_break.py` and the allocation/sweep runs behind it.

**1. Within a position, the model's ordering IS ADP's ordering.** Not similar — identical. The model-vs-ADP top-quintile hit rate differs by **+0% in every position in all five seasons**, and within-position Spearman differences are ±0.02. This is structural, not a five-point inference: for a historical holdout, `rankings_holdout` is built by sorting that season's ADP (`backtest._adp_ranked_frame`), and `PlayerDistribution.mean` is a monotone function of positional rank, so the two orderings cannot disagree.

**The consequence is large: in the backtest the engine cannot out-pick ADP within a position. Positional allocation is its only lever.** Any explanation of §7e has to be an allocation story, and any "better projections" story is ruled out by construction. (The live 2026 pool uses real FantasyPros ECR, which is *not* built from ADP, so this is a property of the backtest, not of draft day — but it means the backtest cannot validate projection quality at all.)

**2. The ADP baseline never drafts a quarterback**, and that is a flaw in the project's load-bearing number.

```
season  arm      1st QB rd    QB   RB   WR   TE
2023    engine        1.6    3.0  4.0  8.0  2.0
2023    adp          45.5    0.9  6.9 10.1  0.2
2024    adp          62.4    0.5  7.1 10.3  0.1
```

`adp_pick_policy` takes the lowest-ADP player available with no roster constraint, so over 18 rounds it takes almost only RB/WR and then starts a **free replacement-level QB (~18 ppg)** every week. That is not a manager following ADP; it exploits the replacement mechanic. `engine - adp` compares against a strategy no human would run, and the baseline gets *stronger* the more generous replacement level is. Fix before quoting the gate again: give the ADP policy the same roster constraints a real manager has.

**3. Dead end: the QB tilt is NOT what breaks 2023-24.** The obvious hypothesis — the engine drafts 3.0 QBs every season and takes the first one earlier each year (round 4.0 → 2.3 → 1.2 → 1.6 → 1.4) while QB replacement rose (16.36 in 2022 → 20.11 in 2024) — is wrong. Re-pricing QB replacement by +2.15 in the value function cuts QBs drafted from 3.00 to 0.99 and makes the collapse **worse**:

```
season    delta=0.0   delta=2.15
2022        23.00%      17.00%
2023         8.50%       6.00%
2024         8.50%       7.50%
```

Recorded so it is not retried. Note this is the per-season view §10c never printed; its pooled mean averaged the two regimes and hid both the collapse and this result.

**Still unexplained.** The engine's allocation is near-constant across seasons (3 QB, 2 TE, ~4 RB, ~8 WR) while ADP's swung from RB-heavy (RB 10 / WR 6 in 2020-22) to WR-heavy (RB 7 / WR 10 in 2023-24). The next thing to measure is where the realized starting-lineup points actually go, by position, engine vs ADP, in a season it wins (2022) against one it loses (2023) — that says whether the engine is drafting a worse roster or a differently-shaped one that scores the same and wins less.

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
.venv/bin/pytest                          # ~3:20
```

### Draft-morning refresh (no ESPN cookies needed)

ADP and rankings are **snapshots, not live fetches** — the app reads `data/*.parquet`. They go stale fast in August: measured 2026-08-22 against a 14-day-old snapshot, the 2026 board had moved a mean of 6.0 places with individual swings of 20-30. Refresh just those two, which needs neither ESPN cookies nor the nflverse download:

```bash
.venv/bin/python -c "from ffdraft.sources import fantasypros, ffcalculator; fantasypros.ingest(); ffcalculator.ingest()"
```

**That will trip `CacheStaleError` on the next run — expected, not a break.** Four caches fingerprint against the refreshed inputs, and the suite fails ~42 tests in 15s until they are refit. Refit them explicitly (3s), then rebuild the tree (~6 min):

```bash
.venv/bin/python -c "
from ffdraft import store
from ffdraft.models.availability import load_or_fit_availability_rates
from ffdraft.models.replacement import load_or_fit_replacement_values
from ffdraft.models.tier_shape import load_or_fit_tier_shapes
from ffdraft.models.rank_curve import load_or_fit_rank_curves
load_or_fit_availability_rates(force_refit=True)
load_or_fit_replacement_values(force_refit=True)
w = store.read('weekly_stats'); a = store.read('adp_history')
load_or_fit_tier_shapes(w, force_refit=True)
load_or_fit_rank_curves(a, w, store.read('rankings_2026'), force_refit=True)
load_or_fit_rank_curves(a, w, force_refit=True)"
.venv/bin/python scripts/build_pick_tree.py && .venv/bin/python scripts/build_pick_tree_page.py --out pick_tree.html
```

**The sources are current; only our snapshot ages.** Verified 2026-08-22 by reading the API's own metadata rather than assuming: Fantasy Football Calculator reports a rolling **7-day window (2026-08-15 → 2026-08-22) over 7,288 drafts**.

**Both sources are PPR and both assume 4-point passing TDs.** `ADP_URL` is `/api/v1/adp/ppr` (the API echoes `type: PPR`) and ECR is `ppr-cheatsheets.php`; the league is full PPR (`rec: 1.0`) but pays **6** for a passing TD. That divergence is the project's designed edge and is corrected in the player model, not in the sources — see `sources/fantasypros.py`. It is also the origin of the QB tilt (§7c, §7e).

Caches are fingerprinted against their inputs and **raise `CacheStaleError`** if the underlying data changed — deliberately loud rather than silently refitting, because a silent refit inside a rollout loop would be an invisible performance cliff. Rank-curve caches are namespaced by rankings context so the 2026 pool and a historical backtest pool coexist without evicting each other.

## 10. Working agreement that has been in force

- Subagent-driven execution: fresh Sonnet subagent per task, then spec and quality review. Small, well-specified modules get a combined review; risky ones get separate passes.
- Reviews are dispatched in parallel (read-only); implementers never run concurrently (git collisions).
- Findings that change a documented number get written back into `docs/superpowers/plans/README.md` so they survive the session.
- Numbers reported by subagents are spot-checked against raw data before being believed. This has caught real errors more than once, including one where the coordinator's own first check used the wrong baseline and nearly flagged a correct model as broken.

## 10b. Environment: Polars was running unsound — RESOLVED

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

## 10c. The runtime could produce silently WRONG NUMBERS — RESOLVED

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

> **The conclusion survives; this argument for it does not (added 2026-08-09).** Read the `mean QB drafted` column: it goes **3.00 → 0.48**. At the large deltas the engine fields *no quarterback at all*, so those arms lose because of an empty starting slot, not because the tilt was removed. The experiment confounds "don't take a QB early" with "don't take a QB". It also corrects one position in isolation, leaving TE inflated — which moves the tilt rather than removing it. Do not cite this table as evidence that the tilt is harmless or that `value.py` should not be touched; for that, cite §7c's resolution, which tested a coherent, all-position correction and a rounds-1-3 timing guard and found both wanting.

**The decisive argument is that the variance reduction is not worth buying.** You draft *once*. For a single season, the objective is expected championship probability, and `delta=0` maximises it. Between-season spread here is not diversifiable risk — it is uncertainty about which kind of season you will get, and shrinking it lowers the expectation without protecting anything you can bank.

**Honest caveat: none of the mean differences are significant.** With 5 seasons the between-season SE is ~3.5pp, so 11.47 vs 9.67 is well inside noise. The *variance* ordering is the more credible signal, and it is the one that does not matter for a one-shot draft. The defensible reading is **"no evidence that shrinking the QB tilt helps; mild evidence it costs"** — so leave the value function alone.

This supersedes §7c's closing recommendation, which was reasoning rather than measurement. The reasoning was plausible and the measurement disagrees; the measurement wins.

`scripts/qb_tilt_sweep.py` is committed and its mechanism is verified (raising QB replacement does reduce QB count: 3.00 → 1.75 at delta=2.15 on 2024).

*(Stale paragraph removed 2026-08-09: it said the five-season results were void pending a native-ARM re-run, directly contradicting the table above, which **is** that re-run. The table stands; only the argument drawn from it is confounded — see the note under it.)*

## 11. Stage 4 — the live assistant, and why precompute was dropped

**Built, measured, simplified.** The owner chose "precompute + live fallback" on the reasonable assumption that precompute would cover the common draft states. **It does not, and the architecture was cut back to live-only after measuring it.**

### The measurement that killed precompute

`scripts/build_recommendation_cache.py` (kept at `docs/superpowers/archive/build_recommendation_cache.py.unused`) samples reachable states from the opponent model. Sampling 300 drafts:

| our round | distinct states seen | top 8 states cover |
|---|---|---|
| 1 | 203 | **17.0%** of drafts |
| 2 | 248 | **11.3%** |
| 3 | 269 | **9.0%** |

**Roughly 195 of 203 round-1 states occur exactly once.** The state space is nearly as large as the sample, and this is not a tuning problem: at pick 8, seven players are gone from a realistic top ~15, which is C(15,7) ≈ 6,400 states before later rounds multiply it. Coarsening the key does not rescue it either — *which* elite players remain is precisely what changes the recommendation, so a key coarse enough to collapse the space is too coarse to be correct.

Cost/benefit: ~78 minutes of precompute bought about **one pick in eight**. A live recommendation costs 13–36s and covers all of them.

### Live-only is comfortable on the clock — measured, not asserted

`LIVE_BUDGET` = **12×16×300**, measured on the native arm64 venv against the live 2026 pool:

| our pick | rounds left | elapsed | leader SE |
|---|---|---|---|
| 8 | 18 | **35.7s** | 0.92pp |
| 13 | 17 | 34.1s | 0.79pp |
| 33 | 15 | 30.4s | 0.50pp |
| 88 | 10 | 21.5s | 0.48pp |
| 148 | 4 | 13.1s | 0.46pp |

Cost scales with **rounds left to simulate**, so pick 8 is the worst case and it falls steadily. Against a ~90s clock that leaves ~54s to think at the worst pick.

Two things make this better than the raw number. **Pick entry does not compete with your clock** — opponents' picks are entered during *their* turns, so at most one is outstanding when yours starts. And **precision improves as picks get cheaper** (SE 0.92pp → 0.46pp), because fewer remaining rounds means less variance to simulate.

Why not the 15×20×400 budget that also fits under 60s: the 90s clock must also cover entry, reading, and deciding. 35.6s leaves ~54s of human time against ~30s. The precision surrendered is 0.25pp of SE against the model's own 4.15pp between-season SE (§10) — a bad trade.

### The rerun hazard, and the memo

Streamlit re-executes the entire script on every interaction. The first version called `recommend()` directly in the render path, so clicking Undo or touching the selectbox after a recommendation appeared would **re-run the simulation on the clock**. `app._memoized_recommendation` keys results by `state_key` in `st.session_state`: unchanged board → instant, genuinely new board → recompute. Pinned by `tests/test_live_app_paths.py::test_recommendation_is_memoized_by_state_key_not_recomputed`, including that an *undo* returns to a memo hit.

### What survives from the precompute work

- `live/cache.py` keeps `state_key`, `recommend` (still the single `recommend_pick` call site), and `candidates_to_rows`. `RecommendationCache` remains but is unused by the app.
- **`TOP_N_FOR_KEY = 60` is validated** (`scripts/validate_cache_key.py`): boards sharing a key produce the same leader 3/3, probability deltas 0.82/0.49/0.08pp consistent with Monte-Carlo noise at the full budget's 0.45pp SE. Still load-bearing, because the memo keys on it.

### Draft-day runbook

1. Re-run the ingest so `rankings_2026` / `adp_2026` are current.
2. `.venv/bin/streamlit run src/ffdraft/live/app.py`
3. Load the page **before** the draft starts — fitting the context takes a few seconds and is cached per session.
4. Enter every pick as it happens, including opponents'. Undo fixes a mis-entry.
5. At our pick the recommendation computes automatically. Read the tie callout before the ranking: candidates flagged indistinguishable from the leader are equivalent, and the ordering among them is noise.

## 12. Draft-day ingest verification — 2026-08-08

`.venv/bin/python scripts/ingest_all.py`, exit 0. Recorded here because **the data itself is gitignored and must stay that way** (ESPN payloads identify real people, §8 / plans README), so this record is the only durable evidence of the refresh.

**Row counts before → after:**

| dataset | before | after | delta |
|---|---|---|---|
| `rankings_2026` | 517×6 | 515×6 | −2 |
| `adp_2026` | 246×7 | 256×7 | +10 |
| `weekly_stats` | 70775×29 | 70775×29 | 0 |
| `adp_history` | 1375×7 | 1375×7 | 0 |
| `league_drafts` | 1390×6 | 1390×6 | 0 |
| `id_crosswalk` | 11897×5 | 11897×5 | 0 |

Only the two live sources moved (FantasyPros −2 ranked players, FFC +10 in 2026 ADP); historical datasets unchanged. That is the expected shape of a same-day refresh — **a collapse in any of these is the signal to restore from backup, not to proceed.**

**Validation:** no tracebacks; ID coverage 90.5–91.7% across gsis/espn/sleeper; all eight draft seasons complete.

**Live context rebuilt and exercised:** pool **508 players**, board advances correctly to overall pick 8, top of board Ja'Marr Chase / Puka Nacua / Jahmyr Gibbs / Bijan Robinson / Jaxon Smith-Njigba. This check matters more than the row counts: `build_player_pool` raises on unmatched players rather than dropping them, so a crosswalk regression shows up as a *smaller pool*, not an error — confirming 508 and a coherent top-5 is what actually rules it out.

**Privacy check on the ingest log**, run before reading it: zero GUID patterns, zero occurrences of the league ID, and the single `SWID` match was the line `league_managers: 18 distinct manager SWIDs` — a count, not values. Do this every time; the log is written by code that has ESPN payloads in scope.

**Before re-running the ingest**, back up `data/` first (`tar czf <somewhere-outside-the-repo>/data_backup.tgz data/`). These are live external sources: an upstream schema change or a partial response would overwrite a working pool with a broken one, and there is no other copy.

### Note on document dates

Stage 4's spec, plan and several §-headers are dated **2026-08-04**; the actual date was **2026-08-08**. A session-start hook reported the earlier date and it was taken at face value. Filenames are left alone to avoid breaking links — treat every "2026-08-04" in Stage 4 material as 2026-08-08.

### Refresh — 2026-08-22 (ADP + rankings only)

`fantasypros.ingest()` and `ffcalculator.ingest()`, not the full `ingest_all.py` — neither needs ESPN cookies or the nflverse pull.

| dataset | before | after |
|---|---|---|
| `rankings_2026` | 515 | 513 |
| `adp_2026` | 256 | 266 |
| `adp_history` | 1375 | 1375 (values unchanged, see below) |

The 2026 board had moved in 14 days: **mean 6.0 places**, swings of 20-30 (Caleb Williams −17.7, Pat Freiermuth +33.8, Wan'Dale Robinson +15.2). After name normalisation, **24 true adds and 14 true drops** (Aiyuk and Keenan Allen in; Pearsall, Mooney, Geno Smith out). Pool rebuilt to 509 players, all 509 ADP-matched, 4 excluded as unmatched.

**`adp_history` changed only in the `name` column — not one ADP value moved.** Finished seasons are fixed; the churn was spelling ("Brian Robinson Jr." → "Brian Robinson"). Worth knowing because **a raw-name diff makes those look like players dropping off the board when they have not** — normalise before comparing.

Refreshing invalidated four fingerprinted caches and the suite failed 42 tests in 15s. That is `CacheStaleError` working as designed. Refit explicitly (3s), 437 tests pass, tree rebuilt (343s). Commands in §9.

## 13. Session log — 2026-08-08/09

The session that took the project from "Stage 3 incomplete, engine apparently broken" to all four stages complete and pushed. Recorded because most of the value was **negative results and overturned conclusions**, which are the easiest things for a future reader to re-derive by accident.

### What was overturned

| Was believed | Now known |
|---|---|
| The engine loses to ADP by 13.5pp; do not trust it | ~6pp was a scoring artifact, and 2024 was the worst of 5 seasons. Multi-season: **+2.86pp ± 4.15pp, 3 of 5** |
| Hindsight lineup-setting costs ≤2.5pp | For a *depth-stacked* roster it is worth **5.75pp**, and it silently decided the backtest |
| The QB tilt might be an edge (6-pt passing TDs) | Directly measured: taking QB/TE in rounds 1-3 is a **leak** — 9 of 9 comparisons favour skill-only, one at 3.6σ |
| "Size the QB bet down" (reasoning, §7c) | Swept it: mean *falls* monotonically as the tilt shrinks. Superseded — see §10c |
| Precompute + live fallback is the right architecture | Built it, measured a **9-17% hit rate**, dropped it. Live-only |
| `rtcompat` fixed the Polars instability | It did not. Native ARM did |

### Decisions made, with their reasons

- **Survival is displayed, never blended into the ranking.** Championship probability is meant to *be* the objective; survival is a correction for a defect in how it is computed. A weighted blend would need an invented weight between quantities sharing no units, and would bury the defect behind a knob. `wait_cost_pp` expresses the combination in real units instead.
- **Manual pick entry, no ESPN polling** (owner). Cannot silently desync mid-draft.
- **Ranking is never silently re-sorted** on any unvalidated score. Caps and tie-breaks change the *choice set*, not the model's ordering.
- **An unsimulated pick carries `championship_probability = None`**, not 0.0. It crashed the mock's logging on first use — the correct failure mode. A 0.0 default would have displayed a fabricated title equity.
- **Aliases key on full names, never first-name tokens.** `kenny → kenneth` would rewrite Kenny Pickett and Kenny Golladay; merging two real players is the Marvin Harrison Jr./Sr. class of bug.

### Defects found by looking at output, not by reading code

Every one produced plausible, non-crashing, wrong results:

1. **Hindsight lineups** — worth 5.75pp to depth-stacked rosters, 0 to ours. Found by comparing roster *shapes*.
2. **Rosetta/Polars silent corruption** — identical code returned 5.33% once and ~17% every other time. Found because a control arm disagreed with a known value.
3. **`filter_capped` disabling itself** — its "never return nothing" fallback returned capped rows, and by the late rounds *everything* is capped. Found in a mock roster with 4 QBs.
4. **My own tie-break causing a R9 kicker** — roster "need" overrode a 100%-survival signal. Found in a mock roster.
5. **`validate_cache_key.py` reporting "OK — 3/3"** while its own output said the key had moved on every trial. It validated nothing and said the opposite.
6. **HTTP 200 from a dead Streamlit app** — the server booted, the script had crashed on a relative import. Renders client-side, so status proved nothing.
7. **u32 underflow** in an ad-hoc rank-gap analysis — `-1` printed as 4294967295, inverting the sort.
8. **Nickname mismatches** silently excluding draftable players (Gainwell rank 96, Okonkwo 144) from the pool *and* from being enterable.

### Owner observations that drove real fixes

Three defects were found by the owner reading a mock roster, not by any automated check:

- *"recommending players a couple rounds before they go"* → led to the survival column and, ultimately, to measuring the QB/TE tilt as a leak.
- *"why would we draft 2 high TEs?"* → the two-TE build; now measured as costing 4.85pp vs 2RB:1WR.
- *"Chase will almost certainly not be available at 8"* → open item 3b, the opponent-model temperature.

**The lesson worth keeping: the model cannot audit its own priors. Domain review of concrete output found things 421 tests did not.**

### Current open items, in priority order

**The tool runs; whether it is any good is now the open question.** Stages 1-4 are complete and the live app works. Item 1 is about whether the engine actually beats doing nothing clever, which is more fundamental than anything below it.

1. **THE ENGINE LOSES 2023 AND 2024 — see §7e.** Championship rate falls from ~21% (2020-22) to ~7% (2023-24) under *both* the old flat temperature and the new fitted schedule, so it is not an artifact of either. The pooled "+1.27pp over ADP" averages two opposite regimes and hides a sign flip in exactly the two seasons most like 2026 (2023: −8.67pp, 2024: −13.67pp). Partly narrowed 2026-08-12 (§7e, "round 1"): the engine can only differ from ADP by *positional allocation* (within a position their orderings are identical), the QB tilt has been ruled out as the cause, and the ADP baseline itself is flawed — it drafts no quarterback. Still unexplained. **Everything below is secondary**: there is little value in tuning a component while the whole loses the most recent seasons. Start by checking whether the collapse tracks the player pool, the fits that grow with each added season, or simply how "chalk" those two seasons were.
2. **The backtest leaks the temperature schedule.** `TEMPERATURE_BY_ROUND` is fit on 2018-2024 — including every holdout — while `backtest.fit_holdout_context` correctly refits only the reach model through the prior season. Fix by threading a per-holdout schedule through `run_rollout` and moving the likelihood replay out of `scripts/fit_temperature.py` into `src/`. **Does not affect draft-day use** (2026 is outside the fitting range), and it cannot have flattered the engine — both contenders share the simulated opponents, and ADP gained most. But no holdout number involving the opponent model is clean until this is done.
3. **Item 4** — state-dependent between-rollout variance.
4. **Item 6** — pin smaller budgets in the two slow real-data tests (suite is ~7:20, 426 tests).
5. **Item 7** — `sources/nflverse.py` → `models/kicking.py` layering inversion.
6. ~~**ADP's championship rate rises monotonically across holdouts.**~~ **EXPLAINED 2026-08-12, §7e — the trend was caused by the temperature schedule, not by the data.** ADP-following is flat under the old flat temperature (corr with season +0.345) and steeply rising under the fitted schedule (+0.967). Related but separate: ADP coverage is genuinely incomplete — ~200 real rows per season against a ~637-player pool, and only 157 in 2022 (K 5, D/ST 6). Verified upstream: the Fantasy Football Calculator API itself returns 157 for 2022, so this is the source's gap, not our ingest. Coverage does not trend, so it explains neither the rise nor §7e.
7. **Replacement-level uncertainty** — the experiment whose rationale was retracted (§10c). Run it with no predicted direction, or drop it.
8. **Three players remain unmatched** (Tommy Myers, Jordan Waters, Elijah Tau-Tolliver), all rank 441+. Genuinely absent from nflverse; no alias helps. If one is drafted, **they cannot be entered** and the board will drift. The owner declined the UI escape hatch that would fix this.

### Closed 2026-08-09

- ~~**Item 3 — the QB/TE tilt in `value.py`'s replacement levels.**~~ **Mechanism localised, two fixes measured, neither works.** See §7c "Resolution". The tilt is real (top-40: QB 6/TE 4 vs ADP's 1/1) and its cause is now known: the waiver baseline prices a slot as if it will never be drafted. But a coherent all-position correction loses 4.40pp (2.1σ) and a rounds-1-3 QB/TE ban is 0.1σ while swinging ±11pp by season. **`value.py` is unchanged and should stay that way absent new evidence.** Anyone reopening it needs a fix that does *not* amount to a fixed bet on that season's QB streaming quality — the part the data says is unknowable at draft time. Do not re-run the §10c delta sweep; it is confounded (see the note there).
- ~~**Item 3b — pick-dependent opponent-model temperature.**~~ **Fixed, see §7d.** `TEMPERATURE_BY_ROUND` replaces the flat 3.0; held-out LL +1.72/pick in 7/7 seasons, board-realism error halved. The gate is statistically unchanged (+1.27pp vs +2.86pp, SE ~5pp): this bought simulation fidelity, not edge. It created open items 1 and 2 above.

> **Every measurement taken before 2026-08-09 used the flat temperature**, including §7c's structure study and its resolution. Their *conclusions* stand — the opponent environment was shared by every contender — but their exact numbers would move on a re-run. Re-derive rather than quote them if a decision turns on the precise value.

## 14. The precomputed pick tree — 2026-08-11, rebuilt 2026-08-22

`scripts/build_pick_tree.py` → `data/pick_tree.json` → `scripts/build_pick_tree_page.py` → `pick_tree.html`.

A decision tree over our **first six picks** at slot 8 (overall 8, 13, 28, 33, 48, 53); a path through it is a complete plan for rounds 1-6. 189 nodes from 94 `recommend_pick` calls at the same `LIVE_BUDGET` the app uses, so the tree says what the app would say rather than something cheaper.

**Only our own choices branch.** Opponent behaviour between our picks is integrated out into `p_available`, measured over 200 separate opponent simulations. **That is the number that says whether a branch is real** — a node at 15% is a contingency to recognise, not a plan. The board each column is evaluated on is one representative opponent draft; `p_available` is the honest correction to that, and it is stated on the page rather than hidden.

Roster shape is enforced by *feasibility*, not a fixed pattern: a position is offered only when some legal final split (RB in 2..4, RB+WR=6) is still reachable, so 3/3, 4/2 and 2/4 all stay open. Since 3+3 = 6 that makes all six picks RB or WR — independently what §7c concluded for rounds 1-3.

**`recommend_pick` gained `restrict_to_positions` for this, and it was necessary rather than convenient.** At pick 28 the entire pruned candidate list came back QB and TE — the value-function tilt (§7c) biting exactly where documented — and filtering the *output* cannot recover a candidate that was never simulated. It restricts **pruning only**; rollouts, opponents and scoring are untouched, so each candidate's championship probability still accounts for every position we might take later. Inert when omitted, pinned by a test.

**Round 1 on the 2026-08-22 board:** Chase 30.7% (23% available), McCaffrey 27.9% (42%), Taylor 25.8% (51%). On the two-week-older board it was Chase / St. Brown / Lamb — **the tree is worth rebuilding whenever the data is refreshed**, since two weeks of movement replaced two of the three round-1 options.

### The page shipped blank once. Read this before trusting a generated artifact.

`PAGE` was a normal Python string, so every `\n` written inside the JavaScript became a literal newline, splitting `lines.join("\n")` into an unterminated string. The whole script failed to parse and the page rendered **completely blank** — no error, nothing to click. It is a raw string now.

**How it shipped matters more than the typo.** The page was "verified" by pasting an equivalent script into a live browser and watching it work — the preview pane strips inline scripts, so the real artifact was never executed even once. Verifying a retyped copy of a deliverable is not verifying the deliverable. `build_pick_tree_page.py` now runs `node --check` on its own output and refuses to write a page that does not parse (proven by reintroducing the bug: exit 1, no file written).

## 15. Performance — `recommend_pick` is 10.7x faster, bit-identical — 2026-08-11

34.2s → 3.2s at `LIVE_BUDGET`, verified against the *original* serial code at picks 8, 28 and 53: every candidate, championship probability and standard error identical. The test suite dropped 7:25 → 3:20 as a side effect.

Profiling, not guessing, found 61% of a call in one place: `value_available` scored the **entire ~380-player pool** at every one of our future picks inside every rollout — 1,239,614 `solve_lineup` calls — when `_choose_our_pick` wants only the argmax.

1. **`value.dominant_candidates` (exact, ~2.7x).** Within a position, value is monotone non-decreasing in projected mean — `lineup_marginal` because any slot open to a lower-mean player is open to a higher-mean one at a higher score; `bench_value` because `over_replacement` rises while the "ahead" set can only shrink; and `max()` preserves monotonicity. So the argmax over 380 players is the argmax over the best-by-mean player at each position — six of them. Zero of 216 real states picked a different player. `prune_candidates` gets its top-k the same way, with the zero-value tail reconstructed exactly (by rounds 17-18 fewer than `n_candidates` players have positive value, so that tail is a tie among zeros broken over the *whole* pool).
2. **Two pure refactors (~1.45x).** `pick_probabilities` called `predicted_reach` 12.4M times for six distinct values (it depends only on position); `run_rollout` rebuilt ~15M identical frozen `AvailablePlayer` objects. Also `run_rollout` re-ran `pool_adp_lookup` — a polars join over the whole pool — 192 times per call for an answer that cannot change.
3. **Parallel candidates (~3.7x).** Candidates are independent and share their random seeds, drawn in the parent, so this cannot move a number.

**Parallelism is opt-in and OFF by default, deliberately.** Every start method available on macOS rebuilds a worker's namespace by executing the *main module*, so a caller without an `if __name__ == "__main__"` guard re-runs its own script in every worker — measured: a guardless script ran itself three times and finished **slower than serial** (44.9s vs 13.6s) while still printing the right answer. **A Streamlit script never has that guard**, so the live app stays serial (12.6s, comfortable against a 60-90s clock); batch scripts opt in with `n_workers=0`.

Two dead ends recorded so they are not retried: `forkserver` does *not* avoid importing `__main__` (it only moves which process does it), and preloading this package into the forkserver crashed every worker on macOS (`+[NSCharacterSet initialize] may have been in progress in another thread when fork() was called`). Two safety rails: a worker never starts its own pool, and any pool failure falls back to serial with a `RuntimeWarning` — identical numbers, just slower. On draft day a broken pool must cost time, not the pick.

**Budgets were deliberately NOT widened.** There is ~10x headroom now, but the leader was already identical at every budget measured, so spending it is a decision to take and re-measure, not a side effect of an optimisation.

## 16. Session log — 2026-08-11/22

**Shipped:** the pick tree and explorer (§14); the 10.7x speedup (§15); the fitted per-round opponent temperature (§7d); the player selector re-ordered by **ADP** with board rank beside it (it is used to record what *other* managers did, and they draft off ADP — the two disagree by 20.7 places on average, and Cincinnati's D/ST is ADP 174 vs board #404).

**Investigated, not solved:** §7e. The engine's collapse in 2023-24 is real and robust across both opponent models. Ruled out: the temperature schedule, ADP coverage, and the QB tilt (re-pricing QB replacement by +2.15 cuts QBs drafted from 3.00 to 0.99 and makes those seasons *worse*).

**Corrections I made to my own earlier claims** — the pattern worth carrying forward is that each was found by looking at data, not by re-reading code:

| claimed | actually |
|---|---|
| "ADP's rising championship rate predates the temperature change" | It was *caused* by it. ADP is flat under the old flat temperature (corr +0.345) and steeply rising under the fitted schedule (+0.967). |
| The `baseline` replacement fix is "a no-op" | It loses 4.40pp (2.1σ) — called off a top-40 board and a 25-seed smoke test, both too weak to see it. |
| The guard's benefit tracks how often the engine takes QB/TE early | Falsified by 2022: the *most* early QB/TE and the guard's *worst* season. Two points looked like a relationship. |
| The pick-tree page works (with screenshots) | It was blank. I verified a retyped copy, never the generated file. |

**Two structural facts found while digging, both worth more than the bug that surfaced them:**

- **In the backtest, the model's within-position ordering IS ADP's ordering** — identical, by construction. So the backtest can only ever validate *positional allocation*, never projection quality. Live 2026 uses real ECR, so this is a backtest property, not a draft-day one.
- **The ADP baseline never drafts a quarterback** (0.5 QB, 0.1 TE in 2024) — it takes the lowest ADP available with no roster constraint and starts a free replacement-level QB. `engine - adp` is the load-bearing number and its denominator is a strategy no human would run.

