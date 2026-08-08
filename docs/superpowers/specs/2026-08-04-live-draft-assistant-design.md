# Stage 4 — Live Draft Assistant: Design

**Date:** 2026-08-04
**Status:** approved by owner (latency strategy and pick-entry mechanism confirmed)
**Prerequisite reading:** [`docs/HANDOFF.md`](../../HANDOFF.md) §3 (status), §7c (structure answer), §9–§10 (runtime), and [`2026-08-02-ff-draft-2026-design.md`](2026-08-02-ff-draft-2026-design.md) for the league configuration.

---

## 1. Goal

A draft-day tool that, at each of our 18 picks, shows a ranked list of candidates with simulated championship probability and honest uncertainty — fast enough to use inside a ~90-second pick clock.

**Stage 4 adds no modelling.** Every number comes from `draft.recommender.recommend_pick`, unchanged. This stage is delivery: state entry, latency engineering, and presentation.

## 2. What is already true, and constrains this design

Measured, not assumed. Do not re-derive these.

- **A full recommendation costs ~3.7 minutes** (15 candidates × 65 rollouts × 500 season sims), down from ~5.6 min after the native-ARM migration (§9). The pick clock is ~90 seconds. This gap is the central engineering problem of Stage 4.
- **The engine's measured edge is small and not statistically demonstrated:** +2.86pp against a between-season SE of 4.15pp, beating ADP in 3 of 5 holdout seasons (§10). The tool must present itself as a decision aid, not an oracle.
- **Draft structure does not matter** (§7c): 0RB, Hero RB and 2RB:1WR are statistically indistinguishable. The tool must not imply a structural plan.
- **Draft slot is 8** in a 10-team snake, so our picks fall at overall **8, 13, 28, 33, 48, 53, 68, 73, 88, 93, 108, 113, 128, 133, 148, 153, 168, 173** — back-to-back pairs at the turn, which are best considered together.
- **Silent corruption is this project's dominant failure mode** (§8, §10). Prefer loud failures everywhere.

## 3. Owner decisions

**Latency: precompute + live fallback.** Cache full-budget recommendations for the draft states most likely to occur; on a cache miss, run a reduced-budget recommendation live. Chosen for robustness — there is always an answer — accepting that it is the most build effort and carries a two-code-path consistency risk.

**Pick entry: manual type-ahead only.** No ESPN polling. The 2026 draft endpoint shape is unverified, and a mid-draft auth expiry or schema change would cost picks at the worst possible moment. Manual entry cannot silently desync.

## 4. Architecture

```
src/ffdraft/live/
  state.py     draft board: pick entry, snake order, whose turn, undo
  cache.py     precomputed recommendations: key, build, lookup, fingerprint
  budget.py    the two budgets, and the measured basis for the live one
  app.py       Streamlit UI
scripts/
  build_recommendation_cache.py   offline precompute (run near draft day)
```

Dependencies point downward: `app.py` → `cache.py`/`state.py` → `draft.recommender` → existing model layers. Nothing in `live/` is imported by anything below it.

### 4.1 The consistency guarantee

The two-path risk is that cache and fallback drift into two behaviours. The design forecloses it structurally:

```python
def recommend(state, ctx, budget: Budget) -> RecommendationResult:
    return recommend_pick(state, ctx.pool, ctx.model, ctx.replacement_by_position,
                          seed=budget.seed, n_candidates=budget.n_candidates,
                          n_rollouts=budget.n_rollouts,
                          n_sims_per_rollout=budget.n_sims_per_rollout,
                          adp_table=ctx.adp_table, rankings=ctx.rankings)
```

There is **one** call site. Precompute passes `FULL_BUDGET`; the fallback passes `LIVE_BUDGET`. They differ only in the budget argument. A test asserts both paths return identical output for the same state at the same budget.

### 4.2 Cache key — a deliberate, documented approximation

Keying on the exact set of drafted players is useless past round 2: the state space branches faster than any precompute can cover. The key is therefore lossy:

```
(our_overall_pick, our_roster_position_counts, frozenset(available ∩ top_N_by_rank))
```

with `N = 60`. **Rationale:** a recommendation depends on which *good* players are available and what our roster needs. Whether the 200th-ranked player is gone cannot change our pick, because we would never take him. Collapsing that detail is what makes precompute feasible.

This is an approximation and is labelled as one. Two guards:

1. `N = 60` is validated empirically (Task 6): sample state pairs that share a key but differ below the top 60, run both at full budget, and confirm the top recommendation agrees. If it does not, raise `N` or abandon precompute for that round.
2. The UI always shows whether an answer came from cache or live, so a suspicious recommendation can be re-run live at any time.

### 4.3 Which states to precompute

Sample them from the opponent model rather than guessing. `build_recommendation_cache.py` runs `run_rollout` forward many times from an empty board, records the state key at each of our 18 picks, and precomputes the most frequently reached keys within a configured budget.

This yields a **measured hit rate per round**, which is the number that tells us whether precompute is working. Expect good coverage in rounds 1–3 and thin coverage later; that is exactly what the fallback is for. The hit rate is reported, not assumed.

### 4.4 Staleness

Cache entries are fingerprinted with `store.fingerprint` over the artifacts that determine a recommendation (player pool rankings, ADP, opponent model training set). On load, a mismatch raises `CacheStaleError` — never serves the stale entry. This machinery already exists and exists precisely because silent staleness is this project's characteristic failure.

## 5. The UI

Single Streamlit page.

**Board panel.** Type-ahead over the 2026 pool, filtered to undrafted players. The app knows whose turn it is from the snake order, so entry is one field: the player. Submitting advances the board. Undo reverts the last pick — a mis-entry mid-draft must be cheap to fix.

**Recommendation panel**, shown only when it is our turn. Ranked candidates with: name, position, championship probability, standard error, gap from leader in pp, and whether it is `indistinguishable_from_leader`. Candidates statistically tied with the leader are visually grouped, because presenting a strict ranking over a tie is the single most misleading thing this tool could do.

**Provenance line**, always visible: whether the answer came from **cache (full budget)** or **live (reduced budget)**, the actual `n_rollouts × n_sims_per_rollout`, and elapsed seconds. A number whose origin cannot be seen cannot be calibrated.

**Standing caveat**, always visible: measured edge +2.86pp with a between-season SE of 4.15pp, 3 of 5 seasons. And, per §7c, no structural plan — best available, not a script.

## 6. Out of scope

- ESPN live polling (owner decision).
- Any change to the model, value function, or recommender.
- Trades, keepers, auction, or any league setting this league does not use.
- Multi-user or hosted deployment. This runs locally, for one person, on one draft day.

## 7. Validation

- `.venv/bin/pytest` passes on the native arm64 venv.
- A recommendation is produced for every one of our 18 picks within the pick clock, by cache or fallback, with no path that can produce no answer.
- `LIVE_BUDGET` is **measured** to fit ~60s on this machine, not chosen by intuition, and the measurement is recorded.
- Cache hit rate per round is reported from simulation.
- Cache and live paths produce identical output at equal budget (test).
- A stale cache raises `CacheStaleError` rather than serving a wrong answer (test).

## 8. Sequencing note

Precompute is only meaningful against the **real 2026 player pool**. The ingest must be re-run close to draft day and the cache rebuilt afterwards. This is an explicit task, not an assumption, and the fingerprint check enforces it: a cache built against a stale pool will refuse to load.
