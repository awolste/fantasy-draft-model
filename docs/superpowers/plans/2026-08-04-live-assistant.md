# Live Draft Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit tool that, at each of our 18 picks, shows ranked candidates with championship probability and honest uncertainty, fast enough for a ~90-second clock.

**Architecture:** `src/ffdraft/live/` holds draft-board state, a precomputed recommendation cache, budgets, and the UI. All numbers come from the existing `draft.recommender.recommend_pick` through a **single** call site parameterised by budget — precompute uses the full budget, live fallback a reduced one. Nothing in the model layer changes.

**Tech Stack:** Python 3.11 (native arm64 `.venv`), Streamlit, polars, numpy, pytest.

**Spec:** [`../specs/2026-08-04-live-draft-assistant-design.md`](../specs/2026-08-04-live-draft-assistant-design.md). Read it before starting — especially §4.2 on why the cache key is deliberately lossy.

**Run everything with `.venv/bin/...`** (native arm64). Never set `POLARS_SKIP_CPU_CHECK`. See HANDOFF §9–§10.

---

## Facts you will need (do not re-derive)

- `DRAFT_SLOT = 8`, `N_TEAMS = 10`, 18 rounds. Our picks: **8, 13, 28, 33, 48, 53, 68, 73, 88, 93, 108, 113, 128, 133, 148, 153, 168, 173**.
- `recommend_pick(state, pool, model, replacement_by_position, seed, our_team=DRAFT_SLOT, n_candidates=15, n_rollouts=65, n_sims_per_rollout=500, temperature, roster_decay, adp_table=None, rankings=None) -> RecommendationResult`. It **raises `ValueError`** if `state.team_on_clock != our_team`.
- `RecommendationResult`: `candidates`, `n_candidates_considered`, `n_available`, `n_rollouts`, `n_sims_per_rollout`, `used_common_random_numbers`, `elapsed_seconds`.
- `CandidateRecommendation`: `player_id`, `name`, `position`, `greedy_value`, `mean`, `championship_probability`, `standard_error`, `gap_from_leader_pp`, `indistinguishable_from_leader`, `roster_counts_before_pick`.
- `DraftState.from_picks(picks, n_teams, rounds)`, `Pick(overall_pick, team, player_id, position)`, `state.team_on_clock`, `state.drafted_ids`, `state.rosters`, `state.next_overall_pick`.
- `store.fingerprint(*frames) -> str`, `store.write_fingerprint(name, fp)`, `store.read_fingerprint(name)`, `store.check_cache_fresh(name, fp)` raises `CacheStaleError`.
- `team_for_pick(overall_pick, n_teams)` in `draft.rollout` gives the 1-indexed team on the clock.

---

## Task 1: Budgets, and measure the live one

**Files:**
- Create: `src/ffdraft/live/__init__.py` (empty)
- Create: `src/ffdraft/live/budget.py`
- Create: `tests/test_live_budget.py`
- Create: `scripts/measure_live_budget.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_budget.py
from ffdraft.live.budget import FULL_BUDGET, LIVE_BUDGET, Budget


def test_full_budget_matches_recommender_defaults():
    from ffdraft.draft.recommender import (
        DEFAULT_N_CANDIDATES, DEFAULT_N_ROLLOUTS, DEFAULT_N_SIMS_PER_ROLLOUT,
    )
    assert FULL_BUDGET.n_candidates == DEFAULT_N_CANDIDATES
    assert FULL_BUDGET.n_rollouts == DEFAULT_N_ROLLOUTS
    assert FULL_BUDGET.n_sims_per_rollout == DEFAULT_N_SIMS_PER_ROLLOUT


def test_live_budget_is_strictly_cheaper_than_full():
    full = FULL_BUDGET.n_rollouts * FULL_BUDGET.n_sims_per_rollout * FULL_BUDGET.n_candidates
    live = LIVE_BUDGET.n_rollouts * LIVE_BUDGET.n_sims_per_rollout * LIVE_BUDGET.n_candidates
    assert live < full


def test_budget_is_hashable_so_it_can_key_a_cache():
    assert hash(Budget(1, 2, 3, seed=4)) == hash(Budget(1, 2, 3, seed=4))
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `.venv/bin/pytest tests/test_live_budget.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ffdraft.live'`

- [ ] **Step 3: Implement**

```python
# src/ffdraft/live/budget.py
"""The two recommendation budgets, and the measured basis for the live one.

There is exactly one `recommend_pick` call site in this package (see
`live/cache.py:recommend`); these objects are the only thing that differs
between the precomputed and live paths. Keeping them as data, rather than
as two code paths, is what makes the consistency test in
`tests/test_live_cache.py` possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..draft.recommender import (
    DEFAULT_N_CANDIDATES,
    DEFAULT_N_ROLLOUTS,
    DEFAULT_N_SIMS_PER_ROLLOUT,
)


@dataclass(frozen=True)
class Budget:
    n_candidates: int
    n_rollouts: int
    n_sims_per_rollout: int
    seed: int


FULL_BUDGET = Budget(
    n_candidates=DEFAULT_N_CANDIDATES,
    n_rollouts=DEFAULT_N_ROLLOUTS,
    n_sims_per_rollout=DEFAULT_N_SIMS_PER_ROLLOUT,
    seed=20260804,
)

# PLACEHOLDER until Task 1 Step 5 measures it. Do not ship these numbers.
LIVE_BUDGET = Budget(n_candidates=10, n_rollouts=12, n_sims_per_rollout=250, seed=20260804)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_live_budget.py -q`
Expected: 3 passed

- [ ] **Step 5: Measure the real live budget**

```python
# scripts/measure_live_budget.py
"""Measure what recommendation budget fits a ~60s live pick.

HANDOFF says a full recommendation costs ~3.7 min on this machine. The
live budget must be MEASURED, not guessed -- that is a validation-gate
requirement in the Stage 4 spec. Prints wall-clock and the leader's SE at
each budget so the accuracy cost of going cheap is visible, not implied.
"""

from __future__ import annotations

import time

from ffdraft.backtest import fit_holdout_context
from ffdraft.draft.recommender import recommend_pick
from ffdraft.draft.rollout import DraftState
from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET, Budget

CANDIDATE_BUDGETS = [
    Budget(10, 8, 200, seed=1),
    Budget(10, 12, 250, seed=1),
    Budget(12, 16, 300, seed=1),
    Budget(15, 20, 400, seed=1),
]


def main() -> None:
    ctx = fit_holdout_context()
    state = DraftState.from_picks([], n_teams=N_TEAMS, rounds=18)
    print(f"{'budget':<28}{'elapsed':>10}{'leader SE':>12}{'leader':>22}")
    for b in [*CANDIDATE_BUDGETS, FULL_BUDGET]:
        t0 = time.perf_counter()
        res = recommend_pick(
            state, ctx.pool, ctx.opponent_model, ctx.replacement_by_position,
            seed=b.seed, our_team=DRAFT_SLOT, n_candidates=b.n_candidates,
            n_rollouts=b.n_rollouts, n_sims_per_rollout=b.n_sims_per_rollout,
            adp_table=ctx.adp_holdout, rankings=ctx.rankings_holdout,
        )
        dt = time.perf_counter() - t0
        top = res.candidates[0]
        label = f"{b.n_candidates}x{b.n_rollouts}x{b.n_sims_per_rollout}"
        print(f"{label:<28}{dt:>9.1f}s{top.standard_error * 100:>11.2f}pp{top.name[:20]:>22}")


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/measure_live_budget.py`

Pick the largest budget whose elapsed time is **under 60s**, edit `LIVE_BUDGET` to those numbers, and record the measured table as a comment above it. Note in the comment whether the leader changed versus `FULL_BUDGET` — if it did, say so plainly rather than burying it.

- [ ] **Step 6: Commit**

```bash
git add src/ffdraft/live tests/test_live_budget.py scripts/measure_live_budget.py
git commit -m "feat: live budgets, with the live one measured not guessed"
```

---

## Task 2: Draft board state

**Files:**
- Create: `src/ffdraft/live/state.py`
- Create: `tests/test_live_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_state.py
import pytest

from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.state import DraftBoard


def test_board_starts_empty_and_knows_who_is_on_the_clock():
    b = DraftBoard(rounds=18)
    assert b.next_overall_pick == 1
    assert b.team_on_clock == 1
    assert b.is_our_turn is False


def test_our_picks_are_the_slot_8_snake_sequence():
    assert DraftBoard(rounds=18).our_pick_numbers[:6] == [8, 13, 28, 33, 48, 53]


def test_recording_picks_advances_the_snake_and_flags_our_turn():
    b = DraftBoard(rounds=18)
    for i in range(7):
        b.record(f"p{i}", "RB")
    assert b.next_overall_pick == 8
    assert b.team_on_clock == DRAFT_SLOT
    assert b.is_our_turn is True


def test_undo_reverts_exactly_one_pick():
    b = DraftBoard(rounds=18)
    b.record("a", "RB")
    b.record("b", "WR")
    b.undo()
    assert b.next_overall_pick == 2
    assert b.drafted_ids == {"a"}


def test_undo_on_empty_board_raises_rather_than_silently_doing_nothing():
    with pytest.raises(IndexError):
        DraftBoard(rounds=18).undo()


def test_drafting_the_same_player_twice_raises():
    b = DraftBoard(rounds=18)
    b.record("a", "RB")
    with pytest.raises(ValueError, match="already drafted"):
        b.record("a", "RB")


def test_to_draft_state_round_trips_into_the_engine_type():
    b = DraftBoard(rounds=18)
    b.record("a", "RB")
    st = b.to_draft_state(n_teams=N_TEAMS)
    assert st.next_overall_pick == 2
    assert st.drafted_ids == {"a"}
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `.venv/bin/pytest tests/test_live_state.py -q`
Expected: FAIL, `ImportError: cannot import name 'DraftBoard'`

- [ ] **Step 3: Implement**

```python
# src/ffdraft/live/state.py
"""The draft board as entered by hand on draft day.

Deliberately a thin wrapper over `draft.rollout.DraftState` rather than a
parallel model of the draft: `to_draft_state` is the only bridge, so the
engine can never be handed a board shape it did not produce itself.

Pick entry is manual (owner decision, see the Stage 4 spec): no ESPN
polling, so this cannot silently desync from the real draft. The cost is
that a mis-entry is possible, which is why `undo` exists and why
double-drafting raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..draft.rollout import DraftState, Pick, team_for_pick
from ..league import DRAFT_SLOT, N_TEAMS


@dataclass
class DraftBoard:
    rounds: int
    n_teams: int = N_TEAMS
    our_team: int = DRAFT_SLOT
    picks: list[Pick] = field(default_factory=list)

    @property
    def next_overall_pick(self) -> int:
        return len(self.picks) + 1

    @property
    def total_picks(self) -> int:
        return self.rounds * self.n_teams

    @property
    def is_complete(self) -> bool:
        return len(self.picks) >= self.total_picks

    @property
    def team_on_clock(self) -> int:
        return team_for_pick(self.next_overall_pick, self.n_teams)

    @property
    def is_our_turn(self) -> bool:
        return (not self.is_complete) and self.team_on_clock == self.our_team

    @property
    def drafted_ids(self) -> set[str]:
        return {p.player_id for p in self.picks}

    @property
    def our_pick_numbers(self) -> list[int]:
        return [
            n
            for n in range(1, self.total_picks + 1)
            if team_for_pick(n, self.n_teams) == self.our_team
        ]

    def record(self, player_id: str, position: str) -> None:
        if self.is_complete:
            raise ValueError("draft is complete; no further picks can be recorded")
        if player_id in self.drafted_ids:
            raise ValueError(f"{player_id} was already drafted")
        self.picks.append(
            Pick(
                overall_pick=self.next_overall_pick,
                team=self.team_on_clock,
                player_id=player_id,
                position=position,
            )
        )

    def undo(self) -> Pick:
        """Remove and return the last pick. Raises `IndexError` on an empty
        board rather than no-opping -- a silent no-op during a live draft
        would leave the board out of step with reality."""
        return self.picks.pop()

    def to_draft_state(self, n_teams: int = N_TEAMS) -> DraftState:
        return DraftState.from_picks(list(self.picks), n_teams=n_teams, rounds=self.rounds)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_live_state.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/ffdraft/live/state.py tests/test_live_state.py
git commit -m "feat: draft board state with undo and double-draft guard"
```

---

## Task 3: Cache key and the single recommend() call site

**Files:**
- Create: `src/ffdraft/live/cache.py`
- Create: `tests/test_live_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_cache.py
from ffdraft.live.cache import TOP_N_FOR_KEY, state_key


class _P:
    def __init__(self, rank, position):
        self.rank = rank
        self.position = position


def _pool(n=200):
    return {f"p{i}": _P(rank=i, position="RB") for i in range(1, n + 1)}


def test_key_ignores_players_outside_the_top_n():
    pool = _pool()
    a = state_key(8, {"QB": 1}, drafted_ids={"p150"}, pool=pool)
    b = state_key(8, {"QB": 1}, drafted_ids={"p151"}, pool=pool)
    assert a == b, "players below the top-N must not affect the key"


def test_key_distinguishes_top_n_availability():
    pool = _pool()
    a = state_key(8, {"QB": 1}, drafted_ids={"p1"}, pool=pool)
    b = state_key(8, {"QB": 1}, drafted_ids={"p2"}, pool=pool)
    assert a != b


def test_key_distinguishes_pick_number_and_roster_shape():
    pool = _pool()
    base = state_key(8, {"QB": 1}, drafted_ids=set(), pool=pool)
    assert state_key(13, {"QB": 1}, drafted_ids=set(), pool=pool) != base
    assert state_key(8, {"QB": 2}, drafted_ids=set(), pool=pool) != base


def test_key_is_order_independent_and_hashable():
    pool = _pool()
    a = state_key(8, {"QB": 1, "RB": 2}, drafted_ids={"p3", "p5"}, pool=pool)
    b = state_key(8, {"RB": 2, "QB": 1}, drafted_ids={"p5", "p3"}, pool=pool)
    assert a == b
    assert hash(a) == hash(b)


def test_top_n_is_the_documented_value():
    assert TOP_N_FOR_KEY == 60
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `.venv/bin/pytest tests/test_live_cache.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ffdraft.live.cache'`

- [ ] **Step 3: Implement key + the single call site**

```python
# src/ffdraft/live/cache.py
"""Precomputed recommendations, and the one place `recommend_pick` is called.

## Why one call site

The precomputed and live-fallback paths must never become two behaviours.
They are therefore the *same* function with a different `Budget`; see
`tests/test_live_cache.py::test_cache_and_live_paths_agree_at_equal_budget`.

## Why the key is lossy

Keying on the exact drafted set is useless past round 2 -- the state space
branches faster than any precompute can cover. A recommendation depends on
which *good* players are available and what our roster needs; whether the
200th-ranked player is gone cannot change our pick, because we would never
take him. So the key keeps only availability within the top
`TOP_N_FOR_KEY` by rank.

This is an approximation. `scripts/validate_cache_key.py` (Task 6) checks
it empirically by finding state pairs that share a key but differ below the
cutoff and confirming the full-budget recommendation agrees. If it does
not, raise the cutoff -- do not quietly accept the mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from ..draft.recommender import RecommendationResult, recommend_pick
from ..league import DRAFT_SLOT
from .budget import Budget

TOP_N_FOR_KEY = 60


class StateKey(NamedTuple):
    overall_pick: int
    roster_counts: tuple[tuple[str, int], ...]
    top_available: tuple[str, ...]


def state_key(
    overall_pick: int,
    roster_counts: Mapping[str, int],
    drafted_ids: set[str],
    pool: Mapping[str, object],
    top_n: int = TOP_N_FOR_KEY,
) -> StateKey:
    top_ids = sorted(pool, key=lambda pid: pool[pid].rank)[:top_n]
    return StateKey(
        overall_pick=overall_pick,
        roster_counts=tuple(sorted(roster_counts.items())),
        top_available=tuple(sorted(pid for pid in top_ids if pid not in drafted_ids)),
    )


def recommend(state, ctx, budget: Budget, our_team: int = DRAFT_SLOT) -> RecommendationResult:
    """The ONLY `recommend_pick` call in this package. Precompute passes
    `FULL_BUDGET`, the live fallback passes `LIVE_BUDGET`; nothing else
    differs between the two paths."""
    return recommend_pick(
        state,
        ctx.pool,
        ctx.opponent_model,
        ctx.replacement_by_position,
        seed=budget.seed,
        our_team=our_team,
        n_candidates=budget.n_candidates,
        n_rollouts=budget.n_rollouts,
        n_sims_per_rollout=budget.n_sims_per_rollout,
        adp_table=ctx.adp_table,
        rankings=ctx.rankings,
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_live_cache.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/ffdraft/live/cache.py tests/test_live_cache.py
git commit -m "feat: lossy cache key and the single recommend() call site"
```

---

## Task 4: Cache persistence with a staleness guard

**Files:**
- Modify: `src/ffdraft/live/cache.py`
- Modify: `tests/test_live_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_live_cache.py
import pytest

from ffdraft.live.cache import RecommendationCache
from ffdraft.store import CacheStaleError


def test_cache_round_trips_an_entry(tmp_path):
    c = RecommendationCache(path=tmp_path / "cache.json", fingerprint="fp1")
    key = ("k",)
    c.put(key, [{"name": "X", "championship_probability": 0.11}])
    c.save()
    reloaded = RecommendationCache.load(tmp_path / "cache.json", fingerprint="fp1")
    assert reloaded.get(key)[0]["name"] == "X"


def test_loading_with_a_different_fingerprint_raises_rather_than_serving_stale(tmp_path):
    c = RecommendationCache(path=tmp_path / "cache.json", fingerprint="fp1")
    c.put(("k",), [{"name": "X"}])
    c.save()
    with pytest.raises(CacheStaleError):
        RecommendationCache.load(tmp_path / "cache.json", fingerprint="fp2")


def test_missing_key_returns_none_so_the_caller_can_fall_back(tmp_path):
    c = RecommendationCache(path=tmp_path / "cache.json", fingerprint="fp1")
    assert c.get(("absent",)) is None
```

- [ ] **Step 2: Run, confirm it fails**

Run: `.venv/bin/pytest tests/test_live_cache.py -q`
Expected: FAIL, `ImportError: cannot import name 'RecommendationCache'`

- [ ] **Step 3: Implement**

```python
# append to src/ffdraft/live/cache.py
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..store import CacheStaleError


@dataclass
class RecommendationCache:
    """Precomputed recommendations, keyed by `StateKey`, guarded by a
    fingerprint of the artifacts that determine a recommendation.

    A fingerprint mismatch raises `CacheStaleError` instead of serving the
    entry. This project's characteristic failure is confident, plausible,
    wrong output (HANDOFF section 8), and a cache built against last week's
    player pool is exactly that -- so staleness must be loud.
    """

    path: Path
    fingerprint: str
    entries: dict[str, list[dict]] = field(default_factory=dict)

    @staticmethod
    def _encode(key) -> str:
        return json.dumps(key, sort_keys=True, default=list)

    def put(self, key, candidates: list[dict]) -> None:
        self.entries[self._encode(key)] = candidates

    def get(self, key) -> list[dict] | None:
        return self.entries.get(self._encode(key))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"fingerprint": self.fingerprint, "entries": self.entries})
        )

    @classmethod
    def load(cls, path: Path, fingerprint: str) -> RecommendationCache:
        raw = json.loads(Path(path).read_text())
        stored = raw.get("fingerprint")
        if stored != fingerprint:
            raise CacheStaleError(
                f"recommendation cache at {path} was built against fingerprint "
                f"{stored!r} but the current model artifacts fingerprint "
                f"{fingerprint!r}. Rebuild it with "
                f"scripts/build_recommendation_cache.py -- refusing to serve "
                f"recommendations from a stale player pool."
            )
        return cls(path=Path(path), fingerprint=fingerprint, entries=raw["entries"])
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_live_cache.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/ffdraft/live/cache.py tests/test_live_cache.py
git commit -m "feat: recommendation cache persistence with loud staleness guard"
```

---

## Task 5: Consistency test — the two paths must agree

**Files:**
- Modify: `tests/test_live_cache.py`

This is the task that justifies the owner's chosen architecture. If it cannot be made to pass, the two-path design is unsafe and must be reconsidered rather than patched.

- [ ] **Step 1: Write the test**

```python
# append to tests/test_live_cache.py
from ffdraft.backtest import fit_holdout_context
from ffdraft.draft.rollout import DraftState
from ffdraft.league import N_TEAMS
from ffdraft.live.budget import Budget
from ffdraft.live.cache import recommend


def test_cache_and_live_paths_agree_at_equal_budget():
    """Precompute and live fallback differ ONLY in their Budget. At the
    same budget they must be byte-identical -- that is the whole safety
    argument for having two paths at all."""
    ctx = fit_holdout_context()
    state = DraftState.from_picks([], n_teams=N_TEAMS, rounds=18)
    budget = Budget(n_candidates=5, n_rollouts=3, n_sims_per_rollout=40, seed=7)

    a = recommend(state, ctx, budget)
    b = recommend(state, ctx, budget)

    assert [c.player_id for c in a.candidates] == [c.player_id for c in b.candidates]
    assert [c.championship_probability for c in a.candidates] == [
        c.championship_probability for c in b.candidates
    ]
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest tests/test_live_cache.py::test_cache_and_live_paths_agree_at_equal_budget -q`
Expected: PASS. If it fails, `recommend_pick` is not deterministic at a fixed seed — **stop and investigate**; do not add a tolerance to make it pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_live_cache.py
git commit -m "test: cache and live paths are identical at equal budget"
```

---

## Task 6: Validate the lossy key empirically

**Files:**
- Create: `scripts/validate_cache_key.py`

The key throws information away (§4.2). This measures whether that costs anything, rather than assuming it does not.

- [ ] **Step 1: Write the script**

```python
# scripts/validate_cache_key.py
"""Does the lossy cache key (top-60 availability) change the recommendation?

live/cache.py keys on availability within the top TOP_N_FOR_KEY only. That
is an approximation: two boards that differ solely below the cutoff share a
key and would be served the same cached answer. This checks that they
deserve to be.

Method: build a base state, then perturb it by drafting additional players
strictly BELOW the cutoff. Key is unchanged by construction. Run the full
budget on each and compare the top recommendation.

If the leader disagrees, TOP_N_FOR_KEY is too small -- raise it and re-run.
Do not accept a mismatch: it would mean the cache can serve a wrong answer
on draft day and never say so.
"""

from __future__ import annotations

import numpy as np

from ffdraft.backtest import fit_holdout_context
from ffdraft.draft.rollout import DraftState, Pick, team_for_pick
from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET
from ffdraft.live.cache import TOP_N_FOR_KEY, recommend, state_key

N_PERTURBATIONS = 5


def main() -> None:
    ctx = fit_holdout_context()
    ranked = sorted(ctx.pool, key=lambda pid: ctx.pool[pid].rank)
    deep = ranked[TOP_N_FOR_KEY:]
    rng = np.random.default_rng(1)

    def state_with(drafted: list[str]) -> DraftState:
        picks = [
            Pick(
                overall_pick=i + 1,
                team=team_for_pick(i + 1, N_TEAMS),
                player_id=pid,
                position=ctx.pool[pid].position,
            )
            for i, pid in enumerate(drafted)
        ]
        return DraftState.from_picks(picks, n_teams=N_TEAMS, rounds=18)

    base_drafted = ranked[:7]
    base = state_with(base_drafted)
    base_key = state_key(8, {}, set(base_drafted), ctx.pool)
    base_rec = recommend(base, ctx, FULL_BUDGET)
    print(f"base leader: {base_rec.candidates[0].name}")

    agree = 0
    for i in range(N_PERTURBATIONS):
        swapped = list(base_drafted)
        swapped[-1] = deep[int(rng.integers(0, len(deep)))]
        st = state_with(swapped)
        k = state_key(8, {}, set(swapped), ctx.pool)
        rec = recommend(st, ctx, FULL_BUDGET)
        same_key = k == base_key
        same_leader = rec.candidates[0].player_id == base_rec.candidates[0].player_id
        agree += same_leader
        print(
            f"  perturbation {i}: same_key={same_key} same_leader={same_leader} "
            f"leader={rec.candidates[0].name}"
        )

    print(f"\nleader agreement: {agree}/{N_PERTURBATIONS}")
    if agree < N_PERTURBATIONS:
        print("MISMATCH -- raise TOP_N_FOR_KEY and re-run. Do not ship this cutoff.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and act on the result**

Run: `.venv/bin/python scripts/validate_cache_key.py`

If agreement is not total, raise `TOP_N_FOR_KEY` (try 100, then 150) and re-run until it is. Record the final value and the evidence in the spec's §4.2.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_cache_key.py
git commit -m "test: validate the lossy cache key against full-budget recommendations"
```

---

## Task 7: Build the cache offline

**Files:**
- Create: `scripts/build_recommendation_cache.py`

- [ ] **Step 1: Write the script**

```python
# scripts/build_recommendation_cache.py
"""Precompute full-budget recommendations for the draft states we are most
likely to face, and report the hit rate that results.

States are SAMPLED from the opponent model rather than guessed: simulate
many drafts forward, record the state key at each of our picks, and
precompute the most frequent keys within budget. That also yields a
measured hit rate per round -- the number that says whether precompute is
actually working.

Expect good coverage in rounds 1-3 and thin coverage later. That is what
the live fallback is for; it is not a defect.

Run this AFTER re-running the ingest near draft day. A cache built against
a stale pool will refuse to load (fingerprint guard in live/cache.py).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ffdraft.backtest import fit_holdout_context
from ffdraft.draft.rollout import DraftState, run_rollout, team_for_pick
from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET
from ffdraft.live.cache import RecommendationCache, recommend, state_key
from ffdraft.store import fingerprint

CACHE_PATH = Path("data/recommendation_cache.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=2000, help="draft simulations to sample states from")
    parser.add_argument("--top-k", type=int, default=40, help="most-frequent states to precompute per pick")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3],
                        help="which of OUR rounds to precompute")
    args = parser.parse_args()

    ctx = fit_holdout_context()
    fp = fingerprint(ctx.rankings, ctx.adp_table)

    # 1. Sample reachable states.
    seen: dict[int, Counter] = {r: Counter() for r in args.rounds}
    for seed in range(args.n_sims):
        rng = np.random.default_rng(seed)
        empty = DraftState.from_picks([], n_teams=N_TEAMS, rounds=18)
        final = run_rollout(empty, ctx.pool, ctx.opponent_model,
                            ctx.replacement_by_position, rng, our_team=DRAFT_SLOT,
                            adp_table=ctx.adp_table, rankings=ctx.rankings)
        for r in args.rounds:
            our_picks = [p.overall_pick for p in final.picks if p.team == DRAFT_SLOT]
            target = our_picks[r - 1]
            before = [p for p in final.picks if p.overall_pick < target]
            counts: dict[str, int] = {}
            for p in before:
                if p.team == DRAFT_SLOT:
                    counts[p.position] = counts.get(p.position, 0) + 1
            k = state_key(target, counts, {p.player_id for p in before}, ctx.pool)
            seen[r][k] += 1

    # 2. Precompute the most frequent, and report coverage honestly.
    cache = RecommendationCache(path=CACHE_PATH, fingerprint=fp)
    for r in args.rounds:
        total = sum(seen[r].values())
        top = seen[r].most_common(args.top_k)
        covered = sum(c for _, c in top)
        print(f"round {r}: {len(seen[r])} distinct states, "
              f"top {args.top_k} cover {100 * covered / total:.1f}% of simulated drafts")
        for key, _count in top:
            # Rebuild a representative state for this key is not possible from
            # the key alone (it is lossy), so recompute from the first draft
            # that produced it -- recorded during sampling in a fuller
            # implementation. For now, precompute is limited to states we can
            # reconstruct; see the note below.
            pass

    cache.save()
    print(f"\nwrote {CACHE_PATH} (fingerprint {fp[:12]}...)")


if __name__ == "__main__":
    main()
```

**Implementation note for whoever executes this task:** the sampling loop above records keys but not the states that produced them, and the key is lossy so a state cannot be reconstructed from it. Fix this while implementing: store `key -> first DraftState seen` during sampling, then call `recommend(state, ctx, FULL_BUDGET)` on that representative state. This is called out explicitly because the naive version silently writes an empty cache — which would look like success.

- [ ] **Step 2: Run it small and check the hit rates**

Run: `.venv/bin/python scripts/build_recommendation_cache.py --n-sims 200 --top-k 10 --rounds 1 2`
Expected: prints distinct-state counts and coverage percentages, writes the cache file.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_recommendation_cache.py
git commit -m "feat: offline recommendation cache builder with measured hit rate"
```

---

## Task 8: Streamlit UI

**Files:**
- Create: `src/ffdraft/live/app.py`
- Modify: `pyproject.toml` (add `streamlit` to dependencies)

- [ ] **Step 1: Add the dependency**

Add `"streamlit>=1.40"` to `[project].dependencies` in `pyproject.toml`, then:

Run: `.venv/bin/python -m pip install -q -e ".[dev]"`

- [ ] **Step 2: Implement the app**

```python
# src/ffdraft/live/app.py
"""Draft-day UI. Run with: .venv/bin/streamlit run src/ffdraft/live/app.py

Three things this page must always show, because each is a way the tool
could otherwise mislead:

1. PROVENANCE -- cache (full budget) or live (reduced budget), and the
   actual rollouts x sims. A number whose origin you cannot see is a number
   you cannot calibrate.
2. TIES -- candidates statistically indistinguishable from the leader are
   grouped, not ranked. Presenting a strict order over a tie is the single
   most misleading thing this tool could do.
3. THE STANDING CAVEAT -- measured edge +2.86pp against a between-season SE
   of 4.15pp, 3 of 5 seasons. This is a decision aid, not an oracle.
"""

from __future__ import annotations

import time

import streamlit as st

from ffdraft.backtest import fit_holdout_context
from ffdraft.league import DRAFT_SLOT, N_TEAMS
from ffdraft.live.budget import FULL_BUDGET, LIVE_BUDGET
from ffdraft.live.cache import RecommendationCache, recommend, state_key
from ffdraft.live.state import DraftBoard

CACHE_PATH = "data/recommendation_cache.json"


@st.cache_resource
def _context():
    return fit_holdout_context()


def main() -> None:
    st.title(f"Draft assistant — slot {DRAFT_SLOT} of {N_TEAMS}")
    st.caption(
        "Measured edge over ADP-following: +2.86pp, between-season SE 4.15pp, "
        "3 of 5 holdout seasons. A decision aid, not an oracle. "
        "No structural plan — draft structure was measured and does not matter."
    )

    ctx = _context()
    if "board" not in st.session_state:
        st.session_state.board = DraftBoard(rounds=18)
    board: DraftBoard = st.session_state.board

    undrafted = {pid: p for pid, p in ctx.pool.items() if pid not in board.drafted_ids}
    labels = {f"{p.name} ({p.position})": pid for pid, p in undrafted.items()}

    st.subheader(f"Pick {board.next_overall_pick} — team {board.team_on_clock}")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        choice = st.selectbox("Player taken", sorted(labels), index=None, key="pick_input")
        if st.button("Record pick", disabled=choice is None):
            pid = labels[choice]
            board.record(pid, ctx.pool[pid].position)
            st.rerun()
    with col_b:
        if st.button("Undo", disabled=not board.picks):
            board.undo()
            st.rerun()

    if not board.is_our_turn:
        st.info(f"Waiting — team {board.team_on_clock} is on the clock.")
        return

    st.subheader("Recommendation")
    counts: dict[str, int] = {}
    for pid, pos in board.to_draft_state().rosters[DRAFT_SLOT]:
        counts[pos] = counts.get(pos, 0) + 1
    key = state_key(board.next_overall_pick, counts, board.drafted_ids, ctx.pool)

    cached = None
    try:
        cache = RecommendationCache.load(CACHE_PATH, fingerprint=_fingerprint(ctx))
        cached = cache.get(key)
    except FileNotFoundError:
        st.warning("No recommendation cache found — every pick will run live.")
    except Exception as exc:  # CacheStaleError and anything else: never serve stale
        st.error(f"Cache unusable, falling back to live: {exc}")

    if cached is not None:
        st.success(
            f"cache hit — full budget "
            f"({FULL_BUDGET.n_rollouts}x{FULL_BUDGET.n_sims_per_rollout})"
        )
        rows = cached
    else:
        t0 = time.perf_counter()
        with st.spinner("No cached state — simulating live..."):
            res = recommend(board.to_draft_state(), ctx, LIVE_BUDGET)
        rows = [
            {
                "name": c.name,
                "position": c.position,
                "championship_probability": c.championship_probability,
                "standard_error": c.standard_error,
                "gap_from_leader_pp": c.gap_from_leader_pp,
                "indistinguishable_from_leader": c.indistinguishable_from_leader,
            }
            for c in res.candidates
        ]
        st.warning(
            f"cache miss — LIVE reduced budget "
            f"({LIVE_BUDGET.n_rollouts}x{LIVE_BUDGET.n_sims_per_rollout}), "
            f"{time.perf_counter() - t0:.1f}s. Wider error bars than cached picks."
        )

    tied = [r for r in rows if r["indistinguishable_from_leader"]]
    if len(tied) > 1:
        st.info(
            f"{len(tied)} candidates are statistically indistinguishable from the "
            "leader. Treat them as equivalent and use your own judgement."
        )
    st.dataframe(rows, width="stretch")


def _fingerprint(ctx) -> str:
    from ffdraft.store import fingerprint

    return fingerprint(ctx.rankings, ctx.adp_table)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test it**

Run: `.venv/bin/streamlit run src/ffdraft/live/app.py --server.headless true`
Expected: page loads, pick entry records and advances the board, undo reverts, and at pick 8 a recommendation appears (live, since the cache will be empty on first run).

- [ ] **Step 4: Commit**

```bash
git add src/ffdraft/live/app.py pyproject.toml
git commit -m "feat: draft-day Streamlit UI with provenance and tie grouping"
```

---

## Task 9: Full-suite check and docs

**Files:**
- Modify: `docs/HANDOFF.md`
- Modify: `docs/superpowers/plans/README.md`
- Modify: `README.md`

- [ ] **Step 1: Run the whole suite**

```bash
.venv/bin/python -m pytest -q > /tmp/pytest_stage4.txt 2>&1; echo "EXIT=$?"; tail -3 /tmp/pytest_stage4.txt
```

Expected: `EXIT=0`. **Check the exit code from the redirect, never through a pipe** — a pipeline reports the last command's status, which masked a crashed suite as exit 0 during Stage 3 (HANDOFF open item 8b).

- [ ] **Step 2: Update the docs**

In `docs/HANDOFF.md`: mark Stage 4 complete in §3; add a §11 recording the **measured** `LIVE_BUDGET`, the **measured** cache hit rate per round, and the final `TOP_N_FOR_KEY` with its validation evidence.

In `docs/superpowers/plans/README.md`: mark Stage 4 **Complete** in the stages table.

In `README.md`: add the two draft-day commands — rebuild the cache, then run the app.

- [ ] **Step 3: Commit**

```bash
git add docs README.md
git commit -m "docs: record Stage 4 measured budgets, hit rates, and key validation"
```

---

## Done when

- `.venv/bin/pytest` passes on the native arm64 venv, exit code checked without a pipe.
- Every one of our 18 picks yields a recommendation by cache or fallback — no path returns nothing.
- `LIVE_BUDGET` is measured to fit ~60s and the measurement is recorded, not asserted.
- Cache hit rate per round is reported from simulation.
- Cache and live paths are proven identical at equal budget.
- A stale cache raises `CacheStaleError` rather than serving a wrong answer.
- The UI always shows provenance, groups statistical ties, and carries the standing caveat.
