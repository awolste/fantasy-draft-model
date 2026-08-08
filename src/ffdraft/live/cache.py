"""Precomputed recommendations, and the one place `recommend_pick` is called.

## Why exactly one call site

The precomputed and live-fallback paths must never become two behaviours.
They are therefore the *same* function with a different `Budget`; see
`tests/test_live_cache.py::test_cache_and_live_paths_agree_at_equal_budget`,
which is the whole safety argument for having two paths at all. If that
test ever fails, the architecture is unsafe -- reconsider it rather than
loosening the assertion.

## Why the key is lossy

Keying on the exact drafted set is useless past round 2: the state space
branches faster than any precompute can cover. A recommendation depends on
which *good* players are available and what our roster needs; whether the
200th-ranked player is gone cannot change our pick, because we would never
take him. So the key keeps only availability within the top
`TOP_N_FOR_KEY` by **overall board rank**.

This is an approximation, and it is checked rather than assumed:
`scripts/validate_cache_key.py` builds state pairs that share a key but
differ below the cutoff, runs both at full budget, and compares the
recommendation. If they disagree, raise the cutoff -- do not accept the
mismatch, because it would mean the cache can serve a wrong answer on
draft day and never say so.

Note `PlayerDistribution.rank` is the **overall** board rank, not a
within-position one (pinned by
`tests/test_live_context.py::test_pool_rank_is_an_overall_rank_and_is_unique`).
"top 60 by rank" therefore means the top 60 players on the board, which is
what makes the approximation defensible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from ..draft.recommender import RecommendationResult, recommend_pick
from ..league import DRAFT_SLOT
from ..store import CacheStaleError
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
    pool: Mapping[str, Any],
    top_n: int = TOP_N_FOR_KEY,
) -> StateKey:
    """Key a draft state for the recommendation cache.

    Raises `ValueError` if `pool` is smaller than `top_n`: that means the
    caller passed a filtered subset rather than the full board, and keying
    on whatever happened to be present would produce keys that collide
    across genuinely different states.
    """
    if len(pool) < top_n:
        raise ValueError(
            f"pool has {len(pool)} players, smaller than the cache-key cutoff "
            f"of {top_n} -- pass the full board, not a filtered subset."
        )
    top_ids = sorted(pool, key=lambda pid: pool[pid].rank)[:top_n]
    return StateKey(
        overall_pick=overall_pick,
        roster_counts=tuple(sorted(roster_counts.items())),
        top_available=tuple(sorted(pid for pid in top_ids if pid not in drafted_ids)),
    )


def recommend(
    state: Any,
    ctx: Any,
    budget: Budget,
    our_team: int = DRAFT_SLOT,
) -> RecommendationResult:
    """The ONLY `recommend_pick` call in this package.

    Precompute passes `FULL_BUDGET`, the live fallback passes
    `LIVE_BUDGET`; nothing else differs between the two paths. `ctx` is a
    `live.context.DraftContext` (or a `HoldoutContext` adapted through
    `DraftContext.from_holdout`).
    """
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


def candidates_to_rows(result: RecommendationResult) -> list[dict]:
    """Flatten a result to JSON-serialisable rows for the cache and the UI.

    Keeps `standard_error` and `indistinguishable_from_leader`: a cached
    recommendation that dropped its uncertainty would present a strict
    ranking over what may be a statistical tie.
    """
    return [
        {
            "player_id": c.player_id,
            "name": c.name,
            "position": c.position,
            "championship_probability": c.championship_probability,
            "standard_error": c.standard_error,
            "gap_from_leader_pp": c.gap_from_leader_pp,
            "indistinguishable_from_leader": c.indistinguishable_from_leader,
        }
        for c in result.candidates
    ]


@dataclass
class RecommendationCache:
    """Precomputed recommendations, keyed by `StateKey`, guarded by a
    fingerprint of the artifacts that determine a recommendation.

    A fingerprint mismatch raises `CacheStaleError` instead of serving the
    entry. This project's characteristic failure is confident, plausible,
    wrong output (HANDOFF section 8), and a cache built against last
    month's player pool is exactly that -- so staleness must be loud.
    """

    path: Path
    fingerprint: str
    entries: dict[str, list[dict]] = field(default_factory=dict)

    @staticmethod
    def _encode(key: Any) -> str:
        return json.dumps(key, sort_keys=True, default=list)

    def put(self, key: Any, candidates: list[dict]) -> None:
        self.entries[self._encode(key)] = candidates

    def get(self, key: Any) -> list[dict] | None:
        return self.entries.get(self._encode(key))

    def __len__(self) -> int:
        return len(self.entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"fingerprint": self.fingerprint, "entries": self.entries})
        )

    @classmethod
    def load(cls, path: Path | str, fingerprint: str) -> RecommendationCache:
        path = Path(path)
        raw = json.loads(path.read_text())
        stored = raw.get("fingerprint")
        if stored != fingerprint:
            raise CacheStaleError(
                f"recommendation cache at {path} was built against fingerprint "
                f"{stored!r} but the current model artifacts fingerprint "
                f"{fingerprint!r}. Rebuild it with "
                f"scripts/build_recommendation_cache.py -- refusing to serve "
                f"recommendations from a stale player pool."
            )
        return cls(path=path, fingerprint=fingerprint, entries=raw["entries"])
