"""Shared interface for weekly fantasy-points distributions.

`sim/` fills a roster's lineup slots by sampling a weekly score for every
player (and for DST) and picking the best combination. To do that uniformly
it needs every slot -- kicker, defense, or skill player -- to expose the same
tiny surface: draw `size` samples given an explicit RNG, and report a mean.

This lives in its own module, separate from `defense.py` and the future
`distribution.py`, so neither has to import the other just to share a type.
`models/defense.py` (Task 2) and `models/distribution.py` (Task 3) both
depend on this module; this module depends on nothing in `models/`.
"""

from typing import Protocol

import numpy as np


class WeeklyDistribution(Protocol):
    """A sampleable distribution of one roster slot's weekly fantasy points."""

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw `size` independent weekly-point samples using `rng`.

        Callers must pass an explicit `np.random.Generator` -- never rely on
        global numpy random state -- so that simulations are reproducible
        from a seed.

        Performance note for callers (Task 7's simulator in particular):
        `sample` is vectorized and cheap per call regardless of `size` --
        one call of 14,000,000 takes ~0.2s. Flatten weeks x sims into a
        single `size` and reshape afterward; do not call `sample` in a
        per-week Python loop.
        """
        ...

    @property
    def mean(self) -> float:
        """The distribution's expected weekly points."""
        ...
