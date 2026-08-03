"""Draft-time decision logic: candidate valuation, rollout, and recommendation.

This package sits above `models/` and `sim/` -- it answers "what should we do
right now, cheaply" (this task, `value.py`) and, in later Stage 3 tasks,
"what happens if we play the draft forward" (`rollout.py`) and "which pick
maximizes championship equity" (`recommender.py`).
"""
