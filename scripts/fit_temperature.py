"""Fit the opponent model's softmax temperature by maximum likelihood,
per round, against this league's real drafts (HANDOFF open item 3b).

`DEFAULT_TEMPERATURE = 3.0` was picked by "a small grid search against the
2024 holdout backtest" optimising top-1 accuracy -- a coarse, single-season
criterion, and a single number applied identically at pick 1 and pick 180.
`scripts/diagnose_temperature.py` shows that one number is wrong in both
directions at once:

```
best-available ADP rank    pick 8: real 5.1  vs  model 4.2   (too random)
                          pick 48: real 32.1 vs  model 41.6  (too rigid)
P(consensus ADP-1 survives) pick 8: real 0%   vs  model 11%
```

So this fits temperature **as a function of round**, by maximum likelihood
over every usable pick in `TRAINING_SEASONS`, rather than by grid-searching
one accuracy number on one season.

## The likelihood

`pick_probabilities` gives player `i` weight `exp(-rank_i / T) * mult_i`,
where `rank_i` is its rank on the manager's reach-adjusted board and
`mult_i` is the roster-need penalty. For an observed pick `a`:

    log P(a) = -rank_a / T + log(mult_a) - log(sum_i exp(-rank_i/T) * mult_i)

`mult` does not depend on `T`, so this is a clean one-parameter problem per
round, and it uses *every* pick rather than only whether the top-1 guess
was right. Nothing is refit except `T`: the reach model and `roster_decay`
are held at their existing fitted values, so this cannot silently absorb a
misfit from either.

## Validation

Leave-one-season-out. Temperature is fit on six seasons and scored on the
seventh, against the incumbent global 3.0, on both held-out log-likelihood
(what was fit) and top-1/top-5 accuracy (what the original grid search
optimised, kept so the comparison is not rigged in the new fit's favour).

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import polars as pl

from ffdraft import store
from ffdraft.ids import normalize_name
from ffdraft.models.opponent import (
    DEFAULT_ROSTER_DECAY,
    DEFAULT_TEMPERATURE,
    TRAINING_SEASONS,
    AvailablePlayer,
    OpponentModel,
    _MIN_ROSTER_MULTIPLIER,
    _POSITION_ALIASES,
    _resolve_picks,
    build_training_set,
    fit_opponent_model,
)

T_GRID = np.round(np.concatenate([np.arange(0.2, 10.0, 0.05), np.arange(10.0, 120.1, 0.5)]), 2)


@dataclass(frozen=True)
class PickObservation:
    """One real pick, reduced to everything the likelihood needs.

    `ranks` and `mults` cover every player available at that pick; `actual`
    indexes the one really taken. Storing it this way means the whole
    temperature grid can be evaluated without re-deriving the board.
    """

    season: int
    round: int
    ranks: np.ndarray
    mults: np.ndarray
    actual: int


def collect_observations(
    model: OpponentModel,
    seasons,
    league_drafts,
    league_managers,
    crosswalk,
    adp_history,
    roster_decay: float = DEFAULT_ROSTER_DECAY,
) -> list[PickObservation]:
    """Replay each season's real draft, recording the reach-adjusted board
    at every pick that can be scored.

    Mirrors `backtest_holdout_season`'s replay exactly -- the pool is that
    season's ADP board, picks are removed as they really happened, and
    roster counts advance on every pick whether or not it was itself
    ADP-matched -- so the observations are drawn from the same process the
    model is asked to predict.
    """
    obs: list[PickObservation] = []
    for season in seasons:
        pool: dict[str, AvailablePlayer] = {}
        for row in adp_history.filter(pl.col("season") == season).iter_rows(named=True):
            if row["position"] == "DST":
                key, position = row["team"], "DST"
            else:
                key = normalize_name(row["name"])
                position = _POSITION_ALIASES.get(row["position"], row["position"])
            pool[key] = AvailablePlayer(player_id=key, position=position, adp=row["adp"])
        if not pool:
            continue

        drafts = league_drafts.filter(pl.col("season") == season).sort("overall_pick")
        resolved = _resolve_picks(drafts, league_managers, crosswalk, adp_history).sort(
            "overall_pick"
        )

        counts_by_manager: dict[str, dict[str, int]] = {}
        for row in resolved.iter_rows(named=True):
            key = row["player_key"]
            counts = counts_by_manager.setdefault(row["manager_id"], {})

            if key in pool and pool:
                players = list(pool.values())
                adjusted = np.array(
                    [
                        np.log(p.adp) - model.predicted_reach(row["manager_id"], p.position)
                        for p in players
                    ]
                )
                order = np.argsort(adjusted, kind="stable")
                ranks = np.empty(len(players))
                ranks[order] = np.arange(len(players))
                mults = np.array(
                    [
                        max(roster_decay ** counts.get(p.position, 0), _MIN_ROSTER_MULTIPLIER)
                        for p in players
                    ]
                )
                actual = next(i for i, p in enumerate(players) if p.player_id == key)
                obs.append(
                    PickObservation(
                        season=season, round=row["round"],
                        ranks=ranks, mults=mults, actual=actual,
                    )
                )

            pool.pop(key, None)
            counts[row["position"]] = counts.get(row["position"], 0) + 1
    return obs


def log_likelihood(obs: list[PickObservation], temperature: float) -> float:
    """Total log-likelihood of `obs` at one temperature."""
    total = 0.0
    for o in obs:
        logw = -o.ranks / temperature + np.log(o.mults)
        total += float(logw[o.actual] - np.logaddexp.reduce(logw))
    return total


def best_temperature(obs: list[PickObservation], grid=T_GRID) -> tuple[float, float]:
    """The grid temperature maximising `obs`' log-likelihood."""
    if not obs:
        return float("nan"), float("nan")
    lls = [log_likelihood(obs, t) for t in grid]
    i = int(np.argmax(lls))
    return float(grid[i]), float(lls[i])


def accuracy(obs: list[PickObservation], temp_of_round) -> tuple[float, float]:
    """Top-1 and top-5 accuracy under a temperature schedule."""
    if not obs:
        return float("nan"), float("nan")
    top1 = top5 = 0
    for o in obs:
        w = -o.ranks / temp_of_round(o.round) + np.log(o.mults)
        order = np.argsort(-w, kind="stable")
        top1 += int(order[0] == o.actual)
        top5 += int(o.actual in order[:5])
    return top1 / len(obs), top5 / len(obs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-round", type=int, default=18)
    ap.add_argument("--shape-only", action="store_true",
                    help="per-round MLE only; skip the leave-one-season-out pass")
    args = ap.parse_args()

    league_drafts = store.read("league_drafts")
    league_managers = store.read("league_managers")
    crosswalk = store.read("id_crosswalk")
    adp_history = store.read("adp_history")

    training, _ = build_training_set(league_drafts, league_managers, crosswalk, adp_history)
    model = fit_opponent_model(training)

    print("collecting observations from the real drafts...", flush=True)
    obs = collect_observations(
        model, TRAINING_SEASONS, league_drafts, league_managers, crosswalk, adp_history
    )
    print(f"  {len(obs)} scorable picks across {len(TRAINING_SEASONS)} seasons\n")

    # --- the shape: MLE temperature per round, no functional form assumed
    print("MLE temperature by round (all seasons pooled):")
    print(f"{'round':>6}{'n':>6}{'T_hat':>9}{'LL/pick':>10}")
    rounds, temps, weights = [], [], []
    for rnd in range(1, args.max_round + 1):
        at = [o for o in obs if o.round == rnd]
        if len(at) < 20:
            continue
        t, ll = best_temperature(at)
        rounds.append(rnd)
        temps.append(t)
        weights.append(len(at))
        print(f"{rnd:>6}{len(at):>6}{t:>9.2f}{ll / len(at):>10.3f}")

    rounds_a = np.array(rounds, dtype=float)
    temps_a = np.array(temps, dtype=float)
    slope, intercept = np.polyfit(rounds_a, temps_a, 1, w=np.array(weights, dtype=float))
    print(f"\nweighted linear fit:  T(round) = {intercept:.3f} + {slope:.3f} * round")
    print(f"incumbent global:     T = {DEFAULT_TEMPERATURE}")

    if args.shape_only:
        return

    # --- leave-one-season-out, comparing candidate functional forms.
    #
    # `per_round` is the least constrained (one free parameter per round,
    # ~70 picks each) and `linear` the most; the incumbent global is the
    # null. Scoring all three on held-out seasons is what distinguishes "the
    # schedule is real" from "17 parameters fit 17 rounds of noise".
    print(f"\n{'=' * 78}\nLEAVE-ONE-SEASON-OUT (fit on 6 seasons, scored on the 7th)\n{'=' * 78}")
    print(f"{'holdout':>8}" + "".join(f"{f'LL {k}':>14}" for k in ("global", "linear", "per_round"))
          + f"{'top1 g':>9}{'top1 pr':>9}{'top5 g':>9}{'top5 pr':>9}")

    results: dict[str, list[float]] = {"linear": [], "per_round": []}
    early_results: dict[str, list[float]] = {"linear": [], "per_round": []}
    acc: dict[str, list[tuple[float, float]]] = {"global": [], "per_round": []}

    def _ll(obs_list, temp_of_round) -> float:
        total = 0.0
        for o in obs_list:
            logw = -o.ranks / temp_of_round(o.round) + np.log(o.mults)
            total += float(logw[o.actual] - np.logaddexp.reduce(logw))
        return total / len(obs_list)

    for holdout in TRAINING_SEASONS:
        fit_seasons = tuple(s for s in TRAINING_SEASONS if s != holdout)
        tr, _ = build_training_set(
            league_drafts, league_managers, crosswalk, adp_history, seasons=fit_seasons
        )
        m = fit_opponent_model(tr)
        fit_obs = collect_observations(
            m, fit_seasons, league_drafts, league_managers, crosswalk, adp_history
        )
        hold_obs = collect_observations(
            m, (holdout,), league_drafts, league_managers, crosswalk, adp_history
        )
        if not hold_obs:
            continue

        table: dict[int, float] = {}
        rr, tt, ww = [], [], []
        for rnd in range(1, args.max_round + 1):
            at = [o for o in fit_obs if o.round == rnd]
            if len(at) < 20:
                continue
            t, _ = best_temperature(at)
            table[rnd] = t
            rr.append(rnd)
            tt.append(t)
            ww.append(len(at))
        s, b = np.polyfit(np.array(rr, float), np.array(tt, float), 1, w=np.array(ww, float))
        deepest = max(table)

        schedules = {
            "global": lambda r: DEFAULT_TEMPERATURE,
            "linear": lambda r, s=s, b=b: max(0.2, b + s * r),
            # Rounds past the fitted table reuse the deepest fitted round
            # rather than extrapolating a line off the end of the data.
            "per_round": lambda r, t=table, d=deepest: t.get(min(r, d), t[d]),
        }

        lls = {k: _ll(hold_obs, f) for k, f in schedules.items()}
        results["linear"].append(lls["linear"] - lls["global"])
        results["per_round"].append(lls["per_round"] - lls["global"])

        # Rounds 1-3 on their own. Aggregate LL is dominated by the ~80% of
        # picks in rounds 6-17, where nothing we decide is at stake; our own
        # first three picks are the ones the whole tool exists to get right,
        # and the candidate forms disagree most there (linear's intercept is
        # dragged up by the late rounds).
        early = [o for o in hold_obs if o.round <= 3]
        if early:
            early_lls = {k: _ll(early, f) for k, f in schedules.items()}
            for k in ("linear", "per_round"):
                early_results[k].append(early_lls[k] - early_lls["global"])
        acc["global"].append(accuracy(hold_obs, schedules["global"]))
        acc["per_round"].append(accuracy(hold_obs, schedules["per_round"]))

        print(f"{holdout:>8}" + "".join(f"{lls[k]:>14.3f}" for k in ("global", "linear", "per_round"))
              + f"{acc['global'][-1][0] * 100:>8.1f}%{acc['per_round'][-1][0] * 100:>8.1f}%"
              f"{acc['global'][-1][1] * 100:>8.1f}%{acc['per_round'][-1][1] * 100:>8.1f}%")

    def _mean_se(xs):
        m = float(np.mean(xs))
        se = float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else float("nan")
        return m, se

    print()
    for name, ds in results.items():
        m, se = _mean_se(ds)
        print(f"held-out LL/pick, {name:>9} - global: {m:+.4f} (SE {se:.4f})  "
              f"better in {sum(1 for d in ds if d > 0)}/{len(ds)} seasons")

    print("\nROUNDS 1-3 ONLY -- the picks the tool actually advises on:")
    for name, ds in early_results.items():
        m, se = _mean_se(ds)
        print(f"held-out LL/pick, {name:>9} - global: {m:+.4f} (SE {se:.4f})  "
              f"better in {sum(1 for d in ds if d > 0)}/{len(ds)} seasons")

    d1 = [p[0] - g[0] for p, g in zip(acc["per_round"], acc["global"])]
    d5 = [p[1] - g[1] for p, g in zip(acc["per_round"], acc["global"])]
    m1, se1 = _mean_se(d1)
    m5, se5 = _mean_se(d5)
    print(f"\nheld-out top-1, per_round - global: {m1 * 100:+.2f}pp (SE {se1 * 100:.2f})")
    print(f"held-out top-5, per_round - global: {m5 * 100:+.2f}pp (SE {se5 * 100:.2f})")
    print("\nTop-k is the criterion the incumbent 3.0 was originally tuned on, kept")
    print("here so this comparison is not judged only by the metric being fit.")
    print("It is the wrong criterion for this model's actual job: rollouts SAMPLE")
    print("from the distribution, so calibration (LL) is what matters, not the")
    print("argmax. Reported to be honest about the trade, not to be optimised.")


if __name__ == "__main__":
    main()
