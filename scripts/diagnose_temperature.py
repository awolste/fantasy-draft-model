"""Is the opponent model too random early? (HANDOFF open item 3b)

`models/opponent.DEFAULT_TEMPERATURE` is a single global 3.0, applied
identically at pick 1 and pick 180. Real drafts are not like that: the top
of round 1 is near-deterministic (everyone agrees who the best player is),
while round 15 is idiosyncratic. A single temperature has to split the
difference, which makes early picks far more random than they really are --
the symptom the owner noticed on draft day, where elite players kept
surviving to pick 8 in the tool's rollouts.

This measures the claim rather than assuming it, by comparing like for
like: **how good is the best player still on the board at each of our
picks**, in ADP-rank terms, in the model versus in this league's seven real
drafts.

`best available ADP rank` is the cleanest statistic for this. If the model
is too random early, elite players survive too long and this number is too
small (too good) at our early picks compared with reality.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`; refers
to teams by draft slot only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import polars as pl

from ffdraft import store
from ffdraft.models.opponent import (
    DEFAULT_ROSTER_DECAY,
    DEFAULT_TEMPERATURE,
    TRAINING_SEASONS,
    AvailablePlayer,
    _POSITION_ALIASES,
    _resolve_picks,
    build_training_set,
    fit_opponent_model,
    sample_pick,
)
from ffdraft.ids import normalize_name

# Our own picks at slot 8 in a 10-team snake, first few rounds -- the only
# ones where "did an elite player survive?" is a live question.
REPORT_PICKS = (8, 13, 28, 33, 48)


def season_pool(adp_history: pl.DataFrame, season: int) -> dict[str, AvailablePlayer]:
    """That season's full ADP board, keyed exactly as `_resolve_picks` keys
    real picks so a real pick can be removed from it directly."""
    pool: dict[str, AvailablePlayer] = {}
    for row in adp_history.filter(pl.col("season") == season).iter_rows(named=True):
        if row["position"] == "DST":
            key, position = row["team"], "DST"
        else:
            key = normalize_name(row["name"])
            position = _POSITION_ALIASES.get(row["position"], row["position"])
        pool[key] = AvailablePlayer(player_id=key, position=position, adp=row["adp"])
    return pool


def real_best_available(adp_history, league_drafts, league_managers, crosswalk, seasons):
    """For each season, the ADP rank of the best player still on the board
    at each of `REPORT_PICKS` -- measured from the real draft."""
    out: dict[int, list[float]] = defaultdict(list)
    for season in seasons:
        pool = season_pool(adp_history, season)
        if not pool:
            continue
        adp_order = {
            pid: i + 1
            for i, pid in enumerate(sorted(pool, key=lambda k: pool[k].adp))
        }
        drafts = league_drafts.filter(pl.col("season") == season).sort("overall_pick")
        resolved = _resolve_picks(drafts, league_managers, crosswalk, adp_history).sort(
            "overall_pick"
        )
        live = dict(pool)
        for row in resolved.iter_rows(named=True):
            if row["overall_pick"] in REPORT_PICKS and live:
                out[row["overall_pick"]].append(
                    min(adp_order[pid] for pid in live)
                )
            live.pop(row["player_key"], None)
    return out


def simulated_best_available(model, adp_history, seasons, temperature, n_sims, seed):
    """The same statistic, with the opponent model doing the drafting.

    `temperature` is either a float (pin one value at every pick) or the
    string "schedule" (use the fitted per-round `temperature_for_round`).
    """
    rng = np.random.default_rng(seed)
    out: dict[int, list[float]] = defaultdict(list)
    for season in seasons:
        pool = season_pool(adp_history, season)
        if not pool:
            continue
        adp_order = {
            pid: i + 1
            for i, pid in enumerate(sorted(pool, key=lambda k: pool[k].adp))
        }
        for _ in range(n_sims):
            live = dict(pool)
            counts: dict[int, dict[str, int]] = {}
            for overall in range(1, max(REPORT_PICKS) + 1):
                if overall in REPORT_PICKS and live:
                    out[overall].append(min(adp_order[pid] for pid in live))
                if not live:
                    break
                # 10-team snake; team identity only drives roster_counts.
                rnd, idx = divmod(overall - 1, 10)
                team = (idx + 1) if rnd % 2 == 0 else (10 - idx)
                c = counts.setdefault(team, {})
                taken = sample_pick(
                    model, f"slot_{team}", list(live.values()), c, rng,
                    temperature=None if temperature == "schedule" else temperature,
                    roster_decay=DEFAULT_ROSTER_DECAY,
                    round_=rnd + 1,
                )
                c[live[taken].position] = c.get(live[taken].position, 0) + 1
                live.pop(taken)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--temperatures", nargs="+",
                    default=[DEFAULT_TEMPERATURE, "schedule"],
                    help="floats, and/or the literal 'schedule' for the fitted per-round fit")
    args = ap.parse_args()
    args.temperatures = [t if t == "schedule" else float(t) for t in args.temperatures]

    league_drafts = store.read("league_drafts")
    league_managers = store.read("league_managers")
    crosswalk = store.read("id_crosswalk")
    adp_history = store.read("adp_history")

    training, _ = build_training_set(
        league_drafts, league_managers, crosswalk, adp_history
    )
    model = fit_opponent_model(training)

    real = real_best_available(
        adp_history, league_drafts, league_managers, crosswalk, TRAINING_SEASONS
    )

    print("\nBest-available ADP rank at each of our picks (slot 8).")
    print("Higher = the good players are gone. The model is too random if its")
    print("numbers sit BELOW the real ones -- elite players surviving too long.\n")

    header = f"{'pick':>5}{'REAL':>12}"
    for t in args.temperatures:
        header += f"{t if t == 'schedule' else 'T=' + format(t, 'g'):>12}"
    print(header)

    sims = {
        t: simulated_best_available(
            model, adp_history, TRAINING_SEASONS, t, args.n_sims, args.seed
        )
        for t in args.temperatures
    }

    for pick in REPORT_PICKS:
        row = f"{pick:>5}{np.mean(real[pick]):>12.1f}"
        for t in args.temperatures:
            row += f"{np.mean(sims[t][pick]):>12.1f}"
        print(row)

    print("\nP(the overall ADP-1 player is still available) at each pick:")
    header = f"{'pick':>5}{'REAL':>12}"
    for t in args.temperatures:
        header += f"{t if t == 'schedule' else 'T=' + format(t, 'g'):>12}"
    print(header)
    for pick in REPORT_PICKS:
        row = f"{pick:>5}{np.mean([v == 1 for v in real[pick]]) * 100:>11.0f}%"
        for t in args.temperatures:
            row += f"{np.mean([v == 1 for v in sims[t][pick]]) * 100:>11.0f}%"
        print(row)


if __name__ == "__main__":
    main()
