"""Turning a ranked candidate list into one pick.

`recommend_pick` often cannot separate its top candidates -- it flags them
`indistinguishable_from_leader`. Something still has to choose, and *how*
matters: a full mock draft showed the naive rules producing picks that can
never start.

## Two rules, both from observed failures

**Caps.** The engine drafts a third quarterback and a second kicker in
every rollout (HANDOFF open item 3), and in the mock draft those were picks
5, 12 and 15 -- roughly 44 projected points parked on the bench for the
season, since only one QB and one K can ever start. `filter_capped` removes
them from the choice set. Measured worth: +0.67pp +/- 0.75 (test 4b), i.e.
statistically insignificant, so this is about not wasting picks rather than
a demonstrated gain.

**Scarcity-weighted need.** The mock draft took Brandon Aubrey, a kicker,
at pick 88 against ADP 131.5 -- a 43-pick reach. He was not the leader; he
won a tie-break. Every tied candidate had `wait_cost_pp` of 0, so the rule
fell through to "unfilled starter slot", and we had no kicker. **The roster
-need rule overrode a survival signal that said wait** (Aubrey: 100%
likely to last). Need must be weighted by scarcity: kickers and defenses
are always available, so their slots are only a real need once the draft is
nearly done.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..league import STARTERS

# Only one QB and one K can start. A second QB is worth having as cover; a
# third never plays. A second kicker never plays either.
POSITION_CAPS: dict[str, int] = {"QB": 2, "K": 1}

# Always available on the wire, so filling these early is a wasted pick.
# D/ST is here for completeness -- the engine never drafts one anyway, since
# the shared-distribution decision makes a drafted D/ST worth exactly the
# replacement D/ST (HANDOFF section 5).
LATE_FILL_POSITIONS = frozenset({"K", "DST"})

# Within this many rounds of the end, a `LATE_FILL_POSITIONS` slot counts as
# a genuine need.
LATE_FILL_WINDOW = 2


def filter_capped(rows: Sequence[dict], counts: Mapping[str, int]) -> list[dict]:
    """Drop candidates at a position we already have enough of.

    **Returns an empty list when every candidate is capped, deliberately.**
    An earlier version fell back to returning the original rows so the
    caller always had something -- and that silently disabled the cap
    exactly when it mattered. By the late rounds the recommender's whole
    candidate list is QBs and kickers (the value function's documented
    positional tilt), so "all capped" is the *common* case, not an edge
    case: the mock draft ended with four quarterbacks.

    Callers must handle the empty case with `best_uncapped_available`,
    which looks past the candidate list to the pool.
    """
    return [
        r for r in rows if counts.get(r["position"], 0) < POSITION_CAPS.get(r["position"], 99)
    ]


def best_uncapped_available(
    pool: Mapping[str, object],
    drafted_ids: set[str],
    counts: Mapping[str, int],
    adp_by_player_id: Mapping[str, float] | None = None,
) -> dict | None:
    """Best available player at a position we are not capped at.

    The escape hatch for when every recommended candidate is capped out.
    Ranked by ADP where known, else board rank -- deliberately the market's
    order rather than the value function's, since it is the value function's
    positional tilt that created the situation.

    Returned in the same row shape the UI and tie-break consume, with
    `championship_probability` set to None: this pick was not simulated and
    must not be displayed as though it were.
    """
    best, best_key = None, None
    for pid, p in pool.items():
        if pid in drafted_ids:
            continue
        if counts.get(p.position, 0) >= POSITION_CAPS.get(p.position, 99):
            continue
        key = (
            adp_by_player_id.get(pid, float(p.rank) + 1000.0)
            if adp_by_player_id
            else float(p.rank)
        )
        if best_key is None or key < best_key:
            best, best_key = pid, key
    if best is None:
        return None
    p = pool[best]
    return {
        "player_id": best,
        "name": p.name,
        "position": p.position,
        "championship_probability": None,
        "standard_error": None,
        "gap_from_leader_pp": None,
        "indistinguishable_from_leader": False,
        "p_survive": None,
        "wait_cost_pp": None,
        "uncapped_fallback": True,
    }


def _is_real_need(position: str, counts: Mapping[str, int], rounds_left: int) -> bool:
    """Is this position's dedicated starter slot genuinely unfilled *and*
    worth filling now?"""
    if counts.get(position, 0) >= STARTERS.get(position, 0):
        return False
    if position in LATE_FILL_POSITIONS:
        return rounds_left <= LATE_FILL_WINDOW
    return True


def _is_deferrable(position: str, rounds_left: int) -> bool:
    """True while a `LATE_FILL_POSITIONS` pick would be premature. Sorts
    those candidates to the back of a tie outright."""
    return position in LATE_FILL_POSITIONS and rounds_left > LATE_FILL_WINDOW


def describe_tie(tied: Sequence[dict]) -> str:
    """Markdown explaining a tie, and what to decide on instead.

    The flag is routinely read as "these have different numbers, why is the
    tool calling them equal" -- so this says the actual reason once
    (the gaps are smaller than the uncertainty *in* the gaps) and then
    redirects to the dimension that is still informative.

    **Ordered by `wait_cost_pp`, not by championship probability.** Once
    the model cannot separate the candidates on title equity, the ranking
    it is displayed in carries no information, and leading with it invites
    reading a tie as an ordering. Availability does carry information, and
    it is the first key `choose_from_tied` sorts on, so the callout and the
    automatic choice agree by construction.

    Three cases, because the honest advice genuinely differs:

    * no next turn (our last pick) -- nothing to wait for, so fall back to
      judgement;
    * somebody is unlikely to last -- name him;
    * everybody is likely to last -- say so, because "no urgency" is a
      real answer and inventing a reason to hurry would be worse.
    """
    n = len(tied)
    head = (
        f"**{n} candidates are statistically indistinguishable from the leader.** "
        f"The gaps between them are smaller than the uncertainty *in* those gaps, "
        f"so the order they are listed in carries no information."
    )

    rated = [r for r in tied if r.get("wait_cost_pp") is not None]
    if not rated:
        return (
            f"{head} There is no later turn to weigh them against, so use your "
            f"own judgement — need, injury news, bye weeks."
        )

    # Secondary key: among candidates that cost nothing to pass, the one
    # least likely to survive still deserves to be read first. Wait cost
    # can be zero for a coin-flip survivor -- it is clamped once a likely
    # survivor scores as well as he does -- and sorting on cost alone would
    # bury a 49% player behind a 99% one.
    ranked = sorted(
        rated,
        key=lambda r: (-(r["wait_cost_pp"] or 0.0),
                       1.0 if r.get("p_survive") is None else r["p_survive"]),
    )
    lines = "\n".join(
        f"- **{r['name']}** ({r['position']}) — "
        + (
            f"{r['wait_cost_pp']:.2f}pp to pass"
            if r["wait_cost_pp"]
            else "free to pass"
        )
        + (
            f" · {r['p_survive'] * 100:.0f}% still there next turn"
            if r.get("p_survive") is not None
            else ""
        )
        for r in ranked
    )

    top = ranked[0]
    if top["wait_cost_pp"]:
        tail = (
            f"\n\n**{top['name']}** costs the most to pass on — he is the one "
            f"least likely to be there at your next pick."
        )
    else:
        tail = (
            "\n\nNone of them costs anything to pass on: they are all likely to "
            "last until your next turn. No urgency here — decide on roster need."
        )

    return (
        f"{head} Decide on **availability** instead — expected title equity "
        f"lost by passing now:\n\n{lines}{tail}"
    )


def choose_from_tied(
    rows: Sequence[dict],
    counts: Mapping[str, int],
    rounds_left: int,
) -> dict:
    """Pick one candidate from a ranked, survival-annotated list.

    Order of preference among candidates the model cannot separate:

    1. **Highest `wait_cost_pp`** -- of two players worth the same, take the
       one who will not be there next time.
    2. **Actively demote K/D/ST** while the draft is young. Merely not
       counting them as a "need" is not enough: with every other key tied,
       the choice fell through to the deterministic id tie-break and a
       kicker still won. They have to be pushed to the back.
    3. **Genuine roster need** among the remaining positions.
    4. Championship probability, then player id, for determinism.
    """
    tied = [r for r in rows if r.get("indistinguishable_from_leader")] or list(rows[:1])
    if len(tied) == 1:
        return tied[0]
    return sorted(
        tied,
        key=lambda r: (
            -(r.get("wait_cost_pp") or 0.0),
            _is_deferrable(r["position"], rounds_left),
            not _is_real_need(r["position"], counts, rounds_left),
            -r["championship_probability"],
            r["player_id"],  # deterministic
        ),
    )[0]
