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
from .survival import LIKELY_SURVIVOR

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
    tool calling them equal", so this states the reason once and then
    redirects to the dimension that still discriminates: availability.

    **A zero `wait_cost_pp` is not shown at all.** It means a *comparable*
    candidate is likely to last -- not that this one will -- so it is zero
    for a player who is 7% to survive sitting beside one who is 69%. An
    earlier version printed it as "free to pass", which was misleading, and
    then spent three sentences explaining why it was not. A number that
    discriminates nothing and needs a paragraph of defence should not be on
    screen: when every cost is zero, survival is the entire decision, and
    the lines say only that. A *non*-zero cost is real information and is
    shown.

    Ordered by `wait_cost_pp` then ascending `p_survive`, matching
    `choose_from_tied`, and the closing line names the same player that
    function would pick.
    """
    n = len(tied)
    head = (
        f"**{n} candidates are statistically indistinguishable from the leader** "
        f"— the gaps between them are smaller than the uncertainty *in* those "
        f"gaps, so their order carries no information."
    )

    rated = [r for r in tied if r.get("wait_cost_pp") is not None]
    if not rated:
        return (
            f"{head} There is no later turn to weigh them against, so use your "
            f"own judgement — need, injury news, bye weeks."
        )

    def survival(r):
        return 1.0 if r.get("p_survive") is None else r["p_survive"]

    ranked = sorted(rated, key=lambda r: (-(r["wait_cost_pp"] or 0.0), survival(r)))

    lines = "\n".join(
        f"- **{r['name']}** ({r['position']}) — "
        + (
            f"**{survival(r) * 100:.0f}%** likely to still be there"
            if r.get("p_survive") is not None
            else "survival unknown"
        )
        + (f" · passing costs **{r['wait_cost_pp']:.2f}pp**" if r["wait_cost_pp"] else "")
        for r in ranked
    )

    top = ranked[0]
    if top["wait_cost_pp"]:
        tail = (
            f"Take **{top['name']}** — passing costs {top['wait_cost_pp']:.2f}pp "
            f"of title equity."
        )
    elif survival(top) < LIKELY_SURVIVOR:
        tail = (
            f"Take **{top['name']}** — at {survival(top) * 100:.0f}% he is the one "
            f"you are least likely to still have."
        )
    else:
        tail = (
            f"No urgency — all of them should last (lowest is "
            f"{survival(top) * 100:.0f}%). Decide on roster need."
        )

    return f"{head} Decide on **availability**:\n\n{lines}\n\n{tail}"


SKILL_POSITIONS = frozenset({"RB", "WR"})


def describe_skill_alternative(rows: Sequence[dict]) -> str | None:
    """When the engine leads with a QB or TE, the best back or receiver
    beside it, and what the simulation thinks that costs. `None` when
    there is no trade to show.

    **Free to compute.** Every row was already simulated at the same
    budget with the same paired seeds, so this is a lookup, not a second
    `recommend_pick` -- which would double a ~17s pick.

    Why it earns the space: `draft.value` prices each pick as a static gap
    to replacement and never asks how fast that gap decays, which tilts it
    toward QB and TE (HANDOFF open item 3; measured 2026-08-22 at 13 of 24
    picks in rounds 3-8 across four drafts, including a *second*
    quarterback in round 7). The simulation cannot referee that, because
    the rollout's own future picks use the same greedy function -- both
    branches end QB-heavy. So the owner is shown the trade rather than
    being asked to trust a number whose bias the number cannot see.

    **The copy is deliberately calibrated to what was measured.** The
    structure study, extended to rounds 4-8 on 2026-08-22, found forcing
    RB/WR through round 5 or 8 worth `+0.22 +- 3.88pp` and
    `+1.00 +- 4.46pp` against the unconstrained engine -- a null. Only the
    rounds 1-3 result survives, and only in sign. Saying more than that
    here would be the tool talking itself into a strategy the backtest
    does not support.
    """
    if not rows or rows[0]["position"] in SKILL_POSITIONS:
        return None
    skill = [r for r in rows if r["position"] in SKILL_POSITIONS]
    # `max`, not `skill[0]`. Callers pass rows already ranked by
    # championship probability, so the two agree today -- but depending on
    # that coupling means a future re-sort of the table would silently
    # start naming the wrong player as "best".
    if not skill:
        return (
            "**No running back or receiver was simulated at all** — every "
            "candidate the value function offered was a QB, TE, K or D/ST. "
            "That is the positional tilt at its worst; treat the leader with "
            "suspicion and use the best skill player on your own board."
        )
    best = max(skill, key=lambda r: r["championship_probability"])
    gap = (rows[0]["championship_probability"] - best["championship_probability"]) * 100
    return (
        f"**Best RB/WR: {best['name']} ({best['position']}) — "
        f"{best['championship_probability'] * 100:.1f}%, {gap:.2f}pp behind.** "
        f"The value function tilts toward QB/TE. Backtested, forcing skill "
        f"positions in **rounds 1-3** was better in 9 of 9 comparisons; past "
        f"round 3 it made **no measured difference**."
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
    1b. **Lowest `p_survive`**, once the costs tie. A zero cost means "a
       comparable player is likely to last", not "this player will last",
       so a tie of zeros is decided by who is actually scarce.
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
            # Scarcity, and it has to come second. `wait_cost_pp` is zero
            # whenever a comparable candidate is likely to last, which makes
            # it zero for a 7%-to-survive player sitting beside a 69% one --
            # so on a tie of zeros this used to fall through to championship
            # probability and take the *available* player over the scarce
            # one, losing the scarce one for nothing. Ascending, so least
            # likely to survive wins.
            1.0 if r.get("p_survive") is None else r["p_survive"],
            _is_deferrable(r["position"], rounds_left),
            not _is_real_need(r["position"], counts, rounds_left),
            -r["championship_probability"],
            r["player_id"],  # deterministic
        ),
    )[0]
