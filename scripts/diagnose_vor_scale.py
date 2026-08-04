"""Diagnostic: is our projections-to-VOR scaling sane?

`vor_only` (static argmax(mean - replacement_mean[pos])) scored 0.00% in
300 backtest realizations -- identical to random. A greedy argmax over a
*static* key is just "best available by a fixed list", so that list is the
whole policy. This prints it, next to ADP order, plus the replacement means
that define it.

Personal-data rule: never reads `.env` or `data/manager_labels.csv`.
"""

from __future__ import annotations

from ffdraft.backtest import fit_holdout_context


def main() -> None:
    ctx = fit_holdout_context()

    print("=== replacement means (points/week) ===")
    for pos, mean in sorted(ctx.replacement_means.items()):
        print(f"  {pos:<5}{mean:6.2f}")

    rows = []
    for pid, p in ctx.pool.items():
        mean = float(p.distribution.mean)
        vor = mean - ctx.replacement_means[p.position]
        rows.append((vor, mean, p.position, p.name, ctx.adp_by_player_id.get(pid, float("nan"))))
    rows.sort(reverse=True)

    print("\n=== top 40 by static VOR (this IS the vor_only draft order) ===")
    print(f"{'#':>3}  {'player':<22}{'pos':<5}{'mean':>7}{'vor':>7}{'adp':>8}")
    for i, (vor, mean, pos, name, adp) in enumerate(rows[:40], start=1):
        print(f"{i:>3}  {name[:21]:<22}{pos:<5}{mean:>7.2f}{vor:>7.2f}{adp:>8.1f}")

    print("\n=== position share of the top 40 / 100 by VOR, vs by ADP ===")
    def shares(seq):
        c: dict[str, int] = {}
        for r in seq:
            c[r[2]] = c.get(r[2], 0) + 1
        return dict(sorted(c.items()))

    by_adp = sorted(rows, key=lambda r: r[4])
    print(f"  VOR top40: {shares(rows[:40])}")
    print(f"  ADP top40: {shares(by_adp[:40])}")
    print(f"  VOR top100: {shares(rows[:100])}")
    print(f"  ADP top100: {shares(by_adp[:100])}")

    print("\n=== projected mean by position: best, and the starter-count-th best ===")
    for pos in ("QB", "RB", "WR", "TE", "K"):
        ms = sorted((r[1] for r in rows if r[2] == pos), reverse=True)
        if not ms:
            continue
        repl = ctx.replacement_means[pos]
        print(
            f"  {pos:<4} best {ms[0]:6.2f}   10th {ms[9]:6.2f}   30th "
            f"{ms[29] if len(ms) > 29 else float('nan'):6.2f}   replacement {repl:6.2f}"
        )


if __name__ == "__main__":
    main()


def with_alternate_replacement() -> None:
    """Re-rank the same pool using 2024's REALIZED replacement levels
    (measured by scripts/validate_replacement_2024.py, top_k=5) instead of
    the through-2023 fit. This is leakage -- deliberately so. It answers a
    diagnostic question, not a predictive one: is the engine's QB-heavy
    positional tilt explained by 2024's replacement levels differing from
    their long-run means?"""
    ctx = fit_holdout_context()
    realized_2024 = {"QB": 20.11, "RB": 8.83, "WR": 8.88, "TE": 7.67, "K": 7.22, "DST": 7.50}

    def shares(means, n):
        rows = sorted(
            (
                (float(p.distribution.mean) - means[p.position], p.position)
                for p in ctx.pool.values()
            ),
            reverse=True,
        )[:n]
        c: dict[str, int] = {}
        for _, pos in rows:
            c[pos] = c.get(pos, 0) + 1
        return dict(sorted(c.items()))

    by_adp = sorted(ctx.pool.items(), key=lambda kv: ctx.adp_by_player_id.get(kv[0], 1e9))

    def adp_shares(n):
        c: dict[str, int] = {}
        for _, p in by_adp[:n]:
            c[p.position] = c.get(p.position, 0) + 1
        return dict(sorted(c.items()))

    for n in (40, 100):
        print(f"\ntop {n} position shares")
        print(f"  VOR, fitted <=2023 repl : {shares(ctx.replacement_means, n)}")
        print(f"  VOR, realized 2024 repl : {shares(realized_2024, n)}")
        print(f"  ADP                     : {adp_shares(n)}")
