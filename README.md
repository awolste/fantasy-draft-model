# ff-draft-2026

A fantasy football draft optimizer for one specific ESPN league. It does not
optimize for projected points. It optimizes for **championship probability**:
a 10-team snake draft with a 6-team, 2-bye playoff rewards a different draft
strategy than "take the highest point total available," and that is the gap
this project is built to close.

This repository currently covers the data foundation stage only: pulling,
normalizing, ID-matching, and validating every input the later season
simulator and draft optimizer will need. There is no simulator or optimizer
here yet.

## League configuration

| Setting | Value |
|---|---|
| Teams | 10 |
| Scoring | Full PPR, **6-point passing TDs** (ESPN default is 4) |
| Roster | 1 QB / 2 RB / 2 WR / 1 TE / 2 FLEX / 1 K / 1 DST |
| Roster size | 18 (8 bench, 1 IR) — 2026 only, see Known gaps |
| Regular season | 14 weeks |
| Playoffs | 6 teams, 2 byes |
| Draft | Snake, 10 teams, our slot is 8 |

The full set of constants lives in `src/ffdraft/league.py` — treat it as the
single source of truth; nothing downstream should hardcode a league number.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

Fill in `.env`:

```
FF_LEAGUE_ID=<your league id>
ESPN_S2=<espn_s2 cookie value>
ESPN_SWID=<SWID cookie value>
```

`ESPN_S2` and `ESPN_SWID` come from a signed-in ESPN Fantasy session: open
DevTools on espn.com → Application → Cookies → `espn.com`, and copy the
`espn_s2` and `SWID` values. These cookies expire periodically — if ingest
starts failing with a 401, recapture them the same way.

`.env` is gitignored and must never be committed.

## Running

Full pipeline (all sources, then validation):

```bash
.venv/bin/python scripts/ingest_all.py
```

This runs every ingest in dependency order — the ID crosswalk is built
before validation so match rates can be checked — and writes one parquet
file per dataset into the gitignored `data/` directory. It is not
idempotent-safe against partial failure: there is no retry or resume logic,
by design, so a failed run should just be re-run from the top. ESPN raw
JSON responses are cached under `data/raw/espn/` and reused on subsequent
runs; delete that cache to force a refetch.

Tests:

```bash
.venv/bin/pytest -v
```

## Datasets

All datasets are written as parquet files in `data/` (gitignored, never
committed).

| Dataset | Contents |
|---|---|
| `weekly_stats` | Per-player, per-week NFL stat lines (2015–2025) plus a `fantasy_points` column computed under this league's scoring rules |
| `id_crosswalk` | One row per player reconciling `gsis_id` (nflverse), `espn_id`, `sleeper_id`, name, and position |
| `crosswalk_collisions` | Rows discarded by crosswalk dedup because they shared a normalized (name, position) key — kept for audit, not used downstream |
| `league_drafts` | Every pick from every league season: overall pick, round, round pick, team id, ESPN player id |
| `league_managers` | Team → manager (ESPN account SWID) → team name, per season |
| `league_results` | Wins, losses, points for, playoff seed, and final rank per team per season |
| `rankings_2026` | FantasyPros expert consensus rankings (ECR) for the 2026 season |
| `adp_2026` | 2026 average draft position from Fantasy Football Calculator |
| `adp_history` | Average draft position by season, 2018–2024, from Fantasy Football Calculator |

Run `scripts/ingest_all.py` to produce all nine; `validate.py`'s report
prints row counts, season coverage, and ID match rates for each.

## Known gaps

- **Kickers and defenses are not usable yet.** Kickers match the ID
  crosswalk at 0.0% and score 0.0 fantasy points, because `ScoringRules` has
  no kicking terms. DST is not ingested at all. The league starts both a K
  and a DST every week, so this must be resolved before the season
  simulator is built.
- **No source has 2025 ADP.** ADP comes from Fantasy Football Calculator
  (FantasyPros fences its ADP report behind registration). FFC covers
  2018–2024 and 2026 but has no 2025 data at all. ESPN only has genuine ADP
  for 2023, 2024, and 2026. There is no way to get 2025 ADP from any source
  currently wired up.
- **Draft size changed mid-history.** Drafts were 17 rounds (170 picks) from
  2018–2022 and 18 rounds (180 picks) from 2023–2025. `ROSTER_SIZE = 18` in
  `league.py` describes the 2026 season only — do not assume it applies to
  earlier `league_drafts` seasons.
- **Team defenses never match the ID crosswalk**, by construction (nflverse's
  player-id table has no DST rows), and are excluded from match-rate checks
  rather than counted as failures.

## Design docs

- `docs/superpowers/specs/2026-08-02-ff-draft-2026-design.md` — the design
  spec: league configuration, approach, rejected alternatives, validation
  gate.
- `docs/superpowers/plans/README.md` — the stage roadmap and the running
  list of non-obvious constraints discovered along the way.

## Handling ESPN data

ESPN league payloads identify real people: every league member's ESPN
account GUID (which, for the authenticated user, is the `SWID` cookie
itself), their real first and last names, and their team names. None of
that belongs in git.

Raw ESPN JSON responses are cached only in the gitignored `data/raw/`
directory and must never be committed. If you need to derive a test fixture
from an ESPN response, run it through `scripts/sanitize_espn_fixture.py`
first — it replaces every identifier with a deterministic fake while
preserving response shape, pick counts, and internal consistency.
