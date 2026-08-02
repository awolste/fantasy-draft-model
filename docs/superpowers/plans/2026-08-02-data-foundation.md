# Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull, normalize, ID-match, and validate every data input the draft optimizer needs — 8 seasons of ESPN league history, 10 seasons of NFL weekly stats scored under this league's exact rules, and 2026 rankings/ADP.

**Architecture:** A thin adapter per external source, each writing to a canonical parquet schema in a local `data/` directory. Sources never talk to each other; a single ID crosswalk module reconciles the four different player-ID systems. A scoring module converts raw stat lines into fantasy points under this league's rules, so no external site's scoring assumptions ever enter the pipeline.

**Tech Stack:** Python 3.11+, polars, pandas (nflverse interop only), requests, beautifulsoup4, pytest, pyarrow.

---

## Design Notes

**Why a scoring module instead of using published fantasy points:** this league uses 6-point passing TDs. Essentially every public fantasy point total assumes 4. Computing from raw stat lines is the only way to get QB values right, and QB value is exactly what changes when passing TDs are worth 50% more.

**Why the ID crosswalk gets its own task:** four ID systems are in play — nflverse `gsis_id`, ESPN `playerId`, Sleeper `player_id`, and FantasyPros names. A silent mismatch here does not crash anything; it quietly drops players from the pool or attaches the wrong projections to the wrong person, and every downstream result is wrong in a way that looks plausible. It gets tested explicitly.

**Manager identity across seasons:** ESPN team IDs are not stable across years, but owner `SWID`s are. The opponent model needs per-manager tendencies over 8 seasons, so managers are keyed by SWID, never by team ID or team name.

## File Structure

```
src/ffdraft/
  league.py           league constants: scoring, roster, playoff format  (source of truth)
  scoring.py          raw stat line -> fantasy points (pure functions)
  ids.py              player ID crosswalk across nflverse/ESPN/Sleeper/FantasyPros
  store.py            parquet read/write helpers, canonical data dir
  sources/
    nflverse.py       weekly NFL stat lines
    espn.py           authenticated ESPN v3 client (raw JSON only)
    espn_parse.py     ESPN JSON -> canonical drafts/rosters/results frames
    fantasypros.py    2026 ECR rankings + consensus ADP
  validate.py         cross-source sanity checks, prints a data health report
tests/
  test_scoring.py  test_ids.py  test_nflverse.py  test_espn_parse.py
  test_fantasypros.py  test_store.py  test_validate.py
  fixtures/           saved JSON/HTML samples so tests never hit the network
data/                 parquet outputs (gitignored)
```

---

### Task 1: Project scaffolding and league configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/ffdraft/__init__.py`
- Create: `src/ffdraft/league.py`
- Create: `tests/__init__.py`
- Test: `tests/test_league.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "ffdraft"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "polars>=1.0",
    "pandas>=2.0",
    "pyarrow>=15.0",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create `.gitignore`**

```
data/
.env
__pycache__/
*.pyc
.pytest_cache/
.venv/
*.egg-info/
```

- [ ] **Step 3: Create the virtualenv and install**

```bash
cd /Users/andrewinvsys/Documents/Code/ff-draft-2026
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Expected: installs without error, ending in `Successfully installed ffdraft-0.1.0 ...`

- [ ] **Step 4: Write the failing test**

Create `tests/test_league.py`:

```python
from ffdraft.league import (
    SCORING, STARTERS, FLEX_ELIGIBLE, ROSTER_SIZE, BENCH_SIZE,
    N_TEAMS, DRAFT_SLOT, REGULAR_SEASON_WEEKS,
    PLAYOFF_TEAMS, PLAYOFF_ROUNDS, PLAYOFF_BYES, starting_slots_total,
)


def test_scoring_uses_six_point_passing_tds():
    assert SCORING.pass_td == 6.0


def test_scoring_matches_espn_ppr_defaults():
    assert SCORING.pass_yd == 0.04
    assert SCORING.rush_yd == 0.1
    assert SCORING.rec_yd == 0.1
    assert SCORING.rec == 1.0
    assert SCORING.interception == -2.0
    assert SCORING.fumble_lost == -2.0
    assert SCORING.two_pt == 2.0


def test_starters_sum_to_ten():
    assert starting_slots_total() == 10


def test_roster_shape():
    assert ROSTER_SIZE == 18
    assert BENCH_SIZE == 8
    assert ROSTER_SIZE == starting_slots_total() + BENCH_SIZE


def test_flex_excludes_qb_k_dst():
    assert FLEX_ELIGIBLE == frozenset({"RB", "WR", "TE"})


def test_league_and_playoff_shape():
    assert N_TEAMS == 10
    assert DRAFT_SLOT == 8
    assert REGULAR_SEASON_WEEKS == 14
    assert PLAYOFF_TEAMS == 6
    assert PLAYOFF_ROUNDS == 3
    assert PLAYOFF_BYES == 2
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_league.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.league'`

- [ ] **Step 6: Write the implementation**

Create `src/ffdraft/__init__.py` (empty file) and `tests/__init__.py` (empty file).

Create `src/ffdraft/league.py`:

```python
"""Single source of truth for league settings.

Every scoring, roster, and playoff constant lives here. Nothing downstream
should hardcode a league rule -- if a number about this league appears in
another module, it is a bug.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringRules:
    """Points per unit. ESPN PPR defaults except pass_td, which is 6 here."""

    pass_yd: float = 0.04       # 1 point per 25 yards
    pass_td: float = 6.0        # league-specific: ESPN default is 4
    interception: float = -2.0
    rush_yd: float = 0.1
    rush_td: float = 6.0
    rec: float = 1.0            # full PPR
    rec_yd: float = 0.1
    rec_td: float = 6.0
    fumble_lost: float = -2.0
    two_pt: float = 2.0


SCORING = ScoringRules()

# Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DST = 10
STARTERS: dict[str, int] = {
    "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DST": 1,
}
FLEX_ELIGIBLE = frozenset({"RB", "WR", "TE"})

ROSTER_SIZE = 18
BENCH_SIZE = 8
IR_SLOTS = 1

N_TEAMS = 10
DRAFT_SLOT = 8
DRAFT_ROUNDS = ROSTER_SIZE

REGULAR_SEASON_WEEKS = 14
PLAYOFF_TEAMS = 6
PLAYOFF_ROUNDS = 3
PLAYOFF_BYES = 2

# Seasons of this league's history available on ESPN.
LEAGUE_SEASONS = tuple(range(2018, 2026))
# Seasons of NFL stats used to fit variance and role models.
STATS_SEASONS = tuple(range(2015, 2026))


def starting_slots_total() -> int:
    return sum(STARTERS.values())
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_league.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/
git commit -m "feat: project scaffolding and league configuration"
```

---

### Task 2: Scoring engine

**Files:**
- Create: `src/ffdraft/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py`:

```python
import polars as pl

from ffdraft.league import SCORING
from ffdraft.scoring import score_stat_line, add_fantasy_points

STAT_COLUMNS = [
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "fumbles_lost", "two_pt_conversions",
]


def blank_line(**overrides) -> dict:
    line = {col: 0.0 for col in STAT_COLUMNS}
    line.update(overrides)
    return line


def test_empty_stat_line_scores_zero():
    assert score_stat_line(blank_line(), SCORING) == 0.0


def test_passing_touchdown_is_worth_six():
    assert score_stat_line(blank_line(passing_tds=1), SCORING) == 6.0


def test_passing_yards_quarter_point_per_25():
    assert score_stat_line(blank_line(passing_yards=300), SCORING) == 12.0


def test_full_ppr_reception_worth_one():
    assert score_stat_line(blank_line(receptions=8), SCORING) == 8.0


def test_interceptions_and_fumbles_are_negative():
    line = blank_line(interceptions=2, fumbles_lost=1)
    assert score_stat_line(line, SCORING) == -6.0


def test_realistic_quarterback_line():
    # 320 pass yd, 3 pass TD, 1 INT, 30 rush yd, 1 rush TD
    line = blank_line(
        passing_yards=320, passing_tds=3, interceptions=1,
        rushing_yards=30, rushing_tds=1,
    )
    # 12.8 + 18 - 2 + 3 + 6 = 37.8
    assert score_stat_line(line, SCORING) == 37.8


def test_realistic_receiver_line():
    # 9 rec, 120 yd, 1 TD
    line = blank_line(receptions=9, receiving_yards=120, receiving_tds=1)
    assert score_stat_line(line, SCORING) == 27.0


def test_six_point_passing_td_beats_four_point_assumption():
    """Guards the single most important league-specific rule."""
    line = blank_line(passing_tds=4)
    assert score_stat_line(line, SCORING) == 24.0
    assert score_stat_line(line, SCORING) != 16.0


def test_add_fantasy_points_appends_column_to_frame():
    df = pl.DataFrame([
        blank_line(passing_tds=1) | {"player_id": "A"},
        blank_line(receptions=5, receiving_yards=50) | {"player_id": "B"},
    ])
    out = add_fantasy_points(df, SCORING)
    assert "fantasy_points" in out.columns
    assert out["fantasy_points"].to_list() == [6.0, 10.0]


def test_add_fantasy_points_treats_missing_stats_as_zero():
    """Kickers have no passing columns; nulls must not poison the sum."""
    df = pl.DataFrame({"player_id": ["K1"], "receptions": [None], "rushing_yards": [12.0]})
    out = add_fantasy_points(df, SCORING)
    assert out["fantasy_points"].to_list() == [1.2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.scoring'`

- [ ] **Step 3: Write the implementation**

Create `src/ffdraft/scoring.py`:

```python
"""Convert raw NFL stat lines into fantasy points under this league's rules.

Never consume a precomputed fantasy point total from an external source.
Public totals assume 4-point passing touchdowns; this league uses 6.
"""

from collections.abc import Mapping

import polars as pl

from .league import SCORING, ScoringRules

# Maps a stat column name to the ScoringRules field that prices it.
STAT_TO_RULE: dict[str, str] = {
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "interceptions": "interception",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "receptions": "rec",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",
    "fumbles_lost": "fumble_lost",
    "two_pt_conversions": "two_pt",
}


def score_stat_line(line: Mapping[str, float], rules: ScoringRules = SCORING) -> float:
    """Score a single stat line. Missing or null stats count as zero."""
    total = 0.0
    for stat, rule_name in STAT_TO_RULE.items():
        value = line.get(stat)
        if value is None:
            continue
        total += float(value) * getattr(rules, rule_name)
    return round(total, 2)


def add_fantasy_points(df: pl.DataFrame, rules: ScoringRules = SCORING) -> pl.DataFrame:
    """Append a `fantasy_points` column. Stat columns absent from the frame
    contribute nothing, so the same call works for QBs and kickers alike."""
    terms = [
        (pl.col(stat).fill_null(0.0) * getattr(rules, rule_name))
        for stat, rule_name in STAT_TO_RULE.items()
        if stat in df.columns
    ]
    if not terms:
        return df.with_columns(pl.lit(0.0).alias("fantasy_points"))
    expr = terms[0]
    for term in terms[1:]:
        expr = expr + term
    return df.with_columns(expr.round(2).alias("fantasy_points"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_scoring.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/ffdraft/scoring.py tests/test_scoring.py
git commit -m "feat: scoring engine with 6-point passing TDs"
```

---

### Task 3: Storage helpers

**Files:**
- Create: `src/ffdraft/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
import polars as pl
import pytest

from ffdraft import store


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def test_write_then_read_roundtrips(tmp_data_dir):
    df = pl.DataFrame({"player_id": ["A", "B"], "fantasy_points": [1.5, 2.5]})
    store.write("weekly_stats", df)
    out = store.read("weekly_stats")
    assert out.to_dicts() == df.to_dicts()


def test_write_creates_the_data_directory(tmp_path, monkeypatch):
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(store, "DATA_DIR", nested)
    store.write("thing", pl.DataFrame({"x": [1]}))
    assert (nested / "thing.parquet").exists()


def test_read_missing_dataset_raises_with_helpful_message(tmp_data_dir):
    with pytest.raises(FileNotFoundError, match="weekly_stats"):
        store.read("weekly_stats")


def test_exists_reports_presence(tmp_data_dir):
    assert store.exists("weekly_stats") is False
    store.write("weekly_stats", pl.DataFrame({"x": [1]}))
    assert store.exists("weekly_stats") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.store'`

- [ ] **Step 3: Write the implementation**

Create `src/ffdraft/store.py`:

```python
"""Local parquet storage. One dataset per file, no database needed."""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def write(name: str, df: pl.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(name)
    df.write_parquet(path)
    return path


def read(name: str) -> pl.DataFrame:
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path}. "
            f"Run the ingest step that produces it first."
        )
    return pl.read_parquet(path)


def exists(name: str) -> bool:
    return _path(name).exists()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/ffdraft/store.py tests/test_store.py
git commit -m "feat: parquet storage helpers"
```

---

### Task 4: nflverse weekly stats adapter

**Files:**
- Create: `src/ffdraft/sources/__init__.py`
- Create: `src/ffdraft/sources/nflverse.py`
- Test: `tests/test_nflverse.py`

**Package note:** nflverse's Python bindings exist as both `nfl_data_py` (pandas) and the newer polars-native `nflreadpy`. Step 1 determines which is installed and available; the adapter's public interface is identical either way, and the tests exercise our normalization rather than the upstream package. Do not skip Step 1 — guessing wrong here wastes an hour.

- [ ] **Step 1: Determine the available nflverse package**

```bash
.venv/bin/pip install nflreadpy && .venv/bin/python -c "import nflreadpy; print('nflreadpy', nflreadpy.__version__)"
```

If that fails, run:

```bash
.venv/bin/pip install nfl_data_py && .venv/bin/python -c "import nfl_data_py; print('nfl_data_py ok')"
```

Record which one succeeded — Step 4 has an implementation branch for each. Add the winner to `pyproject.toml` `dependencies`.

- [ ] **Step 2: Save a real fixture**

```bash
.venv/bin/python - <<'PY'
import polars as pl
try:
    import nflreadpy as nfl
    df = nfl.load_player_stats(seasons=[2024])
    df = df.to_polars() if not isinstance(df, pl.DataFrame) else df
except ImportError:
    import nfl_data_py, polars as pl
    df = pl.from_pandas(nfl_data_py.import_weekly_data([2024]))
df.head(200).write_parquet("tests/fixtures/nflverse_2024_sample.parquet")
print(sorted(df.columns))
PY
```

Expected: prints the upstream column list and writes the fixture. Note the exact names for passing/rushing/receiving TDs and fumbles — they vary by package version and Step 4 depends on them.

- [ ] **Step 3: Write the failing test**

Create `tests/test_nflverse.py`:

```python
import polars as pl
import pytest

from ffdraft.sources.nflverse import CANONICAL_COLUMNS, normalize_weekly

FIXTURE = "tests/fixtures/nflverse_2024_sample.parquet"


@pytest.fixture
def raw():
    return pl.read_parquet(FIXTURE)


def test_normalize_produces_exactly_the_canonical_columns(raw):
    out = normalize_weekly(raw)
    assert out.columns == list(CANONICAL_COLUMNS)


def test_normalize_scores_fantasy_points(raw):
    out = normalize_weekly(raw)
    assert out["fantasy_points"].null_count() == 0
    assert out["fantasy_points"].max() > 0


def test_positions_restricted_to_fantasy_relevant(raw):
    out = normalize_weekly(raw)
    assert set(out["position"].unique()) <= {"QB", "RB", "WR", "TE", "K"}


def test_no_null_player_ids(raw):
    out = normalize_weekly(raw)
    assert out["player_id"].null_count() == 0


def test_week_and_season_are_integers(raw):
    out = normalize_weekly(raw)
    assert out.schema["week"] == pl.Int64
    assert out.schema["season"] == pl.Int64


def test_quarterback_points_reflect_six_point_tds(raw):
    """A QB week with passing TDs must score above the 4-point-TD equivalent."""
    out = normalize_weekly(raw)
    qbs = out.filter((pl.col("position") == "QB") & (pl.col("passing_tds") >= 2))
    assert qbs.height > 0
    row = qbs.row(0, named=True)
    four_pt_equivalent = row["fantasy_points"] - 2 * row["passing_tds"]
    assert row["fantasy_points"] > four_pt_equivalent
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_nflverse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.sources'`

- [ ] **Step 5: Write the implementation**

Create `src/ffdraft/sources/__init__.py` (empty file).

Create `src/ffdraft/sources/nflverse.py`:

```python
"""Weekly NFL stat lines from nflverse, normalized to a canonical schema.

Upstream column names differ between nflreadpy and nfl_data_py and between
versions of each. All that variation is absorbed here so nothing downstream
ever sees an upstream column name.
"""

import polars as pl

from ..league import STATS_SEASONS
from ..scoring import add_fantasy_points
from ..store import write

CANONICAL_COLUMNS = (
    "player_id", "player_name", "position", "team", "season", "week",
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "fumbles_lost", "two_pt_conversions",
    "fantasy_points",
)

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K"}

# Upstream name -> canonical name. Extra aliases are harmless; the first one
# present in the frame wins.
ALIASES: dict[str, tuple[str, ...]] = {
    "player_id": ("player_id", "gsis_id"),
    "player_name": ("player_display_name", "player_name"),
    "position": ("position",),
    "team": ("recent_team", "team"),
    "season": ("season",),
    "week": ("week",),
    "passing_yards": ("passing_yards",),
    "passing_tds": ("passing_tds",),
    "interceptions": ("interceptions", "passing_interceptions"),
    "rushing_yards": ("rushing_yards",),
    "rushing_tds": ("rushing_tds",),
    "receptions": ("receptions",),
    "targets": ("targets",),
    "receiving_yards": ("receiving_yards",),
    "receiving_tds": ("receiving_tds",),
}

# Summed rather than aliased, because upstream splits fumbles by play type.
FUMBLE_PARTS = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
TWO_PT_PARTS = (
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
)


def _resolve(df: pl.DataFrame, canonical: str) -> pl.Expr:
    for candidate in ALIASES[canonical]:
        if candidate in df.columns:
            return pl.col(candidate).alias(canonical)
    return pl.lit(None).alias(canonical)


def _sum_parts(df: pl.DataFrame, parts: tuple[str, ...], name: str) -> pl.Expr:
    present = [pl.col(p).fill_null(0.0) for p in parts if p in df.columns]
    if not present:
        return pl.lit(0.0).alias(name)
    expr = present[0]
    for part in present[1:]:
        expr = expr + part
    return expr.alias(name)


def normalize_weekly(raw: pl.DataFrame) -> pl.DataFrame:
    df = raw.select(
        [_resolve(raw, c) for c in ALIASES]
        + [_sum_parts(raw, FUMBLE_PARTS, "fumbles_lost"),
           _sum_parts(raw, TWO_PT_PARTS, "two_pt_conversions")]
    )
    df = df.filter(
        pl.col("position").is_in(list(FANTASY_POSITIONS))
        & pl.col("player_id").is_not_null()
    )
    df = df.with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
    )
    df = add_fantasy_points(df)
    return df.select(list(CANONICAL_COLUMNS))


def load_raw(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Fetch from whichever nflverse package is installed."""
    try:
        import nflreadpy as nfl

        data = nfl.load_player_stats(seasons=list(seasons))
        return data if isinstance(data, pl.DataFrame) else data.to_polars()
    except ImportError:
        import nfl_data_py

        return pl.from_pandas(nfl_data_py.import_weekly_data(list(seasons)))


def ingest(seasons: tuple[int, ...] = STATS_SEASONS) -> pl.DataFrame:
    df = normalize_weekly(load_raw(seasons))
    write("weekly_stats", df)
    return df
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_nflverse.py -v`
Expected: 6 passed. If a test fails on a column name, correct the entry in `ALIASES` using the column list printed in Step 2 — do not change the test.

- [ ] **Step 7: Run the real ingest**

```bash
.venv/bin/python -c "from ffdraft.sources.nflverse import ingest; df = ingest(); print(df.shape); print(df.head())"
```

Expected: roughly 90,000–130,000 rows across 2015–2025, and a populated `fantasy_points` column.

- [ ] **Step 8: Commit**

```bash
git add src/ffdraft/sources/ tests/test_nflverse.py tests/fixtures/ pyproject.toml
git commit -m "feat: nflverse weekly stats adapter"
```

---

### Task 5: Player ID crosswalk

**Files:**
- Create: `src/ffdraft/ids.py`
- Test: `tests/test_ids.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ids.py`:

```python
import polars as pl
import pytest

from ffdraft.ids import CROSSWALK_COLUMNS, match_by_name, normalize_name

CROSSWALK = pl.DataFrame({
    "gsis_id": ["00-0034796", "00-0036322", "00-0033873"],
    "espn_id": [3139477, 4262921, 3116406],
    "sleeper_id": ["4034", "6794", "4035"],
    "name": ["Lamar Jackson", "Ja'Marr Chase", "Christian McCaffrey"],
    "position": ["QB", "WR", "RB"],
})


def test_normalize_name_strips_punctuation_and_case():
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"
    assert normalize_name("A.J. Brown") == "aj brown"
    assert normalize_name("  Amon-Ra  St. Brown ") == "amon ra st brown"


def test_normalize_name_strips_generational_suffixes():
    assert normalize_name("Odell Beckham Jr.") == "odell beckham"
    assert normalize_name("Michael Pittman Jr") == "michael pittman"
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"


def test_match_by_name_resolves_apostrophes():
    names = pl.DataFrame({"name": ["JaMarr Chase"], "position": ["WR"]})
    out = match_by_name(names, CROSSWALK)
    assert out["gsis_id"].to_list() == ["00-0036322"]


def test_match_by_name_requires_position_agreement():
    """Two players can share a name; position disambiguates."""
    names = pl.DataFrame({"name": ["Lamar Jackson"], "position": ["WR"]})
    out = match_by_name(names, CROSSWALK)
    assert out["gsis_id"].to_list() == [None]


def test_match_by_name_keeps_unmatched_rows_with_null_id():
    names = pl.DataFrame({"name": ["Nonexistent Player"], "position": ["RB"]})
    out = match_by_name(names, CROSSWALK)
    assert out.height == 1
    assert out["gsis_id"].to_list() == [None]


def test_crosswalk_columns_are_the_four_id_systems():
    assert set(CROSSWALK_COLUMNS) == {"gsis_id", "espn_id", "sleeper_id", "name", "position"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.ids'`

- [ ] **Step 3: Write the implementation**

Create `src/ffdraft/ids.py`:

```python
"""Reconcile the four player-ID systems in play.

nflverse uses gsis_id, ESPN uses an integer playerId, Sleeper uses its own
string id, and FantasyPros publishes names only. A silent mismatch here does
not raise -- it drops players or misattributes projections, and every
downstream number stays plausible while being wrong. Hence explicit tests
and an unmatched-rate check in validate.py.
"""

import re

import polars as pl

from .store import write

CROSSWALK_COLUMNS = ("gsis_id", "espn_id", "sleeper_id", "name", "position")

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and generational suffixes, squash spaces."""
    cleaned = name.lower().replace("-", " ")
    cleaned = re.sub(r"[^a-z\s]", "", cleaned)
    tokens = [t for t in cleaned.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def _with_key(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("_key")
    )


def match_by_name(df: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Left-join `df` (needs `name` and `position`) onto crosswalk IDs.

    Unmatched rows are kept with null IDs so callers can measure the miss rate
    rather than silently losing players.
    """
    left = _with_key(df)
    right = _with_key(crosswalk).select(
        ["_key", "position", "gsis_id", "espn_id", "sleeper_id"]
    )
    return left.join(right, on=["_key", "position"], how="left").drop("_key")


def load_crosswalk() -> pl.DataFrame:
    """Build the crosswalk from nflverse's ID table plus Sleeper's player dump."""
    try:
        import nflreadpy as nfl

        raw = nfl.load_ff_playerids()
        ids = raw if isinstance(raw, pl.DataFrame) else raw.to_polars()
    except ImportError:
        import nfl_data_py

        ids = pl.from_pandas(nfl_data_py.import_ids())

    return (
        ids.select(
            pl.col("gsis_id"),
            pl.col("espn_id").cast(pl.Int64, strict=False),
            pl.col("sleeper_id").cast(pl.String, strict=False),
            pl.col("name"),
            pl.col("position"),
        )
        .filter(pl.col("name").is_not_null() & pl.col("position").is_not_null())
        .unique(subset=["gsis_id"])
    )


def ingest() -> pl.DataFrame:
    df = load_crosswalk()
    write("id_crosswalk", df)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ids.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the real ingest and check coverage**

```bash
.venv/bin/python -c "
from ffdraft.ids import ingest
df = ingest()
print('rows:', df.height)
print('with espn_id:', df.filter(df['espn_id'].is_not_null()).height)
print('with sleeper_id:', df.filter(df['sleeper_id'].is_not_null()).height)
"
```

Expected: several thousand rows, with a large majority carrying an `espn_id`. If `espn_id` coverage is under 50%, the upstream column name differs — inspect `ids.columns` and correct `load_crosswalk`.

- [ ] **Step 6: Commit**

```bash
git add src/ffdraft/ids.py tests/test_ids.py
git commit -m "feat: player ID crosswalk across nflverse, ESPN, and Sleeper"
```

---

### Task 6: ESPN API client

**Files:**
- Create: `src/ffdraft/sources/espn.py`
- Create: `.env.example`
- Test: `tests/test_espn_client.py`

- [ ] **Step 1: Capture ESPN credentials**

In a browser logged into ESPN Fantasy, open DevTools → Application → Cookies → `espn.com`, and copy the values of `espn_s2` and `SWID`. Create `.env` in the repo root (already gitignored):

```
FF_LEAGUE_ID=<your league id from the ESPN league URL>
ESPN_S2=<espn_s2 cookie value>
ESPN_SWID={<SWID cookie value, braces included>}
```

Also create `.env.example` with the same keys and empty values, and commit that one.

- [ ] **Step 2: Write the failing test**

Create `tests/test_espn_client.py`:

```python
import pytest

from ffdraft.sources.espn import EspnClient, MissingCredentials


def test_missing_credentials_raise_a_clear_error(monkeypatch):
    monkeypatch.delenv("ESPN_S2", raising=False)
    monkeypatch.delenv("ESPN_SWID", raising=False)
    monkeypatch.delenv("FF_LEAGUE_ID", raising=False)
    with pytest.raises(MissingCredentials, match="ESPN_S2"):
        EspnClient.from_env()


def test_historical_url_uses_league_history_endpoint():
    client = EspnClient(league_id=123, espn_s2="s2", swid="{sw}")
    url, params = client.season_request(2019, views=["mDraftDetail"])
    assert "leagueHistory/123" in url
    assert params["seasonId"] == 2019
    assert params["view"] == ["mDraftDetail"]


def test_current_season_url_uses_seasons_endpoint():
    client = EspnClient(league_id=123, espn_s2="s2", swid="{sw}")
    url, params = client.season_request(2026, views=["mTeam"], current_season=2026)
    assert "seasons/2026/segments/0/leagues/123" in url
    assert "seasonId" not in params


def test_cookies_include_both_required_values():
    client = EspnClient(league_id=123, espn_s2="s2", swid="{sw}")
    assert client.cookies == {"espn_s2": "s2", "SWID": "{sw}"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_espn_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.sources.espn'`

- [ ] **Step 4: Write the implementation**

Create `src/ffdraft/sources/espn.py`:

```python
"""Authenticated ESPN Fantasy v3 client.

Returns raw JSON only -- parsing lives in espn_parse.py so that saved JSON
fixtures can be reparsed without network access.

ESPN exposes completed seasons under a different path than the active one:
  completed: /leagueHistory/{league_id}?seasonId={year}
  active:    /seasons/{year}/segments/0/leagues/{league_id}
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "espn"


class MissingCredentials(RuntimeError):
    pass


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class EspnClient:
    league_id: int
    espn_s2: str
    swid: str

    @classmethod
    def from_env(cls) -> "EspnClient":
        _load_dotenv()
        missing = [k for k in ("FF_LEAGUE_ID", "ESPN_S2", "ESPN_SWID") if not os.environ.get(k)]
        if missing:
            raise MissingCredentials(
                f"Missing environment variables: {', '.join(missing)}. "
                f"Copy .env.example to .env and fill in your ESPN cookies."
            )
        return cls(
            league_id=int(os.environ["FF_LEAGUE_ID"]),
            espn_s2=os.environ["ESPN_S2"],
            swid=os.environ["ESPN_SWID"],
        )

    @property
    def cookies(self) -> dict[str, str]:
        return {"espn_s2": self.espn_s2, "SWID": self.swid}

    def season_request(
        self, season: int, views: list[str], current_season: int | None = None
    ) -> tuple[str, dict]:
        if current_season is not None and season == current_season:
            return f"{BASE}/seasons/{season}/segments/0/leagues/{self.league_id}", {"view": views}
        return f"{BASE}/leagueHistory/{self.league_id}", {"seasonId": season, "view": views}

    def fetch_season(self, season: int, views: list[str], current_season: int | None = None) -> dict:
        url, params = self.season_request(season, views, current_season)
        response = requests.get(url, params=params, cookies=self.cookies, timeout=30)
        response.raise_for_status()
        payload = response.json()
        # leagueHistory returns a single-element list; the active season returns an object.
        return payload[0] if isinstance(payload, list) else payload

    def fetch_and_cache(self, season: int, views: list[str], current_season: int | None = None) -> dict:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cached = RAW_DIR / f"{season}_{'_'.join(views)}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        data = self.fetch_season(season, views, current_season)
        cached.write_text(json.dumps(data))
        return data
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_espn_client.py -v`
Expected: 4 passed

- [ ] **Step 6: Verify against the live API and save fixtures**

```bash
.venv/bin/python - <<'PY'
import json, pathlib
from ffdraft.sources.espn import EspnClient
c = EspnClient.from_env()
data = c.fetch_and_cache(2024, ["mDraftDetail", "mTeam", "mSettings"])
print("top-level keys:", sorted(data.keys()))
print("picks:", len(data["draftDetail"]["picks"]))
print("teams:", len(data["teams"]))
pathlib.Path("tests/fixtures").mkdir(parents=True, exist_ok=True)
pathlib.Path("tests/fixtures/espn_2024.json").write_text(json.dumps(data))
PY
```

Expected: `picks: 180` (10 teams × 18 rounds) and `teams: 10`. A 401 means the cookies are stale — recapture them. If `leagueHistory` 404s for a season, that season predates the league's ESPN history; reduce `LEAGUE_SEASONS` in `league.py` accordingly and note it.

- [ ] **Step 7: Commit**

```bash
git add src/ffdraft/sources/espn.py tests/test_espn_client.py tests/fixtures/espn_2024.json .env.example
git commit -m "feat: authenticated ESPN v3 client with response caching"
```

---

### Task 7: Parse ESPN drafts, managers, and results

**Files:**
- Create: `src/ffdraft/sources/espn_parse.py`
- Test: `tests/test_espn_parse.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_espn_parse.py`:

```python
import json

import polars as pl
import pytest

from ffdraft.sources.espn_parse import (
    DRAFT_COLUMNS, MANAGER_COLUMNS, RESULT_COLUMNS,
    parse_draft, parse_managers, parse_results,
)


@pytest.fixture
def payload():
    with open("tests/fixtures/espn_2024.json") as fh:
        return json.load(fh)


def test_parse_draft_returns_canonical_columns(payload):
    out = parse_draft(payload, season=2024)
    assert out.columns == list(DRAFT_COLUMNS)


def test_parse_draft_has_one_row_per_pick(payload):
    out = parse_draft(payload, season=2024)
    assert out.height == 180
    assert out["overall_pick"].min() == 1
    assert out["overall_pick"].max() == 180


def test_parse_draft_round_and_slot_are_consistent(payload):
    """overall_pick must equal (round - 1) * 10 + round_pick in a 10-team league."""
    out = parse_draft(payload, season=2024)
    derived = (pl.col("round") - 1) * 10 + pl.col("round_pick")
    mismatches = out.filter(derived != pl.col("overall_pick"))
    assert mismatches.height == 0


def test_parse_managers_keys_on_swid_not_team_id(payload):
    out = parse_managers(payload, season=2024)
    assert out.columns == list(MANAGER_COLUMNS)
    assert out.height == 10
    assert out["manager_id"].null_count() == 0
    assert out["manager_id"].n_unique() == 10


def test_parse_results_identifies_exactly_one_champion(payload):
    out = parse_results(payload, season=2024)
    assert out.columns == list(RESULT_COLUMNS)
    champions = out.filter(pl.col("final_rank") == 1)
    assert champions.height == 1


def test_parse_results_playoff_seeds_cover_six_teams(payload):
    out = parse_results(payload, season=2024)
    seeded = out.filter(pl.col("playoff_seed") <= 6)
    assert seeded.height == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_espn_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.sources.espn_parse'`

- [ ] **Step 3: Write the implementation**

Create `src/ffdraft/sources/espn_parse.py`:

```python
"""Turn raw ESPN season JSON into canonical frames.

Managers are keyed by owner SWID, never by team id or team name: ESPN team
ids are reassigned between seasons and names change constantly, but the
opponent model needs to follow the same human across eight drafts.
"""

import polars as pl

from ..league import LEAGUE_SEASONS
from ..store import write
from .espn import EspnClient

DRAFT_COLUMNS = ("season", "overall_pick", "round", "round_pick", "team_id", "espn_player_id")
MANAGER_COLUMNS = ("season", "team_id", "manager_id", "team_name")
RESULT_COLUMNS = ("season", "team_id", "wins", "losses", "points_for", "playoff_seed", "final_rank")


def parse_draft(payload: dict, season: int) -> pl.DataFrame:
    picks = payload["draftDetail"]["picks"]
    return pl.DataFrame(
        {
            "season": [season] * len(picks),
            "overall_pick": [p["overallPickNumber"] for p in picks],
            "round": [p["roundId"] for p in picks],
            "round_pick": [p["roundPickNumber"] for p in picks],
            "team_id": [p["teamId"] for p in picks],
            "espn_player_id": [p["playerId"] for p in picks],
        },
        schema={
            "season": pl.Int64, "overall_pick": pl.Int64, "round": pl.Int64,
            "round_pick": pl.Int64, "team_id": pl.Int64, "espn_player_id": pl.Int64,
        },
    ).sort("overall_pick")


def _primary_owner(team: dict) -> str | None:
    owners = team.get("owners") or []
    return owners[0] if owners else None


def _team_name(team: dict) -> str:
    if team.get("name"):
        return team["name"]
    return f"{team.get('location', '')} {team.get('nickname', '')}".strip()


def parse_managers(payload: dict, season: int) -> pl.DataFrame:
    teams = payload["teams"]
    return pl.DataFrame(
        {
            "season": [season] * len(teams),
            "team_id": [t["id"] for t in teams],
            "manager_id": [_primary_owner(t) for t in teams],
            "team_name": [_team_name(t) for t in teams],
        },
        schema={
            "season": pl.Int64, "team_id": pl.Int64,
            "manager_id": pl.String, "team_name": pl.String,
        },
    )


def parse_results(payload: dict, season: int) -> pl.DataFrame:
    teams = payload["teams"]
    return pl.DataFrame(
        {
            "season": [season] * len(teams),
            "team_id": [t["id"] for t in teams],
            "wins": [t.get("record", {}).get("overall", {}).get("wins") for t in teams],
            "losses": [t.get("record", {}).get("overall", {}).get("losses") for t in teams],
            "points_for": [t.get("record", {}).get("overall", {}).get("pointsFor") for t in teams],
            "playoff_seed": [t.get("playoffSeed") for t in teams],
            "final_rank": [t.get("rankCalculatedFinal") for t in teams],
        },
        schema={
            "season": pl.Int64, "team_id": pl.Int64, "wins": pl.Int64, "losses": pl.Int64,
            "points_for": pl.Float64, "playoff_seed": pl.Int64, "final_rank": pl.Int64,
        },
    )


def ingest(seasons: tuple[int, ...] = LEAGUE_SEASONS) -> dict[str, pl.DataFrame]:
    client = EspnClient.from_env()
    drafts, managers, results = [], [], []
    for season in seasons:
        payload = client.fetch_and_cache(season, ["mDraftDetail", "mTeam", "mSettings"])
        drafts.append(parse_draft(payload, season))
        managers.append(parse_managers(payload, season))
        results.append(parse_results(payload, season))

    frames = {
        "league_drafts": pl.concat(drafts),
        "league_managers": pl.concat(managers),
        "league_results": pl.concat(results),
    }
    for name, df in frames.items():
        write(name, df)
    return frames
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_espn_parse.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the real ingest across all 8 seasons**

```bash
.venv/bin/python -c "
from ffdraft.sources.espn_parse import ingest
frames = ingest()
for name, df in frames.items():
    print(name, df.shape)
print(frames['league_managers'].group_by('manager_id').len().sort('len', descending=True))
"
```

Expected: `league_drafts (1440, 6)` for 8 seasons × 180 picks, and a manager table showing which SWIDs appear in all 8 seasons. Managers appearing in fewer seasons are newer members — record which, since the opponent model will have less data on them.

- [ ] **Step 6: Commit**

```bash
git add src/ffdraft/sources/espn_parse.py tests/test_espn_parse.py
git commit -m "feat: parse ESPN drafts, managers, and season results"
```

---

### Task 8: FantasyPros rankings and ADP

**Files:**
- Create: `src/ffdraft/sources/fantasypros.py`
- Test: `tests/test_fantasypros.py`

- [ ] **Step 1: Save a fixture of the live page**

```bash
.venv/bin/python - <<'PY'
import pathlib, requests
url = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
pathlib.Path("tests/fixtures").mkdir(parents=True, exist_ok=True)
pathlib.Path("tests/fixtures/fantasypros_ecr.html").write_text(html)
print("ecrData present:", "ecrData" in html)
print("bytes:", len(html))
PY
```

Expected: `ecrData present: True`. FantasyPros embeds rankings as a JSON blob in a `var ecrData = {...};` script tag rather than in the HTML table, so the parser targets that blob. If it prints `False`, the page structure changed — inspect the saved HTML for the current JSON variable name and adjust `ECR_PATTERN` in Step 3.

- [ ] **Step 2: Write the failing test**

Create `tests/test_fantasypros.py`:

```python
import polars as pl
import pytest

from ffdraft.sources.fantasypros import RANKING_COLUMNS, parse_ecr


@pytest.fixture
def html():
    with open("tests/fixtures/fantasypros_ecr.html") as fh:
        return fh.read()


def test_parse_ecr_returns_canonical_columns(html):
    out = parse_ecr(html)
    assert out.columns == list(RANKING_COLUMNS)


def test_parse_ecr_returns_a_full_player_pool(html):
    out = parse_ecr(html)
    assert out.height > 150


def test_ranks_start_at_one_and_are_unique(html):
    out = parse_ecr(html)
    assert out["rank"].min() == 1
    assert out["rank"].n_unique() == out.height


def test_positions_are_normalized_without_rank_digits(html):
    """FantasyPros writes 'WR1', 'RB12'; we want the bare position."""
    out = parse_ecr(html)
    assert set(out["position"].unique()) <= {"QB", "RB", "WR", "TE", "K", "DST"}


def test_every_row_has_a_name(html):
    out = parse_ecr(html)
    assert out["name"].null_count() == 0
    assert (out["name"].str.len_chars() > 0).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fantasypros.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.sources.fantasypros'`

- [ ] **Step 4: Write the implementation**

Create `src/ffdraft/sources/fantasypros.py`:

```python
"""FantasyPros expert consensus rankings (ECR) and consensus ADP.

Rankings are embedded as a JSON blob in a script tag, not in the HTML table.
Note that FantasyPros PPR rankings assume 4-point passing touchdowns, so
quarterback ranks are systematically low for this league. That correction is
applied downstream in the player model, not here -- this module reports what
the source actually says.
"""

import json
import re

import polars as pl
import requests

from ..store import write

ECR_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
ADP_URL = "https://www.fantasypros.com/nfl/adp/ppr-overall.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

RANKING_COLUMNS = ("rank", "name", "position", "team", "bye", "tier")
ADP_COLUMNS = ("name", "position", "team", "adp")

ECR_PATTERN = re.compile(r"var\s+ecrData\s*=\s*(\{.*?\});", re.DOTALL)
POSITION_PATTERN = re.compile(r"^([A-Z]+)")


def _bare_position(value: str) -> str:
    """'WR12' -> 'WR'. FantasyPros suffixes positional rank onto the position."""
    match = POSITION_PATTERN.match((value or "").upper())
    return match.group(1) if match else ""


def parse_ecr(html: str) -> pl.DataFrame:
    match = ECR_PATTERN.search(html)
    if not match:
        raise ValueError(
            "Could not find `var ecrData` in the FantasyPros page. "
            "The page structure changed; inspect the saved HTML fixture."
        )
    players = json.loads(match.group(1))["players"]
    df = pl.DataFrame(
        {
            "rank": [int(p["rank_ecr"]) for p in players],
            "name": [p["player_name"] for p in players],
            "position": [_bare_position(p.get("player_position_id", "")) for p in players],
            "team": [p.get("player_team_id") for p in players],
            "bye": [p.get("player_bye_week") for p in players],
            "tier": [p.get("tier") for p in players],
        },
        schema={
            "rank": pl.Int64, "name": pl.String, "position": pl.String,
            "team": pl.String, "bye": pl.String, "tier": pl.Int64,
        },
    )
    return df.select(list(RANKING_COLUMNS)).sort("rank")


def parse_adp(html: str) -> pl.DataFrame:
    """ADP lives in a real HTML table at #data, unlike ECR."""
    import bs4

    soup = bs4.BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="data")
    if table is None:
        raise ValueError("Could not find the ADP table (#data) in the FantasyPros page.")

    names, positions, teams, adps = [], [], [], []
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        player_cell = cells[1]
        anchor = player_cell.find("a")
        if anchor is None:
            continue
        names.append(anchor.get_text(strip=True))
        small = player_cell.find("small")
        teams.append(small.get_text(strip=True) if small else None)
        positions.append(_bare_position(cells[2].get_text(strip=True)))
        try:
            adps.append(float(cells[-1].get_text(strip=True)))
        except ValueError:
            adps.append(None)

    return pl.DataFrame(
        {"name": names, "position": positions, "team": teams, "adp": adps},
        schema={"name": pl.String, "position": pl.String, "team": pl.String, "adp": pl.Float64},
    ).select(list(ADP_COLUMNS))


def _get(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def ingest() -> dict[str, pl.DataFrame]:
    frames = {
        "rankings_2026": parse_ecr(_get(ECR_URL)),
        "adp_2026": parse_adp(_get(ADP_URL)),
    }
    for name, df in frames.items():
        write(name, df)
    return frames
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fantasypros.py -v`
Expected: 5 passed

- [ ] **Step 6: Run the real ingest**

```bash
.venv/bin/python -c "
from ffdraft.sources.fantasypros import ingest
frames = ingest()
for name, df in frames.items():
    print(name, df.shape)
    print(df.head(5))
"
```

Expected: 300+ ranked players and a comparable ADP table, with recognizable names at the top.

- [ ] **Step 7: Commit**

```bash
git add src/ffdraft/sources/fantasypros.py tests/test_fantasypros.py tests/fixtures/fantasypros_ecr.html
git commit -m "feat: FantasyPros ECR rankings and consensus ADP ingest"
```

---

### Task 9: Cross-source validation report

**Files:**
- Create: `src/ffdraft/validate.py`
- Test: `tests/test_validate.py`

This is the task that catches the failure mode described at the top of the plan: an ID mismatch that produces plausible-looking but wrong data. It fails loudly rather than warning quietly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate.py`:

```python
import polars as pl
import pytest

from ffdraft.validate import (
    ValidationError, check_draft_completeness, check_id_match_rate,
    check_season_coverage,
)


def test_id_match_rate_passes_when_most_players_matched():
    df = pl.DataFrame({"gsis_id": ["a", "b", "c", None]})
    assert check_id_match_rate(df, "rankings", threshold=0.7) == 0.75


def test_id_match_rate_raises_below_threshold():
    df = pl.DataFrame({"gsis_id": ["a", None, None, None]})
    with pytest.raises(ValidationError, match="rankings"):
        check_id_match_rate(df, "rankings", threshold=0.7)


def test_draft_completeness_passes_for_full_drafts():
    df = pl.DataFrame({
        "season": [2024] * 180,
        "overall_pick": list(range(1, 181)),
    })
    check_draft_completeness(df, expected_picks=180)


def test_draft_completeness_raises_on_missing_picks():
    df = pl.DataFrame({
        "season": [2024] * 179,
        "overall_pick": list(range(1, 180)),
    })
    with pytest.raises(ValidationError, match="2024"):
        check_draft_completeness(df, expected_picks=180)


def test_season_coverage_raises_when_a_season_is_absent():
    df = pl.DataFrame({"season": [2018, 2019]})
    with pytest.raises(ValidationError, match="2020"):
        check_season_coverage(df, "league_drafts", expected=(2018, 2019, 2020))


def test_season_coverage_passes_when_all_present():
    df = pl.DataFrame({"season": [2018, 2019, 2020]})
    check_season_coverage(df, "league_drafts", expected=(2018, 2019, 2020))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffdraft.validate'`

- [ ] **Step 3: Write the implementation**

Create `src/ffdraft/validate.py`:

```python
"""Cross-source sanity checks.

Every failure mode here is silent by nature -- nothing crashes when an ID
match rate quietly drops to 40%, it just produces a smaller player pool and
confidently wrong recommendations. These checks make that loud.
"""

import polars as pl

from .ids import match_by_name
from .league import LEAGUE_SEASONS, N_TEAMS, ROSTER_SIZE
from .store import read


class ValidationError(AssertionError):
    pass


def check_id_match_rate(df: pl.DataFrame, label: str, threshold: float = 0.7) -> float:
    """Fraction of rows carrying a resolved gsis_id."""
    if df.height == 0:
        raise ValidationError(f"{label}: frame is empty")
    matched = df.filter(pl.col("gsis_id").is_not_null()).height
    rate = matched / df.height
    if rate < threshold:
        unmatched = df.filter(pl.col("gsis_id").is_null())
        sample = unmatched["name"].head(10).to_list() if "name" in df.columns else []
        raise ValidationError(
            f"{label}: only {rate:.1%} of {df.height} rows matched an ID "
            f"(threshold {threshold:.0%}). Unmatched sample: {sample}"
        )
    return rate


def check_draft_completeness(df: pl.DataFrame, expected_picks: int) -> None:
    counts = df.group_by("season").len().sort("season")
    for row in counts.iter_rows(named=True):
        if row["len"] != expected_picks:
            raise ValidationError(
                f"Season {row['season']}: found {row['len']} picks, expected {expected_picks}"
            )


def check_season_coverage(df: pl.DataFrame, label: str, expected: tuple[int, ...]) -> None:
    present = set(df["season"].unique().to_list())
    missing = sorted(set(expected) - present)
    if missing:
        raise ValidationError(f"{label}: missing seasons {missing}")


def report() -> None:
    """Run every check and print a health summary. Raises on the first failure."""
    crosswalk = read("id_crosswalk")

    weekly = read("weekly_stats")
    print(f"weekly_stats: {weekly.height:,} rows, seasons "
          f"{weekly['season'].min()}-{weekly['season'].max()}")

    drafts = read("league_drafts")
    check_season_coverage(drafts, "league_drafts", LEAGUE_SEASONS)
    check_draft_completeness(drafts, expected_picks=N_TEAMS * ROSTER_SIZE)
    print(f"league_drafts: {drafts.height:,} picks across {len(LEAGUE_SEASONS)} seasons")

    managers = read("league_managers")
    tenure = managers.group_by("manager_id").len().sort("len", descending=True)
    veterans = tenure.filter(pl.col("len") == len(LEAGUE_SEASONS)).height
    print(f"league_managers: {tenure.height} distinct managers, "
          f"{veterans} present in all {len(LEAGUE_SEASONS)} seasons")

    results = read("league_results")
    champions = results.filter(pl.col("final_rank") == 1)
    check_season_coverage(champions, "champions", LEAGUE_SEASONS)
    print(f"league_results: {champions.height} champions identified")

    rankings = match_by_name(read("rankings_2026"), crosswalk)
    rate = check_id_match_rate(rankings, "rankings_2026", threshold=0.7)
    print(f"rankings_2026: {rankings.height} players, {rate:.1%} ID-matched")

    adp = match_by_name(read("adp_2026"), crosswalk)
    adp_rate = check_id_match_rate(adp, "adp_2026", threshold=0.7)
    print(f"adp_2026: {adp.height} players, {adp_rate:.1%} ID-matched")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    report()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_validate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/ffdraft/validate.py tests/test_validate.py
git commit -m "feat: cross-source validation report"
```

---

### Task 10: End-to-end ingest and health check

**Files:**
- Create: `scripts/ingest_all.py`
- Create: `README.md`

- [ ] **Step 1: Write the ingest script**

Create `scripts/ingest_all.py`:

```python
"""Run every ingest in dependency order, then validate.

Usage: .venv/bin/python scripts/ingest_all.py
"""

from ffdraft import ids, validate
from ffdraft.sources import espn_parse, fantasypros, nflverse


def main() -> None:
    print("== player ID crosswalk ==")
    ids.ingest()

    print("== nflverse weekly stats ==")
    nflverse.ingest()

    print("== ESPN league history ==")
    espn_parse.ingest()

    print("== FantasyPros rankings and ADP ==")
    fantasypros.ingest()

    print("\n== validation ==")
    validate.report()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pipeline**

```bash
.venv/bin/python scripts/ingest_all.py
```

Expected: each stage prints its row counts, and the run ends with `All validation checks passed.`

If `check_id_match_rate` fails, the unmatched sample it prints tells you what to fix. The two common causes are defenses (FantasyPros lists team names like "San Francisco 49ers", which have no `gsis_id` and should be excluded from the check rather than matched) and rookies absent from the crosswalk until it refreshes. Handle defenses by filtering `position != "DST"` before calling `check_id_match_rate` in `validate.report()`.

- [ ] **Step 3: Run the whole test suite**

Run: `.venv/bin/pytest -v`
Expected: all tests pass, roughly 50 total.

- [ ] **Step 4: Write the README**

Create `README.md`:

```markdown
# 2026 Fantasy Football Draft Optimizer

Recommends draft picks by simulated championship probability for a 10-team,
full-PPR, 6-point-passing-TD ESPN league drafting from slot 8.

See `docs/superpowers/specs/2026-08-02-ff-draft-2026-design.md` for the design.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # then fill in FF_LEAGUE_ID, ESPN_S2, ESPN_SWID
```

ESPN cookies come from a logged-in browser session: DevTools → Application →
Cookies → espn.com → `espn_s2` and `SWID`. They expire periodically; a 401
during ingest means recapture them.

## Ingest data

```bash
.venv/bin/python scripts/ingest_all.py
```

Writes parquet datasets to `data/` and prints a health report.

## Test

```bash
.venv/bin/pytest
```

## Datasets

| Dataset | Contents |
|---|---|
| `id_crosswalk` | Player IDs across nflverse, ESPN, Sleeper |
| `weekly_stats` | Weekly stat lines 2015-2025, scored under league rules |
| `league_drafts` | Every pick, 2018-2025 |
| `league_managers` | Team-to-manager mapping keyed by SWID |
| `league_results` | Records, seeds, and final ranks |
| `rankings_2026` | FantasyPros expert consensus rankings |
| `adp_2026` | FantasyPros consensus ADP |
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ README.md
git commit -m "feat: end-to-end ingest script and README"
```

---

## Done when

- `.venv/bin/python scripts/ingest_all.py` completes and prints `All validation checks passed.`
- `.venv/bin/pytest` passes.
- `data/` contains all seven datasets listed in the README.
- You can answer, from `league_drafts` joined to `adp_2026`, which managers historically reach earliest relative to ADP — the first real input to the opponent model in Plan 3.
