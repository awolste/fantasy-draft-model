"""Local parquet storage. One dataset per file, no database needed."""

import hashlib
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


# ---------------------------------------------------------------------------
# Cache staleness: a `load_or_fit_*` fits parameters from input data and
# caches them here, but deciding whether to reuse the cache on file
# existence alone is a silent-corruption trap -- if the input dataset (e.g.
# `weekly_stats`) gets re-ingested with new data while a cache file already
# exists, every subsequent load silently serves parameters fit from the old
# data, with no warning. `fingerprint`/`check_cache_fresh` close that gap.


def _fingerprint_path(name: str) -> Path:
    return DATA_DIR / f"{name}.fingerprint"


def fingerprint(*frames: pl.DataFrame) -> str:
    """A cheap, content-sensitive, cross-process-stable fingerprint of one
    or more input DataFrames, for detecting whether a cached fitted
    artifact still matches the data it would be fit from now.

    Built from each frame's row count, column names/dtypes (so a re-ingest
    that changes a column's type is caught too), and `hash_rows` -- a
    vectorized, order-independent (rows are summed) content hash using a
    fixed seed. Measured at ~4ms for this project's largest input frame
    (`weekly_stats`, 70,775 rows x 29 cols) against ~100-200ms for the
    cheapest full refit that consumes it (`fit_tier_shapes`/
    `fit_rank_curves`) -- i.e. the fingerprint costs a few percent of a
    refit, not a meaningful fraction of one, so computing it on every call
    (including cache hits) is worth it here. `hash_rows` uses a
    project-independent, non-randomized hash (not Python's `id()` or
    per-process-randomized `hash()`), verified stable across separate
    Python processes -- required so a fingerprint written by one process
    (e.g. an ingest script) is recognized by another (e.g. a long-running
    Stage 3 optimizer process) reading the same cache later.

    Multiple frames are combined positionally, so callers should always
    fingerprint their inputs in the same fixed order.
    """
    parts = []
    for df in frames:
        schema_sig = ",".join(f"{c}:{t}" for c, t in zip(df.columns, df.dtypes))
        content_sig = int(df.hash_rows(seed=0).sum())
        parts.append(f"{df.height}:{schema_sig}:{content_sig}")
    return "|".join(parts)


def write_fingerprint(name: str, fp: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _fingerprint_path(name).write_text(fp)


def read_fingerprint(name: str) -> str | None:
    path = _fingerprint_path(name)
    if not path.exists():
        return None
    return path.read_text()


def cache_namespace(base_name: str, *context_frames: pl.DataFrame) -> str:
    """A cache artifact name namespaced by a short hash of `context_frames`
    only -- deliberately not the full set of inputs a `load_or_fit_*` fits
    from. Use this to build the `name` passed to `write`/`read`/`exists`/
    `write_fingerprint`/`check_cache_fresh` when a cached artifact has more
    than one legitimate "shape" of input in normal use.

    Concretely: `models.rank_curve.load_or_fit_rank_curves` is called with
    two genuinely different `rankings` frames in normal Stage 3 use -- the
    default 2026 pool for the live recommender, and a 2024-ADP-anchored
    frame for the historical backtest (`scripts/season_report.py`, and
    Stage 3's Task 6 backtest). `rankings` only affects how deep the
    fitted tail is built, not whether the fit is "wrong" -- both are
    equally valid, current fits. Caching both under one filename means
    whichever context ran most recently silently evicts the other's cache:
    the next call from the *other* context then raises `CacheStaleError`
    even though nothing is actually stale, just a different legitimate
    context -- and Stage 3 alternates between exactly these two contexts
    repeatedly (backtest vs. live), so this is not a one-off collision.

    The fix is to give each context its own artifact, namespaced by a hash
    of only the frame(s) that *define* the context (e.g. `rankings` alone
    here) -- not a hash of every input, which would also change on a
    genuine re-ingest of e.g. `weekly_stats` and defeat the point: a
    context-only key must stay stable across a real re-ingest so that
    re-ingest is still caught as staleness, not silently treated as "a new
    context." Callers therefore still compute the *full* fingerprint (every
    inbound frame, via `fingerprint`) and check it with `check_cache_fresh`
    against the namespaced name this function returns -- `cache_namespace`
    only decides which file to check against, it does not replace the
    staleness check itself.

    Hashing (not embedding the frames' content directly in the filename)
    keeps the extra filename short and filesystem-safe; using `fingerprint`
    's own content hash keeps it stable across processes (same algorithm
    and seed used everywhere else in this module), so identical context
    frames from any process land in the same slot.
    """
    context_fp = fingerprint(*context_frames)
    digest = hashlib.sha256(context_fp.encode()).hexdigest()[:12]
    return f"{base_name}__{digest}"


class CacheStaleError(RuntimeError):
    """Raised by `check_cache_fresh` when a cached fitted artifact's stored
    fingerprint no longer matches the data it would be fit from now -- the
    underlying dataset has almost certainly been re-ingested since the
    cache was written, so the cached fit is stale."""


def check_cache_fresh(name: str, current_fingerprint: str) -> None:
    """Raise `CacheStaleError` if dataset `name` has a stored fingerprint
    that doesn't match `current_fingerprint`.

    Does nothing (does not raise) if there is no stored fingerprint yet --
    e.g. a cache written before this check existed -- so adopting this
    check doesn't force every pre-existing cache to refit on its first
    post-upgrade load; the next write records a fingerprint and all
    subsequent loads are checked normally.

    Raises rather than silently refitting: a `load_or_fit_*` called in a
    loop (e.g. once per candidate pick in a Stage 3 draft rollout) would
    otherwise refit -- anywhere from ~100ms to several seconds depending on
    the module -- on every single iteration with no visible signal, an
    invisible performance cliff. A loud, actionable exception is safer than
    a warning that a log-scrolling caller could miss, and it forces the
    caller to make an explicit choice (pass `force_refit=True`, or delete
    the stale cache) rather than eating an unexplained slowdown.
    """
    cached = read_fingerprint(name)
    if cached is not None and cached != current_fingerprint:
        raise CacheStaleError(
            f"Cached {name!r} at {_path(name)} was fit from different input data "
            f"than what was just provided (fingerprint mismatch) -- the "
            f"underlying dataset has likely been re-ingested since this cache "
            f"was written, so the cached fit is stale. Refit intentionally by "
            f"passing force_refit=True, or delete {_path(name)} and "
            f"{_fingerprint_path(name)} to clear the stale cache."
        )
