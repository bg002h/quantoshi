"""MC simulation pre-computed cache: generation, loading, and lookup.

Cache structure on disk:
    mc_cache/
        paths_YYYY.npz          — price paths for start year YYYY
        overlays_YYYY.npz       — retire/SC fan percentiles for start year YYYY

Path cache key:  (model_key, entry_pct_bin, mc_years)
    model_key: bub, qr, pl, lppl, exp, ef
    entry_pct_bin: 0.01, 0.10, 0.50
    mc_years: 40

Overlay cache key: (model_key, entry_pct_bin, mc_years, withdrawal, inflation, stack)
    withdrawal: 5000, 7500, 12500, 20000, 32500, 69420
    inflation: 2, 3, 4, 6, 8, 10, 12  (percent, stored as int)
    stack: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0

Fast restart via /dev/shm:
    After first full load from npz (slow, ~7s), the entire cache dict is pickled
    to /dev/shm/quantoshi_mc.pkl (~834 MB). Subsequent restarts load from there
    (~0.7s, 10x faster). The pickle is invalidated when any source npz changes.
"""
import re
import time
import numpy as np
from pathlib import Path

import _app_ctx

# ── Fixed parameters ──────────────────────────────────────────────────────────
MC_BINS = 5
MC_SIMS = 200
MC_FREQ = "Monthly"
MC_PPY = 12
MC_DT = 1.0 / MC_PPY
MC_STEP_DAYS = 30
MC_WINDOW_START = 2010
FAN_PCTS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)

# ── UI defaults (single source of truth for callbacks + figures + mc_overlay) ─
MC_DEFAULT_YEARS = 40
MC_DEFAULT_ENTRY_Q = 10       # percentile, 0–100 scale
MC_DEFAULT_START_YR = 2028    # default start year for MC simulations

# ── Free tier = entire cache (all cached combos are free) ─────────────────────
MC_FREE_SIMS = 200

# ── Variable parameters ──────────────────────────────────────────────────────
CACHED_START_YRS = [2028, 2031, 2035]
ENTRY_PCT_BINS = [0.01, 0.10, 0.50]
MC_YEARS_OPTIONS = [40]
WD_AMOUNTS = [5000, 7500, 12500, 20000, 32500, 69420]
INFL_OPTIONS = [2, 3, 4, 6, 8, 10, 12]
STACK_SIZES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

CACHE_DIR = Path(__file__).parent / "mc_cache"

_CACHE_FILE_RE = re.compile(r"^(paths|overlays)_(.+)_(\d{4})\.npz(?:\.bak)?$")


def _parse_cache_filename(name: str) -> tuple[str, str, int] | None:
    """Parse a cache filename into (kind, model_key, year), or None if not a cache file.

    Pure string parser — does not touch the filesystem. The 4-digit-year
    right-anchor disambiguates greedy `(.+)` capture so multi-underscore
    keys like `ecfg_1d_1u` parse correctly.
    """
    m = _CACHE_FILE_RE.match(name)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def _derive_cached_model_keys() -> frozenset[str]:
    """Build _CACHED_MODEL_KEYS from on-disk cache files.

    Globs overlays_*.npz (NOT paths_*) — overlay files are written second
    in generate_cache, so their presence implies both halves exist
    (avoids partial-write UI lie).

    Returns frozenset() if CACHE_DIR doesn't exist (fresh clone).
    """
    if not CACHE_DIR.exists():
        return frozenset()
    keys = set()
    for f in CACHE_DIR.glob("overlays_*.npz"):
        parsed = _parse_cache_filename(f.name)
        if parsed is not None and parsed[0] == "overlays":
            keys.add(parsed[1])
    return frozenset(keys)


_CACHED_MODEL_KEYS = _derive_cached_model_keys()


def _ef_pkl_path() -> Path:
    """Resolve <repo_root>/model_data_ef.pkl. Mirrors app.py:343."""
    return Path(__file__).parent.parent / "model_data_ef.pkl"


# ── Single source of truth for what the next rebuild will cache ──
# Edited by hand. Used as a runtime sanity oracle (no model_data.pkl load).
# Compare _CACHED_MODEL_KEYS (disk-derived; what's actually on disk now).
_INTENDED_KEYS = frozenset({"bub", "qr", "pl", "ecfg_1d_1u", "ef"})


def intended_models(M) -> dict:
    """Return {short_name: model_instance} for the next rebuild target.

    Single source of truth consumed by tools/rebuild_caches.sh.
    Must instantiate exactly _INTENDED_KEYS — enforced by
    test_intended_models_keys_match_intended_set.
    """
    from btc_core import (BubbleModel, PowerLawModel,
                          EmpiricalFloorModel, QuantileRegressionModel,
                          EPPLConfigModel)
    return {
        "bub":         BubbleModel(M),
        "qr":          QuantileRegressionModel(M),
        "pl":          PowerLawModel(M.ols_intercept, M.ols_slope,
                                     M.price_years, M.price_prices,
                                     M.genesis, M.QR_QUANTILES),
        "ecfg_1d_1u":  EPPLConfigModel("ecfg_1d_1u",
                                       M.price_years, M.price_prices,
                                       M.QR_QUANTILES),
        "ef":          EmpiricalFloorModel(str(_ef_pkl_path())),
    }


def stash_stale_files() -> None:
    """Rename cache files for models NOT in _INTENDED_KEYS to *.npz.bak.

    Atomic and reversible — call commit_stale_files() on success or
    restore_stale_files() on interrupt. Idempotent w.r.t. missing CACHE_DIR.
    Logs each rename via print() (consistent with existing build progress).
    """
    if not CACHE_DIR.exists():
        return
    for f in list(CACHE_DIR.glob("*.npz")):
        parsed = _parse_cache_filename(f.name)
        if parsed is None:
            continue
        _kind, model_key, _year = parsed
        if model_key not in _INTENDED_KEYS:
            bak = f.with_suffix(f.suffix + ".bak")
            f.rename(bak)
            print(f"  stash: {f.name} -> {bak.name}")


def commit_stale_files() -> None:
    """Delete all *.npz.bak files. Call only after generate succeeds."""
    if not CACHE_DIR.exists():
        return
    for bak in list(CACHE_DIR.glob("*.npz.bak")):
        bak.unlink()
        print(f"  commit: deleted {bak.name}")


def restore_stale_files() -> None:
    """Rename *.npz.bak back to *.npz. Call on interrupt or error.
    Idempotent — running twice is a no-op."""
    if not CACHE_DIR.exists():
        return
    for bak in list(CACHE_DIR.glob("*.npz.bak")):
        original = bak.with_suffix("")  # strips .bak, leaves .npz
        if original.exists():
            print(f"  restore: skipping {bak.name} (target {original.name} exists)")
            continue
        bak.rename(original)
        print(f"  restore: {bak.name} -> {original.name}")


# Maps user-facing master keys (dropdown values) to their preferred
# cached variant. Each entry is a transition aid OR a post-rebuild target.
# REMOVE the "lppl" entry after rebuild confirms paths_lppl_*.npz are
# purged from prod (it becomes dead noise, would mislead a future editor).
MASTER_TO_CACHED_FALLBACK: dict[str, str] = {
    "lppl": "lppl",            # transition: kept until LPPL purged from disk
    "eppl": "ecfg_1d_1u",      # post-rebuild target
}


def is_master_cached(key: str) -> bool:
    """True if a dropdown value has a usable cached variant on disk.

    True when either the key itself is in _CACHED_MODEL_KEYS (direct
    cache key like 'bub'), or its master-alias maps to one (e.g.
    'eppl' → 'ecfg_1d_1u').

    Used by the MC source dropdown bolding logic in
    layout/mc_controls.py::_mc_model_src_options.
    """
    return key in _CACHED_MODEL_KEYS or \
           MASTER_TO_CACHED_FALLBACK.get(key) in _CACHED_MODEL_KEYS


SHM_CACHE_PATH = Path("/dev/shm/quantoshi_mc.pkl")


def _path_key_str(pct_bin, mc_years):
    """Deterministic string key for a path combo."""
    return f"p{pct_bin:.1f}_y{mc_years}"


def _overlay_key_str(pct_bin, mc_years, wd, infl, stack):
    """Deterministic string key for an overlay combo."""
    return f"p{pct_bin:.1f}_y{mc_years}_w{wd}_i{infl}_s{stack}"


# ── Generation ────────────────────────────────────────────────────────────────

def generate_cache(start_yr, m, model, progress_cb=None):
    """Generate path + overlay cache files for a single (model, start_yr) combo.

    Args:
        start_yr: Calendar year (e.g. 2031)
        m: ModelData namespace (needs price_prices, price_years, genesis)
        model: PriceModel with find_percentile/interp_price methods
        progress_cb: Optional callback(msg) for progress reporting
    """
    from markov import (build_transition_matrix, monte_carlo_prices,
                        mc_retire, compute_fan_percentiles)
    from btc_core import yr_to_t

    def log(msg):
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

    CACHE_DIR.mkdir(exist_ok=True)
    model_key = model.short_name

    genesis = m.genesis
    t_start = yr_to_t(start_yr, genesis)

    import pandas as pd
    yr_now = pd.Timestamp.today().year
    window_end = min(start_yr, yr_now)

    ws_yr = yr_to_t(MC_WINDOW_START, genesis)
    we_yr = yr_to_t(window_end, genesis)

    trans, bin_edges, _ = build_transition_matrix(
        m.price_prices, m.price_years, model,
        n_bins=MC_BINS,
        window_start_yr=ws_yr,
        window_end_yr=we_yr,
        step_days=MC_STEP_DAYS,
    )

    # ── Generate price paths ──────────────────────────────────────────────
    paths_dict = {}
    n_path_combos = len(ENTRY_PCT_BINS) * len(MC_YEARS_OPTIONS)
    done = 0

    for pct_bin in ENTRY_PCT_BINS:
        for mc_years in MC_YEARS_OPTIONS:
            mc_t_end = t_start + mc_years
            mc_ts = np.arange(t_start, mc_t_end + MC_DT * 0.5, MC_DT)
            n_steps = len(mc_ts)

            # Deterministic seed for the prebuilt cache so every rebuild
            # of mc_cache/ produces the same paths for a given (model_key,
            # mc_years, pct_bin) triple — avoids gratuitous cache
            # invalidation when rebuild_caches.sh reruns.
            _cache_seed = abs(hash((model_key, mc_years, float(pct_bin)))) % (2**32)
            price_paths, _ = monte_carlo_prices(
                trans, bin_edges, pct_bin, n_steps, MC_SIMS,
                model, t_start, MC_DT, rng=_cache_seed,
            )
            key = _path_key_str(pct_bin, mc_years)
            paths_dict[key] = price_paths.astype(np.float32)

            done += 1
            log(f"  [{model_key}] Paths {done}/{n_path_combos}: {key} "
                f"shape={price_paths.shape}")

    # Save paths
    path_file = CACHE_DIR / f"paths_{model_key}_{start_yr}.npz"
    np.savez(path_file, **paths_dict)
    log(f"  Saved {path_file} ({path_file.stat().st_size / 1e6:.1f} MB)")

    # ── Generate overlay fans (retire/SC) ─────────────────────────────────
    overlay_dict = {}
    n_overlay_combos = (len(ENTRY_PCT_BINS) * len(MC_YEARS_OPTIONS) *
                        len(WD_AMOUNTS) * len(INFL_OPTIONS) * len(STACK_SIZES))
    done = 0

    for pct_bin in ENTRY_PCT_BINS:
        for mc_years in MC_YEARS_OPTIONS:
            pkey = _path_key_str(pct_bin, mc_years)
            price_paths = paths_dict[pkey]

            for wd in WD_AMOUNTS:
                for infl in INFL_OPTIONS:
                    infl_frac = infl / 100.0
                    for stack in STACK_SIZES:
                        btc_paths, usd_paths, depl_steps = mc_retire(
                            price_paths, stack, float(wd), infl_frac, MC_DT,
                        )
                        fan_btc = compute_fan_percentiles(btc_paths, FAN_PCTS)
                        fan_usd = compute_fan_percentiles(usd_paths, FAN_PCTS)

                        okey = _overlay_key_str(pct_bin, mc_years, wd, infl, stack)
                        btc_arr = np.array([fan_btc[p] for p in FAN_PCTS],
                                           dtype=np.float32)
                        usd_arr = np.array([fan_usd[p] for p in FAN_PCTS],
                                           dtype=np.float32)
                        overlay_dict[f"{okey}_btc"] = btc_arr
                        overlay_dict[f"{okey}_usd"] = usd_arr

                        done += 1
                        if done % 500 == 0 or done == n_overlay_combos:
                            log(f"  [{model_key}] Overlays {done}/{n_overlay_combos}")

    overlay_file = CACHE_DIR / f"overlays_{model_key}_{start_yr}.npz"
    np.savez(overlay_file, **overlay_dict)
    log(f"  Saved {overlay_file} ({overlay_file.stat().st_size / 1e6:.1f} MB)")

    return path_file, overlay_file


def generate_all_caches(m, models, progress_cb=None):
    """Generate caches for all models × start years.

    Args:
        m: ModelData
        models: dict of {key: model} (quantized models only)
        progress_cb: Optional callback(msg)
    """
    for model_key, model in models.items():
        if not model.quantized:
            continue
        for yr in CACHED_START_YRS:
            msg = f"Generating cache for {model_key}/{yr}..."
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)
            generate_cache(yr, m, model, progress_cb)


# Module-level worker function (must be picklable for ProcessPoolExecutor).
def _generate_one_combo(args):
    """Worker: instantiate one model and generate its cache for one start_yr.

    Each worker gets a fresh M deserialization and reinstantiates only the
    needed model — avoids passing pre-built model instances (some hold
    Cython references that may not pickle cleanly).
    """
    model_key, start_yr, M = args
    models = intended_models(M)
    model = models[model_key]
    generate_cache(start_yr, M, model)
    return f"{model_key}/{start_yr}"


def generate_all_caches_parallel(m, n_workers=4, progress_cb=None):
    """Parallel cache generation: each (model_key, start_yr) runs in its own process.

    Uses concurrent.futures.ProcessPoolExecutor — true parallelism, not
    thread-based (avoids GIL contention since the inner Markov compute is
    NumPy/Cython-heavy).

    Args:
        m: ModelData (must be picklable — it loads from model_data.pkl).
        n_workers: Number of worker processes. Defaults to 4. Cap to
            min(n_workers, total_tasks) to avoid idle workers.
        progress_cb: Optional callback(msg) for progress reporting.

    Worker tasks: every (model_key in _INTENDED_KEYS) × (start_yr in
    CACHED_START_YRS) combo. With current sets that's 5 × 3 = 15 tasks.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    keys = sorted(_INTENDED_KEYS)
    tasks = [(k, y, m) for k in keys for y in CACHED_START_YRS]
    total = len(tasks)
    n_workers = min(n_workers, total) if total else 1

    def log(msg):
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

    log(f"▶ Parallel cache generation: {total} tasks, {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_generate_one_combo, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            label = fut.result()  # raises on worker error → propagates to caller
            done += 1
            log(f"  [{done}/{total}] done: {label}")


# ── Loading ───────────────────────────────────────────────────────────────────

# In-memory cache: {(model_key, start_yr): {"paths": npz_dict, "overlays": npz_dict}}
_CACHE = {}
_FULL_LOADED = False   # True once the full cache has been loaded

# Free-tier defaults → entries to pre-load at startup
# Each tuple: (model_key, start_yr, pct_bin, mc_years)
_STARTUP_PATH_ENTRIES = [
    (mdl, 2028, 0.10, 40) for mdl in _CACHED_MODEL_KEYS
]

# Each tuple: (model_key, start_yr, pct_bin, mc_years, wd, infl_pct, stack)
_STARTUP_OVERLAY_ENTRIES = [
    (mdl, 2028, 0.10, 40, 5000, 4, 1.0) for mdl in _CACHED_MODEL_KEYS
]


def load_startup_cache():
    """Load free-tier path + overlay entries — fast startup."""
    if not CACHE_DIR.exists():
        return
    # ── Paths ────────────────────────────────────────────────────────────
    file_keys = {}  # {(model, yr): [path_key_str, ...]}
    for mdl, syr, pct, yrs in _STARTUP_PATH_ENTRIES:
        file_keys.setdefault((mdl, syr), []).append(_path_key_str(pct, yrs))

    for (mdl, yr), keys in file_keys.items():
        path_file = CACHE_DIR / f"paths_{mdl}_{yr}.npz"
        if not path_file.exists():
            continue
        npz = np.load(path_file)
        paths = {}
        for k in keys:
            if k in npz:
                paths[k] = npz[k]
        npz.close()
        if paths:
            _CACHE.setdefault((mdl, yr), {}).setdefault("paths", {}).update(paths)

    # ── Overlays ─────────────────────────────────────────────────────────
    file_okeys = {}  # {(model, yr): [overlay_key_str, ...]}
    for mdl, syr, pct, yrs, wd, infl, stack in _STARTUP_OVERLAY_ENTRIES:
        okey = _overlay_key_str(pct, yrs, wd, infl, stack)
        file_okeys.setdefault((mdl, syr), []).append(okey)

    for (mdl, yr), okeys in file_okeys.items():
        overlay_file = CACHE_DIR / f"overlays_{mdl}_{yr}.npz"
        if not overlay_file.exists():
            continue
        npz = np.load(overlay_file)
        overlays = {}
        for okey in okeys:
            btc_k = f"{okey}_btc"
            usd_k = f"{okey}_usd"
            if btc_k in npz and usd_k in npz:
                overlays[btc_k] = npz[btc_k]
                overlays[usd_k] = npz[usd_k]
        npz.close()
        if overlays:
            _CACHE.setdefault((mdl, yr), {}).setdefault("overlays", {}).update(overlays)


def _ensure_full_cache():
    """Lazy-load the full cache on first non-startup cache miss."""
    global _FULL_LOADED
    if _FULL_LOADED:
        return
    _FULL_LOADED = True
    load_caches()


from shm_helpers import npz_fingerprint, try_load_shm, save_shm

def _try_load_shm():
    return try_load_shm(SHM_CACHE_PATH, _CACHE, CACHE_DIR, label="MC-CACHE")

def _save_shm():
    save_shm(SHM_CACHE_PATH, _CACHE, CACHE_DIR, label="MC-CACHE")


def load_caches():
    """Load all cached data into RAM.

    Tries /dev/shm pickle first (~0.7s), falls back to npz parsing (~7s).
    After npz load, saves a pickle to /dev/shm for next restart.
    """
    # Fast path: /dev/shm pickle
    t0 = time.perf_counter()
    if _try_load_shm():
        elapsed = time.perf_counter() - t0
        print(f"[MC-CACHE] Loaded from /dev/shm in {elapsed:.2f}s")
        return

    # Slow path: parse npz files from disk
    if not CACHE_DIR.exists():
        return
    for mdl in _CACHED_MODEL_KEYS:
        for yr in CACHED_START_YRS:
            path_file = CACHE_DIR / f"paths_{mdl}_{yr}.npz"
            overlay_file = CACHE_DIR / f"overlays_{mdl}_{yr}.npz"
            cache_key = (mdl, yr)
            if path_file.exists():
                _CACHE.setdefault(cache_key, {})
                _CACHE[cache_key]["paths"] = dict(np.load(path_file))
            if overlay_file.exists():
                _CACHE.setdefault(cache_key, {})
                _CACHE[cache_key]["overlays"] = dict(np.load(overlay_file))
    elapsed = time.perf_counter() - t0
    print(f"[MC-CACHE] Loaded from npz in {elapsed:.2f}s")

    # Persist to /dev/shm for next restart
    _save_shm()


def get_cached_paths(model_key, start_yr, pct_bin, mc_years, max_sims=None):
    """Look up pre-computed price paths. Returns (n_sims, n_steps) array or None.

    If max_sims is set, subsample to at most max_sims paths (numpy view).
    """
    cache_key = (model_key, start_yr)
    data = _CACHE.get(cache_key)
    key = _path_key_str(pct_bin, mc_years)
    result = None
    # Try startup cache first
    if data and data.get("paths", {}).get(key) is not None:
        result = data["paths"][key]
    # Lazy-load the full cache and retry
    if result is None and not _FULL_LOADED:
        _ensure_full_cache()
        data = _CACHE.get(cache_key)
        if data and data.get("paths", {}).get(key) is not None:
            result = data["paths"][key]
    if result is not None and max_sims:
        if result.shape[0] < max_sims:
            return None  # not enough sims — force live computation
        if result.shape[0] > max_sims:
            result = result[:max_sims]
    return result


def get_cached_overlay(model_key, start_yr, pct_bin, mc_years, wd, infl_pct, stack):
    """Look up pre-computed overlay fans.

    Args:
        model_key: model short name (e.g. "bub", "pl")
        infl_pct: inflation as integer percent (e.g. 4 for 4%)

    Returns (fan_btc, fan_usd) dicts or (None, None).
    fan_btc/fan_usd: {pct: np.array} with keys from FAN_PCTS.
    """
    okey = _overlay_key_str(pct_bin, mc_years, wd, int(infl_pct), stack)
    btc_key = f"{okey}_btc"
    usd_key = f"{okey}_usd"

    cache_key = (model_key, start_yr)

    # Try startup cache first (avoids triggering full load)
    data = _CACHE.get(cache_key)
    if data:
        overlays = data.get("overlays")
        if overlays:
            btc_arr = overlays.get(btc_key)
            usd_arr = overlays.get(usd_key)
            if btc_arr is not None and usd_arr is not None:
                fan_btc = {p: btc_arr[i] for i, p in enumerate(FAN_PCTS)}
                fan_usd = {p: usd_arr[i] for i, p in enumerate(FAN_PCTS)}
                return fan_btc, fan_usd

    # Lazy-load the full cache and retry
    if not _FULL_LOADED:
        _ensure_full_cache()
        data = _CACHE.get(cache_key)
        if data:
            overlays = data.get("overlays")
            if overlays:
                btc_arr = overlays.get(btc_key)
                usd_arr = overlays.get(usd_key)
                if btc_arr is not None and usd_arr is not None:
                    fan_btc = {p: btc_arr[i] for i, p in enumerate(FAN_PCTS)}
                    fan_usd = {p: usd_arr[i] for i, p in enumerate(FAN_PCTS)}
                    return fan_btc, fan_usd

    return None, None


def snap_to_bin(raw_pctile):
    """Snap a raw percentile (0–1) to the nearest cached entry bin."""
    return min(ENTRY_PCT_BINS, key=lambda b: abs(raw_pctile - b))


def is_cache_aligned_q(entry_q_pct):
    """Check if entry_q (percentage, e.g. 1, 10, 50) matches a cached bin.

    Returns True for values within 0.5% of a bin boundary.
    """
    raw = float(entry_q_pct) / 100.0
    pct_bin = snap_to_bin(raw)
    return abs(raw - pct_bin) < 0.005


def is_cached(model_key, start_yr, entry_q, mc_years):
    """Check if a specific (model, start_yr, entry_q, mc_years) combo is pre-computed."""
    if model_key not in _CACHED_MODEL_KEYS:
        return False
    if start_yr not in CACHED_START_YRS:
        return False
    if mc_years not in MC_YEARS_OPTIONS:
        return False
    return is_cache_aligned_q(entry_q)


# Backward compat alias
def is_cached_year(yr):
    """Check if a start year has any pre-computed cache."""
    return yr in CACHED_START_YRS
