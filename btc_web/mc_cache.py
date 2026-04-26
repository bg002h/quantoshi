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
_CACHED_MODEL_KEYS = frozenset(["bub", "qr", "pl", "lppl", "exp", "ef"])
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
