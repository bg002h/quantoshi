"""MC simulation pre-computed cache: generation, loading, and lookup.

Cache structure on disk:
    mc_cache/
        paths_YYYY.npz          — price paths for start year YYYY
        overlays_YYYY.npz       — retire/SC fan percentiles for start year YYYY

Path cache key:  (entry_pct_bin, mc_years)
    entry_pct_bin: 0.1, 0.2, ..., 0.9  (9 values, 10% step)
    mc_years: 10, 20, 30, 40

Overlay cache key: (entry_pct_bin, mc_years, withdrawal, inflation, stack)
    withdrawal: 5000, 7500, 12500, 20000, 32500, 69420
    inflation: 2, 3, 4, 6, 8, 10, 12  (percent, stored as int)
    stack: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0

Fast restart via /dev/shm:
    After first full load from npz (slow, ~7s), the entire cache dict is pickled
    to /dev/shm/quantoshi_mc.pkl (~834 MB). Subsequent restarts load from there
    (~0.7s, 10x faster). The pickle is invalidated when any source npz changes.
"""
import pickle
import time
import numpy as np
from pathlib import Path

# ── Fixed parameters ──────────────────────────────────────────────────────────
MC_BINS = 5
MC_SIMS = 800
MC_FREQ = "Monthly"
MC_PPY = 12
MC_DT = 1.0 / MC_PPY
MC_STEP_DAYS = 30
MC_WINDOW_START = 2010
FAN_PCTS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)

# ── UI defaults (single source of truth for callbacks + figures + mc_overlay) ─
MC_DEFAULT_YEARS = 10
MC_DEFAULT_ENTRY_Q = 50       # percentile, 0–100 scale
MC_DEFAULT_START_YR = 2026    # default start year for MC simulations

# ── Free tier constraints (restricted params for LRU-cached free-tier figures) ─
MC_FREE_SIMS = 100
MC_FREE_START_YRS = [2028, 2031]
MC_FREE_ENTRY_Q = 10          # percentage, 0-100 scale
MC_FREE_YEARS = [10, 20]

# ── Variable parameters ──────────────────────────────────────────────────────
CACHED_START_YRS = [2026, 2028, 2031, 2035, 2040]
ENTRY_PCT_BINS = [round(i / 10, 1) for i in range(1, 10)]  # 0.1 .. 0.9
MC_YEARS_OPTIONS = [10, 20, 30, 40]
WD_AMOUNTS = [5000, 7500, 12500, 20000, 32500, 69420]
INFL_OPTIONS = [2, 3, 4, 6, 8, 10, 12]
STACK_SIZES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

CACHE_DIR = Path(__file__).parent / "mc_cache"
SHM_CACHE_PATH = Path("/dev/shm/quantoshi_mc.pkl")


def _path_key_str(pct_bin, mc_years):
    """Deterministic string key for a path combo."""
    return f"p{pct_bin:.1f}_y{mc_years}"


def _overlay_key_str(pct_bin, mc_years, wd, infl, stack):
    """Deterministic string key for an overlay combo."""
    return f"p{pct_bin:.1f}_y{mc_years}_w{wd}_i{infl}_s{stack}"


# ── Generation ────────────────────────────────────────────────────────────────

def generate_cache(start_yr, m, progress_cb=None):
    """Generate path + overlay cache files for a single start year.

    Args:
        start_yr: Calendar year (e.g. 2031)
        m: ModelData namespace (needs qr_fits, price_prices, price_years, genesis)
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

    genesis = m.genesis
    t_start = yr_to_t(start_yr, genesis)

    # Window end: min(start_yr, current_year) — but for future start years,
    # use all available data up to now
    import pandas as pd
    yr_now = pd.Timestamp.today().year
    window_end = min(start_yr, yr_now)

    ws_yr = yr_to_t(MC_WINDOW_START, genesis)
    we_yr = yr_to_t(window_end, genesis)

    trans, bin_edges, _ = build_transition_matrix(
        m.price_prices, m.price_years, m.qr_fits,
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

            price_paths, _ = monte_carlo_prices(
                trans, bin_edges, pct_bin, n_steps, MC_SIMS,
                m.qr_fits, genesis, t_start, MC_DT,
            )
            key = _path_key_str(pct_bin, mc_years)
            paths_dict[key] = price_paths.astype(np.float32)

            done += 1
            log(f"  Paths {done}/{n_path_combos}: {key} "
                f"shape={price_paths.shape}")

    # Save paths
    path_file = CACHE_DIR / f"paths_{start_yr}.npz"
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
            mc_ts = np.arange(t_start, t_start + mc_years + MC_DT * 0.5, MC_DT)

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
                        # Store as (n_pcts, n_steps) arrays — 5 percentiles
                        btc_arr = np.array([fan_btc[p] for p in FAN_PCTS],
                                           dtype=np.float32)
                        usd_arr = np.array([fan_usd[p] for p in FAN_PCTS],
                                           dtype=np.float32)
                        overlay_dict[f"{okey}_btc"] = btc_arr
                        overlay_dict[f"{okey}_usd"] = usd_arr

                        done += 1
                        if done % 500 == 0 or done == n_overlay_combos:
                            log(f"  Overlays {done}/{n_overlay_combos}")

    overlay_file = CACHE_DIR / f"overlays_{start_yr}.npz"
    np.savez(overlay_file, **overlay_dict)
    log(f"  Saved {overlay_file} ({overlay_file.stat().st_size / 1e6:.1f} MB)")

    return path_file, overlay_file


def generate_all_caches(m, progress_cb=None):
    """Generate caches for all cached start years."""
    for yr in CACHED_START_YRS:
        msg = f"Generating cache for start year {yr}..."
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)
        generate_cache(yr, m, progress_cb)


# ── Loading ───────────────────────────────────────────────────────────────────

# In-memory cache: {start_yr: {"paths": npz_dict, "overlays": npz_dict}}
_CACHE = {}
_FULL_LOADED = False   # True once the full 455 MB cache has been loaded

# Free-tier defaults → entries to pre-load at startup (fast, ~1.5 MB)
# Each tuple: (start_yr, pct_bin, mc_years)
_STARTUP_PATH_ENTRIES = [
    (2028, 0.1, 10),   # free tier: 2028/10yr
    (2028, 0.1, 20),   # free tier: 2028/20yr
    (2031, 0.1, 10),   # free tier: 2031/10yr
    (2031, 0.1, 20),   # free tier: 2031/20yr
]

# Each tuple: (start_yr, pct_bin, mc_years, wd, infl_pct, stack)
_STARTUP_OVERLAY_ENTRIES = [
    (2028, 0.1, 10, 5000, 4, 1.0),   # free tier: ret/SC default
    (2028, 0.1, 20, 5000, 4, 1.0),
    (2031, 0.1, 10, 5000, 4, 1.0),
    (2031, 0.1, 20, 5000, 4, 1.0),
]


def load_startup_cache():
    """Load free-tier path + overlay entries — fast startup (~1.5 MB)."""
    if not CACHE_DIR.exists():
        return
    # ── Paths ────────────────────────────────────────────────────────────
    yr_keys = {}
    for syr, pct, yrs in _STARTUP_PATH_ENTRIES:
        yr_keys.setdefault(syr, []).append(_path_key_str(pct, yrs))

    for yr, keys in yr_keys.items():
        path_file = CACHE_DIR / f"paths_{yr}.npz"
        if not path_file.exists():
            continue
        npz = np.load(path_file)
        paths = {}
        for k in keys:
            if k in npz:
                paths[k] = npz[k]
        npz.close()
        if paths:
            _CACHE.setdefault(yr, {}).setdefault("paths", {}).update(paths)

    # ── Overlays ─────────────────────────────────────────────────────────
    yr_okeys = {}
    for syr, pct, yrs, wd, infl, stack in _STARTUP_OVERLAY_ENTRIES:
        okey = _overlay_key_str(pct, yrs, wd, infl, stack)
        yr_okeys.setdefault(syr, []).append(okey)

    for yr, okeys in yr_okeys.items():
        overlay_file = CACHE_DIR / f"overlays_{yr}.npz"
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
            _CACHE.setdefault(yr, {}).setdefault("overlays", {}).update(overlays)


def _ensure_full_cache():
    """Lazy-load the full cache on first non-startup cache miss."""
    global _FULL_LOADED
    if _FULL_LOADED:
        return
    _FULL_LOADED = True
    load_caches()


def _npz_fingerprint():
    """Return a fingerprint (max mtime + total size) of source npz files."""
    if not CACHE_DIR.exists():
        return 0, 0
    max_mt, total_sz = 0, 0
    for yr in CACHED_START_YRS:
        for kind in ("paths", "overlays"):
            f = CACHE_DIR / f"{kind}_{yr}.npz"
            if f.exists():
                st = f.stat()
                max_mt = max(max_mt, st.st_mtime)
                total_sz += st.st_size
    return max_mt, total_sz


def _try_load_shm():
    """Try loading cache from /dev/shm pickle. Returns True on success."""
    if not SHM_CACHE_PATH.exists():
        return False
    try:
        with open(SHM_CACHE_PATH, "rb") as f:
            saved = pickle.load(f)
        fp_saved = saved.pop("_fingerprint", None)
        fp_now = _npz_fingerprint()
        if fp_saved != fp_now:
            print(f"[MC-CACHE] /dev/shm fingerprint mismatch, reloading from npz")
            return False
        _CACHE.update(saved)
        return True
    except Exception as exc:
        print(f"[MC-CACHE] /dev/shm load failed: {exc}")
        return False


def _save_shm():
    """Persist cache dict to /dev/shm for fast restart."""
    try:
        blob = dict(_CACHE)
        blob["_fingerprint"] = _npz_fingerprint()
        with open(SHM_CACHE_PATH, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz_mb = SHM_CACHE_PATH.stat().st_size / 1e6
        print(f"[MC-CACHE] Saved /dev/shm pickle ({sz_mb:.0f} MB)")
    except Exception as exc:
        print(f"[MC-CACHE] /dev/shm save failed: {exc}")


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
    for yr in CACHED_START_YRS:
        path_file = CACHE_DIR / f"paths_{yr}.npz"
        overlay_file = CACHE_DIR / f"overlays_{yr}.npz"
        if path_file.exists():
            _CACHE.setdefault(yr, {})
            _CACHE[yr]["paths"] = dict(np.load(path_file))
        if overlay_file.exists():
            _CACHE.setdefault(yr, {})
            _CACHE[yr]["overlays"] = dict(np.load(overlay_file))
    elapsed = time.perf_counter() - t0
    print(f"[MC-CACHE] Loaded from npz in {elapsed:.2f}s")

    # Persist to /dev/shm for next restart
    _save_shm()


def get_cached_paths(start_yr, pct_bin, mc_years, max_sims=None):
    """Look up pre-computed price paths. Returns (n_sims, n_steps) array or None.

    If max_sims is set, subsample to at most max_sims paths (numpy view).
    """
    yr_data = _CACHE.get(start_yr)
    key = _path_key_str(pct_bin, mc_years)
    result = None
    # Try startup cache first
    if yr_data and yr_data.get("paths", {}).get(key) is not None:
        result = yr_data["paths"][key]
    # Lazy-load the full cache and retry
    if result is None and not _FULL_LOADED:
        _ensure_full_cache()
        yr_data = _CACHE.get(start_yr)
        if yr_data and yr_data.get("paths", {}).get(key) is not None:
            result = yr_data["paths"][key]
    if result is not None and max_sims and result.shape[0] > max_sims:
        result = result[:max_sims]
    return result


def get_cached_overlay(start_yr, pct_bin, mc_years, wd, infl_pct, stack):
    """Look up pre-computed overlay fans.

    Args:
        infl_pct: inflation as integer percent (e.g. 4 for 4%)

    Returns (fan_btc, fan_usd) dicts or (None, None).
    fan_btc/fan_usd: {pct: np.array} with keys from FAN_PCTS.
    """
    okey = _overlay_key_str(pct_bin, mc_years, wd, int(infl_pct), stack)
    btc_key = f"{okey}_btc"
    usd_key = f"{okey}_usd"

    # Try startup cache first (avoids triggering full load)
    yr_data = _CACHE.get(start_yr)
    if yr_data:
        overlays = yr_data.get("overlays")
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
        yr_data = _CACHE.get(start_yr)
        if yr_data:
            overlays = yr_data.get("overlays")
            if overlays:
                btc_arr = overlays.get(btc_key)
                usd_arr = overlays.get(usd_key)
                if btc_arr is not None and usd_arr is not None:
                    fan_btc = {p: btc_arr[i] for i, p in enumerate(FAN_PCTS)}
                    fan_usd = {p: usd_arr[i] for i, p in enumerate(FAN_PCTS)}
                    return fan_btc, fan_usd

    return None, None


def snap_to_bin(raw_pctile):
    """Round a raw percentile (0–1) to the nearest 10% cache bin."""
    binned = round(raw_pctile * 10) / 10
    return max(0.1, min(binned, 0.9))


def is_cache_aligned_q(entry_q_pct):
    """Check if entry_q (percentage, e.g. 10, 20, 43.5) matches a 10% cache bin.

    Returns True for values within 0.5% of a bin boundary (10, 20, ..., 90).
    """
    raw = float(entry_q_pct) / 100.0
    pct_bin = snap_to_bin(raw)
    return abs(raw - pct_bin) < 0.005


def is_cached_year(yr):
    """Check if a start year has pre-computed cache."""
    return yr in _CACHE and "paths" in _CACHE[yr]
