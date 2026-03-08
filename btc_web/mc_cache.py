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
"""
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

# ── Variable parameters ──────────────────────────────────────────────────────
CACHED_START_YRS = [2026, 2028, 2031, 2035, 2040]
ENTRY_PCT_BINS = [round(i / 10, 1) for i in range(1, 10)]  # 0.1 .. 0.9
MC_YEARS_OPTIONS = [10, 20, 30, 40]
WD_AMOUNTS = [5000, 7500, 12500, 20000, 32500, 69420]
INFL_OPTIONS = [2, 3, 4, 6, 8, 10, 12]
STACK_SIZES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

CACHE_DIR = Path(__file__).parent / "mc_cache"


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


def load_caches():
    """Load all cached .npz files into RAM at startup."""
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


def get_cached_paths(start_yr, pct_bin, mc_years):
    """Look up pre-computed price paths. Returns (800, n_steps) array or None."""
    yr_data = _CACHE.get(start_yr)
    if yr_data is None:
        return None
    paths = yr_data.get("paths")
    if paths is None:
        return None
    key = _path_key_str(pct_bin, mc_years)
    return paths.get(key)


def get_cached_overlay(start_yr, pct_bin, mc_years, wd, infl_pct, stack):
    """Look up pre-computed overlay fans.

    Args:
        infl_pct: inflation as integer percent (e.g. 4 for 4%)

    Returns (fan_btc, fan_usd) dicts or (None, None).
    fan_btc/fan_usd: {pct: np.array} with keys from FAN_PCTS.
    """
    yr_data = _CACHE.get(start_yr)
    if yr_data is None:
        return None, None
    overlays = yr_data.get("overlays")
    if overlays is None:
        return None, None

    okey = _overlay_key_str(pct_bin, mc_years, wd, int(infl_pct), stack)
    btc_arr = overlays.get(f"{okey}_btc")
    usd_arr = overlays.get(f"{okey}_usd")
    if btc_arr is None or usd_arr is None:
        return None, None

    fan_btc = {p: btc_arr[i] for i, p in enumerate(FAN_PCTS)}
    fan_usd = {p: usd_arr[i] for i, p in enumerate(FAN_PCTS)}
    return fan_btc, fan_usd


def snap_to_bin(raw_pctile):
    """Round a raw percentile (0–1) to the nearest 10% cache bin."""
    binned = round(raw_pctile * 10) / 10
    return max(0.1, min(binned, 0.9))


def is_cached_year(yr):
    """Check if a start year has pre-computed cache."""
    return yr in _CACHE and "paths" in _CACHE[yr]
