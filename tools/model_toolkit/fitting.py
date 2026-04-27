"""Peak finding, sequential DE bubble fitting, and classification."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from .data import PriceData
from .support import SupportLine
from .bubble_shape import bubble_shape

# Axis-aware helpers for find_peaks t_center + bubble date conversion.
# Mirrors the sys.path bridge in btc_core/__init__.py so this module also
# works when imported standalone (build_bm_model.py inserts ROOT to sys.path
# before importing model_toolkit).
import sys as _sys
from pathlib import Path as _Path
_BTC_WEB = str(_Path(__file__).resolve().parent.parent.parent / "btc_web")
if _BTC_WEB not in _sys.path:
    _sys.path.insert(0, _BTC_WEB)
from time_basis import year_to_t, t_to_calendar  # noqa: E402
del _sys, _Path, _BTC_WEB


# ── Default configuration ────────────────────────────────────────────────────
# All constants used by the fitting process.  Pass a dict to override any
# subset; missing keys fall back to these defaults.

DEFAULT_CONFIG = {
    # Bubble years to search for peaks
    "BUBBLE_YEARS":             [2011, 2013, 2017, 2021, 2025],
    "BUBBLE_YEAR_WINDOW":       0.75,

    # Fitting window
    "FIT_CONTEXT_YR":           1.0,
    "FIT_RISE_LOOKBACK_YR":     0.75,

    # Plateau constraint
    "PLATEAU_PARALLEL_SUPPORT": True,
    "PLAT_POW_RANGE":           8.0,

    # Differential Evolution
    "DE_MAXITER":               2000,
    "DE_POPSIZE":               18,

    # Classification
    "N_MAJOR":                  5,
    "MAX_MAJOR_BUBBLES":        None,
    "MAX_MINOR_BUBBLES":        8,
}

GENESIS = pd.Timestamp("2009-07-25")


def _cfg(config, key):
    """Look up *key* in *config* (if provided), else fall back to DEFAULT_CONFIG."""
    if config and key in config:
        return config[key]
    return DEFAULT_CONFIG[key]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 equivalent: locate bubble peaks
# ═══════════════════════════════════════════════════════════════════════════════

def find_peaks(log_excess, years, bubble_years, window=0.75):
    """Find the highest log-excess within +/-*window* of Jan 1 of each bubble year.

    Parameters
    ----------
    log_excess : ndarray
        Log-excess values (log_price - log_support) at the *years* time points.
    years : ndarray
        Years since genesis corresponding to *log_excess*.
    bubble_years : list[int]
        Calendar years in which a bubble peak is expected.
    window : float
        Half-width of the search window in years (default 0.75).

    Returns
    -------
    list[dict]
        One dict per located peak with keys: bubble_year, peak_t, region_lo,
        region_hi, raw_K.  Years with no data in the search window are skipped.
    """
    peaks = []
    for yr in bubble_years:
        # Phase 2a: year_to_t is axis-aware (calendar: years; block: blocks).
        t_center = year_to_t(float(yr))
        t_lo = t_center - window
        t_hi = t_center + window

        mask = (years >= t_lo) & (years <= t_hi)
        if not mask.any():
            continue

        local_exc = log_excess[mask]
        local_yrs = years[mask]
        peak_i = int(np.argmax(local_exc))

        peaks.append({
            "bubble_year": yr,
            "peak_t":      local_yrs[peak_i],
            "region_lo":   t_lo,
            "region_hi":   t_hi,
            "raw_K":       local_exc[peak_i],
        })
    return peaks


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 equivalent: fit one bubble via Differential Evolution
# ═══════════════════════════════════════════════════════════════════════════════

def fit_bubble(residual, years, peak, support, config=None, seed_idx=0):
    """Fit bubble_shape() to one located peak using Differential Evolution.

    Parameters
    ----------
    residual : ndarray
        Current residual log-excess (same length as *years*).
    years : ndarray
        Years since genesis (fit data points).
    peak : dict
        Peak dict from find_peaks (keys: peak_t, region_lo, region_hi, raw_K,
        bubble_year).
    support : SupportLine
        Fitted support line (need slope for bubble_shape).
    config : dict or None
        Override any DEFAULT_CONFIG keys.
    seed_idx : int
        Original peak index (discovery order) for DE RNG seeding.

    Returns
    -------
    dict
        Fitted bubble parameters and diagnostics.
    """
    fit_context_yr      = _cfg(config, "FIT_CONTEXT_YR")
    fit_rise_lookback   = _cfg(config, "FIT_RISE_LOOKBACK_YR")
    plateau_parallel    = _cfg(config, "PLATEAU_PARALLEL_SUPPORT")
    plat_pow_range      = _cfg(config, "PLAT_POW_RANGE")
    de_maxiter          = _cfg(config, "DE_MAXITER")
    de_popsize          = _cfg(config, "DE_POPSIZE")
    slope_sup           = support.slope

    region_lo = peak["region_lo"]
    region_hi = peak["region_hi"]
    peak_t    = peak["peak_t"]

    # ── Fitting data window ──────────────────────────────────────────────────
    t_lo = max(years[0],  region_lo - fit_context_yr)
    t_hi = min(years[-1], region_hi + fit_context_yr)
    ctx  = (years >= t_lo) & (years <= t_hi)
    t_ctx = years[ctx]
    exc   = np.maximum(0.0, residual[ctx])

    span = (region_hi - region_lo) + 2 * fit_context_yr

    # ── Optimisation bounds ──────────────────────────────────────────────────
    t_rise_lb = max(years[0], region_lo - fit_rise_lookback)

    bounds_5 = [
        (t_rise_lb, peak_t),     # t_rise
        (0.05, 20.0),            # r
        (0.02, span),            # dur_rise
        (0.0,  span),            # dur_plateau
        (0.05, 20.0),            # d
    ]

    if plateau_parallel:
        # 5-D: plat_pow fixed at 0
        def obj5(p5):
            tr, r_, dr, dp, d_ = p5
            pred = bubble_shape(t_ctx, tr, r_, tr + dr, tr + dr + dp, d_, 0.0,
                                slope_sup=slope_sup)
            return float(np.sum((exc - pred) ** 2))

        res    = differential_evolution(obj5, bounds_5, maxiter=de_maxiter,
                                        popsize=de_popsize, seed=42 + seed_idx,
                                        tol=1e-10, polish=True)
        params = list(res.x) + [0.0]
        cost   = res.fun
    else:
        # 6-D: plat_pow free
        bounds_6 = bounds_5 + [(-plat_pow_range, plat_pow_range)]

        def obj6(p6):
            tr, r_, dr, dp, d_, pp = p6
            pred = bubble_shape(t_ctx, tr, r_, tr + dr, tr + dr + dp, d_, pp,
                                slope_sup=slope_sup)
            return float(np.sum((exc - pred) ** 2))

        res    = differential_evolution(obj6, bounds_6, maxiter=de_maxiter,
                                        popsize=de_popsize, seed=42 + seed_idx,
                                        tol=1e-10, polish=True)
        params = list(res.x)
        cost   = res.fun

    # ── Unpack and derive K values ───────────────────────────────────────────
    tr, r_, dr, dp, d_, pp = params
    tplat = tr + dr
    tdec  = tplat + dp
    K_peak = max(r_ * dr + slope_sup * np.log10(max(tr / tplat, 1e-12)), 0.0)
    K_end  = K_peak + pp * np.log10(max(tdec / tplat, 1e-12)) if dp > 0 else K_peak
    K_bub  = max(K_peak, K_end)
    t_end  = tdec + (max(K_end, 0.0) / d_ if d_ > 0 else 0.0)

    return {
        "t_rise":       tr,       "r":            r_,     "dur_rise":     dr,
        "t_plateau":    tplat,    "dur_plateau":  dp,     "t_decay":      tdec,
        "d":            d_,       "plat_pow":     pp,
        "K":            K_bub,    "K_peak":       K_peak, "K_end":        K_end,
        "t_end":        t_end,    "cost":         cost,
        "t_start":      tr,       # alias used by plot helpers
        "bubble_year":  peak["bubble_year"],
        # Phase 2a: t → calendar date via time_basis (axis-aware).
        # In calendar mode equivalent to the old GENESIS + Timedelta(days=t*365.25).
        # In block mode, t is a block offset; t_to_calendar projects via
        # T_PER_YEAR to a calendar date for chart annotations.
        "date_rise":    pd.Timestamp(t_to_calendar(tr)),
        "date_plat":    pd.Timestamp(t_to_calendar(tplat)),
        "date_decay":   pd.Timestamp(t_to_calendar(tdec)),
        "date_end":     pd.Timestamp(t_to_calendar(t_end)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sequential residual fitting loop
# ═══════════════════════════════════════════════════════════════════════════════

def fit_sequential(price_data, support, config=None):
    """Fit all bubbles sequentially: largest raw peak first, subtract each.

    Parameters
    ----------
    price_data : PriceData
        Loaded price data.
    support : SupportLine
        Fitted support line.
    config : dict or None
        Override any DEFAULT_CONFIG keys.

    Returns
    -------
    list[dict]
        Fitted bubble parameter dicts, sorted chronologically by t_rise.
    """
    bubble_years = _cfg(config, "BUBBLE_YEARS")
    window       = _cfg(config, "BUBBLE_YEAR_WINDOW")
    slope_sup    = support.slope

    years   = price_data.years
    log_exc = support.log_excess

    # Locate peaks
    peaks = find_peaks(log_exc, years, bubble_years, window)
    if not peaks:
        return []

    # Sort by magnitude (largest raw_K first) for fitting order,
    # but remember the original discovery index for DE seeding (E3).
    peaks_by_magnitude = sorted(
        enumerate(peaks),
        key=lambda x: x[1]["raw_K"],
        reverse=True,
    )

    residual = log_exc.copy()
    fitted = []

    for rank, (orig_idx, pk) in enumerate(peaks_by_magnitude):
        bp = fit_bubble(residual, years, pk, support, config,
                        seed_idx=orig_idx)
        fitted.append(bp)

        # Subtract this bubble's contribution from the residual
        contrib = bubble_shape(years, bp["t_rise"], bp["r"],
                               bp["t_plateau"], bp["t_decay"], bp["d"],
                               bp.get("plat_pow", 0.0),
                               slope_sup=slope_sup)
        residual = np.maximum(0.0, residual - contrib)

    # Sort chronologically by rise time
    fitted.sort(key=lambda b: b["t_rise"])
    return fitted


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 equivalent: classify fitted bubbles -> major / minor
# ═══════════════════════════════════════════════════════════════════════════════

def classify(fitted_bubbles, n_major=5, max_major=None, max_minor=8):
    """Split fitted bubbles into major and minor by peak K.

    Parameters
    ----------
    fitted_bubbles : list[dict]
        Output of fit_sequential.
    n_major : int
        Top *n_major* bubbles by K are labelled major (default 5, per E4).
    max_major : int or None
        After classification, keep only the *max_major* highest-K major bubbles.
        None = keep all n_major.
    max_minor : int or None
        Keep only the *max_minor* highest-K minor bubbles.
        None = keep all.

    Returns
    -------
    (list[dict], list[dict])
        (major, minor) — each sorted chronologically by t_rise.
    """
    n_det = len(fitted_bubbles)
    n_maj = min(n_major, n_det)
    by_K  = sorted(range(n_det), key=lambda i: fitted_bubbles[i]["K"], reverse=True)
    maj_set = set(by_K[:n_maj])

    major = sorted([fitted_bubbles[i] for i in maj_set],
                   key=lambda b: b["t_rise"])
    minor = sorted([fitted_bubbles[i] for i in range(n_det) if i not in maj_set],
                   key=lambda b: b["t_rise"])

    # Apply optional caps (keep highest-K within each class, re-sort chronologically)
    if max_major is not None and len(major) > max_major:
        major = sorted(sorted(major, key=lambda b: -b["K"])[:max_major],
                       key=lambda b: b["t_rise"])
    if max_minor is not None and len(minor) > max_minor:
        minor = sorted(sorted(minor, key=lambda b: -b["K"])[:max_minor],
                       key=lambda b: b["t_rise"])

    return major, minor
