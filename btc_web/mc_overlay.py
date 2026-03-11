"""mc_overlay.py — Monte Carlo overlay logic for all chart tabs.

Extracted from figures.py to reduce its size and isolate MC concerns.
Contains: transition matrix cache, cache key functions, fan band trace
builders, and the 4 overlay functions (DCA, withdraw/retire/SC, heatmap).

Import chain: mc_overlay imports from _app_ctx, mc_cache, btc_core, markov.
figures.py imports from mc_overlay (one-directional, no circular dependency).
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

import _app_ctx
from btc_core import ModelData, yr_to_t, fmt_price

from mc_cache import (MC_BINS, MC_SIMS, MC_FREQ, MC_DEFAULT_YEARS,
                      MC_DEFAULT_ENTRY_Q, MC_DEFAULT_START_YR)

FREQ_PPY = _app_ctx.FREQ_PPY
FREQ_STEP_DAYS = _app_ctx.FREQ_STEP_DAYS

# ── Markov imports (conditional) ─────────────────────────────────────────────
try:
    from markov import (build_transition_matrix, monte_carlo_prices,
                        mc_dca, mc_retire, compute_fan_percentiles,
                        depletion_stats, max_bins_for_window)
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False

# ── Pre-computed MC cache (conditional) ──────────────────────────────────────
try:
    from mc_cache import (load_startup_cache as _load_startup_cache,
                          get_cached_paths, get_cached_overlay,
                          snap_to_bin, is_cached_year,
                          FAN_PCTS as _MC_CACHE_FAN_PCTS)
    _load_startup_cache()
    _HAS_MC_CACHE = True
except ImportError:
    _HAS_MC_CACHE = False

_MC_FAN_PCTS = _MC_CACHE_FAN_PCTS if _HAS_MC_CACHE else (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)

# ── MC overlay constants ──────────────────────────────────────────────────────
_PCTILE_MIN = 0.05              # min entry percentile (clip to avoid extreme tails)
_PCTILE_MAX = 0.95              # max entry percentile
_ANNOT_AX = 28                  # annotation arrow x-offset (pixels)


# ══════════════════════════════════════════════════════════════════════════════
# Transition matrix cache (server-side, persisted to disk)
# ══════════════════════════════════════════════════════════════════════════════

_TRANS_MATRIX_CACHE = {}
_TRANS_CACHE_PATH = Path(__file__).parent / ".trans_matrix_cache.pkl"
_TRANS_CACHE_DIRTY = False


def _load_trans_cache_from_disk():
    """Load cached transition matrices from disk if they match current model data."""
    global _TRANS_MATRIX_CACHE
    if not _TRANS_CACHE_PATH.exists():
        return
    try:
        import pickle
        with open(_TRANS_CACHE_PATH, "rb") as f:
            saved = pickle.load(f)
        pkl_path = Path(__file__).parent.parent / "btc_app" / "model_data.pkl"
        if pkl_path.exists():
            pkl_mtime = pkl_path.stat().st_mtime
            if saved.get("_pkl_mtime") != pkl_mtime:
                return  # stale cache
        _TRANS_MATRIX_CACHE = saved.get("matrices", {})
    except Exception:
        pass


def save_trans_cache_to_disk() -> None:
    """Flush transition matrix cache to disk for reuse across restarts."""
    global _TRANS_CACHE_DIRTY
    if not _TRANS_CACHE_DIRTY:
        return
    try:
        import pickle
        pkl_path = Path(__file__).parent.parent / "btc_app" / "model_data.pkl"
        pkl_mtime = pkl_path.stat().st_mtime if pkl_path.exists() else None
        with open(_TRANS_CACHE_PATH, "wb") as f:
            pickle.dump({"matrices": _TRANS_MATRIX_CACHE, "_pkl_mtime": pkl_mtime}, f)
        _TRANS_CACHE_DIRTY = False
    except Exception:
        pass


_load_trans_cache_from_disk()


def _get_transition_matrix(m, n_bins, step_days, mc_window):
    """Get transition matrix from cache or build on the fly."""
    global _TRANS_CACHE_DIRTY
    window_start_yr = None
    window_end_yr   = None
    ws_cal = we_cal = None
    if mc_window:
        ws_cal = int(mc_window[0])
        we_cal = int(mc_window[1])
        window_years = max(1, we_cal - ws_cal)
        n_bins = min(n_bins, max_bins_for_window(window_years, step_days))
        window_start_yr = yr_to_t(ws_cal, m.genesis)
        window_end_yr   = yr_to_t(we_cal, m.genesis)

    cache_key = (n_bins, step_days, ws_cal, we_cal)
    cached = _TRANS_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1], n_bins

    trans, bin_edges, _ = build_transition_matrix(
        m.price_prices, m.price_years, m.qr_fits,
        n_bins=n_bins,
        window_start_yr=window_start_yr,
        window_end_yr=window_end_yr,
        step_days=step_days,
    )
    _TRANS_MATRIX_CACHE[cache_key] = (trans, bin_edges)
    _TRANS_CACHE_DIRTY = True
    return trans, bin_edges, n_bins


# ══════════════════════════════════════════════════════════════════════════════
# Cache key functions
# ══════════════════════════════════════════════════════════════════════════════

def _mc_path_key(p, tab):
    """Build dict of params that determine price paths (expensive MC sampling).

    If these match, cached price_paths can be reused — only the overlay
    (DCA amount / withdrawal / inflation / start_stack) needs recomputing.

    Every param comes exclusively from the MC config panel — no fallbacks
    to main-tab controls.  Key names are uniform across ALL tabs (including HM).
    The HM callback is responsible for mapping its entry_yr/entry_q UI values
    into mc_start_yr/mc_entry_q before passing params here.
    """
    return {
        "tab": tab,
        "mc_bins": int(p.get("mc_bins", MC_BINS)),
        "mc_sims": int(p.get("mc_sims", MC_SIMS)),
        "mc_years": int(p.get("mc_years", MC_DEFAULT_YEARS)),
        "mc_freq": p.get("mc_freq", MC_FREQ),
        "mc_window": p.get("mc_window"),
        "mc_start_yr": int(p.get("mc_start_yr", MC_DEFAULT_START_YR)),
        "mc_entry_q": float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)),
    }


def _mc_overlay_key(p, tab, start_stack):
    """Build dict of overlay-specific params (cheap to recompute from paths)."""
    key = {
        "mc_amount": float(p.get("mc_amount", 100)),
        "start_stack": float(start_stack),
    }
    if tab in ("ret", "sc"):
        key["mc_infl"] = float(p.get("mc_infl", 0))
    return key


# ══════════════════════════════════════════════════════════════════════════════
# Pre-computed cache helpers
# ══════════════════════════════════════════════════════════════════════════════

def try_precomputed_paths(p, mc_years):
    """Look up pre-computed price paths using uniform mc_start_yr/mc_entry_q.

    Returns (n_sims, n_steps) ndarray or None.
    Only returns cached data when entry_q is cache-aligned (within 0.5% of a
    10% bin).  Non-aligned values fall through to live simulation.
    """
    if not _HAS_MC_CACHE:
        return None
    syr = int(p.get("mc_start_yr", MC_DEFAULT_START_YR))
    raw_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    pct_bin = snap_to_bin(raw_pctile)
    if abs(raw_pctile - pct_bin) > 0.005:
        return None
    return get_cached_paths(syr, pct_bin, mc_years)


def try_precomputed_overlay(p, mc_years, wd_amount, inflation, mc_stack):
    """Look up pre-computed withdraw overlay fans.

    Returns (fan_btc, fan_usd) dicts or (None, None).
    Only returns cached data when entry_q is cache-aligned (within 0.5% of a
    10% bin).  Non-aligned values fall through to live simulation.
    """
    if not _HAS_MC_CACHE:
        return None, None
    syr = int(p.get("mc_start_yr", MC_DEFAULT_START_YR))
    raw_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    pct_bin = snap_to_bin(raw_pctile)
    if abs(raw_pctile - pct_bin) > 0.005:
        return None, None
    infl_pct = int(round(inflation * 100))
    return get_cached_overlay(syr, pct_bin, mc_years, int(wd_amount), infl_pct, mc_stack)


# ══════════════════════════════════════════════════════════════════════════════
# Serialization helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mc_fan_to_lists(fan):
    """Convert fan dict {pct: ndarray} to JSON-serializable {str: list}."""
    return {str(k): [round(float(v), 4) for v in arr] for k, arr in fan.items()}


def _mc_fan_from_lists(d):
    """Restore fan dict from JSON-serialized form."""
    return {float(k): np.array(v) for k, v in d.items()}


def _mc_paths_to_lists(paths):
    """Convert price_paths ndarray (n_sims, n_steps) to compact JSON list (float32)."""
    return np.asarray(paths, dtype=np.float32).tolist()


def _mc_paths_from_lists(lst):
    """Restore price_paths ndarray from JSON-serialized list."""
    return np.array(lst, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Trace builders
# ══════════════════════════════════════════════════════════════════════════════

def _mc_build_traces(mc_ts, fan, extra_label="", show_median=True,
                     show_final_values=False, fan_usd=None,
                     hide_5_95_legend=False):
    """Build standard MC fan band traces from precomputed fan percentiles.

    fan_usd: if provided, use these values for legend final-value labels
             (always show USD regardless of display mode).
    hide_5_95_legend: if True, suppress 5-95% band from legend.
    """
    lf = fan_usd if fan_usd is not None else fan  # legend fan

    traces = []
    _MC_BANDS = [
        (0.01, 0.95, "rgba(220,120,0,0.06)", "MC 1\u201395%"),
        (0.05, 0.95, "rgba(220,120,0,0.10)", "MC 5\u201395%"),
        (0.25, 0.75, "rgba(220,120,0,0.18)", "MC 25\u201375%"),
    ]
    for p_lo, p_hi, fill_color, label in _MC_BANDS:
        is_5_95 = (p_lo == 0.05 and p_hi == 0.95)
        show_leg = not (is_5_95 and hide_5_95_legend)
        if show_final_values:
            lo_final = fmt_price(float(lf[p_lo][-1])) if len(lf[p_lo]) > 0 else ""
            hi_final = fmt_price(float(lf[p_hi][-1])) if len(lf[p_hi]) > 0 else ""
            label = f"{label}  ({lo_final} \u2013 {hi_final})"
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[p_hi]),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[p_lo]),
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=fill_color, name=label, showlegend=show_leg, hoverinfo="skip",
        ))
    if show_median:
        med_label = "MC median" + extra_label
        if show_final_values and 0.50 in lf and len(lf[0.50]) > 0:
            med_label += f"  \u2192  {fmt_price(float(lf[0.50][-1]))}"
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[0.50]),
            mode="lines", name=med_label,
            line=dict(color="rgba(220,120,0,0.9)", width=1.5, dash="dot"),
        ))
    return traces


def _mc_depletion_annots(mc_ts, fan, mc_start_yr, mc_years, existing_count=0):
    """Detect depletion on MC fan percentiles and return annotations."""
    annots = []
    mc_col = _app_ctx.BTC_ORANGE
    for pct in [0.01, 0.50, 0.95]:
        vals = fan.get(pct)
        if vals is None:
            continue
        depl_i = next((i for i, v in enumerate(vals) if v <= 0.0001), None)
        if depl_i is not None:
            depl_t = mc_ts[depl_i]
            t0 = mc_ts[0]
            t_span = mc_ts[-1] - t0 if mc_ts[-1] > t0 else 1.0
            depl_yr = int(mc_start_yr + (depl_t - t0) / t_span * mc_years)
            _ay = _app_ctx.ANNOT_STAGGER_Y[(existing_count + len(annots)) % 3]
            annots.append(dict(
                x=depl_t, xref="x",
                y=0, yref="paper",
                ax=_ANNOT_AX, ay=_ay,
                text=f"\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=mc_col,
                font=dict(size=_app_ctx.FONT_LEGEND, color=mc_col),
            ))
    return annots


# ══════════════════════════════════════════════════════════════════════════════
# DCA overlay
# ══════════════════════════════════════════════════════════════════════════════

def _mc_dca_overlay(m, p, ts, t_start, dt, start_stack, disp_mode):
    """Build Monte Carlo fan band traces for DCA overlay.

    Returns (traces, result_dict, cf_usd).
    """
    # Cache lookup order (3-level fallthrough):
    #   1. Client-side cache: full match (path_key + overlay_key) → return directly
    #      Partial match (path_key only) → recompute overlay from cached price paths
    #   2. Pre-computed server cache: npz/shm paths → recompute overlay
    #   3. Live simulation: build transition matrix, run MC, compute overlay
    amount     = float(p.get("mc_amount", 100))
    n_bins     = int(p.get("mc_bins", MC_BINS))
    n_sims     = int(p.get("mc_sims", MC_SIMS))
    mc_window  = p.get("mc_window")
    mc_freq    = p.get("mc_freq", MC_FREQ)
    mc_ppy     = FREQ_PPY.get(mc_freq, 12)
    mc_dt      = 1.0 / mc_ppy
    step_days  = FREQ_STEP_DAYS.get(mc_freq, 30)

    mc_years    = int(p.get("mc_years", MC_DEFAULT_YEARS))
    mc_start_yr = int(p.get("mc_start_yr", MC_DEFAULT_START_YR))
    mc_t_start  = yr_to_t(mc_start_yr, m.genesis)
    mc_t_end    = mc_t_start + mc_years
    mc_ts       = np.arange(mc_t_start, mc_t_end + mc_dt * 0.5, mc_dt)

    # Clip MC fan to DCA year range so off-screen points don't distort y-axis
    dca_t_end = ts[-1] if len(ts) > 0 else mc_t_end
    clip_n = int(np.searchsorted(mc_ts, dca_t_end + mc_dt * 0.5))
    clip_n = max(clip_n, 1)

    def _clip(mc_ts_full, fan_full):
        ct = mc_ts_full[:clip_n]
        cf = {k: v[:clip_n] for k, v in fan_full.items()}
        return ct, cf

    path_key    = _mc_path_key(p, "dca")
    overlay_key = _mc_overlay_key(p, "dca", start_stack)

    # ── Check client-side cache ──────────────────────────────────────────
    cached = p.get("mc_cached")
    if cached and cached.get("path_key") == path_key:
        if cached.get("overlay_key") == overlay_key:
            fan_btc = _mc_fan_from_lists(cached["fan_btc"])
            fan_usd = _mc_fan_from_lists(cached["fan_usd"])
            fan = fan_usd if disp_mode == "usd" else fan_btc
            ct, cf = _clip(mc_ts, fan)
            _, cf_usd = _clip(mc_ts, fan_usd)
            return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), None, cf_usd

        # Path hit, overlay miss — recompute DCA from cached price paths
        price_paths = _mc_paths_from_lists(cached["price_paths"])
        btc_paths, usd_paths = mc_dca(price_paths, amount, start_stack)
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        fan = fan_usd if disp_mode == "usd" else fan_btc

        result = {
            "tab": "dca",
            "path_key": path_key,
            "overlay_key": overlay_key,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ts": cached["ts"],
            "price_paths": cached["price_paths"],
            "fan_btc": _mc_fan_to_lists(fan_btc),
            "fan_usd": _mc_fan_to_lists(fan_usd),
        }
        ct, cf = _clip(mc_ts, fan)
        _, cf_usd = _clip(mc_ts, fan_usd)
        return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), result, cf_usd

    # ── Check pre-computed path cache ────────────────────────────────────
    cached_paths = try_precomputed_paths(p, mc_years)
    if cached_paths is not None:
        btc_paths, usd_paths = mc_dca(cached_paths, amount, start_stack)
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        fan = fan_usd if disp_mode == "usd" else fan_btc
        ct, cf = _clip(mc_ts, fan)
        _, cf_usd = _clip(mc_ts, fan_usd)
        return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), None, cf_usd

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window)
    n_steps = len(mc_ts)

    start_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    start_pctile = max(_PCTILE_MIN, min(start_pctile, _PCTILE_MAX))

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, n_steps, n_sims,
        m.qr_fits, m.genesis, mc_t_start, mc_dt,
    )
    btc_paths, usd_paths = mc_dca(price_paths, amount, start_stack)

    fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
    fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
    fan = fan_usd if disp_mode == "usd" else fan_btc

    result = {
        "tab": "dca",
        "path_key": path_key,
        "overlay_key": overlay_key,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
        "fan_btc": _mc_fan_to_lists(fan_btc),
        "fan_usd": _mc_fan_to_lists(fan_usd),
    }

    ct, cf = _clip(mc_ts, fan)
    _, cf_usd = _clip(mc_ts, fan_usd)
    return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), result, cf_usd


# ══════════════════════════════════════════════════════════════════════════════
# Withdraw overlay (Retire + Supercharger)
# ══════════════════════════════════════════════════════════════════════════════

def _mc_withdraw_overlay(m, p, ts, t_start, t_end, dt,
                          start_stack, disp_mode, tab,
                          existing_annot_count=0,
                          show_final_values=False,
                          hide_5_95_legend=False):
    """Build Monte Carlo fan band traces for withdrawal-based overlays (Retire/SC).

    Returns (traces, annots, result).
    """
    # Cache lookup order (3-level fallthrough):
    #   1. Client-side cache: full match (path_key + overlay_key) → return directly
    #      Partial match (path_key only) → recompute overlay from cached price paths
    #   2. Pre-computed server cache: npz/shm paths → recompute overlay
    #   3. Live simulation: build transition matrix, run MC, compute overlay
    wd_amount  = float(p.get("mc_amount", 5000))
    inflation  = float(p.get("mc_infl", 4)) / 100.0
    n_bins     = int(p.get("mc_bins", MC_BINS))
    n_sims     = int(p.get("mc_sims", MC_SIMS))
    mc_window  = p.get("mc_window")
    mc_freq    = p.get("mc_freq", MC_FREQ)
    mc_ppy     = FREQ_PPY.get(mc_freq, 12)
    mc_dt      = 1.0 / mc_ppy
    step_days  = FREQ_STEP_DAYS.get(mc_freq, 30)

    mc_years    = int(p.get("mc_years", MC_DEFAULT_YEARS))
    mc_start_yr = int(p.get("mc_start_yr", 2031))
    mc_t_start  = max(yr_to_t(mc_start_yr, m.genesis), 1.0)
    mc_t_end    = mc_t_start + mc_years
    mc_ts       = np.arange(mc_t_start, mc_t_end + mc_dt * 0.5, mc_dt)

    mc_stack    = float(p.get("mc_start_stack", 1.0))
    path_key    = _mc_path_key(p, tab)
    overlay_key = _mc_overlay_key(p, tab, mc_stack)
    do_annot    = bool(p.get("annotate"))

    def _depl_extra(dstats):
        if dstats.get("pct_depleted", 0) > 0:
            return f"  ({dstats['pct_depleted']:.0%} depleted)"
        return ""

    def _build_return(fan_btc, fan, extra, result=None, fan_usd=None):
        traces = _mc_build_traces(mc_ts, fan, extra,
                                  show_final_values=show_final_values,
                                  fan_usd=fan_usd,
                                  hide_5_95_legend=hide_5_95_legend)
        annots = _mc_depletion_annots(mc_ts, fan_btc, mc_start_yr, mc_years,
                                       existing_annot_count) if do_annot else []
        return traces, annots, result

    # ── Check client-side cache ──────────────────────────────────────────
    cached = p.get("mc_cached")
    if cached and cached.get("path_key") == path_key:
        if cached.get("overlay_key") == overlay_key:
            fan_btc = _mc_fan_from_lists(cached["fan_btc"])
            fan_usd = _mc_fan_from_lists(cached["fan_usd"])
            fan = fan_usd if disp_mode == "usd" else fan_btc
            return _build_return(fan_btc, fan, _depl_extra(cached.get("depletion", {})),
                                fan_usd=fan_usd)

        price_paths = _mc_paths_from_lists(cached["price_paths"])
        btc_paths, usd_paths, depl_steps = mc_retire(
            price_paths, mc_stack, wd_amount, inflation, mc_dt,
        )
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        dstats = depletion_stats(depl_steps, len(mc_ts), mc_dt, mc_t_start)
        fan = fan_usd if disp_mode == "usd" else fan_btc

        result = {
            "tab": tab,
            "path_key": path_key,
            "overlay_key": overlay_key,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ts": cached["ts"],
            "price_paths": cached["price_paths"],
            "fan_btc": _mc_fan_to_lists(fan_btc),
            "fan_usd": _mc_fan_to_lists(fan_usd),
            "depletion": dstats,
        }
        return _build_return(fan_btc, fan, _depl_extra(dstats), result,
                             fan_usd=fan_usd)

    # ── Check pre-computed cache ─────────────────────────────────────────
    fan_btc, fan_usd = try_precomputed_overlay(p, mc_years, wd_amount,
                                                inflation, mc_stack)
    if fan_btc is not None:
        fan = fan_usd if disp_mode == "usd" else fan_btc
        return _build_return(fan_btc, fan, "", fan_usd=fan_usd)

    cached_paths = try_precomputed_paths(p, mc_years)
    if cached_paths is not None:
        btc_paths, usd_paths, depl_steps = mc_retire(
            cached_paths, mc_stack, wd_amount, inflation, mc_dt,
        )
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        dstats = depletion_stats(depl_steps, len(mc_ts), mc_dt, mc_t_start)
        fan = fan_usd if disp_mode == "usd" else fan_btc
        return _build_return(fan_btc, fan, _depl_extra(dstats), fan_usd=fan_usd)

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window)
    n_steps = len(mc_ts)

    start_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    start_pctile = round(start_pctile * 20) / 20
    start_pctile = max(_PCTILE_MIN, min(start_pctile, _PCTILE_MAX))

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, n_steps, n_sims,
        m.qr_fits, m.genesis, mc_t_start, mc_dt,
    )
    btc_paths, usd_paths, depl_steps = mc_retire(
        price_paths, mc_stack, wd_amount, inflation, mc_dt,
    )

    fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
    fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
    dstats = depletion_stats(depl_steps, n_steps, mc_dt, mc_t_start)
    fan = fan_usd if disp_mode == "usd" else fan_btc

    result = {
        "tab": tab,
        "path_key": path_key,
        "overlay_key": overlay_key,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
        "fan_btc": _mc_fan_to_lists(fan_btc),
        "fan_usd": _mc_fan_to_lists(fan_usd),
        "depletion": dstats,
    }

    return _build_return(fan_btc, fan, _depl_extra(dstats), result,
                         fan_usd=fan_usd)


def _mc_retire_overlay(m, p, ts, t_start, t_end, dt,
                        start_stack, disp_mode, existing_annot_count=0):
    return _mc_withdraw_overlay(m, p, ts, t_start, t_end, dt,
                                 start_stack, disp_mode, "ret", existing_annot_count,
                                 show_final_values=True,
                                 hide_5_95_legend=True)


def _mc_supercharge_overlay(m, p, ts, t_start, t_end, dt,
                             start_stack, disp_mode, existing_annot_count=0):
    return _mc_withdraw_overlay(m, p, ts, t_start, t_end, dt,
                                 start_stack, disp_mode, "sc", existing_annot_count)


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap overlay
# ══════════════════════════════════════════════════════════════════════════════

def _mc_heatmap_overlay(m, p, ep, entry_t, eyrs):
    """Compute MC-derived CAGR percentile rows for the heatmap.

    Returns (mc_cagr, mc_prices, mc_mults, mc_labels, mc_result) or
    (None, None, None, None, None) on error.

    Uses mc_start_yr / mc_entry_q uniformly (same as all other tabs).
    The HM callback is responsible for mapping UI values into these keys.
    """
    n_bins    = int(p.get("mc_bins", MC_BINS))
    n_sims    = int(p.get("mc_sims", MC_SIMS))
    mc_window = p.get("mc_window")
    mc_freq   = p.get("mc_freq", MC_FREQ)
    mc_ppy    = FREQ_PPY.get(mc_freq, 12)
    mc_dt     = 1.0 / mc_ppy
    step_days = FREQ_STEP_DAYS.get(mc_freq, 30)

    mc_years  = int(p.get("mc_years", MC_DEFAULT_YEARS))
    t_start   = entry_t
    mc_t_end  = t_start + mc_years
    mc_ts     = np.arange(t_start, mc_t_end + mc_dt * 0.5, mc_dt)

    path_key    = _mc_path_key(p, "hm")
    overlay_key = {"entry_price": float(ep)}

    def _compute_cagr_rows(price_paths, mc_ts):
        n_pcts = len(_MC_FAN_PCTS)
        n_eyrs = len(eyrs)
        mc_cagr  = np.full((n_pcts, n_eyrs), np.nan)
        mc_prices = np.full((n_pcts, n_eyrs), np.nan)
        mc_mults  = np.full((n_pcts, n_eyrs), np.nan)

        for ci, ey in enumerate(eyrs):
            et = yr_to_t(ey, m.genesis)
            nyr = et - entry_t
            if et > mc_ts[-1]:
                continue
            if nyr <= 0:
                et_eff = entry_t + 0.5
                if et_eff > mc_ts[-1]:
                    continue
                idx = int(np.argmin(np.abs(mc_ts - et_eff)))
                if idx == 0:
                    idx = min(1, len(mc_ts) - 1)
            else:
                idx = int(np.argmin(np.abs(mc_ts - et)))
            prices_at_exit = price_paths[:, idx]
            pct_prices = np.percentile(prices_at_exit, [pf * 100 for pf in _MC_FAN_PCTS])
            for ri, pp in enumerate(pct_prices):
                mc_prices[ri, ci] = pp
                mc_mults[ri, ci] = pp / ep if ep > 0 else 0.0
                if nyr <= 0:
                    mc_cagr[ri, ci] = (pp / ep - 1.0) * 100.0 if ep > 0 else 0.0
                else:
                    mc_cagr[ri, ci] = ((pp / ep) ** (1.0 / nyr) - 1.0) * 100.0 if ep > 0 else 0.0

        return mc_cagr, mc_prices, mc_mults

    mc_labels = [f"MC P{int(pf*100)}%" for pf in _MC_FAN_PCTS]

    # ── Check client-side cache ───────────────────────────────────────────
    cached = p.get("mc_cached")
    if cached and cached.get("path_key") == path_key:
        price_paths = _mc_paths_from_lists(cached["price_paths"])
        cached_ts = np.array(cached["ts"])
        mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(price_paths, cached_ts)
        if cached.get("overlay_key") == overlay_key:
            return mc_cagr, mc_prices_arr, mc_mults, mc_labels, None
        result = {
            "tab": "hm",
            "path_key": path_key,
            "overlay_key": overlay_key,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ts": cached["ts"],
            "price_paths": cached["price_paths"],
        }
        return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result

    # ── Check pre-computed server-side path cache ─────────────────────────
    cached_paths = try_precomputed_paths(p, mc_years)
    if cached_paths is not None:
        mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(cached_paths, mc_ts)
        return mc_cagr, mc_prices_arr, mc_mults, mc_labels, None

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window)
    n_steps = len(mc_ts)

    start_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    start_pctile = max(_PCTILE_MIN, min(start_pctile, _PCTILE_MAX))

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, n_steps, n_sims,
        m.qr_fits, m.genesis, t_start, mc_dt,
    )

    mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(price_paths, mc_ts)

    result = {
        "tab": "hm",
        "path_key": path_key,
        "overlay_key": overlay_key,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
    }

    return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result
