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
_CACHE_Q_TOLERANCE = 0.005      # max quantile distance for cache bin alignment

def _mc_setup_vars(p):
    """Extract common MC simulation variables from params dict.

    Returns (n_bins, n_sims, mc_window, mc_freq, mc_ppy, mc_dt,
    step_days, mc_years).
    """
    n_bins    = int(p.get("mc_bins", MC_BINS))
    n_sims    = int(p.get("mc_sims", MC_SIMS))
    mc_window = p.get("mc_window")
    mc_freq   = p.get("mc_freq", MC_FREQ)
    mc_ppy    = FREQ_PPY.get(mc_freq, 12)   # periods per year
    mc_dt     = 1.0 / mc_ppy                # time step in years
    step_days = FREQ_STEP_DAYS.get(mc_freq, 30)
    mc_years  = int(p.get("mc_years", MC_DEFAULT_YEARS))
    return n_bins, n_sims, mc_window, mc_freq, mc_ppy, mc_dt, step_days, mc_years


# ── Bin regime labels (5 bins only; 6+ use percentile ranges) ────────────────
_BIN_NAMES_5 = ("Bargain", "Cheap", "Fair", "Pricey", "Bubble")


def bin_regime_labels(n_bins: int) -> list[str]:
    """Human-readable labels for each percentile bin.

    For 5 bins: named (Bargain, Cheap, Fair, Pricey, Bubble).
    For other counts: percentile range strings like '0–20%'.
    """
    if n_bins == 5:
        return [f"{_BIN_NAMES_5[i]} ({i*20}\u2013{(i+1)*20}%)"
                for i in range(5)]
    width = 100 / n_bins
    return [f"{round(i*width)}\u2013{round((i+1)*width)}%"
            for i in range(n_bins)]


# ── Bin regime filter (applied after building transition matrix) ─────────────

def _apply_bin_mask(trans, blocked_bins):
    """Zero out transition columns for blocked bins, re-normalize rows.

    Parameters
    ----------
    trans : ndarray (n_bins, n_bins) — row-stochastic transition matrix
    blocked_bins : set/list/tuple of int — bin indices to block

    Returns
    -------
    ndarray (n_bins, n_bins) — modified row-stochastic matrix with no
        transitions into blocked bins.  If a row becomes all-zero
        (source bin only had exits to blocked bins), transitions are
        spread uniformly over remaining allowed bins.
    """
    if not blocked_bins:
        return trans
    trans = trans.copy()
    n = trans.shape[0]
    blocked = set(blocked_bins)
    allowed = [i for i in range(n) if i not in blocked]

    # Zero out columns for blocked bins
    for b in blocked:
        if 0 <= b < n:
            trans[:, b] = 0.0

    # Re-normalize rows
    row_sums = trans.sum(axis=1)
    for r in range(n):
        if row_sums[r] > 0:
            trans[r] /= row_sums[r]
        elif allowed:
            # Row is all-zero — distribute uniformly over allowed bins
            trans[r] = 0.0
            for a in allowed:
                trans[r, a] = 1.0 / len(allowed)
        # else: all bins blocked — leave as zero (degenerate; caller validates)

    return trans


def _snap_start_pctile(start_pctile, bin_edges, blocked_bins):
    """Snap starting percentile to nearest allowed bin if it falls in a blocked one.

    Parameters
    ----------
    start_pctile : float — desired starting percentile (0–1)
    bin_edges : ndarray — percentile bin boundaries [0, 0.2, ..., 1.0]
    blocked_bins : set/list/tuple of int — blocked bin indices

    Returns
    -------
    float — adjusted percentile (midpoint of nearest allowed bin),
        or original if already in an allowed bin.
    """
    if not blocked_bins:
        return start_pctile
    n_bins = len(bin_edges) - 1
    blocked = set(blocked_bins)

    # Find which bin the start percentile falls in
    current_bin = min(int(start_pctile * n_bins), n_bins - 1)
    current_bin = max(current_bin, 0)

    if current_bin not in blocked:
        return start_pctile  # already in allowed bin

    # Find nearest allowed bin by expanding distance
    best_bin = None
    best_dist = float("inf")
    midpoint = (bin_edges[current_bin] + bin_edges[current_bin + 1]) / 2
    for b in range(n_bins):
        if b in blocked:
            continue
        b_mid = (bin_edges[b] + bin_edges[b + 1]) / 2
        dist = abs(b_mid - midpoint)
        if dist < best_dist:
            best_dist = dist
            best_bin = b
    if best_bin is None:
        return start_pctile  # all bins blocked — degenerate
    # Return midpoint of nearest allowed bin
    return (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2


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
    blocked = p.get("mc_blocked_bins")
    return {
        "tab": tab,
        "mc_bins": int(p.get("mc_bins", MC_BINS)),
        "mc_sims": int(p.get("mc_sims", MC_SIMS)),
        "mc_years": int(p.get("mc_years", MC_DEFAULT_YEARS)),
        "mc_freq": p.get("mc_freq", MC_FREQ),
        "mc_window": p.get("mc_window"),
        "mc_start_yr": int(p.get("mc_start_yr", MC_DEFAULT_START_YR)),
        "mc_entry_q": float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)),
        "mc_blocked_bins": sorted(blocked) if blocked else [],
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
    Subsamples to mc_sims if fewer than cache size (e.g. free tier 100 sims).
    """
    if not _HAS_MC_CACHE:
        return None
    syr = int(p.get("mc_start_yr", MC_DEFAULT_START_YR))
    raw_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    pct_bin = snap_to_bin(raw_pctile)
    if abs(raw_pctile - pct_bin) > _CACHE_Q_TOLERANCE:
        return None
    max_sims = int(p.get("mc_sims", MC_SIMS))
    return get_cached_paths(syr, pct_bin, mc_years, max_sims=max_sims)


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
    if abs(raw_pctile - pct_bin) > _CACHE_Q_TOLERANCE:
        return None, None
    infl_pct = int(round(inflation * 100))
    return get_cached_overlay(syr, pct_bin, mc_years, int(wd_amount), infl_pct, mc_stack)


# ══════════════════════════════════════════════════════════════════════════════
# Serialization helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mc_metadata(p, tab, mc_years=None):
    """Build human-readable metadata dict for MC save files."""
    n_bins, n_sims, mc_window, mc_freq, _, _, _, yrs = _mc_setup_vars(p)
    if mc_years is None:
        mc_years = yrs
    return {
        "app": "Quantoshi",
        "version": "1.1",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tab": tab,
        "description": f"Monte Carlo {tab.upper()} simulation",
        "config": {
            "start_year": int(p.get("mc_start_yr", MC_DEFAULT_START_YR)),
            "years": mc_years,
            "entry_percentile": float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)),
            "n_bins": n_bins,
            "n_sims": n_sims,
            "frequency": mc_freq,
            "window": list(mc_window) if mc_window else None,
            "blocked_bins": list(p.get("mc_blocked_bins", [])),
            "amount": float(p.get("mc_amount", 0)),
            "inflation_pct": float(p.get("mc_infl", 0)),
            "start_stack_btc": float(p.get("mc_start_stack", 0)),
        },
    }


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

_MC_BANDS = [
    (0.01, 0.95, "rgba(220,120,0,0.06)", "MC 1\u201395%"),
    (0.05, 0.95, "rgba(220,120,0,0.10)", "MC 5\u201395%"),
    (0.25, 0.75, "rgba(220,120,0,0.18)", "MC 25\u201375%"),
]
_GHOST_BANDS = [
    (0.01, 0.95, "rgba(150,150,150,0.04)", "MC ref 1\u201395%"),
    (0.05, 0.95, "rgba(150,150,150,0.08)", "MC ref 5\u201395%"),
    (0.25, 0.75, "rgba(150,150,150,0.14)", "MC ref 25\u201375%"),
]


def _mc_build_traces(mc_ts, fan, extra_label="", show_median=True,
                     show_final_values=False, fan_usd=None,
                     hide_5_95_legend=False, bands=None,
                     suppress_legend=False):
    """Build MC fan band traces from precomputed fan percentiles.

    bands: band definitions (default: _MC_BANDS). Pass _GHOST_BANDS for ghost.
    suppress_legend: if True, hide all traces from legend (ghost mode).
    fan_usd: if provided, use these values for legend final-value labels.
    hide_5_95_legend: if True, suppress 5-95% band from legend.
    """
    if bands is None:
        bands = _MC_BANDS
    lf = fan_usd if fan_usd is not None else fan  # legend fan

    traces = []
    for p_lo, p_hi, fill_color, label in bands:
        if p_lo not in fan or p_hi not in fan:
            continue
        is_5_95 = (p_lo == 0.05 and p_hi == 0.95)
        show_leg = not suppress_legend and not (is_5_95 and hide_5_95_legend)
        if show_final_values and not suppress_legend:
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
    if show_median and 0.50 in fan:
        if suppress_legend:
            med_label = "MC ref median"
            med_color, med_width, med_dash = "rgba(150,150,150,0.4)", 1.2, "dash"
        else:
            med_label = "MC median" + extra_label
            if show_final_values and 0.50 in lf and len(lf[0.50]) > 0:
                med_label += f"  \u2192  {fmt_price(float(lf[0.50][-1]))}"
            med_color, med_width, med_dash = "rgba(220,120,0,0.9)", 1.5, "dot"
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[0.50]),
            mode="lines", name=med_label,
            line=dict(color=med_color, width=med_width, dash=med_dash),
            showlegend=not suppress_legend,
        ))
    return traces


def ghost_traces_from_params(p, x_end, disp_mode):
    """Build ghost fan traces from unblocked cache stored in params.

    Returns list of Scatter traces (empty if no ghost data or no blocked bins).
    Called from figures.py after the main overlay to prepend ghost reference fan.
    """
    ghost_data = p.get("mc_ghost_fan")
    if not ghost_data or not p.get("mc_blocked_bins"):
        return []
    fan_key = "fan_usd" if disp_mode == "usd" else "fan_btc"
    fan_raw = ghost_data.get(fan_key)
    ts_raw = ghost_data.get("ts")
    if not fan_raw or not ts_raw:
        return []
    fan = _mc_fan_from_lists(fan_raw)
    ts = np.array(ts_raw)
    # Clip to visible chart range
    clip_n = int(np.searchsorted(ts, x_end + 0.01))
    clip_n = max(clip_n, 1)
    ct = ts[:clip_n]
    cf = {k: v[:clip_n] for k, v in fan.items()}
    return _mc_build_traces(ct, cf, bands=_GHOST_BANDS, suppress_legend=True)


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


# ── Shared simulation helpers ─────────────────────────────────────────────────

def _build_mc_timeline(p, m, mc_years, mc_dt, clamp_start=False):
    """Build MC simulation timeline from params. Returns (start_yr, t_start, t_end, ts)."""
    mc_start_yr = int(p.get("mc_start_yr", MC_DEFAULT_START_YR))
    mc_t_start = yr_to_t(mc_start_yr, m.genesis)
    if clamp_start:
        mc_t_start = max(mc_t_start, 1.0)
    mc_t_end = mc_t_start + mc_years
    mc_ts = np.arange(mc_t_start, mc_t_end + mc_dt * 0.5, mc_dt)
    return mc_start_yr, mc_t_start, mc_t_end, mc_ts


def _prepare_sim(m, p, n_bins, step_days, mc_window, blocked, snap_grid=0):
    """Build transition matrix and compute start percentile. Returns (trans, bin_edges, n_bins, start_pctile)."""
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window)
    if blocked:
        trans = _apply_bin_mask(trans, blocked)
    start_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    if snap_grid > 0:
        start_pctile = round(start_pctile / snap_grid) * snap_grid
    start_pctile = max(_PCTILE_MIN, min(start_pctile, _PCTILE_MAX))
    if blocked:
        start_pctile = _snap_start_pctile(start_pctile, bin_edges, blocked)
    return trans, bin_edges, n_bins, start_pctile


def _try_cached(p, mc_years, blocked):
    """Check pre-computed path cache (skip when bins blocked). Returns paths or None."""
    if not blocked:
        return try_precomputed_paths(p, mc_years)
    return None


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
    n_bins, n_sims, mc_window, mc_freq, mc_ppy, mc_dt, step_days, mc_years = _mc_setup_vars(p)

    mc_start_yr, mc_t_start, mc_t_end, mc_ts = _build_mc_timeline(p, m, mc_years, mc_dt)

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
    if (cached and cached.get("path_key") == path_key
            and "price_paths" in cached):
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
            "metadata": _mc_metadata(p, "dca", mc_years),
        }
        ct, cf = _clip(mc_ts, fan)
        _, cf_usd = _clip(mc_ts, fan_usd)
        return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), result, cf_usd

    # ── Check pre-computed path cache (skip when bins blocked) ──────────
    blocked = p.get("mc_blocked_bins", [])
    cached_paths = _try_cached(p, mc_years, blocked)
    if cached_paths is not None:
        btc_paths, usd_paths = mc_dca(cached_paths, amount, start_stack)
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        fan = fan_usd if disp_mode == "usd" else fan_btc
        ct, cf = _clip(mc_ts, fan)
        _, cf_usd = _clip(mc_ts, fan_usd)
        return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), None, cf_usd

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins, start_pctile = _prepare_sim(
        m, p, n_bins, step_days, mc_window, blocked)

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, len(mc_ts), n_sims,
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
        "metadata": _mc_metadata(p, "dca", mc_years),
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
    n_bins, n_sims, mc_window, mc_freq, mc_ppy, mc_dt, step_days, mc_years = _mc_setup_vars(p)

    mc_start_yr, mc_t_start, mc_t_end, mc_ts = _build_mc_timeline(
        p, m, mc_years, mc_dt, clamp_start=True)

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
    if (cached and cached.get("path_key") == path_key
            and "price_paths" in cached):
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
            "metadata": _mc_metadata(p, tab, mc_years),
        }
        return _build_return(fan_btc, fan, _depl_extra(dstats), result,
                             fan_usd=fan_usd)

    # ── Check pre-computed cache (skip when bins blocked) ───────────────
    blocked = p.get("mc_blocked_bins", [])
    if not blocked:
        fan_btc, fan_usd = try_precomputed_overlay(p, mc_years, wd_amount,
                                                    inflation, mc_stack)
    else:
        fan_btc = fan_usd = None
    if fan_btc is not None:
        fan = fan_usd if disp_mode == "usd" else fan_btc
        return _build_return(fan_btc, fan, "", fan_usd=fan_usd)

    cached_paths = _try_cached(p, mc_years, blocked)
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
    trans, bin_edges, n_bins, start_pctile = _prepare_sim(
        m, p, n_bins, step_days, mc_window, blocked, snap_grid=0.05)

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, len(mc_ts), n_sims,
        m.qr_fits, m.genesis, mc_t_start, mc_dt,
    )
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
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
        "fan_btc": _mc_fan_to_lists(fan_btc),
        "fan_usd": _mc_fan_to_lists(fan_usd),
        "depletion": dstats,
        "metadata": _mc_metadata(p, tab, mc_years),
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
    n_bins, n_sims, mc_window, mc_freq, mc_ppy, mc_dt, step_days, mc_years = _mc_setup_vars(p)

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
    if (cached and cached.get("path_key") == path_key
            and "price_paths" in cached):
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
            "metadata": _mc_metadata(p, "hm", mc_years),
        }
        return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result

    # ── Check pre-computed server-side path cache (skip when bins blocked) ─
    blocked = p.get("mc_blocked_bins", [])
    cached_paths = _try_cached(p, mc_years, blocked)
    if cached_paths is not None:
        mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(cached_paths, mc_ts)
        return mc_cagr, mc_prices_arr, mc_mults, mc_labels, None

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins, start_pctile = _prepare_sim(
        m, p, n_bins, step_days, mc_window, blocked)

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, len(mc_ts), n_sims,
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
        "metadata": _mc_metadata(p, "hm", mc_years),
    }

    return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result
