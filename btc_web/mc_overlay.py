"""mc_overlay.py — Monte Carlo overlay logic for all chart tabs.

Extracted from figures.py to reduce its size and isolate MC concerns.
Contains: transition matrix cache, cache key functions, fan band trace
builders, and the 4 overlay functions (DCA, withdraw/retire/SC, heatmap).

Import chain: mc_overlay imports from _app_ctx, mc_cache, btc_core, markov.
figures.py imports from mc_overlay (one-directional, no circular dependency).
"""
from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

import _app_ctx
from colors import (
    CITADEL_OVERLAY_COLORS as _CITADEL_MC_COLORS, BTC_ORANGE, _hex_alpha,
    MC_AMBER, MC_GHOST_GRAY, CHART_FONT_LEGEND,
    MC_BAND_OUTER_ALPHA, MC_BAND_INNER_ALPHA,
    MC_GHOST_OUTER_ALPHA, MC_GHOST_INNER_ALPHA,
    MC_GHOST_MEDIAN_ALPHA, MC_MEDIAN_ALPHA,
)

logger = logging.getLogger(__name__)
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
except ImportError:
    pass
_HAS_MARKOV = _app_ctx._HAS_MARKOV

# ── Pre-computed MC cache (conditional) ──────────────────────────────────────
try:
    from mc_cache import (load_startup_cache as _load_startup_cache,
                          load_caches as _load_full_cache,
                          get_cached_paths, get_cached_overlay,
                          snap_to_bin, is_cached_year,
                          FAN_PCTS as _MC_CACHE_FAN_PCTS)
    # In production (gunicorn --preload), eagerly load the full MC cache so
    # all workers share it via copy-on-write.  In DEV mode, skip this — the
    # cache lazy-loads on first MC request via _ensure_full_cache().
    if not os.environ.get("DEV"):
        _load_full_cache()
    _HAS_MC_CACHE = True
except ImportError:
    _HAS_MC_CACHE = False

_MC_FAN_PCTS = _MC_CACHE_FAN_PCTS if _HAS_MC_CACHE else (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)

# ── MC overlay constants ──────────────────────────────────────────────────────
_PCTILE_MIN = 0.05              # min entry percentile (clip to avoid extreme tails)
_PCTILE_MAX = 0.95              # max entry percentile
_ANNOT_AX = 28                  # annotation arrow x-offset (pixels)
_CACHE_Q_TOLERANCE = 0.005      # max quantile distance for cache bin alignment

def _resolve_model(p):
    """Resolve the price model for MC simulation from mc_model_src param.

    Returns the model object from PRICE_MODELS (or DEFAULT_MODEL if
    the source model is not quantized or not found).
    """
    key = p.get("mc_model_src", "bub")
    mdl = _app_ctx.PRICE_MODELS.get(key)
    if mdl is not None and mdl.quantized:
        return mdl
    return _app_ctx.DEFAULT_MODEL


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


def filter_paths_by_regime(paths, t_axis, model, blocked_bins, n_bins=5,
                            tolerance=None):
    """Rank cached paths by alignment with user's regime preference; return
    paths sorted ASC by time spent in blocked bins (best-aligned first).

    Caller subsamples to display count via `paths[:mc_sims]`. With sims=1
    the user gets the SINGLE path that spent the least time in the blocked
    regimes — distinct paths for "only Bargain" vs "only Bubble" etc.

    Why rank-based, not threshold-based: cached paths simulate a
    400+-step Markov walk that visits every bin. A "drop paths above X%
    time in blocked bins" filter either drops everything (low threshold)
    or nothing (high threshold), depending on threshold choice and how
    many bins are blocked. The earlier threshold variant additionally
    capped at `expected × tolerance` which exceeded 1.0 (the maximum
    achievable time fraction) when ≥3 of 5 bins were blocked, making
    the filter a no-op for "only Bargain"/"only Bubble"-style selections.
    Rank-based always yields a meaningful ordering.

    Live (paid) MC sims apply blocked_bins via the transition matrix in
    `_apply_bin_mask` — they never transition INTO blocked bins, so
    rank-based filtering of cached paths is the best-effort analog for
    free-tier scenarios.

    Parameters
    ----------
    paths : np.ndarray (n_sims, n_steps) — price paths
    t_axis : np.ndarray (n_steps,) — time values (years since genesis)
        for each path step.
    model : PriceModel — used for `model.quantiles` + `price_at(q, t)`.
    blocked_bins : iterable of int — bin indices the user excluded.
        Empty/None → no-op (returns paths unchanged).
    n_bins : int — number of regime bins (default 5; uniform 0..1).
    tolerance : ignored (kept for back-compat with old call sites).

    Returns
    -------
    np.ndarray (n_sims, n_steps) — same paths, reordered. Path at index
        0 has the LEAST time in blocked bins.
    """
    if not blocked_bins or paths is None:
        return paths
    if not hasattr(paths, "size") or paths.size == 0:
        return paths
    import numpy as _np
    blocked = set(int(b) for b in blocked_bins)
    n_sims, n_steps = paths.shape

    # Pre-compute log-price at each (t, quantile) — shared across paths.
    # Without this, the per-path-step model.find_percentile path made
    # 100×481×27 ≈ 1.3M model.price_at calls (~33s); pre-computing
    # collapses that to 481×27 ≈ 13k calls (~1s) and lets the per-path
    # step lookup vectorize via np.searchsorted.
    sorted_qs = list(getattr(model, "quantiles", []) or [])
    n_q = len(sorted_qs)
    if n_q < 2:
        return paths  # model lacks quantile fan; can't classify
    qs_arr = _np.asarray(sorted_qs, dtype=float)
    log_ps_per_t = _np.empty((n_steps, n_q), dtype=float)
    for j in range(n_steps):
        t_safe = max(float(t_axis[j]), 0.5)
        for k, q in enumerate(sorted_qs):
            log_ps_per_t[j, k] = _np.log10(
                max(float(model.price_at(q, t_safe)), 1e-10))

    log_paths = _np.log10(_np.maximum(paths.astype(float), 1e-10))

    # For each step j: locate each path's interpolated percentile, convert
    # to bin index, accumulate blocked-step counts per path. Outer loop
    # is over n_steps (~481); inner ops vectorize over n_sims (~100).
    n_blocked_steps = _np.zeros(n_sims, dtype=int)
    for j in range(n_steps):
        log_ps = log_ps_per_t[j]
        idx = _np.searchsorted(log_ps, log_paths[:, j], side="right")
        pcts = _np.empty(n_sims, dtype=float)
        below = idx == 0
        above = idx == n_q
        mid = ~(below | above)
        pcts[below] = qs_arr[0]
        pcts[above] = qs_arr[-1]
        if mid.any():
            i_lo = idx[mid] - 1
            i_hi = idx[mid]
            ps_lo = log_ps[i_lo]
            ps_hi = log_ps[i_hi]
            denom = ps_hi - ps_lo
            denom = _np.where(_np.abs(denom) < 1e-30, 1e-30, denom)
            frac = (log_paths[mid, j] - ps_lo) / denom
            pcts[mid] = qs_arr[i_lo] + frac * (qs_arr[i_hi] - qs_arr[i_lo])
        bin_idx = (pcts * n_bins).astype(int)
        bin_idx = _np.clip(bin_idx, 0, n_bins - 1)
        for b in blocked:
            n_blocked_steps += (bin_idx == b)

    # Sort ascending by blocked-step count; stable sort keeps the cache's
    # natural order among ties.
    order = _np.argsort(n_blocked_steps, kind="stable")
    return paths[order]


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
        pkl_path = Path(__file__).parent.parent / "model_data.pkl"
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
        pkl_path = Path(__file__).parent.parent / "model_data.pkl"
        pkl_mtime = pkl_path.stat().st_mtime if pkl_path.exists() else None
        with open(_TRANS_CACHE_PATH, "wb") as f:
            pickle.dump({"matrices": _TRANS_MATRIX_CACHE, "_pkl_mtime": pkl_mtime}, f)
        _TRANS_CACHE_DIRTY = False
    except Exception:
        pass


_load_trans_cache_from_disk()


def _get_transition_matrix(m, n_bins, step_days, mc_window, model=None):
    """Get transition matrix from cache or build on the fly."""
    global _TRANS_CACHE_DIRTY
    model = model or _app_ctx.DEFAULT_MODEL
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

    model_key = model.short_name if model else "bub"
    cache_key = (n_bins, step_days, ws_cal, we_cal, model_key)
    cached = _TRANS_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1], n_bins

    trans, bin_edges, _ = build_transition_matrix(
        m.price_prices, m.price_years, model,
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
        "mc_model_src": p.get("mc_model_src", "bub"),
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

def _matches_prebuilt_cache_params(p) -> bool:
    """The pre-computed cache was built with fixed mc_bins=MC_BINS and
    mc_window = [MC_WINDOW_START, today]. If the user has overridden
    either, the cache's paths were NOT generated under their parameters —
    return False so we fall through to live simulation instead of silently
    serving wrong-parameter bands.
    """
    from mc_cache import MC_BINS as _CACHE_BINS, MC_WINDOW_START as _CACHE_WIN_START
    # mc_bins: absent or equal to the build-time default.
    user_bins = p.get("mc_bins")
    if user_bins is not None and int(user_bins) != int(_CACHE_BINS):
        return False
    # mc_window: None/absent is fine (caller uses the default). A list that
    # differs from the prebuilt start year invalidates the cache lookup.
    win = p.get("mc_window")
    if win and isinstance(win, (list, tuple)) and len(win) >= 1:
        try:
            if int(win[0]) != int(_CACHE_WIN_START):
                return False
        except (TypeError, ValueError):
            return False
    return True


def try_precomputed_paths(p, mc_years):
    """Look up pre-computed price paths using uniform mc_start_yr/mc_entry_q.

    Returns (n_sims, n_steps) ndarray or None.
    Only returns cached data when entry_q is cache-aligned (within 0.5% of a
    10% bin) AND the user hasn't overridden mc_bins/mc_window (since the
    cache was built with fixed defaults). Subsamples to mc_sims if fewer
    than cache size (e.g. free tier 100 sims).
    """
    if not _HAS_MC_CACHE:
        return None
    if not _matches_prebuilt_cache_params(p):
        return None
    syr = int(p.get("mc_start_yr", MC_DEFAULT_START_YR))
    raw_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    pct_bin = snap_to_bin(raw_pctile)
    if abs(raw_pctile - pct_bin) > _CACHE_Q_TOLERANCE:
        return None
    max_sims = int(p.get("mc_sims", MC_SIMS))
    model_key = p.get("mc_model_src", "bub")
    return get_cached_paths(model_key, syr, pct_bin, mc_years, max_sims=max_sims)


def try_precomputed_overlay(p, mc_years, wd_amount, inflation, mc_stack):
    """Look up pre-computed withdraw overlay fans.

    Same bins/window guards as :func:`try_precomputed_paths`.
    """
    if not _HAS_MC_CACHE:
        return None, None
    if not _matches_prebuilt_cache_params(p):
        return None, None
    syr = int(p.get("mc_start_yr", MC_DEFAULT_START_YR))
    raw_pctile = float(p.get("mc_entry_q", MC_DEFAULT_ENTRY_Q)) / 100.0
    pct_bin = snap_to_bin(raw_pctile)
    if abs(raw_pctile - pct_bin) > _CACHE_Q_TOLERANCE:
        return None, None
    infl_pct = int(round(inflation * 100))
    model_key = p.get("mc_model_src", "bub")
    return get_cached_overlay(model_key, syr, pct_bin, mc_years, int(wd_amount), infl_pct, mc_stack)


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
    """Serialize price_paths ndarray for client caching.

    Returns a dict {"b64": base64-str, "shape": [nsims, nsteps]} — binary
    float64 bytes, base64-encoded. Matches the precision used by
    engines.adapter.encode so round-trips through either path (client
    cache or Celery serialisation) preserve bit-identical paths.

    Legacy list-of-lists form is still accepted by `_mc_paths_from_lists`
    for clients holding the old cached result shape.
    """
    import base64 as _b64
    arr = np.ascontiguousarray(np.asarray(paths, dtype=np.float64))
    return {
        "b64": _b64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": "float64",
    }


def _mc_paths_from_lists(data):
    """Restore price_paths ndarray from client cache.

    Accepts either the new base64+shape dict form or the legacy list-of-lists.
    Reads the dtype tag to support older client caches that encoded as
    float32; defaults to float64 for new payloads.
    """
    if isinstance(data, dict) and "b64" in data:
        import base64 as _b64
        raw = _b64.b64decode(data["b64"])
        shape = tuple(data.get("shape") or ())
        dtype = np.dtype(data.get("dtype", "float32"))
        arr = np.frombuffer(raw, dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        # frombuffer returns a read-only view; caller mutates in some paths.
        # Also upcast any legacy float32 payload to float64 for consistency.
        return np.ascontiguousarray(arr, dtype=np.float64)
    return np.array(data, dtype=np.float64)


def _build_mc_result(tab, path_key, overlay_key, mc_ts, price_paths,
                     p, mc_years, **extras):
    """Build standard MC result dict for client-side caching."""
    d = {
        "tab": tab,
        "path_key": path_key,
        "overlay_key": overlay_key,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
        "metadata": _mc_metadata(p, tab, mc_years),
    }
    d.update(extras)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Trace builders
# ══════════════════════════════════════════════════════════════════════════════

# MC band opacity pattern: matches _build_symmetric_bands (0.08 outer, 0.15 inner)
# so MC visually integrates with other model overlays instead of looking alien.
_MC_BANDS = [
    (0.05, 0.95, _hex_alpha(MC_AMBER, MC_BAND_OUTER_ALPHA), "MC 5\u201395%"),
    (0.25, 0.75, _hex_alpha(MC_AMBER, MC_BAND_INNER_ALPHA), "MC 25\u201375%"),
]
_GHOST_BANDS = [
    (0.05, 0.95, _hex_alpha(MC_GHOST_GRAY, MC_GHOST_OUTER_ALPHA), "MC ref 5\u201395%"),
    (0.25, 0.75, _hex_alpha(MC_GHOST_GRAY, MC_GHOST_INNER_ALPHA), "MC ref 25\u201375%"),
]


def _mc_build_traces(mc_ts, fan, extra_label="", show_median=True,
                     show_final_values=False, fan_usd=None,
                     hide_5_95_legend=False, bands=None,
                     suppress_legend=False, line_shape="linear"):
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
            mode="lines", line=dict(width=0, shape=line_shape), showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[p_lo]),
            mode="lines", line=dict(width=0, shape=line_shape), fill="tonexty",
            fillcolor=fill_color, name=label, showlegend=show_leg, hoverinfo="skip",
        ))
    if show_median and 0.50 in fan:
        if suppress_legend:
            med_label = "MC ref median"
            med_color, med_width, med_dash = _hex_alpha(MC_GHOST_GRAY, MC_GHOST_MEDIAN_ALPHA), 1.2, "dash"
        else:
            med_label = "MC median" + extra_label
            if show_final_values and 0.50 in lf and len(lf[0.50]) > 0:
                med_label += f"  \u2192  {fmt_price(float(lf[0.50][-1]))}"
            med_color, med_width, med_dash = _hex_alpha(MC_AMBER, MC_MEDIAN_ALPHA), 1.5, "dot"
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[0.50]),
            mode="lines", name=med_label,
            line=dict(color=med_color, width=med_width, dash=med_dash, shape=line_shape),
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
                font=dict(size=CHART_FONT_LEGEND, color=mc_col),
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


def _prepare_sim(m, p, n_bins, step_days, mc_window, blocked, snap_grid=0, model=None):
    """Build transition matrix and compute start percentile. Returns (trans, bin_edges, n_bins, start_pctile)."""
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window, model=model)
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


def _check_client_cache(p, path_key):
    """Check client-side MC cache for a path_key match.

    Returns (cached_dict, True) on path_key hit, (None, False) on miss.
    Caller checks overlay_key for full vs partial match.

    Also rejects caches whose path count is below the current ``mc_sims``
    request — otherwise raising the sim count would silently narrow the
    band by averaging over the smaller cached array. (Citadel has this
    check inline; DCA/Retire/SC/HM shared it here post-audit.)
    """
    cached = p.get("mc_cached")
    if not (cached and cached.get("path_key") == path_key
            and "price_paths" in cached):
        return None, False
    shape = cached["price_paths"].get("shape") if isinstance(
        cached["price_paths"], dict) else None
    if shape and len(shape) >= 1:
        cached_sims = int(shape[0])
        requested_sims = int(p.get("mc_sims") or 0)
        if requested_sims and cached_sims < requested_sims:
            return None, False
    return cached, True


def _run_full_simulation(m, p, n_bins, step_days, mc_window, mc_ts,
                         n_sims, mc_t_start, mc_dt, snap_grid=0):
    """Run full MC simulation: build transition matrix + generate price paths.

    Honors ``p["mc_seed"]`` if present so repeat MC runs with the same
    config + seed produce identical paths. Prior to 2026-04-17 the
    monte_carlo_prices call created an unseeded Generator, so results
    were non-reproducible regardless of UI-level seed controls.
    """
    blocked = p.get("mc_blocked_bins", [])
    model = _resolve_model(p)
    trans, bin_edges, n_bins, start_pctile = _prepare_sim(
        m, p, n_bins, step_days, mc_window, blocked, snap_grid=snap_grid, model=model)
    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, len(mc_ts), n_sims,
        model, mc_t_start, mc_dt, rng=p.get("mc_seed"),
    )
    return price_paths


# ══════════════════════════════════════════════════════════════════════════════
# DCA overlay
# ══════════════════════════════════════════════════════════════════════════════

def _mc_dca_overlay(m, p, ts, t_start, dt, start_stack, disp_mode):
    """Build Monte Carlo fan band traces for DCA overlay.

    Returns (traces, result_dict, cf_usd).
    """
    _ls = "hv" if p.get("discrete") else "linear"
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

    # ── Level 1: Client-side cache ────────────────────────────────────────
    cached, cache_hit = _check_client_cache(p, path_key)
    if cache_hit:
        if cached.get("overlay_key") == overlay_key:
            fan_btc = _mc_fan_from_lists(cached["fan_btc"])
            fan_usd = _mc_fan_from_lists(cached["fan_usd"])
            fan = fan_usd if disp_mode == "usd" else fan_btc
            ct, cf = _clip(mc_ts, fan)
            _, cf_usd = _clip(mc_ts, fan_usd)
            return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd, line_shape=_ls), None, cf_usd

        # Path hit, overlay miss — recompute DCA from cached price paths
        price_paths = _mc_paths_from_lists(cached["price_paths"])
        btc_paths, usd_paths = mc_dca(price_paths, amount, start_stack)
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        fan = fan_usd if disp_mode == "usd" else fan_btc

        result = _build_mc_result("dca", path_key, overlay_key,
                                  mc_ts, None, p, mc_years,
                                  fan_btc=_mc_fan_to_lists(fan_btc),
                                  fan_usd=_mc_fan_to_lists(fan_usd))
        result["ts"] = cached["ts"]
        result["price_paths"] = cached["price_paths"]
        ct, cf = _clip(mc_ts, fan)
        _, cf_usd = _clip(mc_ts, fan_usd)
        return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd, line_shape=_ls), result, cf_usd

    # ── Level 2: Pre-computed server cache ─────────────────────────────────
    blocked = p.get("mc_blocked_bins", [])
    cached_paths = _try_cached(p, mc_years, blocked)
    if cached_paths is not None:
        btc_paths, usd_paths = mc_dca(cached_paths, amount, start_stack)
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        fan = fan_usd if disp_mode == "usd" else fan_btc
        ct, cf = _clip(mc_ts, fan)
        _, cf_usd = _clip(mc_ts, fan_usd)
        return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd, line_shape=_ls), None, cf_usd

    # ── Level 3: Full simulation ───────────────────────────────────────────
    price_paths = _run_full_simulation(
        m, p, n_bins, step_days, mc_window, mc_ts, n_sims, mc_t_start, mc_dt)
    btc_paths, usd_paths = mc_dca(price_paths, amount, start_stack)

    fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
    fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
    fan = fan_usd if disp_mode == "usd" else fan_btc

    result = _build_mc_result("dca", path_key, overlay_key,
                              mc_ts, price_paths, p, mc_years,
                              fan_btc=_mc_fan_to_lists(fan_btc),
                              fan_usd=_mc_fan_to_lists(fan_usd))

    ct, cf = _clip(mc_ts, fan)
    _, cf_usd = _clip(mc_ts, fan_usd)
    return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd, line_shape=_ls), result, cf_usd


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
    _ls = "hv" if p.get("discrete") else "linear"
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
                                  hide_5_95_legend=hide_5_95_legend,
                                  line_shape=_ls)
        annots = _mc_depletion_annots(mc_ts, fan_btc, mc_start_yr, mc_years,
                                       existing_annot_count) if do_annot else []
        return traces, annots, result

    # ── Level 1: Client-side cache ────────────────────────────────────────
    cached, cache_hit = _check_client_cache(p, path_key)
    if cache_hit:
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

        result = _build_mc_result(tab, path_key, overlay_key,
                                  mc_ts, None, p, mc_years,
                                  fan_btc=_mc_fan_to_lists(fan_btc),
                                  fan_usd=_mc_fan_to_lists(fan_usd),
                                  depletion=dstats)
        result["ts"] = cached["ts"]
        result["price_paths"] = cached["price_paths"]
        return _build_return(fan_btc, fan, _depl_extra(dstats), result,
                             fan_usd=fan_usd)

    # ── Level 2: Pre-computed server cache ─────────────────────────────────
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

    # ── Level 3: Full simulation ───────────────────────────────────────────
    price_paths = _run_full_simulation(
        m, p, n_bins, step_days, mc_window, mc_ts, n_sims,
        mc_t_start, mc_dt, snap_grid=0.05)
    btc_paths, usd_paths, depl_steps = mc_retire(
        price_paths, mc_stack, wd_amount, inflation, mc_dt,
    )

    fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
    fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
    dstats = depletion_stats(depl_steps, len(mc_ts), mc_dt, mc_t_start)
    fan = fan_usd if disp_mode == "usd" else fan_btc

    result = _build_mc_result(tab, path_key, overlay_key,
                              mc_ts, price_paths, p, mc_years,
                              fan_btc=_mc_fan_to_lists(fan_btc),
                              fan_usd=_mc_fan_to_lists(fan_usd),
                              depletion=dstats)

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


def _mc_supercharge_mode_b_overlay(m, p, syr, delays, target_yr, start_stack,
                                    ppy, dt, inflation, palette,
                                    bisect_iters=60, chart_layout=0,
                                    sel_qs=None):
    """Mode B MC overlay: per-delay distribution of max-spend that depletes
    by target_yr, computed across MC price paths.

    For each delay d:
      1. Subset path columns to t in [syr+d, target_yr]
      2. Vectorized bisection across paths: at each candidate mid value,
         simulate withdrawal-per-step inflated from t_start_d, take cumsum
         of btc_consumed = adj_wd[t] / paths[i, t], find which paths
         survive (stack > 0 the whole way), update lo/hi per path
      3. After bisect_iters, lo[i] is each path's max sustainable wd

    Aggregate across paths: percentile fan at each delay {5,25,50,75,95}.
    Returns (traces, mc_result_or_None). Empty list when MC is disabled
    or paths can't be obtained.
    """
    if not _HAS_MARKOV or not p.get("mc_enabled"):
        return [], None

    n_bins, n_sims, mc_window, mc_freq, mc_ppy, mc_dt, step_days, mc_years = _mc_setup_vars(p)

    # Acquire paths: free-tier cache → live simulation.
    try:
        from mc_cache import get_cached_paths as _get_cached_paths_local
    except ImportError:
        _get_cached_paths_local = None
    pct_bin = float(p.get("mc_entry_q") or 10) / 100.0
    paths = None
    if _HAS_MC_CACHE and _get_cached_paths_local is not None:
        try:
            paths = _get_cached_paths_local(
                p.get("mc_model_src", "bub"),
                int(p.get("mc_start_yr", MC_DEFAULT_START_YR)),
                pct_bin, mc_years)
        except Exception:
            paths = None
    if paths is None or not hasattr(paths, "shape") or paths.size == 0:
        # Live simulation fallback (paid tier)
        try:
            mc_start_yr_, mc_t_start_, mc_t_end_, mc_ts_ = _build_mc_timeline(
                p, m, mc_years, mc_dt)
            paths = _run_full_simulation(
                m, p, n_bins, step_days, mc_window, mc_ts_, n_sims,
                mc_t_start_, mc_dt)
        except Exception as e:
            logger.debug("[MC-SC-B] sim fallback failed: %s", e)
            return [], None
    if paths is None or paths.size == 0:
        return [], None

    # Apply regime filter (rank ascending by time-in-blocked) + trim to mc_sims.
    blocked = p.get("mc_blocked_bins") or []
    n_steps_full = paths.shape[1]
    mc_t_start = yr_to_t(int(p.get("mc_start_yr", MC_DEFAULT_START_YR)),
                         m.genesis)
    mc_t_end = mc_t_start + mc_years
    t_axis_full = np.linspace(mc_t_start, mc_t_end, n_steps_full)
    if blocked:
        regime_model = _app_ctx.PRICE_MODELS.get(
            p.get("mc_model_src", "bub")) or _app_ctx.DEFAULT_MODEL
        if regime_model is not None:
            paths = filter_paths_by_regime(
                paths, t_axis_full, regime_model, blocked, n_bins=n_bins)
    n_paths_max = int(p.get("mc_sims") or 200)
    if paths.shape[0] > n_paths_max:
        paths = paths[:n_paths_max]
    n_paths = paths.shape[0]
    if n_paths == 0:
        return [], None

    # For each delay, vectorize a bisection across paths.
    pct_levels = [5, 25, 50, 75, 95]
    fan_per_pct = {pct: [] for pct in pct_levels}
    target_t = yr_to_t(int(target_yr), m.genesis)

    for d in delays:
        t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
        if target_t <= t_start_d:
            for pct in pct_levels:
                fan_per_pct[pct].append(0.0)
            continue
        # Slice paths to the [t_start_d, target_t] window
        j_start = int(np.searchsorted(t_axis_full, t_start_d, side="left"))
        j_end = int(np.searchsorted(t_axis_full, target_t, side="right"))
        if j_end - j_start < 2:
            for pct in pct_levels:
                fan_per_pct[pct].append(0.0)
            continue
        path_slice = paths[:, j_start:j_end]                # (n_paths, n_steps_d)
        ts_slice = t_axis_full[j_start:j_end]               # (n_steps_d,)
        # Per-step inflation factor relative to t_start_d
        adj_factors = (1.0 + inflation) ** (ts_slice - t_start_d)  # (n_steps_d,)
        # Generous upper bound: 4× annual stack value at first price.
        first_price_per_path = path_slice[:, 0]             # (n_paths,)
        hi = start_stack * first_price_per_path * ppy * 4.0
        lo = np.zeros(n_paths)
        # Bisection — per-path lo/hi update; vectorized survival check.
        for _ in range(int(bisect_iters)):
            mid = (lo + hi) / 2.0                            # (n_paths,)
            adj_wd = mid[:, None] * adj_factors[None, :]     # (n_paths, n_steps_d)
            btc_per_step = adj_wd / np.maximum(path_slice, 1e-12)
            cum_btc = np.cumsum(btc_per_step, axis=1)
            stack_traj = start_stack - cum_btc               # (n_paths, n_steps_d)
            survived = (stack_traj > 0).all(axis=1)          # (n_paths,)
            lo = np.where(survived, mid, lo)
            hi = np.where(survived, hi, mid)
        max_wd_paths = lo                                    # (n_paths,)
        for pct in pct_levels:
            fan_per_pct[pct].append(float(np.percentile(max_wd_paths, pct)))

    # Build traces; layout-aware so the MC overlay aligns with the chart's
    # x-axis (delay vs quantile).
    mc_color = palette.get("mc_band", MC_AMBER) if palette else MC_AMBER
    traces = []
    if chart_layout == 2 and sel_qs:
        # Layout 2 plots quantile on x with one line per delay. Emit a
        # horizontal MC-median line for each delay at the quantile range so
        # the user can see how the deterministic-per-quantile lines compare
        # to MC's path-distribution median (constant across the q axis).
        for di, d in enumerate(delays):
            y_const = [fan_per_pct[50][di]] * len(sel_qs)
            d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
            traces.append(go.Scatter(
                x=list(sel_qs), y=y_const,
                mode="lines",
                line=dict(color=mc_color, width=1.5, dash="dot"),
                name=f"MC median {d_lbl}",
                legendgroup="mc-sc-b",
                hovertemplate=(f"MC median (delay {d_lbl})<br>"
                               "max-spend=%{y:,.0f}<extra></extra>"),
            ))
        return traces, None
    # Layouts 0 + 1: fan band across delay axis.
    # Outer 5-95 band
    traces.append(go.Scatter(
        x=list(delays), y=fan_per_pct[95],
        mode="lines", line=dict(color=mc_color, width=0),
        showlegend=False, hoverinfo="skip", legendgroup="mc-sc-b",
    ))
    traces.append(go.Scatter(
        x=list(delays), y=fan_per_pct[5],
        mode="lines", line=dict(color=mc_color, width=0),
        fill="tonexty",
        fillcolor=_hex_alpha(mc_color, MC_BAND_OUTER_ALPHA),
        showlegend=False, hoverinfo="skip", legendgroup="mc-sc-b",
    ))
    # Inner 25-75 band
    traces.append(go.Scatter(
        x=list(delays), y=fan_per_pct[75],
        mode="lines", line=dict(color=mc_color, width=0),
        showlegend=False, hoverinfo="skip", legendgroup="mc-sc-b",
    ))
    traces.append(go.Scatter(
        x=list(delays), y=fan_per_pct[25],
        mode="lines", line=dict(color=mc_color, width=0),
        fill="tonexty",
        fillcolor=_hex_alpha(mc_color, MC_BAND_INNER_ALPHA),
        showlegend=False, hoverinfo="skip", legendgroup="mc-sc-b",
    ))
    # Median line + marker
    traces.append(go.Scatter(
        x=list(delays), y=fan_per_pct[50],
        mode="lines+markers",
        line=dict(color=mc_color, width=2),
        marker=dict(color=mc_color, size=8, symbol="diamond"),
        name=f"MC median ({n_paths} paths)",
        legendgroup="mc-sc-b",
        hovertemplate=("delay=%{x}<br>median max-spend=%{y:,.0f}"
                       "<extra>MC</extra>"),
    ))
    return traces, None


# ══════════════════════════════════════════════════════════════════════════════
# Citadel Planner overlay
# ══════════════════════════════════════════════════════════════════════════════

def _mc_citadel_overlay(m, p, citadel_config, citadel_model):
    """Build Monte Carlo fan band traces for Citadel Planner.

    Runs the citadel engine with MC-generated price paths.
    Returns (traces, result_dict) where result_dict is SimResult.to_dict()
    or ([], None) if MC is disabled / unavailable.
    """
    if not _HAS_MARKOV or not p.get("mc_enabled"):
        logger.debug("[MC-CITADEL] skipped: _HAS_MARKOV=%s, mc_enabled=%s", _HAS_MARKOV, p.get('mc_enabled'))
        return [], None

    import time as _time
    _mc_t0 = _time.time()
    n_bins, n_sims, mc_window, mc_freq, mc_ppy, mc_dt, step_days, mc_years = _mc_setup_vars(p)

    logger.debug("[MC-CITADEL] starting: n_sims=%s, mc_years=%s, bins=%s", n_sims, mc_years, n_bins)
    mc_start_yr, mc_t_start, mc_t_end, mc_ts = _build_mc_timeline(
        p, m, mc_years, mc_dt, clamp_start=True)

    # Generate price paths
    path_key = _mc_path_key(p, "cp")

    # Check client-side cache (reject if fewer sims than requested)
    cached, cache_hit = _check_client_cache(p, path_key)
    if cache_hit:
        price_paths = _mc_paths_from_lists(cached["price_paths"])
        if price_paths.shape[0] < n_sims:
            logger.debug("[MC-CITADEL] client cache: %d paths < %d requested, bypassing", price_paths.shape[0], n_sims)
            cache_hit = False
    if not cache_hit:
        # Try pre-computed server cache
        blocked = p.get("mc_blocked_bins", [])
        cached_paths = _try_cached(p, mc_years, blocked)
        if cached_paths is not None:
            price_paths = cached_paths
        else:
            # Full simulation
            price_paths = _run_full_simulation(
                m, p, n_bins, step_days, mc_window, mc_ts, n_sims,
                mc_t_start, mc_dt, snap_grid=0.05)

    # Ensure enough periods for citadel engine
    from engines.citadel import FREQ_PPY as _CITADEL_FREQ_PPY, simulate as _citadel_sim
    citadel_ppy = _CITADEL_FREQ_PPY.get(citadel_config.freq, 12)
    n_periods_needed = int((citadel_config.end_yr - citadel_config.start_yr) * citadel_ppy)
    # If MC paths are shorter than citadel range, truncate citadel to available MC data
    if price_paths.shape[1] < n_periods_needed:
        n_periods_needed = price_paths.shape[1]
        logger.debug("[MC-CITADEL] truncating citadel to %d periods (MC shorter than citadel range)", n_periods_needed)
    logger.debug("[MC-CITADEL] paths: %s, using %d periods", price_paths.shape, n_periods_needed)

    # Resample MC price paths to citadel's time grid if frequencies differ
    # MC runs at mc_freq (typically Monthly), citadel may use same or different freq
    # Use linear interpolation along the time axis
    citadel_dt = 1.0 / citadel_ppy
    from btc_core import yr_to_t
    citadel_t0 = yr_to_t(citadel_config.start_yr)
    citadel_ts = np.array([citadel_t0 + i * citadel_dt for i in range(n_periods_needed)])

    # Map citadel time points onto MC time indices
    mc_indices = np.interp(citadel_ts, mc_ts, np.arange(len(mc_ts)))
    # Interpolate in LOG-PRICE space (geometric interpolation). BTC price
    # paths can span many orders of magnitude; linear interpolation between
    # e.g. $50k and $500k would give $275k at the midpoint whereas the true
    # geometric mean is ~$158k. Log-space interpolation preserves the
    # compounding character of the underlying process. Guard against
    # non-positive prices (shouldn't happen for BTC but be safe).
    n_mc_sims = price_paths.shape[0]
    resampled = np.zeros((n_mc_sims, n_periods_needed))
    mc_x = np.arange(price_paths.shape[1])
    for s in range(n_mc_sims):
        path = price_paths[s]
        safe = np.maximum(path, 1e-12)
        resampled[s] = np.exp(np.interp(mc_indices, mc_x, np.log(safe)))

    # Adjust citadel config to match available MC range
    import copy
    citadel_config = copy.deepcopy(citadel_config)
    mc_end_yr = citadel_config.start_yr + n_periods_needed / citadel_ppy
    if mc_end_yr < citadel_config.end_yr:
        citadel_config.end_yr = int(mc_end_yr)

    # Run citadel engine with MC price paths — use Celery if available
    try:
        from engines.adapter import submit_simulation, _HAS_CELERY
        if _HAS_CELERY:
            task_or_result = submit_simulation(
                citadel_config, citadel_model,
                price_paths=resampled, use_celery=True)
            # Check if it's an AsyncResult (Celery) or SimResult (in-process)
            if hasattr(task_or_result, 'ready'):
                # Celery task — check if already done
                if task_or_result.ready():
                    from engines.citadel import SimResult
                    result = SimResult.from_dict(task_or_result.get(timeout=5))
                else:
                    # Task submitted but not done — return pending marker
                    return [], {"_celery_task_id": task_or_result.id, "_pending": True}
            else:
                result = task_or_result
        else:
            logger.debug("[MC-CITADEL] running in-process: %d sims x %d steps", resampled.shape[0], resampled.shape[1])
            result = _citadel_sim(citadel_config, citadel_model, price_paths=resampled)
            logger.debug("[MC-CITADEL] done: %.0fms total", (_time.time()-_mc_t0)*1000)
    except (ValueError, NotImplementedError) as e:
        logger.debug("[MC-CITADEL] error: %s", e)
        return [], None

    # Build fan band traces per asset class
    traces = []
    ts_plot = result.time_axis
    disp_mode = p.get("disp_mode", "usd_per_asset")

    # The "total" aggregate follows the active price model's palette color,
    # matching the tab 1 convention where quantile bands/trace use the driving
    # model's color. Per-asset traces (btc / cash / reserves / investments)
    # keep their distinct colors so the asset breakdown stays readable.
    from figures.common import _get_model_color
    _model_color = _get_model_color(p.get("mc_model_src") or p.get("model_src", "bub"), p)
    _CITADEL_MC_COLORS_ACTIVE = dict(_CITADEL_MC_COLORS)
    _CITADEL_MC_COLORS_ACTIVE["total"] = _model_color

    # In BTC mode, only show BTC holdings (in BTC, not USD)
    if disp_mode == "btc":
        _mc_assets = {"btc": _CITADEL_MC_COLORS_ACTIVE.get("btc_usd", BTC_ORANGE)}
    else:
        _mc_assets = _CITADEL_MC_COLORS_ACTIVE

    for asset_key, color in _mc_assets.items():
        # Extract per-sim arrays for this asset
        if asset_key == "total":
            data_2d = result.total_usd                       # (n_sims, n_periods)
        elif asset_key == "btc_usd":
            data_2d = result.btc_holdings * result.btc_prices  # USD value
        elif asset_key == "btc":
            data_2d = result.btc_holdings                      # BTC units
        elif asset_key == "cash":
            data_2d = result.cash_balances
        elif asset_key == "reserves_total":
            data_2d = result.reserve_balances.sum(axis=2)
        elif asset_key == "investments_total":
            data_2d = result.invest_balances.sum(axis=2)
        else:
            continue

        # Compute percentile bands
        p5  = np.percentile(data_2d, 5, axis=0)
        p25 = np.percentile(data_2d, 25, axis=0)
        p50 = np.median(data_2d, axis=0)
        p75 = np.percentile(data_2d, 75, axis=0)
        p95 = np.percentile(data_2d, 95, axis=0)

        nice_name = {"total": "Total", "btc_usd": "BTC (USD)",
                     "btc": "BTC", "cash": "Cash",
                     "reserves_total": "Reserves",
                     "investments_total": "Investments"}.get(asset_key, asset_key)

        lg = f"mc_{asset_key}"

        # 5-95% band (outer, 0.08 alpha — matches _build_symmetric_bands)
        traces.append(go.Scatter(
            x=list(ts_plot), y=list(p95), mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=lg,
        ))
        traces.append(go.Scatter(
            x=list(ts_plot), y=list(p5), mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor=f"rgba({_hex_to_rgb(color)},{MC_BAND_OUTER_ALPHA})",
            name=f"MC {nice_name} 5\u201395%", showlegend=False, hoverinfo="skip",
            legendgroup=lg,
        ))

        # 25-75% band (inner, 0.15 alpha — matches _build_symmetric_bands)
        traces.append(go.Scatter(
            x=list(ts_plot), y=list(p75), mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=lg,
        ))
        traces.append(go.Scatter(
            x=list(ts_plot), y=list(p25), mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor=f"rgba({_hex_to_rgb(color)},{MC_BAND_INNER_ALPHA})",
            name=f"MC {nice_name} 25\u201375%", showlegend=False, hoverinfo="skip",
            legendgroup=lg,
        ))

        # Median dotted line (legend entry — clicking toggles all bands in group)
        final_val = fmt_price(float(p50[-1])) if len(p50) > 0 else ""
        traces.append(go.Scatter(
            x=list(ts_plot), y=list(p50), mode="lines",
            name=f"MC {nice_name} median  \u2192  {final_val}",
            line=dict(color=color, width=1.5, dash="dot"),
            legendgroup=lg,
        ))

    result_dict = result.to_dict()
    # Attach path_key for client-side caching
    result_dict["path_key"] = path_key
    result_dict["n_sims"] = int(resampled.shape[0])  # actual sim count (may differ from requested)
    result_dict["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return traces, result_dict


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


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

    # ── Level 1: Client-side cache ────────────────────────────────────────
    cached, cache_hit = _check_client_cache(p, path_key)
    if cache_hit:
        price_paths = _mc_paths_from_lists(cached["price_paths"])
        cached_ts = np.array(cached["ts"])
        mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(price_paths, cached_ts)
        if cached.get("overlay_key") == overlay_key:
            return mc_cagr, mc_prices_arr, mc_mults, mc_labels, None
        result = _build_mc_result("hm", path_key, overlay_key,
                                  mc_ts, None, p, mc_years)
        result["ts"] = cached["ts"]
        result["price_paths"] = cached["price_paths"]
        return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result

    # ── Level 2: Pre-computed server cache ─────────────────────────────────
    blocked = p.get("mc_blocked_bins", [])
    cached_paths = _try_cached(p, mc_years, blocked)
    if cached_paths is not None:
        mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(cached_paths, mc_ts)
        return mc_cagr, mc_prices_arr, mc_mults, mc_labels, None

    # ── Level 3: Full simulation ───────────────────────────────────────────
    price_paths = _run_full_simulation(
        m, p, n_bins, step_days, mc_window, mc_ts, n_sims, t_start, mc_dt)

    mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(price_paths, mc_ts)

    result = _build_mc_result("hm", path_key, overlay_key,
                              mc_ts, price_paths, p, mc_years)

    return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result
