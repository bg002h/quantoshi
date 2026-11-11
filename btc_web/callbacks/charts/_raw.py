"""Tab-1 Raw MC: control coercion and the overlay drawn by ``update_bubble``.

Raw never touches BTCPay, ``mc_cache``, the payment modal or the MC stores:
it simulates on request from ``raw_mc``'s price registry and draws onto a
clone of the fast-path figure. Memory (ALARA): a ``RawResult`` itself is
never kept. What is kept is a per-worker redraw cache (review M-3, operator
ruling 2026-11-10) of the last ``_CACHE_MAX`` simulations' drawable parts --
float32, read-only paths -- keyed on the full ``RawRequest``, so a cosmetic
``update_bubble`` trigger (pan, point size, legend, paths/extremes toggles)
skips the 30-126 ms re-simulation. Worst case 2 x 3200 x 481 x 4 B ~ 12 MB per
worker; the default 1000 sims is ~4 MB. Dropped when the price registry
changes (``raw_mc._CACHE_CLEARERS``); the date and live price are in the key.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
from dash import html

import _app_ctx
import raw_mc
from btc_core import yr_to_t
from callbacks.coerce import _ci_or, _cf

RAW_SIMS_DEFAULT = 1000
RAW_YEARS = (10, 20, 30, 40)
RAW_YEARS_DEFAULT = 40
RAW_WINDOW_MIN = 2010
RAW_BINS_RANGE = range(3, 11)
RAW_BINS_DEFAULT = 5
RAW_TREND_DEFAULT = "power"
RAW_FAILED = "Raw simulation failed: no fan drawn"

logger = logging.getLogger(__name__)

# Legend wording: "MC median · raw cycle, power law, 2010–2026"
_VARIANT_NAME = {"raw_iid": "raw independent", "raw_cycle": "raw cycle"}
_TREND_NAME = {"power": "power law", "exp": "exponential decay", "none": "no trend"}


def raw_bins(v) -> int:
    b = _ci_or(v, RAW_BINS_DEFAULT)
    return b if b in RAW_BINS_RANGE else RAW_BINS_DEFAULT


def raw_window(v) -> tuple[int, int]:
    yr = pd.Timestamp.today().year
    try:
        a, b = sorted(int(float(x)) for x in v)
    except (TypeError, ValueError, OverflowError):
        return (RAW_WINDOW_MIN, yr)
    return (min(max(a, RAW_WINDOW_MIN), yr), min(max(b, RAW_WINDOW_MIN), yr))


def raw_trend(v) -> str:
    return v if v in raw_mc.RAW_TRENDS else RAW_TREND_DEFAULT


def raw_controls(sims, years, window, trend, bins, regime, lines) -> dict:
    """Coerce the Raw Inputs; anything a stale or hostile client sends falls
    back to the layout default (0 is a valid sims value and clamps to 1)."""
    b = raw_bins(bins)
    y = _ci_or(years, RAW_YEARS_DEFAULT)
    if isinstance(regime, (list, tuple)):
        checked = []
        for i in regime:
            try:
                checked.append(int(i))
            except (TypeError, ValueError, OverflowError):
                pass
        checked = tuple(checked)
    else:
        checked = tuple(range(b))
    return dict(
        sims=_ci_or(sims, RAW_SIMS_DEFAULT, 1, raw_mc.RAW_SIMS_MAX),
        years=y if y in RAW_YEARS else RAW_YEARS_DEFAULT,
        window=raw_window(window),
        trend=raw_trend(trend),
        bins=b,
        checked=checked,
        lines=_ci_or(lines, raw_mc.RAW_LINES_DEFAULT,
                     raw_mc.RAW_LINES_MIN, raw_mc.RAW_LINES_MAX),
    )


_CACHE_MAX = 2
_cache: OrderedDict = OrderedDict()     # RawRequest -> (ok, status, paths, t_axis)
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


raw_mc._CACHE_CLEARERS.append(clear_cache)


def _frozen(a: np.ndarray, dtype=None) -> np.ndarray:
    a = np.array(a, dtype=dtype)              # always a private copy
    a.setflags(write=False)
    return a


def _simulate_cached(req):
    """``(ok, status, paths, t_axis)`` for ``req``; arrays are read-only."""
    with _cache_lock:
        hit = _cache.get(req)
        if hit is not None:
            _cache.move_to_end(req)
            return hit
    res = raw_mc.simulate(req)
    entry = (res.ok, tuple(res.status), _frozen(res.paths, np.float32),
             _frozen(res.t_axis))
    del res                                   # never hold a RawResult (ALARA)
    with _cache_lock:
        _cache[req] = entry
        _cache.move_to_end(req)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return entry


def _status_children(lines: list[str]):
    return [html.Div(s) for s in lines]


def add_raw_overlay(fig, *, variant, price_data, xrange, stack, show_stack,
                    mc_paths, mc_extremes, sims, years, window, trend, bins,
                    regime, lines):
    """Simulate and draw the Raw fan (+ lines / extremes) onto a clone of
    ``fig``. Returns ``(figure, raw-status children)``; a refused run returns
    the figure untouched with the reason in the status, and so does a run that
    raises (logged, status ``RAW_FAILED``) -- this never raises itself."""
    from figures.bubble import (_add_mc_fan, _add_mc_spaghetti,
                                _add_mc_extremes, select_line_paths)
    c = raw_controls(sims, years, window, trend, bins, regime, lines)
    try:
        live = float(price_data)
    except (TypeError, ValueError):
        live = 0.0
    req = None
    try:
        series, data_fp = raw_mc.get_prices()
        # F-23 N-4: fall back to the registry's last close, the series the fan uses
        p0 = live if np.isfinite(live) and live > 0 else float(series.iloc[-1])
        req = raw_mc.RawRequest(
            variant=variant, trend=c["trend"], window=c["window"], bins=c["bins"],
            checked=c["checked"], sims=c["sims"], years=c["years"],
            today=pd.Timestamp.today().normalize(), p0=p0, data_fp=data_fp,
            asof=None)
        ok, status, paths, t_axis = _simulate_cached(req)
    except Exception:                         # noqa: BLE001 -- final review M-7
        # A Raw fault must not 500 update_bubble (that freezes all of Tab 1):
        # draw the chart without a fan, say so, and log it.
        logger.exception("Raw simulation failed for %r", req)
        return fig, _status_children([RAW_FAILED])
    status = list(status)
    if not ok or paths.size == 0:
        return fig, _status_children(status)

    # Trim at the visible right edge, one step past it (as the model branch).
    n_steps = paths.shape[1]
    t_hi = yr_to_t(int(xrange[1]), _app_ctx.M.genesis)
    keep = int(np.searchsorted(t_axis, t_hi, side="left")) + 1
    keep = min(n_steps, max(keep, 2))
    t_axis = t_axis[:keep]
    paths = paths[:, :keep]
    stack_f = _cf(stack, 0)
    if "yes" in (show_stack or []) and stack_f > 0:
        paths = paths * stack_f

    import plotly.graph_objects as go
    fig = go.Figure(fig)                      # never mutate the cached figure
    w0, w1 = c["window"]
    suffix = (f" · {_VARIANT_NAME[variant]}, {_TREND_NAME[c['trend']]}, "
              f"{w0}–{w1}")
    _add_mc_fan(fig, paths, t_axis, label_suffix=suffix)
    if isinstance(mc_paths, list) and "yes" in mc_paths:
        idx = select_line_paths(paths, c["lines"])
        _add_mc_spaghetti(fig, paths[idx], t_axis, n_display=len(idx))
        if isinstance(mc_extremes, list) and "yes" in mc_extremes:
            _add_mc_extremes(fig, paths, t_axis)
    return fig, _status_children(status)
