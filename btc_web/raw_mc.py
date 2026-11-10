"""Raw (model-free) Monte Carlo for Tab 1: history, cycle table, trend fits.

Spec: docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md ("Data" and
"Trend curves").  This module imports only numpy, pandas, scipy and the
standard library -- never ``figures``, ``_app_ctx`` or ``mc_cache`` -- so the
command-line tool tools/raw_mc_cycle_stats.py can use it without loading the
model stack.

Definitions (shared with the CLI):

  * monthly log returns of month-end closes; the incomplete trailing month
    is dropped;
  * a cycle runs halving to halving (genesis 2009-01-03 is epoch 0); its
    months are its IN-WINDOW months with price data; it is FULL when it
    ends at a known halving and has >= 24 such months, and COMPLETE (usable
    as a cycle-mode block source) when, in addition, those months span
    cycle fraction phi < 0.05 to phi > 0.95;
  * fit set (fit_set): the full cycles in the window if there are >= 3,
    else every full cycle of the whole history;
  * t_c = mean t (years since 2009-07-25, month midpoints) of its months;
    g_c = annual log growth = 12 * mean monthly log return;
    s_c = annual vol = sqrt(12) * std of monthly log returns;
  * power law: g = n / t (unweighted LS, n = sum(g/t) / sum(1/t^2));
    vol = a * t^-b (LS of log s on log t);
  * exponential: g = exp(alpha + beta t), s = exp(gamma + delta t), fitted
    directly in g / s space (scipy curve_fit).

Prices enter through one registry (``set_prices`` / ``get_prices`` /
``months_for``); nothing else in production calls ``load_months`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── calendar ────────────────────────────────────────────────────────────────

ORIGIN = pd.Timestamp("2009-07-25")          # the project's time origin (t = 0)
GENESIS = pd.Timestamp("2009-01-03")         # epoch 0 (genesis block)

# Copied, not imported, from figures.common._HALVING_KNOWN_DATES (that module
# pulls in btc_core); test_halvings_match_figures_common keeps them equal.
_HALVING_KNOWN_DATES = ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")
_HALVING_ANCHOR = pd.Timestamp("2024-04-20")
HALVINGS: tuple[pd.Timestamp, ...] = (GENESIS,
                                      *(pd.Timestamp(d) for d in _HALVING_KNOWN_DATES))

MIN_FULL_MONTHS = 24         # a cycle needs this many in-window months to be full
MIN_OWN_STD_MONTHS = 12      # fewer months: standardise on the trend (F-22 N-4)
_TABLE_COLS = ["cycle", "first", "last", "months", "full", "complete", "t", "g", "s"]


def future_halvings(until: pd.Timestamp) -> list[pd.Timestamp]:
    """Estimated halvings after 2024-04-20: anchor + 4*k years, for every k >= 1
    whose date is <= ``until`` (the rule of figures.common._build_halving_epochs)."""
    out, k = [], 1
    while True:
        dt = _HALVING_ANCHOR + pd.DateOffset(years=4 * k)
        if dt > until:
            return out
        out.append(dt)
        k += 1


def t_of(ts):
    """Years since 2009-07-25 (whole days / 365.25).  A scalar timestamp gives
    a float, an array-like of timestamps an ndarray."""
    if np.ndim(ts) == 0:
        return (pd.Timestamp(ts) - ORIGIN).days / 365.25
    return np.asarray((pd.DatetimeIndex(ts) - ORIGIN).days, dtype=float) / 365.25


# ── history ─────────────────────────────────────────────────────────────────

def load_months(prices: pd.Series, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Monthly log returns of month-end closes.

    Columns ``r`` (log return), ``t`` (years since ORIGIN at the month's
    midpoint), ``cycle`` (the halving interval the month-END falls in; the
    genesis block is epoch 0, so months before 2012-11-28 are cycle 1) and
    ``phi`` (fraction of that cycle elapsed at the midpoint, clipped to
    [0, 1) -- a month can read slightly negative when its month-end and
    midpoint straddle a halving).  Index = month-end.  The incomplete
    trailing month is dropped.  ``asof`` truncates ``prices`` first (the
    frozen-input hook for tests; None in production).
    """
    s = prices if asof is None else prices[prices.index <= asof]
    me = s.resample("ME").last()
    if s.index[-1] < me.index[-1]:               # incomplete trailing month
        me = me.iloc[:-1]
    r = np.log(me).diff().dropna()
    mid = r.index - pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    epochs = list(HALVINGS) + future_halvings(max(r.index[-1], HALVINGS[-1])
                                              + pd.DateOffset(years=4))
    ep = np.array(epochs, dtype="datetime64[ns]")
    cyc = np.searchsorted(ep[:-1], r.index.values, side="right")
    start, end = ep[cyc - 1], ep[cyc]
    phi = ((mid.values - start) / (end - start)).astype(float)
    phi = np.clip(phi, 0.0, np.nextafter(1.0, 0.0))
    return pd.DataFrame({"r": r.values, "t": t_of(mid), "cycle": cyc, "phi": phi},
                        index=r.index)


def _in_window(months: pd.DataFrame, window: tuple[int, int] | None) -> pd.DataFrame:
    """Months whose month-end year lies in ``window`` (inclusive); None = all."""
    if window is None:
        return months
    y0, y1 = window
    return months[(months.index.year >= y0) & (months.index.year <= y1)]


def cycle_table(months: pd.DataFrame, window: tuple[int, int] | None) -> pd.DataFrame:
    """One row per halving cycle from the in-window months: first/last month,
    month count, ``full`` / ``complete`` flags, mean ``t`` and the annual
    ``g`` (12 * mean r) and ``s`` (sqrt(12) * std r)."""
    rows = []
    for c, g in _in_window(months, window).groupby("cycle"):
        full = bool(c < len(HALVINGS) and len(g) >= MIN_FULL_MONTHS)
        complete = bool(full and g.phi.min() < 0.05 and g.phi.max() > 0.95)
        rows.append(dict(cycle=int(c), first=g.index[0].strftime("%Y-%m"),
                         last=g.index[-1].strftime("%Y-%m"), months=len(g),
                         full=full, complete=complete,
                         t=g.t.mean(), g=12 * g.r.mean(),
                         s=np.sqrt(12) * g.r.std()))
    return pd.DataFrame(rows, columns=_TABLE_COLS)


def fit_set(tab_window: pd.DataFrame, tab_all: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Cycles the trend curves are fitted on: the full cycles in the window if
    there are >= 3 (``"window"``), else every full cycle of the whole history
    (``"all history"``)."""
    w = tab_window[tab_window.full]
    return (w, "window") if len(w) >= 3 else (tab_all[tab_all.full], "all history")


# ── trend curves ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Trend:
    """A fitted growth / volatility curve pair, in annual natural-log units.

    ``kind`` is ``"power"`` (params n, a, b), ``"exp"`` (alpha, beta, gamma,
    delta) or ``"none"`` (mu_m, sd_m: the window's pooled MONTHLY mean and
    std, constant in t).  The simulation uses the monthly forms.
    """
    kind: str
    params: dict

    def g(self, t):
        """Annual log growth at time t (years since ORIGIN)."""
        p = self.params
        if self.kind == "power":
            return p["n"] / t
        if self.kind == "exp":
            return np.exp(p["alpha"] + p["beta"] * t)
        return 12 * p["mu_m"] + 0.0 * np.asarray(t, dtype=float)

    def s(self, t):
        """Annual volatility at time t."""
        p = self.params
        if self.kind == "power":
            return p["a"] * t ** -p["b"]
        if self.kind == "exp":
            return np.exp(p["gamma"] + p["delta"] * t)
        return np.sqrt(12) * p["sd_m"] + 0.0 * np.asarray(t, dtype=float)

    def mu_m(self, t):
        return self.g(t) / 12

    def sd_m(self, t):
        return self.s(t) / np.sqrt(12)


def fit_trend(kind: str, fset: pd.DataFrame, months: pd.DataFrame,
              window: tuple[int, int] | None) -> Trend | None:
    """Fit ``kind`` (``"power"`` | ``"exp"`` | ``"none"``) to the fit set.

    Returns None when the curve cannot be fitted: ``curve_fit`` raised, or the
    fit set has fewer than 2 cycles (power / exp).  ``"none"`` needs no fit
    set: it carries the pooled monthly mean and std of the window's months.
    """
    if kind == "none":
        r = _in_window(months, window).r
        return Trend("none", dict(mu_m=float(r.mean()), sd_m=float(r.std())))
    if kind not in ("power", "exp"):
        raise ValueError(f"unknown trend kind {kind!r}")
    if len(fset) < 2:
        return None
    t, g, s = fset.t.values, fset.g.values, fset.s.values
    if kind == "power":
        n = np.sum(g / t) / np.sum(1 / t ** 2)
        nb, la = np.polyfit(np.log(t), np.log(s), 1)
        return Trend("power", dict(n=float(n), a=float(np.exp(la)), b=float(-nb)))

    def curve(x, a, b):
        return np.exp(a + b * x)
    try:
        (al, be), _ = curve_fit(curve, t, g, maxfev=20000,
                                p0=(np.log(max(g.mean(), 1e-3)), -0.1))
        (ga, de), _ = curve_fit(curve, t, s, maxfev=20000,
                                p0=(np.log(max(s.mean(), 1e-3)), -0.05))
    except (RuntimeError, ValueError):
        return None
    return Trend("exp", dict(alpha=float(al), beta=float(be),
                             gamma=float(ga), delta=float(de)))


def standardise(months: pd.DataFrame, trend: Trend) -> pd.Series:
    """Shape ``z`` of each month: ``(r - mu) / sigma`` with the cycle's own
    sample mean and std, except that a cycle with fewer than
    ``MIN_OWN_STD_MONTHS`` months (the thin partial current cycle: its std is
    undefined at 1 month and unstable below ~12) borrows the trend's monthly
    mu and sigma at each month's own t (F-22 N-4).  With ``Trend("none")``
    those are the window's pooled monthly mean and std.  ``months`` should be
    the in-window months."""
    grp = months.groupby("cycle").r
    own = (months.r - grp.transform("mean")) / grp.transform("std")
    t = months.t.to_numpy()
    borrowed = (months.r - trend.mu_m(t)) / trend.sd_m(t)
    return own.where(grp.transform("size") >= MIN_OWN_STD_MONTHS, borrowed)


# ── price registry ──────────────────────────────────────────────────────────
#
# Production registers the series once at startup (btc_web/app.py).  Tests use
# the ``raw_prices`` fixture (raw_mc_testing.py), which restores the previous
# state on teardown.

_UNSET = object()            # the "nothing registered" state of ``_registry``
_registry = _UNSET           # (series, data_fp) or _UNSET
_months_cache: dict[tuple[str, pd.Timestamp | None], pd.DataFrame] = {}
# Functions called whenever the registered prices change, so derived caches
# elsewhere in this module can be dropped (e.g. regime_options.cache_clear).
_CACHE_CLEARERS: list[Callable[[], None]] = []


def _set_registry(value) -> None:
    """Replace the registry (a ``(series, data_fp)`` tuple or ``_UNSET``) and
    drop everything derived from the old one."""
    global _registry
    _registry = value
    _months_cache.clear()
    for clear in _CACHE_CLEARERS:
        clear()


def set_prices(series: pd.Series, data_fp: str) -> None:
    """Register the daily close series (stored as given, not copied) and its
    data fingerprint; clears every cache derived from the previous one."""
    _set_registry((series, data_fp))


def get_prices() -> tuple[pd.Series, str]:
    if _registry is _UNSET:
        raise RuntimeError("raw_mc prices not registered")
    return _registry


def months_for(asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """``load_months`` of the registered series, cached per ``(data_fp, asof)``.
    The returned frame is shared -- callers must not mutate it."""
    series, fp = get_prices()
    key = (fp, asof)
    got = _months_cache.get(key)
    if got is None:
        got = _months_cache[key] = load_months(series, asof)
    return got
