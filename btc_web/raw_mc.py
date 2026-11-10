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

import functools
import hashlib
import json
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


# ── simulator ───────────────────────────────────────────────────────────────
#
# Spec "Generating a path" + plan "F-22 decisions".  Step k spans
# [d_k, d_{k+1}), d_k = today + k calendar months; it belongs to the cycle
# containing d_k and takes phi / mu at its midpoint d_k + 15 days.  sigma is
# held per cycle at the trend's value at the cycle's midpoint t (the current
# cycle: the midpoint of [today, next halving)).

RAW_SRCS = ("raw_iid", "raw_cycle")
RAW_TRENDS = ("power", "exp", "none")
RAW_SIMS_MAX = 3200
RAW_MIN_MONTHS = 36
BLOCK = 3                     # cycle-mode block length (steps / history months)
PHASE_TOLS = (0.04, 0.08)     # then: every complete-cycle block of that length
_STEP_MID = pd.Timedelta(days=15)


@dataclass(frozen=True)
class RawRequest:
    variant: str                       # one of RAW_SRCS
    trend: str                         # one of RAW_TRENDS
    window: tuple[int, int]            # month-end years, inclusive
    bins: int
    checked: tuple[int, ...]
    sims: int
    years: int
    today: pd.Timestamp
    p0: float                          # start price (NOT in the seed)
    data_fp: str
    asof: pd.Timestamp | None = None   # frozen-input hook; None in production


@dataclass
class RawResult:
    ok: bool
    reason: str
    t_axis: np.ndarray
    paths: np.ndarray                  # (sims, steps + 1), prices
    status: list[str]
    schedule: list                     # cycle mode: one dict per grid block
    z_shift: float
    sigma_steps: np.ndarray            # (steps,)
    unchecked_share: float
    bin_labels: list[str]
    edges: np.ndarray | None
    states: np.ndarray | None          # (sims, steps + 1), filter runs only
    trend: Trend | None
    fit_where: str
    # filtered cycle runs: (sims, steps) in-window month index each step copied
    sources: np.ndarray | None = None


def effective_checked(checked, bins: int) -> tuple[int, ...] | None:
    """The regime filter actually in force: sorted, de-duplicated, valid bin
    indices -- or None when there is no filter (nothing valid checked, or
    every bin checked)."""
    valid = sorted({int(i) for i in checked if 0 <= int(i) < bins})
    return None if not valid or len(valid) == bins else tuple(valid)


def seed_for(req: RawRequest) -> int:
    """Stable RNG seed: blake2b (8 bytes) of canonical JSON of every request
    field except ``p0``, with ``checked`` replaced by ``effective_checked``."""
    def day(ts):
        return None if ts is None else pd.Timestamp(ts).isoformat()
    eff = effective_checked(req.checked, req.bins)
    doc = dict(variant=req.variant, trend=req.trend, window=list(req.window),
               bins=int(req.bins), checked=None if eff is None else list(eff),
               sims=int(req.sims), years=int(req.years),
               today=day(None if req.today is None
                         else pd.Timestamp(req.today).normalize()),
               data_fp=req.data_fp, asof=day(req.asof))
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "big")


def _step_dates(today: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """d_0 .. d_n: today + k calendar months (day clamped to month length)."""
    return pd.DatetimeIndex([today + pd.DateOffset(months=k) for k in range(n + 1)])


def _grid(cyc: np.ndarray, k0: int, k1: int) -> list[tuple[int, int, int]]:
    """Blocks (cycle, start, stop) over steps [k0, k1): runs of BLOCK steps
    restarting at each cycle's first step; a cycle's last block may be short."""
    out, k = [], k0
    while k < k1:
        stop = k + 1
        while stop < k1 and stop - k < BLOCK and cyc[stop] == cyc[k]:
            stop += 1
        out.append((int(cyc[k]), k, stop))
        k = stop
    return out


def _pct(g: float) -> str:
    """Annual log growth as a signed compound %/yr."""
    return f"{np.expm1(g) * 100:+.0f} %/yr".replace("-", "−")


def _num(x: float) -> str:
    return f"{x:.2f}".replace("-", "−")


def _refuse(reason: str, status: list[str], trend=None, where="window") -> RawResult:
    return RawResult(False, reason, np.empty(0), np.empty((0, 0)), status + [reason],
                     [], 0.0, np.empty(0), 0.0, [], None, None, trend, where)


def _trend_line(trend: Trend, n_fit: int, where: str) -> str:
    p = trend.params
    fitted = f"fitted on {n_fit} cycles ({where})"
    if trend.kind == "power":
        return f"Power law n = {_num(p['n'])}, vol ∝ t^{_num(-p['b'])}, {fitted}"
    if trend.kind == "exp":
        return (f"Exponential growth e^({_num(p['alpha'])} {_num(p['beta'])}·t), "
                f"vol e^({_num(p['gamma'])} {_num(p['delta'])}·t), {fitted}")
    return "No trend: honest replay of raw monthly returns"


def _fit_for(months: pd.DataFrame, window, kind: str):
    """(trend or None, fit set, where): the trend ``simulate`` uses for
    ``kind`` on ``window`` -- shared with the regime labels so both see the
    same curve."""
    tab_w, tab_all = cycle_table(months, window), cycle_table(months, None)
    fset, where = fit_set(tab_w, tab_all)
    if kind == "none":
        where = "window"
    return fit_trend(kind, fset, months, window), fset, where


# ── regime filter: level relative to trend ──────────────────────────────────
#
# Spec "Regime filter (Raw)".  Historical D_k = log P_k - sum_{j<=k} mu(t_j) - C
# over the eligible (in-window) months, C making mean(D) = 0; state = D / S,
# S = std(D).  Simulated D_{k+1} = D_k + r_{k+1} - mu(t_{k+1}) from D_0 (plan
# F-22 N-3).  Bins are equal-count over the historical state, outer edges the
# historical min / max, all bins CLOSED: a state outside every checked bin's
# closed range is "unchecked".

MAX_TRIES = 20               # draws per steered month (first + 19 redraws)
STEER_SKIP = 12              # months excluded from ``unchecked_share``


def _d_series(months: pd.DataFrame, trend: Trend, window) -> tuple[np.ndarray, int]:
    """Uncentred D (cumsum of r - mu) from the first eligible month through the
    LAST month of ``months`` (the curve continued past the window), and the
    number of eligible (in-window) months at its head."""
    win = _in_window(months, window)
    pos0 = months.index.get_loc(win.index[0])
    tail = months.iloc[pos0:]
    assert tail.index[len(win) - 1] == win.index[-1]          # window is contiguous
    return np.cumsum(tail.r.to_numpy() - trend.mu_m(tail.t.to_numpy())), len(win)


def state_history(months: pd.DataFrame, trend: Trend, window) -> tuple[np.ndarray, float]:
    """Historical state (D / S) of each eligible month, and S."""
    d, n = _d_series(months, trend, window)
    d = d[:n] - d[:n].mean()
    S = float(d.std())
    return d / S, S


def _d0(months: pd.DataFrame, trend: Trend, window, today: pd.Timestamp,
        p0: float) -> float:
    """F-22 N-3: D_0 = D_lastME + log(P_today / P_lastME) - mu(t_today) * f,
    f = days since the last month-end / days in today's month (the fraction of
    the current month elapsed)."""
    d, n = _d_series(months, trend, window)
    d_last = d[-1] - d[:n].mean()
    last_me = months.index[-1]
    series = get_prices()[0]
    p_last = float(series[series.index <= last_me].iloc[-1])
    f = (today - last_me).days / today.days_in_month
    return float(d_last + np.log(p0 / p_last) - trend.mu_m(t_of(today)) * f)


def bin_edges(states: np.ndarray, bins: int) -> np.ndarray:
    """``bins + 1`` equal-count edges; the outer two are the historical
    min and max state exactly."""
    edges = np.quantile(states, np.linspace(0.0, 1.0, int(bins) + 1))
    edges[0], edges[-1] = states.min(), states.max()
    return edges


def _sigma(x: float) -> str:
    x = round(float(x), 1)
    if x == 0:
        return "0.0σ"
    return f"{x:+.1f}σ".replace("-", "−")


def bin_labels(edges: np.ndarray) -> list[str]:
    """One label per bin, coldest first, e.g. ``"−1.5σ to −0.8σ vs trend"``."""
    return [f"{_sigma(a)} to {_sigma(b)} vs trend" for a, b in zip(edges[:-1], edges[1:])]


@functools.lru_cache(maxsize=64)
def _regime_options(bins: int, window: tuple[int, int], trend: str,
                    asof: pd.Timestamp | None, data_fp: str) -> tuple[str, ...]:
    """Bin labels for the checklist; () when no trend / state can be built.
    ``trend`` is the kind string (Trend objects are unhashable); ``data_fp``
    is in the key only so a new series never hits a stale entry."""
    months = months_for(asof)
    if len(_in_window(months, window)) < RAW_MIN_MONTHS:
        return ()
    tr, _, _ = _fit_for(months, window, trend)
    if tr is None:
        return ()
    return tuple(bin_labels(bin_edges(state_history(months, tr, window)[0], bins)))


def regime_options(bins, window, trend: str, asof=None) -> list[dict]:
    """Dash checklist options ``{"label", "value"}`` for the ``bins``
    trend-deviation bins (values 0..bins-1, coldest first), from the same
    ``months_for`` series and trend fit ``simulate`` uses.  Fresh dicts per
    call; the labels are cached per ``(bins, window, trend, asof, data_fp)``."""
    labels = _regime_options(int(bins), tuple(int(y) for y in window), str(trend),
                             asof, get_prices()[1])
    return [{"label": lab, "value": i} for i, lab in enumerate(labels)]


regime_options.cache_clear = _regime_options.cache_clear
_CACHE_CLEARERS.append(regime_options.cache_clear)


class _Steer:
    """Distance of a state to the union of the checked bins' closed ranges
    (adjacent checked bins merged into one interval)."""

    def __init__(self, edges: np.ndarray, checked: tuple[int, ...]):
        runs: list[list[int]] = []
        for i in checked:                       # sorted (effective_checked)
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        self.lo = np.array([edges[a] for a, _ in runs])
        self.hi = np.array([edges[b] for _, b in runs])

    def outside(self, x: np.ndarray) -> np.ndarray:
        """True where ``x`` is in no checked bin (dist > 0)."""
        out = np.ones(x.shape, bool)
        for lo, hi in zip(self.lo, self.hi):
            out &= (x < lo) | (x > hi)
        return out

    def dist(self, x: np.ndarray) -> np.ndarray:
        out = np.full(x.shape, np.inf)
        for lo, hi in zip(self.lo, self.hi):
            np.minimum(out, np.maximum(np.maximum(lo - x, x - hi), 0.0), out=out)
        return out


def _steered_month(steer: _Steer, x_prev: np.ndarray, d: np.ndarray, scale: float,
                   off: float, redraw: Callable[[np.ndarray, int], tuple]) -> tuple:
    """Steer one month for all paths.  ``d`` holds the first try's draw (shape
    (sims,)); the state moves by ``d * scale - off``.  Rows landing outside the
    checked bins get the other MAX_TRIES - 1 tries in one batch:
    ``redraw(rows, n)`` returns ``(draws, tags)``, each ``(len(rows), n)``,
    ``tags`` (or None) travelling with each try (the block candidate in cycle
    mode).  A row keeps its first accepted try, else its closest one -- the
    same choice as redrawing one try at a time.  Returns
    ``(d, x_new, rows, tags)``: the rows whose kept draw is a redraw, and that
    redraw's tag (tags None when ``redraw`` gives none)."""
    x = x_prev + d * scale - off
    dist = steer.dist(x)
    bad = np.flatnonzero(dist > 0)
    if not len(bad):
        return d, x, bad, bad              # nothing redrawn: empty rows and tags
    d2, tag = redraw(bad, MAX_TRIES - 1)
    x2 = x_prev[bad, None] + d2 * scale - off
    dd = steer.dist(x2)
    hit = dd == 0
    best = np.where(hit.any(axis=1), hit.argmax(axis=1), dd.argmin(axis=1))
    ar = np.arange(len(bad))
    take = dd[ar, best] < dist[bad]
    rows, best = bad[take], best[take]
    d[rows], x[rows] = d2[ar[take], best], x2[ar[take], best]
    return d, x, rows, None if tag is None else tag[ar[take], best]


def simulate(req: RawRequest) -> RawResult:
    """Bootstrap ``req.sims`` monthly paths from today for ``req.years``.

    Trend power / exp: ``r = mu_m(t_mid) + sigma_cycle * (z - z_shift)``,
    ``z`` a bootstrapped historical shape (``standardise``).  Trend none:
    the same draws of raw ``r`` (honest replay).  Independent mode draws one
    eligible month per step; cycle mode walks the per-cycle block grid and
    copies runs of consecutive months from one complete cycle whose start
    phase matches the block's.

    Regime filter (``effective_checked`` not None): each path's level-vs-trend
    state is carried month by month and a draw that would leave the checked
    bins is redrawn (independent: that month; cycle: the block from that month
    on, same grid slot), up to MAX_TRIES, keeping the closest try.  Unfiltered
    runs keep the fully vectorised draw and its RNG stream.
    """
    assert req.data_fp == get_prices()[1], "stale data_fp"
    if req.variant not in RAW_SRCS or req.trend not in RAW_TRENDS:
        raise ValueError(f"bad variant/trend {req.variant!r}/{req.trend!r}")
    months = months_for(req.asof)
    win = _in_window(months, req.window)
    status: list[str] = []
    if len(win) < RAW_MIN_MONTHS:
        return _refuse(f"Fewer than {RAW_MIN_MONTHS} months of history in the window "
                       f"({len(win)}): no fan drawn", status)

    trend, fset, where = _fit_for(months, req.window, req.trend)
    if trend is None:
        name = "Exponential" if req.trend == "exp" else "Power-law"
        return _refuse(f"{name} trend could not be fitted to these cycles", status,
                       None, where)
    status.append(_trend_line(trend, len(fset), where))

    tab_w = cycle_table(months, req.window)
    complete = tab_w[tab_w.complete]
    cycle_mode = req.variant == "raw_cycle"
    if cycle_mode and len(complete) < 2:
        return _refuse("Halving-cycle mode needs at least 2 complete cycles in the "
                       f"window ({len(complete)} found): no fan drawn", status,
                       trend, where)

    # ── step calendar ──
    steps = int(req.years) * 12
    today = pd.Timestamp(req.today).normalize()
    epochs = list(HALVINGS) + future_halvings(today + pd.DateOffset(years=int(req.years) + 9))
    ep = np.array(epochs, dtype="datetime64[ns]")
    cyc_now = int(np.searchsorted(ep, today.to_datetime64(), side="right"))
    # extend past the horizon to cover the first whole future cycle (debias)
    first_whole_end = epochs[cyc_now + 1]
    n_dates = max(steps, int(np.ceil((first_whole_end - today).days / 28)) + 1)
    dates = _step_dates(today, n_dates)
    dv = dates.values
    cyc = np.searchsorted(ep, dv[:-1], side="right")         # cycle of d_k
    start, end = ep[cyc - 1], ep[cyc]
    start = np.where(cyc == cyc_now, today.to_datetime64(), start)  # current: remaining span
    mid = dv[:-1] + _STEP_MID.to_timedelta64()
    phi = np.clip(((mid - ep[cyc - 1]) / (end - ep[cyc - 1])).astype(float),
                  0.0, np.nextafter(1.0, 0.0))
    t_mid = t_of(pd.DatetimeIndex(mid))
    sigma_all = trend.sd_m((t_of(pd.DatetimeIndex(start)) + t_of(pd.DatetimeIndex(end))) / 2)
    mu = trend.mu_m(t_mid)[:steps]
    sigma_steps = np.asarray(sigma_all[:steps], dtype=float)

    # ── draw pool: in-window months, positional ──
    raw = win.r.to_numpy()
    zs = standardise(win, trend).to_numpy()
    replay = req.trend == "none"
    pool = raw if replay else zs
    rng = np.random.default_rng(seed_for(req))
    sims = int(req.sims)
    schedule: list = []
    z_shift = 0.0

    # regime filter: state increment per step = draw * scale[k] - off[k]
    eff = effective_checked(req.checked, req.bins)
    steer = states = edges = None
    labels: list[str] = []
    if eff is not None:
        hist, S = state_history(months, trend, req.window)
        edges = bin_edges(hist, req.bins)
        labels = bin_labels(edges)
        steer = _Steer(edges, eff)
        scale = (np.ones(steps) if replay else sigma_steps) / S
        off = (mu if replay else np.zeros(steps)) / S
        # step-major (steps + 1, sims): the loops write one contiguous row per month
        states = np.empty((steps + 1, sims))
        states[0] = _d0(months, trend, req.window, today, req.p0) / S

    if not cycle_mode and steer is None:
        draws = pool[rng.integers(0, len(pool), size=(sims, steps))]
    elif not cycle_mode:
        draws = np.empty((steps, sims))

        def redraw(rows, n):
            return pool[rng.integers(0, len(pool), size=(len(rows), n))], None
        for k in range(steps):
            d = pool[rng.integers(0, len(pool), size=sims)]
            draws[k], states[k + 1], _, _ = _steered_month(
                steer, states[k], d, scale[k], off[k], redraw)
    else:
        wcyc, wphi = win.cycle.to_numpy(), win.phi.to_numpy()
        comp = set(complete.cycle.astype(int))
        cands = {}
        for L in range(1, BLOCK + 1):
            s = np.arange(len(win) - L + 1)
            ok = np.isin(wcyc[s], list(comp)) & (wcyc[s] == wcyc[s + L - 1])
            s = s[ok]
            cands[L] = (s, wphi[s], wcyc[s])

        def candidates(L, ph):
            s, sp, sc = cands[L]
            for level, tol in enumerate(PHASE_TOLS):
                m = np.abs(sp - ph) <= tol
                if len(np.unique(sc[m])) >= 2:
                    return s[m], level
            return s, len(PHASE_TOLS)

        def block_sum(cand, L):
            return float(np.mean([zs[c:c + L].sum() for c in cand]))

        # debias from the first whole future cycle's grid (F-22 N-1)
        k_w = np.flatnonzero(cyc == cyc_now + 1)
        assert len(k_w) == 48, len(k_w)
        e = 0.0
        if not replay:
            for _, k0, k1 in _grid(cyc, k_w[0], k_w[-1] + 1):
                cand, _ = candidates(k1 - k0, phi[k0])
                e += block_sum(cand, k1 - k0)
            z_shift = e / len(k_w)

        draws = np.empty((sims, steps)) if steer is None else np.empty((steps, sims))
        sources = None if steer is None else np.empty((steps, sims), dtype=np.int64)
        widened = [0] * len(PHASE_TOLS)
        for c, k0, k1 in _grid(cyc, 0, steps):
            L = k1 - k0
            cand, lvl = candidates(L, phi[k0])
            if lvl:
                widened[lvl - 1] += 1
            schedule.append(dict(cycle=c, steps=(k0, k1), cand=cand, widened=lvl))
            pick = cand[rng.integers(0, len(cand), size=sims)]
            if steer is None:
                draws[:, k0:k1] = pool[pick[:, None] + np.arange(L)] - z_shift
                continue
            # Month by month; a violation redraws the block from that month on,
            # from this slot's whole candidate set at the same offset.  Block
            # integrity then holds per contiguous run (each run copies one
            # candidate, inside one complete cycle), not per block: restricting
            # suffix redraws to the prefix's cycle broke the 5 % steering bound
            # (review I-1, fallback ruling; measurements in task-3-report.md).
            for j in range(L):
                k = k0 + j

                def redraw(rows, n, j=j, cand=cand):
                    c2 = cand[rng.integers(0, len(cand), size=(len(rows), n))]
                    return pool[c2 + j] - z_shift, c2
                d = pool[pick + j] - z_shift
                draws[k], states[k + 1], rows, tags = _steered_month(
                    steer, states[k], d, scale[k], off[k], redraw)
                pick[rows] = tags           # later months of the block follow the redraw
                sources[k] = pick + j

    if steer is not None:
        draws = draws.T                                    # (sims, steps)
    if steer is not None and cycle_mode:
        sources = np.ascontiguousarray(sources.T)          # (sims, steps)
    else:
        sources = None
    r = draws if replay else mu + sigma_steps * draws
    share = 0.0
    if steer is not None:
        share = float(np.mean(steer.outside(states[STEER_SKIP + 1:])))
        states = np.ascontiguousarray(states.T)            # (sims, steps + 1)
    paths = np.empty((sims, steps + 1))
    paths[:, 0] = 0.0
    np.cumsum(r, axis=1, out=paths[:, 1:])
    paths = req.p0 * np.exp(paths)

    # ── status ──
    cur = win[win.cycle == cyc_now]
    if replay:
        hist_g = 12 * raw.mean()
        if len(cur):
            status.append(f"Current cycle ({len(cur)} months) grows "
                          f"{_pct(12 * cur.r.mean())} vs history {_pct(hist_g)}")
        fan_g = np.median(np.log(paths[:, -1] / req.p0)) / int(req.years)
        status.append(f"Fan median grows {_pct(fan_g)} over {int(req.years)} years "
                      f"(history {win.index[0].year}–{win.index[-1].year} grew "
                      f"{_pct(hist_g)}); the fan compounds replayed history.")
    elif len(cur):
        status.append(f"Current cycle ({len(cur)} months) grows "
                      f"{_pct(12 * cur.r.mean())} vs curve {_pct(trend.g(cur.t.mean()))}")
    else:
        status.append("Current cycle: no complete months yet")
    if cycle_mode:
        spans = ", ".join(f"{HALVINGS[c - 1].year}–{HALVINGS[c].year}"
                          for c in complete.cycle.astype(int))
        status.append(f"Halving-cycle mode replays only the complete past cycles in "
                      f"the window: {spans}")
        status.append(f"Phase tolerance widened on {sum(widened)} of {len(schedule)} "
                      f"blocks (±0.08: {widened[0]}, all blocks: {widened[1]})")
    else:
        status.append("No memory: understates multi-month swings")
    if steer is not None:
        status.append(f"Raw: paths are steered away from unchecked bins "
                      f"({share * 100:.1f} % of months still fall in them).")
    if sims < 100:
        status.append("Fewer than 100 simulations: percentiles are rough below 100 "
                      "simulations")

    return RawResult(True, "", t_of(dates[:steps + 1]), paths, status, schedule,
                     z_shift, sigma_steps, share, labels, edges, states, trend, where,
                     sources)
