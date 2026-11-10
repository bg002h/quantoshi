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
               sims=int(req.sims), years=int(req.years), today=day(req.today),
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


def simulate(req: RawRequest) -> RawResult:
    """Bootstrap ``req.sims`` monthly paths from today for ``req.years``.

    Trend power / exp: ``r = mu_m(t_mid) + sigma_cycle * (z - z_shift)``,
    ``z`` a bootstrapped historical shape (``standardise``).  Trend none:
    the same draws of raw ``r`` (honest replay).  Independent mode draws one
    eligible month per step; cycle mode walks the per-cycle block grid and
    copies runs of consecutive months from one complete cycle whose start
    phase matches the block's.  No regime filter here (Task 3).
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

    tab_w, tab_all = cycle_table(months, req.window), cycle_table(months, None)
    fset, where = fit_set(tab_w, tab_all)
    if req.trend == "none":
        where = "window"
    trend = fit_trend(req.trend, fset, months, req.window)
    if trend is None:
        name = "Exponential" if req.trend == "exp" else "Power-law"
        return _refuse(f"{name} trend could not be fitted to these cycles", status,
                       None, where)
    status.append(_trend_line(trend, len(fset), where))

    complete = tab_w[tab_w.complete]
    cycle_mode = req.variant == "raw_cycle"
    if cycle_mode and len(complete) < 2:
        return _refuse("Halving-cycle mode needs at least 2 complete cycles in the "
                       f"window ({len(complete)} found): no fan drawn", status,
                       trend, where)

    # ── step calendar ──
    steps = int(req.years) * 12
    today = pd.Timestamp(req.today)
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

    if not cycle_mode:
        draws = pool[rng.integers(0, len(pool), size=(sims, steps))]
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

        draws = np.empty((sims, steps))
        widened = [0] * len(PHASE_TOLS)
        for c, k0, k1 in _grid(cyc, 0, steps):
            L = k1 - k0
            cand, lvl = candidates(L, phi[k0])
            if lvl:
                widened[lvl - 1] += 1
            schedule.append(dict(cycle=c, steps=(k0, k1), cand=cand, widened=lvl))
            pick = cand[rng.integers(0, len(cand), size=sims)]
            draws[:, k0:k1] = pool[pick[:, None] + np.arange(L)] - z_shift

    r = draws if replay else mu + sigma_steps * draws
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
    if sims < 100:
        status.append("Fewer than 100 simulations: percentiles are rough below 100 "
                      "simulations")

    return RawResult(True, "", t_of(dates[:steps + 1]), paths, status, schedule,
                     z_shift, sigma_steps, 0.0, [], None, None, trend, where)
