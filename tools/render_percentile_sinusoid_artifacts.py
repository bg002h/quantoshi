#!/usr/bin/env python3
"""Render the percentile-sinusoid study figures into docs/artifacts/.

    btc_venv/bin/python3 tools/render_percentile_sinusoid_artifacts.py

Two PNGs, three panels each (calendar-time / log-time / both terms) fitted to
the percentile of the BTC price within the QR model's quantile fan:

    percentile-sinusoid-fits.png                full record, 2010-2026
    percentile-sinusoid-fits-extrapolated.png   same fits, +10 yr, 2020 on
    percentile-sinusoid-calendar-only-10yr.png   calendar fit alone, one wide
    percentile-sinusoid-calendar-only-20yr.png   panel, full record + 10 / 20 yr

Both mark:
  * every peak and trough OF THE FITTED CURVE — date, fitted percentile, and
    the price the QR fan puts at that percentile on that date;
  * the major highs and lows OF THE ACTUAL BITCOIN PRICE — date, the real
    close, and the percentile that close sat at.

Self-contained: the fits are recomputed here from `model_data.pkl` on every
run, so the artifacts can be regenerated after a price update or a model
refit without depending on any scratch state.

Notes on the analysis, so a future reader does not have to rediscover them:

* The QR fan is non-monotonic in the tails (channels cross on ~16.5 % of
  dates), so percentile -> price uses the FORWARD direction, interpolating
  in log-price across the quantile grid. That direction is well defined;
  the inverse is the one that needs `_bracket_percentile`.
* t is years since the 2009-07-25 origin, anchored at t = 1 (log-log space
  has no t = 0).
* These are descriptive fits, R^2 ~ 0.55-0.70. The extrapolation is drawn
  because it was asked for, not because a sinusoid predicts ten years.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "btc_web"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("TESTING", "1")

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import matplotlib.dates as mdates                            # noqa: E402
from scipy.optimize import minimize, minimize_scalar         # noqa: E402
from scipy.signal import find_peaks                          # noqa: E402

import app                                                   # noqa: E402,F401
import _app_ctx                                              # noqa: E402
from figures.percentile import _percentile_series            # noqa: E402

OUT_DIR = REPO / "docs" / "artifacts"
GENESIS = pd.Timestamp("2009-07-25")
DATA, BLUE, VERM, GREEN, PRICE = "#9A9A9A", "#0072B2", "#D55E00", "#009E73", "#1A1A2E"


# ── data ────────────────────────────────────────────────────────────────────

def load_series():
    M = _app_ctx.M
    qr = _app_ctx.PRICE_MODELS["qr"]
    t_all = np.asarray(M.price_years, float)
    keep = t_all >= 1.0                       # log-log anchor is t = 1
    t = t_all[keep]
    px = np.asarray(M.price_prices, float)[keep]
    dates = pd.to_datetime([str(d) for d in M.price_dates])[keep]
    pct = _percentile_series(qr, t, px)
    if pct is None:
        raise SystemExit("QR model has no quantile fan — cannot build the study")
    return qr, t, px, dates, np.asarray(pct, float)


# ── the three fits ──────────────────────────────────────────────────────────

def _lstsq(cols, y):
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    return coef, fit, float(((y - fit) ** 2).sum())


def _amp_phase(a, b):
    return float(np.hypot(a, b)), float(np.arctan2(-b, a))


def fit_all(t, y):
    """Calendar, log-time and joint fits. Frequencies are scanned on a grid —
    for a fixed frequency the amplitude and phase are linear, so each grid
    node is solved exactly and no starting guess can strand the fit."""
    ones, lnt = np.ones_like(t), np.log(t)
    sst = float(((y - y.mean()) ** 2).sum())
    span = t[-1] - t[0]

    best_f, best = None, np.inf
    for f in np.linspace(1.0 / span, 2.0, 20000):
        w = 2 * np.pi * f * t
        s = _lstsq([ones, np.cos(w), np.sin(w)], y)[2]
        if s < best:
            best, best_f = s, f
    f_cal = float(minimize_scalar(
        lambda f: _lstsq([ones, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)], y)[2],
        bracket=(best_f * 0.98, best_f, best_f * 1.02)).x)
    w = 2 * np.pi * f_cal * t
    c_cal, _, sse_cal = _lstsq([ones, np.cos(w), np.sin(w)], y)
    A_cal, p_cal = _amp_phase(c_cal[1], c_cal[2])

    best_om, best = None, np.inf
    for om in np.linspace(0.2, 40.0, 40000):
        w2 = om * lnt
        s = _lstsq([ones, np.cos(w2), np.sin(w2)], y)[2]
        if s < best:
            best, best_om = s, om
    om_log = float(minimize_scalar(
        lambda om: _lstsq([ones, np.cos(om*lnt), np.sin(om*lnt)], y)[2],
        bracket=(best_om * 0.99, best_om, best_om * 1.01)).x)
    w2 = om_log * lnt
    c_log, _, sse_log = _lstsq([ones, np.cos(w2), np.sin(w2)], y)
    A_log, p_log = _amp_phase(c_log[1], c_log[2])

    def sse_both(v):
        a, b = 2 * np.pi * v[0] * t, v[1] * lnt
        return _lstsq([ones, np.cos(a), np.sin(a), np.cos(b), np.sin(b)], y)[2]

    pair, val = (f_cal, om_log), sse_both((f_cal, om_log))
    for f0 in np.linspace(max(1e-3, f_cal * 0.3), f_cal * 2.5, 60):
        for om0 in np.linspace(max(0.2, om_log * 0.3), om_log * 2.5, 60):
            v = sse_both((f0, om0))
            if v < val:
                val, pair = v, (f0, om0)
    r = minimize(sse_both, pair, method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 6000})
    f_b, om_b = float(r.x[0]), float(r.x[1])
    a, b = 2 * np.pi * f_b * t, om_b * lnt
    c_b, _, sse_b = _lstsq([ones, np.cos(a), np.sin(a), np.cos(b), np.sin(b)], y)
    A1, P1 = _amp_phase(c_b[1], c_b[2])
    A2, P2 = _amp_phase(c_b[3], c_b[4])

    return {
        "calendar": dict(c=float(c_cal[0]), A=A_cal, f=f_cal, phi=p_cal,
                         r2=1 - sse_cal / sst),
        "logtime": dict(c=float(c_log[0]), A=A_log, om=om_log, phi=p_log,
                        r2=1 - sse_log / sst),
        "both": dict(c=float(c_b[0]), A1=A1, f=f_b, phi1=P1, A2=A2, om=om_b,
                     phi2=P2, r2=1 - sse_b / sst),
    }


def make_curves(F):
    def cal(x):
        p = F["calendar"]
        return p["c"] + p["A"] * np.cos(2 * np.pi * p["f"] * x + p["phi"])

    def logt(x):
        p = F["logtime"]
        return p["c"] + p["A"] * np.cos(p["om"] * np.log(x) + p["phi"])

    def both(x):
        p = F["both"]
        return (p["c"]
                + p["A1"] * np.cos(2 * np.pi * p["f"] * x + p["phi1"])
                + p["A2"] * np.cos(p["om"] * np.log(x) + p["phi2"]))
    return cal, logt, both


# ── helpers ─────────────────────────────────────────────────────────────────

def money(v):
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    if v >= 1e3:
        return f"${v/1e3:,.0f}k"
    if v >= 1:
        return f"${v:,.0f}"
    return f"${v:.2f}"


def to_date(t_):
    return GENESIS + pd.Timedelta(days=float(t_) * 365.25)


def implied_price(qr, pct, t_):
    """The USD level the QR fan puts at `pct` on date `t_` (forward direction)."""
    qs = np.asarray(qr.quantiles, float)
    logfan = np.array([
        np.log10(max(float(np.asarray(qr.price_at(q, np.array([t_]))).ravel()[0]),
                     1e-12)) for q in qs])
    return float(10 ** np.interp(np.clip(pct / 100.0, qs[0], qs[-1]), qs, logfan))


def fit_extrema(fn, t0, t1):
    tt = np.arange(t0, t1, 1.0 / 365.25)
    yy = fn(tt)
    dy = np.diff(yy)
    return [(tt[i], yy[i], "peak" if dy[i - 1] > 0 >= dy[i] else "trough")
            for i in range(1, len(dy))
            if (dy[i - 1] > 0 >= dy[i]) or (dy[i - 1] < 0 <= dy[i])]


def price_extrema(t, px, pct, dates, t_lo, n_each=4):
    """The most prominent highs and lows of the ACTUAL price after t_lo.

    Prominence is measured on log price so a 2011 top competes with a 2021
    top instead of being flattened by six orders of magnitude of growth.
    """
    m = t >= t_lo
    idx = np.flatnonzero(m)
    lp = np.log10(px[m])
    out = []
    for sign, kind in ((1, "high"), (-1, "low")):
        pk, props = find_peaks(sign * lp, distance=180, prominence=0.15)
        if len(pk) == 0:
            continue
        order = np.argsort(props["prominences"])[::-1][:n_each]
        for j in sorted(pk[order]):
            k = idx[j]
            out.append((float(t[k]), float(px[k]), float(pct[k]),
                        dates[k], kind))
    return sorted(out, key=lambda r: r[0])


# ── rendering ───────────────────────────────────────────────────────────────

def draw(path, F, curves, t, px, pct, dates, *, extrapolate_years=0.0,
         x_from=None, title_suffix=""):
    cal, logt, both = curves
    last_date = dates[-1]
    t0 = t[0] if x_from is None else (pd.Timestamp(x_from) - GENESIS).days / 365.25
    t1 = t[-1] + extrapolate_years
    x0 = to_date(t0)
    x1 = to_date(t1)
    qr = _app_ctx.PRICE_MODELS["qr"]

    panels = [
        ("Calendar-time sinusoid", cal, BLUE, (0, ()),
         f"y = c + A·cos(2πf·t + φ)   ·   period "
         f"{1/F['calendar']['f']:.3f} yr   ·   A {F['calendar']['A']:.1f} pp"
         f"   ·   R² {F['calendar']['r2']:.3f}"),
        ("Log-time sinusoid", logt, VERM, (0, (6, 2)),
         f"y = c + A·cos(ω·ln t + φ)   ·   ω {F['logtime']['om']:.3f}"
         f"  (λ = e^(2π/ω) = {np.exp(2*np.pi/F['logtime']['om']):.3f})"
         f"   ·   R² {F['logtime']['r2']:.3f}"),
        ("Both terms", both, GREEN, (0, (1, 1.4)),
         f"calendar {1/F['both']['f']:.3f} yr + log-time ω {F['both']['om']:.3f}"
         f"   ·   R² {F['both']['r2']:.3f}"),
    ]

    tt = np.arange(t0, t1, 1.0 / 365.25)
    dd = pd.DatetimeIndex([to_date(v) for v in tt])
    pxt = price_extrema(t, px, pct, dates, t0)

    fig, axes = plt.subplots(3, 1, figsize=(17, 16.5), sharex=True)
    fig.suptitle(
        "Sinusoid fits to the QR-model percentile of BTC price" + title_suffix
        + "\ncoloured labels = peaks/troughs of the FITTED curve "
          "(date · fitted percentile · price the QR fan puts there)   |   "
          "dark labels = actual BTC price highs/lows "
          "(date · real close · percentile it sat at)",
        fontsize=13, y=0.981)

    for ax, (name, fn, colour, dash, sub) in zip(axes, panels):
        yy = fn(tt)
        hist = dd <= last_date
        ax.plot(dates, pct, color=DATA, lw=0.8, zorder=1,
                label="QR percentile (actual)")
        ax.plot(dd[hist], yy[hist], color=colour, lw=2.2, ls=dash, zorder=3,
                label="fit (in sample)")
        if extrapolate_years > 0:
            ax.plot(dd[~hist], yy[~hist], color=colour, lw=2.2, ls=dash,
                    alpha=0.55, zorder=3, label="extrapolation")
            ax.axvspan(last_date, x1, color="#000000", alpha=0.045, zorder=0)
            ax.axvline(last_date, color="#444", lw=1.0, ls=(0, (3, 3)), zorder=2)

        # fitted-curve extrema
        for t_e, y_e, kind in fit_extrema(fn, t0, t1):
            de = to_date(t_e)
            up = kind == "peak"
            ax.plot([de], [y_e], "o", ms=5.5, color=colour, mec="white",
                    mew=1.0, zorder=6)
            frac = (de - x0) / (x1 - x0)
            dx, ha = 0, "center"
            if frac < 0.07:
                dx, ha = 34, "left"
            elif frac > 0.93:
                dx, ha = -34, "right"
            ax.annotate(
                f"{de.date()}\nQ{y_e:.1f}% · {money(implied_price(qr, y_e, t_e))}",
                xy=(de, y_e), xytext=(dx, 28 if up else -28),
                textcoords="offset points", ha=ha,
                va="bottom" if up else "top", fontsize=7.4,
                family="DejaVu Sans Mono", zorder=7,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=colour,
                          alpha=0.94, lw=0.8),
                arrowprops=dict(arrowstyle="-", color=colour, lw=0.8,
                                shrinkA=0, shrinkB=3))

        # actual price highs / lows. Consecutive same-direction marks that
        # fall close together (2021-04 and 2021-11, say) are staggered to two
        # heights rather than drawn on top of each other.
        near = (x1 - x0) * 0.075
        tier, prev_d, prev_kind = 0, None, None
        for t_p, p_p, q_p, d_p, kind in pxt:
            if prev_d is not None and kind == prev_kind and (d_p - prev_d) < near:
                tier = 1 - tier
            else:
                tier = 0
            prev_d, prev_kind = d_p, kind
            up = kind == "high"
            ax.plot([d_p], [q_p], marker="D", ms=5.0, color=PRICE,
                    mec="white", mew=0.9, zorder=6)
            frac = (d_p - x0) / (x1 - x0)
            dx, ha = 0, "center"
            if frac < 0.07:
                dx, ha = 30, "left"
            elif frac > 0.93:
                dx, ha = -30, "right"
            ax.annotate(
                f"{d_p.date()}\n{money(p_p)} · Q{q_p:.1f}%",
                xy=(d_p, q_p), xytext=(dx, (62 + 34*tier) if up else -(62 + 34*tier)),
                textcoords="offset points", ha=ha,
                va="bottom" if up else "top", fontsize=7.0,
                family="DejaVu Sans Mono", zorder=7, color="white",
                bbox=dict(boxstyle="round,pad=0.24", fc=PRICE, ec="white",
                          alpha=0.93, lw=0.7),
                arrowprops=dict(arrowstyle="-", color=PRICE, lw=0.7,
                                ls=(0, (2, 1.5)), shrinkA=0, shrinkB=3))

        ax.axhline(50, color="k", lw=0.6, alpha=0.28, zorder=0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(-88, 188)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("percentile in QR fan")
        ax.set_title(f"{name}   —   {sub}", fontsize=10.5, loc="left", pad=10)
        ax.grid(alpha=0.16)
        # legend is placed once at figure level below, so it cannot
        # collide with a panel title or a label

    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, 0.949),
               ncol=4, fontsize=9, framealpha=0.95)

    span_years = (x1 - x0).days / 365.25
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(1 if span_years < 20 else 2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("date")
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    fig.text(0.5, 0.008,
             f"fitted on all {len(pct):,} daily points, "
             f"{dates[0].date()} – {dates[-1].date()};  t = years since "
             f"{GENESIS.date()} anchored at t = 1.  "
             "Descriptive fits (R² 0.55–0.70), not forecasts.",
             ha="center", fontsize=8.6, color="#555")
    fig.tight_layout(rect=[0, 0.022, 1, 0.934])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")
    return pxt


def draw_single(path, F, curves, t, px, pct, dates, *, extrapolate_years=10.0):
    """One wide panel: the calendar-time fit alone, whole record + extension.

    The three-panel figures stack short axes, which suits comparing forms. For
    reading dates and levels off a single fit, one tall wide axis is better —
    same data, same labels, more room per label.
    """
    cal = curves[0]
    qr = _app_ctx.PRICE_MODELS["qr"]
    p = F["calendar"]
    t0, t1 = t[0], t[-1] + extrapolate_years
    x0, x1 = to_date(t0), to_date(t1)
    last = dates[-1]
    tt = np.arange(t0, t1, 1.0 / 365.25)
    dd = pd.DatetimeIndex([to_date(v) for v in tt])
    yy = cal(tt)
    hist = dd <= last

    span_yr = (x1 - x0).days / 365.25
    fig, ax = plt.subplots(figsize=(max(21.0, 11.0 + span_yr * 0.42), 9.5))
    ax.plot(dates, pct, color=DATA, lw=0.85, zorder=1,
            label="QR percentile (actual)")
    ax.plot(dd[hist], yy[hist], color=BLUE, lw=2.6, zorder=3,
            label="fit (in sample)")
    ax.plot(dd[~hist], yy[~hist], color=BLUE, lw=2.6, alpha=0.55, zorder=3,
            label=f"extrapolation ({extrapolate_years:.0f} yr)")
    ax.axvspan(last, x1, color="#000000", alpha=0.045, zorder=0)
    ax.axvline(last, color="#444", lw=1.2, ls=(0, (3, 3)), zorder=2)

    # Half-amplitude twin: same offset, frequency and phase, A/2. Its extrema
    # fall on the SAME dates as the full fit, so its labels are placed INSIDE
    # the envelope (peaks below the point, troughs above) — the opposite of
    # the convention used for the full fit, which is what keeps the two label
    # sets from landing on top of each other.
    def half(x):
        return p["c"] + 0.5 * p["A"] * np.cos(2 * np.pi * p["f"] * x + p["phi"])

    yh = half(tt)
    ax.plot(dd[hist], yh[hist], color=VERM, lw=2.2, ls=(0, (6, 2)), zorder=3,
            label=f"same phase, ½ amplitude ({0.5*p['A']:.1f} pp)")
    ax.plot(dd[~hist], yh[~hist], color=VERM, lw=2.2, ls=(0, (6, 2)),
            alpha=0.55, zorder=3)
    for t_e, y_e, kind in fit_extrema(half, t0, t1):
        de = to_date(t_e)
        up = kind == "peak"
        ax.plot([de], [y_e], "s", ms=5.2, color=VERM, mec="white", mew=1.0,
                zorder=6)
        hfrac = (de - x0) / (x1 - x0)
        hdx, hha = 0, "center"
        if hfrac < 0.045:                 # would overrun the y-axis and its ticks
            hdx, hha = 30, "left"
        elif hfrac > 0.955:
            hdx, hha = -30, "right"
        ax.annotate(
            f"{de.date()}\nQ{y_e:.1f}% \u00b7 {money(implied_price(qr, y_e, t_e))}",
            xy=(de, y_e), xytext=(hdx, -26 if up else 26),  # inverted on purpose
            textcoords="offset points", ha=hha,
            va="top" if up else "bottom", fontsize=7.2,
            family="DejaVu Sans Mono", zorder=7,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=VERM,
                      alpha=0.95, lw=0.9),
            arrowprops=dict(arrowstyle="-", color=VERM, lw=0.9, shrinkA=0,
                            shrinkB=3))

    for t_e, y_e, kind in fit_extrema(cal, t0, t1):
        de = to_date(t_e)
        up = kind == "peak"
        ax.plot([de], [y_e], "o", ms=6, color=BLUE, mec="white", mew=1.1, zorder=6)
        frac = (de - x0) / (x1 - x0)
        dx, ha = 0, "center"
        if frac < 0.045:
            dx, ha = 30, "left"
        elif frac > 0.955:
            dx, ha = -30, "right"
        ax.annotate(
            f"{de.date()}\nQ{y_e:.1f}% \u00b7 {money(implied_price(qr, y_e, t_e))}",
            xy=(de, y_e), xytext=(dx, 30 if up else -30),
            textcoords="offset points", ha=ha,
            va="bottom" if up else "top", fontsize=7.8,
            family="DejaVu Sans Mono", zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=BLUE,
                      alpha=0.95, lw=0.9),
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9, shrinkA=0,
                            shrinkB=3))

    near = (x1 - x0) * 0.055
    tier, prev_d, prev_kind = 0, None, None
    for t_p, p_p, q_p, d_p, kind in price_extrema(t, px, pct, dates, t0):
        if prev_d is not None and kind == prev_kind and (d_p - prev_d) < near:
            tier = 1 - tier
        else:
            tier = 0
        prev_d, prev_kind = d_p, kind
        up = kind == "high"
        ax.plot([d_p], [q_p], marker="D", ms=5.5, color=PRICE, mec="white",
                mew=1.0, zorder=6)
        ax.annotate(
            f"{d_p.date()}\n{money(p_p)} \u00b7 Q{q_p:.1f}%",
            xy=(d_p, q_p),
            xytext=(0, (70 + 36 * tier) if up else -(70 + 36 * tier)),
            textcoords="offset points", ha="center",
            va="bottom" if up else "top", fontsize=7.4, color="white",
            family="DejaVu Sans Mono", zorder=7,
            bbox=dict(boxstyle="round,pad=0.26", fc=PRICE, ec="white",
                      alpha=0.94, lw=0.8),
            arrowprops=dict(arrowstyle="-", color=PRICE, lw=0.8,
                            ls=(0, (2, 1.5)), shrinkA=0, shrinkB=3))

    ax.axhline(50, color="k", lw=0.7, alpha=0.28, zorder=0)
    ax.set_xlim(x0, x1)
    # just enough for the two label tiers; -96/196 left the data squashed
    # into the middle third of the panel
    ax.set_ylim(-74, 172)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("percentile in QR fan")
    ax.set_xlabel("date")
    ax.grid(alpha=0.17)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95, ncol=3)
    ax.set_title(
        "Calendar-time sinusoid fit to the QR-model percentile of BTC price — "
        f"full record + {extrapolate_years:.0f}-year extrapolation\n"
        f"y = c + A\u00b7cos(2\u03c0f\u00b7t + \u03c6)   \u00b7   "
        f"period {1/p['f']:.3f} yr   \u00b7   A {p['A']:.1f} pp   \u00b7   "
        f"offset {p['c']:.1f}   \u00b7   R\u00b2 {p['r2']:.3f}      |      "
        "blue = fitted peaks/troughs   \u00b7   orange = same phase at half "
        "amplitude (labels inside the envelope)   \u00b7   dark = actual BTC "
        "highs/lows\n"
        "all labels: date \u00b7 percentile \u00b7 price the QR fan puts there "
        "(dark labels show the real close instead)",
        fontsize=12, loc="left", pad=14)
    fig.text(0.5, 0.012,
             f"fitted on all {len(pct):,} daily points, {dates[0].date()} \u2013 "
             f"{dates[-1].date()};  t = years since {GENESIS.date()} anchored at "
             "t = 1.  Descriptive fit, not a forecast.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0.028, 1, 1])
    fig.savefig(path, dpi=145)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qr, t, px, dates, pct = load_series()
    print(f"percentile series: {len(pct)} days  {dates[0].date()} .. {dates[-1].date()}")
    F = fit_all(t, pct)
    for k, v in F.items():
        print(f"  {k:9s} R² = {v['r2']:.4f}")
    curves = make_curves(F)

    draw(OUT_DIR / "percentile-sinusoid-fits.png", F, curves, t, px, pct, dates,
         title_suffix="")
    pxt = draw(OUT_DIR / "percentile-sinusoid-fits-extrapolated.png", F, curves,
               t, px, pct, dates, extrapolate_years=10.0, x_from="2020-01-01",
               title_suffix=" — extrapolated 10 years")

    for yrs in (10, 20):
        draw_single(OUT_DIR / f"percentile-sinusoid-calendar-only-{yrs}yr.png",
                    F, curves, t, px, pct, dates, extrapolate_years=float(yrs))

    print("\nactual BTC price extremes marked (from 2020 for the second figure):")
    for t_p, p_p, q_p, d_p, kind in pxt:
        print(f"  {kind:4s} {d_p.date()}  {money(p_p):>9s}  Q{q_p:5.1f}%")


if __name__ == "__main__":
    main()
