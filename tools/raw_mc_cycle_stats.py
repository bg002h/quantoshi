#!/usr/bin/env python3
"""Per-halving-cycle monthly return statistics and the Raw-MC trend fits.

Reproduces the numbers quoted in
docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md (Measured section
and the default fit), using the spec's definitions:

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
  * g_c = annual log growth = 12 * mean monthly log return;
    s_c = annual vol = sqrt(12) * std of monthly log returns;
  * power law: g = n / t (unweighted LS, n = sum(g/t) / sum(1/t^2));
    vol = a * t^-b (LS of log s on log t);
  * exponential: g = exp(alpha + beta t), s = exp(gamma + delta t), fitted
    directly in g / s space (scipy curve_fit).

Usage: btc_venv/bin/python3 tools/raw_mc_cycle_stats.py [--from YEAR] [--to YEAR]
Not covered: the simulation itself (see btc_web/raw_mc.py once built).
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORIGIN = pd.Timestamp("2009-07-25")
EPOCHS = [pd.Timestamp(d) for d in
          ("2009-01-03", "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")]
MIN_FULL_MONTHS = 24


def monthly_returns():
    df = pd.read_csv(ROOT / "BitcoinPricesDaily.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    s = df.set_index("Date")["Price"].astype(float)
    last = s.index[-1]
    me = s.resample("ME").last()
    if last < me.index[-1]:            # incomplete trailing month
        me = me.iloc[:-1]
    r = np.log(me).diff().dropna()
    mid = r.index - pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    t = (mid - ORIGIN).days / 365.25
    ep = np.array(EPOCHS + [EPOCHS[-1] + pd.DateOffset(years=4)],
                  dtype="datetime64[ns]")
    cyc = np.searchsorted(ep[:-1], r.index.values, side="right")
    start, end = ep[cyc - 1], ep[cyc]
    phi = ((mid.values - start) / (end - start)).astype(float)
    return pd.DataFrame({"r": r.values, "t": t, "cycle": cyc, "phi": phi},
                        index=r.index)


def cycle_table(m, y0=None, y1=None):
    if y0 is not None:
        m = m[m.index.year >= y0]
    if y1 is not None:
        m = m[m.index.year <= y1]
    rows = []
    for c, g in m.groupby("cycle"):
        full = c < len(EPOCHS) and len(g) >= MIN_FULL_MONTHS
        complete = full and g.phi.min() < 0.05 and g.phi.max() > 0.95
        rows.append(dict(cycle=c, first=g.index[0].strftime("%Y-%m"),
                         last=g.index[-1].strftime("%Y-%m"), months=len(g),
                         complete=complete,
                         t=g.t.mean(), g=12 * g.r.mean(),
                         s=np.sqrt(12) * g.r.std(), full=full))
    return pd.DataFrame(rows)


def fit_set(tab_window, tab_all):
    """Cycles the trend curves are fitted on (spec: >= 3 in window, else all)."""
    w = tab_window[tab_window.full]
    return (w, "window") if len(w) >= 3 else (tab_all[tab_all.full], "all history")


def fits(f):
    t, g, s = f.t.values, f.g.values, f.s.values
    n = np.sum(g / t) / np.sum(1 / t ** 2)
    nb, la = np.polyfit(np.log(t), np.log(s), 1)
    (al, be), _ = curve_fit(lambda x, a, b: np.exp(a + b * x), t, g,
                            p0=(np.log(max(g.mean(), 1e-3)), -0.1), maxfev=20000)
    (ga, de), _ = curve_fit(lambda x, a, b: np.exp(a + b * x), t, s,
                            p0=(np.log(max(s.mean(), 1e-3)), -0.05), maxfev=20000)
    return dict(cycles=len(f), n=n, a=np.exp(la), b=-nb,
                alpha=al, beta=be, gamma=ga, delta=de,
                rms_power=np.sqrt(np.mean((g - n / t) ** 2)),
                rms_exp=np.sqrt(np.mean((g - np.exp(al + be * t)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int)
    ap.add_argument("--to", dest="y1", type=int)
    a = ap.parse_args()
    m = monthly_returns()
    tab = cycle_table(m, a.y0, a.y1)
    pd.set_option("display.width", 160)
    shown = tab.assign(growth_pct=np.expm1(tab.g) * 100, g_x_t=tab.g * tab.t,
                       vol_pct=tab.s * 100)
    print(shown[["cycle", "first", "last", "months", "full", "complete", "t", "growth_pct",
                 "g_x_t", "vol_pct"]].to_string(index=False,
                                                float_format=lambda x: f"{x:.2f}"))
    fset, where = fit_set(tab, cycle_table(m))
    try:
        f = fits(fset)
    except RuntimeError as e:          # curve_fit did not converge
        print(f"\nfit set: {where}\nno fit: {e}")
        return
    if True:
        print(f"\nfit set: {where}")
        print("fit on {cycles} full cycles: power n = {n:.2f} (rms {rms_power:.3f}), "
              "vol = {a:.2f} * t^-{b:.2f}; exp growth exp({alpha:.2f} {beta:+.3f} t) "
              "(rms {rms_exp:.3f}), vol exp({gamma:.2f} {delta:+.3f} t)".format(**f))
        for yr in (2030, 2040, 2050, 2066):
            tt = (pd.Timestamp(f"{yr}-07-01") - ORIGIN).days / 365.25
            print(f"  {yr}: power growth {np.expm1(f['n'] / tt):+.0%}/yr "
                  f"vol {f['a'] * tt ** -f['b']:.0%} | exp growth "
                  f"{np.expm1(np.exp(f['alpha'] + f['beta'] * tt)):+.0%}/yr "
                  f"vol {np.exp(f['gamma'] + f['delta'] * tt):.0%}")


if __name__ == "__main__":
    main()
