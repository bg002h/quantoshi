#!/usr/bin/env python3
"""Per-halving-cycle monthly return statistics and the Raw-MC trend fits.

Reproduces the numbers quoted in
docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md (Measured section
and the default fit).  The definitions (monthly log returns, halving cycles,
full / complete, fit set, power and exponential trends) live in
btc_web/raw_mc.py -- see its docstring; this script only loads the price CSV,
calls them and prints.

Usage: btc_venv/bin/python3 tools/raw_mc_cycle_stats.py [--from YEAR] [--to YEAR]
Not covered: the simulation itself.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "btc_web"))   # appended: never shadows an installed module

from raw_mc import (cycle_table, fit_set, fit_trend, load_months,  # noqa: E402
                    t_of)


def read_prices():
    df = pd.read_csv(ROOT / "BitcoinPricesDaily.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    return df.set_index("Date")["Price"].astype(float)


def _rms(fset, trend):
    return np.sqrt(np.mean((fset.g.values - trend.g(fset.t.values)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int)
    ap.add_argument("--to", dest="y1", type=int)
    a = ap.parse_args()
    m = load_months(read_prices())
    window = None
    if a.y0 is not None or a.y1 is not None:     # an open end spans all history
        window = (a.y0 if a.y0 is not None else m.index.year.min(),
                  a.y1 if a.y1 is not None else m.index.year.max())
    tab = cycle_table(m, window)
    pd.set_option("display.width", 160)
    shown = tab.assign(growth_pct=np.expm1(tab.g) * 100, g_x_t=tab.g * tab.t,
                       vol_pct=tab.s * 100)
    print(shown[["cycle", "first", "last", "months", "full", "complete", "t", "growth_pct",
                 "g_x_t", "vol_pct"]].to_string(index=False,
                                                float_format=lambda x: f"{x:.2f}"))
    fset, where = fit_set(tab, cycle_table(m, None))
    power = fit_trend("power", fset, m, window)
    exp = fit_trend("exp", fset, m, window)
    if power is None or exp is None:       # too few cycles, or curve_fit did not converge
        print(f"\nfit set: {where}\nno fit: trend could not be fitted to these cycles")
        return
    pp, pe = power.params, exp.params
    print(f"\nfit set: {where}")
    print(f"fit on {len(fset)} full cycles: power n = {pp['n']:.2f} "
          f"(rms {_rms(fset, power):.3f}), vol = {pp['a']:.2f} * t^-{pp['b']:.2f}; "
          f"exp growth exp({pe['alpha']:.2f} {pe['beta']:+.3f} t) "
          f"(rms {_rms(fset, exp):.3f}), vol exp({pe['gamma']:.2f} {pe['delta']:+.3f} t)")
    for yr in (2030, 2040, 2050, 2066):
        tt = t_of(pd.Timestamp(f"{yr}-07-01"))
        print(f"  {yr}: power growth {np.expm1(power.g(tt)):+.0%}/yr "
              f"vol {power.s(tt):.0%} | exp growth "
              f"{np.expm1(exp.g(tt)):+.0%}/yr "
              f"vol {exp.s(tt):.0%}")


if __name__ == "__main__":
    main()
