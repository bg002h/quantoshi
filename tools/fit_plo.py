#!/usr/bin/env python3
"""Fit Offset Power Law model to Bitcoin price history.

Model: log10(price) = A + m * log10(t + c)

Three parameters:
  A = log10 intercept  (dimensionless)
  m = power-law slope  (dimensionless)
  c = time-origin offset, years (positive: effective origin earlier than
      our chosen genesis 2009-07-25; negative: later)

Compared with PowerLawModel (fixed c=0), this lets the model choose its
own time-zero. If the data prefer c≈0 the result degenerates to plain PL.

Usage:
    btc_venv/bin/python3 tools/fit_plo.py              # fit and print
    btc_venv/bin/python3 tools/fit_plo.py --update      # fit and update btc_core/
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import curve_fit, differential_evolution


def plo_log10(t, A, m, c):
    """Offset power law: log10(price) = A + m * log10(t + c)."""
    return A + m * np.log10(np.maximum(t + c, 1e-9))


def main():
    update = "--update" in sys.argv

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    log_p = prices.df_full["log_price"].values
    mask = t >= 1.0
    t_fit = t[mask]
    lp_fit = log_p[mask]
    print(f"  {len(t_fit)} data points (t >= 1.0)")

    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)

    # Bounds:
    #   A in [-5, 10]    — wide; plain-PL A≈2, but offset-PL can trade A↔c
    #   m in [1, 12]     — slope; plain PL m≈5.5
    #   c in [-0.9, 10]  — must keep t+c > 0 for all t in data (t_min ≈ 1)
    bounds_lo = [-5.0, 1.0, -0.9]
    bounds_hi = [10.0, 12.0, 10.0]

    # Use DE as primary — non-linear offset makes curve_fit sensitive to p0.
    bounds = list(zip(bounds_lo, bounds_hi))

    def objective(params):
        pred = plo_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("  Running differential evolution...")
    res = differential_evolution(objective, bounds, maxiter=5000, seed=42,
                                 tol=1e-14, polish=True, popsize=30,
                                 workers=1)
    # Polish with curve_fit starting from DE result for extra precision.
    try:
        popt, _ = curve_fit(plo_log10, t_fit, lp_fit, p0=res.x,
                            bounds=(bounds_lo, bounds_hi), maxfev=20000)
    except Exception:
        popt = res.x

    A, m, c = popt
    pred = plo_log10(t_fit, *popt)
    r2 = 1.0 - np.sum((lp_fit - pred) ** 2) / ss_tot
    sigma = float(np.std(lp_fit - pred))

    print(f"\nFitted Offset Power Law parameters:")
    print(f"  A     = {A:.6f}   (log10 intercept)")
    print(f"  m     = {m:.6f}   (slope)")
    print(f"  c     = {c:.6f}   (offset, years)")
    print(f"  R²    = {r2:.6f}")
    print(f"  σ     = {sigma:.6f}")

    if update:
        print("\nUpdating btc_core/ ...")
        core_path = os.path.join(ROOT, "btc_core", "_simple.py")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [("_A", A), ("_m", m), ("_c", c)]
        cls_pos = src.find("class OffsetPowerLawModel")
        if cls_pos == -1:
            print("  WARNING: could not find OffsetPowerLawModel class")
            sys.exit(1)
        else:
            next_class = src.find("\nclass ", cls_pos + 1)
            cls_end = next_class if next_class != -1 else len(src)
            section = src[cls_pos:cls_end]
            for name, val in replacements:
                pattern = rf"(    {name}\s*=\s*)[^#\n]+"
                new_val = f"{val:>11.6f}" if val >= 0 else f"{val:.6f}"
                match = re.search(pattern, section)
                if match:
                    old_line = match.group(0)
                    new_line = re.sub(pattern, rf"\g<1>{new_val}  ", old_line)
                    new_section = section.replace(old_line, new_line, 1)
                    src = src[:cls_pos] + new_section + src[cls_end:]
                    cls_pos = src.find("class OffsetPowerLawModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in OffsetPowerLawModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core/ updated.")
    else:
        print("\nRun with --update to write to btc_core/")


if __name__ == "__main__":
    main()
