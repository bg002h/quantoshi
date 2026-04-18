#!/usr/bin/env python3
"""Fit Stretched Exponential model to Bitcoin price history.

Model: log10(price) = A + B * t^beta

Three parameters:
  A    = log10 intercept at t=0
  B    = scale of the stretched term
  beta = stretching exponent (beta=1 → pure exp; beta<1 → decelerating;
         beta>1 → super-exponential)

Usage:
    btc_venv/bin/python3 tools/fit_sexp.py               # fit and print
    btc_venv/bin/python3 tools/fit_sexp.py --update       # fit and update btc_core/
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


def sexp_log10(t, A, B, beta):
    """Stretched exponential: log10(price) = A + B * t^beta."""
    return A + B * np.power(np.maximum(t, 1e-9), beta)


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
    #   A    in [-5, 5]    — intercept has wide range depending on B/beta trade
    #   B    in [0.1, 10]  — positive scale; typical BTC stretched-exp gives B~2-4
    #   beta in [0.1, 2]   — stretching exponent; beta=1 is pure exp
    # beta bounded to [0.25, 1.5] — below 0.25 the t^beta term collapses
    # toward log-like behaviour and the fit becomes degenerate with PL.
    # Above 1.5 the model blows up super-exponentially.
    bounds_lo = [-10.0, 0.1, 0.25]
    bounds_hi = [5.0, 15.0, 1.5]

    bounds = list(zip(bounds_lo, bounds_hi))

    def objective(params):
        pred = sexp_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("  Running differential evolution...")
    res = differential_evolution(objective, bounds, maxiter=5000, seed=42,
                                 tol=1e-14, polish=True, popsize=30,
                                 workers=1)
    try:
        popt, _ = curve_fit(sexp_log10, t_fit, lp_fit, p0=res.x,
                            bounds=(bounds_lo, bounds_hi), maxfev=20000)
    except Exception:
        popt = res.x

    A, B, beta = popt
    pred = sexp_log10(t_fit, *popt)
    r2 = 1.0 - np.sum((lp_fit - pred) ** 2) / ss_tot
    sigma = float(np.std(lp_fit - pred))

    print(f"\nFitted Stretched Exponential parameters:")
    print(f"  A     = {A:.6f}   (log10 intercept at t=0)")
    print(f"  B     = {B:.6f}   (coefficient)")
    print(f"  beta  = {beta:.6f}   (stretching exponent; β=1 → pure exp)")
    print(f"  R²    = {r2:.6f}")
    print(f"  σ     = {sigma:.6f}")

    if update:
        print("\nUpdating btc_core/ ...")
        core_path = os.path.join(ROOT, "btc_core", "_simple.py")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [("_A", A), ("_B", B), ("_beta", beta)]
        cls_pos = src.find("class StretchedExponentialModel")
        if cls_pos == -1:
            print("  WARNING: could not find StretchedExponentialModel class")
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
                    cls_pos = src.find("class StretchedExponentialModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in StretchedExponentialModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core/ updated.")
    else:
        print("\nRun with --update to write to btc_core/")


if __name__ == "__main__":
    main()
