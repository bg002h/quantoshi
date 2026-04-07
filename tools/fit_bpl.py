#!/usr/bin/env python3
"""Fit Broken Power Law model to Bitcoin price history.

Model:
    t < t_break:  log10(price) = a1 + b1 * log10(t)
    t >= t_break: log10(price) = a2 + b2 * log10(t)
    Continuity:   a2 = a1 + (b1 - b2) * log10(t_break)

4 free parameters: a1, b1, t_break, b2 (a2 derived).

Usage:
    btc_venv/bin/python3 tools/fit_bpl.py              # fit and print
    btc_venv/bin/python3 tools/fit_bpl.py --update      # fit and update btc_core.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution


def bpl_log10(t, a1, b1, t_break, b2):
    """Broken power law with continuity at t_break."""
    t_safe = np.maximum(t, 0.1)
    lt = np.log10(t_safe)
    lt_break = np.log10(t_break)
    a2 = a1 + (b1 - b2) * lt_break
    return np.where(t_safe < t_break,
                    a1 + b1 * lt,
                    a2 + b2 * lt)


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

    bounds = [
        (-3.0, 1.0),    # a1 (early intercept)
        (3.0, 8.0),     # b1 (early slope)
        (2.0, 12.0),    # t_break (breakpoint, years since genesis)
        (2.0, 7.0),     # b2 (late slope)
    ]

    def objective(params):
        pred = bpl_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("Running differential evolution (4 params)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=3000, seed=42, tol=1e-12, polish=True, workers=1,
    )

    a1, b1, t_break, b2 = result.x
    a2 = a1 + (b1 - b2) * np.log10(t_break)
    pred = bpl_log10(t_fit, *result.x)
    r2 = 1.0 - np.sum((lp_fit - pred) ** 2) / ss_tot
    sigma = float(np.std(lp_fit - pred))

    # Compare to single power law
    from scipy.stats import linregress
    sl, intc, *_ = linregress(np.log10(np.maximum(t_fit, 0.1)), lp_fit)
    pred_pl = intc + sl * np.log10(np.maximum(t_fit, 0.1))
    r2_pl = 1.0 - np.sum((lp_fit - pred_pl) ** 2) / ss_tot

    # Convert t_break to calendar year
    import pandas as pd
    genesis = pd.Timestamp("2009-07-25")
    break_date = genesis + pd.Timedelta(days=t_break * 365.25)

    print(f"\nFitted Broken Power Law parameters:")
    print(f"  a1      = {a1:.6f}  (early intercept)")
    print(f"  b1      = {b1:.6f}  (early slope)")
    print(f"  t_break = {t_break:.6f}  ({break_date.strftime('%Y-%m-%d')})")
    print(f"  b2      = {b2:.6f}  (late slope)")
    print(f"  a2      = {a2:.6f}  (late intercept, derived)")
    print(f"  R²      = {r2:.6f}  (vs single PL R²={r2_pl:.6f}, Δ={r2-r2_pl:+.6f})")
    print(f"  σ       = {sigma:.6f}")
    if b2 < b1:
        print(f"  Slope decreased: {b1:.3f} → {b2:.3f} (growth slowing)")
    else:
        print(f"  Slope increased: {b1:.3f} → {b2:.3f} (growth accelerating)")

    if update:
        print("\nUpdating btc_core.py...")
        core_path = os.path.join(ROOT, "btc_core.py")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [
            ("_a1", a1), ("_b1", b1), ("_t_break", t_break), ("_b2", b2),
        ]
        cls_pos = src.find("class BrokenPowerLawModel")
        if cls_pos == -1:
            print("  WARNING: could not find BrokenPowerLawModel class")
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
                    cls_pos = src.find("class BrokenPowerLawModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in BrokenPowerLawModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core.py updated.")
    else:
        print("\nRun with --update to write to btc_core.py")


if __name__ == "__main__":
    main()
