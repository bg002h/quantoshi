#!/usr/bin/env python3
"""Fit Gompertz model to Bitcoin price history.

Model: log10(price) = K * exp(-exp(-r * (t - t0)))

K = carrying capacity (log10 of max price), r = growth rate,
t0 = inflection point (years since genesis).

Usage:
    btc_venv/bin/python3 tools/fit_gompertz.py              # fit and print
    btc_venv/bin/python3 tools/fit_gompertz.py --update      # fit and update the btc_core/ package
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


def gompertz_log10(t, K, r, t0):
    """Gompertz: log10(price) = K * exp(-exp(-r * (t - t0)))."""
    return K * np.exp(-np.exp(-r * (t - t0)))


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

    # Fit Gompertz only — the model class always evaluates Gompertz
    bounds_lo = [4.0, 0.01, 1.0]
    bounds_hi = [12.0, 2.0, 20.0]
    p0 = [7.0, 0.15, 8.0]

    try:
        popt, _ = curve_fit(gompertz_log10, t_fit, lp_fit, p0=p0,
                            bounds=(bounds_lo, bounds_hi), maxfev=10000)
    except Exception:
        # Fallback: use DE
        print("  curve_fit failed, trying differential_evolution...")
        bounds = [(4.0, 12.0), (0.01, 2.0), (1.0, 20.0)]

        def objective(params):
            return np.sum((lp_fit - gompertz_log10(t_fit, *params)) ** 2)

        res = differential_evolution(objective, bounds, maxiter=3000, seed=42,
                                     tol=1e-12, polish=True)
        popt = res.x

    K, r, t0 = popt
    pred = gompertz_log10(t_fit, *popt)
    r2 = 1.0 - np.sum((lp_fit - pred) ** 2) / ss_tot
    sigma = float(np.std(lp_fit - pred))
    max_price = 10.0 ** K

    print(f"\nFitted Gompertz parameters:")
    print(f"  K     = {K:.6f}  (log10 max price = ${max_price:,.0f})")
    print(f"  r     = {r:.6f}  (growth rate)")
    print(f"  t0    = {t0:.6f}  (inflection, years since genesis)")
    print(f"  R²    = {r2:.6f}")
    print(f"  σ     = {sigma:.6f}")

    if update:
        print("\nUpdating btc_core/ ...")
        core_path = os.path.join(ROOT, "btc_core", "_simple.py")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [("_K", K), ("_r", r), ("_t0", t0)]
        cls_pos = src.find("class GompertzModel")
        if cls_pos == -1:
            print("  WARNING: could not find GompertzModel class")
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
                    cls_pos = src.find("class GompertzModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in GompertzModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core/ updated.")
    else:
        print("\nRun with --update to write to btc_core/")


if __name__ == "__main__":
    main()
