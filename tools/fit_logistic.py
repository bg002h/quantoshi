#!/usr/bin/env python3
"""Fit Logistic Growth (Gompertz) model to Bitcoin price history.

Model: log10(price) = K * exp(-exp(-r * (t - t0)))

K = carrying capacity (log10 of max price), r = growth rate,
t0 = inflection point (years since genesis).

Usage:
    btc_venv/bin/python3 tools/fit_logistic.py              # fit and print
    btc_venv/bin/python3 tools/fit_logistic.py --update      # fit and update btc_core.py
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


def logistic_log10(t, K, r, t0):
    """Logistic: log10(price) = K / (1 + exp(-r * (t - t0)))."""
    return K / (1.0 + np.exp(-r * (t - t0)))


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

    # Try both Gompertz and Logistic, pick the better fit
    bounds_lo = [4.0, 0.01, 1.0]
    bounds_hi = [12.0, 2.0, 20.0]
    p0 = [7.0, 0.15, 8.0]

    results = {}
    for name, func in [("Gompertz", gompertz_log10), ("Logistic", logistic_log10)]:
        try:
            popt, _ = curve_fit(func, t_fit, lp_fit, p0=p0,
                                bounds=(bounds_lo, bounds_hi), maxfev=10000)
            pred = func(t_fit, *popt)
            r2 = 1.0 - np.sum((lp_fit - pred) ** 2) / ss_tot
            sigma = float(np.std(lp_fit - pred))
            results[name] = {"popt": popt, "r2": r2, "sigma": sigma, "func": func}
            print(f"  {name}: R² = {r2:.6f}, σ = {sigma:.6f}")
        except Exception as exc:
            print(f"  {name}: FAILED ({exc})")

    if not results:
        # Fallback: use DE on Gompertz
        print("  curve_fit failed, trying differential_evolution on Gompertz...")
        bounds = [(5.0, 10.0), (0.01, 1.0), (1.0, 20.0)]

        def objective(params):
            return np.sum((lp_fit - gompertz_log10(t_fit, *params)) ** 2)

        res = differential_evolution(objective, bounds, maxiter=3000, seed=42,
                                     tol=1e-12, polish=True)
        popt = res.x
        pred = gompertz_log10(t_fit, *popt)
        r2 = 1.0 - np.sum((lp_fit - pred) ** 2) / ss_tot
        sigma = float(np.std(lp_fit - pred))
        results["Gompertz"] = {"popt": popt, "r2": r2, "sigma": sigma,
                               "func": gompertz_log10}

    # Pick best
    best_name = max(results, key=lambda k: results[k]["r2"])
    best = results[best_name]
    K, r, t0 = best["popt"]
    r2 = best["r2"]
    sigma = best["sigma"]

    max_price = 10.0 ** K

    print(f"\nBest fit: {best_name}")
    print(f"  K     = {K:.6f}  (log10 max price = ${max_price:,.0f})")
    print(f"  r     = {r:.6f}  (growth rate)")
    print(f"  t0    = {t0:.6f}  (inflection, years since genesis)")
    print(f"  R²    = {r2:.6f}")
    print(f"  σ     = {sigma:.6f}")

    if best_name == "Logistic":
        print("  NOTE: Logistic fit was better than Gompertz.")
        print("        Model class uses Gompertz formula — consider switching if persistent.")

    if update:
        print("\nUpdating btc_core.py...")
        core_path = os.path.join(ROOT, "btc_core.py")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [("_K", K), ("_r", r), ("_t0", t0)]
        cls_pos = src.find("class LogisticModel")
        if cls_pos == -1:
            print("  WARNING: could not find LogisticModel class")
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
                    cls_pos = src.find("class LogisticModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in LogisticModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core.py updated.")
    else:
        print("\nRun with --update to write to btc_core.py")


if __name__ == "__main__":
    main()
