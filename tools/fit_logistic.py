#!/usr/bin/env python3
"""Fit (true) Logistic S-Curve model to Bitcoin price history.

Model: log10(price) = K / (1 + exp(-r * (t - t0)))

Three parameters:
  K  = saturation log10 price  (dimensionless; 10^K is max USD)
  r  = growth rate
  t0 = inflection point (years since genesis)

This is the SYMMETRIC logistic, distinct from GompertzModel's asymmetric
S-curve. Use `fit_gompertz.py` to fit the Gompertz variant.

Usage:
    btc_venv/bin/python3 tools/fit_logistic.py               # fit and print
    btc_venv/bin/python3 tools/fit_logistic.py --update       # fit and update btc_core/
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from _patch_class_attrs import apply_and_report
from model_toolkit.data import load_prices
from scipy.optimize import curve_fit, differential_evolution

CORE_PATH = os.path.join(ROOT, "btc_core", "_simple.py")
CLASS_NAME = "LogisticSCurveModel"


def logistic_log10(t, K, r, t0):
    """Symmetric logistic: log10(price) = K / (1 + exp(-r * (t - t0)))."""
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

    # Bounds:
    #   K  in [3, 15]    — log10 saturation price; BTC plateau anywhere from
    #                      $1000 (K=3) up to $1 quadrillion (K=15)
    #   r  in [0.05, 2]  — growth rate
    #   t0 in [1, 30]    — inflection year; center of the sigmoid
    bounds_lo = [3.0, 0.05, 1.0]
    bounds_hi = [15.0, 2.0, 30.0]

    bounds = list(zip(bounds_lo, bounds_hi))

    def objective(params):
        pred = logistic_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("  Running differential evolution...")
    res = differential_evolution(objective, bounds, maxiter=5000, seed=42,
                                 tol=1e-14, polish=True, popsize=30,
                                 workers=1)
    try:
        popt, _ = curve_fit(logistic_log10, t_fit, lp_fit, p0=res.x,
                            bounds=(bounds_lo, bounds_hi), maxfev=20000)
    except Exception:
        popt = res.x

    K, r, t0 = popt
    pred = logistic_log10(t_fit, *popt)
    r2 = 1.0 - np.sum((lp_fit - pred) ** 2) / ss_tot
    sigma = float(np.std(lp_fit - pred))
    max_price = 10.0 ** K

    print(f"\nFitted Logistic (S-curve) parameters:")
    print(f"  K     = {K:.6f}   (log10 saturation price = ${max_price:,.0f})")
    print(f"  r     = {r:.6f}   (growth rate)")
    print(f"  t0    = {t0:.6f}   (inflection, years since genesis)")
    print(f"  R²    = {r2:.6f}")
    print(f"  σ     = {sigma:.6f}")

    if update:
        # Scoped, atomic, guarded -- tools/_patch_class_attrs.py. `_K`, `_r`
        # and `_t0` are ALL shared with GompertzModel (and `_t0` with
        # SaturatingPowerLawModel too), so an unscoped patch here would
        # silently rewrite another model's fit. Do not inline a regex.
        if apply_and_report(CORE_PATH, CLASS_NAME,
                            {"_K": K, "_r": r, "_t0": t0}):
            print("btc_core/ updated.")
    else:
        print("\nRun with --update to write to btc_core/")


if __name__ == "__main__":
    main()
