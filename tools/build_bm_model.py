#!/usr/bin/env python3
"""Build model_data.pkl -- BM model via model_toolkit.

Usage:
    btc_venv/bin/python3 tools/build_bm_model.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import numpy as np
from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
from model_toolkit.composite import build_composite, build_comp_by_n
from model_toolkit.bands import fit_qr_channels, fit_asymmetric_sigma, BM_QUANTILES
from model_toolkit.export import build_bm_pkl_dict, write_pkl


def main():
    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    print(f"  {len(prices.df)} fitting points, {len(prices.df_full)} total")

    print("Fitting support line...")
    sup = fit_support(prices)
    print(f"  slope={sup.slope:.4f}, intercept={sup.intercept:.4f}")

    print("Fitting bubbles...")
    fitted = fit_sequential(prices, sup)
    major, minor = classify(fitted, n_major=5)
    print(f"  {len(major)} major + {len(minor)} minor")

    print("Predicting future bubbles...")
    f_maj, f_min = predict_future(major, minor, t_last_data=prices.years[-1],
                                   n_major=3, n_minor=1)
    all_future = sorted(f_maj + f_min, key=lambda b: b["t_rise"])
    print(f"  {len(f_maj)} major + {len(f_min)} minor predicted")

    print("Building composite...")
    comp = build_composite(sup, fitted, prices)
    cbn = build_comp_by_n(sup, fitted, all_future, comp.t_grid,
                           comp.hist_K_max, comp.total_plot)
    print(f"  R2={comp.r2:.4f}, comp_by_n: {len(cbn)} arrays")

    print("Fitting QR channels...")
    qr = fit_qr_channels(prices, BM_QUANTILES)
    print(f"  {len(qr.fits)} quantiles, OLS slope={qr.ols_slope:.4f}")

    print("Fitting sigma...")
    # Use comp_by_n[-1] (full composite incl. future bubbles) and df_full data
    # to match original fit_sigma.py behavior
    full_log_prices = np.log10(np.maximum(prices.df_full["price"].values, 1e-10))
    full_years = prices.df_full["years"].values
    sigma = fit_asymmetric_sigma(full_log_prices, np.array(cbn[-1]),
                                  comp.t_grid, full_years)
    print(f"  sigma0_up={sigma.sigma0_up:.4f}, alpha_up={sigma.alpha_up:.4f}")

    print("Writing pkl...")
    pkl_path = os.path.join(ROOT, "model_data.pkl")
    write_pkl(build_bm_pkl_dict(prices, sup, comp, cbn, qr, sigma), pkl_path)
    print("Done.")


if __name__ == "__main__":
    main()
