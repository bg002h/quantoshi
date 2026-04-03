#!/usr/bin/env python3
"""Build model_data_ef.pkl -- Empirical Floor model via model_toolkit.

Usage:
    btc_venv/bin/python3 tools/build_ef_model.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import numpy as np
from model_toolkit.data import load_prices
from model_toolkit.support import fixed_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
from model_toolkit.composite import build_composite, build_comp_by_n
from model_toolkit.bands import fit_asymmetric_sigma, EF_QUANTILES
from model_toolkit.export import build_ef_pkl_dict, write_pkl

EF_SLOPE = 5.248017
EF_INTERCEPT = -1.630623


def main():
    print("=" * 70)
    print("Building Empirical Floor model")
    print("=" * 70)

    prices = load_prices("BitcoinPricesDaily.csv")
    sup = fixed_support(EF_INTERCEPT, EF_SLOPE, prices)
    print(f"  EF support: slope={sup.slope}, intercept={sup.intercept}")

    fitted = fit_sequential(prices, sup)
    major, minor = classify(fitted, n_major=5)
    f_maj, f_min = predict_future(major, minor, t_last_data=prices.years[-1],
                                   n_major=3, n_minor=1)
    all_future = sorted(f_maj + f_min, key=lambda b: b["t_rise"])
    print(f"  {len(major)} major + {len(minor)} minor fitted")
    print(f"  {len(f_maj)} major + {len(f_min)} minor predicted")

    comp = build_composite(sup, fitted, prices)
    cbn = build_comp_by_n(sup, fitted, all_future, comp.t_grid,
                           comp.hist_K_max, comp.total_plot)
    print(f"  R2={comp.r2:.6f}, comp_by_n: {len(cbn)} arrays")

    # Sigma: use full dataset (df_full) prices and the max-future composite
    full_log_prices = np.log10(np.maximum(prices.df_full["price"].values, 1e-10))
    full_years = prices.df_full["years"].values
    sigma = fit_asymmetric_sigma(full_log_prices, np.array(cbn[-1]),
                                  comp.t_grid, full_years)
    print(f"  sigma0_up={sigma.sigma0_up:.4f}, alpha_up={sigma.alpha_up:.4f}")

    pkl_dict = build_ef_pkl_dict(
        sup, comp, cbn, sigma, fitted,
        prices.df_full["years"].tolist(),
        prices.df_full["price"].tolist(),
        EF_QUANTILES,
    )
    write_pkl(pkl_dict, os.path.join(ROOT, "model_data_ef.pkl"))
    print("Done.")


if __name__ == "__main__":
    main()
