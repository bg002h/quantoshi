#!/usr/bin/env python3
"""
Fit asymmetric shrinking sigma parameters for a bubble model pkl.

Computes σ_up(t) = σ₀_up × t^(-α_up) and σ_down(t) = σ₀_down × t^(-α_down)
from windowed residuals of log10(price) vs log10(composite).

Usage:
    btc_venv/bin/python3 tools/fit_sigma.py --pkl btc_app/model_data.pkl --type bm
    btc_venv/bin/python3 tools/fit_sigma.py --pkl btc_app/model_data_ef.pkl --type ef
"""

import argparse
import pickle
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path


def _fit_sigma(pkl_path, model_type):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # Load historical data
    prices = np.asarray(d["price_prices"], float)
    years = np.asarray(d["price_years"], float)

    if model_type == "bm":
        comp_grid_y = np.asarray(d["years_plot_bm"], float)
        comp_grid_p = np.asarray(d["bm_comp_by_n"][-1], float)
    else:  # ef
        comp_grid_y = np.asarray(d["years_plot"], float)
        comp_grid_p = np.asarray(d["comp_by_n"][-1], float)

    log_p = np.log10(prices)
    log_comp = np.interp(years,
                         comp_grid_y,
                         np.log10(np.maximum(comp_grid_p, 1e-10)))
    residuals = log_p - log_comp

    # Windowed fit in 20 log-time bins
    log_t = np.log10(years)
    n_bins = 20
    bin_edges = np.linspace(log_t.min(), log_t.max(), n_bins + 1)

    t_centers, sigma_up_bins, sigma_down_bins = [], [], []
    for b in range(n_bins):
        mask = (log_t >= bin_edges[b]) & (log_t < bin_edges[b + 1])
        if mask.sum() < 10:
            continue
        r = residuals[mask]
        t_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2))
        r_up = r[r >= 0]
        r_down = r[r < 0]
        sigma_up_bins.append(np.std(r_up) if len(r_up) > 3 else np.nan)
        sigma_down_bins.append(np.std(np.abs(r_down)) if len(r_down) > 3 else np.nan)

    t_centers = np.array(t_centers)
    sigma_up_bins = np.array(sigma_up_bins)
    sigma_down_bins = np.array(sigma_down_bins)

    def sigma_model(t, sigma0, alpha):
        return sigma0 * t ** (-alpha)

    # Fit upside
    valid = ~np.isnan(sigma_up_bins) & (sigma_up_bins > 0)
    popt_up, _ = curve_fit(sigma_model, t_centers[valid], sigma_up_bins[valid],
                           p0=[0.5, 0.3], bounds=([0.01, -1], [5.0, 3.0]))

    # Fit downside
    valid = ~np.isnan(sigma_down_bins) & (sigma_down_bins > 0)
    popt_down, _ = curve_fit(sigma_model, t_centers[valid], sigma_down_bins[valid],
                             p0=[0.3, 0.3], bounds=([0.01, -1], [5.0, 3.0]))

    sigma0_up, alpha_up = float(popt_up[0]), float(popt_up[1])
    sigma0_down, alpha_down = float(popt_down[0]), float(popt_down[1])

    # Write to pkl with appropriate key prefix
    prefix = "bm_" if model_type == "bm" else ""
    d[f"{prefix}sigma0_up"] = sigma0_up
    d[f"{prefix}alpha_up"] = alpha_up
    d[f"{prefix}sigma0_down"] = sigma0_down
    d[f"{prefix}alpha_down"] = alpha_down

    # Remove old constant sigma if present (EF)
    d.pop("sigma", None)

    with open(pkl_path, "wb") as f:
        pickle.dump(d, f)

    print(f"σ₀_up={sigma0_up:.4f}  α_up={alpha_up:.4f}")
    print(f"σ₀_down={sigma0_down:.4f}  α_down={alpha_down:.4f}")
    print(f"Asymmetry ratio (α_down/α_up): {alpha_down / alpha_up:.2f}")
    print(f"Written to {pkl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit asymmetric shrinking sigma for bubble model pkl")
    parser.add_argument("--pkl", required=True, help="Path to pkl file")
    parser.add_argument("--type", required=True, choices=["bm", "ef"],
                        help="Model type: bm or ef")
    args = parser.parse_args()
    _fit_sigma(args.pkl, args.type)


if __name__ == "__main__":
    main()
