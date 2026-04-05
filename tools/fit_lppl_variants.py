#!/usr/bin/env python3
"""Monthly refit of all LPPL variants with fitted parameters.

Fits and updates btc_core.py for:
  - LPPLModelW      (1 freq, log-time weighted)
  - LPPL2ModelW     (2 freqs, weighted)
  - LPPL3ModelW     (3 freqs, weighted)
  - LPPL4ModelW     (4 freqs, weighted)
  - LPPL4ModelN13   (4 freqs, unweighted, ω=13 band excluded)
  - LPPL4ModelWN13  (4 freqs, weighted, ω=13 excluded)

Uses log-time-uniform (1/t) weighting when --weighted variants.
Excludes secondary frequencies from [11.5, 14.5] when --no-13 variants.

Usage:
    btc_venv/bin/python3 tools/fit_lppl_variants.py             # fit + print
    btc_venv/bin/python3 tools/fit_lppl_variants.py --update    # fit + write to btc_core.py
"""
import os
import sys
import re
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")


def lp1_log10(t, A, B, C, W, PHI, D):
    ts = np.maximum(t, 0.1)
    return A + B * np.log10(ts) + C * ts**(-D) * np.cos(W * np.log(ts) + PHI)


def lp2_log10(t, A, B, C1, W1, PHI1, D, C2, W2, PHI2):
    ts = np.maximum(t, 0.1)
    return (A + B * np.log10(ts)
            + C1 * ts**(-D) * np.cos(W1 * np.log(ts) + PHI1)
            + C2 * np.cos(W2 * np.log(ts) + PHI2))


def lp3_log10(t, A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3):
    ts = np.maximum(t, 0.1)
    return (A + B * np.log10(ts)
            + C1 * ts**(-D) * np.cos(W1 * np.log(ts) + PHI1)
            + C2 * np.cos(W2 * np.log(ts) + PHI2)
            + C3 * np.cos(W3 * np.log(ts) + PHI3))


def lp4_log10(t, A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3, C4, W4, PHI4):
    ts = np.maximum(t, 0.1)
    return (A + B * np.log10(ts)
            + C1 * ts**(-D) * np.cos(W1 * np.log(ts) + PHI1)
            + C2 * np.cos(W2 * np.log(ts) + PHI2)
            + C3 * np.cos(W3 * np.log(ts) + PHI3)
            + C4 * np.cos(W4 * np.log(ts) + PHI4))


# All primary W bounds widened to 40; secondaries widened from 35 to 40
BOUNDS_LP1 = [
    (-1.5, -0.7), (4.5, 5.5), (0.1, 1.5),
    (2.0, 40.0), (-np.pi, np.pi), (0.0, 1.5),
]
BOUNDS_LP2 = [
    (-1.5, -0.7), (4.5, 5.5), (0.1, 1.5),
    (2.0, 40.0), (-np.pi, np.pi), (0.0, 1.5),
    (0.0, 0.5), (3.0, 40.0), (-np.pi, np.pi),
]
BOUNDS_LP3 = [
    (-1.5, -0.7), (4.5, 5.5), (0.1, 1.5),
    (2.0, 40.0), (-np.pi, np.pi), (0.05, 1.2),
    (0.0, 0.5), (3.0, 40.0), (-np.pi, np.pi),
    (0.0, 0.5), (3.0, 40.0), (-np.pi, np.pi),
]
BOUNDS_LP4 = [
    (-1.5, -0.7), (4.5, 5.5), (0.1, 1.5),
    (2.0, 40.0), (-np.pi, np.pi), (0.05, 1.2),
    (0.0, 0.5), (3.0, 40.0), (-np.pi, np.pi),
    (0.0, 0.5), (3.0, 40.0), (-np.pi, np.pi),
    (0.0, 0.5), (3.0, 40.0), (-np.pi, np.pi),
]

# Indices of W parameters for each model (0-indexed in bounds list)
W_INDICES_LP2 = [7]
W_INDICES_LP3 = [7, 10]
W_INDICES_LP4 = [7, 10, 13]


def fit_variant(fn, bounds, t_fit, lp_fit, weights=None, no_13_indices=None,
                maxiter=4000, label=""):
    """Fit one LPPL variant with optional weighting + W=13 exclusion."""
    if weights is None:
        weights = np.ones_like(t_fit)

    def obj(params):
        if no_13_indices is not None:
            for idx in no_13_indices:
                if 11.5 <= params[idx] <= 14.5:
                    return 1e10
        resid = lp_fit - fn(t_fit, *params)
        return np.sum(weights * resid**2)

    print(f"  Fitting {label}...")
    res = differential_evolution(
        obj, bounds,
        maxiter=maxiter, seed=42, tol=1e-12, polish=True, workers=1,
    )
    pred = fn(t_fit, *res.x)
    ss_res = np.sum((lp_fit - pred)**2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit))**2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))
    print(f"    R²={r2:.6f}, σ={sigma:.6f}")
    return res.x, r2, sigma


def update_class(class_name, param_names, values, core_path):
    """Update a model class's hardcoded params in btc_core.py."""
    with open(core_path) as f:
        src = f.read()

    cls_pos = src.find(f"class {class_name}")
    if cls_pos == -1:
        print(f"  WARNING: could not find {class_name}")
        return src
    next_class = src.find("\nclass ", cls_pos + 1)
    cls_end = next_class if next_class != -1 else len(src)
    section = src[cls_pos:cls_end]

    for name, val in zip(param_names, values):
        pattern = rf"(    {name}\s*=\s*)[^\n]+"
        new_val = f"{val:>11.6f}" if val >= 0 else f"{val:.6f}"
        match = re.search(pattern, section)
        if match:
            old_line = match.group(0)
            new_line = re.sub(pattern, rf"\g<1>{new_val}", old_line)
            new_section = section.replace(old_line, new_line, 1)
            src = src[:cls_pos] + new_section + src[cls_end:]
            cls_pos = src.find(f"class {class_name}")
            next_class = src.find("\nclass ", cls_pos + 1)
            cls_end = next_class if next_class != -1 else len(src)
            section = src[cls_pos:cls_end]
        else:
            print(f"    WARNING: could not find {name} in {class_name}")

    with open(core_path, "w") as f:
        f.write(src)
    return src


def main():
    update = "--update" in sys.argv

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    log_p = prices.df_full["log_price"].values
    mask = t >= 1.0
    t_fit = t[mask]
    lp_fit = log_p[mask]
    n = len(t_fit)
    print(f"  {n} data points (t >= 1.0)")

    # Log-time-uniform weights (normalized so total = N)
    w_log = 1.0 / t_fit
    w_log = w_log / np.sum(w_log) * n

    variants = [
        # (class_name, fit_fn, bounds, weights, no_13_idx, param_names, label)
        ("LPPLModelW",    lp1_log10, BOUNDS_LP1, w_log, None,
         ["_A", "_B", "_C", "_W", "_PHI", "_D"],
         "LPPL_w"),
        ("LPPL2ModelW",   lp2_log10, BOUNDS_LP2, w_log, None,
         ["_A", "_B", "_C", "_W", "_PHI", "_D", "_C2", "_W2", "_PHI2"],
         "LP2_w"),
        ("LPPL3ModelW",   lp3_log10, BOUNDS_LP3, w_log, None,
         ["_A", "_B", "_C", "_W", "_PHI", "_D",
          "_C2", "_W2", "_PHI2", "_C3", "_W3", "_PHI3"],
         "LP3_w"),
        ("LPPL4ModelW",   lp4_log10, BOUNDS_LP4, w_log, None,
         ["_A", "_B", "_C", "_W", "_PHI", "_D",
          "_C2", "_W2", "_PHI2", "_C3", "_W3", "_PHI3",
          "_C4", "_W4", "_PHI4"],
         "LP4_w"),
        ("LPPL4ModelN13", lp4_log10, BOUNDS_LP4, None, W_INDICES_LP4,
         ["_A", "_B", "_C", "_W", "_PHI", "_D",
          "_C2", "_W2", "_PHI2", "_C3", "_W3", "_PHI3",
          "_C4", "_W4", "_PHI4"],
         "LP4_n13"),
        ("LPPL4ModelWN13", lp4_log10, BOUNDS_LP4, w_log, W_INDICES_LP4,
         ["_A", "_B", "_C", "_W", "_PHI", "_D",
          "_C2", "_W2", "_PHI2", "_C3", "_W3", "_PHI3",
          "_C4", "_W4", "_PHI4"],
         "LP4_w_n13"),
    ]

    results = []
    for class_name, fn, bounds, weights, no_13_idx, names, label in variants:
        values, r2, sigma = fit_variant(
            fn, bounds, t_fit, lp_fit, weights, no_13_idx,
            maxiter=4000, label=label,
        )
        results.append((class_name, names, values, r2, sigma, label))

    if update:
        core_path = os.path.join(ROOT, "btc_core.py")
        shutil.copy2(core_path, core_path + ".bak")
        print(f"\nBackup: {core_path}.bak")
        print("Updating btc_core.py...")
        for class_name, names, values, r2, sigma, label in results:
            print(f"  {class_name} ({label}): R²={r2:.4f}")
            update_class(class_name, names, values, core_path)
        print("Done.")
    else:
        print("\nRun with --update to write to btc_core.py")
        print("\nSummary:")
        for class_name, names, values, r2, sigma, label in results:
            print(f"  {label:<12} R²={r2:.6f} σ={sigma:.6f}")


if __name__ == "__main__":
    main()
