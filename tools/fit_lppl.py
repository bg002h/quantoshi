#!/usr/bin/env python3
"""Fit LPPL (Log-Periodic Power Law) model to Bitcoin price history.

Model: log10(price) = A + B*log10(t) + C*t^(-D)*cos(W*ln(t) + PHI)

Uses scipy differential_evolution for global optimization, then
optionally updates btc_core.py with the fitted parameters.

Usage:
    btc_venv/bin/python3 tools/fit_lppl.py              # fit and print
    btc_venv/bin/python3 tools/fit_lppl.py --update      # fit and update btc_core.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution


def lppl_log10(t, A, B, C, W, PHI, D):
    """Evaluate LPPL model in log10 space."""
    t_safe = np.maximum(t, 0.1)
    envelope = C * t_safe ** (-D)
    return A + B * np.log10(t_safe) + envelope * np.cos(W * np.log(t_safe) + PHI)


def main():
    update = "--update" in sys.argv

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    log_p = prices.df_full["log_price"].values

    # Filter to t >= 1.0 (matching LPPLModel.__init__)
    mask = t >= 1.0
    t_fit = t[mask]
    lp_fit = log_p[mask]
    print(f"  {len(t_fit)} data points (t >= 1.0)")

    # Bounds for differential evolution
    # A: intercept (~-2 to 1)
    # B: power law slope (~3 to 7)
    # C: oscillation amplitude (~0.01 to 3)
    # W: angular frequency (~2 to 15, ~4yr cycle in log-time)
    # PHI: phase (~-pi to pi)
    # D: damping exponent (~0.01 to 2)
    bounds = [
        (-3.0, 1.0),    # A
        (3.0, 7.0),     # B
        (0.01, 3.0),    # C
        (2.0, 15.0),    # W
        (-np.pi, np.pi),# PHI
        (0.01, 2.0),    # D
    ]

    def objective(params):
        A, B, C, W, PHI, D = params
        pred = lppl_log10(t_fit, A, B, C, W, PHI, D)
        return np.sum((lp_fit - pred) ** 2)

    print("Running differential evolution (this may take a minute)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=2000,
        seed=42,
        tol=1e-12,
        polish=True,
        workers=1,
    )

    A, B, C, W, PHI, D = result.x
    pred = lppl_log10(t_fit, A, B, C, W, PHI, D)
    ss_res = np.sum((lp_fit - pred) ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))

    print(f"\nFitted LPPL parameters:")
    print(f"  A   = {A:.6f}")
    print(f"  B   = {B:.6f}")
    print(f"  C   = {C:.6f}")
    print(f"  W   = {W:.6f}")
    print(f"  PHI = {PHI:.6f}")
    print(f"  D   = {D:.6f}")
    print(f"  R²  = {r2:.6f}")
    print(f"  σ   = {sigma:.6f}")

    if update:
        print("\nUpdating btc_core.py...")
        core_path = os.path.join(ROOT, "btc_core.py")
        # Backup before modifying
        backup_path = core_path + ".bak"
        import shutil
        shutil.copy2(core_path, backup_path)
        print(f"  Backup saved to {backup_path}")
        with open(core_path) as f:
            src = f.read()

        replacements = [
            ("_A", A), ("_B", B), ("_C", C),
            ("_W", W), ("_PHI", PHI), ("_D", D),
        ]
        for name, val in replacements:
            import re
            # Match lines like: _A   = -1.155084
            pattern = rf"(    {name}\s*=\s*)[^\n]+"
            new_val = f"{val:>11.6f}" if val >= 0 else f"{val:.6f}"
            replacement = rf"\g<1>{new_val}"
            new_src, count = re.subn(pattern, replacement, src, count=1)
            if count != 1:
                print(f"  WARNING: could not find unique match for {name}")
            else:
                src = new_src
                print(f"  {name} = {new_val.strip()}")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core.py updated.")
    else:
        print("\nRun with --update to write these to btc_core.py")


if __name__ == "__main__":
    main()
