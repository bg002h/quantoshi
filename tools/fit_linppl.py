#!/usr/bin/env python3
"""Fit LinPPL (Linear-periodic Power Law) model to Bitcoin price history.

Model: log10(price) = A + B*log10(t) + C*t^(-D)*cos(W_cal*t + PHI)

Unlike LPPL which oscillates in log-time (ω·ln(t)), LinPPL oscillates in
calendar time (ω·t). Designed to match Bitcoin's ~4-year halving cycle
which is constant in calendar years.

Usage:
    btc_venv/bin/python3 tools/fit_linppl.py              # fit and print
    btc_venv/bin/python3 tools/fit_linppl.py --update      # fit and update btc_core.py
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


def linppl_log10(t, A, B, C, W_cal, PHI, D):
    """Evaluate LinPPL model in log10 space.

    W_cal is angular frequency in radians/year (2π/T where T is period in years).
    """
    t_safe = np.maximum(t, 0.1)
    envelope = C * t_safe ** (-D)
    return A + B * np.log10(t_safe) + envelope * np.cos(W_cal * t_safe + PHI)


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

    # Bounds for DE
    # W_cal range: corresponds to calendar period T = 2π/W_cal
    # W_cal ∈ [0.5, 10] → T ∈ [0.63, 12.6] years
    # Primary interest: halving cycle at T=4yr → W_cal ≈ 1.57
    # Mid-cycle rallies at T=2yr → W_cal ≈ 3.14
    # Annual noise at T=1yr → W_cal ≈ 6.28
    bounds = [
        (-3.0, 1.0),     # A
        (3.0, 7.0),      # B
        (0.01, 3.0),     # C
        (0.5, 10.0),     # W_cal (rad/yr)
        (-np.pi, np.pi), # PHI
        (0.01, 2.0),     # D
    ]

    def objective(params):
        pred = linppl_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("Running differential evolution...")
    result = differential_evolution(
        objective, bounds,
        maxiter=2000,
        seed=42,
        tol=1e-12,
        polish=True,
        workers=1,
    )

    A, B, C, W_cal, PHI, D = result.x
    pred = linppl_log10(t_fit, A, B, C, W_cal, PHI, D)
    ss_res = np.sum((lp_fit - pred) ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))
    T_years = 2.0 * np.pi / W_cal

    # Compare to LPPL
    from btc_core import LPPLModel as LM
    ts_ = np.maximum(t_fit, 0.1)
    pred_lppl = (LM._A + LM._B * np.log10(ts_)
                 + LM._C * ts_**(-LM._D) * np.cos(LM._W * np.log(ts_) + LM._PHI))
    r2_lppl = 1.0 - np.sum((lp_fit - pred_lppl)**2) / ss_tot

    print(f"\nFitted LinPPL parameters:")
    print(f"  A     = {A:.6f}")
    print(f"  B     = {B:.6f}")
    print(f"  C     = {C:.6f}")
    print(f"  W_cal = {W_cal:.6f}  rad/yr  (T = {T_years:.2f} years)")
    print(f"  PHI   = {PHI:.6f}")
    print(f"  D     = {D:.6f}")
    print(f"  R²    = {r2:.6f}  (vs LPPL R²={r2_lppl:.6f}, Δ={r2 - r2_lppl:+.6f})")
    print(f"  σ     = {sigma:.6f}")

    if update:
        print("\nUpdating btc_core.py...")
        core_path = os.path.join(ROOT, "btc_core.py")
        import shutil
        shutil.copy2(core_path, core_path + ".bak")
        print(f"  Backup saved to btc_core.py.bak")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [
            ("_A", A), ("_B", B), ("_C", C),
            ("_W", W_cal), ("_PHI", PHI), ("_D", D),
        ]
        # Find LinPPLModel class and update only within it
        cls_pos = src.find("class LinPPLModel")
        if cls_pos == -1:
            print("  WARNING: could not find LinPPLModel class")
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
                    cls_pos = src.find("class LinPPLModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in LinPPLModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core.py updated.")
    else:
        print("\nRun with --update to write to btc_core.py")


if __name__ == "__main__":
    main()
