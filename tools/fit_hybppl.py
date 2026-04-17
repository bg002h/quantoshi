#!/usr/bin/env python3
"""Fit HybPPL (hybrid log+linear PPL) model to Bitcoin price history.

Model: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(ω_log*ln(t)+φ1)
                    + C2*cos(ω_cal*t+φ2)

Combines LPPL's log-periodic damped oscillation with LinPPL's
linear-periodic undamped calendar cycle. 9 parameters.

Usage:
    btc_venv/bin/python3 tools/fit_hybppl.py              # fit and print
    btc_venv/bin/python3 tools/fit_hybppl.py --update      # fit and update the btc_core/ package
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


def hybppl_log10(t, A, B, C1, W_log, PHI1, D, C2, W_cal, PHI2):
    """Hybrid: log-periodic damped + linear-periodic undamped."""
    t_safe = np.maximum(t, 0.1)
    damped = C1 * t_safe ** (-D) * np.cos(W_log * np.log(t_safe) + PHI1)
    undamped = C2 * np.cos(W_cal * t_safe + PHI2)
    return A + B * np.log10(t_safe) + damped + undamped


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

    bounds = [
        (-3.0, 1.0),     # A
        (3.0, 7.0),      # B
        (0.01, 3.0),     # C1
        (2.0, 40.0),     # W_log (log-time angular freq)
        (-np.pi, np.pi), # PHI1
        (0.01, 2.0),     # D
        (0.0, 2.0),      # C2
        (0.5, 10.0),     # W_cal (calendar angular freq, rad/yr)
        (-np.pi, np.pi), # PHI2
    ]

    def objective(params):
        pred = hybppl_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("Running differential evolution (9 params)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=3000, seed=42, tol=1e-12, polish=True, workers=1,
    )

    A, B, C1, W_log, PHI1, D, C2, W_cal, PHI2 = result.x
    pred = hybppl_log10(t_fit, *result.x)
    ss_res = np.sum((lp_fit - pred) ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))
    T_cal_years = 2.0 * np.pi / W_cal

    # Compare to LPPL, LinPPL, LP2
    from btc_core import LPPLModel, LPPL2Model, LinPPLModel
    ts_ = np.maximum(t_fit, 0.1)
    pred_lppl = (LPPLModel._A + LPPLModel._B * np.log10(ts_)
                 + LPPLModel._C * ts_**(-LPPLModel._D)
                   * np.cos(LPPLModel._W * np.log(ts_) + LPPLModel._PHI))
    r2_lppl = 1.0 - np.sum((lp_fit - pred_lppl)**2) / ss_tot

    print(f"\nFitted HybPPL parameters:")
    print(f"  A     = {A:.6f}")
    print(f"  B     = {B:.6f}")
    print(f"  C1    = {C1:.6f}  (log-periodic amplitude)")
    print(f"  W_log = {W_log:.6f}")
    print(f"  PHI1  = {PHI1:.6f}")
    print(f"  D     = {D:.6f}")
    print(f"  C2    = {C2:.6f}  (linear-periodic amplitude)")
    print(f"  W_cal = {W_cal:.6f}  rad/yr  (T = {T_cal_years:.2f} years)")
    print(f"  PHI2  = {PHI2:.6f}")
    print(f"  R²    = {r2:.6f}  (vs LPPL R²={r2_lppl:.6f}, \u0394={r2-r2_lppl:+.6f})")
    print(f"  \u03c3     = {sigma:.6f}")
    print(f"  Amplitude ratio C2/C1 = {C2/C1:.3f}")

    if update:
        print("\nUpdating btc_core/ ...")
        core_path = os.path.join(ROOT, "btc_core", "_hybppl_eppl.py")
        import shutil
        shutil.copy2(core_path, core_path + ".bak")
        print(f"  Backup saved to btc_core.py.bak")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [
            ("_A", A), ("_B", B), ("_C", C1),
            ("_W", W_log), ("_PHI", PHI1), ("_D", D),
            ("_C2", C2), ("_W2", W_cal), ("_PHI2", PHI2),
        ]
        cls_pos = src.find("class HybPPLModel")
        if cls_pos == -1:
            print("  WARNING: could not find HybPPLModel class")
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
                    cls_pos = src.find("class HybPPLModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in HybPPLModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core/ updated.")
    else:
        print("\nRun with --update to write to btc_core/")


if __name__ == "__main__":
    main()
