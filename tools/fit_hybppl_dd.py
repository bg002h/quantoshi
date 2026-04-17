#!/usr/bin/env python3
"""Fit HybPPL_DD (double-damped, non-excess) parameters.

Model: log10(price) = A + B*log10(t)
                    + C1*t^(-D1)*cos(W_log*ln(t) + PHI1)
                    + C2*t^(-D2)*cos(W_cal*t + PHI2)

Both oscillators have independent damping exponents (D1, D2).
10 parameters fit directly to log_price (not excess).

The 10 params are written to btc_core.py::HybPPLDDModel.

Usage:
    btc_venv/bin/python3 tools/fit_hybppl_dd.py             # fit + print
    btc_venv/bin/python3 tools/fit_hybppl_dd.py --update    # write to btc_core/
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


def hybppl_dd_log10(t, A, B, C1, W_log, PHI1, D1, C2, W_cal, PHI2, D2):
    t_safe = np.maximum(t, 0.1)
    damped_log = C1 * t_safe ** (-D1) * np.cos(W_log * np.log(t_safe) + PHI1)
    damped_cal = C2 * t_safe ** (-D2) * np.cos(W_cal * t_safe + PHI2)
    return A + B * np.log10(t_safe) + damped_log + damped_cal


def main():
    update = "--update" in sys.argv

    print("Loading prices...")
    pd_ = load_prices("BitcoinPricesDaily.csv")
    t = pd_.df_full["years"].values
    log_p = pd_.df_full["log_price"].values

    mask = t >= 1.0
    t_fit = t[mask]
    lp_fit = log_p[mask]
    print(f"  {len(t_fit)} data points (t >= 1.0)")

    bounds = [
        (-2.0, 0.0),       # A
        (3.0, 7.0),        # B
        (0.01, 3.0),       # C1
        (2.0, 40.0),       # W_log
        (-np.pi, np.pi),   # PHI1
        (0.01, 2.0),       # D1  (log-periodic damping)
        (0.0, 2.0),        # C2
        (0.5, 10.0),       # W_cal (rad/yr)
        (-np.pi, np.pi),   # PHI2
        (0.001, 2.0),      # D2  (calendar-periodic damping)
    ]

    def objective(params):
        pred = hybppl_dd_log10(t_fit, *params)
        return float(np.sum((lp_fit - pred) ** 2))

    print("Running differential evolution (10 params, double-damped, non-excess)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=3000, seed=42, tol=1e-12, polish=True, workers=1,
    )

    A, B, C1, W_log, PHI1, D1, C2, W_cal, PHI2, D2 = result.x
    pred = hybppl_dd_log10(t_fit, *result.x)
    resid = lp_fit - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((lp_fit - np.mean(lp_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(resid))
    T_yr = 2.0 * np.pi / W_cal

    print(f"\nFitted HybPPL_DD parameters:")
    print(f"  A     = {A:.6f}")
    print(f"  B     = {B:.6f}")
    print(f"  C1    = {C1:.6f}")
    print(f"  W_log = {W_log:.6f}")
    print(f"  PHI1  = {PHI1:.6f}")
    print(f"  D1    = {D1:.6f}")
    print(f"  C2    = {C2:.6f}")
    print(f"  W_cal = {W_cal:.6f} rad/yr  (T = {T_yr:.2f} years)")
    print(f"  PHI2  = {PHI2:.6f}")
    print(f"  D2    = {D2:.6f}")
    print(f"  R\u00b2   = {r2:.6f}")
    print(f"  \u03c3     = {sigma:.6f}")

    if update:
        print("\nUpdating btc_core.py::HybPPLDDModel...")
        core_path = os.path.join(ROOT, "btc_core", "_hybppl_eppl.py")
        import shutil
        shutil.copy2(core_path, core_path + ".bak")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [
            ("_A", A), ("_B", B),
            ("_C1", C1), ("_W_log", W_log), ("_PHI1", PHI1),
            ("_D1", D1), ("_C2", C2), ("_W_cal", W_cal), ("_PHI2", PHI2),
            ("_D2", D2),
        ]
        cls_pos = src.find("class HybPPLDDModel")
        if cls_pos == -1:
            print("  ERROR: could not find HybPPLDDModel class")
            sys.exit(1)
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
                cls_pos = src.find("class HybPPLDDModel")
                next_class = src.find("\nclass ", cls_pos + 1)
                cls_end = next_class if next_class != -1 else len(src)
                section = src[cls_pos:cls_end]
                print(f"  {name} = {new_val.strip()}")
            else:
                print(f"  WARNING: could not find {name} in HybPPLDDModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core/ updated.")
    else:
        print("\nRun with --update to write to btc_core/")


if __name__ == "__main__":
    main()
