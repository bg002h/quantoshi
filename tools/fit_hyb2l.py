#!/usr/bin/env python3
"""Fit Hyb2L (HybPPL + 2nd log-periodic) model to Bitcoin price history.

Model: log10(price) = A + B*log10(t) + C1*t^(-D1)*cos(W1*ln(t)+P1)
                    + C2*cos(Wc*t+P2) + C3*t^(-D2)*cos(W2*ln(t)+P3)

Extends HybPPL with a second log-periodic damped oscillation. 13 parameters.

Usage:
    btc_venv/bin/python3 tools/fit_hyb2l.py              # fit and print
    btc_venv/bin/python3 tools/fit_hyb2l.py --update      # fit and update the btc_core/ package
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


def hyb2l_log10(t, A, B, C1, W1, PHI1, D1, C2, Wc, PHI2, C3, W2, PHI3, D2):
    """HybPPL + 2nd log-periodic damped oscillation."""
    t_safe = np.maximum(t, 0.1)
    damped1 = C1 * t_safe ** (-D1) * np.cos(W1 * np.log(t_safe) + PHI1)
    undamped = C2 * np.cos(Wc * t_safe + PHI2)
    damped2 = C3 * t_safe ** (-D2) * np.cos(W2 * np.log(t_safe) + PHI3)
    return A + B * np.log10(t_safe) + damped1 + undamped + damped2


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
        (2.0, 40.0),     # W1 (log-time angular freq)
        (-np.pi, np.pi), # PHI1
        (0.01, 2.0),     # D1
        (0.0, 2.0),      # C2
        (0.5, 10.0),     # Wc (calendar angular freq, rad/yr)
        (-np.pi, np.pi), # PHI2
        (0.001, 2.0),    # C3
        (2.0, 80.0),     # W2 (2nd log-time angular freq)
        (-np.pi, np.pi), # PHI3
        (0.01, 2.0),     # D2
    ]

    def objective(params):
        pred = hyb2l_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("Running differential evolution (13 params)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=5000, seed=42, tol=1e-12, polish=True, workers=1,
    )

    A, B, C1, W1, PHI1, D1, C2, Wc, PHI2, C3, W2, PHI3, D2 = result.x
    pred = hyb2l_log10(t_fit, *result.x)
    ss_res = np.sum((lp_fit - pred) ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))

    # Compare to HybPPL
    from btc_core import HybPPLModel
    ts_ = np.maximum(t_fit, 0.1)
    pred_hyb = (HybPPLModel._A + HybPPLModel._B * np.log10(ts_)
                + HybPPLModel._C * ts_**(-HybPPLModel._D)
                  * np.cos(HybPPLModel._W * np.log(ts_) + HybPPLModel._PHI)
                + HybPPLModel._C2 * np.cos(HybPPLModel._W2 * ts_ + HybPPLModel._PHI2))
    r2_hyb = 1.0 - np.sum((lp_fit - pred_hyb)**2) / ss_tot

    print(f"\nFitted Hyb2L parameters:")
    print(f"  A    = {A:.6f}")
    print(f"  B    = {B:.6f}")
    print(f"  C1   = {C1:.6f}  (1st log-periodic amplitude)")
    print(f"  W1   = {W1:.6f}  (1st log-time angular freq)")
    print(f"  PHI1 = {PHI1:.6f}")
    print(f"  D1   = {D1:.6f}  (1st damping exponent)")
    print(f"  C2   = {C2:.6f}  (calendar-periodic amplitude)")
    print(f"  Wc   = {Wc:.6f}  rad/yr  (T = {2*np.pi/Wc:.2f} years)")
    print(f"  PHI2 = {PHI2:.6f}")
    print(f"  C3   = {C3:.6f}  (2nd log-periodic amplitude)")
    print(f"  W2   = {W2:.6f}  (2nd log-time angular freq)")
    print(f"  PHI3 = {PHI3:.6f}")
    print(f"  D2   = {D2:.6f}  (2nd damping exponent)")
    print(f"  R\u00b2   = {r2:.6f}  (vs HybPPL R\u00b2={r2_hyb:.6f}, \u0394={r2-r2_hyb:+.6f})")
    print(f"  \u03c3    = {sigma:.6f}")

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
            ("_A", A), ("_B", B), ("_C1", C1),
            ("_W1", W1), ("_PHI1", PHI1), ("_D1", D1),
            ("_C2", C2), ("_Wc", Wc), ("_PHI2", PHI2),
            ("_C3", C3), ("_W2", W2), ("_PHI3", PHI3),
            ("_D2", D2),
        ]
        cls_pos = src.find("class Hyb2LModel")
        if cls_pos == -1:
            print("  WARNING: could not find Hyb2LModel class")
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
                    cls_pos = src.find("class Hyb2LModel")
                    next_class = src.find("\nclass ", cls_pos + 1)
                    cls_end = next_class if next_class != -1 else len(src)
                    section = src[cls_pos:cls_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in Hyb2LModel")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core/ updated.")
    else:
        print("\nRun with --update to write to btc_core/")


if __name__ == "__main__":
    main()
