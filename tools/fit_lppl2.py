#!/usr/bin/env python3
"""Fit two-frequency LPPL model to Bitcoin price history.

Model: log10(price) = A + B*log10(t) + t^(-D) * [C1*cos(W1*ln(t)+φ1) + C2*cos(W2*ln(t)+φ2)]

9 parameters: A, B, C1, W1, PHI1, D (shared damping), C2, W2, PHI2.
W2 is independent of W1 (not constrained to 2×W1).

Usage:
    btc_venv/bin/python3 tools/fit_lppl2.py              # fit and print
    btc_venv/bin/python3 tools/fit_lppl2.py --update      # fit and update the btc_core/ package
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution


def lppl2_log10(t, A, B, C1, W1, PHI1, D, C2, W2, PHI2):
    """Evaluate two-frequency LPPL model in log10 space.

    Primary oscillation is damped; secondary oscillation is undamped.
    """
    t_safe = np.maximum(t, 0.1)
    term1 = C1 * t_safe ** (-D) * np.cos(W1 * np.log(t_safe) + PHI1)
    term2 = C2 * np.cos(W2 * np.log(t_safe) + PHI2)
    return A + B * np.log10(t_safe) + term1 + term2


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

    # Read current LPPL params as starting point
    sys.path.insert(0, ROOT)
    from btc_core import LPPLModel
    A0, B0, C0 = LPPLModel._A, LPPLModel._B, LPPLModel._C
    W0, PHI0, D0 = LPPLModel._W, LPPLModel._PHI, LPPLModel._D
    print(f"  LPPL base: A={A0:.4f} B={B0:.4f} C={C0:.4f} W={W0:.4f} φ={PHI0:.4f} D={D0:.4f}")

    # Bounds: base params near LPPL fit, W2 free to find any frequency
    margin = 0.3
    bounds = [
        (A0 - margin, A0 + margin),              # A
        (B0 - margin, B0 + margin),               # B
        (max(0.01, C0 - margin), C0 + margin),    # C1
        (2.0, 40.0),                               # W1 (widened)
        (-np.pi, np.pi),                           # PHI1
        (max(0.01, D0 - margin), D0 + margin),    # D
        (0.0, 2.0),                                # C2
        (1.0, 40.0),                               # W2 (widened, free frequency)
        (-np.pi, np.pi),                           # PHI2
    ]

    def objective(params):
        pred = lppl2_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("Running differential evolution (9 params, may take a few minutes)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=3000,
        seed=42,
        tol=1e-12,
        polish=True,
        workers=1,
    )

    A, B, C1, W1, PHI1, D, C2, W2, PHI2 = result.x
    pred = lppl2_log10(t_fit, *result.x)
    ss_res = np.sum((lp_fit - pred) ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))

    # Compare to 1st-order LPPL R²
    from btc_core import LPPLModel as LM
    pred1 = LM._A + LM._B * np.log10(np.maximum(t_fit, 0.1)) + \
            LM._C * np.maximum(t_fit, 0.1)**(-LM._D) * np.cos(LM._W * np.log(np.maximum(t_fit, 0.1)) + LM._PHI)
    r2_lppl1 = 1.0 - np.sum((lp_fit - pred1) ** 2) / ss_tot

    # Frequency ratio
    freq_ratio = W2 / W1 if W1 > 0 else 0

    print(f"\nFitted LPPL\u2082 parameters (9-param, free W\u2082):")
    print(f"  A    = {A:.6f}")
    print(f"  B    = {B:.6f}")
    print(f"  C1   = {C1:.6f}")
    print(f"  W1   = {W1:.6f}")
    print(f"  PHI1 = {PHI1:.6f}")
    print(f"  D    = {D:.6f}")
    print(f"  C2   = {C2:.6f}  (second oscillation amplitude)")
    print(f"  W2   = {W2:.6f}  (second frequency, ratio W2/W1 = {freq_ratio:.3f})")
    print(f"  PHI2 = {PHI2:.6f}")
    print(f"  R\u00b2   = {r2:.6f}  (vs LPPL\u2081 R\u00b2={r2_lppl1:.6f}, \u0394={r2 - r2_lppl1:.6f})")
    print(f"  \u03c3    = {sigma:.6f}")

    if C2 < 0.01:
        print("\n  WARNING: C2 is near zero \u2014 second oscillation may not be significant.")

    if update:
        print("\nUpdating btc_core/ ...")
        core_path = os.path.join(ROOT, "btc_core", "_lppl.py")
        import shutil
        shutil.copy2(core_path, core_path + ".bak")
        print(f"  Backup saved to btc_core.py.bak")

        with open(core_path) as f:
            src = f.read()

        import re
        replacements = [
            ("_A", A), ("_B", B), ("_C", C1),
            ("_W", W1), ("_PHI", PHI1), ("_D", D),
            ("_C2", C2), ("_W2", W2), ("_PHI2", PHI2),
        ]
        lp2_pos = src.find("class LPPL2Model")
        if lp2_pos == -1:
            print("  WARNING: could not find LPPL2Model class")
        else:
            next_class = src.find("\nclass ", lp2_pos + 1)
            lp2_end = next_class if next_class != -1 else len(src)
            section = src[lp2_pos:lp2_end]
            for name, val in replacements:
                pattern = rf"(    {name}\s*=\s*)[^\n]+"
                new_val = f"{val:>11.6f}" if val >= 0 else f"{val:.6f}"
                old_match = re.search(pattern, section)
                if old_match:
                    old_line = old_match.group(0)
                    new_line = re.sub(pattern, rf"\g<1>{new_val}", old_line)
                    new_section = section.replace(old_line, new_line, 1)
                    src = src[:lp2_pos] + new_section + src[lp2_end:]
                    lp2_pos = src.find("class LPPL2Model")
                    next_class = src.find("\nclass ", lp2_pos + 1)
                    lp2_end = next_class if next_class != -1 else len(src)
                    section = src[lp2_pos:lp2_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in LPPL2Model")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core/ updated.")
    else:
        print("\nRun with --update to write params to btc_core.py")


if __name__ == "__main__":
    main()
