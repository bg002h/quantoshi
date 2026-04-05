#!/usr/bin/env python3
"""Fit three-frequency LPPL model to Bitcoin price history.

Model: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1)
                    + C2*cos(W2*ln(t)+φ2) + C3*cos(W3*ln(t)+φ3)

12 parameters. W3 initialized near 13.3 (from FFT residual analysis).
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution


def lppl3_log10(t, A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3):
    """Evaluate three-frequency LPPL in log10 space."""
    t_safe = np.maximum(t, 0.1)
    term1 = C1 * t_safe ** (-D) * np.cos(W1 * np.log(t_safe) + PHI1)
    term2 = C2 * np.cos(W2 * np.log(t_safe) + PHI2)
    term3 = C3 * np.cos(W3 * np.log(t_safe) + PHI3)
    return A + B * np.log10(t_safe) + term1 + term2 + term3


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

    # Read current LP2 params as starting point
    sys.path.insert(0, ROOT)
    from btc_core import LPPL2Model
    A0 = LPPL2Model._A
    B0 = LPPL2Model._B
    C0 = LPPL2Model._C
    W0 = LPPL2Model._W
    PHI0 = LPPL2Model._PHI
    D0 = LPPL2Model._D
    C20 = LPPL2Model._C2
    W20 = LPPL2Model._W2
    PHI20 = LPPL2Model._PHI2
    print(f"  LP2 base: W1={W0:.3f}, W2={W20:.3f}")

    # Bounds: base params near LP2 fit, W3 search around 13.3
    margin = 0.2
    bounds = [
        (A0 - margin, A0 + margin),              # A
        (B0 - margin, B0 + margin),               # B
        (max(0.01, C0 - margin), C0 + margin),    # C1
        (2.0, 40.0),                               # W1 (widened)
        (-np.pi, np.pi),                           # PHI1
        (max(0.01, D0 - margin), D0 + margin),    # D
        (max(0.0, C20 - 0.2), C20 + 0.2),         # C2
        (3.0, 40.0),                               # W2 (widened)
        (-np.pi, np.pi),                           # PHI2
        (0.0, 1.0),                                # C3
        (3.0, 40.0),                               # W3 (widened)
        (-np.pi, np.pi),                           # PHI3
    ]

    def objective(params):
        pred = lppl3_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    print("Running differential evolution (12 params)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=4000, seed=42, tol=1e-12, polish=True, workers=1,
    )

    params = result.x
    A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3 = params
    pred = lppl3_log10(t_fit, *params)
    ss_res = np.sum((lp_fit - pred) ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))

    # Compare to LP2
    from btc_core import LPPL2Model as L2
    pred2 = (L2._A + L2._B * np.log10(np.maximum(t_fit, 0.1))
             + L2._C * np.maximum(t_fit, 0.1)**(-L2._D)
               * np.cos(L2._W * np.log(np.maximum(t_fit, 0.1)) + L2._PHI)
             + L2._C2 * np.cos(L2._W2 * np.log(np.maximum(t_fit, 0.1)) + L2._PHI2))
    r2_lp2 = 1.0 - np.sum((lp_fit - pred2) ** 2) / ss_tot

    print(f"\nFitted LPPL\u2083 parameters (12 params):")
    print(f"  A    = {A:.6f}")
    print(f"  B    = {B:.6f}")
    print(f"  C1   = {C1:.6f}")
    print(f"  W1   = {W1:.6f}")
    print(f"  PHI1 = {PHI1:.6f}")
    print(f"  D    = {D:.6f}")
    print(f"  C2   = {C2:.6f}")
    print(f"  W2   = {W2:.6f}  (LP2 had {W20:.3f})")
    print(f"  PHI2 = {PHI2:.6f}")
    print(f"  C3   = {C3:.6f}  (third oscillation amplitude)")
    print(f"  W3   = {W3:.6f}  (initialized at 13.3)")
    print(f"  PHI3 = {PHI3:.6f}")
    print(f"  R\u00b2   = {r2:.6f}  (vs LP2 R\u00b2={r2_lp2:.6f}, \u0394={r2 - r2_lp2:+.6f})")
    print(f"  \u03c3    = {sigma:.6f}")
    print(f"  Ratios: W2/W1={W2/W1:.3f}, W3/W1={W3/W1:.3f}, W3/W2={W3/W2:.3f}")

    if C3 < 0.01:
        print("\n  WARNING: C3 near zero \u2014 3rd oscillation may not be significant.")

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
            ("_A", A), ("_B", B), ("_C", C1),
            ("_W", W1), ("_PHI", PHI1), ("_D", D),
            ("_C2", C2), ("_W2", W2), ("_PHI2", PHI2),
            ("_C3", C3), ("_W3", W3), ("_PHI3", PHI3),
        ]
        lp3_pos = src.find("class LPPL3Model")
        if lp3_pos == -1:
            print("  WARNING: could not find LPPL3Model class")
        else:
            next_class = src.find("\nclass ", lp3_pos + 1)
            lp3_end = next_class if next_class != -1 else len(src)
            section = src[lp3_pos:lp3_end]
            for name, val in replacements:
                pattern = rf"(    {name}\s*=\s*)[^\n]+"
                new_val = f"{val:>11.6f}" if val >= 0 else f"{val:.6f}"
                old_match = re.search(pattern, section)
                if old_match:
                    old_line = old_match.group(0)
                    new_line = re.sub(pattern, rf"\g<1>{new_val}", old_line)
                    new_section = section.replace(old_line, new_line, 1)
                    src = src[:lp3_pos] + new_section + src[lp3_end:]
                    lp3_pos = src.find("class LPPL3Model")
                    next_class = src.find("\nclass ", lp3_pos + 1)
                    lp3_end = next_class if next_class != -1 else len(src)
                    section = src[lp3_pos:lp3_end]
                    print(f"  {name} = {new_val.strip()}")
                else:
                    print(f"  WARNING: could not find {name} in LPPL3Model")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core.py updated.")
    else:
        print("\nRun with --update to write params to btc_core.py")


if __name__ == "__main__":
    main()
