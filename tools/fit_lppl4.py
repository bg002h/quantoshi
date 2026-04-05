#!/usr/bin/env python3
"""Fit LPPL4 (four-frequency LPPL) — 15 parameters.

Model: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1)
                    + C2*cos(W2*ln(t)+φ2) + C3*cos(W3*ln(t)+φ3) + C4*cos(W4*ln(t)+φ4)

--no-13: exclude the ω≈13 band (forces secondaries to avoid the W2-W1 intermod region)
--update: write fitted params to btc_core.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution


def lppl4_log10(t, A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3, C4, W4, PHI4):
    t_safe = np.maximum(t, 0.1)
    return (A + B * np.log10(t_safe)
            + C1 * t_safe**(-D) * np.cos(W1 * np.log(t_safe) + PHI1)
            + C2 * np.cos(W2 * np.log(t_safe) + PHI2)
            + C3 * np.cos(W3 * np.log(t_safe) + PHI3)
            + C4 * np.cos(W4 * np.log(t_safe) + PHI4))


def main():
    update = "--update" in sys.argv
    no_13 = "--no-13" in sys.argv

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    log_p = prices.df_full["log_price"].values
    mask = t >= 1.0
    t_fit = t[mask]
    lp_fit = log_p[mask]
    print(f"  {len(t_fit)} data points (t >= 1.0)")

    # Bounds
    bounds = [
        (-1.5, -0.7),    # A
        (4.5, 5.5),      # B
        (0.1, 1.5),      # C1
        (2.0, 40.0),     # W1 (widened)
        (-np.pi, np.pi), # PHI1
        (0.05, 1.2),     # D
        (0.0, 0.5),      # C2
        (3.0, 40.0),     # W2 (widened)
        (-np.pi, np.pi), # PHI2
        (0.0, 0.5),      # C3
        (3.0, 40.0),     # W3 (widened)
        (-np.pi, np.pi), # PHI3
        (0.0, 0.5),      # C4
        (3.0, 40.0),     # W4 (widened)
        (-np.pi, np.pi), # PHI4
    ]

    # Exclude W=13 band (11.5-14.5) — apply penalty to any W in that range
    W_EXCLUDE = (11.5, 14.5) if no_13 else None
    label = "LPPL4 no-13" if no_13 else "LPPL4"

    def in_exclude(W):
        if W_EXCLUDE is None:
            return False
        return W_EXCLUDE[0] <= W <= W_EXCLUDE[1]

    def objective(params):
        pred = lppl4_log10(t_fit, *params)
        sse = np.sum((lp_fit - pred)**2)
        if W_EXCLUDE is not None:
            # Soft penalty: if any W is in exclude band, penalize heavily
            for W in (params[7], params[10], params[13]):
                if in_exclude(W):
                    return sse * 100  # effective 100× loss
        return sse

    print(f"Running DE for {label} (15 params)...")
    result = differential_evolution(
        objective, bounds,
        maxiter=6000, seed=42, tol=1e-12, polish=True, workers=1,
    )

    params = result.x
    A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3, C4, W4, PHI4 = params
    pred = lppl4_log10(t_fit, *params)
    ss_res = np.sum((lp_fit - pred) ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(lp_fit - pred))

    print(f"\nFitted {label} parameters:")
    print(f"  A    = {A:.6f}")
    print(f"  B    = {B:.6f}")
    print(f"  C1   = {C1:.6f}")
    print(f"  W1   = {W1:.6f}")
    print(f"  PHI1 = {PHI1:.6f}")
    print(f"  D    = {D:.6f}")
    print(f"  C2   = {C2:.6f}")
    print(f"  W2   = {W2:.6f}")
    print(f"  PHI2 = {PHI2:.6f}")
    print(f"  C3   = {C3:.6f}")
    print(f"  W3   = {W3:.6f}")
    print(f"  PHI3 = {PHI3:.6f}")
    print(f"  C4   = {C4:.6f}")
    print(f"  W4   = {W4:.6f}")
    print(f"  PHI4 = {PHI4:.6f}")
    print(f"  R²   = {r2:.6f}")
    print(f"  σ    = {sigma:.6f}")
    secondaries = sorted([W2, W3, W4])
    print(f"\n  Secondaries (sorted): {secondaries[0]:.3f}, {secondaries[1]:.3f}, {secondaries[2]:.3f}")
    print(f"  Ratios vs W1: {secondaries[0]/W1:.3f}, {secondaries[1]/W1:.3f}, {secondaries[2]/W1:.3f}")

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
            ("_C4", C4), ("_W4", W4), ("_PHI4", PHI4),
        ]
        lp4_pos = src.find("class LPPL4Model")
        if lp4_pos == -1:
            print("  WARNING: could not find LPPL4Model class")
            return
        next_class = src.find("\nclass ", lp4_pos + 1)
        lp4_end = next_class if next_class != -1 else len(src)
        section = src[lp4_pos:lp4_end]
        for name, val in replacements:
            pattern = rf"(    {name}\s*=\s*)[^\n]+"
            new_val = f"{val:>11.6f}" if val >= 0 else f"{val:.6f}"
            old_match = re.search(pattern, section)
            if old_match:
                old_line = old_match.group(0)
                new_line = re.sub(pattern, rf"\g<1>{new_val}", old_line)
                new_section = section.replace(old_line, new_line, 1)
                src = src[:lp4_pos] + new_section + src[lp4_end:]
                lp4_pos = src.find("class LPPL4Model")
                next_class = src.find("\nclass ", lp4_pos + 1)
                lp4_end = next_class if next_class != -1 else len(src)
                section = src[lp4_pos:lp4_end]
                print(f"  {name} = {new_val.strip()}")
            else:
                print(f"  WARNING: could not find {name} in LPPL4Model")

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core.py updated.")
    else:
        print("\nRun with --update to write params to btc_core.py")


if __name__ == "__main__":
    main()
