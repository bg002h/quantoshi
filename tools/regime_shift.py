#!/usr/bin/env python3
"""Rolling-Window LPPL Regime Shift Detection.

Fits a 1-frequency LPPL model on 5-year rolling windows of Bitcoin price
history (stepped monthly) and tracks parameter evolution over time.

Outputs a 4-panel stacked figure to regime_shift.svg / .jpg.

Usage:
    btc_venv/bin/python3 tools/regime_shift.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution


GENESIS = pd.Timestamp("2009-07-25")
WINDOW_WIDTH_YRS = 5.0
STEP_YRS = 1.0 / 12.0  # monthly

REGIME_EVENTS = [
    ("2013-11-30", "2013 mania peak"),
    ("2017-12-17", "2017 peak / CME futures"),
    ("2020-03-12", "Covid crash"),
    ("2021-11-10", "2021 peak"),
    ("2022-11-11", "FTX collapse"),
    ("2024-01-10", "ETF approval"),
]

BOUNDS = [
    (-3.0, 1.0),     # A
    (3.0, 7.0),      # B
    (0.01, 3.0),     # C
    (2.0, 15.0),     # W
    (-np.pi, np.pi), # PHI
    (0.01, 2.0),     # D
]


def lppl_log10(t, A, B, C, W, PHI, D):
    t_safe = np.maximum(t, 0.1)
    envelope = C * t_safe ** (-D)
    return A + B * np.log10(t_safe) + envelope * np.cos(W * np.log(t_safe) + PHI)


def fit_window(t_win, lp_win):
    """Fit LPPL to a single window. Returns dict with W, D, sigma, r2 (or NaN)."""
    def objective(params):
        A, B, C, W, PHI, D = params
        pred = lppl_log10(t_win, A, B, C, W, PHI, D)
        return float(np.sum((lp_win - pred) ** 2))

    try:
        result = differential_evolution(
            objective, BOUNDS,
            maxiter=2000, seed=42, tol=1e-10,
            polish=True, workers=1,
        )
        A, B, C, W, PHI, D = result.x
        pred = lppl_log10(t_win, A, B, C, W, PHI, D)
        resid = lp_win - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((lp_win - np.mean(lp_win)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sigma = float(np.std(resid))
        return {"A": A, "B": B, "C": C, "W": W, "PHI": PHI, "D": D,
                "sigma": sigma, "r2": r2}
    except Exception as e:
        print(f"  fit failure: {e}")
        return {"A": np.nan, "B": np.nan, "C": np.nan, "W": np.nan,
                "PHI": np.nan, "D": np.nan, "sigma": np.nan, "r2": np.nan}


def main():
    print("Loading Bitcoin prices...")
    pd_ = load_prices("BitcoinPricesDaily.csv")
    df = pd_.df_full[["date", "years", "log_price"]].copy()
    df = df[df["years"] >= 1.0].reset_index(drop=True)
    print(f"  {len(df)} daily rows (years >= 1.0)")

    t_all = df["years"].values
    lp_all = df["log_price"].values
    dates_all = df["date"].values

    # Build windows: window ends at t_end, starts at t_end - 5.0
    # t_end sweeps from (data_min + width) to data_max, stepped by 1 month
    t_min = float(t_all.min())
    t_max = float(t_all.max())
    first_end = t_min + WINDOW_WIDTH_YRS
    # Align to monthly step
    ends = np.arange(first_end, t_max + 1e-9, STEP_YRS)
    print(f"  {len(ends)} rolling windows (width={WINDOW_WIDTH_YRS}yr, step={STEP_YRS*12:.0f}mo)")
    print(f"  First window end t={ends[0]:.3f}, last t={ends[-1]:.3f}")

    rows = []
    for i, t_end in enumerate(ends):
        t_start = t_end - WINDOW_WIDTH_YRS
        mask = (t_all >= t_start) & (t_all <= t_end)
        t_win = t_all[mask]
        lp_win = lp_all[mask]
        if len(t_win) < 100:
            print(f"[{i+1:3d}/{len(ends)}] t_end={t_end:.3f} SKIP (n={len(t_win)})")
            rows.append({"t_end": t_end, "end_date": GENESIS + pd.Timedelta(days=t_end*365.25),
                         "n": len(t_win), "W": np.nan, "D": np.nan, "sigma": np.nan, "r2": np.nan})
            continue
        fit = fit_window(t_win, lp_win)
        end_date = GENESIS + pd.Timedelta(days=t_end * 365.25)
        print(f"[{i+1:3d}/{len(ends)}] {end_date.date()} n={len(t_win)} "
              f"W={fit['W']:.3f} D={fit['D']:.3f} sigma={fit['sigma']:.4f} r2={fit['r2']:.4f}")
        rows.append({
            "t_end": t_end, "end_date": end_date, "n": len(t_win),
            "W": fit["W"], "D": fit["D"],
            "sigma": fit["sigma"], "r2": fit["r2"],
        })

    results = pd.DataFrame(rows)
    # Save CSV alongside figure
    csv_path = os.path.join(ROOT, "regime_shift.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # ── Build 4-panel figure ────────────────────────────────────────────────
    FACE = "#1a1a2e"
    AXES_FACE = "#16213e"
    TEXT = "#cccccc"
    SPINE = "#555555"
    GRID = "#333344"

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True,
                             facecolor=FACE, dpi=200)
    fig.patch.set_facecolor(FACE)

    x = results["end_date"]

    # Panel 1: W
    ax = axes[0]
    ax.plot(x, results["W"], color="#FF6D00", lw=1.5, label="W (rolling)")
    ax.axhline(7.56, color="#888888", ls="--", lw=0.8, label="Full-history W=7.56")
    ax.set_title("LPPL frequency W (log-time)", color=TEXT, fontsize=12)
    ax.set_ylabel("W", color=TEXT)
    ax.legend(loc="upper left", facecolor=AXES_FACE, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

    # Panel 2: D
    ax = axes[1]
    ax.plot(x, results["D"], color="#FF9F40", lw=1.5, label="D (rolling)")
    ax.axhline(0.0, color="#888888", ls=":", lw=0.8, label="undamped (D=0)")
    ax.axhline(0.61, color="#888888", ls="--", lw=0.8, label="full-history LPPL D=0.61")
    ax.set_title("Damping exponent D", color=TEXT, fontsize=12)
    ax.set_ylabel("D", color=TEXT)
    ax.legend(loc="upper left", facecolor=AXES_FACE, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

    # Panel 3: sigma
    ax = axes[2]
    ax.plot(x, results["sigma"], color="#DAA520", lw=1.5)
    ax.set_title("Residual σ per window", color=TEXT, fontsize=12)
    ax.set_ylabel("σ (log₁₀ price)", color=TEXT)

    # Panel 4: R²
    ax = axes[3]
    ax.plot(x, results["r2"], color="#E8C860", lw=1.5)
    ax.set_title("R² per window (fit quality)", color=TEXT, fontsize=12)
    ax.set_ylabel("R²", color=TEXT)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Window end date", color=TEXT)

    # Apply styling + regime event annotations to all panels
    for ax in axes:
        ax.set_facecolor(AXES_FACE)
        for spine in ax.spines.values():
            spine.set_color(SPINE)
        ax.tick_params(colors=TEXT, which="both")
        ax.grid(True, color=GRID, lw=0.4, alpha=0.6)
        for date_str, label in REGIME_EVENTS:
            ev = pd.Timestamp(date_str)
            ax.axvline(ev, color="#888888", ls="--", lw=0.6, alpha=0.7)
            # tiny rotated label at top of each axis
            ylim = ax.get_ylim()
            ax.text(ev, ylim[1], " " + label, color="#aaaaaa", fontsize=6,
                    rotation=90, va="top", ha="left", alpha=0.8)

    axes[-1].xaxis.set_major_formatter(DateFormatter("%Y"))

    fig.suptitle("Rolling-Window LPPL Regime Shift Detection (5-year windows, monthly steps)",
                 color=TEXT, fontsize=14, y=0.995)
    fig.text(0.5, 0.005,
             "Window width=5yr, step=1mo, LPPL 1-frequency model. Vertical dashes mark known regime events.",
             ha="center", color=TEXT, fontsize=9)
    fig.tight_layout(rect=[0, 0.015, 1, 0.985])

    svg_path = os.path.join(ROOT, "regime_shift.svg")
    jpg_path = os.path.join(ROOT, "regime_shift.jpg")
    fig.savefig(svg_path, facecolor=FACE, bbox_inches="tight")
    fig.savefig(jpg_path, facecolor=FACE, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {svg_path}")
    print(f"Saved {jpg_path}")

    # ── Summary stats ──────────────────────────────────────────────────────
    r = results.dropna(subset=["W", "D"])
    W_arr = r["W"].values
    D_arr = r["D"].values
    dW = np.abs(np.diff(W_arr))
    dD = np.abs(np.diff(D_arr))
    print("\n─── Summary ───")
    print(f"Windows fitted: {len(r)} / {len(results)}")
    print(f"W:  mean={W_arr.mean():.3f}  std={W_arr.std():.3f}  min={W_arr.min():.3f}  max={W_arr.max():.3f}")
    print(f"D:  mean={D_arr.mean():.3f}  std={D_arr.std():.3f}  min={D_arr.min():.3f}  max={D_arr.max():.3f}")
    print(f"|ΔW| > 1.0 adjacent-step breaks:  {int((dW > 1.0).sum())}")
    print(f"|ΔD| > 0.2 adjacent-step breaks:  {int((dD > 0.2).sum())}")


if __name__ == "__main__":
    main()
