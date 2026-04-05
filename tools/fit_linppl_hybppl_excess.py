#!/usr/bin/env python3
"""Fit LinPPL and HybPPL oscillators to EXCESS = log_price - BM support.

Standard LinPPL/HybPPL fits the oscillator jointly with a power-law
trend (A + B*log10(t)). This version fixes the trend to the known
BM support line (A_sup + B_sup*log10(t)) and fits only the
oscillation parameters. Cleaner decomposition: trend is set, cycles
are free.

Models:
  LinPPL_excess:  excess = a0 + C*t^(-D)*cos(ω_cal*t + φ)       [5 params]
  HybPPL_excess:  excess = a0 + C1*t^(-D)*cos(ω_log*ln(t) + φ1)
                              + C2*cos(ω_cal*t + φ2)            [8 params]

The a0 constant captures the DC offset (log-excess is consistently
above zero since the BM support is the floor, not the mean).

Emits SVGs (fit overlay + residuals) and CSVs; served at /F.

MANUAL REGENERATION ONLY.

Usage:
    btc_venv/bin/python3 tools/fit_linppl_hybppl_excess.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices
from model_toolkit.support import fit_support


REGIME_EVENTS = [
    ("2013-11-30", "2013 mania"),
    ("2017-12-17", "2017 peak"),
    ("2020-03-12", "Covid"),
    ("2021-11-10", "2021 peak"),
    ("2022-11-11", "FTX"),
    ("2024-01-10", "ETF"),
]


def linppl_excess(t, a0, C, W_cal, PHI, D):
    t_safe = np.maximum(t, 0.1)
    return a0 + C * t_safe ** (-D) * np.cos(W_cal * t_safe + PHI)


def hybppl_excess(t, a0, C1, W_log, PHI1, D, C2, W_cal, PHI2):
    t_safe = np.maximum(t, 0.1)
    damped = C1 * t_safe ** (-D) * np.cos(W_log * np.log(t_safe) + PHI1)
    undamped = C2 * np.cos(W_cal * t_safe + PHI2)
    return a0 + damped + undamped


def fit_model(name, fn, bounds, t, excess):
    print(f"\nFitting {name} ({len(bounds)} params) via differential_evolution...")

    def objective(params):
        pred = fn(t, *params)
        return float(np.sum((excess - pred) ** 2))

    result = differential_evolution(
        objective, bounds,
        maxiter=2000, seed=42, tol=1e-12, polish=True, workers=1,
    )
    pred = fn(t, *result.x)
    resid = excess - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((excess - np.mean(excess)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(resid))
    return result.x, pred, resid, r2, sigma


def plot_model(title, dates, excess, fit, resid, out_svg,
               coeffs_text, r2, sigma):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#1a1a2e")

    TITLE_COLOR = "#00d4ff"
    LABEL_COLOR = "#cccccc"
    TICK_COLOR = "#aaaaaa"
    DATA_COLOR = "#888888"
    FIT_COLOR = "#FF9F40"
    EVENT_COLOR = "#888888"
    RESID_COLOR = "#4da6ff"

    fig.suptitle(f"{title}  \u2014  R\u00b2={r2:.4f}  \u03c3={sigma:.4f}",
                 color=TITLE_COLOR, fontsize=13, fontweight="bold")

    ax1.set_facecolor("#16213e")
    for spine in ax1.spines.values():
        spine.set_color("#555555")
    ax1.tick_params(colors=TICK_COLOR, labelsize=9)
    ax1.plot(dates, excess, color=DATA_COLOR, linewidth=0.6, alpha=0.7,
             label="log-excess (price \u2212 BM floor)")
    ax1.plot(dates, fit, color=FIT_COLOR, linewidth=1.4, label=title)
    ax1.axhline(0, color="#666666", linewidth=0.5, linestyle=":")
    for date_str, _ in REGIME_EVENTS:
        ax1.axvline(pd.Timestamp(date_str), color=EVENT_COLOR,
                    linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.grid(True, alpha=0.15, color="#555555")
    ax1.set_ylabel("log-excess", color=LABEL_COLOR, fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, facecolor="#101a2e",
               edgecolor="#555555", labelcolor=LABEL_COLOR)
    ax1.text(0.98, 0.02, coeffs_text, transform=ax1.transAxes,
             fontsize=8, color=LABEL_COLOR, ha="right", va="bottom",
             family="monospace",
             bbox=dict(facecolor="#0e1624", edgecolor="#555555",
                       boxstyle="round,pad=0.4"))

    ax2.set_facecolor("#16213e")
    for spine in ax2.spines.values():
        spine.set_color("#555555")
    ax2.tick_params(colors=TICK_COLOR, labelsize=9)
    ax2.plot(dates, resid, color=RESID_COLOR, linewidth=0.6)
    ax2.axhline(0, color="#666666", linewidth=0.5, linestyle=":")
    for date_str, _ in REGIME_EVENTS:
        ax2.axvline(pd.Timestamp(date_str), color=EVENT_COLOR,
                    linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.grid(True, alpha=0.15, color="#555555")
    ax2.set_ylabel("residual", color=LABEL_COLOR, fontsize=10)
    ax2.set_xlabel("Date", color=LABEL_COLOR, fontsize=10)

    fig.savefig(out_svg, format="svg", facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("LinPPL/HybPPL fits on BM-excess (detrended)")
    print("=" * 60)

    print("Loading Bitcoin prices...")
    pd_ = load_prices("BitcoinPricesDaily.csv")
    df = pd_.df_full[["date", "years", "log_price"]].copy()
    df = df[df["years"] >= 1.0].reset_index(drop=True)
    t = df["years"].values
    lp = df["log_price"].values
    dates = pd.to_datetime(df["date"].values)
    print(f"  {len(df)} daily rows (t \u2265 1 yr)")

    sup = fit_support(pd_)
    log_t = np.log10(np.maximum(t, 0.1))
    log_support = sup.intercept + sup.slope * log_t
    excess = lp - log_support
    print(f"  BM support: A_sup={sup.intercept:.4f}  B_sup={sup.slope:.4f}")

    # ── LinPPL on excess (5 params: a0, C, W_cal, PHI, D) ────────────────
    linppl_bounds = [
        (-1.0, 2.0),      # a0 (DC offset)
        (0.01, 3.0),      # C
        (0.5, 10.0),      # W_cal (rad/yr)
        (-np.pi, np.pi),  # PHI
        (0.01, 2.0),      # D
    ]
    params, fit_, resid, r2, sigma = fit_model(
        "LinPPL_excess", linppl_excess, linppl_bounds, t, excess)
    a0, C, W_cal, PHI, D = params
    T_yr = 2.0 * np.pi / W_cal
    print(f"  a0={a0:.4f}  C={C:.4f}  W_cal={W_cal:.4f} rad/yr "
          f"(T={T_yr:.2f}yr)  PHI={PHI:.4f}  D={D:.4f}")
    print(f"  R\u00b2(on excess) = {r2:.5f}   \u03c3 = {sigma:.5f}")
    coeffs_text = (
        f"a0   = {a0:+.4f}\n"
        f"C    = {C:+.4f}\n"
        f"W_cal= {W_cal:+.4f} rad/yr  (T={T_yr:.2f}yr)\n"
        f"PHI  = {PHI:+.4f}\n"
        f"D    = {D:+.4f}"
    )
    plot_model(
        "LinPPL on excess (5 params)", dates, excess, fit_, resid,
        "fit_linppl_excess.svg", coeffs_text, r2, sigma,
    )
    pd.DataFrame({
        "date": dates, "years": t, "excess": excess,
        "fit": fit_, "residual": resid,
    }).to_csv("fit_linppl_excess.csv", index=False, float_format="%.6f")
    print("  Saved fit_linppl_excess.svg + fit_linppl_excess.csv")

    # ── HybPPL on excess (8 params) ──────────────────────────────────────
    hybppl_bounds = [
        (-1.0, 2.0),      # a0
        (0.01, 3.0),      # C1
        (2.0, 40.0),      # W_log (log-time angular freq)
        (-np.pi, np.pi),  # PHI1
        (0.01, 2.0),      # D
        (0.0, 2.0),       # C2
        (0.5, 10.0),      # W_cal (rad/yr)
        (-np.pi, np.pi),  # PHI2
    ]
    params, fit_, resid, r2, sigma = fit_model(
        "HybPPL_excess", hybppl_excess, hybppl_bounds, t, excess)
    a0, C1, W_log, PHI1, D, C2, W_cal, PHI2 = params
    T_yr = 2.0 * np.pi / W_cal
    print(f"  a0={a0:.4f}")
    print(f"  C1={C1:.4f}  W_log={W_log:.4f}  PHI1={PHI1:.4f}  D={D:.4f}")
    print(f"  C2={C2:.4f}  W_cal={W_cal:.4f} rad/yr (T={T_yr:.2f}yr)  "
          f"PHI2={PHI2:.4f}")
    print(f"  R\u00b2(on excess) = {r2:.5f}   \u03c3 = {sigma:.5f}")
    coeffs_text = (
        f"a0   = {a0:+.4f}\n"
        f"C1   = {C1:+.4f}\n"
        f"W_log= {W_log:+.4f}  (log-time)\n"
        f"PHI1 = {PHI1:+.4f}\n"
        f"D    = {D:+.4f}\n"
        f"C2   = {C2:+.4f}\n"
        f"W_cal= {W_cal:+.4f} rad/yr (T={T_yr:.2f}yr)\n"
        f"PHI2 = {PHI2:+.4f}"
    )
    plot_model(
        "HybPPL on excess (8 params)", dates, excess, fit_, resid,
        "fit_hybppl_excess.svg", coeffs_text, r2, sigma,
    )
    pd.DataFrame({
        "date": dates, "years": t, "excess": excess,
        "fit": fit_, "residual": resid,
    }).to_csv("fit_hybppl_excess.csv", index=False, float_format="%.6f")
    print("  Saved fit_hybppl_excess.svg + fit_hybppl_excess.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
