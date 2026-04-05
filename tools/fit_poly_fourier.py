#!/usr/bin/env python3
"""Static full-history fits of Bitcoin log-price to:
  1. Degree-8 polynomial (9 coefficients)
  2. 4-harmonic Fourier series with fixed fundamental period = full span
     (9 params: a0 + 4*(Ak,Bk))

Emits two SVGs (fit overlay + residuals per model) and one CSV per
model with coefficients + full-length fit curves. Served at /F.

MANUAL REGENERATION ONLY.

Usage:
    btc_venv/bin/python3 tools/fit_poly_fourier.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices
from model_toolkit.support import fit_support


GENESIS = pd.Timestamp("2009-07-25")
POLY_DEGREE = 8             # 9 coefficients (a0..a8)
FOURIER_HARMONICS = 4       # 1 + 2*4 = 9 params (a0, A1..4, B1..4)


def fit_polynomial(t, lp, degree=POLY_DEGREE):
    """Fit degree-D polynomial to (t, log_price) — closed-form OLS.

    Returns (coeffs, fit_values, residuals).
    coeffs[0] is the highest-degree coefficient (numpy convention).
    """
    coeffs = np.polyfit(t, lp, degree)
    fit = np.polyval(coeffs, t)
    resid = lp - fit
    return coeffs, fit, resid


def fit_fourier(t, lp, n_harmonics=FOURIER_HARMONICS):
    """Fit Fourier series with fundamental period T = t.max() - t.min().

    Model: log_price = a0 + sum_{k=1..N} [Ak*cos(2π·k·u) + Bk*sin(2π·k·u)]
    where u = (t - t_min) / T in [0, 1].

    Linear least squares (closed-form).
    Returns (a0, As, Bs, fit_values, residuals, T_years).
    """
    t_min = float(t.min())
    T = float(t.max() - t_min)
    u = (t - t_min) / T  # [0, 1]
    # Design matrix: [1, cos(2π·1·u), sin(2π·1·u), cos(2π·2·u), sin(2π·2·u), ...]
    cols = [np.ones_like(u)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * u))
        cols.append(np.sin(2 * np.pi * k * u))
    X = np.column_stack(cols)
    # OLS
    beta, *_ = np.linalg.lstsq(X, lp, rcond=None)
    a0 = float(beta[0])
    As = [float(beta[1 + 2 * i]) for i in range(n_harmonics)]
    Bs = [float(beta[2 + 2 * i]) for i in range(n_harmonics)]
    fit = X @ beta
    resid = lp - fit
    return a0, As, Bs, fit, resid, T, t_min


# ── Plotting ──────────────────────────────────────────────────────────────

REGIME_EVENTS = [
    ("2013-11-30", "2013 mania"),
    ("2017-12-17", "2017 peak"),
    ("2020-03-12", "Covid"),
    ("2021-11-10", "2021 peak"),
    ("2022-11-11", "FTX"),
    ("2024-01-10", "ETF"),
]


def plot_model(title, dates, lp, fit, resid, out_svg,
               coeffs_text, r2, sigma):
    """2-panel figure: fit overlay (top) + residuals (bottom)."""
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

    # Top: fit overlay
    ax1.set_facecolor("#16213e")
    for spine in ax1.spines.values():
        spine.set_color("#555555")
    ax1.tick_params(colors=TICK_COLOR, labelsize=9)
    ax1.plot(dates, lp, color=DATA_COLOR, linewidth=0.6, alpha=0.7,
             label="log\u2081\u2080(price)")
    ax1.plot(dates, fit, color=FIT_COLOR, linewidth=1.4, label=title)
    for date_str, _ in REGIME_EVENTS:
        ax1.axvline(pd.Timestamp(date_str), color=EVENT_COLOR,
                    linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.grid(True, alpha=0.15, color="#555555")
    ax1.set_ylabel("log\u2081\u2080(price)", color=LABEL_COLOR, fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, facecolor="#101a2e",
               edgecolor="#555555", labelcolor=LABEL_COLOR)

    # Coefficients text box (small, bottom-right of top panel)
    ax1.text(0.98, 0.02, coeffs_text, transform=ax1.transAxes,
             fontsize=8, color=LABEL_COLOR, ha="right", va="bottom",
             family="monospace",
             bbox=dict(facecolor="#0e1624", edgecolor="#555555",
                       boxstyle="round,pad=0.4"))

    # Bottom: residuals
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
    print("Polynomial + Fourier full-history fits")
    print("=" * 60)

    print("Loading Bitcoin prices...")
    pd_ = load_prices("BitcoinPricesDaily.csv")
    df = pd_.df_full[["date", "years", "log_price"]].copy()
    df = df[df["years"] >= 1.0].reset_index(drop=True)
    t = df["years"].values
    lp = df["log_price"].values
    dates = pd.to_datetime(df["date"].values)
    print(f"  {len(df)} daily rows (t \u2265 1 yr)")

    # BM support line (floor): A_sup + B_sup * log10(t)
    sup = fit_support(pd_)
    print(f"  BM support: A_sup={sup.intercept:.4f}  B_sup={sup.slope:.4f}")
    log_t = np.log10(np.maximum(t, 0.1))
    log_support = sup.intercept + sup.slope * log_t
    excess = lp - log_support  # residual above floor (log-space)
    ss_tot = float(np.sum((lp - np.mean(lp)) ** 2))
    # ss_tot for excess — used for R² on the residual fit, measures how
    # well the fit captures the excess variance itself (not the raw lp).
    ss_tot_exc = float(np.sum((excess - np.mean(excess)) ** 2))

    # ── Polynomial ───────────────────────────────────────────────────────
    print(f"\nFitting degree-{POLY_DEGREE} polynomial ({POLY_DEGREE + 1} params)...")
    pcoeffs, pfit, presid = fit_polynomial(t, lp, POLY_DEGREE)
    p_r2 = 1.0 - float(np.sum(presid ** 2)) / ss_tot
    p_sigma = float(np.std(presid))
    # coeffs[0] is a8 (highest degree); coeffs[-1] is a0 (constant)
    poly_rev = pcoeffs[::-1]  # now index k is coefficient of t^k
    poly_text = "\n".join(
        f"a{i} = {poly_rev[i]:+.4e}" for i in range(len(poly_rev)))
    print(f"  R\u00b2 = {p_r2:.5f}   \u03c3 = {p_sigma:.5f}")
    plot_model(
        f"Polynomial (deg {POLY_DEGREE}, {POLY_DEGREE + 1} params)",
        dates, lp, pfit, presid,
        "fit_poly8.svg",
        poly_text, p_r2, p_sigma,
    )
    pd.DataFrame({
        "date": dates, "years": t, "log_price": lp, "fit": pfit,
        "residual": presid,
    }).to_csv("fit_poly8.csv", index=False, float_format="%.6f")
    print("  Saved fit_poly8.svg + fit_poly8.csv")

    # ── Fourier ──────────────────────────────────────────────────────────
    print(f"\nFitting {FOURIER_HARMONICS}-harmonic Fourier "
          f"(1 + 2\u00d7{FOURIER_HARMONICS} = {1 + 2 * FOURIER_HARMONICS} params)...")
    a0, As, Bs, ffit, fresid, T, t_min = fit_fourier(t, lp, FOURIER_HARMONICS)
    f_r2 = 1.0 - float(np.sum(fresid ** 2)) / ss_tot
    f_sigma = float(np.std(fresid))
    # Build amplitude/phase representation for the text box (more readable)
    fourier_lines = [f"a0 = {a0:+.4f}   T = {T:.2f} yr"]
    for k in range(FOURIER_HARMONICS):
        A, B = As[k], Bs[k]
        amp = float(np.hypot(A, B))
        phi = float(np.arctan2(B, A))  # sin-phase form
        period_yr = T / (k + 1)
        fourier_lines.append(
            f"k={k+1}  A={A:+.4f}  B={B:+.4f}  "
            f"|amp|={amp:.4f}  T/k={period_yr:.2f}yr")
    fourier_text = "\n".join(fourier_lines)
    print(f"  R\u00b2 = {f_r2:.5f}   \u03c3 = {f_sigma:.5f}")
    print(f"  Fundamental period T = {T:.2f} years")
    plot_model(
        f"Fourier ({FOURIER_HARMONICS} harmonics, "
        f"{1 + 2 * FOURIER_HARMONICS} params, T={T:.2f}yr)",
        dates, lp, ffit, fresid,
        "fit_fourier4.svg",
        fourier_text, f_r2, f_sigma,
    )
    pd.DataFrame({
        "date": dates, "years": t, "log_price": lp, "fit": ffit,
        "residual": fresid,
    }).to_csv("fit_fourier4.csv", index=False, float_format="%.6f")
    print("  Saved fit_fourier4.svg + fit_fourier4.csv")

    # ── Polynomial on excess (log-price minus BM support) ───────────────
    print(f"\nFitting degree-{POLY_DEGREE} polynomial to EXCESS "
          f"(log-price - BM support)...")
    pe_coeffs, pe_fit, pe_resid = fit_polynomial(t, excess, POLY_DEGREE)
    pe_r2 = 1.0 - float(np.sum(pe_resid ** 2)) / ss_tot_exc
    pe_sigma = float(np.std(pe_resid))
    pe_rev = pe_coeffs[::-1]
    pe_text = (f"A_sup={sup.intercept:+.4f}  B_sup={sup.slope:+.4f}\n"
               + "\n".join(f"a{i} = {pe_rev[i]:+.4e}"
                           for i in range(len(pe_rev))))
    print(f"  R\u00b2(on excess) = {pe_r2:.5f}   \u03c3 = {pe_sigma:.5f}")
    plot_model(
        f"Polynomial on excess (deg {POLY_DEGREE}, {POLY_DEGREE + 1} params)",
        dates, excess, pe_fit, pe_resid,
        "fit_poly8_excess.svg",
        pe_text, pe_r2, pe_sigma,
    )
    pd.DataFrame({
        "date": dates, "years": t, "excess": excess,
        "fit": pe_fit, "residual": pe_resid,
    }).to_csv("fit_poly8_excess.csv", index=False, float_format="%.6f")
    print("  Saved fit_poly8_excess.svg + fit_poly8_excess.csv")

    # ── Fourier on excess ─────────────────────────────────────────────────
    print(f"\nFitting {FOURIER_HARMONICS}-harmonic Fourier to EXCESS...")
    fe_a0, fe_As, fe_Bs, fe_fit, fe_resid, fe_T, _ = fit_fourier(
        t, excess, FOURIER_HARMONICS)
    fe_r2 = 1.0 - float(np.sum(fe_resid ** 2)) / ss_tot_exc
    fe_sigma = float(np.std(fe_resid))
    fe_lines = [f"A_sup={sup.intercept:+.4f}  B_sup={sup.slope:+.4f}",
                f"a0 = {fe_a0:+.4f}   T = {fe_T:.2f} yr"]
    for k in range(FOURIER_HARMONICS):
        A, B = fe_As[k], fe_Bs[k]
        amp = float(np.hypot(A, B))
        period_yr = fe_T / (k + 1)
        fe_lines.append(
            f"k={k+1}  A={A:+.4f}  B={B:+.4f}  "
            f"|amp|={amp:.4f}  T/k={period_yr:.2f}yr")
    fe_text = "\n".join(fe_lines)
    print(f"  R\u00b2(on excess) = {fe_r2:.5f}   \u03c3 = {fe_sigma:.5f}")
    plot_model(
        f"Fourier on excess ({FOURIER_HARMONICS} harmonics, "
        f"{1 + 2 * FOURIER_HARMONICS} params, T={fe_T:.2f}yr)",
        dates, excess, fe_fit, fe_resid,
        "fit_fourier4_excess.svg",
        fe_text, fe_r2, fe_sigma,
    )
    pd.DataFrame({
        "date": dates, "years": t, "excess": excess,
        "fit": fe_fit, "residual": fe_resid,
    }).to_csv("fit_fourier4_excess.csv", index=False, float_format="%.6f")
    print("  Saved fit_fourier4_excess.svg + fit_fourier4_excess.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
