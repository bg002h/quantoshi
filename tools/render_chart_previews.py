#!/usr/bin/env python3
"""Render static PNG previews of all chart tabs for instant first-paint.

Generates approximate placeholder images via matplotlib — they don't need
to be pixel-perfect, they just show users the rough shape while Plotly
hydrates. Each is hidden when the real interactive chart renders.

Outputs (all in btc_web/assets/):
    bubble_preview.png       — log-log price + BM support/composite
    heatmap_preview.png      — CAGR heatmap grid (years × years)
    dca_preview.png          — DCA accumulation curves (log-y)
    retire_preview.png       — retirement withdrawal depletion
    supercharge_preview.png  — HODL supercharger bands
    citadel_preview.png      — Citadel wealth trajectory

Run manually or via the daily update pipeline.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive" / "btc_app"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from btc_core import ModelData

ASSETS = ROOT / "btc_web" / "assets"
GENESIS_YR = 2009 + 206/365.25  # 2009-07-25


def _fmt_price(y, _):
    if y >= 1e6: return f"${y/1e6:.0f}M"
    if y >= 1e3: return f"${y/1e3:.0f}K"
    if y >= 1:   return f"${y:.0f}"
    return f"${y:.2f}"


def _base_axes(figsize=(14, 8), dpi=70):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, which="major", color="#888888", linewidth=1.0, alpha=0.6)
    ax.tick_params(colors="#444", labelsize=12)
    for spine in ax.spines.values():
        spine.set_color("#888")
        spine.set_linewidth(1)
    return fig, ax


def _save(fig, name):
    out = ASSETS / f"{name}_preview.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=70, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    plt.close(fig)
    print(f"  {name}: {out.stat().st_size // 1024} KB")


def render_bubble(M):
    yr_min, yr_max = 2010, 2034
    t_min = yr_min - GENESIS_YR
    t_max = yr_max - GENESIS_YR
    fig, ax = _base_axes()

    mask = (M.price_years >= t_min) & (M.price_years <= t_max)
    yrs = M.price_years[mask]
    prs = M.price_prices[mask]
    if len(yrs) > 600:
        idx = np.linspace(0, len(yrs) - 1, 600, dtype=int)
        yrs, prs = yrs[idx], prs[idx]
    ax.scatter(yrs + GENESIS_YR, prs, s=10, c="#f7931a", alpha=0.5,
               edgecolors="none", zorder=3)

    mask_bm = (M.years_plot_bm >= t_min) & (M.years_plot_bm <= t_max)
    t_bm = M.years_plot_bm[mask_bm] + GENESIS_YR
    ax.plot(t_bm, M.support_bm[mask_bm], color="#C8960C", linestyle="--",
            linewidth=2, alpha=0.8, zorder=4)
    comp = M.comp_by_n[3] if len(M.comp_by_n) > 3 else M.comp_by_n[0]
    ax.plot(t_bm, comp[mask_bm], color="#C8960C", linewidth=2.5, alpha=0.9, zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(yr_min, yr_max)
    ax.set_ylim(0.01, 5e6)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_price))
    ax.set_xlabel("Year", color="#444", fontsize=13)
    ax.set_ylabel("BTC price (USD)", color="#444", fontsize=13)
    _save(fig, "bubble")


def render_heatmap(M):
    """Placeholder CAGR heatmap — years × years grid with a diverging colormap."""
    fig, ax = _base_axes(figsize=(10, 8))
    entry_years = np.arange(2011, 2027)
    exit_years = np.arange(2011, 2041)
    # Fake CAGR grid: decreasing with time distance (placeholder pattern)
    Z = np.zeros((len(entry_years), len(exit_years)))
    for i, y_in in enumerate(entry_years):
        for j, y_out in enumerate(exit_years):
            dt = y_out - y_in
            if dt <= 0:
                Z[i, j] = np.nan
            else:
                # approximate: older entries give higher CAGR, scaled
                Z[i, j] = max(-30, 150 * (1.0 / dt ** 0.7) * (1 - (y_in - 2011) / 25))
    im = ax.imshow(Z, cmap="RdBu_r", aspect="auto", vmin=-30, vmax=150,
                   extent=[exit_years[0], exit_years[-1],
                           entry_years[-1], entry_years[0]],
                   interpolation="nearest")
    ax.set_xlabel("Exit Year", color="#444", fontsize=13)
    ax.set_ylabel("Entry Year", color="#444", fontsize=13)
    plt.colorbar(im, ax=ax, label="CAGR %", shrink=0.8)
    _save(fig, "heatmap")


def render_dca(M):
    """Placeholder DCA chart — BTC accumulation over time (log-y)."""
    fig, ax = _base_axes()
    years = np.linspace(2025, 2040, 200)
    # Simple accumulation curve: sats accumulated via fixed $1200/mo DCA
    # Placeholder: exponential growth
    for i, color in enumerate(["#2ecc71", "#3498db", "#9b59b6"]):
        base = 0.01 * (i + 1)
        curve = base * np.exp(0.15 * (years - 2025))
        ax.plot(years, curve, color=color, linewidth=2.5, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlim(2025, 2040)
    ax.set_ylim(0.01, 100)
    ax.set_xlabel("Year", color="#444", fontsize=13)
    ax.set_ylabel("BTC Stack", color="#444", fontsize=13)
    _save(fig, "dca")


def render_retire(M):
    """Placeholder Retire chart — stack depleting over time."""
    fig, ax = _base_axes()
    years = np.linspace(2031, 2075, 300)
    for i, (color, start) in enumerate([("#e74c3c", 1.0), ("#f39c12", 2.0), ("#2ecc71", 5.0)]):
        # Fake depletion curves — high percentile lasts longer
        decay = np.exp(-0.06 * (years - 2031) / (1 + i * 0.5))
        curve = start * decay
        curve[curve < 0.0001] = np.nan
        ax.plot(years, curve, color=color, linewidth=2.5, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlim(2031, 2075)
    ax.set_ylim(0.001, 10)
    ax.set_xlabel("Year", color="#444", fontsize=13)
    ax.set_ylabel("BTC Remaining", color="#444", fontsize=13)
    _save(fig, "retire")


def render_supercharge(M):
    """Placeholder HODL Supercharger — sustainable withdrawal bands."""
    fig, ax = _base_axes()
    years = np.linspace(2033, 2075, 300)
    base = np.linspace(50000, 500000, 300)
    # Three quantile bands
    for q, color, mult in [("Q0.1%", "#e74c3c", 0.6),
                            ("Q10%", "#f39c12", 1.0),
                            ("Q50%", "#2ecc71", 1.8)]:
        curve = base * mult * np.exp(0.03 * np.random.randn(len(years)).cumsum())
        ax.plot(years, curve, color=color, linewidth=2.5, alpha=0.85, label=q)
    ax.set_yscale("log")
    ax.set_xlim(2033, 2075)
    ax.set_xlabel("Year", color="#444", fontsize=13)
    ax.set_ylabel("Withdrawal / Spending (USD)", color="#444", fontsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_price))
    _save(fig, "supercharge")


def render_citadel(M):
    """Placeholder Citadel — portfolio wealth trajectory."""
    fig, ax = _base_axes()
    years = np.linspace(2031, 2075, 300)
    # Fake wealth curves
    np.random.seed(42)
    for i, color in enumerate(["#8B4513", "#D4760A", "#228B22"]):
        base = 1e6 * (i + 1)
        curve = base * np.exp(0.08 * (years - 2031) + 0.2 * np.random.randn(len(years)).cumsum())
        ax.plot(years, curve, color=color, linewidth=2.5, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlim(2031, 2075)
    ax.set_xlabel("Year", color="#444", fontsize=13)
    ax.set_ylabel("Portfolio Value (USD)", color="#444", fontsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_price))
    _save(fig, "citadel")


def main():
    print("Rendering chart previews...")
    M = ModelData(str(ROOT / "model_data.pkl"))
    render_bubble(M)
    render_heatmap(M)
    render_dca(M)
    render_retire(M)
    render_supercharge(M)
    render_citadel(M)
    print("Done.")


if __name__ == "__main__":
    main()
