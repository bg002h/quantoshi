#!/usr/bin/env python3
"""Render a static PNG preview of the bubble chart for instant first-paint.

Runs at app startup (or manually) to create btc_web/assets/bubble_preview.png.
The web app serves this as an <img> overlay that shows while the interactive
Plotly chart is loading. A clientside script hides it once Plotly is ready.

The preview matches the default bubble tab view: log-log axes, price scatter +
BM quantile bands + bubble composite. It does NOT need to be pixel-perfect —
it's a low-fidelity placeholder that hides within 1-2 seconds.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive" / "btc_app"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from btc_core import ModelData

OUT = ROOT / "btc_web" / "assets" / "bubble_preview.png"


def main():
    M = ModelData(str(ROOT / "model_data.pkl"))

    # Default bubble view: log-log, years 2010 → 2034
    yr_min, yr_max = 2010, 2034
    genesis_yr = 2009 + 206/365.25  # 2009-07-25

    t_min = yr_min - genesis_yr
    t_max = yr_max - genesis_yr

    fig, ax = plt.subplots(figsize=(14, 8), dpi=70)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Price data (scatter)
    mask = (M.price_years >= t_min) & (M.price_years <= t_max)
    years_data = M.price_years[mask]
    prices_data = M.price_prices[mask]
    # Downsample to 600 points
    if len(years_data) > 600:
        idx = np.linspace(0, len(years_data) - 1, 600, dtype=int)
        years_data = years_data[idx]
        prices_data = prices_data[idx]
    ax.scatter(years_data + genesis_yr, prices_data,
               s=10, c="#f7931a", alpha=0.5, edgecolors="none", zorder=3)

    # BM support (lower bound)
    mask_bm = (M.years_plot_bm >= t_min) & (M.years_plot_bm <= t_max)
    t_bm = M.years_plot_bm[mask_bm]
    sup = M.support_bm[mask_bm]
    ax.plot(t_bm + genesis_yr, sup, color="#C8960C", linestyle="--",
            linewidth=2, alpha=0.8, zorder=4)

    # BM composite
    comp = M.comp_by_n[3][mask_bm] if len(M.comp_by_n) > 3 else M.comp_by_n[0][mask_bm]
    ax.plot(t_bm + genesis_yr, comp, color="#C8960C",
            linewidth=2.5, alpha=0.9, zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(yr_min, yr_max)
    ax.set_ylim(0.01, 5e6)

    # Clean grid styling matching the Plotly look
    ax.grid(True, which="major", color="#888888", linewidth=1.0, alpha=0.6)
    ax.grid(True, which="minor", color="#B0B0B0", linewidth=0.5, alpha=0.3, linestyle=":")
    ax.tick_params(colors="#444", labelsize=12)
    for spine in ax.spines.values():
        spine.set_color("#888")
        spine.set_linewidth(1)

    # Y-axis price formatting
    from matplotlib.ticker import FuncFormatter
    def fmt(y, _):
        if y >= 1e6: return f"${y/1e6:.0f}M"
        if y >= 1e3: return f"${y/1e3:.0f}K"
        if y >= 1:   return f"${y:.0f}"
        return f"${y:.2f}"
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))

    ax.set_xlabel("Year", color="#444", fontsize=13)
    ax.set_ylabel("BTC price (USD)", color="#444", fontsize=13)

    plt.tight_layout()
    plt.savefig(str(OUT), dpi=70, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    size_kb = OUT.stat().st_size // 1024
    print(f"Wrote {OUT} ({size_kb} KB)")


if __name__ == "__main__":
    main()
