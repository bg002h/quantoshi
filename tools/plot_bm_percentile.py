"""Plot: BTC price's BM-implied percentile over time.

For each historical daily close, compute BM.find_percentile(t, price).
Saves SVG at repo root as bm_percentile.svg (served by /B route).
"""
from __future__ import annotations

import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "btc_web"))

from btc_core import load_model_data
from btc_core._simple import BubbleModel


def main():
    md = load_model_data(os.path.join(_ROOT, "model_data.pkl"))
    bm = BubbleModel(md)

    # Historical data
    dates = pd.to_datetime(md.price_dates)
    prices = np.asarray(md.price_prices, float)
    t_yr = np.asarray(md.price_years, float)

    # Compute percentile at each date (vectorize via loop — ~3.5k points, tolerable)
    pct = np.array([bm.find_percentile(t_yr[i], prices[i]) for i in range(len(prices))])

    fig, ax = plt.subplots(figsize=(13, 7), dpi=140)

    # Shade three "band" regions
    ax.axhspan(0.0, 0.1, alpha=0.15, color="#2166ac", label="_nolegend_")   # deep-bear
    ax.axhspan(0.1, 0.5, alpha=0.08, color="#4393c3", label="_nolegend_")
    ax.axhspan(0.5, 0.9, alpha=0.08, color="#d6604d", label="_nolegend_")
    ax.axhspan(0.9, 1.0, alpha=0.15, color="#b2182b", label="_nolegend_")   # bubble-peak

    # Reference lines
    for q, style in [(0.5, "-"), (0.1, "--"), (0.9, "--"), (0.01, ":"), (0.99, ":")]:
        ax.axhline(q, color="#555", lw=0.9, ls=style, alpha=0.6)

    # Percentile trace
    ax.plot(dates, pct, lw=0.8, color="#000", alpha=0.85)

    # Scatter dots colored by extremity
    extremes_mask = (pct < 0.05) | (pct > 0.95)
    ax.scatter(dates[extremes_mask], pct[extremes_mask], s=6,
               c=np.where(pct[extremes_mask] < 0.5, "#2166ac", "#b2182b"),
               alpha=0.5, zorder=5)

    ax.set_ylim(0, 1)
    ax.set_ylabel("BTC price's BM-implied percentile")
    ax.set_xlabel("Date")
    ax.set_title(
        f"BTC daily close percentile vs Bubble Model (genesis {md.genesis.date().isoformat()})\n"
        f"Asymmetric shrinking σ, q = Φ((log P − log composite)/σ(t))",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"Q{y*100:.0f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="both", ls=":", alpha=0.35)

    # Annotate a few canonical extrema
    for lbl, yr, mo in [
        ("2013 top", 2013, 12), ("2015 trough", 2015, 1),
        ("2017 top", 2017, 12), ("2018 trough", 2018, 12),
        ("2021 top", 2021, 11), ("2022 trough", 2022, 11),
    ]:
        target = pd.Timestamp(year=yr, month=mo, day=15)
        i = int(np.argmin(np.abs(dates - target)))
        ax.annotate(
            lbl, xy=(dates[i], pct[i]),
            xytext=(0, 10 if pct[i] > 0.5 else -18),
            textcoords="offset points",
            ha="center", fontsize=8, color="#333",
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.5),
        )

    fig.tight_layout()
    out = os.path.join(_ROOT, "bm_percentile.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"Saved: {out}")

    # Also save a JPG for quick previews
    jpg = os.path.join(_ROOT, "bm_percentile.jpg")
    fig.savefig(jpg, format="jpg", dpi=140, bbox_inches="tight")
    print(f"Saved: {jpg}")


if __name__ == "__main__":
    main()
