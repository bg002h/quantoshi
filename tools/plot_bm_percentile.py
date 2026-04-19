"""Plot: BTC price's BM-implied percentile over time.

For each historical daily close, compute BM.find_percentile(t, price).
Saves SVG at repo root as bm_percentile.svg (served by /B route).

Variants:
  - default (shrinking sigma): --> bm_percentile.{svg,jpg}
  - --flat (constant sigma):   --> bm_percentile_flat.{svg,jpg}
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "btc_web"))

from btc_core import load_model_data
from btc_core._simple import BubbleModel


def _percentile_flat(md, t_yr, price):
    """Constant-sigma percentile: ignores t^(-alpha) shrinkage.

    log10(composite) at t → residual in log-space → z-score under a
    constant σ (σ_up if price above composite, σ_down if below) → CDF.
    """
    log_p = np.log10(max(float(price), 1e-10))
    log_comp = float(np.interp(t_yr, md.years_plot_bm,
                               np.log10(np.maximum(md.comp_by_n[-1], 1e-10))))
    if log_p >= log_comp:
        sigma = md.bm_sigma0_up
    else:
        sigma = md.bm_sigma0_down
    if sigma < 1e-9:
        return 0.5
    z = (log_p - log_comp) / sigma
    return float(_norm.cdf(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", action="store_true",
                    help="Use non-shrinking (constant) sigma.")
    args = ap.parse_args()

    md = load_model_data(os.path.join(_ROOT, "model_data.pkl"))
    bm = BubbleModel(md)

    # Historical data
    dates = pd.to_datetime(md.price_dates)
    prices = np.asarray(md.price_prices, float)
    t_yr = np.asarray(md.price_years, float)

    # Compute percentile at each date (vectorize via loop — ~3.5k points, tolerable)
    if args.flat:
        pct = np.array([_percentile_flat(md, t_yr[i], prices[i]) for i in range(len(prices))])
        suffix = "_flat"
        subtitle_sigma = (
            f"Constant σ (no shrinkage): σ_up={md.bm_sigma0_up:.4f}, "
            f"σ_down={md.bm_sigma0_down:.4f}"
        )
    else:
        pct = np.array([bm.find_percentile(t_yr[i], prices[i]) for i in range(len(prices))])
        suffix = ""
        subtitle_sigma = (
            f"Shrinking σ: σ(t)=σ₀·t^(−α). "
            f"α_up={md.bm_alpha_up:.3f}, α_down={md.bm_alpha_down:.3f}"
        )

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
        f"{subtitle_sigma}",
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
    out = os.path.join(_ROOT, f"bm_percentile{suffix}.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"Saved: {out}")

    # Also save a JPG for quick previews
    jpg = os.path.join(_ROOT, f"bm_percentile{suffix}.jpg")
    fig.savefig(jpg, format="jpg", dpi=140, bbox_inches="tight")
    print(f"Saved: {jpg}")


if __name__ == "__main__":
    main()
