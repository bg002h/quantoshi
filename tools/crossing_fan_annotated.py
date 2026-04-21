#!/usr/bin/env python3
"""
Annotated crossing-symmetry fan plot — log-density weighted.
Shows offset=0 vs optimal (29,700), with crossing points colour-coded:
  green  = crossing falls within the data range  (x_c in [lt.min, lt.max])
  red    = crossing is extrapolated outside the data range
Extrapolated crossings are projected onto the data boundary with a dashed arrow.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats

ROOT = Path(__file__).resolve().parent.parent

prices = pd.read_csv(ROOT / "BitcoinPricesDaily.csv")
prices["date"] = pd.to_datetime(prices["Date"], format="%m/%d/%y")
blocks = pd.read_csv(ROOT / "BitcoinBlocksDaily.csv", parse_dates=["date"])
df = (prices.merge(blocks, on="date", how="inner")
           .query("Price > 0")
           .dropna(subset=["Price", "blockheight"])
           .sort_values("date")
           .reset_index(drop=True))

LOG_P = np.log10(df["Price"].values)
BLK   = df["blockheight"].values.astype(np.float64)

Q_LEVELS = list(range(5, 96, 5))
PAIRS    = [(q, 100-q) for q in range(5, 50, 5)]
N_BINS   = 10
N_WORKERS = 19
OFFSETS_SHOW = [0, 29_700]

C_IN   = "#44DD66"   # crossing within data range
C_OUT  = "#FF4444"   # crossing extrapolated outside data


def _weights_log_density(log_t):
    n   = len(log_t)
    kde = scipy.stats.gaussian_kde(log_t)
    w   = 1.0 / np.maximum(kde(log_t), 1e-9)
    return w * (n / w.sum())


def _wls_fit(x, y, w):
    w = w / w.sum()
    xm = np.dot(w, x); ym = np.dot(w, y)
    b  = np.dot(w, (x - xm) * (y - ym)) / np.dot(w, (x - xm) ** 2)
    return ym - b * xm, b


def _fit_quantile_lines(off):
    t = BLK - off
    mask = t > 0
    if mask.sum() < 30:
        return None, None, None
    lt    = np.log10(t[mask])
    y     = LOG_P[mask]
    w     = _weights_log_density(lt)
    edges = np.linspace(lt.min(), lt.max(), N_BINS + 1)
    bids  = np.clip(np.digitize(lt, edges[:-1]) - 1, 0, N_BINS - 1)
    result = {}
    for q in Q_LEVELS:
        xs, ys, bws = [], [], []
        for b in range(N_BINS):
            bm = bids == b
            if bm.sum() < 5: continue
            xs.append(np.average(lt[bm], weights=w[bm]))
            ys.append(np.percentile(y[bm], q))
            bws.append(w[bm].sum())
        if len(xs) < 3:
            result[q] = (np.nan, np.nan)
            continue
        xs, ys, bws = map(np.array, (xs, ys, bws))
        result[q] = _wls_fit(xs, ys, bws)
    return result, lt, y[mask] if mask.any() else y


def _worker(off):
    fits, lt, y = _fit_quantile_lines(off)
    if fits is None:
        return off, float("nan"), []
    a50, b50 = fits.get(50, (np.nan, np.nan))
    if np.isnan(a50):
        return off, float("nan"), []
    deviations = []
    details    = []
    for q_lo, q_hi in PAIRS:
        a_lo, b_lo = fits.get(q_lo, (np.nan, np.nan))
        a_hi, b_hi = fits.get(q_hi, (np.nan, np.nan))
        if any(np.isnan(v) for v in [a_lo, b_lo, a_hi, b_hi]):
            continue
        db = b_lo - b_hi
        if abs(db) < 1e-10:
            continue
        x_c  = (a_hi - a_lo) / db
        y_c  = a_lo + b_lo * x_c
        y_50 = a50 + b50 * x_c
        deviations.append(y_c - y_50)
        details.append((x_c, y_c, y_50, q_lo, q_hi))
    score = float(np.sqrt(np.mean(np.array(deviations) ** 2))) if deviations else float("nan")
    return off, score, details


if __name__ == "__main__":
    print("Computing fits for two offsets…", flush=True)

    q_cmap = plt.cm.coolwarm
    q_norm = plt.Normalize(vmin=5, vmax=95)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle(
        "Crossing-symmetry fan — log-density weighted\n"
        "Green dot = crossing within data range  |  Red dot = extrapolated outside data range\n"
        "Dashed line shows extrapolated crossing projected to data boundary",
        fontsize=11,
    )

    for ax, off in zip(axes, OFFSETS_SHOW):
        fits, lt, ydata = _fit_quantile_lines(off)
        _, score, details = _worker(off)

        lt_min, lt_max = lt.min(), lt.max()

        # scatter data
        ax.scatter(lt, ydata, s=2, c="#888", alpha=0.15,
                   linewidths=0, rasterized=True)

        # quantile lines — extend slightly beyond data for context
        pad = (lt_max - lt_min) * 0.04
        lt_grid = np.linspace(lt_min - pad, lt_max + pad, 400)
        for q in Q_LEVELS:
            a, b = fits[q]
            if np.isnan(a): continue
            color = q_cmap(q_norm(q))
            lw    = 2.5 if q == 50 else 0.9
            zord  = 4   if q == 50 else 2
            ax.plot(lt_grid, a + b * lt_grid, color=color, lw=lw,
                    zorder=zord, alpha=0.85)

        # data range boundary lines
        ax.axvline(lt_min, color="#FFFFFF", lw=0.8, ls=":", alpha=0.4, zorder=1)
        ax.axvline(lt_max, color="#FFFFFF", lw=0.8, ls=":", alpha=0.4, zorder=1)

        n_in = n_out = 0
        a50, b50 = fits[50]

        for x_c, y_c, y_50, q_lo, q_hi in details:
            pair_color = q_cmap(q_norm(q_lo))
            in_range = lt_min <= x_c <= lt_max

            if in_range:
                n_in += 1
                ax.scatter([x_c], [y_c], s=70, zorder=6,
                           color=C_IN, edgecolors="white", linewidths=0.6)
                ax.plot([x_c, x_c], [y_c, y_50], color=C_IN,
                        lw=0.8, ls=":", alpha=0.7, zorder=5)
            else:
                n_out += 1
                # project crossing onto nearest data boundary
                x_proj = lt_min if x_c < lt_min else lt_max
                y_proj_line = a50 + b50 * x_proj

                # dot at actual crossing (may be off-axis, clip for annotation)
                ax.scatter([x_c], [y_c], s=70, zorder=6,
                           color=C_OUT, edgecolors="white", linewidths=0.6,
                           clip_on=False)

                # dashed arrow from data boundary to crossing location
                ax.annotate(
                    "",
                    xy=(x_c, y_c),
                    xytext=(x_proj, y_proj_line),
                    arrowprops=dict(
                        arrowstyle="->", color=C_OUT, lw=0.8,
                        linestyle="dashed",
                        connectionstyle="arc3,rad=0.0",
                    ),
                    clip_on=False,
                )

        # Q50 line
        ax.plot(lt_grid, a50 + b50 * lt_grid, color="white",
                lw=2.5, zorder=7, label="Q50%")

        ax.set_title(
            f"Offset = {off:,.0f} blocks  |  score = {score:.5f}\n"
            f"Crossings in range: {n_in}/{len(details)}   extrapolated: {n_out}/{len(details)}",
            fontsize=10,
        )
        ax.set_xlabel("log₁₀(block − offset)", fontsize=9)
        ax.grid(True, alpha=0.15)

        # legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="white", lw=2.5, label="Q50%"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_IN,
                   markersize=8, label=f"Crossing in range ({n_in})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OUT,
                   markersize=8, label=f"Extrapolated ({n_out})"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper left")

    axes[0].set_ylabel("log₁₀(price)", fontsize=9)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=q_cmap, norm=q_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Quantile level (%)", fontsize=9)

    fig.tight_layout()
    out = ROOT / "btc_web" / "assets" / "research" / "crossing_fan_annotated.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
