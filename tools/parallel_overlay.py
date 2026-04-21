#!/usr/bin/env python3
"""
Overlay fitted quantile lines on:
  - log price vs log(block - offset)   [log-log]
  - log price vs date                  [log-linear]

Two figures:
  parallel_overlay_low.jpg   Q5/Q10/Q15/Q20  at offset=33,000
  parallel_overlay_mid.jpg   Q10/Q20/Q30/Q40/Q50  at offset=37,200
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
DATES = df["date"].values   # numpy datetime64

N_BINS = 10

CONFIGS = [
    dict(
        target_qs  = [5, 10, 15, 20],
        offset     = 33_000,
        colors     = ["#FF4444", "#FF9900", "#FFDD00", "#44CC44"],
        out        = "parallel_overlay_low.jpg",
        title      = "Q5/Q10/Q15/Q20 at optimal offset = 33,000 blocks",
    ),
    dict(
        target_qs  = [10, 20, 30, 40, 50],
        offset     = 37_200,
        colors     = ["#FF4444", "#FF9900", "#FFDD00", "#44CC44", "#44AAFF"],
        out        = "parallel_overlay_mid.jpg",
        title      = "Q10/Q20/Q30/Q40/Q50 at optimal offset = 37,200 blocks",
    ),
]


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


def _fit(target_qs, offset):
    t    = BLK - offset
    mask = t > 0
    lt   = np.log10(t[mask])
    y    = LOG_P[mask]
    w    = _weights_log_density(lt)
    edges = np.linspace(lt.min(), lt.max(), N_BINS + 1)
    bids  = np.clip(np.digitize(lt, edges[:-1]) - 1, 0, N_BINS - 1)
    fits  = {}
    for q in target_qs:
        xs, ys, bws = [], [], []
        for b in range(N_BINS):
            bm = bids == b
            if bm.sum() < 5: continue
            xs.append(np.average(lt[bm], weights=w[bm]))
            ys.append(np.percentile(y[bm], q))
            bws.append(w[bm].sum())
        if len(xs) < 3:
            fits[q] = (np.nan, np.nan); continue
        xs, ys, bws = map(np.array, (xs, ys, bws))
        fits[q] = _wls_fit(xs, ys, bws)
    return fits, mask, lt, y


# dense block → date interpolator (linear between data points)
blk_sorted = BLK.copy()
date_num   = mdates.date2num(pd.to_datetime(DATES))


def blk_to_datenum(blk_arr):
    return np.interp(blk_arr, blk_sorted, date_num)


for cfg in CONFIGS:
    target_qs = cfg["target_qs"]
    offset    = cfg["offset"]
    colors    = cfg["colors"]

    fits, mask, lt, y_sc = _fit(target_qs, offset)
    t_pos   = BLK - offset
    blk_vis = BLK[mask]

    slopes = [fits[q][1] for q in target_qs if not np.isnan(fits[q][1])]
    slope_std = np.std(slopes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"{cfg['title']}\nslope std = {slope_std:.5f}  "
        f"(slopes: {', '.join(f'Q{q}%={fits[q][1]:.3f}' for q in target_qs)})",
        fontsize=10,
    )

    # ── Panel 1: log price vs log(block − offset) ─────────────────────────────
    ax = axes[0]
    ax.scatter(lt, y_sc, s=2, c="#888888", alpha=0.15, linewidths=0, rasterized=True)

    lt_grid = np.linspace(lt.min(), lt.max(), 600)
    for q, col in zip(target_qs, colors):
        a, b = fits[q]
        if np.isnan(a): continue
        ax.plot(lt_grid, a + b * lt_grid, color=col, lw=2.0,
                label=f"Q{q}%  b={b:.4f}")

    ax.set_xlabel(f"log₁₀(block − {offset:,})", fontsize=10)
    ax.set_ylabel("log₁₀(price  USD)", fontsize=10)
    ax.set_title("log-log", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    # ── Panel 2: log price vs date  (log-linear) ─────────────────────────────
    ax = axes[1]

    dates_vis = blk_to_datenum(blk_vis)
    ax.scatter(dates_vis, y_sc, s=2, c="#888888", alpha=0.15,
               linewidths=0, rasterized=True)

    # extend line to 2035 for context
    blk_future = BLK.max() + 365 * 6 * 144   # ~6 years of blocks
    blk_line   = np.linspace(blk_vis.min(), blk_future, 800)
    t_line     = blk_line - offset
    lt_line    = np.log10(np.maximum(t_line, 1))
    date_line  = blk_to_datenum(np.clip(blk_line, blk_sorted.min(), blk_sorted.max()))
    # for future blocks, extrapolate dates linearly from last known rate
    blocks_per_day = np.diff(blk_sorted[-30:]).mean() / np.diff(date_num[-30:]).mean()
    date_line_full = np.where(
        blk_line <= blk_sorted[-1],
        np.interp(blk_line, blk_sorted, date_num),
        date_num[-1] + (blk_line - blk_sorted[-1]) / blocks_per_day,
    )

    for q, col in zip(target_qs, colors):
        a, b = fits[q]
        if np.isnan(a): continue
        ax.plot(date_line_full, a + b * lt_line,
                color=col, lw=2.0, label=f"Q{q}%")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("log₁₀(price  USD)", fontsize=10)
    ax.set_title("log-linear", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = ROOT / "btc_web" / "assets" / "research" / cfg["out"]
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
