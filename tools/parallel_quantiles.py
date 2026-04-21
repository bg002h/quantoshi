#!/usr/bin/env python3
"""
Find the block offset that makes Q10/Q20/Q30/Q40/Q50 as parallel as possible.

"Parallel" = all five slopes (power-law exponents) equal.
Score = std of the five fitted slopes.  Lower = more parallel.

Uses log-density weighting + bin-percentile WLS (same method as other sweeps).

Outputs:
  parallel_score.jpg   — score vs offset, with optimal marked
  parallel_fan.jpg     — Q10–Q50 fan at offset=0 vs optimal
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

TARGET_QS = [5, 10, 15, 20]
N_BINS    = 10
N_WORKERS = 19


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


def _fit_slopes(off):
    """Return dict {q: (a, b)} for TARGET_QS, or None."""
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
    for q in TARGET_QS:
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
    return result, lt, y


def _worker(off):
    fits, lt, y = _fit_slopes(off)
    if fits is None:
        return off, float("nan"), {}
    slopes = [fits[q][1] for q in TARGET_QS if not np.isnan(fits[q][1])]
    if len(slopes) < 2:
        return off, float("nan"), fits
    score = float(np.std(slopes))
    return off, score, fits


if __name__ == "__main__":
    offsets = np.arange(0, 50_001, 150)
    n_off   = len(offsets)

    print(f"Sweeping {n_off} offsets, {N_WORKERS} workers…", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, offsets.tolist(), chunksize=8)

    scores = np.full(n_off, float("nan"))
    all_fits = {}
    for off, s, fits in raw:
        i = int(np.searchsorted(offsets, off))
        scores[i] = s
        all_fits[off] = fits

    valid   = np.isfinite(scores)
    i_opt   = np.where(valid, scores, np.inf).argmin()
    off_opt = int(offsets[i_opt])
    s_opt   = scores[i_opt]
    s_zero  = scores[int(np.searchsorted(offsets, 0))]

    print(f"\nOffset=0       score (slope std): {s_zero:.6f}")
    print(f"Optimal={off_opt:,}  score (slope std): {s_opt:.6f}")
    print(f"Improvement: {100*(s_zero-s_opt)/s_zero:.1f}%")

    fits_opt  = all_fits[off_opt]
    fits_zero = all_fits[0]
    print(f"\nSlopes at offset=0:      " +
          "  ".join(f"Q{q}%={fits_zero[q][1]:.4f}" for q in TARGET_QS))
    print(f"Slopes at offset={off_opt:,}: " +
          "  ".join(f"Q{q}%={fits_opt[q][1]:.4f}" for q in TARGET_QS))

    fmt_k = mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x))
    )

    # ── Figure 1: score vs offset ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 5))

    COLOR = "#44AAFF"
    ax1.plot(offsets, scores, color=COLOR, lw=1.8)

    ax1.axvline(off_opt, color=COLOR, lw=1.0, ls="--", alpha=0.7)
    ax1.scatter([off_opt], [s_opt], s=60, color=COLOR, zorder=5)
    ax1.annotate(
        f"Optimal: {off_opt:,} blocks\nstd = {s_opt:.5f}",
        xy=(off_opt, s_opt),
        xytext=(off_opt + 1_200, s_opt + (s_zero - s_opt) * 0.15),
        fontsize=9, color=COLOR,
        arrowprops=dict(arrowstyle="->", color=COLOR, lw=0.8),
    )
    ax1.scatter([0], [s_zero], s=60, color="#AAAAAA", zorder=5)
    ax1.annotate(
        f"Offset=0\nstd = {s_zero:.5f}",
        xy=(0, s_zero),
        xytext=(1_500, s_zero + (s_zero - s_opt) * 0.05),
        fontsize=9, color="#AAAAAA",
        arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.8),
    )
    ax1.text(
        0.98, 0.95,
        f"Improvement: {100*(s_zero-s_opt)/s_zero:.1f}%",
        transform=ax1.transAxes, ha="right", va="top", fontsize=10,
        color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.4", fc="#333333", ec="#555555", alpha=0.9),
    )

    ax1.xaxis.set_major_formatter(fmt_k)
    ax1.set_xlabel("Block offset", fontsize=10)
    ax1.set_ylabel("Std of slopes across Q5–Q20", fontsize=10)
    ax1.set_title(
        "Slope parallelism score vs block offset  —  log-density weighted\n"
        "Score = std(slopes) for Q5/Q10/Q15/Q20  |  Lower = more parallel quantile lines",
        fontsize=10,
    )
    ax1.grid(True, alpha=0.2)
    fig1.tight_layout()
    out1 = ROOT / "btc_web" / "assets" / "research" / "parallel_score_low.jpg"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out1}")

    # ── Figure 2: fan at offset=0 vs optimal ─────────────────────────────────
    Q_COLORS = ["#FF4444", "#FF9900", "#FFDD00", "#44CC44", "#44AAFF"]

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig2.suptitle(
        "Q5/Q10/Q15/Q20 fan — log-density weighted\n"
        "At the optimal offset the lines should be parallel (equal slopes / exponents)",
        fontsize=11,
    )

    for ax, off, label in zip(axes2,
                               [0, off_opt],
                               [f"Offset = 0", f"Offset = {off_opt:,} (optimal)"]):
        t_pos = BLK - off
        mask  = t_pos > 0
        lt    = np.log10(t_pos[mask])
        y_sc  = LOG_P[mask]

        ax.scatter(lt, y_sc, s=2, c="#888", alpha=0.15,
                   linewidths=0, rasterized=True)

        fits = all_fits[off]
        slopes_here = [fits[q][1] for q in TARGET_QS if not np.isnan(fits[q][1])]
        std_here = np.std(slopes_here)

        lt_grid = np.linspace(lt.min(), lt.max(), 400)
        for q, col in zip(TARGET_QS, Q_COLORS):
            a, b = fits[q]
            if np.isnan(a): continue
            ax.plot(lt_grid, a + b * lt_grid, color=col, lw=2.0,
                    label=f"Q{q}%  b={b:.4f}")

        ax.set_title(f"{label}\nslope std = {std_here:.5f}", fontsize=10)
        ax.set_xlabel("log₁₀(block − offset)", fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=8, loc="upper left")

    axes2[0].set_ylabel("log₁₀(price)", fontsize=9)
    fig2.tight_layout()
    out2 = ROOT / "btc_web" / "assets" / "research" / "parallel_fan_low.jpg"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
