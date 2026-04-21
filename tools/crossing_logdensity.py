#!/usr/bin/env python3
"""
Crossing-symmetry score (log-density weighted only) vs block offset.
Single clean plot; prints score at offset=0 vs optimal.
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
        return None
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
    return result


def _crossing_score(fits):
    if fits is None:
        return float("nan")
    a50, b50 = fits.get(50, (np.nan, np.nan))
    if np.isnan(a50):
        return float("nan")
    deviations = []
    for q_lo, q_hi in PAIRS:
        a_lo, b_lo = fits.get(q_lo, (np.nan, np.nan))
        a_hi, b_hi = fits.get(q_hi, (np.nan, np.nan))
        if any(np.isnan(v) for v in [a_lo, b_lo, a_hi, b_hi]):
            continue
        db = b_lo - b_hi
        if abs(db) < 1e-10:
            continue
        x_c = (a_hi - a_lo) / db
        y_c = a_lo + b_lo * x_c
        y_50 = a50 + b50 * x_c
        deviations.append(y_c - y_50)
    if not deviations:
        return float("nan")
    return float(np.sqrt(np.mean(np.array(deviations) ** 2)))


def _worker(off):
    fits  = _fit_quantile_lines(off)
    score = _crossing_score(fits)
    return off, score


if __name__ == "__main__":
    offsets = np.arange(0, 37_501, 150)
    n_off   = len(offsets)

    print(f"Sweeping {n_off} offsets (log-density), {N_WORKERS} workers…", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, offsets.tolist(), chunksize=8)

    scores = np.full(n_off, float("nan"))
    for off, s in raw:
        i = int(np.searchsorted(offsets, off))
        scores[i] = s

    # ── Stats ─────────────────────────────────────────────────────────────────
    valid   = np.isfinite(scores)
    i_opt   = np.where(valid, scores, np.inf).argmin()
    off_opt = offsets[i_opt]
    s_opt   = scores[i_opt]

    i_zero  = int(np.searchsorted(offsets, 0))
    s_zero  = scores[i_zero]

    improvement_pct = 100 * (s_zero - s_opt) / s_zero

    print(f"\nOffset=0      score: {s_zero:.6f}")
    print(f"Optimal ({off_opt:,}) score: {s_opt:.6f}")
    print(f"Improvement:  {s_zero - s_opt:.6f}  ({improvement_pct:.1f}%)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    COLOR = "#FF9900"
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(offsets, scores, color=COLOR, lw=1.8)

    # optimal
    ax.axvline(off_opt, color=COLOR, lw=1.0, ls="--", alpha=0.7)
    ax.scatter([off_opt], [s_opt], s=60, color=COLOR, zorder=5)
    ax.annotate(
        f"Optimal: {off_opt:,} blocks\nscore = {s_opt:.4f}",
        xy=(off_opt, s_opt),
        xytext=(off_opt + 1_200, s_opt + (s_zero - s_opt) * 0.08),
        fontsize=9, color=COLOR,
        arrowprops=dict(arrowstyle="->", color=COLOR, lw=0.8),
    )

    # offset = 0
    ax.scatter([0], [s_zero], s=60, color="#AAAAAA", zorder=5)
    ax.annotate(
        f"Offset = 0\nscore = {s_zero:.4f}",
        xy=(0, s_zero),
        xytext=(1_500, s_zero + (s_zero - s_opt) * 0.08),
        fontsize=9, color="#AAAAAA",
        arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.8),
    )

    # improvement label
    ax.text(
        0.98, 0.95,
        f"Improvement: {improvement_pct:.1f}%\n"
        f"Δscore = {s_zero - s_opt:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.4", fc="#333333", ec="#555555", alpha=0.9),
    )

    fmt_k = mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x))
    )
    ax.xaxis.set_major_formatter(fmt_k)
    ax.set_xlabel("Block offset", fontsize=10)
    ax.set_ylabel("RMS crossing deviation (log-price units)", fontsize=10)
    ax.set_title(
        "Crossing-symmetry score vs block offset  —  log-density weighted\n"
        "Score = RMS distance of symmetric-pair crossings (Q10/Q90, Q25/Q75, …) from Q50% line  |  Lower = more symmetric fan",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = ROOT / "btc_web" / "assets" / "research" / "crossing_logdensity.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
