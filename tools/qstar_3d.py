#!/usr/bin/env python3
"""3D surface plot of A2 cvar(offset, quantile) for log-density weighting."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

Q_LEVELS  = np.arange(5, 96, 5)
N_BINS    = 10
N_WORKERS = 19

def _weights_log_density(log_t):
    n = len(log_t)
    kde = scipy.stats.gaussian_kde(log_t)
    w   = 1.0 / np.maximum(kde(log_t), 1e-9)
    return w * (n / w.sum())

def _wls_fit(x, y, w):
    w = w / w.sum()
    xm = np.dot(w, x); ym = np.dot(w, y)
    b  = np.dot(w, (x - xm) * (y - ym)) / np.dot(w, (x - xm) ** 2)
    return ym - b * xm, b

def _a2_cvar(lt, y, w, bids, pct):
    xs, ys, bws = [], [], []
    for b in range(N_BINS):
        bm = bids == b
        if bm.sum() < 5: continue
        xs.append(np.average(lt[bm], weights=w[bm]))
        ys.append(np.percentile(y[bm], pct))
        bws.append(w[bm].sum())
    if len(xs) < 3:
        return float("nan")
    xs, ys, bws = map(np.array, (xs, ys, bws))
    a, b_fit = _wls_fit(xs, ys, bws)
    fb = []
    for b in range(N_BINS):
        bm = bids == b
        if bm.sum() < 5: continue
        y_line = a + b_fit * np.average(lt[bm], weights=w[bm])
        fb.append(np.mean(y[bm] < y_line))
    return float(np.var(fb)) if len(fb) >= 3 else float("nan")

def _worker(off):
    t = BLK - off
    mask = t > 0
    if mask.sum() < 30:
        return off, None
    lt    = np.log10(t[mask])
    y     = LOG_P[mask]
    w     = _weights_log_density(lt)
    edges = np.linspace(lt.min(), lt.max(), N_BINS + 1)
    bids  = np.clip(np.digitize(lt, edges[:-1]) - 1, 0, N_BINS - 1)
    return off, {q: _a2_cvar(lt, y, w, bids, q) for q in Q_LEVELS}

if __name__ == "__main__":
    offsets = np.arange(0, 37_501, 150)
    n_off   = len(offsets)
    nq      = len(Q_LEVELS)

    print(f"Sweeping {n_off} offsets × {nq} Q-levels (log-density), {N_WORKERS} workers…",
          flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, offsets, chunksize=8)

    grid = np.full((n_off, nq), float("nan"))
    for off, cvars in raw:
        if cvars is None: continue
        i = int(np.searchsorted(offsets, off))
        for j, q in enumerate(Q_LEVELS):
            grid[i, j] = cvars[q]

    # Clip extreme values for visibility
    vmax = np.nanpercentile(grid, 95)
    grid_clipped = np.clip(grid, 0, vmax)

    # q*(offset) for overlay
    qstar = np.full(n_off, float("nan"))
    for i in range(n_off):
        row = grid[i]
        fin = np.isfinite(row)
        if fin.sum() < 2: continue
        j = np.where(fin, row, np.inf).argmin()
        if 0 < j < nq - 1 and fin[j-1] and fin[j+1]:
            y0, y1, y2 = row[j-1], row[j], row[j+1]
            denom = 2 * (y0 - 2*y1 + y2)
            if abs(denom) > 1e-12:
                frac = np.clip((y0 - y2) / denom, -0.5, 0.5)
                qstar[i] = Q_LEVELS[j] + frac * (Q_LEVELS[1] - Q_LEVELS[0])
                continue
        qstar[i] = float(Q_LEVELS[j])

    # ── 3D surface ────────────────────────────────────────────────────────────
    OFF_GRID, Q_GRID = np.meshgrid(offsets, Q_LEVELS, indexing="ij")

    fig = plt.figure(figsize=(14, 9))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        OFF_GRID, Q_GRID, grid_clipped,
        cmap="viridis_r",
        linewidth=0, antialiased=True, alpha=0.88,
        vmin=0, vmax=vmax,
    )

    # q*(offset) ridge line plotted at z = 0 (floor) and elevated
    valid = np.isfinite(qstar)
    qs_z  = np.array([
        grid_clipped[i, int(np.searchsorted(Q_LEVELS, qstar[i])) if np.isfinite(qstar[i]) else 0]
        for i in range(n_off)
    ])
    ax.plot(offsets[valid], qstar[valid], qs_z[valid],
            color="white", lw=2.0, zorder=5, label="q*(offset) ridge")

    # target lines at q=25 and q=50
    for tgt, col in [(25, "#FF4444"), (50, "#44AAFF")]:
        j_tgt = int(np.searchsorted(Q_LEVELS, tgt))
        z_tgt = grid_clipped[:, j_tgt]
        ax.plot(offsets, np.full(n_off, tgt), z_tgt,
                color=col, lw=1.5, ls="--", alpha=0.8, label=f"Q{tgt}% slice")
        ax.plot(offsets, np.full(n_off, tgt), np.zeros(n_off),
                color=col, lw=0.8, ls=":", alpha=0.4)

    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.04,
                 label="cvar (lower = more consistent)")

    ax.set_xlabel("Block offset", fontsize=9, labelpad=8)
    ax.set_ylabel("Quantile level (%)", fontsize=9, labelpad=8)
    ax.set_zlabel("A2 cvar", fontsize=9, labelpad=6)
    ax.set_title(
        "Log-density weighted  —  A2 consistency variance surface\n"
        "cvar(offset, Q)  |  White ridge = q*(offset)  |  "
        "Red dashed = Q25% slice  |  Blue dashed = Q50% slice",
        fontsize=10,
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x)))
    )
    ax.view_init(elev=28, azim=-50)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = ROOT.parent / "debris" / "research" / "qstar_3d.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
