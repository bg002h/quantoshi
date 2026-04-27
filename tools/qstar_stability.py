#!/usr/bin/env python3
"""
Fine-resolution sweep of A2 cvar(offset, q) for log-density weighting.
Investigates stability of two features seen in the 3D surface:
  - Main valley  (offset ~23k, Q ~10–25%)
  - Secondary feature (offset ~0–5k, Q ~75–85%)

Outputs:
  qstar_fine_3d.jpg       — fine-resolution 3D surface
  qstar_slices.jpg        — cvar vs offset at fixed Q slices
  qstar_valley_width.jpg  — valley width (10% above min) per Q level
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

Q_LEVELS  = np.arange(2.5, 97.6, 2.5)   # finer: 2.5% steps
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
        fb.append(np.mean(y[bm] < a + b_fit * np.average(lt[bm], weights=w[bm])))
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
    # Fine step around the two features; coarser elsewhere
    offsets_fine  = np.arange(0,      8_001,   25)
    offsets_mid   = np.arange(8_025,  37_501,  50)
    offsets       = np.concatenate([offsets_fine, offsets_mid])
    n_off         = len(offsets)
    nq            = len(Q_LEVELS)

    print(f"Sweeping {n_off} offsets × {nq} Q-levels (log-density), {N_WORKERS} workers…",
          flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, offsets.tolist(), chunksize=16)

    grid = np.full((n_off, nq), float("nan"))
    for off, cvars in raw:
        if cvars is None: continue
        i = int(np.searchsorted(offsets, off))
        for j, q in enumerate(Q_LEVELS):
            grid[i, j] = cvars[q]

    print("Done. Building plots…", flush=True)

    # Clip for display
    vmax = np.nanpercentile(grid, 95)
    grid_c = np.clip(grid, 0, vmax)

    # q*(offset)
    qstar = np.full(n_off, float("nan"))
    for i in range(n_off):
        row = grid[i]
        fin = np.isfinite(row)
        if fin.sum() < 2: continue
        j = np.where(fin, row, np.inf).argmin()
        if 0 < j < nq - 1 and fin[j-1] and fin[j+1]:
            y0, y1, y2 = row[j-1], row[j], row[j+1]
            denom = 2*(y0 - 2*y1 + y2)
            if abs(denom) > 1e-12:
                frac = np.clip((y0-y2)/denom, -0.5, 0.5)
                qstar[i] = Q_LEVELS[j] + frac*(Q_LEVELS[1]-Q_LEVELS[0])
                continue
        qstar[i] = float(Q_LEVELS[j])

    fmt_k = mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x))
    )

    # ── Figure 1: fine 3D surface ─────────────────────────────────────────────
    OFF_G, Q_G = np.meshgrid(offsets, Q_LEVELS, indexing="ij")

    fig1 = plt.figure(figsize=(15, 9))
    ax3d = fig1.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(
        OFF_G, Q_G, grid_c,
        cmap="viridis_r", linewidth=0, antialiased=True, alpha=0.88,
        vmin=0, vmax=vmax,
    )
    valid = np.isfinite(qstar)
    qs_z  = np.array([
        grid_c[i, max(0, min(nq-1, int(np.searchsorted(Q_LEVELS, qstar[i]))))]
        if np.isfinite(qstar[i]) else 0
        for i in range(n_off)
    ])
    ax3d.plot(offsets[valid], qstar[valid], qs_z[valid],
              color="white", lw=2.0, label="q*(offset)")
    for tgt, col in [(25, "#FF5555"), (50, "#55AAFF"), (80, "#FFCC00")]:
        j_t = int(np.searchsorted(Q_LEVELS, tgt))
        ax3d.plot(offsets, np.full(n_off, tgt), grid_c[:, j_t],
                  color=col, lw=1.5, ls="--", alpha=0.85, label=f"Q{tgt}% slice")
    fig1.colorbar(surf, ax=ax3d, shrink=0.45, pad=0.04, label="cvar")
    ax3d.set_xlabel("Block offset", fontsize=9, labelpad=8)
    ax3d.set_ylabel("Quantile (%)", fontsize=9, labelpad=8)
    ax3d.set_zlabel("A2 cvar", fontsize=9, labelpad=6)
    ax3d.set_title(
        "Log-density weighted — fine resolution cvar surface\n"
        "White = q*(offset) ridge  |  Dashed slices at Q25%, Q50%, Q80%",
        fontsize=10,
    )
    ax3d.xaxis.set_major_formatter(fmt_k)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.legend(fontsize=8, loc="upper right")
    fig1.tight_layout()
    out1 = ROOT.parent / "debris" / "research" / "qstar_fine_3d.jpg"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved → {out1}")

    # ── Figure 2: cvar-vs-offset slices ──────────────────────────────────────
    # Two panels: low-Q (main valley) and high-Q (secondary feature)
    low_qs  = [10, 15, 20, 25, 30]
    high_qs = [65, 70, 75, 80, 85, 90]
    cmap_lo = plt.cm.Blues(np.linspace(0.4, 0.9, len(low_qs)))
    cmap_hi = plt.cm.Oranges(np.linspace(0.4, 0.9, len(high_qs)))

    fig2, (ax_lo, ax_hi) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(
        "cvar vs block offset at fixed quantile levels — log-density weighted\n"
        "Shallower / wider minimum = more stable model",
        fontsize=11,
    )

    for ax, qs, cmap, title in [
        (ax_lo, low_qs,  cmap_lo, "Low quantiles (main valley ~Q10–25%)"),
        (ax_hi, high_qs, cmap_hi, "High quantiles (secondary feature ~Q75–85%)"),
    ]:
        for q, color in zip(qs, cmap):
            j = int(np.searchsorted(Q_LEVELS, q))
            col = grid[:, j]
            # Normalise to min=1 so sharpness is comparable across Q levels
            col_min = np.nanmin(col)
            ax.plot(offsets, col / col_min, lw=1.5, color=color, label=f"Q{q}%")
            # mark minimum
            i_min = np.nanargmin(col)
            ax.axvline(offsets[i_min], color=color, lw=0.7, ls=":", alpha=0.6)

        ax.axhline(1.10, color="#888", lw=1.0, ls="--", alpha=0.7,
                   label="10% above min")
        ax.set_xlabel("Block offset", fontsize=9)
        ax.set_ylabel("cvar / cvar_min  (1 = minimum)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(fmt_k)
        ax.set_ylim(0.9, 2.5)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    fig2.tight_layout()
    out2 = ROOT.parent / "debris" / "research" / "qstar_slices.jpg"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")

    # ── Figure 3: valley width across all Q levels ────────────────────────────
    widths    = np.full(nq, float("nan"))
    min_offsets = np.full(nq, float("nan"))
    min_cvars   = np.full(nq, float("nan"))

    for j, q in enumerate(Q_LEVELS):
        col = grid[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 3: continue
        i_min = np.where(fin, col, np.inf).argmin()
        c_min = col[i_min]
        threshold = c_min * 1.10
        # find contiguous region around minimum below threshold
        in_valley = (col <= threshold) & fin
        # find left and right edges of the region containing i_min
        left  = i_min
        right = i_min
        while left  > 0       and in_valley[left  - 1]: left  -= 1
        while right < n_off-1 and in_valley[right + 1]: right += 1
        widths[j]      = offsets[right] - offsets[left]
        min_offsets[j] = offsets[i_min]
        min_cvars[j]   = c_min

    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig3.suptitle(
        "Valley analysis per quantile level — log-density weighted\n"
        "Valley width = offset range where cvar ≤ 1.10 × cvar_min",
        fontsize=11,
    )

    axes3[0].bar(Q_LEVELS, widths, width=2.0, color="#0066FF", alpha=0.7)
    axes3[0].set_ylabel("Valley width (blocks)", fontsize=9)
    axes3[0].set_title("Valley width (wider = more stable)", fontsize=10)
    axes3[0].grid(True, alpha=0.2, axis="y")

    axes3[1].plot(Q_LEVELS, min_offsets, "o-", color="#DD0000", ms=4, lw=1.5)
    axes3[1].set_ylabel("Optimal offset (blocks)", fontsize=9)
    axes3[1].set_title("Optimal offset at each Q level", fontsize=10)
    axes3[1].yaxis.set_major_formatter(fmt_k)
    axes3[1].grid(True, alpha=0.2)

    axes3[2].plot(Q_LEVELS, min_cvars, "o-", color="#009944", ms=4, lw=1.5)
    axes3[2].set_ylabel("cvar at minimum", fontsize=9)
    axes3[2].set_title("Best achievable cvar at each Q level (lower = more consistent)", fontsize=10)
    axes3[2].set_xlabel("Quantile level (%)", fontsize=9)
    axes3[2].grid(True, alpha=0.2)

    # Shade the two feature regions
    for ax in axes3:
        ax.axvspan(5,  35, alpha=0.07, color="#0066FF", label="Main valley region")
        ax.axvspan(65, 90, alpha=0.07, color="#FF7700", label="Secondary feature region")
        ax.legend(fontsize=7, loc="upper right")

    fig3.tight_layout()
    out3 = ROOT.parent / "debris" / "research" / "qstar_valley_width.jpg"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved → {out3}")
