#!/usr/bin/env python3
"""
For each block offset, find q*(offset) = the quantile level that minimises
A2 temporal-consistency variance (most uniformly-distributed-in-time fit).

Two target crossings:
  q* = 50%  →  optimal offset assuming full distribution follows a power law
  q* = 25%  →  optimal offset assuming only bear markets are robustly predictable

Outputs:
  ../debris/research/qstar_sweep.jpg   — q*(offset) curves + cvar heatmaps
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

# ── Data ──────────────────────────────────────────────────────────────────────

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

WEIGHT_MODES  = ["none", "inv_t", "inv_sqrt_t", "log_density"]
WEIGHT_LABELS = {
    "none":        "Unweighted",
    "inv_t":       "1/t weighted",
    "inv_sqrt_t":  "1/√t weighted",
    "log_density": "Log-density weighted",
}

Q_LEVELS  = np.arange(5, 96, 5)    # 5, 10, 15, ..., 95
N_BINS    = 10
N_WORKERS = 19

# ── Helpers ───────────────────────────────────────────────────────────────────

def _weights(log_t, t_pos, mode):
    n = len(t_pos)
    if mode == "none":
        return np.ones(n)
    if mode == "inv_t":
        w = 1.0 / t_pos
    elif mode == "inv_sqrt_t":
        w = 1.0 / np.sqrt(t_pos)
    elif mode == "log_density":
        kde = scipy.stats.gaussian_kde(log_t)
        w = 1.0 / np.maximum(kde(log_t), 1e-9)
    else:
        return np.ones(n)
    return w * (n / w.sum())


def _wls_fit(x, y, w):
    w = w / w.sum()
    xm = np.dot(w, x);  ym = np.dot(w, y)
    b  = np.dot(w, (x - xm) * (y - ym)) / np.dot(w, (x - xm) ** 2)
    a  = ym - b * xm
    return a, b


def _a2_cvar(lt, y, w, bids, pct):
    """A2 consistency variance for one quantile level."""
    xs, ys, bws = [], [], []
    for b in range(N_BINS):
        bmask = bids == b
        if bmask.sum() < 5:
            continue
        xs.append(np.average(lt[bmask], weights=w[bmask]))
        ys.append(np.percentile(y[bmask], pct))
        bws.append(w[bmask].sum())
    if len(xs) < 3:
        return float("nan")
    xs, ys, bws = map(np.array, (xs, ys, bws))
    a, b_fit = _wls_fit(xs, ys, bws)
    frac_below = []
    for b in range(N_BINS):
        bmask = bids == b
        if bmask.sum() < 5:
            continue
        y_line = a + b_fit * np.average(lt[bmask], weights=w[bmask])
        frac_below.append(np.mean(y[bmask] < y_line))
    return float(np.var(frac_below)) if len(frac_below) >= 3 else float("nan")


def _worker(args):
    off, wt_mode = args
    t = BLK - off
    mask = t > 0
    if mask.sum() < 30:
        return args, None
    lt    = np.log10(t[mask])
    y     = LOG_P[mask]
    t_pos = t[mask]
    w     = _weights(lt, t_pos, wt_mode)
    edges = np.linspace(lt.min(), lt.max(), N_BINS + 1)
    bids  = np.clip(np.digitize(lt, edges[:-1]) - 1, 0, N_BINS - 1)
    return args, {q: _a2_cvar(lt, y, w, bids, q) for q in Q_LEVELS}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    offsets = np.arange(0, 37_501, 150)
    n_off   = len(offsets)
    jobs    = [(off, wt) for wt in WEIGHT_MODES for off in offsets]

    print(f"Sweeping {n_off} offsets × {len(WEIGHT_MODES)} weight modes × "
          f"{len(Q_LEVELS)} Q-levels = {len(jobs)} jobs, {N_WORKERS} workers…",
          flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, jobs, chunksize=8)

    print("Done. Building results…", flush=True)

    # cvar_grid[wt] = (n_off × n_q) array
    nq = len(Q_LEVELS)
    cvar_grid = {
        wt: np.full((n_off, nq), float("nan"))
        for wt in WEIGHT_MODES
    }
    for (off, wt), cvars in raw:
        if cvars is None:
            continue
        i = int(np.searchsorted(offsets, off))
        for j, q in enumerate(Q_LEVELS):
            cvar_grid[wt][i, j] = cvars[q]

    # q*(offset): quantile that minimises cvar at each offset
    def _qstar(grid_row):
        """Interpolated argmin over Q_LEVELS for one offset."""
        finite = np.isfinite(grid_row)
        if finite.sum() < 2:
            return float("nan")
        j = np.where(finite, grid_row, np.inf).argmin()
        # Parabolic interpolation for sub-bin precision
        if 0 < j < nq - 1 and finite[j-1] and finite[j+1]:
            y0, y1, y2 = grid_row[j-1], grid_row[j], grid_row[j+1]
            denom = 2 * (y0 - 2*y1 + y2)
            if abs(denom) > 1e-12:
                frac = (y0 - y2) / denom
                frac = np.clip(frac, -0.5, 0.5)
                return Q_LEVELS[j] + frac * (Q_LEVELS[1] - Q_LEVELS[0])
        return float(Q_LEVELS[j])

    qstar = {
        wt: np.array([_qstar(cvar_grid[wt][i]) for i in range(n_off)])
        for wt in WEIGHT_MODES
    }

    # Find crossing offset for a target quantile by linear interpolation
    def _crossing(qs_arr, target):
        """Return all offsets where q*(offset) crosses target (interpolated)."""
        diff  = qs_arr - target
        signs = np.sign(diff)
        crossings = []
        for i in range(len(diff) - 1):
            if not (np.isfinite(diff[i]) and np.isfinite(diff[i+1])):
                continue
            if signs[i] != signs[i+1] and signs[i] != 0:
                # linear interpolation
                frac = -diff[i] / (diff[i+1] - diff[i])
                crossings.append(offsets[i] + frac * (offsets[i+1] - offsets[i]))
        return crossings

    # ── Report crossings ──────────────────────────────────────────────────────
    targets = [25, 50]
    print(f"\n{'Weight mode':<16} {'Target q*':>10}  {'Crossing offsets'}")
    print("-" * 60)
    for wt in WEIGHT_MODES:
        for tgt in targets:
            crossings = _crossing(qstar[wt], tgt)
            c_str = ", ".join(f"{c:,.0f}" for c in crossings) if crossings else "none"
            print(f"{WEIGHT_LABELS[wt]:<16}  Q{tgt:>2}%       {c_str}")
        print()

    # ── Plot ──────────────────────────────────────────────────────────────────

    TARGET_COLORS = {25: "#DD0000", 50: "#0066FF"}
    WT_COLORS     = ["#222222", "#009944", "#FF7700", "#990099"]

    fmt = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(
        "Optimal block offset via temporal quantile consistency (A2)\n"
        "q*(offset) = quantile level with minimum per-bin fraction-below variance\n"
        "Red dashed = Q25% target (bear-market model)  |  Blue dashed = Q50% target (full-distribution model)",
        fontsize=11, y=1.01,
    )

    # Row 0: cvar heatmaps
    for col, wt in enumerate(WEIGHT_MODES):
        ax  = axes[0, col]
        grid = cvar_grid[wt].T   # shape: (nq, n_off) — q on y, offset on x
        vmax = np.nanpercentile(grid, 90)
        im   = ax.imshow(
            grid,
            aspect="auto", origin="lower",
            extent=[offsets[0], offsets[-1], Q_LEVELS[0], Q_LEVELS[-1]],
            vmin=0, vmax=vmax,
            cmap="viridis_r",
        )
        # overlay q* curve
        ax.plot(offsets, qstar[wt], color="white", lw=1.5, label="q*(offset)")
        for tgt in targets:
            ax.axhline(tgt, color=TARGET_COLORS[tgt], lw=1.2, ls="--", alpha=0.8)
        ax.set_title(f"{WEIGHT_LABELS[wt]}\ncvar heatmap (darker = more consistent)", fontsize=9)
        ax.set_xlabel("Block offset", fontsize=8)
        ax.set_ylabel("Quantile level (%)", fontsize=8)
        ax.xaxis.set_major_formatter(fmt)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.legend(fontsize=7, loc="upper right")

    # Row 1: q*(offset) curves, all weight modes overlaid, per target
    ax_all = axes[1, 0]
    for ci, wt in enumerate(WEIGHT_MODES):
        ax_all.plot(offsets, qstar[wt], color=WT_COLORS[ci],
                    lw=1.5, label=WEIGHT_LABELS[wt])
    for tgt in targets:
        ax_all.axhline(tgt, color=TARGET_COLORS[tgt], lw=1.5, ls="--",
                       label=f"Target Q{tgt}%")
    ax_all.set_title("q*(offset) — all weight modes", fontsize=9)
    ax_all.set_xlabel("Block offset", fontsize=8)
    ax_all.set_ylabel("q* (%)", fontsize=8)
    ax_all.xaxis.set_major_formatter(fmt)
    ax_all.grid(True, alpha=0.2)
    ax_all.legend(fontsize=8)

    # One panel per weight mode showing q*(offset) with crossing annotations
    for col, wt in enumerate(WEIGHT_MODES[1:], start=1):  # use cols 1-3 for individual modes
        ax = axes[1, col]
        ax.plot(offsets, qstar[wt], color=WT_COLORS[col], lw=1.8)
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{WEIGHT_LABELS[wt]}\nq*(offset)", fontsize=9)
        ax.set_xlabel("Block offset", fontsize=8)
        ax.set_ylabel("q* (%)", fontsize=8)
        ax.xaxis.set_major_formatter(fmt)
        for tgt in targets:
            ax.axhline(tgt, color=TARGET_COLORS[tgt], lw=1.5, ls="--")
            for c in _crossing(qstar[wt], tgt):
                ax.axvline(c, color=TARGET_COLORS[tgt], lw=1.0, ls=":", alpha=0.7)
                ax.annotate(
                    f"{c:,.0f}",
                    xy=(c, tgt), xytext=(c + 500, tgt + 3),
                    fontsize=7, color=TARGET_COLORS[tgt],
                    arrowprops=dict(arrowstyle="-", color=TARGET_COLORS[tgt], lw=0.7),
                )

    fig.tight_layout()
    out = ROOT.parent / "debris" / "research" / "qstar_sweep.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
