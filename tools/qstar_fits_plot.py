#!/usr/bin/env python3
"""
Plot the A2 fits at the optimal offsets for q*=25% and q*=50% targets.
Left: log price vs log(block − offset)
Right: log price vs block (linear x)
One row per weight mode.

Output: ../debris/research/qstar_fits.jpg
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

Q_LEVELS = np.arange(5, 96, 5)
N_BINS   = 10
N_WORKERS = 19
TARGETS  = [25, 50]
TARGET_COLORS = {25: "#DD0000", 50: "#0066FF"}
TARGET_LS     = {25: "--", 50: "-"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _weights(log_t, t_pos, mode):
    n = len(t_pos)
    if mode == "none":         return np.ones(n)
    if mode == "inv_t":        w = 1.0 / t_pos
    elif mode == "inv_sqrt_t": w = 1.0 / np.sqrt(t_pos)
    elif mode == "log_density":
        kde = scipy.stats.gaussian_kde(log_t)
        w = 1.0 / np.maximum(kde(log_t), 1e-9)
    else:                      return np.ones(n)
    return w * (n / w.sum())


def _wls_fit(x, y, w):
    w = w / w.sum()
    xm = np.dot(w, x);  ym = np.dot(w, y)
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
        return float("nan"), float("nan"), float("nan")
    xs, ys, bws = map(np.array, (xs, ys, bws))
    a, b_fit = _wls_fit(xs, ys, bws)
    fb = []
    for b in range(N_BINS):
        bm = bids == b
        if bm.sum() < 5: continue
        y_line = a + b_fit * np.average(lt[bm], weights=w[bm])
        fb.append(np.mean(y[bm] < y_line))
    cvar = float(np.var(fb)) if len(fb) >= 3 else float("nan")
    return a, b_fit, cvar


def _worker(args):
    off, wt_mode = args
    t = BLK - off
    mask = t > 0
    if mask.sum() < 30:
        return args, None
    lt    = np.log10(t[mask])
    y     = LOG_P[mask]
    w     = _weights(lt, t[mask], wt_mode)
    edges = np.linspace(lt.min(), lt.max(), N_BINS + 1)
    bids  = np.clip(np.digitize(lt, edges[:-1]) - 1, 0, N_BINS - 1)
    return args, {q: _a2_cvar(lt, y, w, bids, q) for q in Q_LEVELS}


# ── Sweep q*(offset) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    offsets = np.arange(0, 37_501, 150)
    n_off   = len(offsets)
    jobs    = [(off, wt) for wt in WEIGHT_MODES for off in offsets]

    print(f"Sweeping {n_off} offsets × {len(WEIGHT_MODES)} modes × "
          f"{len(Q_LEVELS)} Q-levels…", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, jobs, chunksize=8)

    # Assemble cvar grids
    nq = len(Q_LEVELS)
    cvar_grid = {wt: np.full((n_off, nq), float("nan")) for wt in WEIGHT_MODES}
    a_grid    = {wt: np.full((n_off, nq), float("nan")) for wt in WEIGHT_MODES}
    b_grid    = {wt: np.full((n_off, nq), float("nan")) for wt in WEIGHT_MODES}

    for (off, wt), fits in raw:
        if fits is None: continue
        i = int(np.searchsorted(offsets, off))
        for j, q in enumerate(Q_LEVELS):
            a_val, b_val, cvar_val = fits[q]
            a_grid[wt][i, j]    = a_val
            b_grid[wt][i, j]    = b_val
            cvar_grid[wt][i, j] = cvar_val

    # q*(offset) with parabolic sub-bin interpolation
    def _qstar_arr(grid):
        out = np.full(n_off, float("nan"))
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
                    out[i] = Q_LEVELS[j] + frac * (Q_LEVELS[1] - Q_LEVELS[0])
                    continue
            out[i] = float(Q_LEVELS[j])
        return out

    qstar = {wt: _qstar_arr(cvar_grid[wt]) for wt in WEIGHT_MODES}

    # Find crossing offset (linear interpolation)
    def _crossing_offset(qs_arr, target):
        diff = qs_arr - target
        for i in range(len(diff) - 1):
            if not (np.isfinite(diff[i]) and np.isfinite(diff[i+1])): continue
            if np.sign(diff[i]) != np.sign(diff[i+1]) and diff[i] != 0:
                frac = -diff[i] / (diff[i+1] - diff[i])
                return offsets[i] + frac * (offsets[i+1] - offsets[i])
        return float("nan")

    # For each (wt, target): find crossing offset, then fit directly at that offset
    def _fit_at_offset(wt, target, cross_off):
        """Fit A2 at the exact (interpolated) crossing offset."""
        t = BLK - cross_off
        mask = t > 0
        lt   = np.log10(t[mask])
        y    = LOG_P[mask]
        w    = _weights(lt, t[mask], wt)
        edges = np.linspace(lt.min(), lt.max(), N_BINS + 1)
        bids  = np.clip(np.digitize(lt, edges[:-1]) - 1, 0, N_BINS - 1)
        a, b, cvar = _a2_cvar(lt, y, w, bids, target)
        return a, b, cvar, cross_off

    print("\nOptimal fits:")
    print(f"{'Weight mode':<16}  {'Target':>6}  {'Offset':>8}  {'a':>10}  {'b':>8}  {'cvar':>10}")
    print("-" * 65)
    fits_out = {}
    for wt in WEIGHT_MODES:
        fits_out[wt] = {}
        for tgt in TARGETS:
            cross = _crossing_offset(qstar[wt], tgt)
            a, b, cvar, actual_off = _fit_at_offset(wt, tgt, cross)
            fits_out[wt][tgt] = (a, b, cvar, actual_off)
            print(f"{WEIGHT_LABELS[wt]:<16}  Q{tgt:>2}%  {actual_off:>8,.0f}  {a:>10.4f}  {b:>8.4f}  {cvar:>10.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────

    t0   = BLK
    lt0  = np.log10(t0)
    t_grid  = np.linspace(t0.min(), t0.max(), 800)
    lt_grid = np.log10(t_grid)

    fmt_block = mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x))
    )

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(
        "A2 fits at optimal block offset (q*-crossing method)\n"
        "Red dashed = q*=25% target (bear-market model)  |  "
        "Blue solid = q*=50% target (full-distribution model)",
        fontsize=12, y=1.002,
    )

    for row, wt in enumerate(WEIGHT_MODES):
        ax_ll  = axes[row, 0]
        ax_lin = axes[row, 1]

        # scatter
        ax_ll.scatter(lt0, LOG_P, s=2, c="#888", alpha=0.18, linewidths=0, rasterized=True)
        ax_lin.scatter(t0,  LOG_P, s=2, c="#888", alpha=0.18, linewidths=0, rasterized=True)

        for tgt in TARGETS:
            a, b, cvar, off = fits_out[wt][tgt]
            color = TARGET_COLORS[tgt]
            ls    = TARGET_LS[tgt]
            lbl   = f"Q{tgt}% target  off={off:,.0f}  a={a:.4f}  b={b:.4f}"

            valid = t_grid > off
            y_line = a + b * np.log10(t_grid[valid] - off)

            ax_ll.plot(lt_grid[valid],  y_line, lw=2.0, color=color, ls=ls, label=lbl)
            ax_lin.plot(t_grid[valid],  y_line, lw=2.0, color=color, ls=ls, label=lbl)

        ax_ll.set_title(f"{WEIGHT_LABELS[wt]} — log-log", fontsize=10)
        ax_ll.set_ylabel("log₁₀(price)", fontsize=9)
        ax_ll.set_xlabel("log₁₀(block)", fontsize=9)
        ax_ll.grid(True, alpha=0.2)
        ax_ll.legend(fontsize=8, loc="upper left")

        ax_lin.set_title(f"{WEIGHT_LABELS[wt]} — log-linear", fontsize=10)
        ax_lin.set_ylabel("log₁₀(price)", fontsize=9)
        ax_lin.set_xlabel("block", fontsize=9)
        ax_lin.xaxis.set_major_formatter(fmt_block)
        ax_lin.grid(True, alpha=0.2)
        ax_lin.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = ROOT.parent / "debris" / "research" / "qstar_fits.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
