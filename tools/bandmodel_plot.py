#!/usr/bin/env python3
"""Plot band model fits overlaid on actual data.
Left column: log₁₀(price) vs log₁₀(block − offset)  [log-log]
Right column: log₁₀(price) vs (block − offset)        [log-linear]
One row per weight mode.  Offset fixed at 0.

Output: ../debris/research/bandmodel_fits.jpg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import scipy.stats
from scipy.optimize import linprog

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

OFFSET = 18_750   # block origin

WEIGHT_MODES  = ["none", "inv_t", "inv_sqrt_t", "log_density"]
WEIGHT_LABELS = {
    "none":        "Unweighted",
    "inv_t":       "1/t weighted",
    "inv_sqrt_t":  "1/√t weighted",
    "log_density": "Log-density weighted",
}

BAND_NS = [5, 10, 25]
BAND_COLORS = {
     5: "#009944",
    10: "#FF7700",
    25: "#990099",
}

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


def _wqr(log_t, y, w, q):
    n = len(y)
    nv = 4 + 2 * n
    c = np.zeros(nv)
    c[4:4+n] =      q * w
    c[4+n:]  = (1-q) * w
    A_eq = np.zeros((n, nv))
    A_eq[:, 0] =  1.0
    A_eq[:, 1] = -1.0
    A_eq[:, 2] =  log_t
    A_eq[:, 3] = -log_t
    A_eq[:, 4:4+n] = -np.eye(n)
    A_eq[:, 4+n:]  =  np.eye(n)
    res = linprog(c, A_eq=A_eq, b_eq=y,
                  bounds=[(0, None)] * nv,
                  method="highs", options={"disp": False})
    if not res.success:
        return float("nan"), float("nan")
    ap, an, bp, bn = res.x[:4]
    return ap - an, bp - bn


# ── Fit band models for each weight mode ─────────────────────────────────────

t = BLK - OFFSET
lt = np.log10(t)
fits = {}   # fits[wt][n_pct] = (a, b, band_mask)

for wt in WEIGHT_MODES:
    fits[wt] = {}
    w = _weights(lt, t, wt)
    for n_pct in BAND_NS:
        a_ref, b_ref = _wqr(lt, LOG_P, w, q=n_pct / 100)
        resid = LOG_P - (a_ref + b_ref * lt)
        lo = np.percentile(resid, max(0, n_pct - 5))
        hi = np.percentile(resid, min(100, n_pct + 5))
        band_mask = (resid >= lo) & (resid <= hi)
        lt_sub = lt[band_mask]; y_sub = LOG_P[band_mask]; w_sub = w[band_mask]
        w_sub = w_sub * (len(w_sub) / w_sub.sum())
        a, b = _wqr(lt_sub, y_sub, w_sub, q=n_pct / 100)
        fits[wt][n_pct] = (a, b, band_mask)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle(
    f"Band model fits  (offset = {OFFSET:,} blocks)\n"
    "Fit QN% to ±5-pct residual band around QN% reference line",
    fontsize=12, y=1.002,
)

# x-axis grids for line drawing
lt_grid = np.linspace(lt.min(), lt.max(), 500)
t_grid  = 10 ** lt_grid   # linear block scale

fmt_block = mticker.FuncFormatter(lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x)))

for row, wt in enumerate(WEIGHT_MODES):
    ax_ll  = axes[row, 0]   # log-log
    ax_lin = axes[row, 1]   # log-linear

    # ── scatter: all data ─────────────────────────────────────────────────────
    ax_ll.scatter(lt,     LOG_P, s=2, c="#888", alpha=0.25, linewidths=0, rasterized=True)
    ax_lin.scatter(t,     LOG_P, s=2, c="#888", alpha=0.25, linewidths=0, rasterized=True)

    # ── band model lines ──────────────────────────────────────────────────────
    for n_pct in BAND_NS:
        a, b, band_mask = fits[wt][n_pct]
        color = BAND_COLORS[n_pct]
        y_line = a + b * lt_grid

        ax_ll.plot(lt_grid, y_line, lw=1.8, color=color, label=f"Q{n_pct}%")
        ax_lin.plot(t_grid, y_line, lw=1.8, color=color, label=f"Q{n_pct}%")

        # highlight the band subset used for fitting
        ax_ll.scatter(lt[band_mask], LOG_P[band_mask],
                      s=4, c=color, alpha=0.45, linewidths=0, rasterized=True)
        ax_lin.scatter(t[band_mask], LOG_P[band_mask],
                       s=4, c=color, alpha=0.45, linewidths=0, rasterized=True)

    # ── cosmetics ─────────────────────────────────────────────────────────────
    ax_ll.set_ylabel("log₁₀(price)", fontsize=9)
    ax_ll.set_xlabel("log₁₀(block)", fontsize=9)
    ax_ll.set_title(f"{WEIGHT_LABELS[wt]} — log-log", fontsize=10)
    ax_ll.grid(True, alpha=0.2)
    ax_ll.legend(fontsize=8, loc="upper left")

    ax_lin.set_ylabel("log₁₀(price)", fontsize=9)
    ax_lin.set_xlabel("block", fontsize=9)
    ax_lin.set_title(f"{WEIGHT_LABELS[wt]} — log-linear", fontsize=10)
    ax_lin.xaxis.set_major_formatter(fmt_block)
    ax_lin.grid(True, alpha=0.2)
    ax_lin.legend(fontsize=8, loc="upper left")

fig.tight_layout()
out = ROOT.parent / "debris" / "research" / f"bandmodel_fits_offset{OFFSET}.jpg"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
