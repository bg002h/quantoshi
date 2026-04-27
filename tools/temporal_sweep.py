#!/usr/bin/env python3
"""
Temporal offset sweep — two approaches to finding optimal block offset:

  Approach 2 (temporal quantile consistency):
    Bin log-time into N_BINS bins, compute Pth percentile of log-price per bin,
    fit WLS to the N bin-centroid (log_t, log_price_pct) pairs.
    Also report per-bin fraction-below variance (the direct consistency metric).

  Approach 1 (time-stratified floor selection):
    Same bins, take the bottom P% of prices per bin (by raw log-price),
    pool all selected points, fit global QR at QP%.
    R² on pooled subset.

Outputs:
  ../debris/research/temporal_sweep.jpg   — R² / consistency vs offset
  ../debris/research/temporal_fits.jpg    — best fits at optimal offsets
"""
from __future__ import annotations

import multiprocessing as mp
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

WEIGHT_MODES  = ["none", "inv_t", "inv_sqrt_t", "log_density"]
WEIGHT_LABELS = {
    "none":        "Unweighted",
    "inv_t":       "1/t weighted",
    "inv_sqrt_t":  "1/√t weighted",
    "log_density": "Log-density weighted",
}

N_BINS    = 10
A2_PCTS   = [10, 25, 50]        # percentiles to track per bin
A1_COMBOS = [(10, 10), (20, 25), (30, 25)]   # (bottom_frac%, fit_q%)

N_WORKERS = 19

# ── Visual encoding ───────────────────────────────────────────────────────────

A2_COLORS = {10: "#0066FF", 25: "#DD0000", 50: "#AA6600"}
A1_COLORS = {(10, 10): "#009944", (20, 25): "#990099", (30, 25): "#FF7700"}

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
    """Weighted least-squares line fit. Returns (intercept, slope, r2)."""
    w = w / w.sum()
    xm = np.dot(w, x)
    ym = np.dot(w, y)
    b  = np.dot(w, (x - xm) * (y - ym)) / np.dot(w, (x - xm) ** 2)
    a  = ym - b * xm
    y_hat  = a + b * x
    ss_res = np.dot(w, (y - y_hat) ** 2)
    ss_tot = np.dot(w, (y - ym) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def _wqr(log_t, y, w, q):
    """Weighted quantile regression. Returns (intercept, slope, r2)."""
    n = len(y)
    nv = 4 + 2 * n
    c = np.zeros(nv)
    c[4:4+n] =      q * w
    c[4+n:]  = (1-q) * w
    A_eq = np.zeros((n, nv))
    A_eq[:, 0] =  1.0;  A_eq[:, 1] = -1.0
    A_eq[:, 2] =  log_t; A_eq[:, 3] = -log_t
    A_eq[:, 4:4+n] = -np.eye(n); A_eq[:, 4+n:] = np.eye(n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=[(0, None)] * nv,
                  method="highs", options={"disp": False})
    if not res.success:
        nan = float("nan")
        return nan, nan, nan
    ap, an, bp, bn = res.x[:4]
    a, b = ap - an, bp - bn
    y_hat  = a + b * log_t
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def _bin_ids(lt, n_bins):
    """Assign each point to one of n_bins equal-width log-time bins."""
    edges = np.linspace(lt.min(), lt.max(), n_bins + 1)
    ids = np.digitize(lt, edges[:-1]) - 1
    return np.clip(ids, 0, n_bins - 1)


def _worker(args):
    off, wt_mode = args
    t = BLK - off
    mask = t > 0
    nan = float("nan")
    if mask.sum() < 30:
        return args, None

    lt    = np.log10(t[mask])
    y     = LOG_P[mask]
    t_pos = t[mask]
    w     = _weights(lt, t_pos, wt_mode)
    bids  = _bin_ids(lt, N_BINS)

    out = {}

    # ── Approach 2: temporal quantile consistency ─────────────────────────────
    for pct in A2_PCTS:
        xs, ys, bin_ws = [], [], []
        frac_below = []

        for b in range(N_BINS):
            bmask = bids == b
            if bmask.sum() < 5:
                continue
            y_b  = y[bmask]
            w_b  = w[bmask]
            xs.append(np.average(lt[bmask], weights=w_b))
            ys.append(np.percentile(y_b, pct))        # unweighted pct within bin
            bin_ws.append(w_b.sum())

        if len(xs) < 3:
            out[("a2", pct)] = (nan, nan, nan)
            continue

        xs, ys, bin_ws = map(np.array, (xs, ys, bin_ws))
        a, b_fit, r2 = _wls_fit(xs, ys, bin_ws)

        # consistency metric: per-bin fraction below the fitted line
        frac_below = []
        for b in range(N_BINS):
            bmask = bids == b
            if bmask.sum() < 5:
                continue
            y_line = a + b_fit * np.average(lt[bmask], weights=w[bmask])
            frac_below.append(np.mean(y[bmask] < y_line))

        consistency_var = float(np.var(frac_below)) if len(frac_below) >= 3 else nan
        out[("a2", pct)] = (b_fit, r2, consistency_var)

    # ── Approach 1: time-stratified floor selection ───────────────────────────
    for (frac_pct, q_pct) in A1_COMBOS:
        selected = []
        for b in range(N_BINS):
            bmask_idx = np.where(bids == b)[0]
            if len(bmask_idx) < 5:
                continue
            n_keep = max(3, int(len(bmask_idx) * frac_pct / 100))
            # take the lowest log-price points in this bin
            order  = np.argsort(y[bmask_idx])
            selected.extend(bmask_idx[order[:n_keep]].tolist())

        if len(selected) < 10:
            out[("a1", frac_pct, q_pct)] = (nan, nan)
            continue

        idx    = np.array(selected)
        lt_sub = lt[idx]; y_sub = y[idx]; w_sub = w[idx]
        w_sub  = w_sub * (len(w_sub) / w_sub.sum())
        _, b_fit, r2 = _wqr(lt_sub, y_sub, w_sub, q=q_pct / 100)
        out[("a1", frac_pct, q_pct)] = (b_fit, r2)

    return args, out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    offsets  = np.arange(0, 37_501, 150)
    n_steps  = len(offsets)
    jobs     = [(off, wt) for wt in WEIGHT_MODES for off in offsets]

    print(f"Sweeping {n_steps} offsets × {len(WEIGHT_MODES)} weight modes "
          f"= {len(jobs)} jobs, {N_WORKERS} workers…", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, jobs, chunksize=8)

    print("Fits done. Assembling results…", flush=True)

    a2_keys = [("a2", p) for p in A2_PCTS]
    a1_keys = [("a1", f, q) for f, q in A1_COMBOS]
    all_keys = a2_keys + a1_keys

    # results[wt][key] = {"b": array, "r2": array, ...}
    results = {
        wt: {k: {"b": np.full(n_steps, float("nan")),
                 "r2": np.full(n_steps, float("nan")),
                 "cvar": np.full(n_steps, float("nan"))}
             for k in all_keys}
        for wt in WEIGHT_MODES
    }

    for (off, wt), fits in raw:
        if fits is None:
            continue
        i = int(np.searchsorted(offsets, off))
        for key, vals in fits.items():
            if key[0] == "a2":
                b_fit, r2, cvar = vals
                results[wt][key]["b"][i]    = b_fit
                results[wt][key]["r2"][i]   = r2
                results[wt][key]["cvar"][i] = cvar
            else:
                b_fit, r2 = vals
                results[wt][key]["b"][i]  = b_fit
                results[wt][key]["r2"][i] = r2

    # ── Figure 1: sweep curves ────────────────────────────────────────────────

    fmt = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    fig, axes = plt.subplots(4, 4, figsize=(24, 22), sharex=True)
    fig.suptitle(
        "Temporal offset sweep\n"
        "Approach 2 (bin percentiles): R² and per-bin fraction-below variance  |  "
        "Approach 1 (stratified floor): R² on pooled subset",
        fontsize=11, y=1.002,
    )

    col_titles = [
        "A2 — R² of WLS to bin percentiles",
        "A2 — Consistency (↓ = more uniform fraction-below)",
        "A1 — R² on stratified floor subset",
        "A1 — Exponent b",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9)

    for row, wt in enumerate(WEIGHT_MODES):
        d = results[wt]

        # Approach 2: R²
        ax = axes[row, 0]
        for pct in A2_PCTS:
            ax.plot(offsets, d[("a2", pct)]["r2"],
                    lw=1.2, color=A2_COLORS[pct], label=f"Q{pct}%")
        ax.set_ylabel(WEIGHT_LABELS[wt], fontsize=9)
        ax.grid(True, alpha=0.2)
        if row == 0: ax.legend(fontsize=8)

        # Approach 2: consistency variance (lower = better)
        ax = axes[row, 1]
        for pct in A2_PCTS:
            ax.plot(offsets, d[("a2", pct)]["cvar"],
                    lw=1.2, color=A2_COLORS[pct], label=f"Q{pct}%")
        ax.grid(True, alpha=0.2)
        if row == 0: ax.legend(fontsize=8)

        # Approach 1: R²
        ax = axes[row, 2]
        for frac_pct, q_pct in A1_COMBOS:
            key = ("a1", frac_pct, q_pct)
            ax.plot(offsets, d[key]["r2"],
                    lw=1.2, color=A1_COLORS[(frac_pct, q_pct)],
                    label=f"bot{frac_pct}%/Q{q_pct}%")
        ax.grid(True, alpha=0.2)
        if row == 0: ax.legend(fontsize=8)

        # Approach 1: exponent b
        ax = axes[row, 3]
        for frac_pct, q_pct in A1_COMBOS:
            key = ("a1", frac_pct, q_pct)
            ax.plot(offsets, d[key]["b"],
                    lw=1.2, color=A1_COLORS[(frac_pct, q_pct)],
                    label=f"bot{frac_pct}%/Q{q_pct}%")
        ax.grid(True, alpha=0.2)
        if row == 0: ax.legend(fontsize=8)

    for ax in axes[3]:
        ax.set_xlabel("Block offset (t₀)", fontsize=9)
        ax.xaxis.set_major_formatter(fmt)

    fig.tight_layout()
    out1 = ROOT.parent / "debris" / "research" / "temporal_sweep.jpg"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved → {out1}")

    # ── Figure 2: best fits at optimal offsets overlaid on data ──────────────

    print("Computing fit visualisations…", flush=True)

    def _best_key_and_offset_a2(wt):
        """Return (pct, offset) with globally lowest cvar for this weight mode."""
        best_cvar = np.inf
        best_pct, best_off = A2_PCTS[0], offsets[0]
        for pct in A2_PCTS:
            arr = results[wt][("a2", pct)]["cvar"]
            valid = np.isfinite(arr)
            if not valid.any():
                continue
            idx = np.where(valid, arr, np.inf).argmin()
            if arr[idx] < best_cvar:
                best_cvar = arr[idx]
                best_pct  = pct
                best_off  = offsets[idx]
        return best_pct, best_off

    def _best_key_and_offset_a1(wt):
        """Return ((frac_pct, q_pct), offset) with globally highest R² for this weight mode."""
        best_r2 = -np.inf
        best_combo, best_off = A1_COMBOS[0], offsets[0]
        for combo in A1_COMBOS:
            arr = results[wt][("a1",) + combo]["r2"]
            valid = np.isfinite(arr)
            if not valid.any():
                continue
            idx = np.where(valid, arr, -np.inf).argmax()
            if arr[idx] > best_r2:
                best_r2   = arr[idx]
                best_combo = combo
                best_off   = offsets[idx]
        return best_combo, best_off

    fig2, axes2 = plt.subplots(4, 2, figsize=(16, 20))
    fig2.suptitle(
        "Best fits at optimal (Q-level, block offset) per weight mode\n"
        "Left: log-log  |  Right: log-linear\n"
        "Blue solid = Approach 2 (best Q-level by min cvar)  |  "
        "Red dashed = Approach 1 (best combo by max R²)",
        fontsize=10, y=1.002,
    )

    fmt2 = mticker.FuncFormatter(lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x)))

    for row, wt in enumerate(WEIGHT_MODES):
        ax_ll  = axes2[row, 0]
        ax_lin = axes2[row, 1]

        t0  = BLK
        lt0 = np.log10(t0)
        ax_ll.scatter(lt0, LOG_P, s=2, c="#999", alpha=0.20, linewidths=0, rasterized=True)
        ax_lin.scatter(t0, LOG_P, s=2, c="#999", alpha=0.20, linewidths=0, rasterized=True)

        lt_grid = np.linspace(lt0.min(), lt0.max(), 600)
        t_grid  = 10 ** lt_grid

        # ── Approach 2: best (Q-level, offset) ───────────────────────────────
        best_pct2, off2 = _best_key_and_offset_a2(wt)
        t2    = BLK - off2
        lt2   = np.log10(t2)
        w2    = _weights(lt2, t2, wt)
        bids2 = _bin_ids(lt2, N_BINS)
        xs2, ys2, bw2 = [], [], []
        for b in range(N_BINS):
            bmask = bids2 == b
            if bmask.sum() < 5: continue
            xs2.append(np.average(lt2[bmask], weights=w2[bmask]))
            ys2.append(np.percentile(LOG_P[bmask], best_pct2))
            bw2.append(w2[bmask].sum())
        xs2, ys2, bw2 = map(np.array, (xs2, ys2, bw2))
        a2, b2, _ = _wls_fit(xs2, ys2, bw2)
        cvar2 = results[wt][("a2", best_pct2)]["cvar"][int(np.searchsorted(offsets, off2))]

        valid2 = t_grid > off2
        y2_ll  = a2 + b2 * np.log10(t_grid[valid2] - off2)
        lbl2   = f"A2 Q{best_pct2}% off={off2:,.0f}  a={a2:.3f} b={b2:.3f}  cvar={cvar2:.4f}"
        ax_ll.plot(lt_grid[valid2],  y2_ll, lw=2.0, color="#0066FF", ls="-",  label=lbl2)
        ax_lin.plot(t_grid[valid2],  y2_ll, lw=2.0, color="#0066FF", ls="-",  label=lbl2)

        # ── Approach 1: best (combo, offset) ─────────────────────────────────
        (frac1, q1), off1 = _best_key_and_offset_a1(wt)
        t1    = BLK - off1
        lt1   = np.log10(t1)
        w1    = _weights(lt1, t1, wt)
        bids1 = _bin_ids(lt1, N_BINS)
        selected = []
        for b in range(N_BINS):
            bidx   = np.where(bids1 == b)[0]
            if len(bidx) < 5: continue
            n_keep = max(3, int(len(bidx) * frac1 / 100))
            order  = np.argsort(LOG_P[bidx])
            selected.extend(bidx[order[:n_keep]].tolist())
        idx    = np.array(selected)
        lt_sub = lt1[idx]; y_sub = LOG_P[idx]; w_sub = w1[idx]
        w_sub  = w_sub * (len(w_sub) / w_sub.sum())
        a1, b1, _ = _wqr(lt_sub, y_sub, w_sub, q=q1 / 100)
        r2_1 = results[wt][("a1", frac1, q1)]["r2"][int(np.searchsorted(offsets, off1))]

        valid1 = t_grid > off1
        y1_ll  = a1 + b1 * np.log10(t_grid[valid1] - off1)
        lbl1   = f"A1 bot{frac1}%/Q{q1}% off={off1:,.0f}  a={a1:.3f} b={b1:.3f}  R²={r2_1:.4f}"
        ax_ll.plot(lt_grid[valid1],  y1_ll, lw=2.0, color="#DD0000", ls="--", label=lbl1)
        ax_lin.plot(t_grid[valid1],  y1_ll, lw=2.0, color="#DD0000", ls="--", label=lbl1)

        ax_ll.set_title(WEIGHT_LABELS[wt], fontsize=10)
        ax_ll.set_ylabel("log₁₀(price)", fontsize=9)
        ax_ll.set_xlabel("log₁₀(block)", fontsize=9)
        ax_ll.grid(True, alpha=0.2)
        ax_ll.legend(fontsize=8)

        ax_lin.set_title(WEIGHT_LABELS[wt], fontsize=10)
        ax_lin.set_ylabel("log₁₀(price)", fontsize=9)
        ax_lin.set_xlabel("block", fontsize=9)
        ax_lin.xaxis.set_major_formatter(fmt2)
        ax_lin.grid(True, alpha=0.2)
        ax_lin.legend(fontsize=8)

    fig2.tight_layout()
    out2 = ROOT.parent / "debris" / "research" / "temporal_fits.jpg"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
