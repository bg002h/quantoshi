#!/usr/bin/env python3
"""
Crossing-symmetry method for finding optimal block offset.

For a given offset, fit WLS lines at many quantile levels.
Symmetric pairs (Q10/Q90, Q25/Q75, etc.) should cross exactly on the Q50%
(median) line if the origin is correctly specified.

Score = RMS deviation (in log-price) of pairwise crossing points from Q50% line.
Optimal offset minimises this score.

Outputs:
  qstar_crossing_score.jpg  — score vs offset, all weight modes
  qstar_crossing_fan.jpg    — fan of quantile lines at offset=0 vs optimal offset
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

WEIGHT_MODES  = ["none", "inv_t", "inv_sqrt_t", "log_density"]
WEIGHT_LABELS = {
    "none":        "Unweighted",
    "inv_t":       "1/t weighted",
    "inv_sqrt_t":  "1/√t weighted",
    "log_density": "Log-density weighted",
}

Q_LEVELS = list(range(5, 96, 5))   # 5,10,...,95
PAIRS    = [(q, 100-q) for q in range(5, 50, 5)]   # (5,95),(10,90),...,(45,55)
N_BINS   = 10
N_WORKERS = 19

# ── Helpers ───────────────────────────────────────────────────────────────────

def _weights(log_t, t_pos, mode):
    n = len(t_pos)
    if mode == "none":         return np.ones(n)
    if mode == "inv_t":        w = 1.0 / t_pos
    elif mode == "inv_sqrt_t": w = 1.0 / np.sqrt(t_pos)
    elif mode == "log_density":
        kde = scipy.stats.gaussian_kde(log_t)
        w = 1.0 / np.maximum(kde(log_t), 1e-9)
    else: return np.ones(n)
    return w * (n / w.sum())


def _wls_fit(x, y, w):
    w = w / w.sum()
    xm = np.dot(w, x); ym = np.dot(w, y)
    b  = np.dot(w, (x - xm) * (y - ym)) / np.dot(w, (x - xm) ** 2)
    return ym - b * xm, b


def _fit_quantile_lines(off, wt_mode):
    """Return {q: (a, b)} for all Q_LEVELS using A2 bin-percentile WLS."""
    t = BLK - off
    mask = t > 0
    if mask.sum() < 30:
        return None
    lt    = np.log10(t[mask])
    y     = LOG_P[mask]
    w     = _weights(lt, t[mask], wt_mode)
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
    """RMS deviation of symmetric-pair crossings from the Q50% line.
    Also returns list of (x_c, y_c, y_50_at_xc, pair) for plotting."""
    if fits is None:
        return float("nan"), []
    a50, b50 = fits.get(50, (np.nan, np.nan))
    if np.isnan(a50):
        return float("nan"), []
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
        x_c = (a_hi - a_lo) / db
        y_c = a_lo + b_lo * x_c
        y_50 = a50 + b50 * x_c
        deviations.append(y_c - y_50)
        details.append((x_c, y_c, y_50, q_lo, q_hi))
    if not deviations:
        return float("nan"), []
    return float(np.sqrt(np.mean(np.array(deviations) ** 2))), details


def _worker(args):
    off, wt_mode = args
    fits = _fit_quantile_lines(off, wt_mode)
    score, _ = _crossing_score(fits)
    return args, score


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    offsets  = np.arange(0, 37_501, 150)
    n_off    = len(offsets)
    jobs     = [(off, wt) for wt in WEIGHT_MODES for off in offsets]

    print(f"Sweeping {n_off} offsets × {len(WEIGHT_MODES)} weight modes, "
          f"{N_WORKERS} workers…", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, jobs, chunksize=8)

    scores = {wt: np.full(n_off, float("nan")) for wt in WEIGHT_MODES}
    for (off, wt), s in raw:
        i = int(np.searchsorted(offsets, off))
        scores[wt][i] = s

    print("\nOptimal offsets (min crossing-symmetry score):")
    print(f"{'Weight mode':<20}  {'Optimal offset':>14}  {'Score':>10}")
    print("-" * 50)
    opt_offsets = {}
    for wt in WEIGHT_MODES:
        arr   = scores[wt]
        valid = np.isfinite(arr)
        i_opt = np.where(valid, arr, np.inf).argmin()
        opt_offsets[wt] = offsets[i_opt]
        print(f"{WEIGHT_LABELS[wt]:<20}  {offsets[i_opt]:>14,.0f}  {arr[i_opt]:>10.6f}")

    fmt_k = mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e3):,}k" if x >= 1000 else str(int(x))
    )

    # ── Figure 1: score vs offset ─────────────────────────────────────────────
    WT_COLORS = ["#222222", "#009944", "#FF7700", "#990099"]

    fig1, ax1 = plt.subplots(figsize=(13, 5))
    for ci, wt in enumerate(WEIGHT_MODES):
        arr = scores[wt]
        ax1.plot(offsets, arr, color=WT_COLORS[ci], lw=1.8,
                 label=WEIGHT_LABELS[wt])
        i_opt = np.where(np.isfinite(arr), arr, np.inf).argmin()
        ax1.axvline(offsets[i_opt], color=WT_COLORS[ci], lw=0.8, ls=":", alpha=0.6)
        ax1.annotate(f"{offsets[i_opt]:,.0f}",
                     xy=(offsets[i_opt], arr[i_opt]),
                     xytext=(offsets[i_opt] + 400, arr[i_opt] * 1.04),
                     fontsize=8, color=WT_COLORS[ci])

    ax1.set_xlabel("Block offset", fontsize=10)
    ax1.set_ylabel("RMS crossing deviation from Q50% (log-price units)", fontsize=10)
    ax1.set_title(
        "Crossing-symmetry score vs block offset\n"
        "Score = RMS distance of symmetric-pair (Q10/Q90, Q25/Q75, …) crossings "
        "from the Q50% line  |  Lower = more symmetric fan",
        fontsize=10,
    )
    ax1.xaxis.set_major_formatter(fmt_k)
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    out1 = ROOT.parent / "debris" / "research" / "qstar_crossing_score.jpg"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out1}")

    # ── Figure 2: fan plots — offset=0 vs optimal, log-density weighted ───────
    wt_plot  = "log_density"
    off_opt  = opt_offsets[wt_plot]
    offsets_show = [0, off_opt]

    # colormap for quantile levels
    q_cmap = plt.cm.coolwarm
    q_norm = plt.Normalize(vmin=5, vmax=95)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig2.suptitle(
        f"Quantile fan — log-density weighted\n"
        f"At the correct origin, symmetric pairs (Q10/Q90, Q25/Q75, …) "
        f"should cross exactly on the Q50% line",
        fontsize=11,
    )

    for ax, off in zip(axes2, offsets_show):
        fits = _fit_quantile_lines(off, wt_plot)
        _, details = _crossing_score(fits)

        t_pos = BLK - off
        mask  = t_pos > 0
        lt    = np.log10(t_pos[mask])

        # scatter data
        ax.scatter(lt, LOG_P[mask], s=2, c="#888", alpha=0.15,
                   linewidths=0, rasterized=True)

        # draw quantile lines
        lt_grid = np.linspace(lt.min(), lt.max(), 400)
        for q in Q_LEVELS:
            a, b = fits[q]
            if np.isnan(a): continue
            color = q_cmap(q_norm(q))
            lw    = 2.5 if q == 50 else 1.0
            zord  = 4   if q == 50 else 2
            ax.plot(lt_grid, a + b * lt_grid, color=color, lw=lw,
                    zorder=zord, alpha=0.85)

        # mark crossing points
        for x_c, y_c, y_50, q_lo, q_hi in details:
            if lt.min() <= x_c <= lt.max() + 1.0:
                ax.scatter([x_c], [y_c], s=60, zorder=5,
                           color=q_cmap(q_norm(q_lo)),
                           edgecolors="white", linewidths=0.5)
                # vertical line from crossing to Q50%
                ax.plot([x_c, x_c], [y_c, y_50], color="#fff",
                        lw=0.6, ls=":", alpha=0.5, zorder=4)

        # Q50% line thicker in a distinct color
        a50, b50 = fits[50]
        ax.plot(lt_grid, a50 + b50 * lt_grid, color="white",
                lw=2.5, zorder=6, label="Q50% (median)")

        score_val = scores[wt_plot][int(np.searchsorted(offsets, off))]
        ax.set_title(
            f"Offset = {off:,.0f} blocks  |  score = {score_val:.5f}",
            fontsize=10,
        )
        ax.set_xlabel("log₁₀(block − offset)", fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=8)

    axes2[0].set_ylabel("log₁₀(price)", fontsize=9)

    # colorbar for quantile levels
    sm = plt.cm.ScalarMappable(cmap=q_cmap, norm=q_norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=axes2, fraction=0.02, pad=0.02)
    cbar.set_label("Quantile level (%)", fontsize=9)

    fig2.tight_layout()
    out2 = ROOT.parent / "debris" / "research" / "qstar_crossing_fan.jpg"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
