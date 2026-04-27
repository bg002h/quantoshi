#!/usr/bin/env python3
"""Block offset sweep: BM floor model R² and exponent for
2 bottom-fractions × 2 fit-quantiles × 4 weight modes,
plus band model (±5 pct band around Nth percentile) for N=5,10,25.

Output: ../debris/research/blocksweep.jpg  (8-panel, 4 rows × 2 cols)
"""
from __future__ import annotations

import multiprocessing as mp
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import scipy.stats
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent

# ── Load + merge ──────────────────────────────────────────────────────────────

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

FRACS     = [5, 20]   # bottom % of data to keep (by Q[N]%-ile QR residuals)
QUANTILES = [5, 20]   # quantile % to fit within that subset

BAND_NS = [5, 10, 25]   # band model: fit QN% to ±5-pct band around QN% line

N_WORKERS = 19

# ── Visual encoding ───────────────────────────────────────────────────────────

FRAC_COLORS = {
     5: "#0066FF",   # bright blue
    20: "#DD0000",   # red
}
Q_STYLES = {
     5: "-",    # solid
    20: "--",   # dashed
}

BAND_COLORS = {
     5: "#009944",   # green
    10: "#FF7700",   # orange
    25: "#990099",   # purple
}
BAND_STYLE = ":"   # dotted — visually distinct from BM floor traces

# ── Helpers ───────────────────────────────────────────────────────────────────

def _weights(log_t: np.ndarray, t_pos: np.ndarray, mode: str) -> np.ndarray:
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


_N_APPROX = 800   # points used for fast reference QR (doubled from 400)


def _wqr_fast(log_t: np.ndarray, y: np.ndarray, w: np.ndarray,
              q: float) -> tuple[float, float]:
    """Stratified-subsample QR for slope/intercept only (reference fit).
    Fits on ~_N_APPROX evenly-spaced points along the log_t axis, then
    returns (a, b) to compute full-data residuals for subset selection."""
    n = len(y)
    if n <= _N_APPROX:
        a, b, _ = _wqr(log_t, y, w, q)
        return a, b
    order = np.argsort(log_t)
    idx   = np.round(np.linspace(0, n - 1, _N_APPROX)).astype(int)
    idx   = order[np.unique(idx)]
    w_s   = w[idx]; w_s = w_s * (len(w_s) / w_s.sum())
    a, b, _ = _wqr(log_t[idx], y[idx], w_s, q)
    return a, b


def _wqr(log_t: np.ndarray, y: np.ndarray, w: np.ndarray,
         q: float) -> tuple[float, float, float]:
    """Weighted quantile regression at quantile q ∈ (0,1).
    Returns (intercept, slope, r2-on-this-dataset)."""
    n = len(y)
    nv = 4 + 2 * n
    c = np.zeros(nv)
    c[4:4+n] =      q * w   # cost of over-prediction residuals
    c[4+n:]  = (1-q) * w   # cost of under-prediction residuals
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
        nan = float("nan")
        return nan, nan, nan
    ap, an, bp, bn = res.x[:4]
    a, b = ap - an, bp - bn
    y_hat = a + b * log_t
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def _worker(args: tuple) -> tuple:
    """For one (offset, weight_mode), compute BM floor fits + band model fits."""
    off, wt_mode = args
    t = BLK - off
    mask = t > 0
    nan = float("nan")
    all_keys = list(product(FRACS, QUANTILES)) + [("band", n) for n in BAND_NS]
    if mask.sum() < 30:
        return args, {k: (nan, nan) for k in all_keys}

    lt    = np.log10(t[mask])
    y     = LOG_P[mask]
    t_pos = t[mask]
    w     = _weights(lt, t_pos, wt_mode)

    out = {}

    # ── BM floor model ────────────────────────────────────────────────────────
    for frac_pct in FRACS:
        a_ref, b_ref = _wqr_fast(lt, y, w, q=frac_pct / 100)
        resid = y - (a_ref + b_ref * lt)
        sorted_idx = np.argsort(resid)   # ascending: most bear-market first

        n_keep = max(10, int(len(y) * frac_pct / 100))
        idx    = sorted_idx[:n_keep]
        lt_sub = lt[idx];  y_sub = y[idx];  w_sub = w[idx]
        w_sub  = w_sub * (len(w_sub) / w_sub.sum())

        for q_pct in QUANTILES:
            _, b, r2 = _wqr(lt_sub, y_sub, w_sub, q=q_pct / 100)
            out[(frac_pct, q_pct)] = (b, r2)

    # ── Band model: ±5 pct band around QN% line ───────────────────────────────
    for n_pct in BAND_NS:
        a_ref, b_ref = _wqr_fast(lt, y, w, q=n_pct / 100)
        resid = y - (a_ref + b_ref * lt)
        lo = np.percentile(resid, max(0, n_pct - 5))
        hi = np.percentile(resid, min(100, n_pct + 5))
        band_mask = (resid >= lo) & (resid <= hi)
        if band_mask.sum() < 10:
            out[("band", n_pct)] = (nan, nan)
            continue
        lt_sub = lt[band_mask]; y_sub = y[band_mask]; w_sub = w[band_mask]
        w_sub  = w_sub * (len(w_sub) / w_sub.sum())
        _, b, r2 = _wqr(lt_sub, y_sub, w_sub, q=n_pct / 100)
        out[("band", n_pct)] = (b, r2)

    return args, out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    offsets = np.arange(0, 37_501, 150)
    n_steps = len(offsets)
    jobs = [(off, wt) for wt in WEIGHT_MODES for off in offsets]

    print(f"Sweeping {n_steps} offsets × {len(WEIGHT_MODES)} weight modes "
          f"= {len(jobs)} jobs, {N_WORKERS} workers…",
          flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        raw = pool.map(_worker, jobs, chunksize=8)

    print("Fits done. Building results…", flush=True)
    csv_rows = []

    combo_keys = list(product(FRACS, QUANTILES)) + [("band", n) for n in BAND_NS]
    results: dict = {
        wt: {k: {"b":  np.full(n_steps, float("nan")),
                  "r2": np.full(n_steps, float("nan"))}
             for k in combo_keys}
        for wt in WEIGHT_MODES
    }
    for (off, wt), fits in raw:
        i = int(np.searchsorted(offsets, off))
        for key, (b, r2) in fits.items():
            results[wt][key]["b"][i]  = b
            results[wt][key]["r2"][i] = r2
            frac_pct, q_pct = key
            csv_rows.append({"offset": int(off), "weight": wt,
                              "frac_pct": frac_pct, "q_pct": q_pct,
                              "r2": r2, "exponent": b})

    csv_out = ROOT.parent / "debris" / "research" / "blocksweep_qr_residuals.csv"
    pd.DataFrame(csv_rows).sort_values(
        ["weight", "frac_pct", "q_pct", "offset"]
    ).to_csv(csv_out, index=False, float_format="%.8f")
    print(f"CSV saved → {csv_out}")

    # ── Plot ──────────────────────────────────────────────────────────────────

    fmt = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    fig, axes = plt.subplots(4, 2, figsize=(18, 22), sharex=True)
    fig.suptitle(
        "BM floor model: R² and exponent vs block time origin\n"
        "log₁₀(price) ~ a + b·log₁₀(block − offset)\n"
        "BM floor (solid/dashed): color = bottom fraction, style = fit quantile  |  "
        "Band model (dotted): fit QN% to ±5-pct band around QN% line",
        fontsize=11, y=1.002,
    )

    R2_NOTE = ("R² computed on bear-market subset only —\n"
               "does not attempt to fit bubble prices")

    for row, wt in enumerate(WEIGHT_MODES):
        ax_r2  = axes[row, 0]
        ax_exp = axes[row, 1]
        d = results[wt]

        # BM floor traces
        for frac_pct in FRACS:
            for q_pct in QUANTILES:
                key   = (frac_pct, q_pct)
                color = FRAC_COLORS[frac_pct]
                ls    = Q_STYLES[q_pct]
                ax_r2.plot(offsets, d[key]["r2"],
                           lw=1.2, color=color, ls=ls, alpha=0.85)
                ax_exp.plot(offsets, d[key]["b"],
                            lw=1.2, color=color, ls=ls, alpha=0.85)

        # Band model traces
        for n_pct in BAND_NS:
            key   = ("band", n_pct)
            color = BAND_COLORS[n_pct]
            ax_r2.plot(offsets, d[key]["r2"],
                       lw=1.5, color=color, ls=BAND_STYLE, alpha=0.9)
            ax_exp.plot(offsets, d[key]["b"],
                        lw=1.5, color=color, ls=BAND_STYLE, alpha=0.9)

        ax_r2.set_ylabel("R²", fontsize=10)
        ax_r2.set_title(f"{WEIGHT_LABELS[wt]} — R²", fontsize=10)
        ax_r2.grid(True, alpha=0.25)
        ax_r2.text(0.97, 0.05, R2_NOTE, transform=ax_r2.transAxes,
                   fontsize=7, color="#555", ha="right", va="bottom",
                   style="italic",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax_exp.set_ylabel("Exponent b", fontsize=10)
        ax_exp.set_title(f"{WEIGHT_LABELS[wt]} — Exponent", fontsize=10)
        ax_exp.grid(True, alpha=0.25)

    for ax in axes[3]:
        ax.set_xlabel("Block offset (t₀)", fontsize=10)
        ax.xaxis.set_major_formatter(fmt)

    # ── Legend ────────────────────────────────────────────────────────────────
    frac_handles = [
        mlines.Line2D([], [], color=FRAC_COLORS[f], lw=2, ls="-",
                      label=f"BM floor bottom {f}%")
        for f in FRACS
    ]
    q_handles = [
        mlines.Line2D([], [], color="#444", lw=2, ls=Q_STYLES[q],
                      label=f"Q{q}% fit")
        for q in QUANTILES
    ]
    band_handles = [
        mlines.Line2D([], [], color=BAND_COLORS[n], lw=2, ls=BAND_STYLE,
                      label=f"Band Q{n}% ±5pct")
        for n in BAND_NS
    ]
    fig.legend(handles=frac_handles + q_handles,
               title="BM floor model", title_fontsize=9,
               fontsize=8, loc="lower left",
               bbox_to_anchor=(0.01, -0.05), ncol=4, framealpha=0.9)
    fig.legend(handles=band_handles,
               title="Band model (dotted)", title_fontsize=9,
               fontsize=8, loc="lower right",
               bbox_to_anchor=(0.99, -0.05), ncol=3, framealpha=0.9)

    fig.tight_layout()
    out = ROOT.parent / "debris" / "research" / "blocksweep.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
