#!/usr/bin/env python3
"""Sensitivity analysis: sweep support line (percentile, quantile) through full bubble model pipeline."""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = "/scratch/code/bitcoinprojections"
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
from model_toolkit.composite import build_composite

warnings.filterwarnings("ignore")

# ── Grid ─────────────────────────────────────────────────────────────────────
PERCENTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
QUANTILES   = [0.05, 0.125, 0.25, 0.50, 0.75, 0.875, 0.95]
DEFAULT_P, DEFAULT_Q = 0.20, 0.50

GENESIS_STR = "2009-07-25"
import pandas as pd
GENESIS = pd.Timestamp(GENESIS_STR)

# ── Load data once ───────────────────────────────────────────────────────────
print("Loading prices...")
prices = load_prices("BitcoinPricesDaily.csv")
print(f"  {len(prices.df)} fitting points")

# ── Result arrays ────────────────────────────────────────────────────────────
nP, nQ = len(PERCENTILES), len(QUANTILES)
slope_grid      = np.full((nP, nQ), np.nan)
intercept_grid  = np.full((nP, nQ), np.nan)
r2_grid         = np.full((nP, nQ), np.nan)
onset_grid      = np.full((nP, nQ), np.nan)
interval_grid   = np.full((nP, nQ), np.nan)
mean_K_grid     = np.full((nP, nQ), np.nan)
n_bubbles_grid  = np.full((nP, nQ), np.nan)

# ── Sweep ────────────────────────────────────────────────────────────────────
total = nP * nQ
for ip, pct in enumerate(PERCENTILES):
    for iq, qnt in enumerate(QUANTILES):
        idx = ip * nQ + iq + 1
        tag = f"[{idx:2d}/{total}] pct={pct:.2f} qnt={qnt:.3f}"
        try:
            sup = fit_support(prices, percentile=pct, quantile=qnt)
            slope_grid[ip, iq] = sup.slope
            intercept_grid[ip, iq] = sup.intercept

            fitted = fit_sequential(prices, sup)
            n_det = len(fitted)
            n_bubbles_grid[ip, iq] = n_det

            if n_det == 0:
                print(f"  {tag}  -> 0 bubbles detected, skipping")
                continue

            major, minor = classify(fitted, n_major=5)

            f_maj, f_min = predict_future(major, minor,
                                          t_last_data=prices.years[-1],
                                          n_major=3, n_minor=1)

            # Next bubble onset year (calendar)
            if f_maj:
                t_onset = f_maj[0]["t_rise"]
                onset_yr = GENESIS + pd.Timedelta(days=t_onset * 365.25)
                onset_grid[ip, iq] = onset_yr.year + onset_yr.dayofyear / 365.25
            elif f_min:
                t_onset = f_min[0]["t_rise"]
                onset_yr = GENESIS + pd.Timedelta(days=t_onset * 365.25)
                onset_grid[ip, iq] = onset_yr.year + onset_yr.dayofyear / 365.25

            # Mean bubble interval (from major bubbles)
            if len(major) >= 2:
                starts = [b["t_rise"] for b in major]
                intervals = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
                interval_grid[ip, iq] = np.mean(intervals)

            # Mean amplitude K (all fitted)
            mean_K_grid[ip, iq] = np.mean([b["K"] for b in fitted])

            # Composite R2
            all_future = sorted(f_maj + f_min, key=lambda b: b["t_rise"])
            comp = build_composite(sup, fitted, prices)
            r2_grid[ip, iq] = comp.r2

            print(f"  {tag}  -> {n_det} bubbles, R2={comp.r2:.4f}, onset={onset_grid[ip,iq]:.1f}")

        except Exception as e:
            print(f"  {tag}  -> FAILED: {e}")

# ── Plotting ─────────────────────────────────────────────────────────────────
print("\nGenerating heatmaps...")

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#181818",
    "axes.facecolor":   "#222222",
    "axes.edgecolor":   "#555555",
    "text.color":       "#dddddd",
    "axes.labelcolor":  "#dddddd",
    "xtick.color":      "#aaaaaa",
    "ytick.color":      "#aaaaaa",
    "grid.color":       "#333333",
})

panels = [
    ("Predicted Next Bubble Onset (year)",  onset_grid,    "plasma",   None),
    ("Composite R$^2$",                     r2_grid,       "viridis",  None),
    ("Support Slope",                       slope_grid,    "coolwarm", None),
    ("Mean Major Bubble Interval (yr)",     interval_grid, "magma",    None),
    ("Mean Amplitude K",                    mean_K_grid,   "inferno",  None),
    ("Number of Detected Bubbles",          n_bubbles_grid,"cividis",  None),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Bubble Model Sensitivity: Support Line Parameters\n"
             "(percentile filter x quantile regression target)",
             fontsize=15, fontweight="bold", color="#eeeeee", y=0.98)

# Find default indices
ip_def = PERCENTILES.index(DEFAULT_P)
iq_def = QUANTILES.index(DEFAULT_Q)

q_labels = [f"{q:.3f}" for q in QUANTILES]
p_labels = [f"{p:.2f}" for p in PERCENTILES]

for idx, (title, data, cmap, vrange) in enumerate(panels):
    ax = axes.flat[idx]
    kwargs = {}
    if vrange:
        kwargs["vmin"], kwargs["vmax"] = vrange

    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower", **kwargs)
    ax.set_xticks(range(nQ))
    ax.set_xticklabels(q_labels, fontsize=8, rotation=45)
    ax.set_yticks(range(nP))
    ax.set_yticklabels(p_labels, fontsize=8)
    ax.set_xlabel("Quantile target", fontsize=9)
    ax.set_ylabel("Percentile filter", fontsize=9)
    ax.set_title(title, fontsize=11, color="#eeeeee", pad=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors="#aaaaaa")

    # Annotate cells with values
    for i in range(nP):
        for j in range(nQ):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "X", ha="center", va="center",
                        fontsize=7, color="#ff4444", fontweight="bold")
            else:
                # Format based on panel
                if "onset" in title.lower():
                    txt = f"{val:.1f}"
                elif "R$^2$" in title or "R2" in title:
                    txt = f"{val:.4f}"
                elif "slope" in title.lower():
                    txt = f"{val:.3f}"
                elif "interval" in title.lower():
                    txt = f"{val:.2f}"
                elif "amplitude" in title.lower():
                    txt = f"{val:.2f}"
                else:
                    txt = f"{val:.0f}"
                # Pick text color for contrast
                norm_val = im.norm(val)
                rgba = im.cmap(norm_val)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc = "#111111" if lum > 0.5 else "#eeeeee"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=tc)

    # Mark default
    ax.plot(iq_def, ip_def, marker="s", markersize=14,
            markerfacecolor="none", markeredgecolor="#00ff88",
            markeredgewidth=2.5, zorder=10)
    ax.text(iq_def, ip_def - 0.42, "default", ha="center", va="top",
            fontsize=6.5, color="#00ff88", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.94])

svg_path = os.path.join(ROOT, "sensitivity_pq.svg")
jpg_path = os.path.join(ROOT, "sensitivity_pq.jpg")
fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(jpg_path, format="jpg", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {svg_path}")
print(f"Saved: {jpg_path}")
print("Done.")
