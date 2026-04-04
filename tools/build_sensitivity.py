#!/usr/bin/env python3
"""Sensitivity analysis: bubble model parameters vs support line slope/intercept.

Generates /B (BM reference) and /BB (EF reference) with deuteranomaly-friendly
colormaps and BM/EF values marked on colorbars.
"""
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import ticker

ROOT = "/scratch/code/bitcoinprojections"
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices
from model_toolkit.support import fit_support, fixed_support
from model_toolkit.fitting import find_peaks, DEFAULT_CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Load data and get reference values ───────────────────────────────────────
print("Loading prices...")
prices = load_prices("BitcoinPricesDaily.csv")
print(f"  {len(prices.df)} fitting points")

print("Fitting BM support line...")
ref_sup = fit_support(prices)
BM_SLOPE = ref_sup.slope
BM_INTERCEPT = ref_sup.intercept
print(f"  BM: slope={BM_SLOPE:.4f}, intercept={BM_INTERCEPT:.4f}")

EF_SLOPE = 5.248017
EF_INTERCEPT = -1.630623
print(f"  EF: slope={EF_SLOPE:.4f}, intercept={EF_INTERCEPT:.4f}")

# ── Grid parameters ──────────────────────────────────────────────────────────
N_SLOPE = 50
N_INTERCEPT = 50

slope_lo, slope_hi = 4.0, 7.0
intc_lo = BM_INTERCEPT - 2.5
intc_hi = BM_INTERCEPT + 2.5

slopes = np.linspace(slope_lo, slope_hi, N_SLOPE)
intercepts = np.linspace(intc_lo, intc_hi, N_INTERCEPT)

print(f"Grid: slope [{slope_lo}, {slope_hi}] x intercept [{intc_lo:.2f}, {intc_hi:.2f}]")

# ── Precompute arrays ────────────────────────────────────────────────────────
log_t = prices.log_years
log_p = prices.log_prices
years = prices.years
SS_tot = np.sum((log_p - np.mean(log_p)) ** 2)

bubble_years = DEFAULT_CONFIG["BUBBLE_YEARS"]
window = DEFAULT_CONFIG["BUBBLE_YEAR_WINDOW"]
t_last = years[-1]

# ── Sweep ────────────────────────────────────────────────────────────────────
R2_grid = np.full((N_INTERCEPT, N_SLOPE), np.nan)
onset_grid = np.full((N_INTERCEPT, N_SLOPE), np.nan)
K_grid = np.full((N_INTERCEPT, N_SLOPE), np.nan)
interval_grid = np.full((N_INTERCEPT, N_SLOPE), np.nan)
rise_interval_grid = np.full((N_INTERCEPT, N_SLOPE), np.nan)

total = N_SLOPE * N_INTERCEPT
done = 0

for si, slp in enumerate(slopes):
    for ii, intc in enumerate(intercepts):
        done += 1
        if done % 100 == 0 or done == total:
            print(f"  [{done}/{total}]", end="\r")

        log_sup = intc + slp * log_t
        SS_res = np.sum((log_p - log_sup) ** 2)
        R2_grid[ii, si] = 1.0 - SS_res / SS_tot

        log_excess = log_p - log_sup
        peaks = find_peaks(log_excess, years, bubble_years, window)

        if len(peaks) < 2:
            continue

        peak_ts = np.array([p["peak_t"] for p in peaks])
        peak_Ks = np.array([p["raw_K"] for p in peaks])

        pos_mask = peak_Ks > 0
        if pos_mask.sum() < 2:
            continue

        peak_ts_pos = peak_ts[pos_mask]
        peak_Ks_pos = peak_Ks[pos_mask]

        K_grid[ii, si] = np.mean(peak_Ks_pos)

        intervals = np.diff(peak_ts_pos)
        interval_grid[ii, si] = np.mean(intervals)

        # t_rise intervals: find trough (min excess) between consecutive peaks
        rise_ts = []
        for pi in range(len(peak_ts_pos)):
            if pi == 0:
                # First bubble: scan from start of data to peak
                t_lo_search = years[0]
            else:
                # Trough between previous peak and this peak
                t_lo_search = peak_ts_pos[pi - 1]
            t_hi_search = peak_ts_pos[pi]
            mask_between = (years >= t_lo_search) & (years <= t_hi_search)
            if mask_between.any():
                trough_idx = np.argmin(log_excess[mask_between])
                rise_ts.append(years[mask_between][trough_idx])
        if len(rise_ts) >= 2:
            rise_intervals = np.diff(rise_ts)
            rise_interval_grid[ii, si] = np.mean(rise_intervals)

        if len(intervals) >= 2:
            idx = np.arange(len(intervals), dtype=float)
            coeffs = np.polyfit(idx, intervals, 1)
            next_intv = np.polyval(coeffs, len(intervals))
            next_intv = max(next_intv, 1.0)
        else:
            next_intv = intervals[0]

        next_onset_t = peak_ts_pos[-1] + next_intv
        next_onset_year = 2009 + (207 / 365.25) + next_onset_t
        onset_grid[ii, si] = next_onset_year

print()

# ── R2 ridge line ────────────────────────────────────────────────────────────
ridge_intc = np.full(N_SLOPE, np.nan)
for si in range(N_SLOPE):
    col = R2_grid[:, si]
    if np.any(~np.isnan(col)):
        ridge_intc[si] = intercepts[np.nanargmax(col)]

# ── Deuteranomaly-friendly colormaps ─────────────────────────────────────────
# Avoid red-green gradients. Use blue-orange, blue-yellow, purple-orange.
CMAPS = {
    "r2":       "cividis",       # blue-yellow, excellent for deutan
    "onset":    "plasma",        # purple-yellow, deutan-safe
    "K":        "inferno",       # black-purple-orange-yellow, deutan-safe
    "interval": "cividis",       # blue-yellow
}

TITLE_COLOR = "#00d4ff"
LABEL_COLOR = "#cccccc"
TICK_COLOR = "#aaaaaa"
BM_COLOR = "#ff6600"      # orange — visible to deutans
EF_COLOR = "#00aaff"       # blue — contrasts with orange for deutans
RIDGE_COLOR = "#ffffff"
CBAR_LABEL_COLOR = "#bbbbbb"


def _get_ref_value(grid, slope, intercept):
    """Get the grid value at a reference point."""
    si = int(np.argmin(np.abs(slopes - slope)))
    ii = int(np.argmin(np.abs(intercepts - intercept)))
    return grid[ii, si]


def _mark_colorbar(cb, value, color, label, side="right"):
    """Add a horizontal marker line on a colorbar at a specific value."""
    if np.isnan(value):
        return
    cb.ax.axhline(y=value, color=color, linewidth=2, linestyle="-")
    # Label on the side
    if side == "right":
        cb.ax.text(1.4, value, f" {label}", color=color, fontsize=7,
                   fontweight="bold", va="center", ha="left",
                   transform=cb.ax.get_yaxis_transform(),
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a1a2e",
                             edgecolor=color, alpha=0.9, linewidth=0.8))
    else:
        cb.ax.text(-0.4, value, f"{label} ", color=color, fontsize=7,
                   fontweight="bold", va="center", ha="right",
                   transform=cb.ax.get_yaxis_transform(),
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a1a2e",
                             edgecolor=color, alpha=0.9, linewidth=0.8))


def build_figure(primary_slope, primary_intercept, primary_label, primary_color,
                 secondary_slope, secondary_intercept, secondary_label, secondary_color,
                 suptitle_suffix=""):
    """Build the 5-panel sensitivity figure."""

    fig = plt.figure(figsize=(16, 18))
    fig.patch.set_facecolor("#1a1a2e")
    # 3 rows × 2 cols, last row has 1 panel centered
    axes = [
        fig.add_subplot(3, 2, 1),
        fig.add_subplot(3, 2, 2),
        fig.add_subplot(3, 2, 3),
        fig.add_subplot(3, 2, 4),
        fig.add_subplot(3, 2, 5),
    ]

    panels = [
        (axes[0], R2_grid, "Support Line R$^2$", "R$^2$", CMAPS["r2"]),
        (axes[1], onset_grid, "Predicted Next Bubble Onset", "Calendar Year", CMAPS["onset"]),
        (axes[2], K_grid, "Mean Bubble Amplitude (K)", "log$_{10}$ excess above support", CMAPS["K"]),
        (axes[3], interval_grid, "Mean Peak-to-Peak Interval", "Years between peaks", CMAPS["interval"]),
        (axes[4], rise_interval_grid, "Mean Rise-to-Rise Interval", "Years between cycle onsets", CMAPS["interval"]),
    ]

    for ax, data, title, cbar_label, cmap in panels:
        ax.set_facecolor("#16213e")
        masked = np.ma.masked_invalid(data)

        if "R$^2$" in title:
            vmin = max(np.nanmin(data), -1.0)
            vmax = np.nanmax(data)
            im = ax.pcolormesh(slopes, intercepts, masked,
                               cmap=cmap, shading="nearest",
                               vmin=vmin, vmax=vmax)
        else:
            im = ax.pcolormesh(slopes, intercepts, masked,
                               cmap=cmap, shading="nearest")

        cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.04)
        cb.set_label(cbar_label, color=CBAR_LABEL_COLOR, fontsize=10)
        cb.ax.tick_params(colors=TICK_COLOR, labelsize=8)
        cb.outline.set_edgecolor("#555555")

        # Mark reference values on colorbar
        prim_val = _get_ref_value(data, primary_slope, primary_intercept)
        sec_val = _get_ref_value(data, secondary_slope, secondary_intercept)
        _mark_colorbar(cb, prim_val, primary_color, primary_label, side="right")
        _mark_colorbar(cb, sec_val, secondary_color, secondary_label, side="left")

        # R2 ridge line
        valid_ridge = ~np.isnan(ridge_intc)
        ax.plot(slopes[valid_ridge], ridge_intc[valid_ridge],
                "--", color=RIDGE_COLOR, linewidth=0.8, alpha=0.5)

        # Primary reference point
        ax.plot(primary_slope, primary_intercept, "o", color=primary_color,
                markersize=10, markeredgecolor="white", markeredgewidth=1.8, zorder=10)
        ax.annotate(
            f"{primary_label}\n({primary_slope:.2f}, {primary_intercept:.2f})",
            xy=(primary_slope, primary_intercept),
            xytext=(18, 18), textcoords="offset points",
            color=primary_color, fontsize=8.5, fontweight="bold",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=primary_color, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor=primary_color, alpha=0.85),
            zorder=11,
        )

        # Secondary reference point
        ax.plot(secondary_slope, secondary_intercept, "s", color=secondary_color,
                markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=10)
        ax.annotate(
            f"{secondary_label}\n({secondary_slope:.2f}, {secondary_intercept:.2f})",
            xy=(secondary_slope, secondary_intercept),
            xytext=(-18, -22), textcoords="offset points",
            color=secondary_color, fontsize=8.5, fontweight="bold",
            ha="right", va="top",
            arrowprops=dict(arrowstyle="->", color=secondary_color, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor=secondary_color, alpha=0.85),
            zorder=11,
        )

        ax.set_xlabel("Support Slope (power-law exponent)", color=LABEL_COLOR, fontsize=10)
        ax.set_ylabel("Support Intercept (log$_{10}$)", color=LABEL_COLOR, fontsize=10)
        ax.set_title(title, color=TITLE_COLOR, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=TICK_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#555555")

    # R2 contour lines
    R2_smooth = np.where(np.isnan(R2_grid), -999, R2_grid)
    try:
        cs = axes[0].contour(slopes, intercepts, R2_smooth,
                             levels=[0.5, 0.7, 0.8, 0.9, 0.95],
                             colors="white", linewidths=0.7, linestyles=":")
        axes[0].clabel(cs, fmt="%.2f", fontsize=7, colors="white")
    except Exception:
        pass

    # Onset year contours
    onset_smooth = np.where(np.isnan(onset_grid), 0, onset_grid)
    try:
        cs2 = axes[1].contour(slopes, intercepts, onset_smooth,
                              levels=[2026, 2028, 2030, 2032, 2034],
                              colors="white", linewidths=0.7, linestyles=":")
        axes[1].clabel(cs2, fmt="%d", fontsize=7, colors="white")
    except Exception:
        pass

    fig.suptitle(f"Bubble Model Sensitivity Analysis{suptitle_suffix}\nSupport Line Slope vs Intercept",
                 color=TITLE_COLOR, fontsize=17, fontweight="bold", y=0.99)

    fig.text(0.5, 0.945,
             f"log$_{{10}}$(price) = intercept + slope "
             r"$\times$"
             f" log$_{{10}}$(t)  |  t = years since 2009-07-25  |  "
             f"BM: ({BM_SLOPE:.3f}, {BM_INTERCEPT:.3f})  |  "
             f"EF: ({EF_SLOPE:.3f}, {EF_INTERCEPT:.3f})",
             ha="center", color="#888888", fontsize=9.5)

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    return fig


# ── Generate /B (BM primary, EF secondary) ──────────────────────────────────
print("Generating /B (BM primary)...")
fig_b = build_figure(
    BM_SLOPE, BM_INTERCEPT, "BM", BM_COLOR,
    EF_SLOPE, EF_INTERCEPT, "EF", EF_COLOR,
)
fig_b.savefig(os.path.join(ROOT, "sensitivity_sweep.svg"), format="svg",
              facecolor=fig_b.get_facecolor(), edgecolor="none", bbox_inches="tight")
fig_b.savefig(os.path.join(ROOT, "sensitivity_sweep.jpg"), format="jpg",
              facecolor=fig_b.get_facecolor(), edgecolor="none", bbox_inches="tight", dpi=200)
plt.close(fig_b)
print(f"  SVG: {os.path.getsize(os.path.join(ROOT, 'sensitivity_sweep.svg')) / 1024:.0f} KB")
print(f"  JPG: {os.path.getsize(os.path.join(ROOT, 'sensitivity_sweep.jpg')) / 1024:.0f} KB")

# ── Generate /BB (EF primary, BM secondary) ─────────────────────────────────
print("Generating /BB (EF primary)...")
fig_bb = build_figure(
    EF_SLOPE, EF_INTERCEPT, "EF", EF_COLOR,
    BM_SLOPE, BM_INTERCEPT, "BM", BM_COLOR,
    suptitle_suffix=" — Empirical Floor",
)
fig_bb.savefig(os.path.join(ROOT, "sensitivity_sweep_ef.svg"), format="svg",
               facecolor=fig_bb.get_facecolor(), edgecolor="none", bbox_inches="tight")
fig_bb.savefig(os.path.join(ROOT, "sensitivity_sweep_ef.jpg"), format="jpg",
               facecolor=fig_bb.get_facecolor(), edgecolor="none", bbox_inches="tight", dpi=200)
plt.close(fig_bb)
print(f"  SVG: {os.path.getsize(os.path.join(ROOT, 'sensitivity_sweep_ef.svg')) / 1024:.0f} KB")
print(f"  JPG: {os.path.getsize(os.path.join(ROOT, 'sensitivity_sweep_ef.jpg')) / 1024:.0f} KB")

# ── Print reference values ───────────────────────────────────────────────────
for label, slp, intc in [("BM", BM_SLOPE, BM_INTERCEPT), ("EF", EF_SLOPE, EF_INTERCEPT)]:
    print(f"\n{label} reference values:")
    print(f"  R2 = {_get_ref_value(R2_grid, slp, intc):.4f}")
    print(f"  Next onset = {_get_ref_value(onset_grid, slp, intc):.1f}")
    print(f"  Mean K = {_get_ref_value(K_grid, slp, intc):.3f}")
    print(f"  Mean interval = {_get_ref_value(interval_grid, slp, intc):.2f} yr")
