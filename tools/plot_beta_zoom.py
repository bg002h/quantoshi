#!/usr/bin/env python3
"""Zoomed β(t₀) plot — show only t₀ dates where 5 ≤ β ≤ 6, with dense
horizontal gridlines at 0.1-unit increments.

Reads docs/sweep_t0_floor.csv (log_density rows) which contains:
  * beta_ols  → β_floor  (QR q=0.5 on bottom-20% support subset)
  * beta_qr   → β_all-data (OLS on the full t > 1 yr dataset, reference)

Output:
  * docs/beta_zoom_5_6.svg
  * docs/beta_zoom_5_6.jpg
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "btc_web"))
os.chdir(ROOT)

from colors import (
    MODEL_TRACE_COLORS, PLOT_BG_COLOR, GRID_MAJOR_COLOR,
    TEXT_COLOR, FALLBACK_MODEL_GRAY, SPINE_COLOR,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PL_C = MODEL_TRACE_COLORS["pl"]
QR_C = MODEL_TRACE_COLORS["qr"]

BETA_LO = 4.5
BETA_HI = 6.0
DATE_LO = pd.Timestamp("2009-01-03")
DATE_HI = pd.Timestamp("2009-12-31")
CANONICAL_T0 = pd.Timestamp("2009-07-25")


def main():
    df = pd.read_csv(os.path.join(ROOT, "docs", "sweep_t0_floor.csv"))
    # Keep only log_density rows (BM floor method doesn't use weightings;
    # all three rows are duplicates but only log_density has data).
    df = df[df["weighting"] == "log_density"].copy()
    df["t0_date"] = pd.to_datetime(df["t0_date"])
    df = df.sort_values("t0_date").reset_index(drop=True)

    # In floor mode the CSV columns were repurposed:
    #   beta_ols = β_floor  (QR on support subset)
    #   beta_qr  = β_all-data (OLS on full t > 1 yr)
    beta_floor    = df["beta_ols"].to_numpy()
    beta_all_data = df["beta_qr"].to_numpy()
    t0_arr        = df["t0_date"].to_numpy()

    # Slice to the explicit date window [DATE_LO, DATE_HI].
    date_mask = (t0_arr >= np.datetime64(DATE_LO)) & (t0_arr <= np.datetime64(DATE_HI))
    window_idx = np.where(date_mask)[0]
    if len(window_idx) == 0:
        raise SystemExit(f"No t₀ dates in [{DATE_LO.date()}, {DATE_HI.date()}] — "
                         "CSV sweep range may not cover this window.")
    i_lo = window_idx.min()
    i_hi = window_idx.max()
    t0_slice = t0_arr[i_lo:i_hi + 1]
    beta_floor_slice    = beta_floor[i_lo:i_hi + 1]
    beta_all_data_slice = beta_all_data[i_lo:i_hi + 1]
    print(f"Plotting {i_hi - i_lo + 1} t₀ candidates: "
          f"{pd.Timestamp(t0_slice[0]).date()} → "
          f"{pd.Timestamp(t0_slice[-1]).date()}")

    # ──────────────────────────────────────────────────────────────
    # Plot
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    ax.plot(t0_slice, beta_floor_slice,
            color=PL_C, linewidth=2.0,
            label="β_floor (QR q=0.5 on bottom-20% support subset)")
    ax.plot(t0_slice, beta_all_data_slice,
            color=QR_C, linewidth=1.6, linestyle="--",
            label="β_all-data (OLS on full t > 1 yr)")

    # Zoom y-axis
    ax.set_ylim(BETA_LO, BETA_HI)

    # Horizontal gridlines at every 0.1
    major_y = np.arange(BETA_LO, BETA_HI + 0.001, 0.1)
    ax.set_yticks(major_y)
    ax.grid(True, axis="y", color=GRID_MAJOR_COLOR, linewidth=0.7, alpha=0.9)
    # Minor gridlines at every 0.05
    minor_y = np.arange(BETA_LO, BETA_HI + 0.001, 0.05)
    ax.set_yticks(minor_y, minor=True)
    ax.grid(True, axis="y", which="minor",
            color=GRID_MAJOR_COLOR, linewidth=0.35, alpha=0.6)

    # Canonical marker
    ax.axvline(CANONICAL_T0, color=TEXT_COLOR, linewidth=1.5, alpha=0.7,
                label="canonical 2009-07-25")

    # Horizontal reference at β=5.08 (prod PowerLawModel β value)
    ax.axhline(5.08, color=FALLBACK_MODEL_GRAY, linewidth=1.0,
                linestyle=":", alpha=0.8,
                label="β = 5.08 (prod PowerLawModel)")

    # X axis — explicit date window + monthly gridlines.
    ax.set_xlim(DATE_LO, DATE_HI)
    # Major ticks every month (these are the gridline anchors).
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # Minor ticks weekly for sub-grid reference (no gridline).
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    # Vertical gridlines every month, matching the horizontal gridline
    # visual weight (they anchor the canonical + reference lines).
    ax.grid(True, axis="x", which="major",
            color=GRID_MAJOR_COLOR, linewidth=0.7, alpha=0.9)
    # Rotate date labels so monthly ticks don't overlap.
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    ax.set_xlabel("t₀ (time origin)", color=TEXT_COLOR)
    ax.set_ylabel("β (power-law exponent)", color=TEXT_COLOR)
    ax.set_title(
        f"Power-law exponent vs t₀ — calendar year 2009 (β ∈ [{BETA_LO:.1f}, {BETA_HI:.1f}])",
        fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)

    # Footer — annotate canonical β value
    can_mask = np.abs((t0_slice -
                        np.datetime64(CANONICAL_T0)).astype("timedelta64[D]").astype(int)) < 7
    if can_mask.any():
        can_idx = np.argmax(can_mask)
        can_floor = beta_floor_slice[can_idx]
        can_all   = beta_all_data_slice[can_idx]
        fig.text(0.5, 0.02,
                  f"At canonical 2009-07-25:  β_floor = {can_floor:.3f}  ·  "
                  f"β_all-data = {can_all:.3f}",
                  ha="center", fontsize=9,
                  alpha=0.75, color=TEXT_COLOR)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    svg_path = os.path.join(ROOT, "docs", "beta_zoom_2009.svg")
    jpg_path = os.path.join(ROOT, "docs", "beta_zoom_2009.jpg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor=PLOT_BG_COLOR)
    fig.savefig(jpg_path, bbox_inches="tight", facecolor=PLOT_BG_COLOR, dpi=150)
    plt.close(fig)
    print(f"Wrote {svg_path}")
    print(f"Wrote {jpg_path}")


if __name__ == "__main__":
    main()
