#!/usr/bin/env python3
"""Post-process PL regime-shift CSVs to add A_center = mean(log10(price))
in each window. Centered-log-time intercept A' is orthogonal to B —
decouples "level" from "slope" in the sliding PL fit.

Regenerates regime_shift_pl_{2,5,7,9}yr.{csv,svg} with A_center column
and a 3-panel plot (A, A_center, B) plus R²/sigma.

MANUAL REGENERATION ONLY.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices

GENESIS = pd.Timestamp("2009-07-25")
REGIME_EVENTS = [
    ("2013-11-30", "2013 mania"),
    ("2017-12-17", "2017 peak / CME"),
    ("2020-03-12", "Covid crash"),
    ("2021-11-10", "2021 peak"),
    ("2022-11-11", "FTX collapse"),
    ("2024-01-10", "ETF approval"),
]

CONFIGS = [
    ("PL 2yr windows", 2.0, "regime_shift_pl_2yr"),
    ("PL 5yr windows", 5.0, "regime_shift_pl_5yr"),
    ("PL 7yr windows", 7.0, "regime_shift_pl_7yr"),
    ("PL 9yr windows", 9.0, "regime_shift_pl_9yr"),
]


def compute_a_center(df_csv, width_yrs, t_all, lp_all):
    """For each window (t_end, width), compute x_bar = mean(log10(t_win))
    and A_center = A + B * x_bar (== mean log-price in window)."""
    x_bars = []
    for t_end in df_csv["t_end"].values:
        t_start = t_end - width_yrs
        mask = (t_all >= t_start) & (t_all <= t_end)
        t_win = t_all[mask]
        if len(t_win) < 2:
            x_bars.append(float("nan"))
            continue
        log_t = np.log10(np.maximum(t_win, 0.1))
        x_bars.append(float(np.mean(log_t)))
    x_bar = np.asarray(x_bars)
    a_center = df_csv["A"].values + df_csv["B"].values * x_bar
    return x_bar, a_center


def plot_pl(df, label, width_yrs, out_svg):
    panels = [
        ("A", "A (intercept @ t=1)"),
        ("A_center", "A' (centered = mean log\u2081\u2080 price)"),
        ("B", "B (slope)"),
        ("sigma", "\u03c3"),
        ("r2", "R\u00b2"),
    ]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n),
                             sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")

    TITLE_COLOR = "#00d4ff"
    LABEL_COLOR = "#cccccc"
    TICK_COLOR = "#aaaaaa"
    LINE_COLOR = "#FF9F40"
    LINE_COLOR_ALT = "#00d4ff"
    EVENT_COLOR = "#888888"

    fig.suptitle(
        f"{label} \u2014 Rolling {width_yrs}-year PL windows, monthly steps",
        color=TITLE_COLOR, fontsize=14, fontweight="bold",
    )

    for i, (col, title) in enumerate(panels):
        ax = axes[i]
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors=TICK_COLOR, labelsize=8)
        color = LINE_COLOR_ALT if col == "A_center" else LINE_COLOR
        ax.plot(df["end_date"], df[col], color=color, linewidth=1.2)
        ax.set_ylabel(title, color=LABEL_COLOR, fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=30)
        ax.grid(True, alpha=0.15, color="#555555")
        for date_str, _ in REGIME_EVENTS:
            ax.axvline(pd.Timestamp(date_str), color=EVENT_COLOR,
                       linewidth=0.6, linestyle="--", alpha=0.5)

    ax0 = axes[0]
    y_top = ax0.get_ylim()[1]
    for date_str, name in REGIME_EVENTS:
        ax0.annotate(name, xy=(pd.Timestamp(date_str), y_top),
                     xytext=(2, 2), textcoords="offset points",
                     rotation=90, color=EVENT_COLOR,
                     fontsize=7, ha="left", va="top")

    axes[-1].set_xlabel("Window end date", color=LABEL_COLOR, fontsize=10)
    fig.savefig(out_svg, format="svg", facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading Bitcoin prices...")
    pd_ = load_prices("BitcoinPricesDaily.csv")
    df_price = pd_.df_full[["years", "log_price"]].copy()
    df_price = df_price[df_price["years"] >= 1.0].reset_index(drop=True)
    t_all = df_price["years"].values
    lp_all = df_price["log_price"].values
    print(f"  {len(df_price)} daily rows\n")

    for label, width, stem in CONFIGS:
        csv_in = f"{stem}.csv"
        if not os.path.exists(csv_in):
            print(f"  SKIP {csv_in} (not found)")
            continue
        df_csv = pd.read_csv(csv_in)
        x_bar, a_center = compute_a_center(df_csv, width, t_all, lp_all)
        df_csv["x_bar"] = x_bar
        df_csv["A_center"] = a_center
        df_csv["end_date"] = pd.to_datetime(df_csv["end_date"])

        csv_out = csv_in
        cols = ["end_date", "t_end", "A", "B", "x_bar", "A_center",
                "sigma", "r2"]
        df_csv[cols].to_csv(csv_out, index=False, float_format="%.6f")
        print(f"  {label}: wrote A_center to {csv_out}")

        svg_out = f"{stem}.svg"
        plot_pl(df_csv, label, width, svg_out)
        print(f"          plot  {svg_out}")

        # Report correlation of A/B vs A'/B
        corr_ab = float(np.corrcoef(df_csv["A"].dropna(),
                                    df_csv["B"].dropna())[0, 1])
        mask = df_csv["A_center"].notna() & df_csv["B"].notna()
        corr_acb = float(np.corrcoef(df_csv.loc[mask, "A_center"],
                                     df_csv.loc[mask, "B"])[0, 1])
        print(f"          corr(A,B)={corr_ab:+.3f}  "
              f"corr(A_center,B)={corr_acb:+.3f}\n")


if __name__ == "__main__":
    main()
