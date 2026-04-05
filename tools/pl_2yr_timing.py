#!/usr/bin/env python3
"""PL 2yr slope vs price: timing analysis.

For each of the clipped and unbounded PL 2yr rolling-window CSVs:
  1. Align window-end B with interpolated log_price at the same t_end.
  2. Compute Pearson correlation with:
       - log_price (level)
       - log_price detrended vs long-run PL (log-excess)
       - d log_price / dt  (momentum)
  3. Compute lagged cross-correlation of B vs log_price over ±24 months
     and report the lag that maximizes |corr| (positive lag = B leads price).
  4. Emit a 2-panel SVG: twin-axis B + log_price over time, then the
     lag-correlation curve.

Outputs:
  regime_shift_pl_2yr_timing_unbounded.svg + .csv
  regime_shift_pl_2yr_timing_clipped.svg + .csv
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices

GENESIS = pd.Timestamp("2009-07-25")
MAX_LAG_MONTHS = 24
REGIME_EVENTS = [
    ("2013-11-30", "2013 mania"),
    ("2017-12-17", "2017 peak / CME"),
    ("2020-03-12", "Covid crash"),
    ("2021-11-10", "2021 peak"),
    ("2022-11-11", "FTX collapse"),
    ("2024-01-10", "ETF approval"),
]


def compute_timing(csv_path, t_all, lp_all, label):
    df = pd.read_csv(csv_path)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.dropna(subset=["B"]).reset_index(drop=True)

    # Interpolate log_price at each window t_end
    lp_at_end = np.interp(df["t_end"].values, t_all, lp_all)
    df["log_price"] = lp_at_end

    # Long-run PL baseline: OLS over full series
    log_t_all = np.log10(np.maximum(t_all, 0.1))
    B_long, A_long = np.polyfit(log_t_all, lp_all, 1)
    lp_baseline = A_long + B_long * np.log10(np.maximum(df["t_end"], 0.1))
    df["log_excess"] = df["log_price"] - lp_baseline

    # Momentum: d log_price / dt using centered 3-month diff
    # (approx 0.25yr spacing since monthly steps)
    B = df["B"].values
    lp = df["log_price"].values
    excess = df["log_excess"].values

    # Momentum: forward 6-month difference of log_price
    dt = 6.0 / 12.0  # 6 months
    momentum = np.full(len(df), np.nan)
    for i in range(len(df)):
        t_fut = df["t_end"].iloc[i] + dt
        if t_fut <= t_all[-1]:
            lp_fut = np.interp(t_fut, t_all, lp_all)
            momentum[i] = (lp_fut - lp[i]) / dt
    df["momentum_6m"] = momentum

    # Zero-lag correlations
    def _corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            return float("nan")
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    corr_level   = _corr(B, lp)
    corr_excess  = _corr(B, excess)
    corr_moment  = _corr(B, momentum)

    # Lagged cross-correlation: B shifted by L months vs log_price
    # Positive lag = B leads price by L months
    lags = np.arange(-MAX_LAG_MONTHS, MAX_LAG_MONTHS + 1)
    xcorr_level = []
    xcorr_excess = []
    for L in lags:
        if L >= 0:
            x = B[: len(B) - L]
            y_level = lp[L:]
            y_excess = excess[L:]
        else:
            x = B[-L:]
            y_level = lp[: len(lp) + L]
            y_excess = excess[: len(excess) + L]
        xcorr_level.append(_corr(x, y_level))
        xcorr_excess.append(_corr(x, y_excess))
    xcorr_level = np.asarray(xcorr_level)
    xcorr_excess = np.asarray(xcorr_excess)

    peak_idx_level  = int(np.nanargmax(np.abs(xcorr_level)))
    peak_idx_excess = int(np.nanargmax(np.abs(xcorr_excess)))
    peak_lag_level    = int(lags[peak_idx_level])
    peak_corr_level   = float(xcorr_level[peak_idx_level])
    peak_lag_excess   = int(lags[peak_idx_excess])
    peak_corr_excess  = float(xcorr_excess[peak_idx_excess])

    print(f"\n=== {label} ===")
    print(f"  rows: {len(df)}")
    print(f"  corr(B, log_price)       = {corr_level:+.3f}")
    print(f"  corr(B, log_excess vs PL)= {corr_excess:+.3f}")
    print(f"  corr(B, momentum_6m)     = {corr_moment:+.3f}")
    print(f"  peak |xcorr(B, log_price)|       = {peak_corr_level:+.3f}"
          f"  at lag = {peak_lag_level:+d} months"
          f"  ({'B leads' if peak_lag_level > 0 else 'B trails' if peak_lag_level < 0 else 'synchronous'})")
    print(f"  peak |xcorr(B, log_excess)|      = {peak_corr_excess:+.3f}"
          f"  at lag = {peak_lag_excess:+d} months"
          f"  ({'B leads' if peak_lag_excess > 0 else 'B trails' if peak_lag_excess < 0 else 'synchronous'})")

    # Detect log_excess peaks and troughs.
    # Monthly spacing -> distance=18 enforces >=1.5yr between extrema.
    exc = excess.copy()
    exc[~np.isfinite(exc)] = 0.0
    prom = 0.30  # prominence in log-units — keeps only major cycle extrema
    # Distance=30 months enforces halving-cycle-scale spacing (~2.5yr min),
    # which suppresses mid-cycle blips like the 2019-06 summer rally.
    peak_idxs, _ = find_peaks(exc, distance=30, prominence=prom)
    trough_idxs, _ = find_peaks(-exc, distance=30, prominence=prom)

    # Predicted B extrema: shift log_excess extrema forward by |lag| months.
    # log_excess PEAK  -> B TROUGH (neg corr at neg lag)
    # log_excess TROUGH -> B PEAK
    L = abs(peak_lag_excess)
    shift = pd.DateOffset(months=L)
    predicted_B_troughs = [df["end_date"].iloc[i] + shift for i in peak_idxs]
    predicted_B_peaks   = [df["end_date"].iloc[i] + shift for i in trough_idxs]

    stats = dict(
        corr_level=corr_level, corr_excess=corr_excess, corr_moment=corr_moment,
        peak_lag_level=peak_lag_level, peak_corr_level=peak_corr_level,
        peak_lag_excess=peak_lag_excess, peak_corr_excess=peak_corr_excess,
        predicted_B_troughs=predicted_B_troughs,
        predicted_B_peaks=predicted_B_peaks,
        lag_months=L,
    )
    return df, lags, xcorr_level, xcorr_excess, stats


def plot_timing(df, lags, xcorr_level, xcorr_excess, stats, label, out_svg):
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.3])

    TITLE_COLOR = "#00d4ff"
    LABEL_COLOR = "#cccccc"
    TICK_COLOR = "#aaaaaa"
    B_COLOR = "#FF9F40"
    PRICE_COLOR = "#4da6ff"
    EVENT_COLOR = "#888888"
    ZERO_COLOR = "#666666"

    # Title with stats
    title = (
        f"PL 2yr rolling slope B vs log\u2081\u2080(price) \u2014 {label}\n"
        f"corr(B, log_price)={stats['corr_level']:+.3f}  |  "
        f"corr(B, log_excess)={stats['corr_excess']:+.3f}  |  "
        f"corr(B, 6m momentum)={stats['corr_moment']:+.3f}\n"
        f"peak |xcorr(B, log_price)|={stats['peak_corr_level']:+.3f} "
        f"at lag {stats['peak_lag_level']:+d} mo  \u00b7  "
        f"peak |xcorr(B, log_excess)|={stats['peak_corr_excess']:+.3f} "
        f"at lag {stats['peak_lag_excess']:+d} mo"
    )
    fig.suptitle(title, color=TITLE_COLOR, fontsize=11.5, fontweight="bold")

    # Panel 1: twin-axis B + log_price
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#16213e")
    for spine in ax1.spines.values():
        spine.set_color("#555555")
    ax1.tick_params(colors=TICK_COLOR, labelsize=9)
    ax1.plot(df["end_date"], df["B"], color=B_COLOR, linewidth=1.4,
             label="B (2yr slope)")
    ax1.set_ylabel("B (slope)", color=B_COLOR, fontsize=10)
    ax1.axhline(0, color=ZERO_COLOR, linewidth=0.6, linestyle=":")

    ax1b = ax1.twinx()
    ax1b.plot(df["end_date"], df["log_price"], color=PRICE_COLOR,
              linewidth=1.4, linestyle="-", alpha=0.9, label="log\u2081\u2080(price)")
    ax1b.set_ylabel("log\u2081\u2080(price)", color=PRICE_COLOR, fontsize=10)
    ax1b.tick_params(colors=PRICE_COLOR, labelsize=9)
    for spine in ax1b.spines.values():
        spine.set_color("#555555")

    ax1.grid(True, alpha=0.15, color="#555555")
    for date_str, _ in REGIME_EVENTS:
        ax1.axvline(pd.Timestamp(date_str), color=EVENT_COLOR,
                    linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Window end date", color=LABEL_COLOR, fontsize=10)

    # Overlay predicted B extrema (shifted from log_excess extrema by |lag|)
    TROUGH_COLOR = "#ff6a4a"  # red — predicted B trough (post-bubble)
    PEAK_COLOR = "#70e8b0"    # green — predicted B peak (post-bear)
    date_min, date_max = ax1.get_xlim()
    for d in stats["predicted_B_troughs"]:
        d_num = pd.Timestamp(d).to_pydatetime().toordinal() - \
                pd.Timestamp("1970-01-01").to_pydatetime().toordinal()
        ax1.axvline(d, color=TROUGH_COLOR, linewidth=1.0,
                    linestyle="-", alpha=0.55)
    for d in stats["predicted_B_peaks"]:
        ax1.axvline(d, color=PEAK_COLOR, linewidth=1.0,
                    linestyle="-", alpha=0.55)

    # Date labels at bottom for predicted extrema
    y_bot = ax1.get_ylim()[0]
    for d in stats["predicted_B_troughs"]:
        ax1.annotate(
            f"B\u2193 {pd.Timestamp(d).strftime('%Y-%m')}",
            xy=(d, y_bot), xytext=(3, 3), textcoords="offset points",
            rotation=90, color=TROUGH_COLOR, fontsize=7,
            ha="left", va="bottom",
        )
    for d in stats["predicted_B_peaks"]:
        ax1.annotate(
            f"B\u2191 {pd.Timestamp(d).strftime('%Y-%m')}",
            xy=(d, y_bot), xytext=(3, 3), textcoords="offset points",
            rotation=90, color=PEAK_COLOR, fontsize=7,
            ha="left", va="bottom",
        )

    # Event labels
    y_top = ax1.get_ylim()[1]
    for date_str, name in REGIME_EVENTS:
        ax1.annotate(name, xy=(pd.Timestamp(date_str), y_top),
                     xytext=(2, 2), textcoords="offset points",
                     rotation=90, color=EVENT_COLOR,
                     fontsize=7, ha="left", va="top")

    # Combined legend — add synthetic handles for the prediction markers
    from matplotlib.lines import Line2D
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    h_trough = Line2D([0], [0], color=TROUGH_COLOR, linewidth=1.0,
                      label=f"predicted B\u2193 (log-exc peak + {stats['lag_months']}mo)")
    h_peak = Line2D([0], [0], color=PEAK_COLOR, linewidth=1.0,
                    label=f"predicted B\u2191 (log-exc trough + {stats['lag_months']}mo)")
    ax1.legend(h1 + h2 + [h_trough, h_peak],
               l1 + l2 + [h_trough.get_label(), h_peak.get_label()],
               loc="upper left", fontsize=8, facecolor="#101a2e",
               edgecolor="#555555", labelcolor=LABEL_COLOR)

    # Panel 2: cross-correlation vs lag
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#16213e")
    for spine in ax2.spines.values():
        spine.set_color("#555555")
    ax2.tick_params(colors=TICK_COLOR, labelsize=9)
    ax2.plot(lags, xcorr_level, color=PRICE_COLOR, linewidth=1.3,
             label="B vs log\u2081\u2080(price)")
    ax2.plot(lags, xcorr_excess, color="#a0e060", linewidth=1.3,
             label="B vs log-excess (detrended)")
    ax2.axhline(0, color=ZERO_COLOR, linewidth=0.6, linestyle=":")
    ax2.axvline(0, color=ZERO_COLOR, linewidth=0.6, linestyle=":")
    ax2.axvline(stats["peak_lag_level"], color=PRICE_COLOR,
                linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axvline(stats["peak_lag_excess"], color="#a0e060",
                linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_xlabel("Lag (months, positive = B leads price)",
                   color=LABEL_COLOR, fontsize=10)
    ax2.set_ylabel("Pearson corr", color=LABEL_COLOR, fontsize=10)
    ax2.set_xlim(-MAX_LAG_MONTHS, MAX_LAG_MONTHS)
    ax2.grid(True, alpha=0.15, color="#555555")
    ax2.legend(loc="upper left", fontsize=9, facecolor="#101a2e",
               edgecolor="#555555", labelcolor=LABEL_COLOR)

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

    variants = [
        ("PL 6mo unbounded", "regime_shift_pl_6mo.csv",
         "regime_shift_pl_6mo_timing_unbounded.svg",
         "regime_shift_pl_6mo_timing_unbounded.csv"),
        ("PL 1yr unbounded", "regime_shift_pl_1yr.csv",
         "regime_shift_pl_1yr_timing_unbounded.svg",
         "regime_shift_pl_1yr_timing_unbounded.csv"),
        ("PL 2yr unbounded OLS",  "regime_shift_pl_2yr.csv",
         "regime_shift_pl_2yr_timing_unbounded.svg",
         "regime_shift_pl_2yr_timing_unbounded.csv"),
    ]

    for label, csv_in, svg_out, csv_out in variants:
        df, lags, xc_level, xc_excess, stats = compute_timing(
            csv_in, t_all, lp_all, label)
        plot_timing(df, lags, xc_level, xc_excess, stats, label, svg_out)
        # Save the aligned timeseries + xcorr curve
        out = df[["end_date", "t_end", "B", "log_price",
                  "log_excess", "momentum_6m"]].copy()
        out.to_csv(csv_out, index=False, float_format="%.6f")
        lag_df = pd.DataFrame({
            "lag_months": lags,
            "xcorr_level": xc_level,
            "xcorr_excess": xc_excess,
        })
        lag_df.to_csv(csv_out.replace(".csv", "_xcorr.csv"),
                      index=False, float_format="%.6f")
        print(f"  Saved {svg_out} + {csv_out}")


if __name__ == "__main__":
    main()
