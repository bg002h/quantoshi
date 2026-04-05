#!/usr/bin/env python3
"""Rolling-window BM (Bubble Model) regime shift detection.

Tracks the core BM parameters over rolling time windows:
  - Support slope (B_sup) and intercept (A_sup) — power-law baseline
  - Support R² (how well the baseline fits window data)
  - Number of bubble-year peaks within window
  - Mean peak K (mean log-excess amplitude of bubbles in window)
  - Max peak K in window
  - Residual σ (log-price - support)

Avoids full sequential bubble fitting per window (too expensive).
Instead: fits support line, then characterizes bubble peaks at known years.

MANUAL REGENERATION ONLY.

Usage:
    btc_venv/bin/python3 tools/regime_shift_bm.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.stats import linregress
import statsmodels.api as sm


GENESIS = pd.Timestamp("2009-07-25")
STEP_YRS = 1.0 / 12.0
BUBBLE_YEARS = [2011, 2013, 2017, 2021, 2025]
BUBBLE_WINDOW = 0.75  # ± years around Jan 1 of each bubble year
SUPPORT_PCT = 0.20    # bottom 20% of OLS residuals as support points
SUPPORT_Q = 0.50      # quantile regression at median

REGIME_EVENTS = [
    ("2013-11-30", "2013 mania"),
    ("2017-12-17", "2017 peak / CME"),
    ("2020-03-12", "Covid crash"),
    ("2021-11-10", "2021 peak"),
    ("2022-11-11", "FTX collapse"),
    ("2024-01-10", "ETF approval"),
]


def _fit_bm_window(args):
    """Fit support + find peaks for a single window. Module-level for picklability."""
    t_end, t_win, lp_win = args
    try:
        if len(t_win) < 200:
            return _nan_result(t_end)

        log_t = np.log10(np.maximum(t_win, 0.1))

        # 1. OLS on all window data
        slope_ols, intercept_ols, _, _, _ = linregress(log_t, lp_win)
        ols_resid = lp_win - (intercept_ols + slope_ols * log_t)

        # 2. Bottom 20% filter
        cutoff = np.percentile(ols_resid, SUPPORT_PCT * 100)
        support_mask = ols_resid <= cutoff
        if support_mask.sum() < 20:
            return _nan_result(t_end)

        # 3. Quantile regression at Q50% on support subset
        X_sup = sm.add_constant(log_t[support_mask])
        try:
            res_sup = sm.QuantReg(lp_win[support_mask], X_sup).fit(
                q=SUPPORT_Q, max_iter=5000)
            intercept_sup = float(res_sup.params[0])
            slope_sup = float(res_sup.params[1])
        except Exception:
            # Fall back to OLS on support points if QR fails
            intercept_sup, slope_sup = intercept_ols, slope_ols

        # 4. Compute log-excess above support line
        log_support = intercept_sup + slope_sup * log_t
        log_excess = lp_win - log_support

        # 5. Compute R² of support line against window data
        ss_tot = float(np.sum((lp_win - np.mean(lp_win))**2))
        ss_res_sup = float(np.sum((lp_win - log_support)**2))
        r2_sup = 1.0 - ss_res_sup / ss_tot if ss_tot > 0 else float("nan")
        sigma_sup = float(np.std(lp_win - log_support))

        # 6. Find bubble peaks within window
        t_lo = t_win.min()
        t_hi = t_win.max()
        peak_Ks = []
        for yr in BUBBLE_YEARS:
            t_center = (pd.Timestamp(f"{yr}-01-01") - GENESIS).days / 365.25
            if t_center < t_lo or t_center > t_hi:
                continue
            search_lo = t_center - BUBBLE_WINDOW
            search_hi = t_center + BUBBLE_WINDOW
            mask = (t_win >= search_lo) & (t_win <= search_hi)
            if not mask.any():
                continue
            peak_K = float(np.max(log_excess[mask]))
            peak_Ks.append(peak_K)

        n_bubbles = len(peak_Ks)
        mean_K = float(np.mean(peak_Ks)) if peak_Ks else float("nan")
        max_K = float(np.max(peak_Ks)) if peak_Ks else float("nan")

        return {
            "t_end": t_end,
            "A_sup": intercept_sup,
            "B_sup": slope_sup,
            "r2_sup": r2_sup,
            "sigma_sup": sigma_sup,
            "n_bubbles": n_bubbles,
            "mean_K": mean_K,
            "max_K": max_K,
        }
    except BaseException as e:
        print(f"[fit_bm] t_end={t_end:.3f} failed: {type(e).__name__}: {e}", flush=True)
        return _nan_result(t_end)


def _nan_result(t_end):
    return {
        "t_end": t_end,
        "A_sup": float("nan"), "B_sup": float("nan"),
        "r2_sup": float("nan"), "sigma_sup": float("nan"),
        "n_bubbles": 0, "mean_K": float("nan"), "max_K": float("nan"),
    }


def run_config(label, width_yrs, t_all, lp_all, n_workers, csv_path=None):
    """Run rolling-window BM fits."""
    t_min = float(t_all.min())
    t_max = float(t_all.max())
    first_end = t_min + width_yrs
    ends = np.arange(first_end, t_max + 1e-9, STEP_YRS)

    args_list = []
    for t_end in ends:
        t_start = t_end - width_yrs
        mask = (t_all >= t_start) & (t_all <= t_end)
        args_list.append((float(t_end), t_all[mask].copy(), lp_all[mask].copy()))

    print(f"  {label}: {len(args_list)} windows, {n_workers} workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_fit_bm_window, args_list))
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s ({elapsed/len(args_list):.2f}s/window)")

    df = pd.DataFrame(results)
    df["end_date"] = [GENESIS + pd.Timedelta(days=t * 365.25) for t in df["t_end"]]

    if csv_path:
        cols = ["end_date", "t_end", "A_sup", "B_sup", "r2_sup", "sigma_sup",
                "n_bubbles", "mean_K", "max_K"]
        df[cols].to_csv(csv_path, index=False, float_format="%.6f")
        print(f"    Saved {csv_path}")
    return df


def plot_config(df, label, width_yrs, out_svg):
    """8-panel plot: A_sup, B_sup, r2_sup, sigma_sup, n_bubbles, mean_K, max_K."""
    panels = [
        ("A_sup", "Support intercept (A)"),
        ("B_sup", "Support slope (B)"),
        ("r2_sup", "Support R\u00b2"),
        ("sigma_sup", "Support \u03c3"),
        ("n_bubbles", "# bubbles in window"),
        ("mean_K", "Mean bubble K"),
        ("max_K", "Max bubble K"),
    ]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n),
                              sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")

    TITLE_COLOR = "#00d4ff"
    LABEL_COLOR = "#cccccc"
    TICK_COLOR = "#aaaaaa"
    LINE_COLOR = "#DAA520"  # goldenrod (BM family)
    EVENT_COLOR = "#888888"

    fig.suptitle(f"{label} \u2014 Rolling {width_yrs}-year windows, monthly steps",
                 color=TITLE_COLOR, fontsize=14, fontweight="bold")

    for i, (col, title) in enumerate(panels):
        ax = axes[i]
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors=TICK_COLOR, labelsize=8)
        ax.plot(df["end_date"], df[col], color=LINE_COLOR, linewidth=1.2)
        ax.set_ylabel(title, color=LABEL_COLOR, fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=30)
        ax.grid(True, alpha=0.15, color="#555555")
        for date_str, _ in REGIME_EVENTS:
            ax.axvline(pd.Timestamp(date_str), color=EVENT_COLOR,
                       linewidth=0.6, linestyle="--", alpha=0.5)

    # Event labels on top panel only
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
    print("=" * 60)
    print("Rolling-window BM regime shift detection")
    print("=" * 60)

    print("Loading Bitcoin prices...")
    pd_ = load_prices("BitcoinPricesDaily.csv")
    df = pd_.df_full[["date", "years", "log_price"]].copy()
    df = df[df["years"] >= 1.0].reset_index(drop=True)
    t_all = df["years"].values
    lp_all = df["log_price"].values
    print(f"  {len(df)} daily rows\n")

    n_workers = max(1, os.cpu_count() - 1)
    print(f"  Using {n_workers} workers\n")

    configs = [
        ("BM 7yr windows", 7.0, "regime_shift_bm_7yr.svg"),
        ("BM 9yr windows", 9.0, "regime_shift_bm_9yr.svg"),
    ]

    for label, width, svg_path in configs:
        csv_path = svg_path.replace(".svg", ".csv")
        df = run_config(label, width, t_all, lp_all, n_workers, csv_path=csv_path)
        plot_config(df, label, width, svg_path)
        print(f"  Saved {svg_path}\n")

    print("Done. Integrate into /E separately.")


if __name__ == "__main__":
    main()
