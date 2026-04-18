#!/usr/bin/env python3
"""t₀ sensitivity sweep for the Bitcoin power-law — OUT-OF-SAMPLE edition.

Produces a 4-panel diagnostic showing β, R², σ, and n_samples as
functions of the chosen time origin t₀ across 2007-01-01 → 2017-01-01
at 7-day step.

Train / test split (strict, no overlap):
  fit data     = dates where (t > 1 yr) AND (date < 2015-01-01)
  holdout data = dates >= 2015-01-01

The fit is performed on pre-2015 data only. σ_holdout / R²_holdout
evaluate the fit on post-2015 data the model has NEVER SEEN during
fitting, so the metrics are genuinely out-of-sample predictive-power
measures — no in-sample contamination.

  * OLS primary + median QR overlay on β
  * Weightings: log_density (primary), unweighted, 1/t
  * σ_fit    = residual std on the pre-2015 fit data  (in-sample)
  * σ_holdout= residual std on the post-2015 holdout  (out-of-sample)
  * R²_fit, R²_holdout analogous
  * Sweep stops being valid when pre-2015 fit data < 500 samples.
    At ~7-day step, that happens around t₀ ≈ 2013-07; the curves
    break visibly at that point.

Output: docs/sweep_t0.svg + docs/sweep_t0.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "btc_web"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from model_toolkit.data import load_prices

# Centralized SSOT for colors. pulled at module load so we keep hex
# literals out of this file per the `test_no_hex_literals_outside_colors_module`
# lint.
from colors import (
    MODEL_TRACE_COLORS, PLOT_BG_COLOR, GRID_MAJOR_COLOR,
    TEXT_COLOR, FALLBACK_MODEL_GRAY, SPINE_COLOR,
)
GRID_COLOR = GRID_MAJOR_COLOR

# ══════════════════════════════════════════════════════════════════════
# Sweep configuration
# ══════════════════════════════════════════════════════════════════════

T0_START        = pd.Timestamp("2007-01-01")
T0_END          = pd.Timestamp("2017-01-01")
T0_STEP_DAYS    = 7
T_MIN_FILTER    = 1.0                         # years
N_SAMPLES_MIN   = 500
HOLDOUT_START   = pd.Timestamp("2015-01-01")  # R²_fixed evaluation window
CANONICAL_T0    = pd.Timestamp("2009-07-25")
QR_SEED         = 42

WEIGHTINGS = ("log_density", "unweighted", "1_over_t")


def _compute_weights(t: np.ndarray, scheme: str) -> np.ndarray:
    """Return weights of shape (n,), normalised so Σw == n."""
    n = len(t)
    if scheme == "unweighted":
        w = np.ones(n)
    elif scheme == "1_over_t":
        w = 1.0 / np.maximum(t, 1e-6)
    elif scheme == "log_density":
        # Quarter-decade bins: floor(log10(t) * 4). Each sample's
        # weight is inverse its bin's population, giving each
        # quarter-decade equal total influence.
        log_t = np.log10(np.maximum(t, 1e-6))
        bins = np.floor(log_t * 4).astype(int)
        bins_shifted = bins - bins.min()
        counts = np.bincount(bins_shifted)
        w = 1.0 / counts[bins_shifted]
    else:
        raise ValueError(f"unknown weighting: {scheme}")
    return w * (n / w.sum())


def _fit_ols(log_t: np.ndarray, log_p: np.ndarray, weights: np.ndarray):
    """Return (alpha, beta) of log_p ≈ alpha + beta·log_t via weighted OLS."""
    slope, intercept = np.polyfit(log_t, log_p, 1, w=np.sqrt(weights))
    return float(intercept), float(slope)


def _fit_qr_median(log_t: np.ndarray, log_p: np.ndarray,
                    weights: np.ndarray, rng: np.random.Generator) -> float:
    """Median quantile regression slope. Weighting via 5×n multinomial
    resampling (same pattern as btc_web/engines/custom_fit.py:fit_qr)."""
    n = len(log_t)
    probs = weights / weights.sum()
    idx = rng.choice(n, size=5 * n, replace=True, p=probs)
    X = sm.add_constant(log_t[idx])
    try:
        res = sm.QuantReg(log_p[idx], X).fit(q=0.5, max_iter=10000)
        return float(res.params[1])
    except Exception:
        return float("nan")


def _compute_r2_weighted(log_p, pred, weights):
    ss_res = float(np.sum(weights * (log_p - pred) ** 2))
    wbar = float(np.sum(weights * log_p) / np.sum(weights))
    ss_tot = float(np.sum(weights * (log_p - wbar) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _compute_r2_unweighted(log_p, pred):
    ss_res = float(np.sum((log_p - pred) ** 2))
    ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oos", "full"), default="oos",
                    help=("oos: strict pre-{HOLDOUT}/post-{HOLDOUT} train/test; "
                          "full: fit on ALL data each t₀, in-sample metrics only"))
    args = ap.parse_args()
    mode = args.mode
    is_full = (mode == "full")

    print(f"Mode: {mode} "
          + ("(full-fit, in-sample metrics only, no holdout)"
             if is_full else "(strict out-of-sample)"))
    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    df = prices.df_full.copy()
    # Ensure date is pandas Timestamp
    dates = pd.to_datetime(df["date"])
    close = df["price"].values.astype(float)
    log_p_all = np.log10(np.maximum(close, 1e-10))
    date_vals = dates.values  # np.datetime64 for vectorised math
    print(f"  n={len(close)} samples, {dates.min()} → {dates.max()}")

    # Build t₀ grid
    t0_grid = pd.date_range(T0_START, T0_END, freq=f"{T0_STEP_DAYS}D")
    print(f"  t₀ grid: {len(t0_grid)} candidates "
          f"({t0_grid[0].date()} → {t0_grid[-1].date()}, {T0_STEP_DAYS}-day step)")

    # Pre-compute holdout mask over the full data
    holdout_mask_all = dates >= HOLDOUT_START
    n_holdout = int(holdout_mask_all.sum())
    print(f"  holdout window: {HOLDOUT_START.date()} → today, "
          f"n_holdout = {n_holdout}")
    print(f"  weightings: {WEIGHTINGS}  (primary = log_density)")
    print(f"  filter: t > {T_MIN_FILTER} yr, n_min = {N_SAMPLES_MIN}")

    # Per-weighting result arrays
    results = {
        w: {
            "beta_ols":     np.full(len(t0_grid), np.nan),
            "beta_qr":      np.full(len(t0_grid), np.nan),
            "alpha_ols":    np.full(len(t0_grid), np.nan),
            "r2_fit":       np.full(len(t0_grid), np.nan),
            "r2_holdout":   np.full(len(t0_grid), np.nan),
            "sigma_fit":    np.full(len(t0_grid), np.nan),
            "sigma_holdout":np.full(len(t0_grid), np.nan),
            "n_fit":        np.zeros(len(t0_grid), dtype=int),
            "n_holdout":    np.zeros(len(t0_grid), dtype=int),
        }
        for w in WEIGHTINGS
    }

    import time as _time
    t_start = _time.perf_counter()
    print("Sweeping...")
    rng = np.random.default_rng(QR_SEED)

    for i, t0 in enumerate(t0_grid):
        t_all = (date_vals - t0.to_datetime64()).astype("timedelta64[D]").astype(float) / 365.25

        if is_full:
            # Full-fit mode: use ALL available data (t > 1 yr) each t₀.
            # No train/test split; in-sample metrics only. Holdout arrays
            # are empty; σ_holdout / R²_holdout stay NaN.
            fit_mask = t_all > T_MIN_FILTER
            hold_mask = np.zeros_like(fit_mask)
        else:
            # OOS mode: strict pre-/post-HOLDOUT_START split.
            fit_mask = (t_all > T_MIN_FILTER) & (~holdout_mask_all.values)
            hold_mask = holdout_mask_all.values

        n_fit  = int(fit_mask.sum())
        n_hold = int(hold_mask.sum())

        for w_name in WEIGHTINGS:
            results[w_name]["n_fit"][i]     = n_fit
            results[w_name]["n_holdout"][i] = n_hold
        if n_fit < N_SAMPLES_MIN:
            continue

        # Fit arrays (pre-2015)
        t_fit     = t_all[fit_mask]
        log_p_fit = log_p_all[fit_mask]
        log_t_fit = np.log10(t_fit)

        # Holdout arrays (2015+)
        t_hold     = t_all[hold_mask]
        log_p_hold = log_p_all[hold_mask]

        for w_name in WEIGHTINGS:
            weights = _compute_weights(t_fit, w_name)

            # OLS fit on pre-2015 data only
            alpha, beta = _fit_ols(log_t_fit, log_p_fit, weights)
            pred_fit = alpha + beta * log_t_fit

            # In-sample metrics (on the fit data)
            sigma_fit = float(np.sqrt(
                np.sum((log_p_fit - pred_fit) ** 2) / max(n_fit - 2, 1)))
            r2_fit = _compute_r2_weighted(log_p_fit, pred_fit, weights)

            # Out-of-sample metrics (on 2015+ holdout — never seen in fit)
            if len(t_hold) >= 10:
                pred_hold = alpha + beta * np.log10(t_hold)
                r2_holdout = _compute_r2_unweighted(log_p_hold, pred_hold)
                sigma_holdout = float(np.sqrt(
                    np.sum((log_p_hold - pred_hold) ** 2) /
                    max(len(t_hold) - 2, 1)))
            else:
                r2_holdout = float("nan")
                sigma_holdout = float("nan")

            # Median QR on the fit data
            beta_qr = _fit_qr_median(log_t_fit, log_p_fit, weights, rng)

            results[w_name]["beta_ols"][i]      = beta
            results[w_name]["alpha_ols"][i]     = alpha
            results[w_name]["beta_qr"][i]       = beta_qr
            results[w_name]["r2_fit"][i]        = r2_fit
            results[w_name]["r2_holdout"][i]    = r2_holdout
            results[w_name]["sigma_fit"][i]     = sigma_fit
            results[w_name]["sigma_holdout"][i] = sigma_holdout

        if (i + 1) % 50 == 0 or i == len(t0_grid) - 1:
            elapsed = _time.perf_counter() - t_start
            print(f"  [{i+1:>4}/{len(t0_grid)}]  t₀={t0.date()}  "
                  f"n_fit={n_fit:>5}  n_hold={n_hold}  "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = _time.perf_counter() - t_start
    print(f"Sweep complete in {elapsed:.1f}s.")

    # ──────────────────────────────────────────────────────────────
    # CSV output
    # ──────────────────────────────────────────────────────────────
    rows = []
    for i, t0 in enumerate(t0_grid):
        for w_name in WEIGHTINGS:
            r = results[w_name]
            rows.append({
                "t0_date":       t0.strftime("%Y-%m-%d"),
                "weighting":     w_name,
                "n_fit":         int(r["n_fit"][i]),
                "n_holdout":     int(r["n_holdout"][i]),
                "beta_ols":      r["beta_ols"][i],
                "beta_qr":       r["beta_qr"][i],
                "alpha_ols":     r["alpha_ols"][i],
                "r2_fit":        r["r2_fit"][i],
                "r2_holdout":    r["r2_holdout"][i],
                "sigma_fit":     r["sigma_fit"][i],
                "sigma_holdout": r["sigma_holdout"][i],
            })
    suffix = "_fullfit" if is_full else ""
    out_csv = os.path.join(ROOT, "docs", f"sweep_t0{suffix}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # ──────────────────────────────────────────────────────────────
    # Plot — 4 stacked panels
    # ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import AutoMinorLocator

    PL_C = MODEL_TRACE_COLORS["pl"]
    QR_C = MODEL_TRACE_COLORS["qr"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    fig.patch.set_facecolor(PLOT_BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PLOT_BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)

    # Panel 1: β
    ax = axes[0]
    ax.plot(t0_grid, results["log_density"]["beta_ols"],
            color=PL_C, linewidth=2.0, label="OLS · log_density")
    ax.plot(t0_grid, results["unweighted"]["beta_ols"],
            color=PL_C, linewidth=1.2, alpha=0.45, label="OLS · unweighted")
    ax.plot(t0_grid, results["1_over_t"]["beta_ols"],
            color=PL_C, linewidth=1.2, alpha=0.45, linestyle="--",
            label="OLS · 1/t")
    ax.plot(t0_grid, results["log_density"]["beta_qr"],
            color=QR_C, linewidth=2.0, linestyle="--",
            label="QR median · log_density")
    ax.set_ylabel("β (power-law exponent)", color=TEXT_COLOR)
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)

    # Panel 2: R²
    ax = axes[1]
    if is_full:
        # Full-fit mode: only in-sample R² exists; show all 3 weightings.
        ax.plot(t0_grid, results["log_density"]["r2_fit"],
                color=PL_C, linewidth=2.0,
                label="R² · log_density")
        ax.plot(t0_grid, results["unweighted"]["r2_fit"],
                color=PL_C, linewidth=1.2, alpha=0.55,
                label="R² · unweighted")
        ax.plot(t0_grid, results["1_over_t"]["r2_fit"],
                color=PL_C, linewidth=1.2, alpha=0.55, linestyle=":",
                label="R² · 1/t")
        all_r2 = np.concatenate([
            results[w]["r2_fit"] for w in WEIGHTINGS])
    else:
        # OOS mode: R²_holdout across 3 weightings + R²_fit log_density ref.
        ax.plot(t0_grid, results["log_density"]["r2_holdout"],
                color=PL_C, linewidth=2.0,
                label="R²_holdout · log_density")
        ax.plot(t0_grid, results["unweighted"]["r2_holdout"],
                color=PL_C, linewidth=1.2, alpha=0.5,
                label="R²_holdout · unweighted")
        ax.plot(t0_grid, results["1_over_t"]["r2_holdout"],
                color=PL_C, linewidth=1.2, alpha=0.5, linestyle=":",
                label="R²_holdout · 1/t")
        ax.plot(t0_grid, results["log_density"]["r2_fit"],
                color=PL_C, linewidth=1.5, linestyle="--", alpha=0.7,
                label="R²_fit · log_density (in-sample ref.)")
        all_r2 = np.concatenate([
            results["log_density"]["r2_fit"],
            results["log_density"]["r2_holdout"],
            results["unweighted"]["r2_holdout"],
            results["1_over_t"]["r2_holdout"],
        ])
    all_r2 = all_r2[~np.isnan(all_r2)]
    if len(all_r2):
        lo = max(all_r2.min() - 0.01, 0.0)
        hi = min(all_r2.max() + 0.01, 1.0)
        ax.set_ylim(lo, hi)
    ax.set_ylabel("R² (log-space)", color=TEXT_COLOR)
    ax.legend(loc="best", fontsize=9, framealpha=0.8, ncol=2)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)

    # Panel 3: σ
    ax = axes[2]
    if is_full:
        ax.plot(t0_grid, results["log_density"]["sigma_fit"],
                color=PL_C, linewidth=2.0,
                label="σ · log_density")
        ax.plot(t0_grid, results["unweighted"]["sigma_fit"],
                color=PL_C, linewidth=1.2, alpha=0.55,
                label="σ · unweighted")
        ax.plot(t0_grid, results["1_over_t"]["sigma_fit"],
                color=PL_C, linewidth=1.2, alpha=0.55, linestyle=":",
                label="σ · 1/t")
    else:
        ax.plot(t0_grid, results["log_density"]["sigma_holdout"],
                color=PL_C, linewidth=2.0,
                label="σ_holdout · log_density")
        ax.plot(t0_grid, results["unweighted"]["sigma_holdout"],
                color=PL_C, linewidth=1.2, alpha=0.5,
                label="σ_holdout · unweighted")
        ax.plot(t0_grid, results["1_over_t"]["sigma_holdout"],
                color=PL_C, linewidth=1.2, alpha=0.5, linestyle=":",
                label="σ_holdout · 1/t")
        ax.plot(t0_grid, results["log_density"]["sigma_fit"],
                color=PL_C, linewidth=1.5, linestyle="--", alpha=0.7,
                label="σ_fit · log_density (in-sample ref.)")
    ax.set_ylabel("σ (log₁₀ price residual std)", color=TEXT_COLOR)
    ax.legend(loc="best", fontsize=9, framealpha=0.8, ncol=2)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)

    # Panel 4: sample counts
    ax = axes[3]
    if is_full:
        ax.plot(t0_grid, results["log_density"]["n_fit"],
                color=FALLBACK_MODEL_GRAY, linewidth=1.5,
                label=f"n_fit (t > {T_MIN_FILTER} yr, all data)")
    else:
        ax.plot(t0_grid, results["log_density"]["n_fit"],
                color=FALLBACK_MODEL_GRAY, linewidth=1.5,
                label=f"n_fit (t > {T_MIN_FILTER} yr, pre-{HOLDOUT_START.date()})")
        ax.plot(t0_grid, results["log_density"]["n_holdout"],
                color=FALLBACK_MODEL_GRAY, linewidth=1.5, linestyle=":",
                label=f"n_holdout (≥ {HOLDOUT_START.date()})")
    ax.axhline(y=N_SAMPLES_MIN, color=FALLBACK_MODEL_GRAY,
                linestyle="--", alpha=0.5, linewidth=1.0,
                label=f"fit-data floor (n ≥ {N_SAMPLES_MIN})")
    ax.set_ylabel("sample count", color=TEXT_COLOR)
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)

    # ──────────────────────────────────────────────────────────────
    # Annotations: canonical t₀ + optima
    # ──────────────────────────────────────────────────────────────
    # Pick the right arrays for the mode.
    if is_full:
        # Full-fit: in-sample metrics are the only option.
        sigma_primary = results["log_density"]["sigma_fit"]
        r2_primary    = results["log_density"]["r2_fit"]
        sigma_label   = "σ_fit"
        r2_label      = "R²_fit"
    else:
        sigma_primary = results["log_density"]["sigma_holdout"]
        r2_primary    = results["log_density"]["r2_holdout"]
        sigma_label   = "σ_holdout"
        r2_label      = "R²_holdout"

    valid = ~np.isnan(sigma_primary)
    if valid.any():
        argmin_sigma_idx = int(np.nanargmin(sigma_primary))
        argmax_r2_idx    = int(np.nanargmax(r2_primary))
        opt_sigma_t0     = t0_grid[argmin_sigma_idx]
        opt_r2_t0        = t0_grid[argmax_r2_idx]
    else:
        argmin_sigma_idx = argmax_r2_idx = None
        opt_sigma_t0 = opt_r2_t0 = None

    for idx, ax in enumerate(axes):
        ax.axvline(CANONICAL_T0, color=TEXT_COLOR,
                    linewidth=1.5, alpha=0.7,
                    label=("canonical 2009-07-25" if idx == 0 else None))
        if opt_sigma_t0 is not None:
            ax.axvline(opt_sigma_t0, color=PL_C, linestyle=":",
                        linewidth=1.2, alpha=0.75,
                        label=(f"argmin {sigma_label} = {opt_sigma_t0.date()}"
                                if idx == 0 else None))
        if (opt_r2_t0 is not None and
                opt_sigma_t0 is not None and
                abs((opt_r2_t0 - opt_sigma_t0).days) > T0_STEP_DAYS):
            ax.axvline(opt_r2_t0, color=QR_C, linestyle=":",
                        linewidth=1.2, alpha=0.75,
                        label=(f"argmax {r2_label} = {opt_r2_t0.date()}"
                                if idx == 0 else None))
    # Refresh legend on top panel to include the new vertical-line entries.
    axes[0].legend(loc="best", fontsize=9, framealpha=0.8)

    # ──────────────────────────────────────────────────────────────
    # X-axis formatting
    # ──────────────────────────────────────────────────────────────
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_minor_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("t₀ (time origin)", color=TEXT_COLOR)

    # Title + subtitle + footer
    mode_suffix_title = (
        " — full-fit (all data, in-sample)" if is_full
        else " — strict out-of-sample"
    )
    fig.suptitle(
        f"Power-law fit quality vs time origin{mode_suffix_title}",
        fontsize=14, fontweight="bold", y=0.995, color=TEXT_COLOR,
    )
    mode_subtitle = (
        "  ·  fit on all t > 1 yr, in-sample metrics"
        if is_full
        else f"  ·  fit pre-{HOLDOUT_START.date()}, eval post-{HOLDOUT_START.date()}"
    )
    fig.text(
        0.5, 0.973,
        r"$\log_{10}(\mathrm{price}) = \alpha + \beta\,\log_{10}(t - t_0)$"
        f"  ·  OLS primary + median QR overlay{mode_subtitle}",
        ha="center", fontsize=10, alpha=0.8, color=TEXT_COLOR,
    )
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    fig.text(
        0.5, 0.005,
        f"Data: BitcoinPricesDaily.csv (through {today_str}, n={len(close)})  "
        f"·  Filter: t > {T_MIN_FILTER} yr, n_min = {N_SAMPLES_MIN}  "
        f"·  Step = {T0_STEP_DAYS} days  "
        f"·  R²_fixed holdout ≥ {HOLDOUT_START.date()}",
        ha="center", fontsize=8, alpha=0.6, color=TEXT_COLOR,
    )

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    out_svg = os.path.join(ROOT, "docs", f"sweep_t0{suffix}.svg")
    fig.savefig(out_svg, bbox_inches="tight", facecolor=PLOT_BG_COLOR)
    # Also emit a JPG alongside the SVG — some viewers render SVG
    # matplotlib output poorly (clipped dashes, missing text). JPG is
    # the universal fallback. Use 150 dpi for readable screen output.
    out_jpg = os.path.join(ROOT, "docs", f"sweep_t0{suffix}.jpg")
    fig.savefig(out_jpg, bbox_inches="tight", facecolor=PLOT_BG_COLOR,
                dpi=150)
    plt.close(fig)
    print(f"Wrote {out_svg}")
    print(f"Wrote {out_jpg}")

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    canonical_idx = int(np.argmin(np.abs(
        np.array([(t - CANONICAL_T0).days for t in t0_grid]))))
    can_sigma = sigma_primary[canonical_idx]
    can_beta  = results["log_density"]["beta_ols"][canonical_idx]
    can_n_fit = int(results["log_density"]["n_fit"][canonical_idx])

    print()
    print("=" * 76)
    mode_label = ("FULL-FIT (all data, in-sample)" if is_full
                  else f"OUT-OF-SAMPLE (pre-{HOLDOUT_START.date()} fit)")
    print(f"Summary — OLS · log_density weighting · {mode_label}")
    print("-" * 76)
    if argmin_sigma_idx is not None:
        opt_sigma = sigma_primary[argmin_sigma_idx]
        opt_beta  = results["log_density"]["beta_ols"][argmin_sigma_idx]
        opt_n_fit = int(results["log_density"]["n_fit"][argmin_sigma_idx])
        delta     = (opt_sigma_t0 - CANONICAL_T0).days
        print(f"  Optimal t₀ (argmin {sigma_label}):  {opt_sigma_t0.strftime('%Y-%m-%d')}  "
              f"({sigma_label}={opt_sigma:.4f}, β={opt_beta:.3f}, n_fit={opt_n_fit})")
        print(f"  Canonical t₀:                  {CANONICAL_T0.strftime('%Y-%m-%d')}  "
              f"({sigma_label}={can_sigma:.4f}, β={can_beta:.3f}, n_fit={can_n_fit})")
        print(f"  Δt₀ from canonical:            {abs(delta):>4} days "
              f"({'after' if delta > 0 else 'before' if delta < 0 else '='})")
    if argmax_r2_idx is not None:
        print(f"  Optimal t₀ (argmax {r2_label}):  "
              f"{opt_r2_t0.strftime('%Y-%m-%d')}")
        d_r2 = (opt_r2_t0 - CANONICAL_T0).days
        print(f"  Δt₀ from canonical:            {abs(d_r2):>4} days")
    if (argmin_sigma_idx is not None and argmax_r2_idx is not None and
            abs((opt_sigma_t0 - opt_r2_t0).days) > 30):
        print(f"  ⚠  argmin {sigma_label} and argmax {r2_label} "
              f"disagree by > 30 days — worth investigating.")
    print("=" * 76)


if __name__ == "__main__":
    main()
