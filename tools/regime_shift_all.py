#!/usr/bin/env python3
"""Rolling-window LPPL regime shift detection — comprehensive edition.

Tracks evolution of ALL parameters of LP1, LP2, and LP3 across rolling
time windows of Bitcoin price history. Generates a single HTML page at
`regime_shift_all.html` with four anchor-linked sections:

  1. LP1 (6 params) — 5-year windows
  2. LP2 (9 params) — 5-year windows
  3. LP3 (12 params) — 7-year windows
  4. LP3 (12 params) — 9-year windows

Each section has one panel per parameter, plus residual σ and R² panels,
with vertical dashed lines at known Bitcoin regime events.

MANUAL REGENERATION ONLY — this script is NOT called by any systemd
timer, update_prices.py, or rebuild_caches.sh. It produces diagnostic
output for human analysis; automatic refreshes would waste compute.

Uses ProcessPoolExecutor (all CPU cores) for parallel window fitting.
Expected runtime: 5-15 minutes depending on CPU count.

Usage:
    btc_venv/bin/python3 tools/regime_shift_all.py

Output:
    regime_shift_all.html      — served at /E
    regime_shift_lp1_5yr.svg   — LP1 5yr section
    regime_shift_lp2_5yr.svg   — LP2 5yr section
    regime_shift_lp3_7yr.svg   — LP3 7yr section
    regime_shift_lp3_9yr.svg   — LP3 9yr section
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
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution


GENESIS = pd.Timestamp("2009-07-25")
STEP_YRS = 1.0 / 12.0  # monthly

REGIME_EVENTS = [
    ("2013-11-30", "2013 mania"),
    ("2017-12-17", "2017 peak / CME"),
    ("2020-03-12", "Covid crash"),
    ("2021-11-10", "2021 peak"),
    ("2022-11-11", "FTX collapse"),
    ("2024-01-10", "ETF approval"),
]


# ── Model functions (module-level for multiprocessing picklability) ──────

def lp1_model(t_safe, A, B, C, W, PHI, D):
    return A + B * np.log10(t_safe) + C * t_safe ** (-D) * np.cos(W * np.log(t_safe) + PHI)


def lp2_model(t_safe, A, B, C1, W1, PHI1, D, C2, W2, PHI2):
    return (A + B * np.log10(t_safe)
            + C1 * t_safe ** (-D) * np.cos(W1 * np.log(t_safe) + PHI1)
            + C2 * np.cos(W2 * np.log(t_safe) + PHI2))


def lp3_model(t_safe, A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3):
    return (A + B * np.log10(t_safe)
            + C1 * t_safe ** (-D) * np.cos(W1 * np.log(t_safe) + PHI1)
            + C2 * np.cos(W2 * np.log(t_safe) + PHI2)
            + C3 * np.cos(W3 * np.log(t_safe) + PHI3))


def lp4_model(t_safe, A, B, C1, W1, PHI1, D, C2, W2, PHI2, C3, W3, PHI3, C4, W4, PHI4):
    return (A + B * np.log10(t_safe)
            + C1 * t_safe ** (-D) * np.cos(W1 * np.log(t_safe) + PHI1)
            + C2 * np.cos(W2 * np.log(t_safe) + PHI2)
            + C3 * np.cos(W3 * np.log(t_safe) + PHI3)
            + C4 * np.cos(W4 * np.log(t_safe) + PHI4))


def pl_model(t_safe, A, B):
    """Pure power law (log-log linear): log10(price) = A + B*log10(t)."""
    return A + B * np.log10(t_safe)


def linppl_model(t_safe, A, B, C, W_cal, PHI, D):
    """LinPPL — oscillation in CALENDAR time, not log-time."""
    return A + B * np.log10(t_safe) + C * t_safe**(-D) * np.cos(W_cal * t_safe + PHI)


# W_max widened to 40 to match production fitters
LP1_BOUNDS = [
    (-3.0, 1.0), (3.0, 7.0), (0.01, 3.0),
    (2.0, 40.0), (-np.pi, np.pi), (0.01, 2.0),
]
LP1_NAMES = ["A", "B", "C", "W", "PHI", "D"]

LP2_BOUNDS = [
    (-3.0, 1.0), (3.0, 7.0), (0.01, 3.0),
    (2.0, 40.0), (-np.pi, np.pi), (0.01, 2.0),
    (0.0, 1.5), (3.0, 40.0), (-np.pi, np.pi),
]
LP2_NAMES = ["A", "B", "C1", "W1", "PHI1", "D", "C2", "W2", "PHI2"]

LP3_BOUNDS = [
    (-3.0, 1.0), (3.0, 7.0), (0.01, 3.0),
    (2.0, 40.0), (-np.pi, np.pi), (0.01, 2.0),
    (0.0, 1.5), (3.0, 40.0), (-np.pi, np.pi),
    (0.0, 1.5), (3.0, 40.0), (-np.pi, np.pi),
]
LP3_NAMES = ["A", "B", "C1", "W1", "PHI1", "D", "C2", "W2", "PHI2", "C3", "W3", "PHI3"]

LP4_BOUNDS = [
    (-3.0, 1.0), (3.0, 7.0), (0.01, 3.0),
    (2.0, 40.0), (-np.pi, np.pi), (0.01, 2.0),
    (0.0, 1.5), (3.0, 40.0), (-np.pi, np.pi),
    (0.0, 1.5), (3.0, 40.0), (-np.pi, np.pi),
    (0.0, 1.5), (3.0, 40.0), (-np.pi, np.pi),
]
LP4_NAMES = ["A", "B", "C1", "W1", "PHI1", "D",
             "C2", "W2", "PHI2", "C3", "W3", "PHI3", "C4", "W4", "PHI4"]

PL_BOUNDS = [(-3.0, 1.0), (3.0, 7.0)]
PL_NAMES = ["A", "B"]

# LinPPL: calendar-periodic. W_cal ∈ [0.5, 10] rad/yr → T ∈ [0.63, 12.6] yr
LINPPL_BOUNDS = [
    (-3.0, 1.0), (3.0, 7.0), (0.01, 3.0),
    (0.5, 10.0), (-np.pi, np.pi), (0.01, 2.0),
]
LINPPL_NAMES = ["A", "B", "C", "W_cal", "PHI", "D"]


# ── Fit workers (module-level, picklable) ────────────────────────────────

def _fit_worker(args):
    """Fit one window. Returns dict of param values + sigma + r2.
    Catches BaseException to avoid killing the ProcessPoolExecutor worker."""
    model_name, t_end, t_win, lp_win = args
    try:
        if len(t_win) < 100:
            return _nan_result(model_name, t_end)
        if model_name == "lp1":
            bounds, names, fn = LP1_BOUNDS, LP1_NAMES, lp1_model
        elif model_name == "lp2":
            bounds, names, fn = LP2_BOUNDS, LP2_NAMES, lp2_model
        elif model_name == "lp3":
            bounds, names, fn = LP3_BOUNDS, LP3_NAMES, lp3_model
        elif model_name == "lp4":
            bounds, names, fn = LP4_BOUNDS, LP4_NAMES, lp4_model
        elif model_name == "pl":
            bounds, names, fn = PL_BOUNDS, PL_NAMES, pl_model
        elif model_name == "linppl":
            bounds, names, fn = LINPPL_BOUNDS, LINPPL_NAMES, linppl_model
        else:
            return _nan_result(model_name, t_end)

        t_safe = np.maximum(t_win, 0.1)

        def objective(params):
            pred = fn(t_safe, *params)
            val = float(np.sum((lp_win - pred) ** 2))
            if not np.isfinite(val):
                return 1e20
            return val

        result = differential_evolution(
            objective, bounds,
            maxiter=1500, seed=42, tol=1e-10,
            polish=True, workers=1,
        )
        pred = fn(t_safe, *result.x)
        resid = lp_win - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((lp_win - np.mean(lp_win)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sigma = float(np.std(resid))
        row = {"t_end": t_end}
        for n, v in zip(names, result.x):
            row[n] = float(v)
        row["sigma"] = sigma
        row["r2"] = r2
        return row
    except BaseException as e:
        # Catch absolutely everything (including SystemExit, KeyboardInterrupt)
        # so the worker doesn't die and bring down the whole pool
        print(f"[fit_worker] {model_name} t_end={t_end:.3f} failed: {type(e).__name__}: {e}",
              flush=True)
        return _nan_result(model_name, t_end)


def _nan_result(model_name, t_end):
    if model_name == "lp1":
        names = LP1_NAMES
    elif model_name == "lp2":
        names = LP2_NAMES
    elif model_name == "lp3":
        names = LP3_NAMES
    elif model_name == "lp4":
        names = LP4_NAMES
    elif model_name == "pl":
        names = PL_NAMES
    elif model_name == "linppl":
        names = LINPPL_NAMES
    else:
        names = []
    row = {"t_end": t_end}
    for n in names:
        row[n] = float("nan")
    row["sigma"] = float("nan")
    row["r2"] = float("nan")
    return row


# ── Config driver ────────────────────────────────────────────────────────

def run_config(label, model_name, width_yrs, t_all, lp_all, param_names, n_workers,
               csv_path=None):
    """Run rolling-window fits for one (model, width) config. Returns DataFrame.
    If csv_path given, also saves fit results to CSV.
    """
    t_min = float(t_all.min())
    t_max = float(t_all.max())
    first_end = t_min + width_yrs
    ends = np.arange(first_end, t_max + 1e-9, STEP_YRS)

    # Build per-window args
    args_list = []
    for t_end in ends:
        t_start = t_end - width_yrs
        mask = (t_all >= t_start) & (t_all <= t_end)
        args_list.append((model_name, float(t_end), t_all[mask].copy(), lp_all[mask].copy()))

    print(f"  {label}: {len(args_list)} windows, {n_workers} workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_fit_worker, args_list))
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s ({elapsed/len(args_list):.2f}s/window)")

    df = pd.DataFrame(results)
    df["end_date"] = [GENESIS + pd.Timedelta(days=t * 365.25) for t in df["t_end"]]

    if csv_path:
        # Save all fit results for analysis
        cols = ["end_date", "t_end"] + param_names + ["sigma", "r2"]
        df[cols].to_csv(csv_path, index=False, float_format="%.6f")
        print(f"    Saved {csv_path}")
    return df


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_config(df, param_names, label, width_yrs, out_svg):
    """Plot one config — panel per parameter + sigma + R²."""
    # All params + sigma + r2
    panels = param_names + ["sigma", "r2"]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n),
                              sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")
    if n == 1:
        axes = [axes]

    TITLE_COLOR = "#00d4ff"
    LABEL_COLOR = "#cccccc"
    TICK_COLOR = "#aaaaaa"
    LINE_COLOR = "#FF9F40"  # LPPL family lighter orange
    EVENT_COLOR = "#888888"

    fig.suptitle(f"{label} — Rolling {width_yrs}-year windows, monthly steps",
                 color=TITLE_COLOR, fontsize=14, fontweight="bold")

    for i, p in enumerate(panels):
        ax = axes[i]
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors=TICK_COLOR, labelsize=8)
        ax.plot(df["end_date"], df[p], color=LINE_COLOR, linewidth=1.2)
        ax.set_ylabel(p, color=LABEL_COLOR, fontsize=10, rotation=0,
                      ha="right", va="center", labelpad=20)
        ax.grid(True, alpha=0.15, color="#555555")

        # Regime event markers
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
    return out_svg


# ── HTML assembly ────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LPPL Regime Shift Detection — Quantoshi</title>
<style>
body {{ background:#1a1a2e; color:#cccccc; font-family:system-ui,sans-serif;
       max-width:1300px; margin:0 auto; padding:24px 16px; line-height:1.5; }}
h1 {{ color:#00d4ff; font-size:22px; }}
h2 {{ color:#00d4ff; margin-top:0; }}
a {{ color:#FF9F40; text-decoration:none; }}
a:hover {{ color:#FFD080; text-decoration:underline; }}
nav {{ background:#16213e; padding:12px 16px; border-radius:8px;
       margin-bottom:24px; border:1px solid #555; }}
nav a {{ display:inline-block; margin-right:16px; font-size:14px; }}
hr {{ border:none; border-top:1px solid #444; margin:40px 0; }}
section {{ margin-bottom:32px; }}
img {{ max-width:100%; height:auto; display:block; border-radius:6px;
       margin-top:12px; }}
.back-link {{ display:inline-block; margin-top:24px; color:#888; }}
.muted {{ color:#888; font-size:12px; }}
.formula {{ background:#0e1624; padding:12px 18px; border-radius:8px;
            border-left:3px solid #FF9F40; margin:12px 0;
            font-size:13px; overflow-x:auto; }}
</style>
</head>
<body>
<h1>LPPL Regime Shift Detection</h1>
<p class="muted">
Rolling-window LPPL fits tracking parameter evolution over time.
Generated {timestamp} by <code>tools/regime_shift_all.py</code>.
Regenerate manually — not auto-refreshed.
</p>
<nav>
<strong>Jump to:</strong>
<a href="#pl-5yr">PL (5yr)</a>
<a href="#lp1-5yr">LPPL\u2081 (5yr)</a>
<a href="#lp2-5yr">LPPL\u2082 (5yr)</a>
<a href="#lp3-7yr">LPPL\u2083 (7yr)</a>
<a href="#lp3-9yr">LPPL\u2083 (9yr)</a>
<a href="#lp4-7yr">LPPL\u2084 (7yr)</a>
<a href="#lp4-9yr">LPPL\u2084 (9yr)</a>
<a href="#linppl-5yr">LinPPL (5yr)</a>
</nav>
{sections}
<a href="/" class="back-link">\u2190 Back to Quantoshi</a>
</body>
</html>
"""

SECTION_TEMPLATE = """<section id="{anchor}">
<h2>{title}</h2>
<div class="formula">{formula}</div>
<p class="muted">{subtitle}</p>
<img src="/regime_shift/{svg_name}" alt="{title}">
</section>
<hr>"""


def build_html(configs_info, timestamp):
    """Assemble final HTML with anchor-linked sections."""
    sections = []
    for info in configs_info:
        section = SECTION_TEMPLATE.format(
            anchor=info["anchor"],
            title=info["title"],
            subtitle=info["subtitle"],
            formula=info["formula"],
            svg_name=info["svg_name"],
        )
        sections.append(section)
    return HTML_TEMPLATE.format(
        timestamp=timestamp,
        sections="\n".join(sections),
    )


# Model formulae (Unicode) — embedded in each /E section
LP1_FORMULA = (
    "log\u2081\u2080(price) = A + B\u00b7log\u2081\u2080(t) "
    "+ C\u00b7t\u207b\u1d30\u00b7cos(\u03c9\u00b7ln t + \u03c6)"
)
LP2_FORMULA = (
    "log\u2081\u2080(price) = A + B\u00b7log\u2081\u2080(t) "
    "+ C\u2081\u00b7t\u207b\u1d30\u00b7cos(\u03c9\u2081\u00b7ln t + \u03c6\u2081) "
    "+ C\u2082\u00b7cos(\u03c9\u2082\u00b7ln t + \u03c6\u2082)"
)
LP3_FORMULA = (
    "log\u2081\u2080(price) = A + B\u00b7log\u2081\u2080(t) "
    "+ C\u2081\u00b7t\u207b\u1d30\u00b7cos(\u03c9\u2081\u00b7ln t + \u03c6\u2081) "
    "+ C\u2082\u00b7cos(\u03c9\u2082\u00b7ln t + \u03c6\u2082) "
    "+ C\u2083\u00b7cos(\u03c9\u2083\u00b7ln t + \u03c6\u2083)"
)
LP4_FORMULA = (
    "log\u2081\u2080(price) = A + B\u00b7log\u2081\u2080(t) "
    "+ C\u2081\u00b7t\u207b\u1d30\u00b7cos(\u03c9\u2081\u00b7ln t + \u03c6\u2081) "
    "+ C\u2082\u00b7cos(\u03c9\u2082\u00b7ln t + \u03c6\u2082) "
    "+ C\u2083\u00b7cos(\u03c9\u2083\u00b7ln t + \u03c6\u2083) "
    "+ C\u2084\u00b7cos(\u03c9\u2084\u00b7ln t + \u03c6\u2084)"
)
PL_FORMULA = "log\u2081\u2080(price) = A + B\u00b7log\u2081\u2080(t)"
LINPPL_FORMULA = (
    "log\u2081\u2080(price) = A + B\u00b7log\u2081\u2080(t) "
    "+ C\u00b7t\u207b\u1d30\u00b7cos(\u03c9_cal\u00b7t + \u03c6)    "
    "[oscillation in calendar time, not log-time]"
)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    html_only = "--html-only" in sys.argv
    # Skip configs whose output SVG already exists (for resuming failed runs)
    skip_existing = "--skip-existing" in sys.argv

    print("=" * 60)
    print("Rolling-window LPPL regime shift detection — ALL models")
    if html_only:
        print("(HTML-only mode: skipping fits, regenerating HTML only)")
    elif skip_existing:
        print("(Skip-existing mode: configs with existing SVG are skipped)")
    print("=" * 60)

    if not html_only:
        print("Loading Bitcoin prices...")
        pd_ = load_prices("BitcoinPricesDaily.csv")
        df = pd_.df_full[["date", "years", "log_price"]].copy()
        df = df[df["years"] >= 1.0].reset_index(drop=True)
        t_all = df["years"].values
        lp_all = df["log_price"].values
        print(f"  {len(df)} daily rows (years >= 1.0)")

        n_workers = max(1, os.cpu_count() - 1)
        print(f"  Using {n_workers} workers\n")

    configs = [
        ("Power Law (2 params, 5yr windows)", "pl", 5.0, PL_NAMES,
         "pl-5yr", "regime_shift_pl_5yr.svg", PL_FORMULA,
         "Pure power law log-log OLS (no oscillation). Tracks how the slope B "
         "evolves over time \u2014 the cleanest regime-change signal for growth "
         "rate. B climbing = faster-than-power-law growth; B falling = slowing."),
        ("LPPL\u2081 (6 params, 5yr windows)", "lp1", 5.0, LP1_NAMES,
         "lp1-5yr", "regime_shift_lp1_5yr.svg", LP1_FORMULA,
         "Single damped log-periodic oscillation. W hits upper bound (15) "
         "from ~2020 onward \u2014 a signature of Bitcoin's cycle stretching."),
        ("LPPL\u2082 (9 params, 5yr windows)", "lp2", 5.0, LP2_NAMES,
         "lp2-5yr", "regime_shift_lp2_5yr.svg", LP2_FORMULA,
         "Damped primary + undamped secondary. Track how W\u2082 evolves \u2014 "
         "flips between ~9 and ~21 depending on which regime dominates the window."),
        ("LPPL\u2083 (12 params, 7yr windows)", "lp3", 7.0, LP3_NAMES,
         "lp3-7yr", "regime_shift_lp3_7yr.svg", LP3_FORMULA,
         "Three frequencies in a 7-year window. Enough data for primary + "
         "secondary \u03c9\u224821 cycle identification; tight on \u03c9\u22489."),
        ("LPPL\u2083 (12 params, 9yr windows)", "lp3", 9.0, LP3_NAMES,
         "lp3-9yr", "regime_shift_lp3_9yr.svg", LP3_FORMULA,
         "Same model as above, wider 9-year windows. More stable fits at "
         "the cost of slower regime-change response."),
        ("LPPL\u2084 (15 params, 7yr windows) \u2014 likely overfit",
         "lp4", 7.0, LP4_NAMES,
         "lp4-7yr", "regime_shift_lp4_7yr.svg", LP4_FORMULA,
         "Four frequencies. LPPL\u2084's 4th frequency is likely an "
         "intermodulation artifact \u2014 watch for erratic jumps "
         "window-to-window in any of the W\u2082/W\u2083/W\u2084 panels. "
         "R\u00b2 improvements over LPPL\u2083 may be cosmetic (overfitting)."),
        ("LPPL\u2084 (15 params, 9yr windows) \u2014 likely overfit",
         "lp4", 9.0, LP4_NAMES,
         "lp4-9yr", "regime_shift_lp4_9yr.svg", LP4_FORMULA,
         "Same caveats as LPPL\u2084 7yr. Wider windows give more data per "
         "fit, reducing (but not eliminating) noise-fitting of the 4th "
         "frequency."),
        ("LinPPL (6 params, 5yr windows)", "linppl", 5.0, LINPPL_NAMES,
         "linppl-5yr", "regime_shift_linppl_5yr.svg", LINPPL_FORMULA,
         "Linear-periodic variant: oscillation in CALENDAR time (W_cal\u00b7t) "
         "rather than log-time. Period T=2\u03c0/W_cal stays constant in years "
         "\u2014 designed to match Bitcoin's ~4-year halving cycle directly. "
         "Expect T to track the halving cycle more stably than LPPL\u2081's W."),
    ]

    configs_info = []
    for label, model_name, width, names, anchor, svg_name, formula, subtitle in configs:
        if not html_only:
            if skip_existing and os.path.exists(svg_name):
                print(f"  Skipping {label} (SVG exists)")
            else:
                csv_path = svg_name.replace(".svg", ".csv")
                svg_df = run_config(label, model_name, width, t_all, lp_all,
                                    names, n_workers, csv_path=csv_path)
                plot_config(svg_df, names, label, width, svg_name)
                print(f"  Saved {svg_name}")
        configs_info.append({
            "anchor": anchor, "title": label, "subtitle": subtitle,
            "svg_name": svg_name, "formula": formula,
        })

    # Build HTML
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(configs_info, timestamp)
    html_path = "regime_shift_all.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nSaved {html_path}")
    print("\nDone. Commit outputs and deploy to serve at /E.")


if __name__ == "__main__":
    main()
