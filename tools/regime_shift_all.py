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


def hybppl_model(t_safe, A, B, C1, W_log, PHI1, D, C2, W_cal, PHI2):
    """HybPPL — log-periodic damped + linear-periodic undamped."""
    damped = C1 * t_safe**(-D) * np.cos(W_log * np.log(t_safe) + PHI1)
    undamped = C2 * np.cos(W_cal * t_safe + PHI2)
    return A + B * np.log10(t_safe) + damped + undamped


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

HYBPPL_BOUNDS = [
    (-3.0, 1.0), (3.0, 7.0), (0.01, 3.0),
    (2.0, 40.0), (-np.pi, np.pi), (0.01, 2.0),
    (0.0, 2.0), (0.5, 10.0), (-np.pi, np.pi),
]
HYBPPL_NAMES = ["A", "B", "C1", "W_log", "PHI1", "D", "C2", "W_cal", "PHI2"]


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
        elif model_name == "hybppl":
            bounds, names, fn = HYBPPL_BOUNDS, HYBPPL_NAMES, hybppl_model
        else:
            return _nan_result(model_name, t_end)

        t_safe = np.maximum(t_win, 0.1)

        # PL is a closed-form OLS problem; don't run differential_evolution
        # through a box constraint that clips the true slope at bubble/bear
        # extremes. Use unbounded linear regression on (log10 t, log10 price).
        if model_name == "pl":
            log_t = np.log10(t_safe)
            B_hat, A_hat = np.polyfit(log_t, lp_win, 1)
            pred = A_hat + B_hat * log_t
            resid = lp_win - pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((lp_win - np.mean(lp_win)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            sigma = float(np.std(resid))
            return {"t_end": t_end, "A": float(A_hat), "B": float(B_hat),
                    "sigma": sigma, "r2": r2}

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
    elif model_name == "hybppl":
        names = HYBPPL_NAMES
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
.model-desc {{ background:#101a2e; padding:10px 16px; border-radius:6px;
               border-left:3px solid #00d4ff; margin:8px 0 10px 0;
               font-size:13px; color:#b8ccd8; line-height:1.55; }}
.model-desc strong {{ color:#00d4ff; }}
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
<a href="#pl-6mo">PL (6mo)</a>
<a href="#pl-1yr">PL (1yr)</a>
<a href="#pl-2yr">PL (2yr)</a>
<a href="#pl-5yr">PL (5yr)</a>
<a href="#pl-7yr">PL (7yr)</a>
<a href="#pl-9yr">PL (9yr)</a>
<a href="#pl-2yr-clipped">PL (2yr, clipped)</a>
<a href="#pl-5yr-clipped">PL (5yr, clipped)</a>
<a href="#pl-7yr-clipped">PL (7yr, clipped)</a>
<a href="#pl-9yr-clipped">PL (9yr, clipped)</a>
<a href="#lp1-5yr">LPPL\u2081 (5yr)</a>
<a href="#lp2-5yr">LPPL\u2082 (5yr)</a>
<a href="#lp3-7yr">LPPL\u2083 (7yr)</a>
<a href="#lp3-9yr">LPPL\u2083 (9yr)</a>
<a href="#lp4-7yr">LPPL\u2084 (7yr)</a>
<a href="#lp4-9yr">LPPL\u2084 (9yr)</a>
<a href="#linppl-5yr">LinPPL (5yr)</a>
<a href="#hybppl-5yr">HybPPL (5yr)</a>
<a href="#bm-7yr">BM (7yr)</a>
<a href="#bm-9yr">BM (9yr)</a>
<a href="#pl-6mo-timing">PL 6mo timing</a>
<a href="#pl-1yr-timing">PL 1yr timing</a>
<a href="#pl-2yr-timing">PL 2yr timing</a>
<a href="#predict-bottoms">Predicting bottoms</a>
</nav>
{sections}
<section id="predict-bottoms">
<h2>Using PL 1yr + 2yr to predict cycle bottoms</h2>
<p class="model-desc">
<strong>Signal mechanics.</strong> The N-year rolling PL slope B<sub>N</sub>(t)
is the net log-log growth rate over the window [t\u2212N, t]. It's a trailing
indicator: B<sub>N</sub> peaks when a bull run has <em>just fully entered</em>
the window, then declines as the window fills with the post-peak crash, and
bottoms roughly N years after the bubble top.
</p>
<p class="model-desc">
<strong>Observed cross-correlation.</strong> corr(B<sub>N</sub>, log-excess)
peaks at lag \u2212L months, where L grows with window width but not
linearly. Measured across three widths:
</p>
<ul class="model-desc" style="margin:0; padding-left:24px;">
<li>6mo window: L = 12 months, peak |corr| = 0.49</li>
<li>1yr window: L = 13 months, peak |corr| = 0.66</li>
<li>2yr window: L = 22 months, peak |corr| = 0.80</li>
</ul>
<p class="model-desc">
The peak is negative at each width because B is low <em>after</em> bubble
peaks and high <em>after</em> bear bottoms. For windows shorter than the
typical 12-month crash duration the lag bottoms out near ~12 months (the
crash fills the window before the window width does). For windows wider
than the crash, the lag grows roughly with window width. Correlation
strength grows monotonically with window width because the signal
averages more noise out.
</p>
<p class="model-desc">
<strong>Bottom-prediction recipe.</strong>
</p>
<ol class="model-desc" style="margin:0; padding-left:24px;">
<li><strong>Anchor on the last bubble top.</strong> Identify the most recent
log-excess peak date T<sub>peak</sub> from the /A palette-chart or the QR
bubble overlay.</li>
<li><strong>Expected B<sub>6mo</sub> trough:</strong> T<sub>peak</sub> +
~12 months (fast alert — the half-year window is fully saturated with
crash data by month 12).</li>
<li><strong>Expected B<sub>1yr</sub> trough:</strong> T<sub>peak</sub> +
~13 months. 12 months of crash have replaced 12 months of rally in the
1-year window.</li>
<li><strong>Expected B<sub>2yr</sub> trough:</strong> T<sub>peak</sub> +
~22 months. The wider window captures the full rally + crash + post-crash
chop before bottoming.</li>
<li><strong>Zero-crossings as confirmation.</strong> Watch B<sub>1</sub>
cross zero upward (window's net return turns positive) \u2014 typically
arrives a few months <em>before</em> the actual price bottom and well before
B<sub>1</sub> reaches its trough. B<sub>2</sub> crossing zero is a
stronger, later confirmation (usually in early bull).</li>
<li><strong>Corroboration rule.</strong> When B<sub>1</sub> troughs AND starts
rising, and B<sub>2</sub> is still falling but decelerating, you are in the
late-bear / early-accumulation window. When B<sub>2</sub> troughs and
B<sub>1</sub> is already above zero climbing, the cycle bottom is in the
rear-view mirror.</li>
<li><strong>Asymmetry caveat.</strong> The recipe assumes bear-length
comparable to prior cycles. If the current bear is shorter than the window
width, B<sub>N</sub> can bottom earlier than the formula predicts \u2014 in
that case B<sub>1</sub> leads B<sub>2</sub> by more than 9\u201310 months.
Divergence between 1yr and 2yr trough timings is itself diagnostic of
cycle-length shifts.</li>
</ol>
<p class="muted" style="margin-top:12px;">
Applied to the 2021-11 top (T<sub>peak</sub> \u2248 2021-11): B<sub>1</sub>
trough predicted ~2022-12 (observed ~2022-11, spot-on); B<sub>2</sub> trough
predicted ~2023-09 (observed ~2023-10). Applied to 2024-03 / 2025-01 local
highs, the recipe points to B<sub>1</sub>/B<sub>2</sub> troughs in 2026 and
2026-2027 respectively \u2014 watch the live CSVs for updates.
</p>
</section>
<hr>
<a href="/" class="back-link">\u2190 Back to Quantoshi</a>
</body>
</html>
"""

SECTION_TEMPLATE = """<section id="{anchor}">
<h2>{title}</h2>
<div class="formula">{formula}</div>
<p class="model-desc">{model_desc}</p>
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
            model_desc=info.get("model_desc", ""),
            svg_name=info["svg_name"],
        )
        sections.append(section)
    return HTML_TEMPLATE.format(
        timestamp=timestamp,
        sections="\n".join(sections),
    )


# ── Model descriptions (one per model family) ────────────────────────────
# These describe WHAT the model is. Window-specific notes go in subtitle.

PL_DESC = (
    "<strong>Power Law (PL)</strong> \u2014 the simplest Bitcoin price model. "
    "Pure log-log linear fit: price grows as a fixed power of time since "
    "genesis. Two parameters (intercept A, slope B) and no oscillation. "
    "Sliding-window fits reveal how the effective growth exponent evolves "
    "as different eras of Bitcoin history enter and leave the fit window. "
    "Here we fit by unbounded OLS (closed-form) so the true slope is never "
    "clipped; a centered intercept A' = mean log-price in the window is "
    "also plotted \u2014 A' is orthogonal to B, while raw A is a mechanical "
    "mirror of B (see FAQ on A/B anti-correlation)."
)
PL_CLIPPED_DESC = (
    "<strong>Power Law (clipped)</strong> \u2014 same PL model, but fit via "
    "differential_evolution with a box constraint (A\u2208[-3,1], "
    "B\u2208[3,7]). The bounds reflect the typical range of the long-run PL "
    "over Bitcoin's full history, but in short windows during bubbles/crashes "
    "the true slope can blow past them. Red dotted lines mark the rails \u2014 "
    "watch how often the optimizer parks there. Kept alongside the unbounded "
    "version as a visual demonstration of what constraint-clipping looks like."
)
LP1_DESC = (
    "<strong>LPPL\u2081 (Log-Periodic Power Law, 1 frequency)</strong> \u2014 "
    "Sornette's classic bubble model: power-law trend plus one "
    "log-periodic oscillation whose peaks compress in log-time toward a "
    "finite-time singularity. 6 parameters. The angular frequency "
    "\u03c9 encodes the bubble's discrete scale invariance; historically "
    "\u03c9\u22487 was the canonical value, but modern Bitcoin fits prefer "
    "\u03c9\u224815\u201320 (cycle-stretching). D is the critical exponent."
)
LP2_DESC = (
    "<strong>LPPL\u2082 (2 frequencies)</strong> \u2014 adds a second, "
    "undamped cosine to LPPL\u2081 (9 parameters). The primary remains a "
    "damped log-periodic; the secondary is a constant-amplitude modulation. "
    "Typically captures a faster \u03c9\u2082\u224820 beat on top of the "
    "\u03c9\u22489 primary. Whether \u03c9\u2082 is a genuine cycle or a "
    "harmonic/artifact of the primary is model-dependent \u2014 watch its "
    "stability across windows."
)
LP3_DESC = (
    "<strong>LPPL\u2083 (3 frequencies)</strong> \u2014 three log-periodic "
    "cosines (12 parameters). With carefully separated frequencies this is "
    "the <em>honest</em> LPPL extension: enough degrees of freedom to "
    "represent Bitcoin's observed fast (\u03c9\u22487\u201310), medium "
    "(\u03c9\u224815\u201320), and slow (\u03c9\u224825\u201335) cycle "
    "components, but not so many that you start fitting noise. R\u00b2 on "
    "full history is comparable to LPPL\u2084 with one fewer frequency."
)
LP4_DESC = (
    "<strong>LPPL\u2084 (4 frequencies) \u2014 likely overfit</strong>. "
    "Four damped log-periodic cosines (15 parameters). The 4th frequency is "
    "typically an <em>intermodulation artifact</em> of the first three "
    "(e.g. \u03c9\u2084 \u2248 \u03c9\u2082 \u2212 \u03c9\u2081 or "
    "\u03c9\u2081 + \u03c9\u2083). When you exclude \u03c9\u224813 "
    "manually, the optimizer finds the next intermod product instead. "
    "Included here for completeness \u2014 watch for erratic "
    "window-to-window jumps in W\u2082/W\u2083/W\u2084 as the signature "
    "of overfitting."
)
LINPPL_DESC = (
    "<strong>LinPPL (Linear-Periodic Power Law)</strong> \u2014 our variant: "
    "the oscillation lives in CALENDAR time (cos(\u03c9_cal\u00b7t)) "
    "rather than log-time (cos(\u03c9\u00b7ln t)). 6 parameters. Period "
    "T = 2\u03c0/\u03c9_cal stays constant in years, designed to match "
    "Bitcoin's ~4-year halving cycle directly. Unlike LPPL\u2081, the cycle "
    "does not compress toward a singularity \u2014 it's a steady calendar "
    "metronome on top of the power-law baseline. Full-history R\u00b2 is "
    "close to LPPL\u2081 but with very different interpretation."
)
HYBPPL_DESC = (
    "<strong>HybPPL (Hybrid log+linear PPL)</strong> \u2014 our best-fitting "
    "9-parameter model: combines a damped log-periodic primary "
    "(cos(\u03c9_log\u00b7ln t)) with an undamped linear-periodic secondary "
    "(cos(\u03c9_cal\u00b7t)). Two cleanly separated mechanisms in one "
    "model: one log-time chirp converging toward a singularity and one "
    "calendar metronome tracking halvings. Full-history R\u00b2\u22480.989 "
    "\u2014 ties LPPL\u2083 (12 params) at 9 params, beats LPPL\u2082 "
    "at the same param count. No intermod artifacts \u2014 the two "
    "frequencies live on different time axes so they can't alias each "
    "other."
)
BM_DESC = (
    "<strong>BM (Bubble Model)</strong> \u2014 Quantoshi's production "
    "model: a two-stage support line fit (OLS on all window data \u2192 "
    "bottom-20% residual filter \u2192 quantile regression at Q50% on the "
    "filtered set) combined with bubble-peak characterization at known "
    "halving-cycle years (2011, 2013, 2017, 2021, 2025). Decouples the "
    "secular power-law baseline from the transient bubble amplitudes \u2014 "
    "rolling-window fits track how the baseline slope, residual \u03c3, "
    "and per-cycle peak K amplitudes evolve together. Falling bubble "
    "amplitudes + tightening \u03c3 are consistent with diminishing-returns "
    "cycle dynamics."
)

MODEL_DESCRIPTIONS = {
    "pl": PL_DESC,
    "pl-clipped": PL_CLIPPED_DESC,
    "lp1": LP1_DESC,
    "lp2": LP2_DESC,
    "lp3": LP3_DESC,
    "lp4": LP4_DESC,
    "linppl": LINPPL_DESC,
    "hybppl": HYBPPL_DESC,
    "bm": BM_DESC,
}


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
HYBPPL_FORMULA = (
    "log\u2081\u2080(price) = A + B\u00b7log\u2081\u2080(t) "
    "+ C\u2081\u00b7t\u207b\u1d30\u00b7cos(\u03c9_log\u00b7ln t + \u03c6\u2081) "
    "+ C\u2082\u00b7cos(\u03c9_cal\u00b7t + \u03c6\u2082)    "
    "[log-periodic damped + linear-periodic undamped]"
)
BM_FORMULA = (
    "log\u2081\u2080(price) = A_sup + B_sup\u00b7log\u2081\u2080(t)    "
    "[support line from bottom-20% OLS filter \u2192 QR at Q50; "
    "bubble peaks characterized at known halving years]"
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
        ("Power Law (2 params, 6mo windows)", "pl", 0.5, PL_NAMES,
         "pl-6mo", "regime_shift_pl_6mo.svg", PL_FORMULA,
         "Fastest-reacting PL window \u2014 6 months of data. B is "
         "essentially a noisy 6-month trailing growth rate; raw panel "
         "looks jittery but leads 1yr and 2yr signals by half a window. "
         "Useful for very-early warning of regime shifts. See the "
         "corresponding timing section at the bottom of /E for the "
         "lead/lag structure."),
        ("Power Law (2 params, 1yr windows)", "pl", 1.0, PL_NAMES,
         "pl-1yr", "regime_shift_pl_1yr.svg", PL_FORMULA,
         "Shortest viable PL window \u2014 1 year of data. B reacts almost "
         "in real time to the current trend, so the panel essentially "
         "tracks Bitcoin's 12-month trailing growth rate. Most volatile "
         "of all widths. Pair with the 2yr window for cycle-timing "
         "inferences (see PL 1yr/2yr timing sections at the bottom of /E)."),
        ("Power Law (2 params, 2yr windows)", "pl", 2.0, PL_NAMES,
         "pl-2yr", "regime_shift_pl_2yr.svg", PL_FORMULA,
         "Shortest power-law window \u2014 reacts fastest to regime changes. "
         "B_pl oscillates strongly around bubble peaks and bear bottoms "
         "(slope over-fits to local cycle phase). Use alongside wider "
         "windows to separate transient spikes from secular drift."),
        ("Power Law (2 params, 5yr windows)", "pl", 5.0, PL_NAMES,
         "pl-5yr", "regime_shift_pl_5yr.svg", PL_FORMULA,
         "Pure power law log-log OLS (no oscillation). Tracks how the slope B "
         "evolves over time \u2014 the cleanest regime-change signal for growth "
         "rate. B climbing = faster-than-power-law growth; B falling = slowing."),
        ("Power Law (2 params, 7yr windows)", "pl", 7.0, PL_NAMES,
         "pl-7yr", "regime_shift_pl_7yr.svg", PL_FORMULA,
         "Same pure power-law fit with a 7-year window. Typically spans two "
         "halving cycles \u2014 slope/intercept drift more slowly, giving a "
         "smoother view of secular regime changes."),
        ("Power Law (2 params, 9yr windows)", "pl", 9.0, PL_NAMES,
         "pl-9yr", "regime_shift_pl_9yr.svg", PL_FORMULA,
         "Same pure power-law fit with a 9-year window. Captures 2\u20133 "
         "bubble cycles per window; B trajectory is the most stable of the "
         "three widths but slowest to react to new regimes."),
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
        ("HybPPL (9 params, 5yr windows)", "hybppl", 5.0, HYBPPL_NAMES,
         "hybppl-5yr", "regime_shift_hybppl_5yr.svg", HYBPPL_FORMULA,
         "Combines log-periodic damped primary + linear-periodic undamped "
         "secondary. Full-history R\u00b2=0.989 \u2014 beats LPPL\u2082 at same "
         "9-param count. Watch how the relative contribution of W_log vs W_cal "
         "evolves over windows."),
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
            "model_desc": MODEL_DESCRIPTIONS.get(model_name, ""),
        })

    # Clipped PL sections: historical record showing the box-constrained
    # differential_evolution fits with A∈[-3,1], B∈[3,7] bounds. 2yr window
    # is 75% rail-clipped; wider windows barely touch the bounds. Generated
    # by tools/pl_clipped_fit.py.
    PL_CLIPPED_NOTE = (
        "Original box-bounded DE fit (A\u2208[-3,1], B\u2208[3,7]). "
        "Red dotted lines mark the rails. Useful for seeing exactly where "
        "the optimizer was being clipped by the constraint \u2014 compare "
        "against the unbounded OLS version above for the same window width."
    )
    for width_label, width_svg, pct in (
        ("2yr", "2yr", "75% of windows hit a rail"),
        ("5yr", "5yr", "6% hit a rail"),
        ("7yr", "7yr", "3% hit a rail"),
        ("9yr", "9yr", "0% hit a rail"),
    ):
        configs_info.append({
            "anchor": f"pl-{width_label}-clipped",
            "title": f"Power Law (2 params, {width_label} windows) \u2014 clipped",
            "subtitle": f"{PL_CLIPPED_NOTE} {pct}.",
            "svg_name": f"regime_shift_pl_{width_svg}_clipped.svg",
            "formula": PL_FORMULA,
            "model_desc": MODEL_DESCRIPTIONS["pl-clipped"],
        })

    # BM sections: generated by tools/regime_shift_bm.py (separate fit pipeline,
    # different parameter set). SVGs must be present on disk.
    configs_info.append({
        "anchor": "bm-7yr", "title": "BM (Bubble Model, 7yr windows)",
        "subtitle": (
            "Rolling support-line fit (bottom-20% OLS \u2192 QR at Q50%) + "
            "bubble peak characterization at known halving years. Tracks A_sup, "
            "B_sup, R\u00b2, \u03c3, and per-window bubble amplitude (mean/max K). "
            "7-year windows typically catch 2 bubble peaks. Generated by "
            "tools/regime_shift_bm.py \u2014 run separately to regenerate."
        ),
        "svg_name": "regime_shift_bm_7yr.svg", "formula": BM_FORMULA,
        "model_desc": MODEL_DESCRIPTIONS["bm"],
    })
    configs_info.append({
        "anchor": "bm-9yr", "title": "BM (Bubble Model, 9yr windows)",
        "subtitle": (
            "Same pipeline as BM 7yr with 9-year windows \u2014 usually captures "
            "2\u20133 bubble peaks per window, giving more stable support slopes "
            "at the cost of slower regime-change response."
        ),
        "svg_name": "regime_shift_bm_9yr.svg", "formula": BM_FORMULA,
        "model_desc": MODEL_DESCRIPTIONS["bm"],
    })

    # PL 2yr timing analysis: slope B vs log10(price) twin-axis + lagged
    # cross-correlation. Generated by tools/pl_2yr_timing.py.
    PL_TIMING_DESC = (
        "<strong>PL 2yr timing analysis</strong> \u2014 aligns the 2yr "
        "rolling slope B(t) with log\u2081\u2080(price) at the same window "
        "end, then computes Pearson correlations (zero-lag) and lagged "
        "cross-correlation over \u00b124 months. The B trace is a "
        "trailing 2-year growth rate; its peaks lag price rallies until "
        "the bubble fully enters the window, and subsequent crashes drag "
        "B down for another ~2 years. Key measurements to compare: "
        "peak |xcorr(B, log-excess)| and the lag at which it occurs. "
        "B vs 6-month forward momentum is the most direct timing read \u2014 "
        "strongly-negative means a high 2yr slope predicts near-term "
        "drawdown."
    )
    configs_info.append({
        "anchor": "pl-2yr-timing",
        "title": "PL 2yr timing \u2014 unbounded OLS",
        "subtitle": (
            "Twin-axis B + log\u2081\u2080(price) over time, plus lagged "
            "cross-correlation out to \u00b124 months. Unbounded OLS fit "
            "so B reflects the true window slope (range ~\u22128 to +15). "
            "Header stats: zero-lag correlations of B vs level, log-excess "
            "(detrended), and 6m forward momentum; peak cross-correlation "
            "lag in months (positive lag = B leads price)."
        ),
        "svg_name": "regime_shift_pl_2yr_timing_unbounded.svg",
        "formula": PL_FORMULA,
        "model_desc": PL_TIMING_DESC,
    })
    configs_info.append({
        "anchor": "pl-6mo-timing",
        "title": "PL 6mo timing \u2014 unbounded OLS",
        "subtitle": (
            "6-month rolling slope aligned with log\u2081\u2080(price). "
            "Fastest early-warning signal. Observed peak |xcorr(B, "
            "log-excess)| \u2248 -0.485 at lag \u221212 months \u2014 "
            "the lag bottoms out at ~12 months because it's set by the "
            "typical 12-month crash duration rather than the window "
            "width (for very short windows the crash takes longer to "
            "fill the window than the window itself). Correlation is "
            "noticeably weaker (0.49 vs 0.66/0.80 for 1yr/2yr) because "
            "the 6-month slope is much noisier. Useful for confirming "
            "momentum shifts in real time, not for precise timing."
        ),
        "svg_name": "regime_shift_pl_6mo_timing_unbounded.svg",
        "formula": PL_FORMULA,
        "model_desc": PL_TIMING_DESC,
    })
    configs_info.append({
        "anchor": "pl-1yr-timing",
        "title": "PL 1yr timing \u2014 unbounded OLS",
        "subtitle": (
            "1-year rolling slope aligned with log\u2081\u2080(price). "
            "Faster-reacting than 2yr: peak |xcorr(B, log-excess)| "
            "\u2248 -0.66 at lag -13 months vs -0.80 at lag -22 months "
            "for 2yr. The lag scales roughly with window width "
            "(~1.1\u00d7 the window in months), and correlation strength "
            "grows with window width (2yr is a cleaner signal). "
            "Use 1yr as the fast early-warning and 2yr as confirmation "
            "\u2014 see the 'Predicting cycle bottoms' notes at the "
            "bottom of the 2yr section for the recipe."
        ),
        "svg_name": "regime_shift_pl_1yr_timing_unbounded.svg",
        "formula": PL_FORMULA,
        "model_desc": PL_TIMING_DESC,
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
