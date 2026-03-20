#!/usr/bin/env python3
"""
Build the Empirical Floor (EF) model data pickle.

Uses the same bubble fitting pipeline as SP.ipynb cell 0, but with a hardcoded
support line instead of fitting one from data.  The support line is the
"Empirical Floor" — a two-point power law through 2010-10-05 ($0.06) and
2026-02-09 ($70,339).

Output: btc_app/model_data_ef.pkl

Usage:
    btc_venv/bin/python3 tools/build_ef_model.py
"""

import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Suppress matplotlib GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

# Use the notebook backup that has the 2011 bubble and correct FIT_MIN_DATE
NB_PATH = REPO / "SP.ipynb.2026-03-20_1059.bak"
if not NB_PATH.exists():
    NB_PATH = REPO / "SP.ipynb"

OUT_DIR = REPO / "btc_app"
OUT_PKL = OUT_DIR / "model_data_ef.pkl"

# ── Empirical Floor constants ────────────────────────────────────────────────
EF_SLOPE     = 5.3106
EF_INTERCEPT = -1.6246

# ── Standard quantile list ───────────────────────────────────────────────────
QR_QUANTILES = [
    0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999,
]


def _extract_code():
    """
    Extract cell 0 source from the notebook, from the first '# =' line
    (after the IPython magic block) through the end of STEP 6 and the
    composite_future generation.

    We need everything through bm_composite_future (line ~996) but NOT
    the plotting code that follows.
    """
    with open(NB_PATH) as f:
        nb = json.load(f)
    src = ''.join(nb['cells'][0]['source']) if isinstance(
        nb['cells'][0]['source'], list) else nb['cells'][0]['source']
    lines = src.split('\n')

    # Find where real code starts (after the try/except IPython block)
    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith('# =') and i > 3:
            start_line = i
            break

    # Find the end: after bm_composite_future = 10 ** (...)
    # We want to include through the line that sets bm_composite_future
    end_line = 0
    for i, line in enumerate(lines):
        if 'bm_composite_future' in line and '=' in line and '10 **' in line:
            end_line = i

    # Include a couple lines past bm_composite_future for any trailing code
    end_line += 1

    kept = lines[start_line:end_line + 1]
    return '\n'.join(kept)


def _override_support(code):
    """
    Replace the fitted support line with hardcoded EF values.

    The notebook computes intercept_sup and slope_sup from a quantile
    regression.  We replace those assignments with our fixed constants.
    """
    # Replace the two assignment lines
    code = code.replace(
        "intercept_sup = res_sup.params[0]",
        f"intercept_sup = {EF_INTERCEPT}"
    )
    code = code.replace(
        "slope_sup     = res_sup.params[1]",
        f"slope_sup     = {EF_SLOPE}"
    )

    # Also need to skip the actual QuantReg fitting since we don't need it
    # and it would set res_sup which we're not using.
    # The fitting block is:
    #   X_support = sm.add_constant(...)
    #   res_sup   = sm.QuantReg(...).fit(...)
    # We replace these with no-ops (they run but we override their results)
    # Actually, let them run — the override comes after.
    # But we need to make sure A_sup and B_sup are recalculated from our values.
    # They are: A_sup = 10 ** intercept_sup, B_sup = slope_sup — these come
    # AFTER our override, so they'll use our values. Good.

    return code


def main():
    print("=" * 70)
    print("Building Empirical Floor model pickle")
    print("=" * 70)
    print(f"  Notebook:  {NB_PATH}")
    print(f"  Output:    {OUT_PKL}")
    print(f"  EF slope:  {EF_SLOPE}")
    print(f"  EF intercept: {EF_INTERCEPT}")
    print()

    # Extract and patch code
    code = _extract_code()
    code = _override_support(code)
    n_lines = len(code.split('\n'))
    print(f"  Extracted {n_lines} lines of fitting code")

    # Suppress stdout from the exec'd code
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    namespace = {"__name__": "__cell0__"}
    try:
        exec(code, namespace)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"ERROR during exec: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        sys.stdout = old_stdout
        plt.close('all')

    # ── Extract results from namespace ───────────────────────────────────────
    intercept_sup     = namespace['intercept_sup']
    slope_sup         = namespace['slope_sup']
    B_sup             = namespace['B_sup']
    A_sup             = namespace['A_sup']
    years_plot_bm     = namespace['years_plot_bm']
    log_support_plot  = namespace['log_support_plot_bm']
    support_plot_bm   = namespace['support_plot_bm']
    bm_r2_comp        = namespace['bm_r2_comp']
    bm_r2_support     = namespace['bm_r2_support']
    bm_major          = namespace['bm_major']
    bm_minor          = namespace['bm_minor']
    bm_future_major   = namespace['bm_future_major']
    bm_future_minor   = namespace.get('bm_future_minor', [])
    bm_composite_future = namespace['bm_composite_future']
    years_all         = namespace['years_all']
    y_all             = namespace['y_all']
    log_p_all         = namespace['log_p_all']
    log_support_all   = namespace['log_support_all']
    bm_total_plot     = namespace['bm_total_plot']
    _all_fitted       = namespace['_all_fitted']
    _hist_K_max       = namespace['_hist_K_max']

    # bubble_shape function for comp_by_n computation
    bubble_shape      = namespace['bubble_shape']
    bm_total_bubble   = namespace['bm_total_bubble']
    CAP_COMPOSITE_OVERLAP = namespace['CAP_COMPOSITE_OVERLAP']

    # ── Compute sigma (residual std in log10 space) ──────────────────────────
    # Interpolate composite at data points
    composite_at_data = namespace.get('composite_all_bm')
    if composite_at_data is not None:
        sigma = float(np.std(log_p_all - composite_at_data))
    else:
        # Fallback: compute from support + bubbles at data points
        total_at_data = bm_total_bubble(
            years_all, _all_fitted,
            np.full(len(years_all), _hist_K_max) if CAP_COMPOSITE_OVERLAP else None
        )
        composite_at_data = log_support_all + total_at_data
        sigma = float(np.std(log_p_all - composite_at_data))

    # ── Compute comp_by_n (composite for N = 0..max future bubbles) ──────────
    fut_all = sorted(bm_future_major + bm_future_minor,
                     key=lambda b: b['t_rise'])
    max_n = len(fut_all)

    comp_by_n = np.zeros((max_n + 1, len(years_plot_bm)))

    for n in range(max_n + 1):
        subset = fut_all[:n]
        if CAP_COMPOSITE_OVERLAP:
            budget = np.full(len(years_plot_bm), _hist_K_max)
            for bp in _all_fitted:
                c = bubble_shape(years_plot_bm, bp['t_rise'], bp['r'],
                                 bp['t_plateau'], bp['t_decay'], bp['d'],
                                 bp.get('plat_pow', 0.0))
                budget = np.maximum(0.0, budget - np.minimum(c, budget))
            tot = bm_total_plot.copy()
            rem = budget.copy()
            for fb in subset:
                c = bubble_shape(years_plot_bm, fb['t_rise'], fb['r'],
                                 fb['t_plateau'], fb['t_decay'], fb['d'],
                                 fb.get('plat_pow', 0.0))
                cap = np.minimum(c, rem)
                tot = tot + cap
                rem = np.maximum(0.0, rem - cap)
        else:
            tot = bm_total_plot.copy()
            for fb in subset:
                tot += bubble_shape(years_plot_bm, fb['t_rise'], fb['r'],
                                    fb['t_plateau'], fb['t_decay'], fb['d'],
                                    fb.get('plat_pow', 0.0))
        comp_by_n[n] = 10.0 ** (log_support_plot + tot)

    # ── Collect fitted bubble parameters ─────────────────────────────────────
    fitted_params = []
    for b in sorted(bm_major + bm_minor, key=lambda b: b['t_rise']):
        fitted_params.append({
            't_rise': b['t_rise'],
            'r': b['r'],
            't_plateau': b['t_plateau'],
            't_decay': b['t_decay'],
            'd': b['d'],
            'K': b['K'],
            'plat_pow': b.get('plat_pow', 0.0),
            'dur_rise': b.get('dur_rise', 0.0),
            'dur_plateau': b.get('dur_plateau', 0.0),
        })

    # ── Build pkl dict ───────────────────────────────────────────────────────
    model = {
        "ef_support_slope":     EF_SLOPE,
        "ef_support_intercept": EF_INTERCEPT,
        "genesis":              "2009-07-25",
        "years_plot":           years_plot_bm.tolist(),
        "support_plot":         support_plot_bm.tolist(),
        "comp_by_n":            comp_by_n.tolist(),
        "bm_r2":                float(bm_r2_comp),
        "n_future_max":         int(max_n),
        # σ parameters fitted separately by tools/fit_sigma.py
        "price_years":          years_all.tolist(),
        "price_prices":         y_all.tolist(),
        "QR_QUANTILES":         QR_QUANTILES,
        "fitted_bubbles":       fitted_params,
    }

    # ── Write pkl ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(model, f, protocol=4)

    size_kb = os.path.getsize(OUT_PKL) // 1024
    print(f"\n{'=' * 70}")
    print(f"  Wrote: {OUT_PKL}  ({size_kb} KB)")
    print(f"  R² (composite):     {bm_r2_comp:.6f}")
    print(f"  R² (support only):  {bm_r2_support:.6f}")
    print(f"  Sigma (log10):      {sigma:.6f}")
    print(f"  N future max:       {max_n}")
    print(f"  comp_by_n shape:    ({max_n + 1}, {len(years_plot_bm)})")
    print(f"  Fitted bubbles:     {len(fitted_params)} "
          f"({len(bm_major)} major + {len(bm_minor)} minor)")
    print(f"  Future bubbles:     {len(fut_all)} "
          f"({len(bm_future_major)} major + {len(bm_future_minor)} minor)")
    print(f"  Support: price = {A_sup:.4e} × t^{B_sup:.4f}")
    print(f"  EF intercept={intercept_sup}, slope={slope_sup}")
    print(f"{'=' * 70}")

    # Verify pkl is loadable
    with open(OUT_PKL, 'rb') as f:
        check = pickle.load(f)
    assert len(check['years_plot']) == len(years_plot_bm)
    assert len(check['comp_by_n']) == max_n + 1
    assert check['ef_support_slope'] == EF_SLOPE
    print("  Verification: OK")


if __name__ == "__main__":
    main()
