#!/usr/bin/env python3
"""Fit the Greedy Select model (v3) — pick 5 oscillatory terms by
forward-greedy BIC minimisation from a 36-entry dictionary.

Dictionary:
  * 3 log frequencies (from the best-fit LPPL₃ model)
  * 3 calendar frequencies (from a new 3-freq calendar-space DE fit)
  * Each paired with 3 damping flavours:
      - undamped      : bare cos / sin
      - hybrid-damped : t^(-D) · cos / sin              (D from LPPL₃)
      - entropy-damped: E(w_e · t) · cos / sin          (w_e user grid)
  * Each paired with 2 phase parts (sin, cos)

Support: log10(price) = α + β·log₁₀(t) (always included)

Modes:
  --mode=grid  : freqs frozen at the chosen 3+3 grid values
  --mode=de    : grid provides seed; DE refines freqs + damping params

Usage:
    btc_venv/bin/python3 tools/fit_grdy.py                    # grid only, print
    btc_venv/bin/python3 tools/fit_grdy.py --mode=de          # DE refine, print
    btc_venv/bin/python3 tools/fit_grdy.py --compare          # grid + DE, compare
    btc_venv/bin/python3 tools/fit_grdy.py --mode=de --update # DE refine + update btc_core/
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import scipy.optimize as sopt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices

N_SELECTED = 5          # greedy picks this many oscillatory terms
W_E_DEFAULT = 0.10      # entropy envelope parameter (fixed in grid mode)


# ────────────────────────────────────────────────────────────────────────
# Basis term construction
# ────────────────────────────────────────────────────────────────────────

def _entropy_env(t, w):
    """Shannon entropy envelope used by EPPL / Greedy v2."""
    x = w * t
    raw = -x * np.log(np.maximum(x, 1e-30))
    return np.maximum(raw, 0.0) / (1.0 / np.e)


def _eval_term(t, space, damping, freq, phase, d_param):
    """Evaluate a single basis term at t. Returns array same shape as t.

    space    : "log" or "cal"
    damping  : "none" | "hybrid" | "entropy"
    freq     : angular frequency (radians per ln(t) for log, per year for cal)
    phase    : "sin" or "cos"
    d_param  : None for undamped; D for hybrid (t^-D); w_e for entropy
    """
    t = np.asarray(t, float)
    ts = np.maximum(t, 0.1)
    if space == "log":
        arg = freq * np.log(ts)
    else:
        arg = freq * ts
    osc = np.sin(arg) if phase == "sin" else np.cos(arg)
    if damping == "none":
        env = 1.0
    elif damping == "hybrid":
        env = ts ** (-d_param)
    else:  # entropy
        env = _entropy_env(ts, d_param)
    return env * osc


def _build_dictionary(t, log_freqs, cal_freqs, D_hybrid, w_e):
    """Build a list of (term_spec, column) for all 36 candidates.

    Returns list of (spec, column_array). spec is a dict suitable for
    storage in the model class.
    """
    dictionary = []
    for (space, freqs) in (("log", log_freqs), ("cal", cal_freqs)):
        for freq in freqs:
            for damping in ("none", "hybrid", "entropy"):
                d_param = (None if damping == "none"
                           else (D_hybrid if damping == "hybrid" else w_e))
                for phase in ("sin", "cos"):
                    spec = {
                        "space": space, "damping": damping, "freq": freq,
                        "phase": phase, "d_param": d_param,
                    }
                    col = _eval_term(t, space, damping, freq, phase, d_param)
                    dictionary.append((spec, col))
    return dictionary


# ────────────────────────────────────────────────────────────────────────
# OLS + BIC helpers
# ────────────────────────────────────────────────────────────────────────

def _ols_fit(design, y):
    """Return (beta, residuals, rss)."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    return beta, resid, rss


def _bic(rss, n, k):
    """Schwarz Bayesian Information Criterion (log-likelihood-based)."""
    if rss <= 0:
        rss = 1e-300
    return n * np.log(rss / n) + k * np.log(n)


# ────────────────────────────────────────────────────────────────────────
# 3-freq calendar-space seed fit (for the cal-freq grid)
# ────────────────────────────────────────────────────────────────────────

def fit_three_cal_freqs(t, log_p):
    """Fit log10(p) = A + B·log10(t) + Σᵢ C_i·cos(w_i·t + φ_i)
    for i=1..3 via DE, return the 3 cal frequencies sorted ascending."""
    def _model(params, tt):
        A, B, C1, w1, P1, C2, w2, P2, C3, w3, P3 = params
        lp = np.log10(np.maximum(tt, 0.1))
        return (A + B * lp
                + C1 * np.cos(w1 * tt + P1)
                + C2 * np.cos(w2 * tt + P2)
                + C3 * np.cos(w3 * tt + P3))

    def _obj(params):
        pred = _model(params, t)
        return float(np.sum((log_p - pred) ** 2))

    # Bounds: let DE find the 3 strongest freqs. Seed around BTC's known
    # halving harmonics (~1.88 rad/yr) and sub-harmonics.
    bounds = [
        (-3.0, 3.0),   # A
        (2.0, 8.0),    # B
        (0.01, 1.5),   # C1
        (0.3, 8.0),    # w1 (wide: 0.3-8 rad/yr ≈ 20yr to 9-month period)
        (-np.pi, np.pi), # P1
        (0.01, 1.5),
        (0.3, 8.0),
        (-np.pi, np.pi),
        (0.01, 1.5),
        (0.3, 8.0),
        (-np.pi, np.pi),
    ]
    print("  Running 3-freq cal DE fit (may take ~30s)...")
    res = sopt.differential_evolution(_obj, bounds, maxiter=3000, seed=42,
                                      tol=1e-12, polish=True, popsize=25,
                                      workers=1)
    freqs = sorted([res.x[3], res.x[6], res.x[9]])
    return freqs, res.fun


# ────────────────────────────────────────────────────────────────────────
# Forward-greedy BIC selection
# ────────────────────────────────────────────────────────────────────────

def greedy_select(t, log_p, dictionary, n_select=N_SELECTED):
    """Pick n_select terms from dictionary by forward-greedy BIC.

    Baseline design matrix: [1, log10(t)]  (α, β).
    At each step we try appending each unselected candidate column; keep
    the one that yields the lowest BIC on the joint re-fit.
    """
    n = len(t)
    lp_arr = np.log10(np.maximum(t, 0.1))
    baseline = np.column_stack([np.ones_like(t), lp_arr])

    selected = []            # list of dict specs
    selected_cols = []       # corresponding columns
    remaining = list(range(len(dictionary)))

    # Baseline fit
    beta, resid, rss = _ols_fit(baseline, log_p)
    cur_bic = _bic(rss, n, baseline.shape[1])
    print(f"  Baseline  k=2 BIC={cur_bic:12.2f}  "
          f"α={beta[0]:.4f}  β={beta[1]:.4f}  "
          f"R²={1 - rss / np.sum((log_p - log_p.mean()) ** 2):.4f}")

    for step in range(n_select):
        best_idx = None
        best_bic = cur_bic
        best_beta = None
        best_rss = None
        design_sel = (baseline if not selected_cols
                      else np.column_stack([baseline] + selected_cols))
        for i in remaining:
            cand_col = dictionary[i][1]
            design = np.column_stack([design_sel, cand_col])
            beta, _, rss = _ols_fit(design, log_p)
            b = _bic(rss, n, design.shape[1])
            if b < best_bic:
                best_bic = b
                best_idx = i
                best_beta = beta
                best_rss = rss
        if best_idx is None:
            print(f"  Step {step+1}: no candidate improves BIC; stopping.")
            break
        spec, col = dictionary[best_idx]
        selected.append(spec)
        selected_cols.append(col)
        remaining.remove(best_idx)
        cur_bic = best_bic
        r2 = 1 - best_rss / np.sum((log_p - log_p.mean()) ** 2)
        print(f"  Step {step+1}: +{_spec_repr(spec):40s}  "
              f"BIC={cur_bic:12.2f}  R²={r2:.4f}")

    # Final joint refit
    design = (baseline if not selected_cols
              else np.column_stack([baseline] + selected_cols))
    beta, resid, rss = _ols_fit(design, log_p)
    r2 = 1 - rss / np.sum((log_p - log_p.mean()) ** 2)
    sigma = float(np.sqrt(rss / n))
    alpha, beta_s = float(beta[0]), float(beta[1])
    weights = [float(w) for w in beta[2:]]
    return {
        "alpha": alpha,
        "beta": beta_s,
        "selected": selected,
        "weights": weights,
        "r2": r2,
        "sigma": sigma,
        "bic": cur_bic,
        "n_params": design.shape[1],
    }


def _spec_repr(spec):
    d = spec["damping"][0].upper()  # N/H/E
    return (f"{spec['space']:3s} {d} freq={spec['freq']:6.3f} "
            f"{spec['phase']:3s}")


# ────────────────────────────────────────────────────────────────────────
# DE refinement mode
# ────────────────────────────────────────────────────────────────────────

def de_refine(t, log_p, fit, log_freqs, cal_freqs):
    """Starting from grid fit's selected basis, DE-refine continuous
    parameters: each term's frequency + damping param (if applicable).

    Reuses the same selected (space, damping, phase) types — only the
    continuous parameters are varied.
    """
    n = len(t)
    lp_arr = np.log10(np.maximum(t, 0.1))
    baseline = np.column_stack([np.ones_like(t), lp_arr])

    selected = fit["selected"]
    # Free params: per term, freq + damping_param_if_any
    param_slots = []  # list of (term_idx, "freq" or "d_param", bounds)
    x0 = []
    for i, spec in enumerate(selected):
        # Freq bounds: ±30% around the seed freq
        f_seed = spec["freq"]
        param_slots.append((i, "freq", (f_seed * 0.7, f_seed * 1.3)))
        x0.append(f_seed)
        if spec["damping"] == "hybrid":
            param_slots.append((i, "d_param", (0.05, 5.0)))
            x0.append(spec["d_param"])
        elif spec["damping"] == "entropy":
            param_slots.append((i, "d_param", (0.01, 0.50)))
            x0.append(spec["d_param"])

    def _obj(params):
        # Apply params to selected specs → build cols → OLS fit → rss
        specs_mut = [dict(s) for s in selected]
        for val, (idx, kind, _) in zip(params, param_slots):
            specs_mut[idx][kind] = val
        cols = []
        for s in specs_mut:
            cols.append(_eval_term(
                t, s["space"], s["damping"], s["freq"],
                s["phase"], s["d_param"]))
        design = np.column_stack([baseline] + cols)
        _, _, rss = _ols_fit(design, log_p)
        return rss

    bounds = [b for (_, _, b) in param_slots]
    print("  Running DE refinement (continuous freq + damping search)...")
    res = sopt.differential_evolution(_obj, bounds, maxiter=2000, seed=42,
                                      tol=1e-12, polish=True, popsize=20,
                                      workers=1, x0=x0)
    # Apply refined params
    specs_new = [dict(s) for s in selected]
    for val, (idx, kind, _) in zip(res.x, param_slots):
        specs_new[idx][kind] = val
    cols = [_eval_term(t, s["space"], s["damping"], s["freq"],
                        s["phase"], s["d_param"])
            for s in specs_new]
    design = np.column_stack([baseline] + cols)
    beta, _, rss = _ols_fit(design, log_p)
    r2 = 1 - rss / np.sum((log_p - log_p.mean()) ** 2)
    sigma = float(np.sqrt(rss / n))
    return {
        "alpha": float(beta[0]),
        "beta": float(beta[1]),
        "selected": specs_new,
        "weights": [float(w) for w in beta[2:]],
        "r2": r2,
        "sigma": sigma,
        "bic": _bic(rss, n, design.shape[1]),
        "n_params": design.shape[1],
    }


# ────────────────────────────────────────────────────────────────────────
# Write results into btc_core/_basis.py
# ────────────────────────────────────────────────────────────────────────

def update_basis_file(fit):
    """Rewrite the _GRDY_* class attrs and _BASIS list in _basis.py."""
    import re
    path = os.path.join(ROOT, "btc_core", "_basis.py")
    with open(path) as f:
        src = f.read()

    # Build a Python-source fragment representing the basis. We use a tuple
    # of tuples so the class attribute is immutable-ish and picklable.
    def _spec_tuple(s, w):
        dp = "None" if s["d_param"] is None else f"{s['d_param']:.6f}"
        return (f"        ({s['space']!r}, {s['damping']!r}, "
                f"{s['freq']:.6f}, {s['phase']!r}, "
                f"{w:.6f}, {dp}),")
    lines = ["    _BASIS = ("]
    for s, w in zip(fit["selected"], fit["weights"]):
        lines.append(_spec_tuple(s, w))
    lines.append("    )")
    basis_block = "\n".join(lines)

    # Replace the two sections inside GreedyModel:
    #  1) _alpha / _beta numeric values
    #  2) _BASIS tuple
    def _replace_scalar(src, name, val):
        pat = rf"(    {name}\s*=\s*)[^#\n]+"
        new = f"{val:>16.6f}" if val >= 0 else f"{val:.6f}"
        return re.sub(pat, rf"\g<1>{new}  ", src, count=1)

    src = _replace_scalar(src, "_alpha", fit["alpha"])
    src = _replace_scalar(src, "_beta",  fit["beta"])

    # Replace _BASIS = (...) block. Use non-greedy .*? with DOTALL so we
    # match across the inner tuple-of-tuples without getting stuck on the
    # first closing paren.
    pat_basis = re.compile(
        r"    _BASIS\s*=\s*\(.*?\n    \)", re.DOTALL)
    if not pat_basis.search(src):
        raise RuntimeError(
            "Could not find existing _BASIS block in _basis.py. "
            "Run the class refactor first.")
    src = pat_basis.sub(basis_block, src)

    with open(path, "w") as f:
        f.write(src)
    print(f"  Wrote α={fit['alpha']:.6f}  β={fit['beta']:.6f}  "
          f"and _BASIS ({len(fit['selected'])} terms) to _basis.py")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("grid", "de"), default="grid",
                    help="grid: freqs frozen; de: DE-refine freqs + damping")
    ap.add_argument("--compare", action="store_true",
                    help="Run both modes and print comparison table")
    ap.add_argument("--update", action="store_true",
                    help="Write fitted params into btc_core/_basis.py")
    args = ap.parse_args()

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    log_p = prices.df_full["log_price"].values
    mask = t > 0
    t = t[mask]; log_p = log_p[mask]
    print(f"  n={len(t)} samples, t range [{t.min():.3f}, {t.max():.3f}]")

    # Log frequencies from the already-fitted LPPL₃ model (hardcoded
    # in btc_core/_lppl.py:LPPL3Model). Re-import at runtime so we
    # track any future --update to that model.
    from btc_core import LPPL3Model
    log_freqs = sorted([LPPL3Model._W, LPPL3Model._W2, LPPL3Model._W3])
    D_hybrid = LPPL3Model._D
    print(f"\nLog frequencies (from LPPL₃): {[round(f,3) for f in log_freqs]}")
    print(f"Hybrid damping D (from LPPL₃): {D_hybrid:.4f}")

    # Calendar frequencies from a new 3-freq DE fit.
    print("\n3-freq calendar DE fit...")
    cal_freqs, cal_rss = fit_three_cal_freqs(t, log_p)
    print(f"Cal frequencies (new fit):    {[round(f,3) for f in cal_freqs]}  "
          f"(rss={cal_rss:.3f})")

    print(f"\nEntropy envelope w_e = {W_E_DEFAULT} (fixed in grid mode)")
    print(f"Dictionary size: 6 freqs × 3 dampings × 2 phases = 36 candidates")

    # Build dictionary
    dictionary = _build_dictionary(t, log_freqs, cal_freqs, D_hybrid, W_E_DEFAULT)
    assert len(dictionary) == 36, f"Expected 36 entries, got {len(dictionary)}"

    # Grid fit
    print("\n=== GRID MODE — pick 5 via forward-greedy BIC ===")
    grid_fit = greedy_select(t, log_p, dictionary)

    if args.compare or args.mode == "de":
        print("\n=== DE REFINEMENT — continuous freq + damping from grid seeds ===")
        de_fit = de_refine(t, log_p, grid_fit, log_freqs, cal_freqs)
    else:
        de_fit = None

    # Print comparison
    print("\n" + "=" * 60)
    print(f"{'Mode':<8}{'R²':>8}{'σ':>10}{'BIC':>12}{'# params':>10}")
    print("-" * 60)
    print(f"{'grid':<8}{grid_fit['r2']:>8.4f}{grid_fit['sigma']:>10.4f}"
          f"{grid_fit['bic']:>12.2f}{grid_fit['n_params']:>10}")
    if de_fit is not None:
        print(f"{'de':<8}{de_fit['r2']:>8.4f}{de_fit['sigma']:>10.4f}"
              f"{de_fit['bic']:>12.2f}{de_fit['n_params']:>10}")
    print("=" * 60)

    # Print selected terms for the chosen mode
    chosen = de_fit if args.mode == "de" else grid_fit
    print(f"\nSelected terms ({args.mode.upper()}):")
    print(f"  α={chosen['alpha']:.6f}  β={chosen['beta']:.6f}")
    for i, (s, w) in enumerate(zip(chosen["selected"], chosen["weights"]), 1):
        dp = "" if s["d_param"] is None else f", d_param={s['d_param']:.4f}"
        print(f"  f{i}: space={s['space']}  damping={s['damping']}  "
              f"freq={s['freq']:.4f}  phase={s['phase']}  "
              f"weight={w:.6f}{dp}")

    if args.update:
        print(f"\nUpdating btc_core/_basis.py with {args.mode} fit...")
        update_basis_file(chosen)


if __name__ == "__main__":
    main()
