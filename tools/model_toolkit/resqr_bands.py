"""Residual quantile regression bands with piecewise-linear log-t basis.

Produces per-model, per-quantile coefficient arrays suitable for storage
in model_data.pkl and runtime evaluation via one matrix multiply.

See:
- docs/sigma_bakeoff_report.md (knot choice, divergence analysis)
- docs/sigma_bakeoff_knots_report.md (global (3, 6, 9, 12) validation)
- docs/superpowers/specs/2026-04-15-residual-qr-sigma-bands-design.md
"""
from __future__ import annotations

import logging

import numpy as np

_LOG = logging.getLogger("resqr_bands")

# Frozen by the bake-off — do not change without re-running validation.
DEFAULT_KNOTS = (3.0, 6.0, 9.0, 12.0)
DEFAULT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)

MIN_SAMPLES = 500
OOS_TOLERANCE = 0.05  # ±5pp on interior quantiles
# Tail Q01/Q99 excluded from the OOS assertion due to small-N noise.
INTERIOR_QS = (0.05, 0.25, 0.75, 0.95)


def _basis(t, knots=DEFAULT_KNOTS):
    """Piecewise-linear-in-log10(t) basis with hinges at knot positions.

    Returns a (n_samples, 2 + len(knots)) design matrix. For t ≤ 0 the log10
    clamps to 1e-6 (avoids -inf). The first column is a constant 1; the
    second is log10(t); remaining columns are hinge functions ``relu(log10(t)
    - log10(knot))`` for each knot.
    """
    log_t = np.log10(np.maximum(np.asarray(t, dtype=np.float64), 1e-6))
    cols = [np.ones_like(log_t), log_t]
    for k in knots:
        cols.append(np.maximum(log_t - np.log10(k), 0.0))
    return np.column_stack(cols)


def fit_residual_qr_pwl(t, residuals, quantiles=DEFAULT_QUANTILES,
                         knots=DEFAULT_KNOTS):
    """Fit PWL QR on residuals per quantile. Raises on hard failure.

    Returns
    -------
    sorted_qs : np.ndarray shape (n_q,)
    coef_matrix : np.ndarray shape (n_q, 2 + len(knots))
    coverage : np.ndarray shape (n_q,)
        In-sample empirical coverage (fraction of residuals ≤ fitted Q_p curve).
    raw_crossing_frac : float
        Fraction of (t, q) cells where raw Q_p+1(t) < Q_p(t) pre-sort.
        Diagnostic only; the caller applies monotone sort at query time.

    Raises
    ------
    ValueError
        Too few samples (< MIN_SAMPLES), non-finite residuals, or solver
        non-convergence on any individual quantile. Callers treat this as
        Policy A (per-model skip).
    RuntimeError
        QuantReg returned non-finite coefficients from an otherwise-successful
        fit. Callers treat this as Policy B (global abort trigger).
    """
    import statsmodels.api as sm

    t = np.asarray(t, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    if not np.isfinite(residuals).all():
        n_nans = int((~np.isfinite(residuals)).sum())
        raise ValueError(f"residuals contain {n_nans} non-finite values")

    mask = t > 1.0  # match existing training convention
    t_fit = t[mask]
    r_fit = residuals[mask]
    if len(t_fit) < MIN_SAMPLES:
        raise ValueError(
            f"only {len(t_fit)} samples after t>1 filter (need ≥{MIN_SAMPLES})"
        )

    sorted_qs = np.array(sorted(quantiles), dtype=np.float64)
    X = _basis(t_fit, knots)
    n_basis = X.shape[1]
    coef_matrix = np.zeros((len(sorted_qs), n_basis), dtype=np.float64)
    coverage = np.zeros(len(sorted_qs), dtype=np.float64)

    for i, q in enumerate(sorted_qs):
        try:
            res = sm.QuantReg(r_fit, X).fit(q=float(q), max_iter=10000)
        except Exception as exc:
            raise ValueError(
                f"QuantReg solver failed at q={float(q)}: {exc}"
            ) from exc
        coefs = np.asarray(res.params, dtype=np.float64)
        if not np.isfinite(coefs).all():
            raise RuntimeError(
                f"q={float(q)} returned non-finite coefs: {coefs.tolist()}"
            )
        coef_matrix[i] = coefs
        pred = X @ coefs
        coverage[i] = float((r_fit <= pred).mean())

    # Raw crossing diagnostic (pre-sort) — informational only.
    all_preds = X @ coef_matrix.T  # shape (n_t_fit, n_q)
    diffs = np.diff(all_preds, axis=1)
    n_cells = diffs.size
    raw_crossing_frac = float((diffs < 0).sum() / n_cells) if n_cells else 0.0

    return sorted_qs, coef_matrix, coverage, raw_crossing_frac


def eval_resqr_offsets(t, sorted_qs, coef_matrix, knots=DEFAULT_KNOTS):
    """Return log10-offsets-from-median as a (n_t, n_q) matrix.

    Query-time clipping at the last knot ensures the basis is evaluated at
    ``t = knots[-1]`` for all ``t ≥ knots[-1]``. Combined with the flat
    terminal-segment property, this produces a constant-log-offset plateau
    past the last knot — no extrapolation drift, no crossings in the
    extrapolation region by construction.

    The monotone sort across q is kept as a belt-and-suspenders correction
    for numerical wiggle in the interior (build-time raw crossing diagnostic
    fires at >5% crossings and logs a warning).
    """
    t_arr = np.asarray(t, dtype=np.float64)
    t_clipped = np.minimum(t_arr, float(knots[-1]))
    X = _basis(t_clipped, knots)
    offsets = X @ coef_matrix.T      # (n_t, n_q)
    offsets = np.sort(offsets, axis=1)
    return offsets


def fit_and_validate(t, residuals, model_key,
                      quantiles=DEFAULT_QUANTILES, knots=DEFAULT_KNOTS,
                      oos_tolerance=OOS_TOLERANCE):
    """80/20 random holdout fit + OOS coverage assertion.

    Raises ValueError (Policy A — per-model skip) on:
    - pre-flight failures (propagated from fit_residual_qr_pwl)
    - OOS coverage deviation > oos_tolerance at any interior quantile

    Raises RuntimeError (Policy B — global abort trigger) on:
    - non-finite coefficients (propagated from fit_residual_qr_pwl)

    Returns a dict with sorted_qs, coef_matrix (fit on ALL data),
    in_sample_coverage, oos_coverage, raw_crossing_frac, n_samples.
    """
    t = np.asarray(t, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    rng = np.random.default_rng(42)
    n = len(t)
    if n < MIN_SAMPLES * 2:
        raise ValueError(
            f"only {n} total samples (need ≥{2 * MIN_SAMPLES} for holdout)"
        )

    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    # Train
    _, train_coefs, _, raw_crossings_train = fit_residual_qr_pwl(
        t[train_idx], residuals[train_idx], quantiles, knots,
    )

    # OOS evaluate
    X_te = _basis(t[test_idx], knots)
    pred_te = X_te @ train_coefs.T  # (n_test, n_q)
    pred_te = np.sort(pred_te, axis=1)
    oos_cov = (residuals[test_idx][:, None] <= pred_te).mean(axis=0)

    sorted_qs_arr = np.array(sorted(quantiles), dtype=np.float64)
    for i, q in enumerate(sorted_qs_arr):
        if float(q) in INTERIOR_QS:
            err = abs(oos_cov[i] - float(q))
            if err > oos_tolerance:
                raise ValueError(
                    f"{model_key} q={float(q)}: OOS coverage "
                    f"{oos_cov[i]:.3f} deviates from nominal by "
                    f"{err:.3f} (>{oos_tolerance})"
                )

    if raw_crossings_train > 0.05:
        _LOG.warning(
            "%s raw quantile crossings (train): %.1f%%",
            model_key, raw_crossings_train * 100,
        )

    # Refit on full data for the stored coefficients.
    sorted_qs, coef_matrix, in_cov, raw_crossings_full = fit_residual_qr_pwl(
        t, residuals, quantiles, knots,
    )

    return {
        "model_key": model_key,
        "sorted_qs": sorted_qs,
        "coef_matrix": coef_matrix,
        "in_sample_coverage": in_cov,
        "oos_coverage": oos_cov,
        "raw_crossing_frac": raw_crossings_full,
        "n_samples": int(n),
    }
