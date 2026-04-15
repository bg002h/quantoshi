"""Unit tests for tools/model_toolkit/resqr_bands.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from tools.model_toolkit import resqr_bands as rb  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Basis construction
# ──────────────────────────────────────────────────────────────────────────

def test_basis_shape_matches_knots_plus_two():
    t = np.linspace(1.0, 20.0, 100)
    X = rb._basis(t, knots=(3.0, 6.0, 9.0, 12.0))
    assert X.shape == (100, 6)  # 1 constant + log_t + 4 hinges


def test_basis_first_column_is_ones():
    t = np.array([1.0, 5.0, 10.0])
    X = rb._basis(t)
    assert np.allclose(X[:, 0], 1.0)


def test_basis_hinge_zero_before_knot():
    t = np.array([1.0, 2.0, 2.9])
    X = rb._basis(t, knots=(3.0, 6.0, 9.0, 12.0))
    assert np.all(X[:, 2] == 0.0)  # log10(t) - log10(3) < 0 for all, clamped


def test_basis_hinge_linear_after_knot():
    t = np.array([3.0, 6.0, 12.0])
    X = rb._basis(t, knots=(3.0, 6.0, 9.0, 12.0))
    expected_k3 = np.array([0.0, np.log10(6) - np.log10(3),
                             np.log10(12) - np.log10(3)])
    assert np.allclose(X[:, 2], expected_k3)


def test_basis_handles_t_at_zero():
    t = np.array([0.0, 0.5, 1.0])
    X = rb._basis(t)
    assert np.isfinite(X).all()  # log10 clamp prevents -inf


def test_basis_single_scalar_t():
    X = rb._basis(np.array([5.0]))
    assert X.shape == (1, 6)


# ──────────────────────────────────────────────────────────────────────────
# fit_residual_qr_pwl
# ──────────────────────────────────────────────────────────────────────────

def _synthetic_residuals(n=2000, seed=0):
    """Generate residuals with known structure: Gaussian noise * σ(t)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(1.0, 16.0, n)
    sigma_of_t = 0.15 * np.exp(-0.05 * t) + 0.08  # mildly decreasing
    noise = rng.normal(0.0, 1.0, n)
    residuals = sigma_of_t * noise
    return t, residuals


def test_fit_returns_correct_shapes():
    t, r = _synthetic_residuals()
    sorted_qs, coef_matrix, coverage, raw_cross = rb.fit_residual_qr_pwl(t, r)
    assert sorted_qs.shape == (8,)
    assert coef_matrix.shape == (8, 6)
    assert coverage.shape == (8,)
    assert 0.0 <= raw_cross <= 1.0


def test_fit_raises_on_nan_residuals():
    t, r = _synthetic_residuals()
    r[100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rb.fit_residual_qr_pwl(t, r)


def test_fit_raises_on_too_few_samples():
    t = np.linspace(1.0, 10.0, 100)  # < MIN_SAMPLES (500)
    r = np.random.default_rng(0).normal(0, 0.1, 100)
    with pytest.raises(ValueError, match="samples"):
        rb.fit_residual_qr_pwl(t, r)


def test_fit_coverage_approximately_nominal_in_sample():
    t, r = _synthetic_residuals(n=3000)
    sorted_qs, coef, coverage, _ = rb.fit_residual_qr_pwl(t, r)
    # In-sample coverage from QuantReg is near-tautological
    for i, q in enumerate(sorted_qs):
        assert abs(coverage[i] - float(q)) < 0.03


def test_fit_finite_coefs_on_clean_data():
    t, r = _synthetic_residuals()
    _, coef_matrix, _, _ = rb.fit_residual_qr_pwl(t, r)
    assert np.isfinite(coef_matrix).all()


# ──────────────────────────────────────────────────────────────────────────
# eval_resqr_offsets
# ──────────────────────────────────────────────────────────────────────────

def test_eval_shape():
    t, r = _synthetic_residuals()
    sorted_qs, coef, _, _ = rb.fit_residual_qr_pwl(t, r)
    offsets = rb.eval_resqr_offsets(np.array([2.0, 8.0, 14.0]), sorted_qs, coef)
    assert offsets.shape == (3, 8)


def test_eval_monotone_across_q_at_each_t():
    t, r = _synthetic_residuals()
    sorted_qs, coef, _, _ = rb.fit_residual_qr_pwl(t, r)
    offsets = rb.eval_resqr_offsets(
        np.linspace(1.0, 15.0, 20), sorted_qs, coef,
    )
    for i in range(offsets.shape[0]):
        row = offsets[i]
        assert np.all(np.diff(row) >= 0), (
            f"non-monotone across q at t_idx={i}: {row}"
        )


def test_eval_clips_at_last_knot():
    """For t > knots[-1], offsets should equal those at t = knots[-1]."""
    t_train, r = _synthetic_residuals()
    sorted_qs, coef, _, _ = rb.fit_residual_qr_pwl(t_train, r)
    at_last_knot = rb.eval_resqr_offsets(
        np.array([12.0]), sorted_qs, coef,
    )
    at_far_future = rb.eval_resqr_offsets(
        np.array([70.0]), sorted_qs, coef,
    )
    assert np.allclose(at_last_knot, at_far_future)


def test_eval_interior_monotone_in_t_at_wide_q():
    """At the tails, the offset grows (in absolute value) vs the median
    in the early era for these synthetic residuals — just a sanity check
    that the fit isn't wildly wrong."""
    t, r = _synthetic_residuals(n=3000)
    sorted_qs, coef, _, _ = rb.fit_residual_qr_pwl(t, r)
    t_eval = np.array([2.0, 5.0, 10.0])
    offsets = rb.eval_resqr_offsets(t_eval, sorted_qs, coef)
    # Q99 should be positive (above median), Q01 should be negative
    q99_idx = int(np.where(sorted_qs == 0.99)[0][0])
    q01_idx = int(np.where(sorted_qs == 0.01)[0][0])
    assert (offsets[:, q99_idx] > 0).all()
    assert (offsets[:, q01_idx] < 0).all()


# ──────────────────────────────────────────────────────────────────────────
# fit_and_validate
# ──────────────────────────────────────────────────────────────────────────

def test_fit_and_validate_succeeds_on_clean_data():
    t, r = _synthetic_residuals(n=5000)
    result = rb.fit_and_validate(t, r, model_key="synthetic")
    assert "sorted_qs" in result
    assert result["sorted_qs"].shape == (8,)
    assert result["coef_matrix"].shape == (8, 6)
    assert result["n_samples"] == 5000


def test_fit_and_validate_raises_on_too_few_samples():
    t = np.linspace(1.0, 10.0, 500)  # < MIN_SAMPLES * 2 = 1000
    r = np.random.default_rng(0).normal(0, 0.1, 500)
    with pytest.raises(ValueError, match="samples"):
        rb.fit_and_validate(t, r, model_key="tiny")


def test_fit_and_validate_oos_coverage_within_tolerance_on_clean_data():
    """Clean synthetic data should produce OOS coverage within ±5pp."""
    t, r = _synthetic_residuals(n=5000)
    result = rb.fit_and_validate(t, r, model_key="synthetic")
    sorted_qs = result["sorted_qs"]
    oos_cov = result["oos_coverage"]
    for i, q in enumerate(sorted_qs):
        if float(q) in rb.INTERIOR_QS:
            err = abs(oos_cov[i] - float(q))
            assert err <= rb.OOS_TOLERANCE, (
                f"q={float(q)} OOS cov {oos_cov[i]:.3f} deviates "
                f"by {err:.3f}"
            )
