"""Runtime tests for the sigma_mode='resqr' dispatch in btc_core.

Covers the three parallel ``price_at`` sites (_ShrinkingBandsMixin,
_FitsBasedModel, _CompositeModel) plus the shared ``_resqr_price_at``
helper: fallback when _resqr is absent, array/scalar shape preservation,
q-interpolation, and flatline-past-last-knot extrapolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import btc_core as bc  # noqa: E402
from tools.model_toolkit.resqr_bands import (  # noqa: E402
    DEFAULT_KNOTS,
    DEFAULT_QUANTILES,
)


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def md():
    return bc.load_model_data(str(_ROOT / "model_data.pkl"))


@pytest.fixture(scope="module")
def bubble_model(md):
    return bc.BubbleModel(md)


@pytest.fixture(scope="module")
def pl_model(md):
    from btc_core import PowerLawModel
    return PowerLawModel(
        md.ols_intercept, md.ols_slope, md.price_years, md.price_prices,
        "2009-07-25", sorted(md.qr_fits.keys()),
    )


@pytest.fixture(scope="module")
def lppl_model(md):
    return bc.LPPLModel(md.price_years, md.price_prices, sorted(md.qr_fits.keys()))


def _fake_resqr_bundle(offset_per_q=None):
    """Build a _resqr dict whose basis coefs yield a known constant offset
    per quantile (coefs are [offset, 0, 0, 0, 0, 0] so X @ c == offset)."""
    qs = np.array(sorted(DEFAULT_QUANTILES), dtype=np.float64)
    n_basis = 2 + len(DEFAULT_KNOTS)
    coefs = np.zeros((len(qs), n_basis), dtype=np.float64)
    if offset_per_q is not None:
        for i, q in enumerate(qs):
            coefs[i, 0] = offset_per_q(float(q))
    return {
        "sorted_qs": qs,
        "coef_matrix": coefs,
        "knots": DEFAULT_KNOTS,
    }


# ── _ShrinkingBandsMixin (LPPL family) ──────────────────────────────────────

def test_shrinking_fallback_when_no_resqr(lppl_model):
    # No _resqr attribute → resqr mode should silently fall back to constant.
    assert not hasattr(lppl_model, "_resqr")
    p_constant = lppl_model.price_at(0.5, 10.0)
    p_resqr = lppl_model.price_at(0.5, 10.0, sigma_mode="resqr")
    assert np.isclose(p_constant, p_resqr)


def test_shrinking_resqr_applies_offset(lppl_model):
    lppl_model._resqr = _fake_resqr_bundle(lambda q: 0.0 if q == 0.5 else 0.2)
    try:
        log_median = lppl_model._model_log10(np.array([10.0]))[0]
        p = lppl_model.price_at(0.75, 10.0, sigma_mode="resqr")
        assert np.isclose(np.log10(p), log_median + 0.2)
    finally:
        del lppl_model._resqr


# ── _FitsBasedModel (PowerLaw) ──────────────────────────────────────────────

def test_fits_based_fallback_when_no_resqr(pl_model):
    assert not hasattr(pl_model, "_resqr")
    p_constant = pl_model.price_at(0.5, 10.0)
    p_resqr = pl_model.price_at(0.5, 10.0, sigma_mode="resqr")
    assert np.isclose(p_constant, p_resqr)


def test_fits_based_model_log10_uses_q50(pl_model):
    # _model_log10 must return the Q50 fit exactly: intercept + slope * log10(t).
    f = pl_model.fits[0.5]
    t = 10.0
    expected = f["intercept"] + f["slope"] * np.log10(t)
    assert np.isclose(pl_model._model_log10(t), expected)


def test_fits_based_resqr_zero_offset_matches_q50(pl_model):
    pl_model._resqr = _fake_resqr_bundle()  # all zeros
    try:
        p = pl_model.price_at(0.75, 10.0, sigma_mode="resqr")
        expected = 10.0 ** pl_model._model_log10(10.0)
        assert np.isclose(p, expected)
    finally:
        del pl_model._resqr


# ── _CompositeModel (Bubble) ────────────────────────────────────────────────

def test_composite_model_log10_matches_composite_curve(bubble_model):
    """_model_log10 for BubbleModel must return the full composite curve —
    i.e. support + bubble cycles — so resqr residuals parallel the visible
    median line rather than the straight support line."""
    t = 10.0
    assert np.isclose(bubble_model._model_log10(t),
                       bubble_model._composite_log10(t))


def test_composite_fallback_when_no_resqr(bubble_model):
    # Fresh BubbleModel instance — can't rely on module-scope fixture state
    # after test_composite_resqr_matches_composite mutates ._resqr.
    import btc_core as bc
    fresh = bc.BubbleModel(bubble_model.__dict__.get("_md", None) or
                            bc.load_model_data(str(_ROOT / "model_data.pkl")))
    assert not hasattr(fresh, "_resqr")
    p_constant = fresh.price_at(0.5, 10.0)
    p_resqr = fresh.price_at(0.5, 10.0, sigma_mode="resqr")
    assert np.isclose(p_constant, p_resqr)


def test_composite_resqr_matches_composite(bubble_model):
    """With zero offsets, resqr output should match the composite curve
    (since _model_log10 now returns composite_log10)."""
    bubble_model._resqr = _fake_resqr_bundle()  # zero offsets
    try:
        p = bubble_model.price_at(0.5, 10.0, sigma_mode="resqr")
        composite_pred = 10.0 ** bubble_model._composite_log10(10.0)
        assert np.isclose(p, composite_pred)
    finally:
        del bubble_model._resqr


# ── shared _resqr_price_at helper ───────────────────────────────────────────

def test_resqr_scalar_returns_scalar():
    class Dummy:
        pass
    d = Dummy()
    d._resqr = _fake_resqr_bundle()
    out = bc._resqr_price_at(d, 0.5, np.asarray(5.0), np.asarray(3.0))
    assert isinstance(out, float)


def test_resqr_array_preserves_shape():
    class Dummy:
        pass
    d = Dummy()
    d._resqr = _fake_resqr_bundle()
    t = np.array([2.0, 5.0, 10.0])
    log_med = np.array([2.0, 3.0, 4.0])
    out = bc._resqr_price_at(d, 0.5, t, log_med)
    assert out.shape == t.shape


def test_resqr_q_interpolation_between_stored_quantiles():
    class Dummy:
        pass
    d = Dummy()
    # Offsets symmetric around Q50 so Q50-centering is a no-op on this case.
    # q=0.075 is between stored 0.05 and 0.10; expected interp = 2*(0.075-0.5)
    # = -0.85.
    d._resqr = _fake_resqr_bundle(lambda q: 2.0 * (q - 0.5))
    out = bc._resqr_price_at(d, 0.075, np.asarray(5.0), np.asarray(0.0))
    assert np.isclose(np.log10(out), -0.85, atol=0.05)


def test_resqr_flatline_past_last_knot():
    """For t > knots[-1]=12 the query-time clip means evaluation is at t=12.
    So price should be identical at t=12 and t=70 for the same quantile."""
    class Dummy:
        pass
    d = Dummy()
    # Build a bundle with a non-trivial log_t coefficient so the difference
    # would show up absent clipping.
    qs = np.array(sorted(DEFAULT_QUANTILES), dtype=np.float64)
    n_basis = 2 + len(DEFAULT_KNOTS)
    coefs = np.zeros((len(qs), n_basis), dtype=np.float64)
    coefs[:, 1] = 0.5  # positive log_t slope
    coefs[:, 2] = -0.5  # offset at first knot
    d._resqr = {
        "sorted_qs": qs,
        "coef_matrix": coefs,
        "knots": DEFAULT_KNOTS,
    }
    # Same log_median for both query points.
    p12 = bc._resqr_price_at(d, 0.5, np.asarray(12.0), np.asarray(3.0))
    p70 = bc._resqr_price_at(d, 0.5, np.asarray(70.0), np.asarray(3.0))
    assert np.isclose(p12, p70)


def test_resqr_no_resqr_returns_none():
    class Dummy:
        pass
    d = Dummy()
    out = bc._resqr_price_at(d, 0.5, np.asarray(5.0), np.asarray(3.0))
    assert out is None
