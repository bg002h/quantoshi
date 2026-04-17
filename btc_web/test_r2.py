"""Unit tests for `_compute_log_r2` and `compute_model_r2` in btc_core._helpers.

These functions feed the R² values shown on the Model Info tab (/8.N) and
residual diagnostic charts. The review agent flagged them as having zero
direct test coverage — silent sign/denominator drift would publish wrong
R² for every model.
"""
import numpy as np
import pytest

from btc_core import _compute_log_r2, compute_model_r2


class TestComputeLogR2:
    def test_perfect_fit_returns_one(self):
        actual = np.array([1.0, 10.0, 100.0, 1000.0])
        predicted = actual.copy()
        r2 = _compute_log_r2(actual, predicted)
        assert r2 == pytest.approx(1.0, abs=1e-10)

    def test_constant_predictions_equal_to_mean(self):
        """When predictions = geometric mean of actual (constant in log),
        R² = 0 because SS_res = SS_tot."""
        # Log10 values: 1, 2, 3, 4 → mean 2.5 → geo mean 10^2.5
        actual = np.array([10.0, 100.0, 1000.0, 10000.0])
        predicted = np.full(4, 10.0 ** 2.5)
        r2 = _compute_log_r2(actual, predicted)
        assert r2 == pytest.approx(0.0, abs=1e-10)

    def test_degenerate_constant_actual_returns_none(self):
        """If all actual values are equal, ss_tot=0 → can't compute R²."""
        actual = np.array([100.0, 100.0, 100.0, 100.0])
        predicted = np.array([100.0, 100.0, 100.0, 100.0])
        r2 = _compute_log_r2(actual, predicted)
        assert r2 is None

    def test_zero_and_negative_inputs_clamped(self):
        """Zero/negative inputs must not raise; the function clamps to 1e-10."""
        actual = np.array([0.0, 1.0, 10.0, 100.0])
        predicted = np.array([1.0, 1.0, 10.0, 100.0])
        r2 = _compute_log_r2(actual, predicted)
        assert r2 is not None
        assert -10 < r2 < 1.0  # finite, somewhat poor

    def test_worse_than_mean_gives_negative_r2(self):
        """Standard convention: R² can go negative when model is worse
        than a horizontal mean line."""
        actual = np.array([1.0, 10.0, 100.0, 1000.0])
        # Anti-correlated prediction
        predicted = actual[::-1].copy()
        r2 = _compute_log_r2(actual, predicted)
        assert r2 < 0, f"anti-correlated predictions should give R²<0, got {r2}"

    def test_matches_sklearn_formula(self):
        """Hand-computed log-space R² matches sklearn's r2_score on log data."""
        rng = np.random.default_rng(42)
        log_true = rng.normal(4.0, 2.0, size=50)
        noise = rng.normal(0, 0.3, size=50)
        log_pred = log_true + noise
        actual = 10.0 ** log_true
        predicted = 10.0 ** log_pred

        r2 = _compute_log_r2(actual, predicted)

        ss_res = np.sum((log_true - log_pred) ** 2)
        ss_tot = np.sum((log_true - log_true.mean()) ** 2)
        expected = 1.0 - ss_res / ss_tot

        assert r2 == pytest.approx(expected, abs=1e-8)


class TestComputeModelR2:
    def test_populates_r2_per_quantile_dict(self):
        """compute_model_r2 sets mdl.r2_per_quantile for each fits key."""
        class M:
            def __init__(self):
                self.quantiles = [0.1, 0.5, 0.9]

            def price_at(self, q, t):
                t = np.asarray(t, dtype=float)
                # Exact log-linear model at quantile 0.5; shifted for others
                base = 10.0 ** (2.0 + 1.5 * np.log10(np.maximum(t, 0.1)))
                shift = {0.1: 0.5, 0.5: 1.0, 0.9: 2.0}[q]
                return base * shift

        m = M()
        years = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        # Actual = M's Q50 prediction → perfect R² at 0.5, worse elsewhere
        actual = m.price_at(0.5, years)

        compute_model_r2(m, years, actual)

        assert set(m.r2_per_quantile.keys()) == {0.1, 0.5, 0.9}
        assert m.r2_per_quantile[0.5] == pytest.approx(1.0, abs=1e-8)
        assert m.r2_per_quantile[0.1] < m.r2_per_quantile[0.5]
        assert m.r2_per_quantile[0.9] < m.r2_per_quantile[0.5]

    def test_skips_years_below_one(self):
        """compute_model_r2 masks out years < 1.0 (pre-log-transform undefined)."""
        class M:
            quantiles = [0.5]
            def price_at(self, q, t):
                return np.full_like(np.asarray(t, dtype=float), 1000.0)

        m = M()
        # Includes 0.5, 0.9 (below threshold) + 2.0 (above)
        years = np.array([0.5, 0.9, 2.0, 5.0, 10.0])
        actual = np.array([100.0, 200.0, 1000.0, 1000.0, 1000.0])

        compute_model_r2(m, years, actual)

        # Only t>=1.0 entries (1000, 1000, 1000) contribute. Actual and
        # predicted are both [1000, 1000, 1000] → ss_tot = 0 → R² is None
        # (suppressed from the dict per _compute_log_r2's contract).
        assert 0.5 not in m.r2_per_quantile

    def test_falls_back_to_q50_when_no_quantiles(self):
        """Non-quantized models (no `quantiles` attribute OR empty) still get
        an R² at Q50."""
        class M:
            quantiles = []  # empty
            def price_at(self, q, t):
                t = np.asarray(t, dtype=float)
                return 10.0 ** (2.0 + 1.5 * np.log10(np.maximum(t, 0.1)))

        m = M()
        years = np.array([1.0, 2.0, 5.0, 10.0])
        actual = m.price_at(0.5, years)

        compute_model_r2(m, years, actual)
        # Only Q50 is populated
        assert 0.5 in m.r2_per_quantile
        assert m.r2_per_quantile[0.5] == pytest.approx(1.0, abs=1e-8)

    def test_handles_price_at_exception_gracefully(self):
        """If a model's price_at raises for a given q, compute_model_r2 must
        skip that quantile instead of crashing the whole run."""
        class M:
            quantiles = [0.1, 0.5]
            def price_at(self, q, t):
                if q == 0.1:
                    raise RuntimeError("bad q")
                return np.full_like(np.asarray(t, dtype=float), 1000.0)

        m = M()
        years = np.array([2.0, 5.0, 10.0])
        actual = np.array([1000.0, 1100.0, 1200.0])

        compute_model_r2(m, years, actual)
        assert 0.1 not in m.r2_per_quantile  # skipped on exception
        assert 0.5 in m.r2_per_quantile       # still computed
