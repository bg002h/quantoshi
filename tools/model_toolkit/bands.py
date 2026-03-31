"""Uncertainty bands: QR channels + asymmetric gaussian sigma."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from statsmodels.regression.quantile_regression import QuantReg
from .data import PriceData

BM_QUANTILES = [
    0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999,
]  # 27 quantiles -- BM model

EF_QUANTILES = [
    0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999,
]  # 21 quantiles -- EF model


@dataclass
class QRResult:
    fits: dict          # {q: {"intercept": float, "slope": float, "r2": float}}
    ols_intercept: float
    ols_slope: float


@dataclass
class SigmaParams:
    sigma0_up: float
    alpha_up: float
    sigma0_down: float
    alpha_down: float


def fit_qr_channels(price_data: PriceData, quantiles=None) -> QRResult:
    """Fit quantile regression at each quantile level.

    From Cell 1 lines 212-241. OLS + statsmodels QuantReg.
    Model: log10(price) = intercept + slope * log10(t_years)
    """
    if quantiles is None:
        quantiles = BM_QUANTILES

    log_years = price_data.log_years
    log_prices = price_data.log_prices

    # OLS (mean) regression
    ols_slope, ols_intercept, ols_r, _, _ = linregress(log_years, log_prices)

    # Design matrix: [ones, log_years]
    X_fit = np.column_stack([np.ones(len(log_years)), log_years])

    # Per-quantile R2 (OLS formula applied to QR predictions)
    ss_tot_fit = np.sum((log_prices - log_prices.mean()) ** 2)

    qr_fits = {}
    for q in quantiles:
        res = QuantReg(log_prices, X_fit).fit(q=q, max_iter=2000)
        pred = res.params[0] + res.params[1] * log_years
        r2_q = 1 - np.sum((log_prices - pred) ** 2) / ss_tot_fit
        qr_fits[q] = {
            "intercept": float(res.params[0]),
            "slope": float(res.params[1]),
            "r2": float(r2_q),
        }

    return QRResult(fits=qr_fits, ols_intercept=ols_intercept, ols_slope=ols_slope)


def fit_asymmetric_sigma(
    prices_log: np.ndarray,
    composite_grid: np.ndarray,
    t_grid: np.ndarray,
    t_data: np.ndarray,
    n_bins: int = 20,
    min_pts: int = 10,
) -> SigmaParams:
    """Fit asymmetric shrinking gaussian to residuals.

    Absorbed from tools/fit_sigma.py.

    1. Convert composite_grid to log10 (E7 -- input is linear USD)
    2. Interpolate composite from t_grid to t_data
    3. residuals = prices_log - composite_at_data
    4. Split into upper (>=0) and lower (<0)
    5. Bin by time, compute sigma in each bin
    6. Fit sigma(t) = sigma0 * t^(-alpha) via curve_fit
    """
    # E7: composite_grid is in linear USD, convert to log10
    log_comp_grid = np.log10(np.maximum(composite_grid, 1e-10))

    # Interpolate composite from grid to data timestamps
    log_comp_at_data = np.interp(t_data, t_grid, log_comp_grid)

    # Residuals in log10 space
    residuals = prices_log - log_comp_at_data

    # Windowed fit in n_bins log-time bins
    log_t = np.log10(t_data)
    bin_edges = np.linspace(log_t.min(), log_t.max(), n_bins + 1)

    t_centers = []
    sigma_up_bins = []
    sigma_down_bins = []

    for b in range(n_bins):
        mask = (log_t >= bin_edges[b]) & (log_t < bin_edges[b + 1])
        if mask.sum() < min_pts:
            continue
        r = residuals[mask]
        t_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2))
        r_up = r[r >= 0]
        r_down = r[r < 0]
        sigma_up_bins.append(np.std(r_up) if len(r_up) > 3 else np.nan)
        sigma_down_bins.append(np.std(np.abs(r_down)) if len(r_down) > 3 else np.nan)

    t_centers = np.array(t_centers)
    sigma_up_bins = np.array(sigma_up_bins)
    sigma_down_bins = np.array(sigma_down_bins)

    def sigma_model(t, sigma0, alpha):
        return sigma0 * t ** (-alpha)

    # Fit upside
    valid = ~np.isnan(sigma_up_bins) & (sigma_up_bins > 0)
    popt_up, _ = curve_fit(
        sigma_model, t_centers[valid], sigma_up_bins[valid],
        p0=[0.5, 0.3], bounds=([0.01, -1], [5.0, 3.0]),
    )

    # Fit downside
    valid = ~np.isnan(sigma_down_bins) & (sigma_down_bins > 0)
    popt_down, _ = curve_fit(
        sigma_model, t_centers[valid], sigma_down_bins[valid],
        p0=[0.3, 0.3], bounds=([0.01, -1], [5.0, 3.0]),
    )

    return SigmaParams(
        sigma0_up=float(popt_up[0]),
        alpha_up=float(popt_up[1]),
        sigma0_down=float(popt_down[0]),
        alpha_down=float(popt_down[1]),
    )
