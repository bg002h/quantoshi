"""Fit or define a power-law support line."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
from .data import PriceData


@dataclass
class SupportLine:
    intercept: float          # log10 intercept
    slope: float              # power-law exponent
    A: float                  # 10^intercept
    B: float                  # slope alias
    log_excess: np.ndarray    # log_price - log_support at fit data points
    log_support: np.ndarray   # support values at fit data points (log10)
    log_excess_all: np.ndarray    # log_price - log_support at all data points (full dataset)
    log_support_all: np.ndarray   # support values at all data points (log10)


def fit_support(price_data: PriceData, percentile: float = 0.20, quantile: float = 0.50) -> SupportLine:
    """Fit support: OLS -> bottom percentile filter -> QuantReg.

    Matches Cell 0 lines 325-354 exactly.

    Parameters
    ----------
    price_data : PriceData
        Loaded price data from load_prices().
    percentile : float
        Bottom fraction of OLS residuals used as support points (default 0.20 = bottom 20%).
    quantile : float
        QuantReg quantile for the support fit (default 0.50).
    """
    log_t = price_data.log_years
    log_p = price_data.log_prices
    log_t_all = price_data.df_full["log_years"].values
    log_p_all = price_data.df_full["log_price"].values

    # 1. OLS on fit subset
    slope_ols, intercept_ols, _, _, _ = linregress(log_t, log_p)
    ols_resid = log_p - (intercept_ols + slope_ols * log_t)

    # 2. Bottom percentile filter
    cutoff = np.percentile(ols_resid, percentile * 100)
    support_mask = ols_resid <= cutoff

    # 3. QuantReg on support subset
    X_support = sm.add_constant(log_t[support_mask])
    res_sup = sm.QuantReg(log_p[support_mask], X_support).fit(q=quantile, max_iter=10000)
    intercept_sup = res_sup.params[0]
    slope_sup = res_sup.params[1]

    # 4. Compute support and excess for fit points and all points
    log_support_fit = intercept_sup + slope_sup * log_t
    log_excess_fit = log_p - log_support_fit
    log_support_all = intercept_sup + slope_sup * log_t_all
    log_excess_all = log_p_all - log_support_all

    return SupportLine(
        intercept=intercept_sup,
        slope=slope_sup,
        A=10 ** intercept_sup,
        B=slope_sup,
        log_excess=log_excess_fit,
        log_support=log_support_fit,
        log_excess_all=log_excess_all,
        log_support_all=log_support_all,
    )


def fixed_support(intercept: float, slope: float, price_data: PriceData) -> SupportLine:
    """Use a hardcoded support line (e.g. Empirical Floor).

    Parameters
    ----------
    intercept : float
        log10 intercept of the support line.
    slope : float
        Power-law exponent of the support line.
    price_data : PriceData
        Loaded price data for computing excess values.
    """
    log_t = price_data.log_years
    log_p = price_data.log_prices
    log_t_all = price_data.df_full["log_years"].values
    log_p_all = price_data.df_full["log_price"].values

    log_support_fit = intercept + slope * log_t
    log_excess_fit = log_p - log_support_fit
    log_support_all = intercept + slope * log_t_all
    log_excess_all = log_p_all - log_support_all

    return SupportLine(
        intercept=intercept,
        slope=slope,
        A=10 ** intercept,
        B=slope,
        log_excess=log_excess_fit,
        log_support=log_support_fit,
        log_excess_all=log_excess_all,
        log_support_all=log_support_all,
    )
