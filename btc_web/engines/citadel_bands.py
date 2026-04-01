"""Citadel Planner — percentile band aggregation from simulation results.

Computes 7 percentile levels (P5/P10/P25/P50/P75/P90/P95) across 11 output
series from a multi-sim SimResult.
"""
from __future__ import annotations

import numpy as np

from .citadel_types import SimResult

__all__ = ["compute_bands", "BAND_PERCENTILES", "BAND_SERIES"]

BAND_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

BAND_SERIES = (
    "total",              # Total portfolio (USD)
    "btc_stack",          # BTC stack (native BTC units)
    "btc_usd",            # BTC holdings (USD)
    "cash",               # Cash (USD)
    "reserves_total",     # Reserves total (USD)
    "investments_total",  # Investments total (USD)
    "td_total",           # Tax-deferred total (USD)
    "tf_total",           # Tax-free total (USD)
    "cumulative_spend",   # Cumulative spending (USD)
    "taxes_paid",         # Cumulative taxes paid (USD)
    "depletion",          # Depletion fraction (0-1 per step)
)


def compute_bands(result: SimResult) -> dict[int, dict[str, np.ndarray]]:
    """Compute percentile bands from a SimResult.

    Returns dict keyed by percentile (5..95), each value a dict of
    series_name -> ndarray of shape (n_periods,).
    """
    n_sims = result.total_usd.shape[0]
    n_periods = result.total_usd.shape[1]
    _zero = np.zeros((n_sims, n_periods))

    series_data = {
        "total": result.total_usd,
        "btc_stack": result.btc_holdings,
        "btc_usd": result.btc_holdings * result.btc_prices,
        "cash": result.cash_balances,
        "reserves_total": result.reserve_balances.sum(axis=2),
        "investments_total": result.invest_balances.sum(axis=2),
        "td_total": result.td_total if result.td_total is not None else _zero,
        "tf_total": result.tf_total if result.tf_total is not None else _zero,
        "cumulative_spend": result.cumulative_spend,
        "taxes_paid": result.taxes_paid if result.taxes_paid is not None else _zero,
    }

    depletion_frac = (result.total_usd <= 0).astype(np.float64).mean(axis=0)

    bands: dict[int, dict[str, np.ndarray]] = {}
    for pct in BAND_PERCENTILES:
        band = {key: np.percentile(data, pct, axis=0)
                for key, data in series_data.items()}
        band["depletion"] = depletion_frac
        bands[pct] = band
    return bands
