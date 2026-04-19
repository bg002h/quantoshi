"""Leverage calculator — figure builder and math helpers.

Design spec: docs/superpowers/specs/2026-04-18-leverage-calculator-design.md
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

import _app_ctx

# Shared project genesis — every PriceModel in btc_core uses this as t=0
# (CLAUDE.md: "All models use 2009-07-25 as their time origin").
_GENESIS = pd.Timestamp("2009-07-25")


def floor_price(model_short: str, q: float, target_date) -> float:
    """Return the `model_short`-q floor price at `target_date` in USD.

    Args:
        model_short: key into _app_ctx.PRICE_MODELS (e.g. "bub", "pl", "lppl").
        q: quantile in (0, 1), e.g. 0.01 for Q1%.
        target_date: datetime.date or datetime.datetime.

    Returns:
        Floor price in USD (positive float).
    """
    model = _app_ctx.PRICE_MODELS[model_short]
    t_yr = (pd.Timestamp(target_date) - _GENESIS).days / 365.25
    return float(model.interp_price(q, t_yr))
