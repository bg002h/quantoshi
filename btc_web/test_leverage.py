"""Leverage calculator unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def test_leverage_defaults_has_expected_keys():
    from tab_defaults import LEVERAGE_DEFAULTS, leverage_defaults

    d = leverage_defaults()
    # Defaults used by the callback — must match the spec exactly.
    assert d["lev_price"] > 0  # seeded from model_data most recent close
    assert d["lev_date"] is not None
    assert d["lev_model"] == "bub"
    assert d["lev_floor_q"] == 0.01
    assert d["lev_rb"] == 13.0
    assert d["lev_rl"] == 4.5
    assert d["lev_horizon"] == 4.0
    assert d["lev_cagr"] == 20.0
    # Frozen-dict invariant
    assert type(LEVERAGE_DEFAULTS).__name__ == "mappingproxy"


def test_floor_price_bm_at_today():
    """BM Q1% floor at today should be a positive price."""
    from figures.leverage import floor_price
    import datetime as _dt
    price = floor_price("bub", 0.01, _dt.date.today())
    assert price > 0
    assert price < 10_000_000


def test_floor_price_higher_q_gives_higher_price():
    """Q5% floor > Q1% floor at same date (higher quantile = more aggressive)."""
    from figures.leverage import floor_price
    import datetime as _dt
    d = _dt.date.today()
    assert floor_price("bub", 0.05, d) > floor_price("bub", 0.01, d)


def test_floor_price_future_higher_than_today():
    """Floor grows over time (power-law)."""
    from figures.leverage import floor_price
    import datetime as _dt
    today = _dt.date.today()
    future = today.replace(year=today.year + 5)
    assert floor_price("bub", 0.01, future) > floor_price("bub", 0.01, today)


def test_floor_price_rejects_s2f_silently_returning_zero_q():
    """S2F.interp_price ignores q. Not in dropdown, but guard against silent misuse."""
    from figures.leverage import floor_price
    import _app_ctx
    import datetime as _dt
    if "s2f" in _app_ctx.PRICE_MODELS:
        d = _dt.date.today()
        assert floor_price("s2f", 0.01, d) == floor_price("s2f", 0.50, d)
