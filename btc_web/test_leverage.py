"""Leverage calculator unit tests."""
from __future__ import annotations

import sys
from datetime import timedelta
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
    future = today + timedelta(days=5 * 365)
    assert floor_price("bub", 0.01, future) > floor_price("bub", 0.01, today)


def test_P_max_basic():
    from figures.leverage import P_max
    # sell=181649, H=4, c=0.20 -> P_max ≈ 87601
    assert P_max(181649, 4, 0.20) == pytest.approx(87601, rel=1e-3)


def test_P_max_zero_cagr_equals_sell_price():
    from figures.leverage import P_max
    assert P_max(100_000, 4, 0.0) == 100_000


def test_implied_cagr_basic():
    from figures.leverage import implied_cagr
    # sell=181649, P_now=72926, H=4 -> c ≈ 0.256
    assert implied_cagr(181649, 72926, 4) == pytest.approx(0.256, rel=1e-2)


def test_implied_cagr_zero_price_returns_none():
    from figures.leverage import implied_cagr
    assert implied_cagr(181649, 0, 4) is None


def test_build_leverage_figure_returns_plotly_figure():
    from figures.leverage import build_leverage_figure
    p = {
        "lev_price": 72926.0,
        "lev_date": "2026-04-18",
        "lev_model": "bub",
        "lev_floor_q": 0.01,
        "lev_rb": 13.0,
        "lev_rl": 4.5,
        "lev_horizon": 4.0,
        "lev_cagr": 20.0,
        "palette": "default",
    }
    fig = build_leverage_figure(p)
    assert fig is not None
    assert len(fig.data) >= 4  # at least 4 curves
    assert fig.layout.yaxis.type == "log"
