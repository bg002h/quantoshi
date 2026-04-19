"""Leverage calculator unit tests."""
from __future__ import annotations

import sys
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")

import datetime as _dt

import pytest


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
