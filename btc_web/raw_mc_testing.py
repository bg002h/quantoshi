"""Shared inputs and the price-registry fixture for the Raw-MC tests.

Not a test module (pytest does not collect it).  Every ``test_raw_mc_*.py``
imports what it needs from here and requests ``raw_prices`` explicitly --
the fixture is deliberately NOT autouse, so a test that never touches the
registry never pays for it (follow-up F-23 M-2).
"""
import contextlib
import pathlib

import pandas as pd
import pytest

import raw_mc
from raw_mc import RawRequest

_CSV = pathlib.Path(__file__).resolve().parent.parent / "BitcoinPricesDaily.csv"


def _read_prices() -> pd.Series:
    df = pd.read_csv(_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    return df.set_index("Date")["Price"].astype(float)


PRICES = _read_prices()
FROZEN = pd.Timestamp("2026-11-08")     # "today" for statistical tests on the real series
RAW_FP = "csv"                          # data_fp the ``raw_prices`` fixture registers

# Simulator / regime-filter frozen inputs (spec "Tests"): prices truncated at
# the 2024-04-20 halving, which is also "today"; start price that day's close.
SIM_TODAY = pd.Timestamp("2024-04-20")
SIM_P0 = float(PRICES[SIM_TODAY])
SIM_WINDOW = (2010, 2024)


def _req(variant="raw_cycle", trend="power", **kw) -> RawRequest:
    """A RawRequest on the simulator's frozen inputs (today = asof =
    SIM_TODAY, data_fp = RAW_FP); keywords override any field.  Tests that
    simulate must also request the ``raw_prices`` fixture (F-23 M-2)."""
    base = dict(variant=variant, trend=trend, window=SIM_WINDOW, bins=5,
                checked=(), sims=400, years=40, today=SIM_TODAY, p0=SIM_P0,
                data_fp=RAW_FP, asof=SIM_TODAY)
    base.update(kw)
    return RawRequest(**base)


@contextlib.contextmanager
def registry_restored():
    """Save the price registry; on exit put it back (or unregister, when
    nothing was registered).  The state is named by ``raw_mc._UNSET``."""
    saved = raw_mc._registry
    try:
        yield
    finally:
        raw_mc._set_registry(saved)


@pytest.fixture
def raw_prices():
    """Register ``PRICES`` (data_fp "csv") for one test, then restore whatever
    was registered before, so registry tests never leak across tests that
    share an xdist worker."""
    with registry_restored():
        raw_mc.set_prices(PRICES, RAW_FP)
        yield PRICES
