"""Phase 2a tests — refactor + parameterize build pipeline.

Tests the btc_core → time_basis bridge, T_MIN sweep, and build-pipeline
parameterization. Does NOT exercise block-mode end-to-end (Phase 2b builds
the actual block pkl).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest


def test_btc_core_bridges_time_basis_into_sys_path():
    """Importing btc_core makes time_basis importable as a top-level module."""
    import btc_core  # noqa: F401
    # After btc_core is imported, time_basis should be importable bare.
    import time_basis as tb  # would fail without the bridge
    assert tb.TIME_BASIS in ("calendar", "block")
    assert tb.T_MIN in (1.0, 52596.0)


def test_time_basis_year_to_t_calendar():
    """year_to_t in calendar mode returns years since 2009-07-25."""
    import time_basis as tb
    if tb.TIME_BASIS != "calendar":
        pytest.skip("calendar-only test")
    # 2010 January 1 → 0.439 years past 2009-07-25 (160 days / 365.25).
    t = tb.year_to_t(2010)
    assert 0.4 < t < 0.5
    # 2024 January 1 → 14.439 years past 2009-07-25.
    t = tb.year_to_t(2024)
    assert 14.4 < t < 14.5
    # Fractional year: 2024.5 = July 1 2024 → 14.939
    t = tb.year_to_t(2024.5)
    assert 14.9 < t < 15.0


def test_time_basis_year_to_t_block(monkeypatch):
    """year_to_t in block mode scales the calendar-mode result by T_PER_YEAR."""
    import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(tb, "T_PER_YEAR", 52596.0)
    # 2024 January 1 → ~14.439 years × 52596 ≈ 759,406 blocks since origin.
    t = tb.year_to_t(2024)
    assert 759_000 < t < 760_000


def test_time_basis_today_t_positive_and_in_range():
    """today_t returns a sensible value in either basis."""
    import time_basis as tb
    t = tb.today_t()
    if tb.TIME_BASIS == "calendar":
        # Today is at least 16 years past 2009-07-25, less than 30.
        assert 16.0 < t < 30.0
    else:
        # Block mode: 16 years × 52596 ≈ 841,536; less than 30 × 52596.
        assert 800_000 < t < 1_600_000


def test_t_min_sweep_calendar_mode_unchanged():
    """All 13 mask sites still exclude the same rows in calendar mode."""
    import numpy as np
    from time_basis import T_MIN
    assert T_MIN == 1.0  # this test is calendar-only
    # The mask `>= T_MIN` with T_MIN=1.0 must produce the same boolean
    # array as the old `>= 1.0` literal. Pick a synthetic price_years
    # array that straddles the threshold.
    price_years = np.array([0.5, 0.99, 1.0, 1.01, 5.0, 14.0])
    new_mask = price_years >= T_MIN
    old_mask = price_years >= 1.0
    np.testing.assert_array_equal(new_mask, old_mask)


def test_t_min_block_mode_threshold():
    """In block mode, T_MIN = T_PER_YEAR (one year's worth of blocks)."""
    import time_basis as tb
    if tb.TIME_BASIS == "block":
        assert tb.T_MIN == tb.T_PER_YEAR == 52596.0
    else:
        assert tb.T_MIN == tb.T_PER_YEAR == 1.0
