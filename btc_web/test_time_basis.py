"""Phase 1 plumbing tests for time_basis configuration."""
from __future__ import annotations
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOML = _REPO_ROOT / "quantoshi.toml"


def test_quantoshi_toml_exists():
    assert _TOML.exists(), f"{_TOML} should exist after Phase 1"


def test_quantoshi_toml_has_required_fields():
    with open(_TOML, "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["time_basis"] in ("calendar", "block")
    assert isinstance(cfg["block_origin"], int)
    # Sanity bound: block at 2009-07-25 UTC is in early-2009 chain history.
    # Actual value resolved via bitcoind RPC: 20188 (last block of that
    # UTC day, timestamp 2009-07-25T15:00:18Z). Keep bound loose so the
    # test isn't brittle if precision is revised.
    assert 19000 <= cfg["block_origin"] <= 21000
    assert cfg["blocks_per_year"] == 52596


def test_quantoshi_toml_default_is_calendar():
    """Default ships as calendar so Phase 1 changes nothing user-visible."""
    with open(_TOML, "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["time_basis"] == "calendar"


import datetime as _dt


def test_module_imports_with_calendar_default():
    from btc_web import time_basis as tb
    assert tb.TIME_BASIS == "calendar"
    assert tb.T_LABEL == "years"
    assert tb.T_PER_YEAR == 1.0
    assert tb.T_MIN == 1.0
    assert tb.T_ORIGIN_DATE == _dt.date(2009, 7, 25)
    assert isinstance(tb.T_ORIGIN_BLOCK, int)


def test_calendar_to_t_calendar_mode():
    from btc_web import time_basis as tb
    assert tb.TIME_BASIS == "calendar"
    # 2009-07-25 → t=0
    assert tb.calendar_to_t(_dt.date(2009, 7, 25)) == 0.0
    # 2010-07-25 → t≈1.0 (one year; 365 days / 365.25 ≈ 0.9993)
    t = tb.calendar_to_t(_dt.date(2010, 7, 25))
    assert abs(t - 1.0) < 1e-3


def test_t_to_calendar_calendar_mode():
    from btc_web import time_basis as tb
    assert tb.t_to_calendar(0.0) == _dt.date(2009, 7, 25)
    # t=1 year → close to 2010-07-25 (within 1 day for 365.25 rounding)
    d = tb.t_to_calendar(1.0)
    assert abs((d - _dt.date(2010, 7, 25)).days) <= 1


def test_round_trip_calendar_mode():
    from btc_web import time_basis as tb
    for d in [_dt.date(2010, 1, 1), _dt.date(2024, 12, 31),
              _dt.date(2050, 6, 15)]:
        t = tb.calendar_to_t(d)
        d2 = tb.t_to_calendar(t)
        assert abs((d - d2).days) <= 1


def test_block_mode_constants(monkeypatch):
    """Verify block-mode constants without rewriting the TOML.

    4-year date chosen so the calendar→block conversion is exact:
    2009-07-25 → 2013-07-25 spans exactly 1461 days = 4 × 365.25,
    so years = 1461/365.25 = 4.0 (no rounding) and the result is
    4 × 52596 = 210384 blocks.
    """
    from btc_web import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(tb, "T_LABEL", "blocks")
    monkeypatch.setattr(tb, "T_PER_YEAR", 52596.0)
    monkeypatch.setattr(tb, "T_MIN", 52596.0)
    t = tb.calendar_to_t(_dt.date(2013, 7, 25))
    assert abs(t - 210384.0) < 1.0  # 4 years × 52596 blocks/yr


def test_load_config_returns_dict_with_required_keys():
    from btc_web import time_basis as tb
    cfg = tb._load_config()
    assert "time_basis" in cfg
    assert "block_origin" in cfg
    assert "blocks_per_year" in cfg


def test_load_config_with_missing_file_falls_back_to_default(tmp_path):
    from btc_web import time_basis as tb
    cfg = tb._load_config(tmp_path / "nonexistent.toml")
    assert cfg["time_basis"] == "calendar"
    assert cfg["blocks_per_year"] == 52596
