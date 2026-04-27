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
