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
