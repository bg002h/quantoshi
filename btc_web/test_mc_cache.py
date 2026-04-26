"""Tests for MC cache SSOT layer (_parse_cache_filename, _INTENDED_KEYS,
intended_models, MASTER_TO_CACHED_FALLBACK, is_master_cached, stash/commit/
restore)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def test_parse_cache_filename_handles_multi_underscore():
    """Model keys with internal underscores (ecfg_1d_1u, cfg_2d_1u) parse correctly."""
    from mc_cache import _parse_cache_filename

    assert _parse_cache_filename("paths_ecfg_1d_1u_2028.npz") == ("paths", "ecfg_1d_1u", 2028)
    assert _parse_cache_filename("overlays_cfg_2d_1u_2031.npz.bak") == ("overlays", "cfg_2d_1u", 2031)
    assert _parse_cache_filename("paths_bub_2028.npz") == ("paths", "bub", 2028)
    assert _parse_cache_filename("overlays_lppl_2035.npz") == ("overlays", "lppl", 2035)


def test_parse_cache_filename_rejects_garbage():
    """Non-cache filenames return None."""
    from mc_cache import _parse_cache_filename

    assert _parse_cache_filename("paths_2028.npz") is None       # no model key
    assert _parse_cache_filename("random.txt") is None
    assert _parse_cache_filename("") is None
    assert _parse_cache_filename("paths_bub_2028") is None       # missing .npz
    assert _parse_cache_filename("paths_bub_abcd.npz") is None   # year not 4 digits
