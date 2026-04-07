"""Shared fixtures and helpers for Quantoshi web app tests."""
# ruff: noqa: F401
# pylint: disable=unused-import

import sys
import os
import json
import gzip
import base64
import math
from pathlib import Path
from unittest.mock import patch, MagicMock
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

# ── Setup sys.path so imports work ──────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btc_core import (load_model_data, fmt_price, yr_to_t, today_t,
                      qr_price, _find_lot_percentile, leo_weighted_entry)

# ── Load model data (once per session) ──────────────────────────────────────
M = load_model_data()

# ── Mock network calls, then import app + utilities ─────────────────────────
os.environ["TESTING"] = "1"

_original_urlopen = None

def _mock_urlopen(*args, **kwargs):
    """Prevent real network calls during test import."""
    raise Exception("mocked")

import urllib.request
_original_urlopen = urllib.request.urlopen
urllib.request.urlopen = _mock_urlopen

try:
    import app as _app_module
    from utils import _q3, _quantize_params, _nearest_quantile
    from snapshot import (_list_to_mask, _mask_to_list,
                          _encode_snapshot, _decode_snapshot,
                          _SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS,
                          _SNAP_PREFIX)
    from callbacks import (_parse_mc_upload, _extract_mc_key_val as _pk, _lots_summary,
                           _MC_UPLOAD_FIELDS, _mc_years_options,
                           _build_mc_params,
                           update_bubble, update_heatmap, update_dca,
                           update_retire, update_supercharge,
                           manage_lots, preview_percentile,
                           update_effective_lots, restore_from_url, apply_snapshot,
                           update_sc_info,
                           _TAB_CONTROLS, _TAB_TO_PATH)
    import _app_ctx
    _ALL_QS = _app_ctx._ALL_QS
except Exception:
    _q3 = _quantize_params = _list_to_mask = _mask_to_list = None
    _encode_snapshot = _decode_snapshot = _SNAPSHOT_CONTROLS = None
    _CHECKLIST_OPTIONS = _SNAP_PREFIX = _nearest_quantile = None
    _ALL_QS = None
    _app_ctx = None
finally:
    urllib.request.urlopen = _original_urlopen

# ── Import figure builders (depend on app init above) ───────────────────────
try:
    from figures import (build_bubble_figure, build_heatmap_figure,
                         build_dca_figure, build_retire_figure,
                         build_supercharge_figure, build_citadel_figure)
except Exception:
    build_bubble_figure = build_heatmap_figure = build_dca_figure = None
    build_retire_figure = build_supercharge_figure = build_citadel_figure = None

# ── Shared constants ────────────────────────────────────────────────────────
_FUZZ_QS = [q for q in [0.001, 0.01, 0.10, 0.50, 0.90] if q in M.qr_fits]
_FUZZ_Q1 = _FUZZ_QS[:1]
_FUZZ_Q2 = _FUZZ_QS[:2] if len(_FUZZ_QS) >= 2 else _FUZZ_QS
_YR_NOW = pd.Timestamp.today().year
_Q50 = min(M.qr_fits, key=lambda q: abs(q - 0.5))


# ── Shared helpers ──────────────────────────────────────────────────────────

def _assert_figure(result, expect_tuple=False):
    """Assert result is a valid figure (or (figure, mc_result) tuple)."""
    if expect_tuple:
        assert isinstance(result, tuple) and len(result) == 2
        fig, _ = result
    else:
        fig = result
    assert isinstance(fig, go.Figure)


class _CallbackCtx:
    """Minimal mock for dash.ctx (dash._callback_context)."""
    def __init__(self, triggered_id=None):
        self.triggered_id = triggered_id


def _patch_ctx(triggered_id=None):
    """Context manager that patches dash.ctx across all callback submodules."""
    ctx_obj = _CallbackCtx(triggered_id)
    _targets = [
        "callbacks", "callbacks.charts", "callbacks.lots",
        "callbacks.mc_helpers", "callbacks.mc_payment",
        "callbacks.snapshot_cb",
    ]
    stack = ExitStack()
    for t in _targets:
        stack.enter_context(patch.multiple(t, ctx=ctx_obj))
    return stack


class _MockPriceModel:
    """Minimal mock of a PriceModel for citadel tax tests."""
    quantized = True

    def __init__(self, fits=None, genesis=None):
        self.fits = fits or {0.25: {"slope": 5.0, "intercept": 2.0}}
        self.genesis = genesis or pd.Timestamp("2009-07-25")

    def price_at(self, q, t):
        return 50_000.0 * (1 + t / 100)

    def quantile_at(self, price, t):
        return 0.50


def _test_model():
    return _MockPriceModel()


class _ControlledPriceModel:
    """Price model that returns configurable quantiles for trigger testing."""
    def __init__(self, quantile=0.50, price=50_000.0):
        self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
        self.genesis = pd.Timestamp("2009-07-25")
        self._quantile = quantile
        self._price = price

    def price_at(self, q, t):
        return self._price

    def quantile_at(self, price, t):
        return self._quantile


def _bare_config(**kw):
    """SimConfig with zero-volatility, deterministic, short horizon for unit tests."""
    from engines.citadel import SimConfig
    defaults = dict(
        start_stack=1.0, start_yr=2031, end_yr=2035,
        freq="Annually", monthly_spend=5000,
        cash_initial=100_000, selected_qs=[0.25],
        reserve_bins=[
            {"label": "Short", "initial": 50_000, "rate": 0, "volatility": 0},
            {"label": "Medium", "initial": 50_000, "rate": 0, "volatility": 0},
            {"label": "Long", "initial": 50_000, "rate": 0, "volatility": 0},
        ],
        invest_bins=[
            {"label": "Equities", "initial": 100_000, "return_rate": 0, "volatility": 0},
            {"label": "Bonds", "initial": 50_000, "return_rate": 0, "volatility": 0},
        ],
    )
    defaults.update(kw)
    return SimConfig(**defaults)
