"""Tests for the Quantoshi web app — covers all tabs and key functions.

Run:  btc_venv/bin/python3 -m pytest btc_web/test_web.py -v
"""

import sys
import os
import json
import gzip
import base64
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ── Setup sys.path so imports work ──────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_BTC_APP = _ROOT / "btc_app"
for _p in (str(_ROOT), str(_BTC_APP), str(_ROOT / "btc_web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: btc_core utility tests
# ═══════════════════════════════════════════════════════════════════════════════

from btc_core import (load_model_data, fmt_price, yr_to_t, today_t,
                      qr_price, _find_lot_percentile, leo_weighted_entry)
import pandas as pd

M = load_model_data()


class TestFmtPrice:
    def test_under_1k(self):
        assert fmt_price(999) == "$999"

    def test_exact_1k(self):
        assert fmt_price(1000) == "$1.0K"

    def test_thousands(self):
        assert fmt_price(12345) == "$12.3K"

    def test_millions(self):
        assert fmt_price(1234567) == "$1.23M"

    def test_zero(self):
        assert fmt_price(0) == "$0"

    def test_small(self):
        assert fmt_price(0.5) == "$0"


class TestYrToT:
    def test_genesis_year(self):
        t = yr_to_t(2009, M.genesis)
        assert abs(t - (-2 / 365.25)) < 0.01  # Jan 1 vs Jan 3

    def test_2025(self):
        t = yr_to_t(2025, M.genesis)
        assert 15.9 < t < 16.1

    def test_monotonic(self):
        assert yr_to_t(2030, M.genesis) > yr_to_t(2025, M.genesis)


class TestQrPrice:
    def test_returns_positive(self):
        t = yr_to_t(2025, M.genesis)
        for q in [0.05, 0.5, 0.95]:
            if q in M.qr_fits:
                p = qr_price(q, t, M.qr_fits)
                assert p > 0

    def test_higher_quantile_higher_price(self):
        t = yr_to_t(2025, M.genesis)
        if 0.1 in M.qr_fits and 0.9 in M.qr_fits:
            p10 = qr_price(0.1, t, M.qr_fits)
            p90 = qr_price(0.9, t, M.qr_fits)
            assert p90 > p10

    def test_array_input(self):
        ts = np.array([yr_to_t(2020, M.genesis), yr_to_t(2025, M.genesis)])
        if 0.5 in M.qr_fits:
            prices = qr_price(0.5, ts, M.qr_fits)
            assert len(prices) == 2
            assert prices[1] > prices[0]  # later year → higher price


class TestFindLotPercentile:
    def test_returns_in_range(self):
        t = yr_to_t(2025, M.genesis)
        price = 100000
        pct = _find_lot_percentile(t, price, M.qr_fits)
        assert pct is not None
        assert 0 < pct < 1

    def test_high_price_high_percentile(self):
        t = yr_to_t(2025, M.genesis)
        pct_lo = _find_lot_percentile(t, 10000, M.qr_fits)
        pct_hi = _find_lot_percentile(t, 500000, M.qr_fits)
        assert pct_hi > pct_lo


class TestLeoWeightedEntry:
    def test_empty_lots(self):
        assert leo_weighted_entry([]) is None

    def test_single_lot(self):
        lots = [{"btc": 1.0, "price": 50000, "date": "2024-01-15", "pct_q": 0.5}]
        result = leo_weighted_entry(lots)
        assert result is not None
        entry_price, entry_t, avg_pct_q, total_btc = result
        assert abs(entry_price - 50000) < 1
        assert abs(total_btc - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: app.py utility tests (_q3, _quantize_params, snapshot encode/decode)
# ═══════════════════════════════════════════════════════════════════════════════

# Import app.py utilities — need to suppress Dash app creation side effects
# by patching network calls at startup
os.environ["TESTING"] = "1"

# We need the functions but not the full app startup. Import carefully.
# The app module does network calls at import time (_startup_heatmap_defaults).
# We'll mock those.

_original_urlopen = None

def _mock_urlopen(*args, **kwargs):
    """Prevent real network calls during test import."""
    raise Exception("mocked")


# Patch before importing app
import urllib.request
_original_urlopen = urllib.request.urlopen
urllib.request.urlopen = _mock_urlopen

try:
    from app import (_q3, _quantize_params, _list_to_mask, _mask_to_list,
                     _encode_snapshot, _decode_snapshot,
                     _SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS, _ALL_QS,
                     _nearest_quantile, _parse_mc_upload, _pk)
except Exception:
    # If app import fails, define stubs for the test — tests will be skipped
    _q3 = _quantize_params = None
finally:
    urllib.request.urlopen = _original_urlopen


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestQ3:
    def test_zero(self):
        assert _q3(0) == 0

    def test_none(self):
        assert _q3(None) is None

    def test_small_number(self):
        assert _q3(0.0623) == pytest.approx(0.0623, rel=0.02)

    def test_large_number(self):
        result = _q3(95437)
        assert result == pytest.approx(95400, rel=0.02)

    def test_thousands(self):
        result = _q3(1234.5)
        assert result == pytest.approx(1230, rel=0.02)

    def test_negative(self):
        result = _q3(-456.7)
        assert result == pytest.approx(-457, rel=0.02)


@pytest.mark.skipif(_quantize_params is None, reason="app.py import failed")
class TestQuantizeParams:
    def test_floats_quantized(self):
        p = {"price": 95437.0, "amount": 100.0}
        out = _quantize_params(p)
        assert out["price"] == _q3(95437.0)
        assert out["amount"] == _q3(100.0)

    def test_selected_qs_exempt(self):
        qs = [0.05, 0.1, 0.5]
        out = _quantize_params({"selected_qs": qs})
        assert out["selected_qs"] == qs

    def test_exit_qs_exempt(self):
        qs = [0.25, 0.75]
        out = _quantize_params({"exit_qs": qs})
        assert out["exit_qs"] == qs

    def test_zero_float_unchanged(self):
        out = _quantize_params({"val": 0.0})
        assert out["val"] == 0.0

    def test_int_unchanged(self):
        out = _quantize_params({"year": 2025})
        assert out["year"] == 2025

    def test_string_unchanged(self):
        out = _quantize_params({"mode": "usd"})
        assert out["mode"] == "usd"

    def test_list_of_floats(self):
        out = _quantize_params({"vals": [1234.5, 0.0, 5678.9]})
        assert out["vals"][0] == _q3(1234.5)
        assert out["vals"][1] == 0.0  # zero unchanged
        assert out["vals"][2] == _q3(5678.9)


# ── Bitmask encoding ─────────────────────────────────────────────────────────

@pytest.mark.skipif(_list_to_mask is None, reason="app.py import failed")
class TestBitmaskEncoding:
    def test_empty_list(self):
        assert _list_to_mask([], ["a", "b", "c"]) == 0

    def test_single_item(self):
        assert _list_to_mask(["b"], ["a", "b", "c"]) == 0b010

    def test_all_items(self):
        assert _list_to_mask(["a", "b", "c"], ["a", "b", "c"]) == 0b111

    def test_roundtrip(self):
        opts = ["shade", "show_ols", "show_data", "show_today", "show_legend"]
        val = ["show_data", "show_today"]
        mask = _list_to_mask(val, opts)
        restored = _mask_to_list(mask, opts)
        assert set(restored) == set(val)

    def test_quantile_roundtrip(self):
        """Test bitmask with actual quantile float values."""
        if not _ALL_QS:
            pytest.skip("No quantiles loaded")
        opts = list(_ALL_QS)
        val = opts[:3]  # first 3 quantiles
        mask = _list_to_mask(val, opts)
        restored = _mask_to_list(mask, opts)
        assert restored == val


# ── Snapshot encode/decode ────────────────────────────────────────────────────

@pytest.mark.skipif(_encode_snapshot is None, reason="app.py import failed")
class TestSnapshotRoundtrip:
    def test_basic_roundtrip(self):
        state = {"bub-xscale:value": "Log", "bub-yscale:value": "Log"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded.get("bub-xscale:value") == "Log"
        assert decoded.get("bub-yscale:value") == "Log"

    def test_checklist_bitmask_roundtrip(self):
        state = {
            "bub-toggles:value": ["shade", "show_data"],
            "bub-qs:value": list(_ALL_QS)[:2] if _ALL_QS else [],
        }
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        if "bub-toggles:value" in decoded:
            assert set(decoded["bub-toggles:value"]) == {"shade", "show_data"}

    def test_tab_filter(self):
        state = {
            "bub-xscale:value": "Log",
            "dca-amount:value": 500,
            "main-tabs:active_tab": "bubble",
        }
        tab_filter = {"bub-xscale", "bub-yscale"}
        encoded = _encode_snapshot(state, tab_filter=tab_filter)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded.get("bub-xscale:value") == "Log"
        # dca-amount should be filtered out
        assert "dca-amount:value" not in decoded
        # main-tabs always included
        assert decoded.get("main-tabs:active_tab") == "bubble"

    def test_lots_roundtrip(self):
        state = {
            "bub-xscale:value": "Log",
            "_lots": [{"btc": 1.0, "price": 69420, "date": "2024-01-15"}],
        }
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded["_lots"][0]["price"] == 69420

    def test_invalid_decode(self):
        assert _decode_snapshot("not-valid-base64!!!") is None


# ── _nearest_quantile ─────────────────────────────────────────────────────────

@pytest.mark.skipif(_nearest_quantile is None, reason="app.py import failed")
class TestNearestQuantile:
    def test_exact_match(self):
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        assert _nearest_quantile(0.5, qs) == 0.5

    def test_nearest(self):
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        assert _nearest_quantile(0.48, qs) == 0.5
        assert _nearest_quantile(0.12, qs) == 0.1


# ── MC upload helpers ─────────────────────────────────────────────────────────

@pytest.mark.skipif(_parse_mc_upload is None, reason="app.py import failed")
class TestParseMcUpload:
    def test_none_input(self):
        data, err = _parse_mc_upload(None)
        assert data is None
        assert err is None

    def test_valid_json(self):
        payload = {"path_key": {"mc_years": 10}, "overlay_key": {}}
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        contents = f"data:application/json;base64,{b64}"
        data, err = _parse_mc_upload(contents)
        assert err is None
        assert data["path_key"]["mc_years"] == 10

    def test_invalid_json(self):
        payload = {"no_path_key": True}
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        contents = f"data:application/json;base64,{b64}"
        data, err = _parse_mc_upload(contents)
        assert data is None
        assert "Invalid" in err

    def test_legacy_params_key(self):
        payload = {"params": {"mc_years": 20}}
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        contents = f"data:application/json;base64,{b64}"
        data, err = _parse_mc_upload(contents)
        assert err is None
        assert data["params"]["mc_years"] == 20


    def test_cross_tab_rejected(self):
        """Uploading a retire sim on DCA tab should be rejected."""
        payload = {"tab": "ret", "path_key": {"mc_years": 10}, "overlay_key": {}}
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        contents = f"data:application/json;base64,{b64}"
        data, err = _parse_mc_upload(contents, expected_tab="dca")
        assert data is None
        assert "Wrong tab" in err
        assert "Retire" in err

    def test_same_tab_accepted(self):
        payload = {"tab": "dca", "path_key": {"mc_years": 10}, "overlay_key": {}}
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        contents = f"data:application/json;base64,{b64}"
        data, err = _parse_mc_upload(contents, expected_tab="dca")
        assert err is None
        assert data is not None

    def test_no_tab_field_accepted(self):
        """Legacy files without tab field should still load."""
        payload = {"path_key": {"mc_years": 10}, "overlay_key": {}}
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        contents = f"data:application/json;base64,{b64}"
        data, err = _parse_mc_upload(contents, expected_tab="dca")
        assert err is None
        assert data is not None


@pytest.mark.skipif(_pk is None, reason="app.py import failed")
class TestPkHelper:
    def test_from_path_key(self):
        data = {"path_key": {"mc_years": 10, "mc_start_yr": 2026}, "overlay_key": {}}
        assert _pk(data, "mc_years") == 10
        assert _pk(data, "mc_start_yr") == 2026

    def test_from_overlay_key(self):
        data = {"path_key": {}, "overlay_key": {"mc_amount": 500}}
        assert _pk(data, "mc_amount") == 500

    def test_path_key_priority(self):
        data = {"path_key": {"val": 1}, "overlay_key": {"val": 2}}
        assert _pk(data, "val") == 1

    def test_default(self):
        data = {"path_key": {}, "overlay_key": {}}
        assert _pk(data, "missing", 42) == 42

    def test_empty_data(self):
        assert _pk({}, "anything", "default") == "default"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: mc_cache.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from mc_cache import (snap_to_bin, _path_key_str, _overlay_key_str,
                      CACHED_START_YRS, ENTRY_PCT_BINS, MC_YEARS_OPTIONS,
                      WD_AMOUNTS, INFL_OPTIONS, STACK_SIZES, FAN_PCTS,
                      is_cached_year)


class TestSnapToBin:
    def test_exact_bins(self):
        for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            assert snap_to_bin(b) == b

    def test_rounds_to_nearest(self):
        assert snap_to_bin(0.14) == 0.1
        assert snap_to_bin(0.16) == 0.2
        assert snap_to_bin(0.55) == 0.6

    def test_clamps_low(self):
        assert snap_to_bin(0.0) == 0.1
        assert snap_to_bin(0.04) == 0.1

    def test_clamps_high(self):
        assert snap_to_bin(1.0) == 0.9
        assert snap_to_bin(0.96) == 0.9


class TestPathKeyStr:
    def test_format(self):
        assert _path_key_str(0.5, 10) == "p0.5_y10"
        assert _path_key_str(0.1, 30) == "p0.1_y30"

    def test_deterministic(self):
        assert _path_key_str(0.3, 20) == _path_key_str(0.3, 20)


class TestOverlayKeyStr:
    def test_format(self):
        result = _overlay_key_str(0.5, 10, 5000, 4, 1.0)
        assert result == "p0.5_y10_w5000_i4_s1.0"

    def test_all_params(self):
        result = _overlay_key_str(0.9, 30, 69420, 12, 10.0)
        assert "w69420" in result
        assert "i12" in result
        assert "s10.0" in result


class TestCacheConstants:
    def test_cached_years(self):
        assert 2026 in CACHED_START_YRS
        assert 2040 in CACHED_START_YRS
        assert len(CACHED_START_YRS) == 5

    def test_entry_bins(self):
        assert len(ENTRY_PCT_BINS) == 9
        assert ENTRY_PCT_BINS[0] == 0.1
        assert ENTRY_PCT_BINS[-1] == 0.9

    def test_mc_years_options(self):
        assert 10 in MC_YEARS_OPTIONS
        assert 20 in MC_YEARS_OPTIONS
        assert 30 in MC_YEARS_OPTIONS

    def test_fan_pcts(self):
        assert 0.50 in FAN_PCTS  # median must exist
        assert len(FAN_PCTS) == 6


class TestIsCachedYear:
    def test_uncached_year(self):
        # 9999 should never be cached
        assert is_cached_year(9999) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: figures.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from figures import (_mc_path_key, _mc_overlay_key, _mc_fan_to_lists,
                     _mc_fan_from_lists, _mc_paths_to_lists, _mc_paths_from_lists,
                     _apply_watermark, _interp_qr_price,
                     _FREQ_STEP_DAYS, _MC_FAN_PCTS)
import plotly.graph_objects as go


class TestMcPathKey:
    def test_dca_defaults(self):
        p = {}
        key = _mc_path_key(p, "dca")
        assert key["tab"] == "dca"
        assert key["mc_bins"] == 5
        assert key["mc_sims"] == 800
        assert key["mc_years"] == 10
        assert key["mc_freq"] == "Monthly"
        assert key["mc_start_yr"] == 2026
        assert "mc_entry_q" in key

    def test_ret_defaults(self):
        key = _mc_path_key({}, "ret")
        assert key["mc_start_yr"] == 2031

    def test_sc_defaults(self):
        key = _mc_path_key({}, "sc")
        assert key["mc_start_yr"] == 2031

    def test_hm_uses_entry_q(self):
        key = _mc_path_key({"entry_q": 75}, "hm")
        assert key["entry_q"] == 75.0
        assert "mc_entry_q" not in key

    def test_dca_uses_mc_entry_q(self):
        key = _mc_path_key({"mc_entry_q": 30}, "dca")
        assert key["mc_entry_q"] == 30.0
        assert "entry_q" not in key

    def test_custom_params(self):
        p = {"mc_bins": 10, "mc_sims": 400, "mc_years": 20,
             "mc_freq": "Daily", "mc_start_yr": 2028, "mc_entry_q": 60}
        key = _mc_path_key(p, "dca")
        assert key["mc_bins"] == 10
        assert key["mc_sims"] == 400
        assert key["mc_years"] == 20
        assert key["mc_freq"] == "Daily"
        assert key["mc_start_yr"] == 2028
        assert key["mc_entry_q"] == 60.0

    def test_path_key_deterministic(self):
        p = {"mc_start_yr": 2026, "mc_entry_q": 50}
        assert _mc_path_key(p, "dca") == _mc_path_key(p, "dca")


class TestMcOverlayKey:
    def test_dca_no_inflation(self):
        key = _mc_overlay_key({"mc_amount": 500}, "dca", 0.0)
        assert key["mc_amount"] == 500.0
        assert key["start_stack"] == 0.0
        assert "mc_infl" not in key

    def test_ret_has_inflation(self):
        key = _mc_overlay_key({"mc_amount": 5000, "mc_infl": 4}, "ret", 1.0)
        assert key["mc_infl"] == 4.0

    def test_sc_has_inflation(self):
        key = _mc_overlay_key({"mc_amount": 10000, "mc_infl": 6}, "sc", 2.0)
        assert key["mc_infl"] == 6.0
        assert key["start_stack"] == 2.0


class TestMcFanSerialization:
    def test_roundtrip(self):
        fan = {0.05: np.array([1.0, 2.0, 3.0]),
               0.50: np.array([10.0, 20.0, 30.0]),
               0.95: np.array([100.0, 200.0, 300.0])}
        serialized = _mc_fan_to_lists(fan)
        restored = _mc_fan_from_lists(serialized)
        for k in fan:
            np.testing.assert_allclose(restored[k], fan[k], atol=0.001)

    def test_keys_are_strings_in_json(self):
        fan = {0.50: np.array([1.0])}
        serialized = _mc_fan_to_lists(fan)
        assert "0.5" in serialized

    def test_empty_fan(self):
        fan = {}
        serialized = _mc_fan_to_lists(fan)
        assert serialized == {}
        restored = _mc_fan_from_lists(serialized)
        assert restored == {}


class TestMcPathsSerialization:
    def test_roundtrip(self):
        paths = np.random.randn(10, 5).astype(np.float32)
        serialized = _mc_paths_to_lists(paths)
        restored = _mc_paths_from_lists(serialized)
        np.testing.assert_allclose(restored, paths, atol=1e-6)

    def test_dtype(self):
        paths = np.array([[1.0, 2.0]], dtype=np.float64)
        restored = _mc_paths_from_lists(_mc_paths_to_lists(paths))
        assert restored.dtype == np.float32


class TestApplyWatermark:
    def test_adds_annotation(self):
        fig = go.Figure()
        _apply_watermark(fig)
        annots = fig.layout.annotations
        assert len(annots) >= 1
        texts = [a.text for a in annots]
        assert "quantoshi.xyz" in texts

    def test_returns_figure(self):
        fig = go.Figure()
        result = _apply_watermark(fig)
        assert isinstance(result, go.Figure)


class TestInterpQrPrice:
    def test_known_quantile(self):
        t = yr_to_t(2025, M.genesis)
        if 0.5 in M.qr_fits:
            price = _interp_qr_price(0.5, t, M.qr_fits)
            expected = qr_price(0.5, t, M.qr_fits)
            assert abs(price - expected) / expected < 0.01

    def test_interpolated_quantile(self):
        t = yr_to_t(2025, M.genesis)
        p_interp = _interp_qr_price(0.075, t, M.qr_fits)
        if 0.05 in M.qr_fits and 0.1 in M.qr_fits:
            p_05 = qr_price(0.05, t, M.qr_fits)
            p_10 = qr_price(0.10, t, M.qr_fits)
            assert p_05 <= p_interp <= p_10


class TestFreqStepDays:
    def test_all_frequencies(self):
        assert _FREQ_STEP_DAYS["Daily"] == 1
        assert _FREQ_STEP_DAYS["Weekly"] == 7
        assert _FREQ_STEP_DAYS["Monthly"] == 30
        assert _FREQ_STEP_DAYS["Quarterly"] == 91
        assert _FREQ_STEP_DAYS["Annually"] == 365


class TestMcFanPcts:
    def test_has_median(self):
        assert 0.50 in _MC_FAN_PCTS

    def test_count(self):
        assert len(_MC_FAN_PCTS) == 6


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Figure builder smoke tests — verify each tab produces a go.Figure
# ═══════════════════════════════════════════════════════════════════════════════

from figures import (build_bubble_figure, build_heatmap_figure,
                     build_dca_figure, build_retire_figure,
                     build_supercharge_figure)


class TestBuildBubbleFigure:
    def test_returns_figure(self):
        p = {
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "xscale": "Log",
            "yscale": "Log",
            "xmin": 2012,
            "xmax": 2030,
            "ymin": -2,
            "ymax": 8,
            "shade": True,
            "show_ols": False,
            "show_data": True,
            "show_today": True,
            "show_legend": False,
            "show_comp": True,
            "show_sup": False,
            "n_future": 3,
            "ptsize": 2,
            "ptalpha": 0.2,
            "stack": 1.0,
            "show_stack": False,
            "lots": [],
            "use_lots": False,
            "auto_y": True,
        }
        fig = build_bubble_figure(M, p)
        assert isinstance(fig, go.Figure)

    def test_no_quantiles(self):
        p = {
            "selected_qs": [],
            "xscale": "Log",
            "yscale": "Log",
            "xmin": 2012,
            "xmax": 2030,
            "ymin": -2,
            "ymax": 8,
            "shade": False,
            "show_ols": False,
            "show_data": True,
            "show_today": True,
            "show_legend": False,
            "show_comp": False,
            "show_sup": False,
            "n_future": 0,
            "ptsize": 2,
            "ptalpha": 0.2,
            "stack": 1.0,
            "show_stack": False,
            "lots": [],
            "use_lots": False,
            "auto_y": False,
        }
        fig = build_bubble_figure(M, p)
        assert isinstance(fig, go.Figure)


class TestBuildHeatmapFigure:
    def test_returns_figure(self):
        yr_now = pd.Timestamp.today().year
        p = {
            "entry_yr": yr_now,
            "entry_q": 50.0,
            "exit_yrs": list(range(yr_now + 1, yr_now + 6)),
            "exit_qs": [0.1, 0.5, 0.9] if all(q in M.qr_fits for q in [0.1, 0.5, 0.9]) else [0.5],
            "mode": "Segmented",
            "b1": 0,
            "b2": 20,
            "c_lo": "#d73027",
            "c_mid1": "#fee08b",
            "c_mid2": "#d9ef8b",
            "c_hi": "#1a9850",
            "grad": 32,
            "vfmt": "cagr",
            "cell_fs": 10,
            "show_colorbar": True,
            "live_price": None,
            "stack": 1.0,
            "use_lots": False,
            "lots": [],
        }
        fig = build_heatmap_figure(M, p)
        assert isinstance(fig, go.Figure)


class TestBuildDcaFigure:
    """build_dca_figure returns (fig, mc_result) tuple."""

    def test_returns_figure(self):
        p = {
            "start_yr": 2024,
            "end_yr": 2034,
            "start_stack": 0.0,
            "amount": 500,
            "freq": "Monthly",
            "disp_mode": "btc",
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "log_y": True,
            "show_today": True,
            "dual_y": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": False,
        }
        fig, mc_result = build_dca_figure(M, p)
        assert isinstance(fig, go.Figure)

    def test_usd_display_mode(self):
        p = {
            "start_yr": 2024,
            "end_yr": 2034,
            "start_stack": 0.0,
            "amount": 500,
            "freq": "Monthly",
            "disp_mode": "usd",
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "log_y": False,
            "show_today": False,
            "dual_y": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": False,
        }
        fig, _ = build_dca_figure(M, p)
        assert isinstance(fig, go.Figure)

    def test_with_stack_cellerator(self):
        p = {
            "start_yr": 2024,
            "end_yr": 2034,
            "start_stack": 0.0,
            "amount": 500,
            "freq": "Monthly",
            "disp_mode": "btc",
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "log_y": True,
            "show_today": True,
            "dual_y": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": True,
            "sc_loan": 1200,
            "sc_rate": 13,
            "sc_term": 12,
            "sc_type": "interest_only",
            "sc_repeats": 0,
            "sc_entry_mode": "model",
            "sc_custom_price": 100000,
            "sc_tax": 33,
            "sc_rollover": False,
        }
        fig, _ = build_dca_figure(M, p)
        assert isinstance(fig, go.Figure)

    def test_annotations_present_with_dual_y(self):
        """Verify right-edge USD annotations are added when dual_y is on."""
        p = {
            "start_yr": 2024,
            "end_yr": 2034,
            "start_stack": 0.0,
            "amount": 500,
            "freq": "Monthly",
            "disp_mode": "btc",
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "log_y": True,
            "show_today": False,
            "dual_y": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": False,
        }
        fig, _ = build_dca_figure(M, p)
        annots = fig.layout.annotations
        # Should have watermark + at least 1 USD annotation per quantile
        has_price_annot = any("$" in (a.text or "") for a in annots)
        if p["selected_qs"]:
            assert has_price_annot, "Expected USD price annotations with dual_y enabled"


class TestBuildRetireFigure:
    """build_retire_figure returns (fig, mc_result) tuple."""

    def test_returns_figure(self):
        p = {
            "start_yr": 2031,
            "end_yr": 2075,
            "start_stack": 1.0,
            "withdrawal": 5000,
            "freq": "Monthly",
            "inflation": 4.0,
            "disp_mode": "usd",
            "selected_qs": [0.1, 0.25] if all(q in M.qr_fits for q in [0.1, 0.25]) else [0.5],
            "log_y": True,
            "show_today": True,
            "dual_y": True,
            "annotate": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
        }
        fig, _ = build_retire_figure(M, p)
        assert isinstance(fig, go.Figure)


class TestBuildSuperchargeFigure:
    """build_supercharge_figure returns (fig, mc_result) tuple."""

    def test_mode_a_returns_figure(self):
        p = {
            "mode": "A",
            "start_stack": 1.0,
            "delays": [0, 1, 2],
            "start_yr": 2033,
            "end_yr": 2075,
            "freq": "Annually",
            "inflation": 4.0,
            "withdrawal": 100000,
            "disp_mode": "usd",
            "selected_qs": [0.1] if 0.1 in M.qr_fits else [0.5],
            "log_y": True,
            "show_today": True,
            "annotate": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "shade": True,
            "display_q": 0.05 if 0.05 in M.qr_fits else 0.5,
        }
        fig, _ = build_supercharge_figure(M, p)
        assert isinstance(fig, go.Figure)

    def test_mode_b_returns_figure(self):
        p = {
            "mode": "B",
            "start_stack": 1.0,
            "delays": [0, 1, 2],
            "start_yr": 2033,
            "end_yr": 2075,
            "target_yr": 2060,
            "freq": "Annually",
            "inflation": 4.0,
            "withdrawal": 100000,
            "disp_mode": "usd",
            "selected_qs": [0.1] if 0.1 in M.qr_fits else [0.5],
            "log_y": True,
            "show_today": True,
            "annotate": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "shade": True,
            "display_q": 0.05 if 0.05 in M.qr_fits else 0.5,
        }
        fig, _ = build_supercharge_figure(M, p)
        assert isinstance(fig, go.Figure)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Edge case / regression tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFalsyZeroGuard:
    """Verify that 0 is handled correctly as input (not treated as falsy)."""

    def test_zero_inflation_retire(self):
        """Inflation=0 should not become a default value."""
        p = {
            "start_yr": 2031,
            "end_yr": 2075,
            "start_stack": 1.0,
            "withdrawal": 5000,
            "freq": "Monthly",
            "inflation": 0.0,
            "disp_mode": "usd",
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "log_y": False,
            "show_today": False,
            "dual_y": False,
            "annotate": False,
            "show_legend": False,
            "lots": [],
            "use_lots": False,
        }
        fig, _ = build_retire_figure(M, p)
        assert isinstance(fig, go.Figure)

    def test_zero_sc_rate(self):
        """SC interest rate=0 should work (interest-free loan)."""
        p = {
            "start_yr": 2024,
            "end_yr": 2034,
            "start_stack": 0.0,
            "amount": 500,
            "freq": "Monthly",
            "disp_mode": "btc",
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "log_y": True,
            "show_today": False,
            "dual_y": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": True,
            "sc_loan": 1200,
            "sc_rate": 0,  # zero interest
            "sc_term": 12,
            "sc_type": "interest_only",
            "sc_repeats": 0,
            "sc_entry_mode": "model",
            "sc_custom_price": 100000,
            "sc_tax": 0,  # zero tax too
            "sc_rollover": False,
        }
        fig, _ = build_dca_figure(M, p)
        assert isinstance(fig, go.Figure)


class TestAnnotationStagger:
    """Verify the annotation stagger logic doesn't crash with various quantile counts."""

    def _make_dca_params(self, n_qs):
        available = [q for q in M.QR_QUANTILES if 0.001 <= q <= 0.999]
        sel = available[:n_qs]
        return {
            "start_yr": 2024,
            "end_yr": 2034,
            "start_stack": 0.0,
            "amount": 500,
            "freq": "Monthly",
            "disp_mode": "btc",
            "selected_qs": sel,
            "log_y": True,
            "show_today": False,
            "dual_y": True,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": False,
        }

    def test_zero_quantiles(self):
        fig, _ = build_dca_figure(M, self._make_dca_params(0))
        assert isinstance(fig, go.Figure)

    def test_one_quantile(self):
        fig, _ = build_dca_figure(M, self._make_dca_params(1))
        assert isinstance(fig, go.Figure)

    def test_three_quantiles(self):
        fig, _ = build_dca_figure(M, self._make_dca_params(3))
        assert isinstance(fig, go.Figure)

    def test_five_quantiles(self):
        fig, _ = build_dca_figure(M, self._make_dca_params(5))
        assert isinstance(fig, go.Figure)

    def test_many_quantiles(self):
        fig, _ = build_dca_figure(M, self._make_dca_params(10))
        assert isinstance(fig, go.Figure)


class TestSnapshotControlsCompleteness:
    """Verify snapshot controls list is self-consistent."""

    @pytest.mark.skipif(_SNAPSHOT_CONTROLS is None, reason="app.py import failed")
    def test_no_duplicate_controls(self):
        ids = [cid for cid, _ in _SNAPSHOT_CONTROLS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    @pytest.mark.skipif(_CHECKLIST_OPTIONS is None, reason="app.py import failed")
    def test_checklist_options_cover_snapshot(self):
        """Every checklist in snapshot should have options defined."""
        snapshot_ids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
        for cid in _CHECKLIST_OPTIONS:
            assert cid in snapshot_ids, f"{cid} in _CHECKLIST_OPTIONS but not in _SNAPSHOT_CONTROLS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
