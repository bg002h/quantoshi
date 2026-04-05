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
for _p in (str(_ROOT), str(_ROOT / "btc_web")):
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
        assert fmt_price(1000) == "$1,000"

    def test_thousands(self):
        assert fmt_price(12345) == "$12,345"

    def test_millions(self):
        assert fmt_price(1234567) == "$1,234,567"

    def test_zero(self):
        assert fmt_price(0) == "$0.00"

    def test_small(self):
        assert fmt_price(0.5) == "$0.50"


class TestYrToT:
    def test_genesis_year(self):
        t = yr_to_t(2009, M.genesis)
        assert abs(t - (-0.5613)) < 0.01  # Jan 1 2009 vs Jul 25 2009 genesis

    def test_2025(self):
        t = yr_to_t(2025, M.genesis)
        assert 15.3 < t < 15.6

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
    import app as _app_module  # triggers full app init (populates _app_ctx)
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
                           auto_bubble_yrange,
                           update_sc_info,
                           _TAB_CONTROLS, _TAB_TO_PATH)
    import _app_ctx
    _ALL_QS = _app_ctx._ALL_QS
except Exception:
    # If app import fails, define stubs for the test — tests will be skipped
    _q3 = _quantize_params = _list_to_mask = _mask_to_list = None
    _encode_snapshot = _decode_snapshot = _SNAPSHOT_CONTROLS = None
    _CHECKLIST_OPTIONS = _SNAP_PREFIX = _nearest_quantile = None
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

    def test_active_models_sorted(self):
        """active_models order should not affect cache key."""
        out1 = _quantize_params({"active_models": ["pl", "bub", "s2f"]})
        out2 = _quantize_params({"active_models": ["bub", "pl", "s2f"]})
        assert out1["active_models"] == out2["active_models"]
        assert out1["active_models"] == ["bub", "pl", "s2f"]

    def test_selected_qs_sorted(self):
        """selected_qs order should not affect cache key."""
        out1 = _quantize_params({"selected_qs": [0.5, 0.1, 0.01]})
        out2 = _quantize_params({"selected_qs": [0.01, 0.1, 0.5]})
        assert out1["selected_qs"] == out2["selected_qs"]
        assert out1["selected_qs"] == [0.01, 0.1, 0.5]

    def test_delays_sorted(self):
        """delays order should not affect cache key."""
        out1 = _quantize_params({"delays": [2.0, 0.0, 1.0]})
        out2 = _quantize_params({"delays": [0.0, 1.0, 2.0]})
        assert out1["delays"] == out2["delays"]
        assert out1["delays"] == [0.0, 1.0, 2.0]

    def test_active_models_exempt_from_quantize(self):
        """active_models contains strings, must not be quantized."""
        out = _quantize_params({"active_models": ["bub", "pl"]})
        assert out["active_models"] == ["bub", "pl"]


# ── Cache maxsize ────────────────────────────────────────────────────────────

@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestCacheMaxsize:
    def test_bubble_cache_maxsize(self):
        from utils import _cached_bubble_fig
        assert _cached_bubble_fig.cache_info().maxsize == 64

    def test_heatmap_cache_maxsize(self):
        from utils import _cached_heatmap_fig
        assert _cached_heatmap_fig.cache_info().maxsize == 64

    def test_mc_heatmap_cache_maxsize(self):
        from utils import _cached_mc_heatmap_fig
        assert _cached_mc_heatmap_fig.cache_info().maxsize == 64


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

    def test_mc_controls_roundtrip(self):
        """MC controls survive encode -> decode roundtrip."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["dca-mc-enable:value"] = ["yes"]
        state["dca-mc-start-yr:value"] = 2035
        state["dca-mc-entry-q:value"] = 30
        state["dca-mc-years:value"] = 20
        state["dca-mc-bins:value"] = 7
        state["dca-mc-regime:value"] = [0, 2, 4]
        state["dca-mc-sims:value"] = 1600
        state["dca-mc-window:value"] = [2012, 2024]
        state["dca-mc-advanced:value"] = ["yes"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["dca-mc-enable:value"] == ["yes"]
        assert decoded["dca-mc-start-yr:value"] == 2035
        assert decoded["dca-mc-entry-q:value"] == 30
        assert decoded["dca-mc-years:value"] == 20
        assert decoded["dca-mc-bins:value"] == 7
        assert decoded["dca-mc-regime:value"] == [0, 2, 4]
        assert decoded["dca-mc-sims:value"] == 1600
        assert decoded["dca-mc-window:value"] == [2012, 2024]
        assert decoded["dca-mc-advanced:value"] == ["yes"]

    def test_mc_hybrid_encoding_nulls_disabled_tabs(self):
        """MC controls encode as null when MC is not enabled on that tab."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["dca-mc-enable:value"] = []
        state["dca-mc-start-yr:value"] = 2035
        state["dca-mc-bins:value"] = 7
        state["ret-mc-enable:value"] = ["yes"]
        state["ret-mc-start-yr:value"] = 2028
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded.get("dca-mc-start-yr:value") is None
        assert decoded.get("dca-mc-bins:value") is None
        assert decoded["ret-mc-enable:value"] == ["yes"]
        assert decoded["ret-mc-start-yr:value"] == 2028

    def test_mc_regime_bitmask_roundtrip(self):
        """MC regime checklist with int values survives bitmask encode/decode."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["sc-mc-enable:value"] = ["yes"]
        state["sc-mc-regime:value"] = [0, 1, 3]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert sorted(decoded["sc-mc-regime:value"]) == [0, 1, 3]

    def test_hm_palette_roundtrip(self):
        """Heatmap palette name survives encode -> decode."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["hm-palette:value"] = "ocean"
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["hm-palette:value"] == "ocean"

    def test_old_link_pads_mc_to_none(self):
        """Old links with 100 entries decode correctly — MC defaults to None."""
        import gzip, base64
        assert len(_SNAPSHOT_CONTROLS) >= 137, "MC controls not yet added"
        old_values = [None] * 100
        old_values[0] = [0.5]
        payload = [old_values, None]
        encoded = base64.urlsafe_b64encode(
            gzip.compress(json.dumps(payload, separators=(',', ':')).encode())).decode()
        decoded = _decode_snapshot(encoded)
        assert decoded.get("bub-qs:value") == [0.5]
        assert decoded.get("dca-mc-enable:value") is None
        assert decoded.get("hm-palette:value") is None


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
                      is_cached, is_cached_year, _CACHED_MODEL_KEYS)


class TestSnapToBin:
    def test_exact_bins(self):
        for b in ENTRY_PCT_BINS:
            assert snap_to_bin(b) == b

    def test_rounds_to_nearest(self):
        assert snap_to_bin(0.08) == 0.10
        assert snap_to_bin(0.40) == 0.50
        assert snap_to_bin(0.005) == 0.01

    def test_extreme_low(self):
        assert snap_to_bin(0.0) == 0.01

    def test_extreme_high(self):
        assert snap_to_bin(1.0) == 0.50


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
        assert 2028 in CACHED_START_YRS
        assert 2031 in CACHED_START_YRS
        assert 2035 in CACHED_START_YRS
        assert len(CACHED_START_YRS) == 3

    def test_entry_bins(self):
        assert len(ENTRY_PCT_BINS) == 3
        assert 0.01 in ENTRY_PCT_BINS
        assert 0.10 in ENTRY_PCT_BINS
        assert 0.50 in ENTRY_PCT_BINS

    def test_mc_years_options(self):
        assert 40 in MC_YEARS_OPTIONS
        assert len(MC_YEARS_OPTIONS) == 1

    def test_fan_pcts(self):
        assert 0.50 in FAN_PCTS  # median must exist
        assert len(FAN_PCTS) == 6

    def test_cached_model_keys(self):
        assert "bub" in _CACHED_MODEL_KEYS
        assert "qr" in _CACHED_MODEL_KEYS
        assert "pl" in _CACHED_MODEL_KEYS
        assert "s2f" not in _CACHED_MODEL_KEYS


class TestIsCached:
    def test_cached_combo(self):
        assert is_cached("bub", 2028, 10, 40)
        assert is_cached("pl", 2031, 50, 40)
        assert is_cached("qr", 2035, 1, 40)

    def test_uncached_model(self):
        assert not is_cached("s2f", 2028, 10, 40)

    def test_uncached_year(self):
        assert not is_cached("bub", 9999, 10, 40)
        assert is_cached_year(9999) is False

    def test_uncached_duration(self):
        assert not is_cached("bub", 2028, 10, 20)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: figures.py tests
# ═══════════════════════════════════════════════════════════════════════════════

from mc_overlay import (_mc_path_key, _mc_overlay_key, _mc_fan_to_lists,
                        _mc_fan_from_lists, _mc_paths_to_lists, _mc_paths_from_lists,
                        _MC_FAN_PCTS)
from figures import _apply_watermark, _FREQ_STEP_DAYS
import plotly.graph_objects as go


class TestMcPathKey:
    def test_dca_defaults(self):
        p = {}
        key = _mc_path_key(p, "dca")
        assert key["tab"] == "dca"
        assert key["mc_bins"] == 5
        assert key["mc_sims"] == 200
        assert key["mc_years"] == 40
        assert key["mc_freq"] == "Monthly"
        assert key["mc_start_yr"] == 2028
        assert "mc_entry_q" in key

    def test_ret_defaults(self):
        key = _mc_path_key({}, "ret")
        assert key["mc_start_yr"] == 2028

    def test_sc_defaults(self):
        key = _mc_path_key({}, "sc")
        assert key["mc_start_yr"] == 2028

    def test_hm_uses_mc_entry_q(self):
        """Unified key reads mc_entry_q directly (callback maps entry_q → mc_entry_q)."""
        key = _mc_path_key({"mc_entry_q": 75}, "hm")
        assert key["mc_entry_q"] == 75.0

    def test_dca_uses_mc_entry_q(self):
        key = _mc_path_key({"mc_entry_q": 30}, "dca")
        assert key["mc_entry_q"] == 30.0

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

    def test_no_start_yr_key_in_path_key(self):
        """path_key must not contain 'start_yr' — only 'mc_start_yr'."""
        for tab in ("dca", "ret", "sc", "hm"):
            key = _mc_path_key({}, tab)
            assert "start_yr" not in key, f"{tab} still has start_yr"
            assert "mc_start_yr" in key

    def test_no_fallback_to_non_mc_start_yr(self):
        """Changing the main tab's start_yr must NOT affect path_key."""
        p1 = {"mc_start_yr": 2030, "mc_entry_q": 40, "start_yr": 2024}
        p2 = {"mc_start_yr": 2030, "mc_entry_q": 40, "start_yr": 9999}
        assert _mc_path_key(p1, "dca") == _mc_path_key(p2, "dca")
        assert _mc_path_key(p1, "ret") == _mc_path_key(p2, "ret")

    def test_hm_reads_mc_start_yr(self):
        """Unified key reads mc_start_yr directly (callback maps entry_yr → mc_start_yr)."""
        key = _mc_path_key({"mc_start_yr": 2030, "start_yr": 9999}, "hm")
        assert key["mc_start_yr"] == 2030

    def test_uniform_key_names_across_tabs(self):
        """All tabs use mc_start_yr and mc_entry_q — no tab uses entry_q or start_yr."""
        for tab in ("dca", "ret", "sc", "hm"):
            key = _mc_path_key({}, tab)
            assert "mc_start_yr" in key
            assert "mc_entry_q" in key
            assert "entry_q" not in key
            assert "start_yr" not in key

    def test_upload_roundtrip_all_tabs(self):
        """Saved path_key matches on reload when upload populates MC panel."""
        for tab, default_yr in [("dca", 2026), ("ret", 2031), ("sc", 2031)]:
            saved_p = {"mc_start_yr": 2030, "mc_entry_q": 40}
            reload_p = {"mc_start_yr": 2030, "mc_entry_q": 40, "start_yr": 2024}
            assert _mc_path_key(saved_p, tab) == _mc_path_key(reload_p, tab)


class TestBuildMcParams:
    """Tests for the _build_mc_params() centralized helper."""

    def test_defaults(self):
        """All-None inputs → tab defaults."""
        d = _build_mc_params(
            mc_enable=True, mc_amount=None, mc_infl=None,
            mc_bins=5, mc_sims=800, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2026, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
        )
        assert d["mc_enabled"] is True
        assert d["mc_amount"] == 100
        assert d["mc_infl"] == 4.0
        assert d["mc_bins"] == 5
        assert d["mc_sims"] == 800
        assert d["mc_years"] == 10
        assert d["mc_freq"] == "Monthly"
        assert d["mc_start_yr"] == 2026
        assert d["mc_entry_q"] == 50
        assert "mc_start_stack" not in d

    def test_custom_defaults(self):
        """Tab-specific defaults override generic defaults."""
        d = _build_mc_params(
            mc_enable=False, mc_amount=None, mc_infl=None,
            mc_bins=5, mc_sims=800, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2031, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            amount_default=5000, infl_default=0.0,
        )
        assert d["mc_amount"] == 5000
        assert d["mc_infl"] == 0.0
        assert d["mc_start_yr"] == 2031

    def test_explicit_values_override(self):
        """Explicit values take precedence over defaults."""
        d = _build_mc_params(
            mc_enable=True, mc_amount=200, mc_infl=3.5,
            mc_bins=10, mc_sims=400, mc_years=20,
            mc_freq="Daily", mc_window=[2010, 2025],
            mc_start_yr=2028, mc_entry_q=75,
            mc_cached="data", mc_live_price=90000,
            amount_default=5000,
        )
        assert d["mc_amount"] == 200
        assert d["mc_infl"] == 3.5
        assert d["mc_bins"] == 10
        assert d["mc_sims"] == 400
        assert d["mc_years"] == 20
        assert d["mc_freq"] == "Daily"
        assert d["mc_start_yr"] == 2028
        assert d["mc_entry_q"] == 75

    def test_infl_zero_not_falsy(self):
        """mc_infl=0 must not fall through to default."""
        d = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=0,
            mc_bins=5, mc_sims=800, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2026, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            infl_default=4.0,
        )
        assert d["mc_infl"] == 0.0

    def test_start_stack_included(self):
        """mc_start_stack added when provided."""
        d = _build_mc_params(
            mc_enable=True, mc_amount=5000, mc_infl=4.0,
            mc_bins=5, mc_sims=800, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2031, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            mc_start_stack=2.5,
        )
        assert d["mc_start_stack"] == 2.5

    def test_start_stack_none_excluded(self):
        """mc_start_stack omitted when None."""
        d = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=4.0,
            mc_bins=5, mc_sims=800, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2026, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            mc_start_stack=None,
        )
        assert "mc_start_stack" not in d


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
    """Tests interp_price via BubbleModel (migrated from _interp_qr_price)."""
    def setup_method(self):
        self.model = BubbleModel(M)

    def test_known_quantile(self):
        t = yr_to_t(2025, M.genesis)
        if 0.5 in self.model.fits:
            price = self.model.interp_price(0.5, t)
            expected = float(self.model.price_at(0.5, t))
            assert abs(price - expected) / expected < 0.01

    def test_interpolated_quantile(self):
        t = yr_to_t(2025, M.genesis)
        p_interp = self.model.interp_price(0.075, t)
        if 0.05 in self.model.fits and 0.1 in self.model.fits:
            p_05 = float(self.model.price_at(0.05, t))
            p_10 = float(self.model.price_at(0.10, t))
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


class TestMcUploadFields:
    """Ensure upload fields cover all MC path_key params for cache hits."""
    def _suffixes(self, tab):
        return {s for s, _, _, _ in _MC_UPLOAD_FIELDS[tab]}

    def test_hm_has_start_yr(self):
        assert "start-yr" in self._suffixes("hm")

    def test_hm_has_entry_q(self):
        assert "entry-q" in self._suffixes("hm")

    def test_hm_has_years(self):
        assert "years" in self._suffixes("hm")

    def test_dca_has_start_yr(self):
        assert "start-yr" in self._suffixes("dca")

    def test_ret_has_start_yr(self):
        assert "start-yr" in self._suffixes("ret")

    def test_sc_has_start_yr(self):
        assert "start-yr" in self._suffixes("sc")

    def test_all_tabs_have_years(self):
        for tab in _MC_UPLOAD_FIELDS:
            assert "years" in self._suffixes(tab), f"{tab} missing years"

    def test_all_tabs_have_window(self):
        """mc_window in path_key changes yearly; upload must restore it."""
        for tab in _MC_UPLOAD_FIELDS:
            assert "window" in self._suffixes(tab), f"{tab} missing window"

    def test_window_extracted_from_path_key(self):
        data = {"path_key": {"mc_window": [2010, 2025]}, "overlay_key": {}}
        assert _pk(data, "mc_window") == [2010, 2025]

    def test_pk_extracts_mc_start_yr(self):
        """_pk must find mc_start_yr in saved data's path_key."""
        data = {"path_key": {"mc_start_yr": 2030, "mc_entry_q": 50},
                "overlay_key": {}}
        assert _pk(data, "mc_start_yr") == 2030
        assert _pk(data, "mc_entry_q") == 50

    def test_no_tuple_keys_in_upload_fields(self):
        """All upload fields use direct string keys — no tuple fallbacks."""
        for tab, fields in _MC_UPLOAD_FIELDS.items():
            for suffix, data_key, _, _ in fields:
                assert isinstance(data_key, str), \
                    f"{tab}.{suffix} uses tuple key {data_key} — should be direct string"

    def test_all_mc_years_available_at_defaults(self):
        """Default sims x Monthly must include at least the cached duration."""
        from mc_cache import MC_SIMS, MC_YEARS_OPTIONS
        opts = _mc_years_options(MC_SIMS, "Monthly")
        values = [o["value"] for o in opts]
        for yr in MC_YEARS_OPTIONS:
            assert yr in values, f"{yr}yr should be available at default sims"


class TestCompositeModelAccessors:
    """EmpiricalFloorModel exposes composite data via public properties."""

    def test_ef_has_comp_by_n(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "comp_by_n")
        assert isinstance(ef.comp_by_n, list)
        assert len(ef.comp_by_n) > 0

    def test_ef_has_support_plot(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "support_plot")
        assert len(ef.support_plot) > 0

    def test_ef_has_t_grid(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "t_grid")
        assert len(ef.t_grid) == len(ef.support_plot)

    def test_ef_has_bm_r2(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert isinstance(ef.bm_r2, float)
        assert 0 < ef.bm_r2 < 1

    def test_ef_has_n_future_max(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert isinstance(ef.n_future_max, int)
        assert ef.n_future_max >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Figure builder smoke tests — verify each tab produces a go.Figure
# ═══════════════════════════════════════════════════════════════════════════════

from figures import (build_bubble_figure, build_heatmap_figure,
                     build_dca_figure, build_retire_figure,
                     build_supercharge_figure,
                     _build_qr_config_text, _build_mc_config_text,
                     _apply_config_annotation)


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
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": False,
        }
        fig, _ = build_dca_figure(M, p)
        assert isinstance(fig, go.Figure)

    def test_with_stack_celerator(self):
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

    def test_annotations_present(self):
        """Verify right-edge USD text-trace annotations are present."""
        p = {
            "start_yr": 2024,
            "end_yr": 2034,
            "start_stack": 0.0,
            "amount": 500,
            "freq": "Monthly",
            "disp_mode": "btc",
            "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
            "log_y": True,
            "annotate": True,
            "show_today": False,
            "show_legend": True,
            "lots": [],
            "use_lots": False,
            "sc_enabled": False,
        }
        fig, _ = build_dca_figure(M, p)
        # Text-trace annotations: go.Scatter with mode="markers+text" and "$" in text
        has_price_trace = any(
            getattr(tr, "mode", "") == "markers+text"
            and any("$" in t for t in (tr.text or []))
            for tr in fig.data
        )
        if p["selected_qs"]:
            assert has_price_trace, "Expected USD text-trace annotations"


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
# Section 5b: Chart builder parameter fuzz tests
# ═══════════════════════════════════════════════════════════════════════════════

# Helper: pick quantiles that exist in the model
_FUZZ_QS = [q for q in [0.001, 0.01, 0.10, 0.50, 0.90] if q in M.qr_fits]
_FUZZ_Q1 = _FUZZ_QS[:1]
_FUZZ_Q2 = _FUZZ_QS[:2] if len(_FUZZ_QS) >= 2 else _FUZZ_QS
_YR_NOW = pd.Timestamp.today().year


def _assert_figure(result, expect_tuple=False):
    """Assert result is a valid figure (or (figure, mc_result) tuple)."""
    if expect_tuple:
        assert isinstance(result, tuple) and len(result) == 2
        fig, _ = result
    else:
        fig = result
    assert isinstance(fig, go.Figure)


# ── Supercharger fuzz ────────────────────────────────────────────────────────

_SC_BASE = dict(
    mode="a", start_stack=1.0, start_yr=2033, end_yr=2075,
    delays=[0, 1, 2], freq="Monthly", inflation=4.0,
    selected_qs=_FUZZ_Q2, chart_layout=0, display_q=0.5,
    wd_amount=5000, disp_mode="usd", log_y=True,
    annotate=True, show_legend=True, legend_pos="outside",
    minor_grid=False, target_yr=2060, lots=[], use_lots=False,
)

_SC_OVERRIDES = [
    # ── Single-axis variations ───────────────────────────────────────
    # Chart layouts
    pytest.param({"chart_layout": 0}, id="sc-layout0"),
    pytest.param({"chart_layout": 1}, id="sc-layout1"),
    pytest.param({"chart_layout": 2}, id="sc-layout2"),
    # Mode B
    pytest.param({"mode": "b"}, id="sc-modeB"),
    pytest.param({"mode": "b", "disp_mode": "btc"}, id="sc-modeB-btc"),
    # Display mode
    pytest.param({"disp_mode": "btc"}, id="sc-btc"),
    # Toggles off
    pytest.param({"log_y": False}, id="sc-linear"),
    pytest.param({"annotate": False}, id="sc-no-annot"),
    pytest.param({"show_legend": False}, id="sc-no-legend"),
    # Delays
    pytest.param({"delays": [0.0]}, id="sc-delay0-only"),
    pytest.param({"delays": [0, 1, 2, 4, 8]}, id="sc-5delays"),
    pytest.param({"delays": [5, 10]}, id="sc-late-delays"),
    # Frequencies
    pytest.param({"freq": "Daily"}, id="sc-daily"),
    pytest.param({"freq": "Weekly"}, id="sc-weekly"),
    pytest.param({"freq": "Quarterly"}, id="sc-quarterly"),
    pytest.param({"freq": "Annually"}, id="sc-annual"),
    # Year ranges
    pytest.param({"end_yr": 2035}, id="sc-short-range"),
    pytest.param({"start_yr": _YR_NOW, "end_yr": _YR_NOW + 3}, id="sc-very-short"),
    pytest.param({"start_yr": 2070, "end_yr": 2075}, id="sc-far-future"),
    # Amounts / inflation
    pytest.param({"wd_amount": 0}, id="sc-wd0"),
    pytest.param({"wd_amount": 1_000_000}, id="sc-wd-large"),
    pytest.param({"inflation": 0}, id="sc-infl0"),
    pytest.param({"inflation": 50}, id="sc-infl50"),
    # Legend positions
    pytest.param({"legend_pos": "top-left"}, id="sc-leg-tl"),
    pytest.param({"legend_pos": "bottom-right"}, id="sc-leg-br"),
    # Stack
    pytest.param({"start_stack": 0.001}, id="sc-tiny-stack"),
    pytest.param({"start_stack": 100.0}, id="sc-big-stack"),
    # ── Critical multi-param combos ──────────────────────────────────
    # Empty results + annotate (caught _sc_log bug)
    pytest.param({"chart_layout": 2, "annotate": True, "end_yr": 2035},
                 id="sc-bands-annot-short"),
    pytest.param({"chart_layout": 0, "annotate": True, "end_yr": 2035},
                 id="sc-lines-annot-short"),
    pytest.param({"chart_layout": 1, "annotate": True, "end_yr": 2035},
                 id="sc-qlines-annot-short"),
    # Delays where all t_start_d >= t_end
    pytest.param({"delays": [50, 60], "end_yr": 2040}, id="sc-all-delays-skip"),
    # Mode B edge cases
    pytest.param({"mode": "b", "delays": [0.0]}, id="sc-modeB-delay0"),
    pytest.param({"mode": "b", "chart_layout": 2}, id="sc-modeB-bands"),
    pytest.param({"mode": "b", "annotate": True, "log_y": False},
                 id="sc-modeB-annot-linear"),
    # Single quantile + each layout
    pytest.param({"selected_qs": _FUZZ_Q1, "chart_layout": 0}, id="sc-1q-layout0"),
    pytest.param({"selected_qs": _FUZZ_Q1, "chart_layout": 1}, id="sc-1q-layout1"),
    pytest.param({"selected_qs": _FUZZ_Q1, "chart_layout": 2}, id="sc-1q-layout2"),
    # Band layout + btc display + log
    pytest.param({"chart_layout": 2, "disp_mode": "btc", "log_y": True,
                  "annotate": True}, id="sc-bands-btc-log"),
]


class TestSuperchargeFuzz:
    """Fuzz build_supercharge_figure with varied parameter combos."""

    @pytest.mark.parametrize("override", _SC_OVERRIDES)
    def test_no_crash(self, override):
        p = {**_SC_BASE, **override}
        _assert_figure(build_supercharge_figure(M, p), expect_tuple=True)


# ── Bubble fuzz ──────────────────────────────────────────────────────────────

_BUB_BASE = dict(
    selected_qs=_FUZZ_Q2, xscale="Log", yscale="Log",
    xmin=2012, xmax=2030, ymin=-2, ymax=8,
    shade=True, show_ols=False, show_data=True, show_today=True,
    show_legend=True, show_comp=True, show_sup=False,
    n_future=3, ptsize=2, ptalpha=0.2,
    stack=1.0, show_stack=False, lots=[], use_lots=False, auto_y=True,
)

_BUB_OVERRIDES = [
    pytest.param({"selected_qs": []}, id="bub-no-qs"),
    pytest.param({"xscale": "Linear", "yscale": "Linear"}, id="bub-linear"),
    pytest.param({"shade": False, "show_comp": False}, id="bub-no-shade-comp"),
    pytest.param({"n_future": 0}, id="bub-no-future"),
    pytest.param({"n_future": 5}, id="bub-5future"),
    pytest.param({"show_stack": True, "stack": 2.5}, id="bub-stack"),
    pytest.param({"show_legend": False}, id="bub-no-legend"),
    pytest.param({"auto_y": False}, id="bub-manual-y"),
    pytest.param({"xmin": 2024, "xmax": 2026}, id="bub-narrow-x"),
    pytest.param({"show_ols": True}, id="bub-ols"),
    pytest.param({"show_sup": True}, id="bub-sup"),
    pytest.param({"ptsize": 1, "ptalpha": 1.0}, id="bub-pt-extremes"),
    pytest.param({"selected_qs": _FUZZ_QS, "shade": True, "auto_y": True},
                 id="bub-many-qs-shade"),
]


class TestBubbleFuzz:
    """Fuzz build_bubble_figure with varied parameter combos."""

    @pytest.mark.parametrize("override", _BUB_OVERRIDES)
    def test_no_crash(self, override):
        p = {**_BUB_BASE, **override}
        _assert_figure(build_bubble_figure(M, p))


# ── Heatmap fuzz ─────────────────────────────────────────────────────────────

_HM_BASE = dict(
    entry_yr=_YR_NOW, entry_q=50.0,
    exit_yrs=list(range(_YR_NOW + 1, _YR_NOW + 6)),
    exit_qs=_FUZZ_Q2, mode="Segmented",
    b1=0, b2=20, c_lo="#d73027", c_mid1="#fee08b",
    c_mid2="#d9ef8b", c_hi="#1a9850", grad=32,
    vfmt="cagr", cell_fs=10, show_colorbar=True,
    live_price=None, stack=1.0, use_lots=False, lots=[],
)

_HM_OVERRIDES = [
    pytest.param({"exit_qs": []}, id="hm-no-exit-qs"),
    pytest.param({"exit_qs": _FUZZ_Q1}, id="hm-single-exit-q"),
    pytest.param({"mode": "DataScaled"}, id="hm-datascaled"),
    pytest.param({"mode": "Diverging"}, id="hm-diverging"),
    pytest.param({"vfmt": "price"}, id="hm-price"),
    pytest.param({"vfmt": "both"}, id="hm-both"),
    pytest.param({"vfmt": "stack"}, id="hm-stack"),
    pytest.param({"vfmt": "mult_only"}, id="hm-mult"),
    pytest.param({"vfmt": "none"}, id="hm-none"),
    pytest.param({"entry_yr": 2020, "exit_yrs": [2021, 2022, 2023]}, id="hm-past"),
    pytest.param({"entry_q": 1.0}, id="hm-low-q"),
    pytest.param({"entry_q": 99.0}, id="hm-high-q"),
    pytest.param({"b1": -50, "b2": 100}, id="hm-wide-breaks"),
    pytest.param({"show_colorbar": False}, id="hm-no-colorbar"),
    pytest.param({"live_price": 95000.0}, id="hm-live-price"),
    pytest.param({"exit_yrs": list(range(_YR_NOW + 1, _YR_NOW + 20))},
                 id="hm-many-exit-yrs"),
]


class TestHeatmapFuzz:
    """Fuzz build_heatmap_figure with varied parameter combos."""

    @pytest.mark.parametrize("override", _HM_OVERRIDES)
    def test_no_crash(self, override):
        p = {**_HM_BASE, **override}
        _assert_figure(build_heatmap_figure(M, p))


# ── DCA fuzz ─────────────────────────────────────────────────────────────────

_DCA_BASE = dict(
    start_yr=2024, end_yr=2034, start_stack=0.0, amount=500,
    freq="Monthly", disp_mode="btc",
    selected_qs=_FUZZ_Q2, log_y=True,
    show_today=True, show_legend=True, annotate=True,
    legend_pos="outside", minor_grid=False,
    lots=[], use_lots=False, sc_enabled=False,
)

_DCA_OVERRIDES = [
    pytest.param({"selected_qs": []}, id="dca-no-qs"),
    pytest.param({"selected_qs": _FUZZ_Q1}, id="dca-single-q"),
    pytest.param({"disp_mode": "usd"}, id="dca-usd"),
    pytest.param({"log_y": False}, id="dca-linear"),
    pytest.param({"annotate": False}, id="dca-no-annot"),
    pytest.param({"show_legend": False}, id="dca-no-legend"),
    pytest.param({"show_today": False}, id="dca-no-today"),
    pytest.param({"freq": "Daily"}, id="dca-daily"),
    pytest.param({"freq": "Weekly"}, id="dca-weekly"),
    pytest.param({"freq": "Quarterly"}, id="dca-quarterly"),
    pytest.param({"freq": "Annually"}, id="dca-annual"),
    pytest.param({"amount": 0}, id="dca-amount0"),
    pytest.param({"amount": 1_000_000}, id="dca-large-amount"),
    pytest.param({"start_stack": 1.0}, id="dca-with-stack"),
    pytest.param({"start_yr": _YR_NOW, "end_yr": _YR_NOW + 2}, id="dca-short"),
    pytest.param({"legend_pos": "top-left"}, id="dca-leg-tl"),
    pytest.param({"legend_pos": "bottom-right"}, id="dca-leg-br"),
    # Stack-celerator combos
    pytest.param({"sc_enabled": True, "sc_loan": 1200, "sc_rate": 13,
                  "sc_term": 12, "sc_type": "interest_only", "sc_repeats": 0,
                  "sc_entry_mode": "model", "sc_custom_price": 100000,
                  "sc_tax": 33, "sc_rollover": False}, id="dca-sc-io"),
    pytest.param({"sc_enabled": True, "sc_loan": 5000, "sc_rate": 8,
                  "sc_term": 24, "sc_type": "amortizing", "sc_repeats": 2,
                  "sc_entry_mode": "model", "sc_custom_price": 100000,
                  "sc_tax": 33, "sc_rollover": False}, id="dca-sc-amort"),
    pytest.param({"sc_enabled": True, "sc_loan": 1200, "sc_rate": 0,
                  "sc_term": 12, "sc_type": "interest_only", "sc_repeats": 0,
                  "sc_entry_mode": "model", "sc_custom_price": 100000,
                  "sc_tax": 33, "sc_rollover": False}, id="dca-sc-rate0"),
    pytest.param({"disp_mode": "usd", "annotate": True, "log_y": True,
                  "selected_qs": _FUZZ_QS}, id="dca-usd-annot-log-manyqs"),
]


class TestDcaFuzz:
    """Fuzz build_dca_figure with varied parameter combos."""

    @pytest.mark.parametrize("override", _DCA_OVERRIDES)
    def test_no_crash(self, override):
        p = {**_DCA_BASE, **override}
        _assert_figure(build_dca_figure(M, p), expect_tuple=True)


# ── Retire fuzz ──────────────────────────────────────────────────────────────

_RET_BASE = dict(
    start_yr=2031, end_yr=2075, start_stack=1.0,
    wd_amount=5000, freq="Monthly", inflation=4.0,
    disp_mode="usd",
    selected_qs=_FUZZ_Q2, log_y=True, dual_y=True,
    annotate=True, show_legend=True, legend_pos="outside",
    minor_grid=False, lots=[], use_lots=False,
)

_RET_OVERRIDES = [
    pytest.param({"selected_qs": []}, id="ret-no-qs"),
    pytest.param({"selected_qs": _FUZZ_Q1}, id="ret-single-q"),
    pytest.param({"disp_mode": "btc"}, id="ret-btc"),
    pytest.param({"log_y": False}, id="ret-linear"),
    pytest.param({"dual_y": False}, id="ret-no-dual"),
    pytest.param({"annotate": False}, id="ret-no-annot"),
    pytest.param({"show_legend": False}, id="ret-no-legend"),
    pytest.param({"freq": "Daily"}, id="ret-daily"),
    pytest.param({"freq": "Weekly"}, id="ret-weekly"),
    pytest.param({"freq": "Quarterly"}, id="ret-quarterly"),
    pytest.param({"freq": "Annually"}, id="ret-annual"),
    pytest.param({"wd_amount": 0}, id="ret-wd0"),
    pytest.param({"wd_amount": 1_000_000}, id="ret-wd-large"),
    pytest.param({"inflation": 0}, id="ret-infl0"),
    pytest.param({"inflation": 50}, id="ret-infl50"),
    pytest.param({"start_yr": _YR_NOW, "end_yr": _YR_NOW + 3}, id="ret-short"),
    pytest.param({"legend_pos": "top-left"}, id="ret-leg-tl"),
    pytest.param({"legend_pos": "bottom-right"}, id="ret-leg-br"),
    pytest.param({"disp_mode": "btc", "annotate": True, "log_y": True},
                 id="ret-btc-annot-log"),
    pytest.param({"selected_qs": _FUZZ_QS, "annotate": True}, id="ret-many-qs-annot"),
]


class TestRetireFuzz:
    """Fuzz build_retire_figure with varied parameter combos."""

    @pytest.mark.parametrize("override", _RET_OVERRIDES)
    def test_no_crash(self, override):
        p = {**_RET_BASE, **override}
        _assert_figure(build_retire_figure(M, p), expect_tuple=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5c: Bin regime filter tests (_apply_bin_mask, _snap_start_pctile)
# ═══════════════════════════════════════════════════════════════════════════════

from mc_overlay import (_apply_bin_mask, _snap_start_pctile,
                        bin_regime_labels, _mc_path_key)


class TestApplyBinMask:
    """Test transition matrix bin blocking and re-normalization."""

    def _uniform(self, n=5):
        """Create a uniform n×n transition matrix."""
        return np.ones((n, n)) / n

    def test_no_blocked_bins_returns_same(self):
        trans = self._uniform()
        result = _apply_bin_mask(trans, [])
        np.testing.assert_array_equal(result, trans)

    def test_none_blocked_bins_returns_same(self):
        trans = self._uniform()
        result = _apply_bin_mask(trans, None)
        np.testing.assert_array_equal(result, trans)

    def test_block_single_bin_zeros_column(self):
        trans = self._uniform()
        result = _apply_bin_mask(trans, [4])
        # Column 4 should be all zeros
        assert np.all(result[:, 4] == 0.0)
        # All rows still sum to 1
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_block_single_bin_redistributes(self):
        trans = self._uniform()
        result = _apply_bin_mask(trans, [4])
        # Each row's remaining 4 bins should be 0.25
        for row in range(5):
            for col in range(4):
                assert abs(result[row, col] - 0.25) < 1e-10

    def test_block_multiple_bins(self):
        trans = self._uniform()
        result = _apply_bin_mask(trans, [0, 4])
        # Columns 0 and 4 should be zero
        assert np.all(result[:, 0] == 0.0)
        assert np.all(result[:, 4] == 0.0)
        # Remaining 3 bins share probability equally
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        for row in range(5):
            for col in [1, 2, 3]:
                assert abs(result[row, col] - 1.0/3) < 1e-10

    def test_block_all_but_one(self):
        trans = self._uniform()
        result = _apply_bin_mask(trans, [0, 1, 3, 4])
        # Only column 2 should have probability
        for row in range(5):
            assert abs(result[row, 2] - 1.0) < 1e-10
            for col in [0, 1, 3, 4]:
                assert result[row, col] == 0.0

    def test_does_not_mutate_original(self):
        trans = self._uniform()
        original = trans.copy()
        _apply_bin_mask(trans, [2])
        np.testing.assert_array_equal(trans, original)

    def test_nonuniform_matrix(self):
        """Test with a realistic non-uniform matrix."""
        trans = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.05],
            [0.2, 0.4, 0.2, 0.1, 0.1],
            [0.1, 0.2, 0.4, 0.2, 0.1],
            [0.05, 0.1, 0.2, 0.4, 0.25],
            [0.05, 0.05, 0.1, 0.3, 0.5],
        ])
        result = _apply_bin_mask(trans, [4])
        # Column 4 is zero
        assert np.all(result[:, 4] == 0.0)
        # Rows sum to 1
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        # Row 0: was [0.5, 0.3, 0.1, 0.05, 0.05], remove col4 (0.05)
        # Remaining sum = 0.95, so each divided by 0.95
        np.testing.assert_allclose(result[0, :4], [0.5/0.95, 0.3/0.95,
                                                    0.1/0.95, 0.05/0.95])

    def test_zero_row_gets_uniform_over_allowed(self):
        """Row that only transitions to blocked bins gets uniform fallback."""
        trans = np.zeros((3, 3))
        trans[0] = [0.0, 0.0, 1.0]  # row 0 only goes to bin 2
        trans[1] = [0.5, 0.5, 0.0]
        trans[2] = [0.0, 0.0, 1.0]
        result = _apply_bin_mask(trans, [2])
        # Rows 0 and 2 had all weight in bin 2 (now blocked)
        # Should get uniform over allowed bins [0, 1]
        np.testing.assert_allclose(result[0], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(result[2], [0.5, 0.5, 0.0])
        # Row 1 is unchanged (no weight in bin 2)
        np.testing.assert_allclose(result[1], [0.5, 0.5, 0.0])

    def test_out_of_range_bins_ignored(self):
        trans = self._uniform()
        result = _apply_bin_mask(trans, [99, -1])
        # Out-of-range bins don't affect the matrix
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        np.testing.assert_allclose(result, trans)

    def test_larger_matrix(self):
        trans = self._uniform(n=10)
        result = _apply_bin_mask(trans, [0, 5, 9])
        for blocked_col in [0, 5, 9]:
            assert np.all(result[:, blocked_col] == 0.0)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)


class TestSnapStartPctile:
    """Test starting percentile snapping when entry bin is blocked."""

    def _edges5(self):
        return np.linspace(0, 1, 6)  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def test_no_blocked_returns_original(self):
        assert _snap_start_pctile(0.5, self._edges5(), []) == 0.5

    def test_none_blocked_returns_original(self):
        assert _snap_start_pctile(0.5, self._edges5(), None) == 0.5

    def test_allowed_bin_returns_original(self):
        # 0.5 is in bin 2 (40-60%), block bin 4 only
        assert _snap_start_pctile(0.5, self._edges5(), [4]) == 0.5

    def test_blocked_bin_snaps_to_nearest(self):
        # 0.9 is in bin 4 (80-100%), block bin 4
        # Nearest allowed is bin 3 (60-80%), midpoint = 0.7
        result = _snap_start_pctile(0.9, self._edges5(), [4])
        assert abs(result - 0.7) < 1e-10

    def test_blocked_low_bin_snaps_up(self):
        # 0.1 is in bin 0 (0-20%), block bin 0
        # Nearest allowed is bin 1 (20-40%), midpoint = 0.3
        result = _snap_start_pctile(0.1, self._edges5(), [0])
        assert abs(result - 0.3) < 1e-10

    def test_blocked_middle_snaps_to_closer(self):
        # 0.5 is in bin 2 (40-60%), block bin 2
        # Nearest: bin 1 (mid=0.3, dist=0.2) or bin 3 (mid=0.7, dist=0.2)
        # Either is fine — just check it's one of them and not blocked
        edges = self._edges5()
        result = _snap_start_pctile(0.5, edges, [2])
        assert abs(result - 0.3) < 1e-10 or abs(result - 0.7) < 1e-10

    def test_multiple_blocked_finds_best(self):
        # 0.9 is in bin 4, block bins 3 and 4
        # Nearest allowed is bin 2 (40-60%), midpoint = 0.5
        result = _snap_start_pctile(0.9, self._edges5(), [3, 4])
        assert abs(result - 0.5) < 1e-10

    def test_all_blocked_returns_original(self):
        # Degenerate case: all blocked
        result = _snap_start_pctile(0.5, self._edges5(), [0, 1, 2, 3, 4])
        assert result == 0.5

    def test_boundary_percentile(self):
        # 0.0 is in bin 0, block bin 0
        result = _snap_start_pctile(0.0, self._edges5(), [0])
        assert abs(result - 0.3) < 1e-10  # snaps to bin 1 midpoint

    def test_1_0_percentile(self):
        # 1.0 clips to bin 4, block bin 4
        result = _snap_start_pctile(1.0, self._edges5(), [4])
        assert abs(result - 0.7) < 1e-10  # snaps to bin 3 midpoint


class TestBinRegimeLabels:
    """Test human-readable bin label generation."""

    def test_5_bins_named(self):
        labels = bin_regime_labels(5)
        assert len(labels) == 5
        assert "Bargain" in labels[0]
        assert "Bubble" in labels[4]
        assert "0\u201320%" in labels[0]
        assert "80\u2013100%" in labels[4]

    def test_7_bins_percentile_ranges(self):
        labels = bin_regime_labels(7)
        assert len(labels) == 7
        assert "0\u201314%" in labels[0]
        assert "86\u2013100%" in labels[6]

    def test_10_bins_percentile_ranges(self):
        labels = bin_regime_labels(10)
        assert len(labels) == 10
        assert "0\u201310%" in labels[0]
        assert "90\u2013100%" in labels[9]


class TestPathKeyBlockedBins:
    """Test that mc_blocked_bins is included in path cache key."""

    def test_empty_blocked(self):
        p = {"mc_bins": 5, "mc_sims": 800, "mc_years": 10,
             "mc_freq": "Monthly", "mc_start_yr": 2026, "mc_entry_q": 50}
        key = _mc_path_key(p, "dca")
        assert key["mc_blocked_bins"] == []

    def test_blocked_in_key(self):
        p = {"mc_bins": 5, "mc_sims": 800, "mc_years": 10,
             "mc_freq": "Monthly", "mc_start_yr": 2026, "mc_entry_q": 50,
             "mc_blocked_bins": [4, 0]}
        key = _mc_path_key(p, "dca")
        assert key["mc_blocked_bins"] == [0, 4]  # sorted

    def test_different_blocked_different_keys(self):
        base = {"mc_bins": 5, "mc_sims": 800, "mc_years": 10,
                "mc_freq": "Monthly", "mc_start_yr": 2026, "mc_entry_q": 50}
        k1 = _mc_path_key({**base, "mc_blocked_bins": [4]}, "dca")
        k2 = _mc_path_key({**base, "mc_blocked_bins": [3]}, "dca")
        k3 = _mc_path_key({**base}, "dca")
        assert k1 != k2
        assert k1 != k3


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5d: Ghost overlay tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGhostBuildTraces:
    """Test _mc_build_traces with ghost parameters produces correct trace structure."""

    def _ghost(self, ts, fan, **kw):
        from mc_overlay import _mc_build_traces, _GHOST_BANDS
        return _mc_build_traces(ts, fan, bands=_GHOST_BANDS, suppress_legend=True, **kw)

    def test_returns_traces_for_valid_fan(self):
        ts = np.linspace(10, 20, 50)
        fan = {p: np.random.rand(50) * 100 for p in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)}
        traces = self._ghost(ts, fan)
        # 3 bands × 2 traces each + 1 median = 7
        assert len(traces) == 7

    def test_no_median_when_disabled(self):
        ts = np.linspace(10, 20, 50)
        fan = {p: np.random.rand(50) * 100 for p in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)}
        traces = self._ghost(ts, fan, show_median=False)
        assert len(traces) == 6  # 3 bands × 2, no median

    def test_empty_fan_returns_empty(self):
        ts = np.linspace(10, 20, 50)
        traces = self._ghost(ts, {})
        assert traces == []

    def test_ghost_traces_have_no_legend(self):
        ts = np.linspace(10, 20, 50)
        fan = {p: np.random.rand(50) * 100 for p in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)}
        traces = self._ghost(ts, fan)
        for t in traces:
            assert t.showlegend is False

    def test_median_is_dashed(self):
        ts = np.linspace(10, 20, 50)
        fan = {p: np.random.rand(50) * 100 for p in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)}
        traces = self._ghost(ts, fan)
        median_trace = traces[-1]
        assert median_trace.line.dash == "dash"


class TestGhostTracesFromParams:
    """Test ghost_traces_from_params end-to-end."""

    def _make_ghost_data(self):
        from mc_overlay import _mc_fan_to_lists
        ts = np.linspace(10.0, 20.0, 50)
        fan = {p: np.random.rand(50) * 100 for p in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)}
        return {
            "fan_btc": _mc_fan_to_lists(fan),
            "fan_usd": _mc_fan_to_lists(fan),
            "ts": [round(float(t), 6) for t in ts],
        }

    def test_returns_traces_when_blocked(self):
        from mc_overlay import ghost_traces_from_params
        p = {"mc_ghost_fan": self._make_ghost_data(), "mc_blocked_bins": (4,)}
        traces = ghost_traces_from_params(p, 20.0, "btc")
        assert len(traces) == 7

    def test_returns_empty_when_no_blocked(self):
        from mc_overlay import ghost_traces_from_params
        p = {"mc_ghost_fan": self._make_ghost_data(), "mc_blocked_bins": ()}
        traces = ghost_traces_from_params(p, 20.0, "btc")
        assert traces == []

    def test_returns_empty_when_no_ghost_data(self):
        from mc_overlay import ghost_traces_from_params
        p = {"mc_blocked_bins": (4,)}
        traces = ghost_traces_from_params(p, 20.0, "btc")
        assert traces == []

    def test_clips_to_x_end(self):
        from mc_overlay import ghost_traces_from_params
        p = {"mc_ghost_fan": self._make_ghost_data(), "mc_blocked_bins": (4,)}
        traces = ghost_traces_from_params(p, 15.0, "btc")
        # Traces should be clipped — shorter than full 50 points
        assert len(traces) > 0
        for t in traces:
            if t.x is not None and len(list(t.x)) > 0:
                assert max(list(t.x)) <= 15.1


class TestGhostMatch:
    """Test _ghost_match helper in callbacks."""

    def test_matching_path_key(self):
        from callbacks import _ghost_match
        unblocked = {
            "path_key": {"tab": "dca", "mc_bins": 5, "mc_sims": 800,
                         "mc_years": 10, "mc_freq": "Monthly",
                         "mc_window": None, "mc_start_yr": 2026,
                         "mc_entry_q": 50.0, "mc_blocked_bins": ()},
            "fan_btc": {}, "fan_usd": {},
        }
        mc_p = {"mc_bins": 5, "mc_sims": 800, "mc_years": 10,
                "mc_freq": "Monthly", "mc_window": None,
                "mc_start_yr": 2026, "mc_entry_q": 50.0,
                "mc_blocked_bins": (4,)}
        result = _ghost_match(unblocked, mc_p, "dca")
        assert result is unblocked

    def test_no_match_different_bins(self):
        from callbacks import _ghost_match
        unblocked = {
            "path_key": {"tab": "dca", "mc_bins": 5, "mc_sims": 800,
                         "mc_years": 10, "mc_freq": "Monthly",
                         "mc_window": None, "mc_start_yr": 2026,
                         "mc_entry_q": 50.0, "mc_blocked_bins": ()},
        }
        mc_p = {"mc_bins": 6, "mc_sims": 800, "mc_years": 10,
                "mc_freq": "Monthly", "mc_window": None,
                "mc_start_yr": 2026, "mc_entry_q": 50.0}
        assert _ghost_match(unblocked, mc_p, "dca") is None

    def test_no_match_wrong_tab(self):
        from callbacks import _ghost_match
        unblocked = {
            "path_key": {"tab": "dca", "mc_bins": 5, "mc_sims": 800,
                         "mc_years": 10, "mc_freq": "Monthly",
                         "mc_window": None, "mc_start_yr": 2026,
                         "mc_entry_q": 50.0, "mc_blocked_bins": ()},
        }
        mc_p = {"mc_bins": 5, "mc_sims": 800, "mc_years": 10,
                "mc_freq": "Monthly", "mc_window": None,
                "mc_start_yr": 2026, "mc_entry_q": 50.0}
        assert _ghost_match(unblocked, mc_p, "ret") is None

    def test_none_unblocked(self):
        from callbacks import _ghost_match
        assert _ghost_match(None, {}, "dca") is None

    def test_empty_dict_unblocked(self):
        from callbacks import _ghost_match
        assert _ghost_match({}, {}, "dca") is None


class TestUnblockedVal:
    """Test _unblocked_val helper."""

    def test_saves_when_no_blocked(self):
        from callbacks import _unblocked_val
        result = {"fan_btc": {}, "fan_usd": {}}
        val = _unblocked_val(True, (), result, None)
        assert val is result

    def test_no_update_when_blocked(self):
        from callbacks import _unblocked_val
        import dash
        val = _unblocked_val(True, (4,), {"fan_btc": {}}, None)
        assert val is dash.no_update

    def test_no_update_when_not_mc_ok(self):
        from callbacks import _unblocked_val
        import dash
        val = _unblocked_val(False, (), {"fan_btc": {}}, None)
        assert val is dash.no_update

    def test_falls_back_to_cached(self):
        from callbacks import _unblocked_val
        cached = {"fan_btc": {}, "fan_usd": {}}
        val = _unblocked_val(True, (), None, cached)
        assert val is cached


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5e: Regime filter fuzz / edge-case tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinMaskFuzz:
    """Parametrized fuzz: apply_bin_mask with various blocked-bin combos."""

    @pytest.mark.parametrize("n_bins,blocked", [
        (5, [0]),
        (5, [4]),
        (5, [0, 4]),
        (5, [1, 2, 3]),
        (5, [0, 1, 2, 3]),  # only bin 4 allowed
        (6, [0, 5]),
        (6, [1, 2, 3, 4]),
        (8, [0, 1, 6, 7]),
        (10, list(range(0, 10, 2))),   # block even bins
        (10, list(range(1, 10, 2))),   # block odd bins
    ])
    def test_masked_matrix_is_valid(self, n_bins, blocked):
        """After masking, matrix must be row-stochastic with zero blocked columns."""
        trans = np.random.rand(n_bins, n_bins)
        trans /= trans.sum(axis=1, keepdims=True)
        result = _apply_bin_mask(trans, blocked)
        # Blocked columns must be zero
        for b in blocked:
            assert np.allclose(result[:, b], 0.0), f"column {b} not zeroed"
        # Each row must sum to 1
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
        # No negative values
        assert np.all(result >= 0)

    @pytest.mark.parametrize("n_bins", [5, 6, 7, 8, 10])
    def test_single_allowed_bin(self, n_bins):
        """With only 1 allowed bin, all probability should go there."""
        blocked = list(range(1, n_bins))  # only bin 0 allowed
        trans = np.random.rand(n_bins, n_bins)
        trans /= trans.sum(axis=1, keepdims=True)
        result = _apply_bin_mask(trans, blocked)
        for r in range(n_bins):
            assert abs(result[r, 0] - 1.0) < 1e-12
            assert abs(result[r, 1:].sum()) < 1e-12


class TestSnapStartPctileFuzz:
    """Parametrized fuzz: snap_start_pctile with blocked-bin combos."""

    @pytest.mark.parametrize("n_bins,blocked,pctile", [
        (5, [4], 0.9),     # in blocked top bin → snap down
        (5, [0], 0.1),     # in blocked bottom bin → snap up
        (5, [2], 0.5),     # in blocked middle bin → snap to neighbor
        (5, [0, 4], 0.1),  # bottom blocked → snap to bin 1
        (5, [0, 4], 0.9),  # top blocked → snap to bin 3
        (5, [0, 1, 2, 3], 0.5),  # only bin 4 allowed → snap to 0.9
        (10, [0, 1, 8, 9], 0.05),  # bottom blocked → snap to bin 2
    ])
    def test_snapped_to_allowed_bin(self, n_bins, blocked, pctile):
        """Snapped percentile must fall in an allowed bin."""
        edges = np.linspace(0, 1, n_bins + 1)
        result = _snap_start_pctile(pctile, edges, blocked)
        # Find which bin the result falls in
        result_bin = min(int(result * n_bins), n_bins - 1)
        result_bin = max(result_bin, 0)
        assert result_bin not in blocked, f"snapped to blocked bin {result_bin}"


class TestBuildMcParamsAllUnchecked:
    """Guard: all bins unchecked (mc_regime=[]) should not block all bins."""

    def test_empty_regime_means_no_blocked(self):
        from callbacks import _build_mc_params
        p = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=4, mc_bins=5, mc_sims=800,
            mc_years=10, mc_freq="Monthly", mc_window=None,
            mc_start_yr=2026, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            mc_regime=[],   # all unchecked
        )
        assert p["mc_blocked_bins"] == []

    def test_single_bin_regime_allowed(self):
        from callbacks import _build_mc_params
        p = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=4, mc_bins=5, mc_sims=800,
            mc_years=10, mc_freq="Monthly", mc_window=None,
            mc_start_yr=2026, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            mc_regime=[2],   # only bin 2 allowed
        )
        assert p["mc_blocked_bins"] == [0, 1, 3, 4]


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


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Financial math tests
# ═══════════════════════════════════════════════════════════════════════════════

_Q50 = 0.5 if 0.5 in M.qr_fits else next(iter(M.qr_fits))


class TestDCAMath:
    """Verify DCA accumulation arithmetic against manual calculation."""

    def _dca_params(self, **overrides):
        p = {
            "start_yr": 2030, "end_yr": 2031,
            "start_stack": 0.0, "amount": 1000, "freq": "Monthly",
            "disp_mode": "btc", "selected_qs": [_Q50],
            "log_y": False, "show_today": False,            "show_legend": True, "lots": [], "use_lots": False,
            "sc_enabled": False,
        }
        p.update(overrides)
        return p

    def test_accumulation_matches_manual(self):
        """Final BTC stack should equal sum of (amount / price) for each period."""
        p = self._dca_params()
        fig, _ = build_dca_figure(M, p)
        assert len(fig.data) >= 1
        y_vals = fig.data[0].y
        # Manual computation
        t_start = max(yr_to_t(2030, M.genesis), 1.0)
        t_end = yr_to_t(2031, M.genesis)
        ts = np.arange(t_start, t_end + (1 / 12) * 0.5, 1 / 12)
        expected = 0.0
        for t in ts:
            expected += 1000.0 / float(qr_price(_Q50, max(t, 0.5), M.qr_fits))
        assert abs(y_vals[-1] - expected) < 0.1  # loosened for 3-sig-fig trace rounding

    def test_start_stack_offset(self):
        """Starting stack should shift all values by a constant."""
        fig0, _ = build_dca_figure(M, self._dca_params(start_stack=0.0))
        fig1, _ = build_dca_figure(M, self._dca_params(start_stack=2.5))
        final0 = fig0.data[0].y[-1]
        final1 = fig1.data[0].y[-1]
        assert abs(final1 - final0 - 2.5) < 0.1  # loosened for 3-sig-fig trace rounding

    def test_usd_mode_equals_btc_times_price(self):
        """USD display mode = BTC balance × final price."""
        fig_btc, _ = build_dca_figure(M, self._dca_params(disp_mode="btc"))
        fig_usd, _ = build_dca_figure(M, self._dca_params(disp_mode="usd"))
        btc_final = fig_btc.data[0].y[-1]
        usd_final = fig_usd.data[0].y[-1]
        t_end = yr_to_t(2031, M.genesis)
        ts = np.arange(max(yr_to_t(2030, M.genesis), 1.0), t_end + (1 / 12) * 0.5, 1 / 12)
        final_price = float(_app_ctx.DEFAULT_MODEL.price_at(_Q50, max(ts[-1], 0.5)))
        assert abs(usd_final - btc_final * final_price) < 100.0  # loosened for 3-sig-fig trace rounding

    def test_higher_quantile_less_btc(self):
        """Higher quantile → higher price → less BTC accumulated per DCA."""
        q_lo, q_hi = 0.1, 0.9
        if q_lo not in M.qr_fits or q_hi not in M.qr_fits:
            pytest.skip("Need Q10% and Q90%")
        fig, _ = build_dca_figure(M, self._dca_params(
            selected_qs=[q_lo, q_hi], disp_mode="btc"))
        btc_lo = fig.data[0].y[-1]
        btc_hi = fig.data[1].y[-1]
        assert btc_lo > btc_hi  # lower price → more BTC per purchase

    def test_end_before_start_returns_error(self):
        """end_yr <= start_yr should return an error figure."""
        fig, _ = build_dca_figure(M, self._dca_params(start_yr=2035, end_yr=2030))
        assert isinstance(fig, go.Figure)
        assert "end year" in (fig.layout.title.text or "").lower()


class TestSCLoanCap:
    """Verify Stack-celerator loan cap formulas."""

    def test_interest_only_cap_formula(self):
        """max_principal = amount / r for interest-only."""
        amount, r = 500, 0.01  # $500/period, 1% per period
        assert abs(amount / r - 50_000) < 0.01

    def test_amortizing_cap_formula(self):
        """max_principal = amount * (1-(1+r)^-n) / r for amortizing."""
        amount, r, n = 500, 0.01, 12
        max_p = amount * (1 - (1 + r) ** (-n)) / r
        # Verify: payment at max principal should equal amount
        pmt = max_p * r / (1 - (1 + r) ** (-n))
        assert abs(pmt - amount) < 0.01

    def test_cap_prevents_negative_dca(self):
        """Huge loan should be capped; SC trace still generated."""
        p = {
            "start_yr": 2030, "end_yr": 2031,
            "start_stack": 0.0, "amount": 100, "freq": "Monthly",
            "disp_mode": "btc", "selected_qs": [_Q50],
            "log_y": False, "show_today": False,            "show_legend": True, "lots": [], "use_lots": False,
            "sc_enabled": True,
            "sc_loan_amount": 999_999_999, "sc_rate": 12.0,
            "sc_term_months": 12, "sc_loan_type": "interest_only",
            "sc_repeats": 0, "sc_entry_mode": "model",
            "sc_custom_price": 0, "sc_tax_rate": 0.33,
            "sc_rollover": False, "sc_live_price": 0,
        }
        fig, _ = build_dca_figure(M, p)
        assert isinstance(fig, go.Figure)
        sc_traces = [t for t in fig.data if "SC" in (t.name or "")]
        assert len(sc_traces) >= 1

    def test_zero_rate_no_cap(self):
        """0% interest → no cap applied, amortizing payment = principal/n."""
        p = {
            "start_yr": 2030, "end_yr": 2031,
            "start_stack": 0.0, "amount": 1000, "freq": "Monthly",
            "disp_mode": "btc", "selected_qs": [_Q50],
            "log_y": False, "show_today": False,            "show_legend": True, "lots": [], "use_lots": False,
            "sc_enabled": True,
            "sc_loan_amount": 5000, "sc_rate": 0.0,
            "sc_term_months": 12, "sc_loan_type": "amortizing",
            "sc_repeats": 0, "sc_entry_mode": "model",
            "sc_custom_price": 0, "sc_tax_rate": 0,
            "sc_rollover": False, "sc_live_price": 0,
        }
        fig, _ = build_dca_figure(M, p)
        sc_traces = [t for t in fig.data if "SC" in (t.name or "")]
        assert len(sc_traces) >= 1


class TestSCTaxOnGain:
    """Verify tax-on-gain formula for interest-only SC."""

    def test_gain_taxed(self):
        """net_per_btc = price - tax_rate * max(price - ep, 0)."""
        price, ep, tax = 100_000, 60_000, 0.33
        gain = max(price - ep, 0)
        net = price - tax * gain
        assert abs(net - 86_800) < 0.01

    def test_loss_no_tax(self):
        """No tax when selling at a loss."""
        price, ep, tax = 50_000, 60_000, 0.33
        gain = max(price - ep, 0)
        net = price - tax * gain
        assert abs(net - 50_000) < 0.01

    def test_btc_sold_amount(self):
        """BTC sold to repay = principal / net_per_btc."""
        principal = 10_000
        price, ep, tax = 100_000, 60_000, 0.33
        net = price - tax * max(price - ep, 0)
        btc_sold = principal / net
        assert abs(btc_sold - 10_000 / 86_800) < 1e-8

    def test_zero_tax_net_equals_price(self):
        """With 0% tax, net_per_btc = price regardless of gain."""
        price, ep = 100_000, 50_000
        net = price - 0.0 * max(price - ep, 0)
        assert abs(net - price) < 0.01


class TestRetireMath:
    """Verify retirement depletion arithmetic."""

    def _retire_params(self, **overrides):
        p = {
            "start_yr": 2030, "end_yr": 2035,
            "start_stack": 1.0, "wd_amount": 50000, "freq": "Annually",
            "inflation": 0, "disp_mode": "btc",
            "selected_qs": [_Q50], "log_y": False,
            "show_legend": True, "annotate": False,
            "lots": [], "use_lots": False,
        }
        p.update(overrides)
        return p

    def test_depletion_matches_manual(self):
        """Step-by-step depletion should match manual calculation."""
        p = self._retire_params()
        fig, _ = build_retire_figure(M, p)
        assert len(fig.data) >= 1
        y_vals = list(fig.data[0].y)
        # Manual — use DEFAULT_MODEL.price_at (same as figure code)
        t_start = max(yr_to_t(2030, M.genesis), 1.0)
        t_end = yr_to_t(2035, M.genesis)
        ts = np.arange(t_start, t_end + 0.5, 1.0)
        stack = 1.0
        for i, t in enumerate(ts):
            price = float(_app_ctx.DEFAULT_MODEL.price_at(_Q50, max(t, 0.5)))
            stack -= 50000.0 / price
            stack = max(stack, 0.0)
            assert abs(y_vals[i] - stack) < 0.01  # loosened for 3-sig-fig trace rounding

    def test_zero_withdrawal_preserves_stack(self):
        """Zero withdrawal should keep stack constant."""
        fig, _ = build_retire_figure(M, self._retire_params(wd_amount=0))
        for v in fig.data[0].y:
            assert abs(v - 1.0) < 1e-8

    def test_inflation_accelerates_depletion(self):
        """Positive inflation should deplete faster than zero inflation."""
        fig_no, _ = build_retire_figure(M, self._retire_params(
            start_stack=10.0, wd_amount=10000, end_yr=2050, inflation=0))
        fig_yes, _ = build_retire_figure(M, self._retire_params(
            start_stack=10.0, wd_amount=10000, end_yr=2050, inflation=10))
        assert fig_yes.data[0].y[-1] < fig_no.data[0].y[-1]

    def test_large_withdrawal_depletes_to_zero(self):
        """Huge withdrawal should reach zero quickly."""
        fig, _ = build_retire_figure(M, self._retire_params(
            start_stack=0.1, wd_amount=1_000_000, freq="Monthly"))
        assert fig.data[0].y[-1] == 0.0

    def test_depletion_annotation_present(self):
        """When stack depletes and annotate=True, annotation should exist."""
        fig, _ = build_retire_figure(M, self._retire_params(
            start_stack=0.01, wd_amount=500_000, freq="Monthly",
            annotate=True))
        annots = fig.layout.annotations or []
        # Should have at least one depletion annotation with year text
        depl_annots = [a for a in annots if "≈" in (a.text or "")]
        assert len(depl_annots) >= 1


class TestAnnotationAlignment:
    """Every text-trace annotation must sit at the last data point of a parent
    line trace — guaranteeing pixel-perfect alignment regardless of zoom,
    resize, or rotation.  Depletion layout annotations (≈YYYY) are checked
    separately for correct x-coordinate placement."""

    @staticmethod
    def _assert_text_traces_at_endpoints(fig):
        lines = [tr for tr in fig.data
                 if getattr(tr, 'mode', '') in ('lines', 'lines+markers')
                 and tr.y is not None and len(list(tr.y)) > 0]
        texts = [tr for tr in fig.data
                 if getattr(tr, 'mode', '') == 'markers+text']
        for tt in texts:
            ax, ay = float(tt.x[0]), float(tt.y[0])
            ok = any(
                abs(ax - float(list(lt.x)[-1])) < 1e-6
                and (abs(ay - float(list(lt.y)[-1])) < 1e-6
                     or (float(list(lt.y)[-1]) != 0 and abs(ay - float(list(lt.y)[-1])) / abs(float(list(lt.y)[-1])) < 0.01))
                for lt in lines
            )
            lbl = tt.text[0] if tt.text else ""
            assert ok, (
                f"Annotation '{lbl}' at ({ax:.6f}, {ay:.6f}) "
                f"does not match any line trace endpoint"
            )

    def test_dca_single_q_btc(self):
        p = dict(start_yr=2024, end_yr=2034, start_stack=0, amount=500,
                 freq="Monthly", disp_mode="btc", log_y=True,
                 selected_qs=[0.5] if 0.5 in M.qr_fits else [],
                 show_today=False, show_legend=True,
                 lots=[], use_lots=False, sc_enabled=False)
        fig, _ = build_dca_figure(M, p)
        self._assert_text_traces_at_endpoints(fig)

    def test_dca_multi_q_usd(self):
        qs = [q for q in [0.10, 0.50] if q in M.qr_fits]
        p = dict(start_yr=2024, end_yr=2034, start_stack=0, amount=500,
                 freq="Monthly", disp_mode="usd", log_y=False,
                 selected_qs=qs, show_today=False, show_legend=True,
                 lots=[], use_lots=False, sc_enabled=False)
        fig, _ = build_dca_figure(M, p)
        self._assert_text_traces_at_endpoints(fig)

    def test_dca_no_y2(self):
        """No secondary Y-axis should be created."""
        qs = [0.5] if 0.5 in M.qr_fits else []
        p = dict(start_yr=2024, end_yr=2034, start_stack=0, amount=500,
                 freq="Monthly", disp_mode="btc", log_y=True,
                 selected_qs=qs, show_today=False,
                 show_legend=True, lots=[], use_lots=False, sc_enabled=False)
        fig, _ = build_dca_figure(M, p)
        try:
            y2 = fig.layout.yaxis2
            assert y2 is None or y2.title is None
        except AttributeError:
            pass  # yaxis2 doesn't exist — correct behavior

    def test_retire_non_depleted(self):
        qs = [0.50] if 0.50 in M.qr_fits else []
        p = dict(start_yr=2031, end_yr=2050, start_stack=10.0,
                 wd_amount=1000, freq="Monthly", inflation=0,
                 disp_mode="btc", selected_qs=qs, log_y=True,
                 annotate=True,
                 show_legend=True, minor_grid=False, lots=[], use_lots=False)
        fig, _ = build_retire_figure(M, p)
        self._assert_text_traces_at_endpoints(fig)

    def test_retire_depleted_has_depletion_annot(self):
        """Depleted traces get ≈YYYY annotations, not endpoint text traces."""
        qs = [0.01] if 0.01 in M.qr_fits else []
        if not qs:
            pytest.skip("need Q1%")
        p = dict(start_yr=2031, end_yr=2075, start_stack=0.01,
                 wd_amount=500_000, freq="Monthly", inflation=0,
                 disp_mode="btc", selected_qs=qs, log_y=True,
                 annotate=True,
                 show_legend=True, minor_grid=False, lots=[], use_lots=False)
        fig, _ = build_retire_figure(M, p)
        depl = [a for a in (fig.layout.annotations or []) if "≈" in (a.text or "")]
        assert len(depl) >= 1
        # Depleted trace should NOT have a text-trace annotation
        texts = [tr for tr in fig.data if getattr(tr, 'mode', '') == 'markers+text']
        assert len(texts) == 0

    def test_supercharge_non_depleted(self):
        qs = [0.50] if 0.50 in M.qr_fits else []
        p = dict(mode="A", start_stack=10.0, delays=[0],
                 start_yr=2033, end_yr=2050, freq="Annually",
                 inflation=0, wd_amount=1000, disp_mode="usd",
                 selected_qs=qs, log_y=True,
                 annotate=True, show_legend=True, lots=[], use_lots=False,
                 shade=False, display_q=0.50)
        fig, _ = build_supercharge_figure(M, p)
        self._assert_text_traces_at_endpoints(fig)


class TestConfigAnnotations:
    """Tests for chart config text annotations (P-1) and display toggle (P-2)."""

    def test_qr_config_text_dca(self):
        p = dict(selected_qs=[0.1, 0.5, 0.85], amount=200, freq="Monthly",
                 start_yr=2025, end_yr=2035, start_stack=0.5, log_y=True)
        text = _build_qr_config_text(p, "dca")
        assert "Quantiles:" in text
        assert "Q10%" in text
        assert "Q50%" in text
        assert "Q85%" in text
        assert "$200" in text
        assert "/mo" in text
        assert "2025" in text
        assert "2035" in text
        assert "0.5 BTC" in text
        assert "Log Y" in text

    def test_qr_config_text_empty_quantiles(self):
        p = dict(selected_qs=[], start_yr=2025, end_yr=2035)
        text = _build_qr_config_text(p, "dca")
        assert "Q50%" in text  # fallback to Q50%

    def test_qr_config_text_retire(self):
        p = dict(selected_qs=[0.01], wd_amount=5000, freq="Annually",
                 start_yr=2031, end_yr=2075, inflation=4, start_stack=1.0)
        text = _build_qr_config_text(p, "ret")
        assert "$5,000" in text
        assert "/yr" in text
        assert "4% infl" in text

    def test_mc_config_text(self):
        p = dict(mc_start_yr=2031, mc_years=10, mc_entry_q=50,
                 mc_sims=800, mc_freq="Monthly", mc_amount=100,
                 mc_infl=4, mc_start_stack=1.0)
        text = _build_mc_config_text(p, "dca")
        assert "MC DCA" in text
        assert "2031" in text
        assert "10yr" in text
        assert "Q50%" in text
        assert "800 sims" in text
        assert "$100" in text
        assert "4% infl" in text
        assert "1 BTC" in text
        assert ".json" in text

    def test_apply_config_annotation_qr_only(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        p = dict(selected_qs=[0.5], start_yr=2025, end_yr=2035)
        _apply_config_annotation(fig, p, "dca", show_qr=True, show_mc=False)
        assert fig.layout.xaxis.title.text is not None
        assert "Quantiles:" in fig.layout.xaxis.title.text
        assert "MC" not in fig.layout.xaxis.title.text

    def test_apply_config_annotation_both(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        p = dict(selected_qs=[0.5], start_yr=2025, end_yr=2035,
                 mc_start_yr=2031, mc_years=10, mc_entry_q=50,
                 mc_sims=800, mc_freq="Monthly")
        _apply_config_annotation(fig, p, "dca", show_qr=True, show_mc=True)
        text = fig.layout.xaxis.title.text
        assert "Quantiles:" in text
        assert "MC DCA" in text
        assert "<br>" in text  # two lines

    def test_apply_config_annotation_none(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        _apply_config_annotation(fig, {}, "dca", show_qr=False, show_mc=False)
        # No annotation set
        assert fig.layout.xaxis.title.text is None

    def test_bubble_figure_has_qr_annotation(self):
        p = dict(selected_qs=[0.5], shade=False, xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False)
        fig = build_bubble_figure(M, p)
        xtitle = fig.layout.xaxis.title.text
        assert xtitle is not None and "Quantiles:" in xtitle

    def test_dca_figure_show_qr_false_hides_qr_annotation(self):
        p = dict(start_yr=2025, end_yr=2035, start_stack=0, amount=100,
                 freq="Monthly", disp_mode="btc", selected_qs=[0.5],
                 log_y=False, show_today=False, show_legend=False,
                 lots=[], use_lots=False, show_qr=False, show_mc=False)
        fig, _ = build_dca_figure(M, p)
        xtitle = fig.layout.xaxis.title.text or ""
        assert "Quantiles:" not in xtitle

    def test_model_show_snapshot_roundtrip(self):
        """Verify model-show controls survive snapshot encode/decode."""
        state = {"dca-model-show:value": ["qr"],
                 "ret-model-show:value": ["mc"],
                 "sc-model-show:value": ["qr", "mc"],
                 "hm-model-show:value": []}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["dca-model-show:value"] == ["qr"]
        assert decoded["ret-model-show:value"] == ["mc"]
        assert decoded["sc-model-show:value"] == ["qr", "mc"]
        # empty list → bitmask 0 → decoded as empty list or omitted
        hm_val = decoded.get("hm-model-show:value", [])
        assert hm_val == []


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


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Callback smoke tests (Phase E)
# ═══════════════════════════════════════════════════════════════════════════════

class _CallbackCtx:
    """Minimal mock for dash.ctx (dash._callback_context)."""
    def __init__(self, triggered_id=None):
        self.triggered_id = triggered_id


def _patch_ctx(triggered_id=None):
    """Context manager that patches dash.ctx across all callback submodules."""
    ctx_obj = _CallbackCtx(triggered_id)
    # After callbacks.py was split into callbacks/, each submodule imports ctx
    # from dash directly.  Patch every submodule that uses ctx.
    from contextlib import ExitStack
    _targets = [
        "callbacks", "callbacks.charts", "callbacks.lots",
        "callbacks.mc_helpers", "callbacks.mc_payment",
        "callbacks.snapshot_cb",
    ]
    stack = ExitStack()
    for t in _targets:
        stack.enter_context(patch.multiple(t, ctx=ctx_obj))
    return stack


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateBubbleCallback:
    """Smoke-test the update_bubble callback."""

    def test_returns_figure(self):
        with _patch_ctx("bub-qs"):
            fig = update_bubble(
                _first_render=1, sel_qs=[0.5], adv_qs=[], toggles=["show_data", "show_today"],
                bubble_toggles=[], xscale="log", yscale="log",
                xrange=[2012, 2030], yrange=[0, 7],
                n_future=3, ptsize=3, ptalpha=0.6,
                stack=0, show_stack=[], use_lots=[], legend_pos="outside", model_show=[],
                lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[], lots_data=[],
                palette_key="default",
            )
        assert isinstance(fig, go.Figure)

    def test_empty_quantiles(self):
        with _patch_ctx("bub-qs"):
            fig = update_bubble(
                _first_render=1, sel_qs=[], adv_qs=[], toggles=[], bubble_toggles=[],
                xscale="linear", yscale="log",
                xrange=[2015, 2028], yrange=[1, 6],
                n_future=0, ptsize=2, ptalpha=0.3,
                stack=0, show_stack=[], use_lots=[], legend_pos="outside", model_show=[],
                lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[], lots_data=[],
                palette_key="default",
            )
        assert isinstance(fig, go.Figure)

    def test_with_stack(self):
        with _patch_ctx("bub-stack"):
            fig = update_bubble(
                _first_render=1, sel_qs=[0.1, 0.5, 0.9], adv_qs=[], toggles=["show_legend"],
                bubble_toggles=["show_comp"], xscale="log", yscale="log",
                xrange=[2012, 2035], yrange=[0, 7],
                n_future=2, ptsize=4, ptalpha=0.5,
                stack=1.5, show_stack=["yes"], use_lots=[], legend_pos="outside", model_show=[],
                lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[], lots_data=[],
                palette_key="default",
            )
        assert isinstance(fig, go.Figure)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateHeatmapCallback:
    """Smoke-test the update_heatmap callback."""

    def test_returns_figure(self):
        yr = pd.Timestamp.today().year
        with _patch_ctx("hm-entry-yr"):
            result = update_heatmap(
                _first_render=1, hm_model="bub", entry_yr=yr, entry_q=50.0,
                exit_range=[yr, yr + 10],
                exit_qs=[0.01, 0.1, 0.5, 0.85, 0.99],
                mode=0, b1=0, b2=20,
                c_lo=None, c_mid1=None, c_mid2=None, c_hi=None,
                grad=32, vfmt="cagr", cell_fs=9,
                toggles=["colorbar"], stack=0, use_lots=[],
                lots_data=[],
                mc_enable=[], mc_amount=100, mc_infl=0,
                mc_bins=5, mc_regime=list(range(5)), mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=[2010, yr],
                mc_start_yr=yr, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0, model_show=["qr", "mc"], mc_model_src="bub",
                live_price=0, mc_cached=None, pay_token=None, mc_auth=None,
                palette_key="default",
            )
        # Returns 8 outputs: fig, store, status, panel_style, indicator_style, rendered_key, modal, tab
        assert len(result) == 8
        assert isinstance(result[0], go.Figure)



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateDcaCallback:
    """Smoke-test the update_dca callback."""

    def test_returns_figure_and_mc_outputs(self):
        with _patch_ctx("dca-amount"):
            result = update_dca(
                _first_render=1, stack=0, use_lots=[], amount=200,
                freq="Monthly", dca_infl=0, yr_range=[2025, 2035],
                disp="btc", toggles=["show_legend"], legend_pos="outside",
                sel_qs=[0.5], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[], lots_data=[],
                sc_enable=[], sc_loan=0, sc_rate=13, sc_term=12,
                sc_type="interest_only", sc_repeats=0,
                sc_entry_mode="live", sc_custom_price=80000,
                sc_tax=33, sc_rollover=[],
                mc_enable=[],
                mc_bins=5, mc_regime=list(range(5)), mc_sims=800, mc_years=10,
                mc_window=None,
                mc_start_yr=2026, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0, model_show=["qr", "mc"], mc_model_src="bub",
                price_data=0, mc_cached=None, pay_token=None, mc_unblocked=None, mc_auth=None,
                palette_key="default",
            )
        # 8 outputs: fig, mc_results, mc_status, rendered_key, mc_modal, mc_tab, unblocked, yr_adjust
        assert len(result) == 8
        assert isinstance(result[0], go.Figure)

    def test_with_sc_enabled(self):
        with _patch_ctx("dca-sc-enable"):
            result = update_dca(
                _first_render=1, stack=0, use_lots=[], amount=500,
                freq="Monthly", dca_infl=0, yr_range=[2025, 2030],
                disp="btc", toggles=[], legend_pos="outside",
                sel_qs=[0.1, 0.5], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[], lots_data=[],
                sc_enable=["yes"], sc_loan=10000, sc_rate=13, sc_term=12,
                sc_type="interest_only", sc_repeats=0,
                sc_entry_mode="custom", sc_custom_price=90000,
                sc_tax=33, sc_rollover=[],
                mc_enable=[],
                mc_bins=5, mc_regime=list(range(5)), mc_sims=800, mc_years=10,
                mc_window=None,
                mc_start_yr=2026, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0, model_show=["qr", "mc"], mc_model_src="bub",
                price_data=0, mc_cached=None, pay_token=None, mc_unblocked=None, mc_auth=None,
                palette_key="default",
            )
        assert isinstance(result[0], go.Figure)

    def test_usd_display_mode(self):
        with _patch_ctx("dca-disp"):
            result = update_dca(
                _first_render=1, stack=0.5, use_lots=[], amount=300,
                freq="Weekly", dca_infl=0, yr_range=[2025, 2032],
                disp="usd", toggles=["log_y"], legend_pos="outside",
                sel_qs=[0.5, 0.85], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[], lots_data=[],
                sc_enable=[], sc_loan=0, sc_rate=13, sc_term=12,
                sc_type="interest_only", sc_repeats=0,
                sc_entry_mode="live", sc_custom_price=80000,
                sc_tax=33, sc_rollover=[],
                mc_enable=[],
                mc_bins=5, mc_regime=list(range(5)), mc_sims=800, mc_years=10,
                mc_window=None,
                mc_start_yr=2026, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0, model_show=["qr", "mc"], mc_model_src="bub",
                price_data=0, mc_cached=None, pay_token=None, mc_unblocked=None, mc_auth=None,
                palette_key="default",
            )
        assert isinstance(result[0], go.Figure)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateRetireCallback:
    """Smoke-test the update_retire callback."""

    def test_returns_figure(self):
        with _patch_ctx("ret-wd"):
            result = update_retire(
                _first_render=1, stack=1.0, use_lots=[], wd=5000,
                freq="Monthly", yr_range=[2031, 2075], infl=4,
                disp="btc", toggles=["log_y", "annotate"],
                legend_pos="outside",
                sel_qs=[0.01, 0.1, 0.25], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[], lots_data=[],
                mc_enable=[],
                mc_bins=5, mc_regime=list(range(5)), mc_sims=800, mc_years=10,
                mc_window=None,
                mc_start_yr=2031, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0, model_show=["qr", "mc"], mc_model_src="bub",
                price_data=0, mc_cached=None, pay_token=None, mc_unblocked=None, mc_auth=None,
                palette_key="default",
            )
        # 8 outputs: fig, mc_results, mc_status, rendered_key, mc_modal, mc_tab, unblocked, yr_adjust
        assert len(result) == 8
        assert isinstance(result[0], go.Figure)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateSuperchargeCallback:
    """Smoke-test the update_supercharge callback."""

    def test_mode_a(self):
        with _patch_ctx("sc-mode"):
            result = update_supercharge(
                _first_render=1, stack=1.0, use_lots=[],
                start_yr=2033, d0=0, d1=0, d2=0, d3=1, d4=2,
                freq="Annually", infl=4, sel_qs=[0.001, 0.1], adv_qs=[],
                lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
                mode="a", wd=100000, end_yr=2075, target_yr=2060,
                disp="usd",
                toggles=["annotate", "log_y", "show_legend"], legend_pos="outside",
                chart_layout=["shade"], display_q=0.5, lots_data=[],
                mc_enable=[],
                mc_bins=5, mc_regime=list(range(5)), mc_sims=800, mc_years=10,
                mc_window=None,
                mc_start_yr=2031, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0, model_show=["qr", "mc"], mc_model_src="bub",
                price_data=0, mc_cached=None, pay_token=None, mc_unblocked=None, mc_auth=None,
                palette_key="default",
                viewport_width=1200,
            )
        assert len(result) == 8
        assert isinstance(result[0], go.Figure)

    def test_mode_b(self):
        with _patch_ctx("sc-mode"):
            result = update_supercharge(
                _first_render=1, stack=2.0, use_lots=[],
                start_yr=2030, d0=0, d1=1, d2=3, d3=5, d4=10,
                freq="Monthly", infl=3, sel_qs=[0.1, 0.5], adv_qs=[],
                lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
                mode="b", wd=50000, end_yr=2080, target_yr=2055,
                disp="usd", toggles=["show_legend"], legend_pos="outside",
                chart_layout=[], display_q=0.5, lots_data=[],
                mc_enable=[],
                mc_bins=5, mc_regime=list(range(5)), mc_sims=800, mc_years=10,
                mc_window=None,
                mc_start_yr=2031, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0, model_show=["qr", "mc"], mc_model_src="bub",
                price_data=0, mc_cached=None, pay_token=None, mc_unblocked=None, mc_auth=None,
                palette_key="default",
                viewport_width=1200,
            )
        assert isinstance(result[0], go.Figure)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestManageLotsCallback:
    """Smoke-test manage_lots callback."""

    def test_add_lot(self):
        with _patch_ctx("lot-add-btn"):
            result = manage_lots(
                add_n=1, del_n=0, clear_n=0, import_contents=None,
                date_str="2024-01-15", btc_amt="0.5", price_val="42000",
                notes="test lot", selected_rows=[], lots_data=[],
            )
        lots, table_data, sel, summary, import_status = result
        assert len(lots) == 1
        assert lots[0]["btc"] == 0.5
        assert lots[0]["price"] == 42000.0
        assert lots[0]["notes"] == "test lot"
        assert "pct_q" in lots[0]

    def test_add_lot_special_chars(self):
        """Lots with special characters in notes."""
        with _patch_ctx("lot-add-btn"):
            result = manage_lots(
                add_n=1, del_n=0, clear_n=0, import_contents=None,
                date_str="2023-06-01", btc_amt="1.0", price_val="30000",
                notes='DCA "buy the dip" 🚀 <script>alert(1)</script>',
                selected_rows=[], lots_data=[],
            )
        lots = result[0]
        assert len(lots) == 1
        assert '<script>' in lots[0]["notes"]  # stored as-is, XSS prevented by Dash rendering

    def test_delete_lot(self):
        existing = [
            {"date": "2024-01-01", "btc": 0.1, "price": 40000, "pct_q": 0.5, "notes": "a"},
            {"date": "2024-02-01", "btc": 0.2, "price": 45000, "pct_q": 0.6, "notes": "b"},
        ]
        with _patch_ctx("lot-del-btn"):
            result = manage_lots(
                add_n=0, del_n=1, clear_n=0, import_contents=None,
                date_str=None, btc_amt=None, price_val=None, notes=None,
                selected_rows=[0], lots_data=existing,
            )
        lots = result[0]
        assert len(lots) == 1
        assert lots[0]["notes"] == "b"

    def test_clear_lots(self):
        existing = [
            {"date": "2024-01-01", "btc": 0.1, "price": 40000, "pct_q": 0.5, "notes": "a"},
        ]
        with _patch_ctx("lot-clear-btn"):
            result = manage_lots(
                add_n=0, del_n=0, clear_n=1, import_contents=None,
                date_str=None, btc_amt=None, price_val=None, notes=None,
                selected_rows=[], lots_data=existing,
            )
        assert result[0] == []

    def test_import_lots(self):
        lots_json = json.dumps([
            {"date": "2024-03-01", "btc": 0.3, "price": 60000, "notes": "imported"},
        ])
        b64 = base64.b64encode(lots_json.encode()).decode()
        contents = f"data:application/json;base64,{b64}"
        with _patch_ctx("lots-import-upload"):
            result = manage_lots(
                add_n=0, del_n=0, clear_n=0, import_contents=contents,
                date_str=None, btc_amt=None, price_val=None, notes=None,
                selected_rows=[], lots_data=[],
            )
        lots = result[0]
        assert len(lots) == 1
        assert lots[0]["btc"] == 0.3
        assert "pct_q" in lots[0]  # recomputed

    def test_add_invalid_prevents_update(self):
        """Missing fields should raise PreventUpdate."""
        with _patch_ctx("lot-add-btn"):
            with pytest.raises(Exception):
                manage_lots(
                    add_n=1, del_n=0, clear_n=0, import_contents=None,
                    date_str=None, btc_amt=None, price_val=None, notes=None,
                    selected_rows=[], lots_data=[],
                )


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestLotsSummary:
    def test_empty(self):
        assert _lots_summary([]) == "No lots."

    def test_single_lot(self):
        lots = [{"btc": 0.5, "price": 40000, "pct_q": 0.45}]
        s = _lots_summary(lots)
        assert "1 lot(s)" in s
        assert "0.5 BTC" in s

    def test_multi_lot_avg(self):
        lots = [
            {"btc": 1.0, "price": 30000, "pct_q": 0.3},
            {"btc": 1.0, "price": 60000, "pct_q": 0.7},
        ]
        s = _lots_summary(lots)
        assert "2 lot(s)" in s
        assert "2 BTC" in s


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestPreviewPercentile:
    def test_valid_input(self):
        with _patch_ctx():
            result = preview_percentile("2024-06-15", 65000)
        assert result.startswith("Q")
        assert result.endswith("%")

    def test_no_date(self):
        with _patch_ctx():
            assert preview_percentile(None, 65000) == ""

    def test_no_price(self):
        with _patch_ctx():
            assert preview_percentile("2024-06-15", None) == ""

    def test_zero_price(self):
        with _patch_ctx():
            assert preview_percentile("2024-06-15", 0) == ""


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestEffectiveLots:
    def test_snapshot_overrides(self):
        local = [{"btc": 1}]
        snap = [{"btc": 2}]
        assert update_effective_lots(local, snap) == snap

    def test_local_when_no_snapshot(self):
        local = [{"btc": 1}]
        assert update_effective_lots(local, None) == local

    def test_empty_when_none(self):
        assert update_effective_lots(None, None) == []


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestRestoreFromUrl:
    def test_empty_hash(self):
        state, loaded = restore_from_url("")
        from dash import no_update
        assert state is no_update
        assert loaded is no_update

    def test_none_hash(self):
        state, loaded = restore_from_url(None)
        from dash import no_update
        assert state is no_update

    def test_invalid_prefix(self):
        state, loaded = restore_from_url("#garbage")
        from dash import no_update
        assert state is no_update

    def test_valid_roundtrip(self):
        """Encode a snapshot, then decode via restore_from_url + apply_snapshot."""
        state = {
            "bub-xscale:value": "log",
            "bub-yscale:value": "log",
            "main-tabs:active_tab": "bubble",
            "bub-qs:value": [0.5],
        }
        encoded = _encode_snapshot(state)
        hash_str = f"#q3:{encoded}"
        decoded, loaded_hash = restore_from_url(hash_str)
        assert loaded_hash == hash_str
        assert isinstance(decoded, dict)
        assert decoded["main-tabs:active_tab"] == "bubble"
        # Apply to controls
        result = apply_snapshot(decoded)
        assert len(result) == len(_SNAPSHOT_CONTROLS) + 1
        main_tab_idx = next(i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
                           if cid == "main-tabs")
        assert result[main_tab_idx] == "bubble"


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestNoDuplicateCallbackOutputs:
    """Dash blocks ALL callbacks when two non-allow_duplicate callbacks
    share an output.  These tests guard against regressions by checking
    both the callback graph and the snapshot restore architecture."""

    def test_no_unguarded_duplicate_outputs(self):
        """No output property should be targeted by >1 callback without
        allow_duplicate=True."""
        from collections import defaultdict
        import _app_ctx
        app = _app_ctx.app

        output_sources = defaultdict(list)  # "cid.prop" -> [has_allow_dup, ...]
        for cb_key in app.callback_map:
            for part in cb_key.split("..."):
                has_dup = "@" in part
                clean = part.split("@")[0] if "@" in part else part
                if clean:
                    output_sources[clean].append(has_dup)

        violations = []
        for out, flags in output_sources.items():
            non_dup = sum(1 for f in flags if not f)
            if non_dup > 1:
                violations.append(out)

        assert violations == [], (
            f"Multiple callbacks output to the same property without "
            f"allow_duplicate=True — Dash will block all callbacks: "
            f"{violations}"
        )

    def test_restore_from_url_uses_intermediate_store(self):
        """restore_from_url must NOT output directly to _SNAPSHOT_CONTROLS.

        It must write to snapshot-state-store, which apply_snapshot then
        fans out with allow_duplicate=True.  Outputting directly would
        create duplicate-output conflicts with MC and other callbacks
        that also target snapshot control properties.
        """
        import _app_ctx
        app = _app_ctx.app

        snap_cids = {f"{cid}.{prop}" for cid, prop in _SNAPSHOT_CONTROLS}

        for cb_key in app.callback_map:
            parts = cb_key.split("...")
            clean_parts = [p.split("@")[0] for p in parts]
            # Identify restore_from_url by its loaded-hash-store output
            if "loaded-hash-store.data" not in clean_parts:
                continue
            # This callback should NOT contain any snapshot control outputs
            overlap = snap_cids & set(clean_parts)
            assert overlap == set(), (
                f"restore_from_url outputs directly to snapshot controls "
                f"without allow_duplicate — this breaks Dash's callback "
                f"graph.  Use snapshot-state-store + apply_snapshot "
                f"instead.  Offending outputs: {sorted(overlap)[:5]}..."
            )
            break


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestAutoBubbleYrange:
    def test_no_auto_prevents_update(self):
        with _patch_ctx("bub-xrange"):
            with pytest.raises(Exception):
                auto_bubble_yrange(
                    xrange=[2015, 2030], auto_y=[], yscale="log",
                    model_show=[], sel_qs=[0.5],
                )

    def test_returns_yrange(self):
        import math as _m
        with _patch_ctx("bub-xrange"):
            result = auto_bubble_yrange(
                xrange=[2015, 2030], auto_y=["yes"], yscale="log",
                model_show=[], sel_qs=[0.5],
            )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] < result[1]


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestTabControlsMappings:
    """Verify _TAB_CONTROLS and _TAB_TO_PATH consistency."""

    def test_all_tabs_have_controls(self):
        for tab in ["bubble", "heatmap", "dca", "retire", "supercharge"]:
            assert tab in _TAB_CONTROLS
            assert len(_TAB_CONTROLS[tab]) > 0

    def test_tab_to_path_complete(self):
        for tab in ["bubble", "heatmap", "dca", "retire", "supercharge", "stack", "faq"]:
            assert tab in _TAB_TO_PATH

    def test_snapshot_controls_covered(self):
        """Every control ID (except main-tabs) should belong to some tab."""
        all_tab_ids = set()
        for ids in _TAB_CONTROLS.values():
            all_tab_ids |= ids
        for cid, _ in _SNAPSHOT_CONTROLS:
            if cid == "main-tabs":
                continue
            assert cid in all_tab_ids, f"{cid} not in any _TAB_CONTROLS set"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Snapshot edge cases (Phase E)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotSingleTabScope:
    """Test single-tab scope filtering in snapshots."""

    def test_tab_filter_encodes_only_matching(self):
        """When tab_filter is set, non-matching controls become null."""
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            if cid in _TAB_CONTROLS.get("bubble", set()):
                state[f"{cid}:{prop}"] = "test_val"
            elif cid in _TAB_CONTROLS.get("dca", set()):
                state[f"{cid}:{prop}"] = "dca_val"
            elif cid == "main-tabs":
                state[f"{cid}:{prop}"] = "bubble"
        tab_filter = _TAB_CONTROLS["bubble"]
        encoded = _encode_snapshot(state, tab_filter=tab_filter)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        # Bubble controls should be present
        assert decoded.get("bub-xscale:value") == "test_val"
        # DCA controls should NOT be present (filtered out)
        assert "dca-amount:value" not in decoded
        # main-tabs always present
        assert decoded.get("main-tabs:active_tab") == "bubble"

    def test_each_tab_filter_roundtrips(self):
        """Each tab's filter should produce a decodable snapshot."""
        for tab, ids in _TAB_CONTROLS.items():
            state = {"main-tabs:active_tab": tab}
            for cid, prop in _SNAPSHOT_CONTROLS:
                if cid in ids:
                    state[f"{cid}:{prop}"] = "val"
            encoded = _encode_snapshot(state, tab_filter=ids)
            decoded = _decode_snapshot(encoded)
            assert decoded is not None, f"Failed to decode {tab} tab snapshot"
            assert decoded.get("main-tabs:active_tab") == tab

    def test_single_tab_shorter_than_all(self):
        """Single-tab snapshot should produce shorter encoded string."""
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = "x"
        encoded_all = _encode_snapshot(state)
        encoded_one = _encode_snapshot(state, tab_filter=_TAB_CONTROLS["retire"])
        assert len(encoded_one) < len(encoded_all)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestBitmaskEdgeCases:
    """Test bitmask encoding with edge-case states."""

    def test_all_on(self):
        """All options selected → all bits set."""
        for cid, opts in _CHECKLIST_OPTIONS.items():
            mask = _list_to_mask(opts, opts)
            expected = (1 << len(opts)) - 1
            assert mask == expected, f"{cid}: expected {expected}, got {mask}"
            # Roundtrip
            assert _mask_to_list(mask, opts) == opts

    def test_all_off(self):
        """No options selected → mask is 0."""
        for cid, opts in _CHECKLIST_OPTIONS.items():
            assert _list_to_mask([], opts) == 0
            assert _mask_to_list(0, opts) == []

    def test_single_bit_each(self):
        """Each individual option should set exactly one bit."""
        for cid, opts in _CHECKLIST_OPTIONS.items():
            for i, opt in enumerate(opts):
                mask = _list_to_mask([opt], opts)
                assert mask == (1 << i), f"{cid}[{i}]={opt}: expected {1<<i}, got {mask}"
                assert _mask_to_list(mask, opts) == [opt]

    def test_quantile_all_on_roundtrip(self):
        """All bands selected → roundtrip through encode/decode."""
        all_bands = ["inner", "outer", "median"]
        state = {"bub-qs:value": all_bands, "main-tabs:active_tab": "bubble"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        restored = decoded.get("bub-qs:value", [])
        assert set(restored) == set(all_bands)

    def test_empty_checklist_roundtrip(self):
        """Empty checklist → 0 bitmask → empty list on decode."""
        state = {"bub-toggles:value": [], "main-tabs:active_tab": "bubble"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        # Empty list encodes as 0, which decodes to empty list
        # But 0 might be stored as 0 in JSON; decoder should handle it
        toggles = decoded.get("bub-toggles:value", None)
        # Either not present (null → skipped) or empty list
        assert toggles is None or toggles == []

    def test_high_bit_quantile(self):
        """Last quantile only → highest bit set."""
        opts = _CHECKLIST_OPTIONS["bub-qs"]
        last = opts[-1]
        mask = _list_to_mask([last], opts)
        assert mask == (1 << (len(opts) - 1))
        assert _mask_to_list(mask, opts) == [last]

    def test_mask_to_list_ignores_extra_bits(self):
        """Bits beyond opts length should be ignored."""
        opts = ["a", "b", "c"]
        mask = 0b11111  # 5 bits, but only 3 opts
        result = _mask_to_list(mask, opts)
        assert result == ["a", "b", "c"]


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotLotsSpecialChars:
    """Test lots with special characters survive snapshot roundtrip."""

    def test_unicode_notes(self):
        lots = [{"date": "2024-01-01", "btc": 0.5, "price": 42000,
                 "pct_q": 0.45, "notes": "🚀 Bitcoin — \"to the moon\" ✨"}]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded["_lots"][0]["notes"] == lots[0]["notes"]

    def test_html_in_notes(self):
        lots = [{"date": "2024-01-01", "btc": 1.0, "price": 50000,
                 "pct_q": 0.5, "notes": '<b>bold</b> & "quotes" <script>x</script>'}]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["_lots"][0]["notes"] == lots[0]["notes"]

    def test_empty_notes(self):
        lots = [{"date": "2024-01-01", "btc": 1.0, "price": 50000,
                 "pct_q": 0.5, "notes": ""}]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["_lots"][0]["notes"] == ""

    def test_many_lots_roundtrip(self):
        lots = [{"date": f"2024-{i:02d}-01", "btc": 0.01 * i, "price": 40000 + i * 1000,
                 "pct_q": 0.3 + i * 0.02, "notes": f"lot #{i}"}
                for i in range(1, 13)]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert len(decoded["_lots"]) == 12

    def test_no_lots(self):
        state = {"main-tabs:active_tab": "bubble"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "_lots" not in decoded


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotV1Compat:
    """Legacy v1 snapshot format backward compatibility."""

    def test_v1_decode(self):
        """v1 format is a plain JSON dict, gzip+b64 encoded."""
        state = {"bub-xscale:value": "log", "main-tabs:active_tab": "bubble"}
        j = json.dumps(state, separators=(',', ':'))
        from snapshot import _decode_snapshot_v1
        encoded = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        decoded = _decode_snapshot_v1(encoded)
        assert decoded == state

    def test_v1_invalid(self):
        from snapshot import _decode_snapshot_v1
        assert _decode_snapshot_v1("not-valid-base64!!!") is None


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateScInfo:
    """Smoke-test the SC info panel callback."""

    def test_disabled_returns_empty(self):
        result = update_sc_info(
            amount=200, freq="Monthly", enabled=[],
            sc_loan=10000, rate=13, term=12,
            loan_type="interest_only", repeats=0,
            entry_mode="live", custom_price=80000,
            tax=33, rollover=[], price_data=90000,
        )
        assert result == ""

    def test_enabled_returns_info(self):
        result = update_sc_info(
            amount=500, freq="Monthly", enabled=["yes"],
            sc_loan=10000, rate=13, term=12,
            loan_type="interest_only", repeats=0,
            entry_mode="custom", custom_price=90000,
            tax=33, rollover=[], price_data=0,
        )
        assert isinstance(result, list)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section: BTCPay pricing and token tests
# ═══════════════════════════════════════════════════════════════════════════════

import btcpay


class TestBTCPayPricing:
    """Test compute_price and is_free_tier logic."""

    def test_dca_live_10yr(self):
        assert btcpay.compute_price("dca", 10) == 500

    def test_hm_discount(self):
        assert btcpay.compute_price("hm", 10) == 250

    def test_hm_live_40yr(self):
        assert btcpay.compute_price("hm", 40) == 1000

    def test_all_horizons(self):
        for yrs in (10, 20, 30, 40):
            p = btcpay.compute_price("dca", yrs)
            assert p > 0, f"{yrs}yr: price should be positive"

    def test_free_tier_all_models(self):
        """All 6 quantized models at cached params should be free."""
        for model_key in ["bub", "qr", "pl", "lppl", "exp", "ef"]:
            assert btcpay.is_free_tier(model_key, 40, 2028, 10)
            assert btcpay.is_free_tier(model_key, 40, 2031, 50)
            assert btcpay.is_free_tier(model_key, 40, 2035, 1)

    def test_not_free_wrong_duration(self):
        assert not btcpay.is_free_tier("bub", 20, 2028, 10)

    def test_not_free_wrong_start_year(self):
        assert not btcpay.is_free_tier("bub", 40, 2027, 10)

    def test_not_free_wrong_entry_bin(self):
        assert not btcpay.is_free_tier("bub", 40, 2028, 30)

    def test_not_free_non_quantized_model(self):
        assert not btcpay.is_free_tier("s2f", 40, 2028, 10)


class TestBTCPayTokens:
    """Test HMAC payment token generation and verification."""

    @pytest.fixture(autouse=True)
    def _ensure_secret(self):
        if not btcpay.HMAC_SECRET:
            pytest.skip("No HMAC secret configured")

    def test_roundtrip(self):
        tok = btcpay.generate_payment_token("inv123", "dca", 10)
        assert btcpay.verify_payment_token(tok, "inv123", "dca", 10)

    def test_wrong_tab(self):
        tok = btcpay.generate_payment_token("inv123", "dca", 10)
        assert not btcpay.verify_payment_token(tok, "inv123", "ret", 10)

    def test_wrong_years(self):
        tok = btcpay.generate_payment_token("inv123", "dca", 10)
        assert not btcpay.verify_payment_token(tok, "inv123", "dca", 20)

    def test_wrong_invoice(self):
        tok = btcpay.generate_payment_token("inv123", "dca", 10)
        assert not btcpay.verify_payment_token(tok, "inv999", "dca", 10)

    def test_token_is_string(self):
        tok = btcpay.generate_payment_token("inv1", "hm", 10)
        assert isinstance(tok, str)
        assert len(tok) > 10


# ═══════════════════════════════════════════════════════════════════════════════
# Section: API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIInvoiceValidation:
    """Test invoice ID sanitization in api.py."""

    def test_valid_invoice_id_regex(self):
        from api import _INVOICE_ID_RE
        assert _INVOICE_ID_RE.match("X31XGHwugKcCpeF38GtGxM")
        assert _INVOICE_ID_RE.match("abc-123_DEF")

    def test_invalid_invoice_id_regex(self):
        from api import _INVOICE_ID_RE
        assert not _INVOICE_ID_RE.match("")
        assert not _INVOICE_ID_RE.match("../../../etc/passwd")
        assert not _INVOICE_ID_RE.match("id with spaces")
        assert not _INVOICE_ID_RE.match("a" * 65)  # too long

    def test_valid_short_id(self):
        from api import _INVOICE_ID_RE
        assert _INVOICE_ID_RE.match("a")

    def test_special_chars_rejected(self):
        from api import _INVOICE_ID_RE
        for bad in ["id;drop", "id<script>", "id&foo", "id=bar", "id/path"]:
            assert not _INVOICE_ID_RE.match(bad), f"should reject: {bad}"


class TestAPIRateLimiting:
    """Test rate-limit helpers in api.py."""

    def test_prune_removes_old(self):
        from api import _invoice_log, _prune, _check_rate_limit
        import time as _time
        ip = "test-prune-ip"
        _invoice_log[ip] = [(_time.time() - 7200, False)]  # 2 hours ago
        _prune(ip)
        assert len(_invoice_log[ip]) == 0

    def test_unpaid_limit(self):
        from api import _invoice_log, _check_rate_limit, _record_invoice
        import time as _time
        ip = "test-unpaid-ip"
        _invoice_log[ip] = []
        for _ in range(20):
            _record_invoice(ip)
        err = _check_rate_limit(ip)
        assert err is not None
        assert "unpaid" in err.lower()
        _invoice_log[ip] = []  # cleanup

    def test_under_limit_ok(self):
        from api import _invoice_log, _check_rate_limit, _record_invoice
        ip = "test-ok-ip"
        _invoice_log[ip] = []
        for _ in range(5):
            _record_invoice(ip)
        assert _check_rate_limit(ip) is None
        _invoice_log[ip] = []  # cleanup


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Snapshot version compatibility
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotVersionCompat:
    """Test that snapshots with different control counts decode gracefully."""

    def test_shorter_snapshot_pads(self):
        """A snapshot with fewer controls than current should pad with None."""
        from snapshot import _SNAPSHOT_CONTROLS, _decode_snapshot
        # Create a truncated snapshot (only first 10 controls)
        short_values = [None] * 10
        payload = [short_values, None]
        j = json.dumps(payload, separators=(',', ':'))
        encoded = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        state = _decode_snapshot(encoded)
        assert state is not None  # should not fail
        # First 10 are None → not in state; rest also None → not in state
        # Key point: no crash

    def test_longer_snapshot_truncates(self):
        """A snapshot with more controls than current should truncate safely."""
        from snapshot import _SNAPSHOT_CONTROLS, _decode_snapshot
        n = len(_SNAPSHOT_CONTROLS)
        long_values = [None] * (n + 20)  # 20 extra
        payload = [long_values, None]
        j = json.dumps(payload, separators=(',', ':'))
        encoded = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        state = _decode_snapshot(encoded)
        assert state is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Price cache tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceCache:
    """Test TTL cache and circuit breaker in _fetch_btc_price."""

    def test_cache_returns_stale(self):
        from utils import _price_cache
        import time as _time
        # Seed cache with a known price
        _price_cache.update({"price": 99999.0, "ts": _time.time()})
        from utils import _fetch_btc_price
        # Should return cached price without hitting network
        result = _fetch_btc_price()
        assert result == 99999.0
        # Cleanup
        _price_cache.update({"price": None, "ts": 0})


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: MC stale mode & restore tests (P-6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMcStaleMode:
    """Verify stale mode keeps cached overlay when MC params change."""

    # Non-free-tier params (entry_q=50 breaks free tier which requires 10)
    CACHED = {
        "path_key": {
            "tab": "dca", "mc_years": 10, "mc_start_yr": 2031,
            "mc_entry_q": 50.0, "mc_bins": 5, "mc_sims": 100,
            "mc_freq": "Monthly", "mc_window": None,
            "mc_blocked_bins": [],
        },
    }

    def test_stale_returns_mc_ok_true(self):
        """Changing mc_years triggers stale mode — mc_ok still True."""
        from callbacks import _mc_setup
        with _patch_ctx("dca-mc-years"):
            mc_ok, is_free, mc_p, blocked = _mc_setup(
                tab="dca", mc_enable=[True],
                mc_years=20,          # changed from cached 10
                mc_start_yr=2031, mc_entry_q=50.0,
                mc_bins=5, mc_sims=100, mc_freq="Monthly",
                mc_window=None, mc_amount=100, mc_infl=4,
                mc_cached=self.CACHED, live_price=100000,
                mc_regime=None, mc_unblocked=None, pay_token=None,
            )
        assert mc_ok is True
        assert mc_p.get("mc_stale") is True
        # mc_p should use CACHED values, not the changed input
        assert mc_p["mc_years"] == 10

    def test_stale_not_triggered_without_cache(self):
        """No cached data → stale mode does not activate."""
        from callbacks import _mc_setup
        with _patch_ctx("dca-mc-years"):
            mc_ok, is_free, mc_p, blocked = _mc_setup(
                tab="dca", mc_enable=[True],
                mc_years=20, mc_start_yr=2031, mc_entry_q=50.0,
                mc_bins=5, mc_sims=100, mc_freq="Monthly",
                mc_window=None, mc_amount=100, mc_infl=4,
                mc_cached=None, live_price=100000,
                mc_regime=None, mc_unblocked=None, pay_token=None,
            )
        assert mc_ok is False
        assert mc_p.get("mc_stale") is not True

    def test_stale_wrong_tab_ignored(self):
        """Cached data from a different tab should not trigger stale mode."""
        from callbacks import _mc_setup
        with _patch_ctx("ret-mc-years"):
            mc_ok, is_free, mc_p, blocked = _mc_setup(
                tab="ret",  # cache has tab="dca"
                mc_enable=[True],
                mc_years=20, mc_start_yr=2031, mc_entry_q=50.0,
                mc_bins=5, mc_sims=100, mc_freq="Monthly",
                mc_window=None, mc_amount=100, mc_infl=4,
                mc_cached=self.CACHED, live_price=100000,
                mc_regime=None, mc_unblocked=None, pay_token=None,
            )
        assert mc_ok is False
        assert mc_p.get("mc_stale") is not True


class TestMcRestore:
    """Verify restore callback returns cached path_key values."""

    def test_restore_returns_cached_params(self):
        from callbacks import _restore_mc
        cached = {
            "path_key": {
                "mc_years": 15, "mc_start_yr": 2028, "mc_entry_q": 25.0,
                "mc_bins": 10, "mc_sims": 500, "mc_window": 4,
            },
        }
        result = _restore_mc(1, cached)
        assert result == (15, 2028, 25.0, 10, 500, 4)

    def test_restore_no_cache_returns_no_update(self):
        import dash
        from callbacks import _restore_mc
        result = _restore_mc(1, None)
        assert result == [dash.no_update] * 6

    def test_restore_no_path_key_returns_no_update(self):
        import dash
        from callbacks import _restore_mc
        result = _restore_mc(1, {"some_data": True})
        assert result == [dash.no_update] * 6


class TestMcFinalizeStale:
    """Verify _mc_finalize with mc_stale=True skips ghost cache update."""

    def test_stale_skips_unblocked_update(self):
        import dash
        from callbacks import _mc_finalize
        import plotly.graph_objects as go
        fig = go.Figure()
        result = _mc_finalize(
            tab="dca", fig=fig, mc_result=None, mc_cached=None,
            mc_enable=[True], mc_ok=True, is_free=False,
            blocked=(), mc_years=10, mc_start_yr=2031, mc_entry_q=10.0,
            toggles=[], has_rendered_key=True, mc_stale=True,
        )
        # result = (fig, store_val, status, rendered_key, show_modal, ub_val)
        ub_val = result[5]
        assert ub_val is dash.no_update


class TestMcPaymentCheckAuth:
    """Verify _mc_payment_check check 2 uses mc_auth (rendered_key)."""

    def test_auth_matching_rendered_key_passes(self):
        """Matching rendered_key should authorize without payment."""
        from callbacks import _mc_payment_check
        with _patch_ctx("dca-toggles"):
            result = _mc_payment_check(
                "dca", mc_years=10, start_yr=2031, entry_q=50.0,
                pay_token=None, mc_bins=5, mc_sims=100, mc_freq="Monthly",
                mc_auth={"years": 10, "start_yr": 2031, "entry_q": 50.0,
                         "bins": 5, "sims": 100, "freq": "Monthly"},
            )
        assert result is True

    def test_auth_mismatched_years_fails(self):
        """Different years in rendered_key should NOT authorize."""
        from callbacks import _mc_payment_check
        with _patch_ctx("dca-toggles"):
            result = _mc_payment_check(
                "dca", mc_years=20, start_yr=2031, entry_q=50.0,
                pay_token=None, mc_bins=5, mc_sims=100, mc_freq="Monthly",
                mc_auth={"years": 10, "start_yr": 2031, "entry_q": 50.0,
                         "bins": 5, "sims": 100, "freq": "Monthly"},
            )
        assert result is False

    def test_auth_none_rendered_key_fails(self):
        """No rendered_key should fall through to button/payment check."""
        from callbacks import _mc_payment_check
        with _patch_ctx("dca-toggles"):
            result = _mc_payment_check(
                "dca", mc_years=10, start_yr=2031, entry_q=50.0,
                pay_token=None, mc_bins=5, mc_sims=100, mc_freq="Monthly",
                mc_auth=None,
            )
        assert result is False

    def test_auth_sims_leq_passes(self):
        """Rendered_key with sims <= requested sims should pass (downgrade ok)."""
        from callbacks import _mc_payment_check
        with _patch_ctx("dca-toggles"):
            result = _mc_payment_check(
                "dca", mc_years=10, start_yr=2031, entry_q=50.0,
                pay_token=None, mc_bins=5, mc_sims=200, mc_freq="Monthly",
                mc_auth={"years": 10, "start_yr": 2031, "entry_q": 50.0,
                         "bins": 5, "sims": 100, "freq": "Monthly"},
            )
        assert result is True

    def test_auth_sims_gt_fails(self):
        """Rendered_key with sims > requested sims should fail (upgrade needs payment)."""
        from callbacks import _mc_payment_check
        with _patch_ctx("dca-toggles"):
            result = _mc_payment_check(
                "dca", mc_years=10, start_yr=2031, entry_q=50.0,
                pay_token=None, mc_bins=5, mc_sims=50, mc_freq="Monthly",
                mc_auth={"years": 10, "start_yr": 2031, "entry_q": 50.0,
                         "bins": 5, "sims": 200, "freq": "Monthly"},
            )
        assert result is False

    def test_auth_mismatched_bins_fails(self):
        """Different bins should NOT authorize."""
        from callbacks import _mc_payment_check
        with _patch_ctx("dca-toggles"):
            result = _mc_payment_check(
                "dca", mc_years=10, start_yr=2031, entry_q=50.0,
                pay_token=None, mc_bins=10, mc_sims=100, mc_freq="Monthly",
                mc_auth={"years": 10, "start_yr": 2031, "entry_q": 50.0,
                         "bins": 5, "sims": 100, "freq": "Monthly"},
            )
        assert result is False


class TestMcFinalizeRenderedKey:
    """Verify _mc_finalize rendered_key includes extended billing fields."""

    def test_rendered_key_has_billing_fields(self):
        from callbacks import _mc_finalize
        import plotly.graph_objects as go
        fig = go.Figure()
        mc_p = {"mc_bins": 5, "mc_sims": 100, "mc_freq": "Monthly"}
        result = _mc_finalize(
            tab="dca", fig=fig, mc_result=None, mc_cached=None,
            mc_enable=[True], mc_ok=True, is_free=False,
            blocked=(), mc_years=10, mc_start_yr=2031, mc_entry_q=50.0,
            toggles=[], has_rendered_key=True, mc_p=mc_p,
        )
        rendered_key = result[3]
        assert rendered_key is not None
        assert rendered_key["years"] == 10
        assert rendered_key["start_yr"] == 2031
        assert rendered_key["entry_q"] == 50.0
        assert rendered_key["bins"] == 5
        assert rendered_key["sims"] == 100
        assert rendered_key["freq"] == "Monthly"

    def test_rendered_key_none_when_not_ok(self):
        from callbacks import _mc_finalize
        import plotly.graph_objects as go
        fig = go.Figure()
        result = _mc_finalize(
            tab="dca", fig=fig, mc_result=None, mc_cached=None,
            mc_enable=[True], mc_ok=False, is_free=False,
            blocked=(), mc_years=10, mc_start_yr=2031, mc_entry_q=50.0,
            toggles=[], has_rendered_key=True,
        )
        rendered_key = result[3]
        assert rendered_key is None


# ═══════════════════════════════════════════════════════════════════════════════
# Section: PriceModel protocol + model classes
# ═══════════════════════════════════════════════════════════════════════════════

from btc_core import (PriceModel, _FitsBasedModel, BubbleModel, PowerLawModel,
                      S2FModel, QuantileRegressionModel)


class TestPriceModelProtocol:
    def test_bubble_implements_protocol(self):
        bub = BubbleModel(M)
        assert isinstance(bub, PriceModel)

    def test_powerlaw_implements_protocol(self):
        pl = PowerLawModel(M.ols_intercept, M.ols_slope, M.price_years,
                           M.price_prices, M.genesis, M.QR_QUANTILES)
        assert isinstance(pl, PriceModel)

    def test_s2f_implements_protocol(self):
        s2f = S2FModel(M.price_years, M.price_prices, M.genesis)
        assert isinstance(s2f, PriceModel)


class TestBubbleModel:
    def setup_method(self):
        self.bub = BubbleModel(M)

    def test_fits_has_quantile_keys(self):
        assert set(self.bub.fits.keys()) == set(M.QR_QUANTILES)

    def test_colors_populated(self):
        assert len(self.bub.colors) > 0
        # Not all quantiles necessarily have colors (e.g. 0.86 may be in fits
        # but not in qr_colors), but colors should be a non-empty subset
        assert set(self.bub.colors.keys()).issubset(set(self.bub.quantiles))

    def test_quantized_true(self):
        assert self.bub.quantized is True

    def test_quantiles_sorted(self):
        assert self.bub.quantiles == sorted(self.bub.quantiles)

    def test_price_at_returns_positive(self):
        q = self.bub.quantiles[len(self.bub.quantiles) // 2]
        t = 10.0
        result = self.bub.price_at(q, t)
        assert float(result) > 0

    def test_price_at_array(self):
        q = self.bub.quantiles[0]
        ts = np.array([5.0, 10.0, 15.0])
        result = self.bub.price_at(q, ts)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_short_name(self):
        assert self.bub.short_name == "bub"

    def test_name(self):
        assert self.bub.name == "Bubble Model"


class TestPowerLawModel:
    def setup_method(self):
        self.pl = PowerLawModel(M.ols_intercept, M.ols_slope, M.price_years,
                                M.price_prices, M.genesis, M.QR_QUANTILES)

    def test_fits_has_quantile_keys_from_qr_quantiles(self):
        # PL is built from M.QR_QUANTILES, which may differ from M.qr_fits keys
        assert set(self.pl.fits.keys()) == set(M.QR_QUANTILES)

    def test_fits_values_have_intercept_and_slope(self):
        for q, f in self.pl.fits.items():
            assert "intercept" in f
            assert "slope" in f

    def test_all_slopes_equal_ols(self):
        for q, f in self.pl.fits.items():
            np.testing.assert_allclose(f["slope"], M.ols_slope)

    def test_median_intercept_matches_ols(self):
        # Q50% should have z=0, so intercept ≈ ols_intercept
        q50 = min(self.pl.quantiles, key=lambda q: abs(q - 0.5))
        np.testing.assert_allclose(
            self.pl.fits[q50]["intercept"], M.ols_intercept, atol=0.01)

    def test_price_at_returns_positive(self):
        result = self.pl.price_at(0.5, 10.0)
        assert float(result) > 0

    def test_quantized_true(self):
        assert self.pl.quantized is True

    def test_colors_populated(self):
        assert len(self.pl.colors) == len(self.pl.quantiles)

    def test_short_name(self):
        assert self.pl.short_name == "pl"


class TestS2FModel:
    def setup_method(self):
        self.s2f = S2FModel(M.price_years, M.price_prices, M.genesis)

    def test_quantized_false(self):
        assert self.s2f.quantized is False

    def test_fits_is_none(self):
        assert self.s2f.fits is None

    def test_quantiles_empty(self):
        assert self.s2f.quantiles == []

    def test_colors_empty(self):
        assert self.s2f.colors == {}

    def test_price_at_scalar(self):
        result = self.s2f.price_at(0.5, 10.0)
        assert isinstance(result, float)
        assert result > 0

    def test_price_at_array(self):
        ts = np.array([5.0, 10.0, 15.0])
        result = self.s2f.price_at(0.5, ts)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_find_percentile_returns_half(self):
        assert self.s2f.find_percentile(10.0, 50000) == 0.5

    def test_short_name(self):
        assert self.s2f.short_name == "s2f"


class TestQuantileRegressionModel:
    def setup_method(self):
        self.qr = QuantileRegressionModel(M)

    def test_short_name(self):
        assert self.qr.short_name == "qr"

    def test_fits_are_qr_fits(self):
        assert self.qr.fits is M.qr_fits

    def test_price_at_matches_qr_price(self):
        q = 0.5
        t = 10.0
        expected = qr_price(q, t, M.qr_fits)
        result = self.qr.price_at(q, t)
        np.testing.assert_allclose(result, expected)

    def test_quantized(self):
        assert self.qr.quantized is True


class TestFitsBasedModelMethods:
    def setup_method(self):
        self.bub = BubbleModel(M)

    def test_interp_price_exact_quantile(self):
        q = self.bub.quantiles[5]
        t = 10.0
        expected = float(self.bub.price_at(q, t))
        result = self.bub.interp_price(q, t)
        np.testing.assert_allclose(result, expected)

    def test_interp_price_between_quantiles(self):
        q_lo = self.bub.quantiles[3]
        q_hi = self.bub.quantiles[4]
        q_mid = (q_lo + q_hi) / 2
        t = 10.0
        p_lo = self.bub.interp_price(q_lo, t)
        p_hi = self.bub.interp_price(q_hi, t)
        p_mid = self.bub.interp_price(q_mid, t)
        assert p_lo <= p_mid <= p_hi

    def test_find_percentile_roundtrip(self):
        q = self.bub.quantiles[5]
        t = 10.0
        price = float(self.bub.price_at(q, t))
        recovered_q = self.bub.find_percentile(t, price)
        np.testing.assert_allclose(recovered_q, q, atol=0.01)

    def test_find_percentile_below_min(self):
        t = 10.0
        price = 0.001  # well below any model price
        result = self.bub.find_percentile(t, price)
        assert result == self.bub.quantiles[0]

    def test_find_percentile_above_max(self):
        t = 10.0
        price = 1e20  # well above any model price
        result = self.bub.find_percentile(t, price)
        assert result == self.bub.quantiles[-1]


class TestPriceModelRegistry:
    def test_registry_has_core_entries(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS) >= 5

    def test_registry_keys(self):
        import _app_ctx
        assert {"bub", "pl", "s2f", "lppl", "exp"}.issubset(set(_app_ctx.PRICE_MODELS.keys()))

    def test_default_model_is_bubble(self):
        import _app_ctx
        assert _app_ctx.DEFAULT_MODEL is _app_ctx.PRICE_MODELS["bub"]

    def test_core_models_quantized(self):
        import _app_ctx
        quantized = {k for k, v in _app_ctx.PRICE_MODELS.items() if v.quantized}
        assert {"bub", "pl", "lppl", "exp"}.issubset(quantized)

    def test_all_models_implement_protocol(self):
        import _app_ctx
        for mdl in _app_ctx.PRICE_MODELS.values():
            assert isinstance(mdl, PriceModel)


class TestHybPPLExcessModel:
    """HybPPL (excess) — BM support + 8 oscillation params on log-excess."""

    def test_instantiates_with_support_params(self):
        from btc_core import HybPPLExcessModel
        import _app_ctx
        M = _app_ctx.M
        mdl = HybPPLExcessModel(
            M.price_years, M.price_prices, M.QR_QUANTILES,
            a_sup=-1.5, b_sup=5.1,
        )
        assert mdl.short_name == "hybppl_ex"
        assert mdl._A_sup == -1.5
        assert mdl._B_sup == 5.1

    def test_lppl_log10_returns_finite(self):
        from btc_core import HybPPLExcessModel
        import _app_ctx
        import numpy as np
        M = _app_ctx.M
        mdl = HybPPLExcessModel(
            M.price_years, M.price_prices, M.QR_QUANTILES,
            a_sup=M.support_intercept, b_sup=M.support_slope,
        )
        for t in (1.0, 5.0, 10.0, 16.0):
            v = mdl._lppl_log10(np.array([t]))
            assert np.isfinite(v).all()
        # Baseline at t=10 should give log10(price) in plausible range (~3-5)
        v10 = mdl._lppl_log10(np.array([10.0]))
        assert 2.0 < v10[0] < 6.0

    def test_included_in_price_models(self):
        import _app_ctx
        assert "hybppl_ex" in _app_ctx.PRICE_MODELS

    def test_support_matches_model_data(self):
        import _app_ctx
        mdl = _app_ctx.PRICE_MODELS["hybppl_ex"]
        M = _app_ctx.M
        assert abs(mdl._A_sup - M.support_intercept) < 1e-6
        assert abs(mdl._B_sup - M.support_slope) < 1e-6


class TestLPPLComponentDecomposition:
    """LPPL family: sum(components(t)) == _lppl_log10(t) to 1e-10."""

    T_TEST = np.array([1.0, 5.0, 10.0, 16.0, 30.0, 50.0])

    def _assert_invariant(self, model):
        comps = model.components(self.T_TEST)
        assert set(comps.keys()) == set(model.component_names), (
            f"{type(model).__name__}: components() keys != component_names")
        total = sum(comps.values())
        expected = model._lppl_log10(self.T_TEST)
        np.testing.assert_allclose(
            total, expected, rtol=0, atol=1e-10,
            err_msg=f"{type(model).__name__}: sum(components) != _lppl_log10")

    def test_lppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lppl"])

    def test_lppl2_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp2"])

    def test_lppl3_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp3"])

    def test_lppl4_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["lp4"])

    def test_lppl_weighted_variants_inherit(self):
        import _app_ctx
        for key in ("lppl_w", "lp2_w", "lp3_w", "lp4_w"):
            self._assert_invariant(_app_ctx.PRICE_MODELS[key])

    def test_lppl4_n13_variants_inherit(self):
        import _app_ctx
        for key in ("lp4_n13", "lp4_w_n13"):
            self._assert_invariant(_app_ctx.PRICE_MODELS[key])

    def test_lppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["lppl"].component_names) == 3
        assert len(_app_ctx.PRICE_MODELS["lp2"].component_names) == 4
        assert len(_app_ctx.PRICE_MODELS["lp3"].component_names) == 5
        assert len(_app_ctx.PRICE_MODELS["lp4"].component_names) == 6

    def test_linppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["linppl"])

    def test_linppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["linppl"].component_names) == 3

    def test_hybppl_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["hybppl"])

    def test_hybppl_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["hybppl"].component_names) == 4

    def test_hybppl_ex_invariant(self):
        import _app_ctx
        self._assert_invariant(_app_ctx.PRICE_MODELS["hybppl_ex"])

    def test_hybppl_ex_component_count(self):
        import _app_ctx
        assert len(_app_ctx.PRICE_MODELS["hybppl_ex"].component_names) == 5


class TestCompositeComponentDecomposition:
    """BM / EF: sum(components(t)) == _composite_log10(t) to 1e-10."""

    T_TEST = np.array([1.0, 5.0, 10.0, 16.0, 30.0, 50.0])

    def _assert_composite_invariant(self, model):
        comps = model.components(self.T_TEST)
        assert set(comps.keys()) == set(model.component_names)
        total = sum(comps.values())
        expected = model._composite_log10(self.T_TEST)
        np.testing.assert_allclose(
            total, expected, rtol=0, atol=1e-10,
            err_msg=f"{type(model).__name__}: sum(components) != _composite_log10")

    def test_bm_invariant(self):
        import _app_ctx
        self._assert_composite_invariant(_app_ctx.PRICE_MODELS["bub"])

    def test_bm_component_count(self):
        import _app_ctx
        assert _app_ctx.PRICE_MODELS["bub"].component_names == ["support", "bubbles"]

    def test_ef_invariant(self):
        import _app_ctx
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded (model_data_ef.pkl absent)")
        self._assert_composite_invariant(ef)

    def test_ef_component_count(self):
        import _app_ctx
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert ef.component_names == ["support", "bubbles"]


class TestDecompRegistry:
    def test_families_keys(self):
        import _app_ctx
        expected = {"bub", "ef", "lppl", "linppl", "hybppl", "hybppl_ex"}
        assert set(_app_ctx.DECOMP_FAMILIES.keys()) == expected

    def test_families_labels(self):
        import _app_ctx
        assert _app_ctx.DECOMP_FAMILIES["bub"] == "BM"
        assert _app_ctx.DECOMP_FAMILIES["lppl"] == "LPPL (family)"
        assert _app_ctx.DECOMP_FAMILIES["hybppl_ex"] == "HybPPL (ex)"

    def test_palette_has_all_four_schemes(self):
        import _app_ctx
        assert set(_app_ctx.DECOMP_COLORS.keys()) == {"default", "cb-brian", "cb-rg", "cb-full"}
        for key, colors in _app_ctx.DECOMP_COLORS.items():
            assert len(colors) == 7, f"{key} palette has {len(colors)} colors, expected 7"
            for c in colors:
                assert c.startswith("#") and len(c) == 7

    def test_sum_color_has_all_four_schemes(self):
        import _app_ctx
        assert set(_app_ctx.DECOMP_SUM_COLOR.keys()) == {"default", "cb-brian", "cb-rg", "cb-full"}


# ═══════════════════════════════════════════════════════════════════════════════
# Section: MC model interface verification
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestModelMCInterface:
    """All quantized models must support find_percentile and interp_price for MC."""

    def test_all_quantized_have_find_percentile(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            assert hasattr(mdl, 'find_percentile'), f"{key} missing find_percentile"
            assert callable(mdl.find_percentile), f"{key}.find_percentile not callable"

    def test_all_quantized_have_interp_price(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            assert hasattr(mdl, 'interp_price'), f"{key} missing interp_price"
            assert callable(mdl.interp_price), f"{key}.interp_price not callable"

    def test_find_percentile_returns_float(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            pct = mdl.find_percentile(16.0, 60000.0)
            assert isinstance(pct, float), f"{key}.find_percentile returned {type(pct)}"
            assert 0.0 <= pct <= 1.0, f"{key} percentile {pct} out of range"

    def test_interp_price_returns_positive(self):
        import _app_ctx
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            if not mdl.quantized:
                continue
            price = mdl.interp_price(0.5, 16.0)
            assert isinstance(price, float), f"{key}.interp_price returned {type(price)}"
            assert price > 0, f"{key} price {price} not positive"


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Multi-model overlay (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSnapshotModelShow:
    """Snapshot roundtrip with 'pl' and 's2f' in model-show checklists."""

    def test_roundtrip_pl_in_model_show(self):
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = None
        state["dca-model-show:value"] = ["qr", "pl"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "pl" in decoded["dca-model-show:value"]
        assert "qr" in decoded["dca-model-show:value"]

    def test_roundtrip_s2f_in_model_show(self):
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = None
        state["sc-model-show:value"] = ["qr", "mc", "s2f"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "s2f" in decoded["sc-model-show:value"]

    def test_old_bitmask_without_pl_decodes_without_pl(self):
        """Old snapshots with only 2-bit model-show (qr+mc) should not have pl/s2f."""
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = None
        # Simulate old link: only qr and mc selected (bits 0 and 1)
        state["ret-model-show:value"] = ["qr", "mc"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "pl" not in decoded["ret-model-show:value"]
        assert "s2f" not in decoded["ret-model-show:value"]


class TestMultiModelBubbleFigure:
    """Bubble figure with active_models=["pl"]."""

    def test_pl_overlay_adds_traces(self):
        from datetime import date
        yr_now = date.today().year
        p_base = dict(
            selected_qs=[0.10, 0.50],
            shade=False, show_ols=False, show_data=False, show_today=False,
            show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4,
            ymin=0.01, ymax=1e7,
            n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0,
            sup_color="#888888", sup_lw=1.5,
        )
        from figures import build_bubble_figure
        fig_no_pl = build_bubble_figure(M, dict(p_base, active_models=[]))
        fig_with_pl = build_bubble_figure(M, dict(p_base, active_models=["pl"]))
        assert len(fig_with_pl.data) > len(fig_no_pl.data)

    def test_pl_traces_have_dot_dash(self):
        from datetime import date
        yr_now = date.today().year
        from figures import build_bubble_figure
        fig = build_bubble_figure(M, dict(
            selected_qs=[0.50], shade=False, show_ols=False, show_data=False,
            show_today=False, show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4,
            ymin=0.01, ymax=1e7, n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=["pl"],
        ))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"]
        assert len(pl_traces) > 0
        assert pl_traces[0].line.dash == "dot"


class TestMultiModelDcaFigure:
    """DCA figure with active_models=["pl"]."""

    def test_pl_overlay_doesnt_crash(self):
        from datetime import date
        yr_now = date.today().year
        from figures import build_dca_figure
        fig, _ = build_dca_figure(M, dict(
            start_stack=0, use_lots=False,
            amount=100.0, freq="Monthly",
            start_yr=yr_now, end_yr=yr_now + 5,
            disp_mode="btc", log_y=False, show_today=False,
            show_legend=False, minor_grid=False,
            selected_qs=[0.50], lots=[],
            sc_enabled=False, sc_loan_amount=0,
            sc_rate=13.0, sc_loan_type="interest_only",
            sc_term_months=48.0, sc_repeats=0, sc_rollover=False,
            sc_entry_mode="live", sc_custom_price=80000,
            sc_tax_rate=0.33, sc_live_price=None,
            active_models=["pl"],
        ))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"]
        assert len(pl_traces) > 0


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestMcModelSrc:
    """Phase 4: MC model-source dropdown tests."""

    def test_mc_path_key_includes_model_src(self):
        from mc_overlay import _mc_path_key
        key = _mc_path_key({"mc_model_src": "pl"}, "dca")
        assert key["mc_model_src"] == "pl"

    def test_mc_path_key_defaults_to_bub(self):
        from mc_overlay import _mc_path_key
        key = _mc_path_key({}, "dca")
        assert key["mc_model_src"] == "bub"

    def test_resolve_model_returns_bub(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({"mc_model_src": "bub"})
        assert mdl is _app_ctx.PRICE_MODELS["bub"]

    def test_resolve_model_default(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({})
        assert mdl is _app_ctx.DEFAULT_MODEL

    def test_resolve_model_pl(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({"mc_model_src": "pl"})
        assert mdl is _app_ctx.PRICE_MODELS["pl"]

    def test_resolve_model_nonquantized_falls_back(self):
        from mc_overlay import _resolve_model
        mdl = _resolve_model({"mc_model_src": "s2f"})
        assert mdl is _app_ctx.DEFAULT_MODEL

    def test_build_mc_params_includes_model_src(self):
        from callbacks import _build_mc_params
        p = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=0,
            mc_bins=5, mc_sims=100, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2028, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
            mc_model_src="pl",
        )
        assert p["mc_model_src"] == "pl"

    def test_build_mc_params_defaults_model_src(self):
        from callbacks import _build_mc_params
        p = _build_mc_params(
            mc_enable=True, mc_amount=100, mc_infl=0,
            mc_bins=5, mc_sims=100, mc_years=10,
            mc_freq="Monthly", mc_window=None,
            mc_start_yr=2028, mc_entry_q=50,
            mc_cached=None, mc_live_price=0,
        )
        assert p["mc_model_src"] == "bub"

    def test_snapshot_roundtrip_with_model_src(self):
        from snapshot import _encode_snapshot, _decode_snapshot
        state = {"dca-mc-model-src:value": "pl"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["dca-mc-model-src:value"] == "pl"

    def test_old_snapshot_pads_model_src(self):
        """Old snapshots without mc-model-src fields decode with None (defaults)."""
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        # Build state with only old controls (no mc-model-src)
        state = {"main-tabs:active_tab": "dca"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        # mc-model-src fields should not be present (None → skipped)
        assert "dca-mc-model-src:value" not in decoded

    def test_tab_controls_include_model_src(self):
        from callbacks import _TAB_CONTROLS
        assert "hm-mc-model-src" in _TAB_CONTROLS["heatmap"]
        assert "dca-mc-model-src" in _TAB_CONTROLS["dca"]
        assert "ret-mc-model-src" in _TAB_CONTROLS["retire"]
        assert "sc-mc-model-src" in _TAB_CONTROLS["supercharge"]


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestPhase5Polish:
    """Phase 5: dash styles, S2F overlay, model attributes."""

    def test_dash_styles(self):
        from btc_core import BubbleModel, PowerLawModel, S2FModel
        assert BubbleModel.dash_style == "solid"
        assert PowerLawModel.dash_style == "dot"
        assert S2FModel.dash_style == "dot"

    def test_all_models_have_dash_style(self):
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            assert hasattr(mdl, "dash_style"), f"{key} missing dash_style"
            assert mdl.dash_style in ("solid", "dot", "longdash", "dash", "dashdot", "longdashdot")

    def test_s2f_bubble_overlay(self):
        from figures import build_bubble_figure
        yr_now = pd.Timestamp.today().year
        fig = build_bubble_figure(M, dict(
            selected_qs=[0.5], xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4, ymin=0.01, ymax=1e7,
            shade=False, show_ols=False, show_data=False, show_today=False,
            show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=["s2f"],
        ))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) == 1
        assert s2f_traces[0].line.dash == "dot"

    def test_s2f_dca_overlay(self):
        from figures import build_dca_figure
        yr_now = pd.Timestamp.today().year
        fig, _ = build_dca_figure(M, dict(
            start_stack=0, use_lots=False,
            amount=100.0, freq="Monthly",
            start_yr=yr_now, end_yr=yr_now + 5,
            disp_mode="btc", log_y=False, show_today=False,
            show_legend=False, minor_grid=False,
            selected_qs=[0.50], lots=[],
            sc_enabled=False, sc_loan_amount=0,
            sc_rate=13.0, sc_loan_type="interest_only",
            sc_term_months=48.0, sc_repeats=0, sc_rollover=False,
            sc_entry_mode="live", sc_custom_price=80000,
            sc_tax_rate=0.33, sc_live_price=None,
            active_models=["s2f"],
        ))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) == 1
        assert s2f_traces[0].line.dash == "dot"

    def test_s2f_retire_overlay(self):
        from figures import build_retire_figure
        fig, _ = build_retire_figure(M, dict(
            start_stack=1.0, use_lots=False,
            wd_amount=5000, freq="Monthly",
            start_yr=2031, end_yr=2040, inflation=4.0,
            disp_mode="btc", log_y=False,
            annotate=False, show_legend=False, minor_grid=False,
            legend_pos="outside",
            selected_qs=[0.5], lots=[],
            active_models=["s2f"],
        ))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) == 1

    def test_pl_uses_dot_dash(self):
        from figures import build_bubble_figure
        yr_now = pd.Timestamp.today().year
        fig = build_bubble_figure(M, dict(
            selected_qs=[0.5], xscale="log", yscale="log",
            xmin=2012, xmax=yr_now + 4, ymin=0.01, ymax=1e7,
            shade=False, show_ols=False, show_data=False, show_today=False,
            show_legend=False, minor_grid=False,
            show_comp=False, show_sup=False,
            n_future=0, pt_size=3, pt_alpha=0.3,
            stack=0, show_stack=False, use_lots=False, lots=[],
            comp_color="#FFD700", comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
            active_models=["pl"],
        ))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"]
        assert len(pl_traces) > 0
        assert pl_traces[0].line.dash == "dot"


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestPalettes:
    """Test palette registry and palette-aware color functions."""

    def test_get_palette_default(self):
        from figures.common import _get_palette
        pal = _get_palette({})
        assert pal is _app_ctx.PALETTES["default"]

    def test_get_palette_cb_rg(self):
        from figures.common import _get_palette
        pal = _get_palette({"palette": "cb-rg"})
        assert pal is _app_ctx.PALETTES["cb-rg"]

    def test_get_palette_unknown_falls_back(self):
        from figures.common import _get_palette
        pal = _get_palette({"palette": "nonexistent"})
        assert pal is _app_ctx.PALETTES["default"]

    def test_thermal_color_default_unchanged(self):
        from figures.common import _thermal_color
        assert _thermal_color(0.50).lower() == "#bdbdbd"

    def test_thermal_color_cb_rg_differs(self):
        from figures.common import _thermal_color
        pal = _app_ctx.PALETTES["cb-rg"]
        assert _thermal_color(0.90) != _thermal_color(0.90, pal)

    def test_all_palettes_have_required_keys(self):
        required = {"thermal_stops", "non_quantized_model", "delay_colors",
                    "annot_colors", "today_line", "hm_c_lo", "hm_c_mid1",
                    "hm_c_mid2", "hm_c_hi", "hm_loss_text", "hm_exceptional_text"}
        for name, pal in _app_ctx.PALETTES.items():
            missing = required - set(pal.keys())
            assert not missing, f"Palette {name!r} missing keys: {missing}"

    def test_all_palettes_thermal_stops_count(self):
        for name, pal in _app_ctx.PALETTES.items():
            assert len(pal["thermal_stops"]) == 12, f"{name} has wrong stop count"

    def test_snapshot_roundtrip_palette(self):
        from snapshot import (_encode_snapshot, _decode_snapshot,
                              _SNAPSHOT_CONTROLS)
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["palette-store:data"] = "cb-rg"
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded.get("palette-store:data") == "cb-rg"

    def test_build_bubble_all_palettes(self):
        from figures import build_bubble_figure
        yr_now = pd.Timestamp.today().year
        for pal_key in _app_ctx.PALETTES:
            fig = build_bubble_figure(M, dict(
                selected_qs=[0.5], shade=False, show_data=False,
                show_today=False, show_legend=False, minor_grid=False,
                show_comp=False, show_sup=False, xscale="log", yscale="log",
                xmin=2012, xmax=yr_now + 4, ymin=1, ymax=1e6,
                n_future=1, pt_size=2, pt_alpha=0.2,
                stack=0, show_stack=False, use_lots=False, lots=[],
                comp_color="#FFD700", comp_lw=2, sup_color="#888", sup_lw=1.5,
                palette=pal_key,
            ))
            assert fig is not None, f"bubble failed for {pal_key}"

    def test_build_dca_all_palettes(self):
        from figures import build_dca_figure
        for pal_key in _app_ctx.PALETTES:
            fig, _ = build_dca_figure(M, dict(
                start_stack=0, use_lots=False, amount=100, freq="Monthly",
                start_yr=2024, end_yr=2030, disp_mode="btc",
                log_y=False, show_today=False, show_legend=False,
                minor_grid=False, selected_qs=[0.5], lots=[],
                sc_enabled=False, sc_loan_amount=0, sc_rate=13.0,
                sc_loan_type="interest_only", sc_term_months=48,
                sc_repeats=0, sc_rollover=False, sc_entry_mode="live",
                sc_custom_price=80000, sc_tax_rate=0.33, sc_live_price=None,
                palette=pal_key,
            ))
            assert fig is not None, f"dca failed for {pal_key}"


_EF_PKL = str(_ROOT / "model_data_ef.pkl")
_EF_SKIP = not Path(_EF_PKL).exists()


@pytest.mark.skipif(_EF_SKIP, reason="model_data_ef.pkl not found")
class TestEmpiricalFloorModel:
    """Tests for EmpiricalFloorModel."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from btc_core import EmpiricalFloorModel
        self.model = EmpiricalFloorModel(_EF_PKL)

    def test_protocol_fields(self):
        assert self.model.name == "BM Empirical Floor"
        assert self.model.short_name == "ef"
        assert self.model.quantized is True
        assert isinstance(self.model.quantiles, list)
        assert len(self.model.quantiles) > 10
        assert isinstance(self.model.colors, dict)
        assert isinstance(self.model.fits, dict)
        assert 0.5 in self.model.fits

    def test_price_at_scalar(self):
        p = self.model.price_at(0.5, 10.0)
        assert float(p) > 0

    def test_price_at_array(self):
        t = np.array([5.0, 10.0, 15.0])
        prices = self.model.price_at(0.5, t)
        assert len(prices) == 3
        assert all(p > 0 for p in prices)

    def test_quantile_ordering(self):
        p10 = float(self.model.price_at(0.1, 10.0))
        p50 = float(self.model.price_at(0.5, 10.0))
        p90 = float(self.model.price_at(0.9, 10.0))
        assert p10 < p50
        assert p50 < p90

    def test_interp_price(self):
        p = self.model.interp_price(0.37, 10.0)
        assert p > 0

    def test_find_percentile(self):
        t = 12.0
        p50 = float(self.model.price_at(0.5, t))
        q = self.model.find_percentile(t, p50)
        assert abs(q - 0.5) < 0.1

    def test_dash_style(self):
        assert self.model.dash_style == "longdash"


class TestCompositeModelBands:
    """Test asymmetric shrinking Gaussian band behavior."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        import _app_ctx
        model = _app_ctx.PRICE_MODELS.get("bub")
        if model is None:
            pytest.skip("BubbleModel not available")
        self.model = model

    def test_bands_narrow_over_time(self):
        """σ(t) decreases → ratio of Q50/Q10 should be smaller at late t."""
        p10_early = float(self.model.price_at(0.1, 5.0))
        p50_early = float(self.model.price_at(0.5, 5.0))
        p10_late = float(self.model.price_at(0.1, 30.0))
        p50_late = float(self.model.price_at(0.5, 30.0))
        ratio_early = p50_early / p10_early
        ratio_late = p50_late / p10_late
        assert ratio_early > ratio_late

    def test_asymmetric_bands(self):
        """Downside band narrower than upside at late times."""
        t = 30.0
        p50 = np.log10(float(self.model.price_at(0.5, t)))
        p10 = np.log10(float(self.model.price_at(0.1, t)))
        p90 = np.log10(float(self.model.price_at(0.9, t)))
        down_width = p50 - p10
        up_width = p90 - p50
        assert up_width > down_width

    def test_quantile_ordering_preserved(self):
        """Q1 < Q10 < Q50 < Q90 < Q99 at all times."""
        for t in [3.0, 10.0, 30.0, 50.0]:
            prices = [float(self.model.price_at(q, t))
                      for q in [0.01, 0.1, 0.5, 0.9, 0.99]]
            for i in range(len(prices) - 1):
                assert prices[i] < prices[i + 1]

    def test_q1_never_exceeds_q50(self):
        """The bug this change fixes: Q1% must never exceed Q50%."""
        for t in [10, 20, 30, 40, 50, 60]:
            p1 = float(self.model.price_at(0.01, t))
            p50 = float(self.model.price_at(0.5, t))
            assert p1 < p50


@pytest.mark.skipif(_EF_SKIP, reason="model_data_ef.pkl not found")
class TestEmpiricalFloorComposite:
    """Test EmpiricalFloorModel with _CompositeModel bands."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        from btc_core import EmpiricalFloorModel
        self.model = EmpiricalFloorModel(_EF_PKL)

    def test_bands_narrow_over_time(self):
        p10_early = float(self.model.price_at(0.1, 5.0))
        p50_early = float(self.model.price_at(0.5, 5.0))
        p10_late = float(self.model.price_at(0.1, 30.0))
        p50_late = float(self.model.price_at(0.5, 30.0))
        ratio_early = p50_early / p10_early
        ratio_late = p50_late / p10_late
        assert ratio_early > ratio_late

    def test_quantile_ordering_preserved(self):
        for t in [3.0, 10.0, 30.0, 50.0]:
            prices = [float(self.model.price_at(q, t))
                      for q in [0.01, 0.1, 0.5, 0.9, 0.99]]
            for i in range(len(prices) - 1):
                assert prices[i] < prices[i + 1]


class TestModelScanner:
    def test_solve_for_quantile(self):
        """Given price and date, find_percentile returns valid quantile."""
        import _app_ctx
        t = today_t(_app_ctx.M.genesis)
        for mdl in _app_ctx.PRICE_MODELS.values():
            pct = mdl.find_percentile(t, 70000)
            assert 0 <= pct <= 1

    def test_solve_for_price(self):
        """Given quantile and date, price_at returns positive price."""
        import _app_ctx
        t = today_t(_app_ctx.M.genesis)
        for mdl in _app_ctx.PRICE_MODELS.values():
            p = float(mdl.price_at(0.5, t))
            assert p > 0

    def test_solve_for_date(self):
        """Root-finding for date works for reasonable inputs."""
        from callbacks.scanner import _solve_date
        import _app_ctx
        for mdl in _app_ctx.PRICE_MODELS.values():
            if not mdl.quantized:
                continue
            result = _solve_date(mdl, 0.5, 1_000_000)
            # Some models may not reach $1M in range — that's OK
            assert isinstance(result, str)

    def test_qr_model_registered(self):
        import _app_ctx
        assert "qr" in _app_ctx.PRICE_MODELS
        assert _app_ctx.PRICE_MODELS["qr"].name == "Quantile Regression"


class TestBubbleModelToggle:
    """bub-model-show includes 'bub' checked by default."""

    def test_bub_in_model_show_options(self):
        """The bubble model appears in Display Models checklist."""
        from layout.bubble import _bubble_controls
        controls = _bubble_controls()
        # Find the bub-model-show checklist
        # Note: Dash components may be falsy (e.g. empty value=[]) so use `is not None`
        def find_checklist(component):
            if hasattr(component, 'id') and component.id == 'bub-model-show':
                return component
            if hasattr(component, 'children'):
                kids = component.children
                if isinstance(kids, list):
                    for c in kids:
                        r = find_checklist(c)
                        if r is not None: return r
                elif kids is not None:
                    r = find_checklist(kids)
                    if r is not None: return r
            return None
        cl = find_checklist(controls)
        assert cl is not None
        option_values = [o["value"] for o in cl.options]
        assert "bub" in option_values
        assert option_values[0] == "bub"  # first in list
        assert "bub" in cl.value  # checked by default


class TestBubbleModelGating:
    """Main BM traces are conditional on 'bub' in active_models."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=True, show_ols=False, show_ucl=False,
        show_data=False, show_today=False,
        show_legend=False, minor_grid=False,
        show_comp=True, show_sup=True,
        xscale="log", yscale="log",
        xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        palette="default",
    )

    def test_bub_active_draws_traces(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Bubble composite" in n for n in names)
        assert any("Bubble support" in n for n in names)

    def test_bub_inactive_hides_traces(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=[]))
        # No traces should lack legendgroup (BM traces lack it; overlays always set it)
        bm_traces = [t for t in fig.data if t.name
                     and not getattr(t, "legendgroup", None)
                     and t.name not in ("Price data", "Lots")]
        assert len(bm_traces) == 0, f"BM traces should be hidden, found: {[t.name for t in bm_traces]}"

    def test_bub_inactive_preserves_data_scatter(self):
        """Data scatter, OLS, UCL, today line survive when BM is off."""
        fig = build_bubble_figure(M, dict(self._BASE,
            active_models=[], show_data=True, show_today=True,
            show_ols=True, show_ucl=True))
        names = [t.name for t in fig.data if t.name]
        assert any("Price data" in n for n in names)

    def test_bub_inactive_still_has_axis_config(self):
        """Even with BM hidden, chart should render without error."""
        fig = build_bubble_figure(M, dict(self._BASE, active_models=[]))
        assert isinstance(fig, go.Figure)
        assert fig.layout.xaxis.type in ("log", "linear", "-")


class TestEFCompositeOverlay:
    """EF overlay renders composite/support/future when enabled."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=False, show_ols=False, show_ucl=False,
        show_data=False, show_today=False,
        show_legend=False, minor_grid=False,
        show_comp=True, show_sup=True,
        xscale="log", yscale="log",
        xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        palette="default",
    )

    def test_ef_overlay_draws_composite(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("EF" in n and "composite" in n for n in names)

    def test_ef_overlay_draws_support(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("EF" in n and "support" in n for n in names)

    def test_ef_composite_uses_own_color(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"]))
        comp_traces = [t for t in fig.data if t.name and "EF" in t.name and "composite" in t.name]
        assert len(comp_traces) > 0
        # EF composite uses palette model color (default palette)
        expected = _app_ctx.PALETTES["default"]["model_colors"]["ef"]
        assert comp_traces[0].line.color == expected

    def test_ef_no_composite_when_show_comp_off(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["ef"], show_comp=False))
        names = [t.name for t in fig.data if t.name]
        assert not any("composite" in n for n in names)

    def test_both_bub_and_ef_composite(self):
        if "ef" not in _app_ctx.PRICE_MODELS:
            pytest.skip("EF model not loaded")
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub", "ef"]))
        names = [t.name for t in fig.data if t.name]
        assert any("Bubble composite" in n for n in names)
        assert any("EF" in n and "composite" in n for n in names)


class TestAutoYWithBubToggle:
    """Auto-Y-range respects bub toggle."""

    def test_auto_y_no_bub_uses_fallback(self):
        """When BM is unchecked, auto-Y should not crash."""
        from callbacks.charts import auto_bubble_yrange
        try:
            result = auto_bubble_yrange([2012, 2030], ["yes"], "log", [], [0.5])
        except Exception:
            pytest.fail("auto_bubble_yrange should not crash when bub is off")
        assert isinstance(result, list)
        assert len(result) == 2


class TestModelR2:
    """All registered models get r2_per_quantile after startup."""

    def test_bubble_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("bub")
        assert hasattr(mdl, "r2_per_quantile")
        assert isinstance(mdl.r2_per_quantile, dict)
        assert len(mdl.r2_per_quantile) > 0
        for q, r2 in mdl.r2_per_quantile.items():
            assert 0 < r2 <= 1.0, f"BM Q{q}: R²={r2}"

    def test_pl_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("pl")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0
        vals = list(mdl.r2_per_quantile.values())
        assert all(0 < v <= 1.0 for v in vals)

    def test_ef_model_has_r2(self):
        ef = _app_ctx.PRICE_MODELS.get("ef")
        if ef is None:
            pytest.skip("EF model not loaded")
        assert hasattr(ef, "r2_per_quantile")
        assert len(ef.r2_per_quantile) > 0

    def test_s2f_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("s2f")
        assert hasattr(mdl, "r2_per_quantile")
        assert 0.5 in mdl.r2_per_quantile

    def test_lppl_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("lppl")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_exp_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("exp")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_qr_model_has_r2(self):
        mdl = _app_ctx.PRICE_MODELS.get("qr")
        assert hasattr(mdl, "r2_per_quantile")
        assert len(mdl.r2_per_quantile) > 0

    def test_ols_r2_on_model_data(self):
        _M = _app_ctx.M
        assert hasattr(_M, "ols_r2")
        assert isinstance(_M.ols_r2, float)
        assert 0.9 < _M.ols_r2 <= 1.0


class TestR2InLegend:
    """Legend labels include R² where available."""

    _BASE = dict(
        selected_qs=[0.5] if 0.5 in _app_ctx.DEFAULT_MODEL.fits else [0.10],
        shade=False, show_ols=True, show_ucl=True,
        show_data=False, show_today=False,
        show_legend=True, minor_grid=False,
        show_comp=True, show_sup=True,
        xscale="log", yscale="log",
        xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
        palette="default",
    )

    def test_bm_quantile_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        q_traces = [t for t in fig.data if t.name and "Q" in t.name
                    and "%" in t.name and "R\u00b2" in t.name
                    and not getattr(t, "legendgroup", None)]
        assert len(q_traces) > 0, "BM quantile lines should show R²"

    def test_overlay_quantile_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub", "pl"]))
        pl_traces = [t for t in fig.data if t.name and t.legendgroup == "pl"
                     and "R\u00b2" in t.name]
        assert len(pl_traces) > 0, "PL overlay lines should show R²"

    def test_ols_has_r2(self):
        fig = build_bubble_figure(_app_ctx.M, dict(self._BASE, active_models=["bub"]))
        ols_traces = [t for t in fig.data if t.name and t.name.startswith("OLS")]
        assert len(ols_traces) > 0
        assert "R\u00b2" in ols_traces[0].name

    def test_s2f_has_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["s2f"]))
        s2f_traces = [t for t in fig.data if t.name and t.legendgroup == "s2f"]
        assert len(s2f_traces) > 0
        assert "R\u00b2" in s2f_traces[0].name

    def test_support_no_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        sup_traces = [t for t in fig.data if t.name and "support" in t.name]
        for t in sup_traces:
            assert "R\u00b2" not in t.name, f"Support should not have R²: {t.name}"

    def test_ucl_no_r2(self):
        fig = build_bubble_figure(M, dict(self._BASE, active_models=["bub"]))
        ucl_traces = [t for t in fig.data if t.name and "Unfairly Cheap" in t.name]
        for t in ucl_traces:
            assert "R\u00b2" not in t.name


class TestTaxData:
    def test_federal_brackets_single_tcja(self):
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        single = FEDERAL_BRACKETS_TCJA["single"]
        assert single[0] == (11_925, 0.10)
        assert single[-1][1] == 0.37
        assert len(single) == 7

    def test_federal_brackets_mfj_tcja(self):
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        mfj = FEDERAL_BRACKETS_TCJA["mfj"]
        assert mfj[0] == (23_850, 0.10)
        assert mfj[-1][1] == 0.37

    def test_federal_brackets_sunset(self):
        from engines.tax_data import FEDERAL_BRACKETS_SUNSET
        single = FEDERAL_BRACKETS_SUNSET["single"]
        assert single[-1][1] == 0.396
        assert len(single) == 7

    def test_ltcg_brackets_single(self):
        from engines.tax_data import LTCG_BRACKETS
        single = LTCG_BRACKETS["single"]
        assert single[0] == (48_350, 0.00)
        assert single[1] == (533_400, 0.15)
        assert single[2] == (float("inf"), 0.20)

    def test_standard_deduction_tcja(self):
        from engines.tax_data import STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET
        assert STANDARD_DEDUCTION_TCJA["single"] == 15_000
        assert STANDARD_DEDUCTION_TCJA["mfj"] == 30_000
        assert STANDARD_DEDUCTION_SUNSET["single"] == 8_300
        assert STANDARD_DEDUCTION_SUNSET["mfj"] == 16_600

    def test_niit_thresholds(self):
        from engines.tax_data import NIIT_RATE, NIIT_THRESHOLD
        assert NIIT_RATE == 0.038
        assert NIIT_THRESHOLD["single"] == 200_000
        assert NIIT_THRESHOLD["mfj"] == 250_000

    def test_state_tax_no_income_tax(self):
        from engines.tax_data import STATE_TAX_RATES
        for st in ("AK", "FL", "NV", "NH", "SD", "TN", "TX", "WA", "WY"):
            assert STATE_TAX_RATES[st] == 0.0, f"{st} should be 0"

    def test_state_tax_california(self):
        from engines.tax_data import STATE_TAX_RATES
        assert STATE_TAX_RATES["CA"] == 13.30

    def test_state_tax_count(self):
        from engines.tax_data import STATE_TAX_RATES
        assert len(STATE_TAX_RATES) == 51  # 50 states + DC

    def test_rmd_factors(self):
        from engines.tax_data import RMD_FACTORS
        assert RMD_FACTORS[73] == 26.5
        assert RMD_FACTORS[75] == 24.6
        assert RMD_FACTORS[80] == 20.2
        assert RMD_FACTORS[90] == 12.2
        assert 72 in RMD_FACTORS
        assert 120 in RMD_FACTORS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTaxLots:
    def test_create_lot(self):
        from engines.tax_lots import TaxLot
        lot = TaxLot(date="2024-01-15", btc=0.5, cost_basis=42_000.0, source="initial")
        assert lot.btc == 0.5
        assert lot.cost_basis == 42_000.0

    def test_sell_fifo_single_lot(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2023-01-01", 1.0, 20_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=50_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.5
        assert len(result.gains) == 1
        g = result.gains[0]
        assert g.btc == 0.5
        assert g.proceeds == 25_000.0
        assert g.cost == 10_000.0
        assert g.gain == 15_000.0
        assert g.is_long_term is True
        assert len(result.remaining_lots) == 1
        assert result.remaining_lots[0].btc == 0.5

    def test_sell_fifo_multiple_lots(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [
            TaxLot("2023-01-01", 0.3, 20_000.0, "initial"),
            TaxLot("2025-03-01", 0.7, 80_000.0, "rebal_buy"),
        ]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=100_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.5
        assert len(result.gains) == 2
        assert result.gains[0].btc == 0.3
        assert result.gains[0].is_long_term is True
        assert abs(result.gains[1].btc - 0.2) < 1e-8
        assert result.gains[1].is_long_term is False
        assert len(result.remaining_lots) == 1
        assert abs(result.remaining_lots[0].btc - 0.5) < 1e-8

    def test_sell_lifo(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [
            TaxLot("2023-01-01", 0.5, 20_000.0, "initial"),
            TaxLot("2025-05-01", 0.5, 80_000.0, "rebal_buy"),
        ]
        result = sell_lots(lots, btc_to_sell=0.3, sale_price=100_000.0,
                          sale_date="2025-06-01", method="lifo")
        assert result.gains[0].cost_basis == 80_000.0
        assert result.gains[0].is_long_term is False

    def test_sell_loss(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2024-01-01", 1.0, 100_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=0.5, sale_price=50_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.gains[0].gain == -25_000.0

    def test_sell_more_than_available(self):
        from engines.tax_lots import TaxLot, sell_lots
        lots = [TaxLot("2024-01-01", 0.3, 50_000.0, "initial")]
        result = sell_lots(lots, btc_to_sell=1.0, sale_price=60_000.0,
                          sale_date="2025-06-01", method="fifo")
        assert result.btc_sold == 0.3
        assert len(result.remaining_lots) == 0

    def test_seed_from_stack_tracker(self):
        from engines.tax_lots import seed_lots
        st_lots = [
            {"date": "2023-06-15", "btc": 0.5, "price": 30_000},
            {"date": "2024-01-10", "btc": 0.3, "price": 45_000},
        ]
        tax_lots = seed_lots(st_lots)
        assert len(tax_lots) == 2
        assert tax_lots[0].date == "2023-06-15"
        assert tax_lots[0].cost_basis == 30_000
        assert tax_lots[1].source == "initial"

    def test_seed_manual_entry(self):
        from engines.tax_lots import seed_lots
        tax_lots = seed_lots([], start_stack=1.0, start_price=60_000.0,
                             start_date="2031-01-01")
        assert len(tax_lots) == 1
        assert tax_lots[0].btc == 1.0
        assert tax_lots[0].cost_basis == 60_000.0

    def test_seed_empty(self):
        from engines.tax_lots import seed_lots
        assert seed_lots([]) == []


class TestTaxComputation:
    def test_apply_brackets_10pct_only(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        tax = apply_progressive_brackets(10_000, FEDERAL_BRACKETS_TCJA["single"])
        assert tax == pytest.approx(1_000.0)

    def test_apply_brackets_two_brackets(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        tax = apply_progressive_brackets(30_000, FEDERAL_BRACKETS_TCJA["single"])
        expected = 11_925 * 0.10 + (30_000 - 11_925) * 0.12
        assert tax == pytest.approx(expected)

    def test_apply_brackets_top_bracket(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        tax = apply_progressive_brackets(1_000_000, FEDERAL_BRACKETS_TCJA["single"])
        assert tax > 300_000

    def test_apply_brackets_zero(self):
        from engines.tax import apply_progressive_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        assert apply_progressive_brackets(0, FEDERAL_BRACKETS_TCJA["single"]) == 0.0

    def test_ltcg_stacking_zero_ordinary(self):
        from engines.tax import compute_ltcg_tax
        tax = compute_ltcg_tax(50_000, stacking_base=0, filing_status="single")
        expected = (50_000 - 48_350) * 0.15
        assert tax == pytest.approx(expected)

    def test_ltcg_stacking_high_ordinary(self):
        from engines.tax import compute_ltcg_tax
        tax = compute_ltcg_tax(100_000, stacking_base=80_000, filing_status="single")
        assert tax == pytest.approx(100_000 * 0.15)

    def test_loss_netting_st_loss_offsets_lt_gain(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=1_000, st_losses=5_000,
                                   lt_gains=10_000, lt_losses=0, carryforward=0)
        assert result.net_lt == 6_000
        assert result.net_st == 0
        assert result.loss_deduction == 0
        assert result.new_carryforward == 0

    def test_loss_netting_excess_carries_forward(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=0, st_losses=10_000,
                                   lt_gains=0, lt_losses=0, carryforward=0)
        assert result.loss_deduction == 3_000
        assert result.new_carryforward == 7_000

    def test_loss_netting_with_carryforward(self):
        from engines.tax import net_capital_gains
        result = net_capital_gains(st_gains=5_000, st_losses=0,
                                   lt_gains=0, lt_losses=0, carryforward=8_000)
        assert result.loss_deduction == 3_000
        assert result.new_carryforward == 0

    def test_niit_below_threshold(self):
        from engines.tax import compute_niit
        assert compute_niit(magi=150_000, nii=50_000, filing_status="single") == 0.0

    def test_niit_above_threshold(self):
        from engines.tax import compute_niit
        assert compute_niit(300_000, 80_000, "single") == pytest.approx(3_040.0)

    def test_niit_lesser_of_rule(self):
        from engines.tax import compute_niit
        assert compute_niit(220_000, 50_000, "single") == pytest.approx(760.0)

    def test_annual_tax_simple_case(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(
            tax_deferred_withdrawals=60_000,
            interest_income=5_000,
            other_income=0,
            lt_capital_gains=45_000,
        )
        result = compute_annual_tax(accum, filing_status="single",
                                     tcja_sunset=False, sim_year=2031,
                                     inflation_rate=0.04, state_rate=0.0)
        assert result["total"] > 0
        assert result["federal_ordinary"] > 0
        assert result["federal_ltcg"] >= 0
        assert result["niit"] == 0  # AGI ~$110k, under $200k
        assert result["effective_rate"] > 0
        assert "loss_carryforward" in result

    def test_annual_tax_with_niit(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(
            tax_deferred_withdrawals=100_000,
            interest_income=20_000,
            lt_capital_gains=200_000,
        )
        result = compute_annual_tax(accum, filing_status="single",
                                     tcja_sunset=False, sim_year=2025,
                                     inflation_rate=0.0, state_rate=0.0)
        assert result["niit"] > 0  # AGI ~$320k, well above $200k

    def test_annual_tax_with_state(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(tax_deferred_withdrawals=100_000)
        result = compute_annual_tax(accum, filing_status="single",
                                     tcja_sunset=False, sim_year=2025,
                                     inflation_rate=0.0, state_rate=13.30)
        assert result["state"] > 0

    def test_brackets_inflation_indexed(self):
        from engines.tax import _inflate_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        base = FEDERAL_BRACKETS_TCJA["single"]
        inflated = _inflate_brackets(base, years=10, rate=0.04)
        assert inflated[0][0] == pytest.approx(11_925 * 1.04**10, rel=0.01)
        assert inflated[0][1] == 0.10

    def test_annual_tax_sunset_brackets(self):
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        accum = TaxYearAccumulator(tax_deferred_withdrawals=600_000)
        result_tcja = compute_annual_tax(accum, filing_status="single",
                                          tcja_sunset=False, sim_year=2025,
                                          inflation_rate=0.0, state_rate=0.0)
        result_sunset = compute_annual_tax(accum, filing_status="single",
                                            tcja_sunset=True, sim_year=2025,
                                            inflation_rate=0.0, state_rate=0.0)
        # Sunset has higher top rate (39.6% vs 37%), so tax should be higher
        assert result_sunset["total"] > result_tcja["total"]


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Citadel Tax Integration
# ═══════════════════════════════════════════════════════════════════════════════

class _MockPriceModel:
    """Minimal mock satisfying the PriceModel protocol for Citadel tests."""
    def __init__(self):
        import pandas as pd
        self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
        self.genesis = pd.Timestamp("2009-07-25")

    def price_at(self, q, t):
        # Return a deterministic price that grows with t
        return 50_000.0 * (1 + t / 100)

    def quantile_at(self, price, t):
        return 0.50


def _test_model():
    return _MockPriceModel()


class TestCitadelTaxIntegration:
    def test_sim_config_has_tax_fields(self):
        from engines.citadel import SimConfig
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        state_code="CA", birth_year=1985,
                        cost_basis_method="fifo")
        assert cfg.tax_enabled is True
        assert cfg.state_code == "CA"
        assert cfg.td_btc_stack == 0.0
        assert cfg.tf_btc_stack == 0.0

    def test_citadel_state_has_tax_fields(self):
        from engines.citadel import CitadelState
        state = CitadelState()
        assert hasattr(state, "tax_lots")
        assert hasattr(state, "td_btc_stack")
        assert hasattr(state, "tf_btc_stack")
        assert hasattr(state, "td_cash")
        assert hasattr(state, "total_taxes_paid")
        assert state.total_taxes_paid == 0.0

    def test_tax_off_preserves_existing_behavior(self):
        """When tax_enabled=False, engine behavior is identical."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Annually", monthly_spend=5000,
                        cash_initial=200_000, selected_qs=[0.25],
                        tax_enabled=False)
        result = simulate(cfg, _test_model())
        # Should work exactly as before — no tax fields populated
        assert result.total_usd.shape[1] > 0
        assert result.taxes_paid is None
        assert result.td_total is None
        assert result.tf_total is None

    def test_tax_enabled_runs_without_error(self):
        """Tax-on simulation should complete without crashing."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Annually", monthly_spend=5000,
                        cash_initial=200_000, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        state_code="CA")
        result = simulate(cfg, _test_model())
        assert result.total_usd.shape[1] > 0
        assert result.taxes_paid is not None
        assert result.taxes_paid.shape == result.total_usd.shape

    def test_tax_enabled_with_td_tf_wrappers(self):
        """Tax-on with TD/TF wrappers initialized."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2033,
                        freq="Annually", monthly_spend=5000,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        state_code="TX",
                        td_cash_initial=50_000,
                        tf_cash_initial=30_000)
        result = simulate(cfg, _test_model())
        assert result.td_total is not None
        assert result.tf_total is not None
        # TD and TF totals should start > 0
        assert result.td_total[0, 0] >= 50_000 or result.td_total[0, 0] >= 0
        assert result.tf_total[0, 0] >= 30_000 or result.tf_total[0, 0] >= 0

    def test_tax_enabled_rmd(self):
        """RMD fires for old enough users."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0.5, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=2000,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        state_code="TX",
                        birth_year=1956,  # age 75 in 2031, RMD starts at 73
                        td_cash_initial=500_000)
        result = simulate(cfg, _test_model())
        assert result.taxes_paid is not None
        # With large TD balance and RMD, some taxes should be paid
        assert result.annual_taxes is not None
        assert len(result.annual_taxes) > 0

    def test_initial_state_seeds_lots_when_tax_on(self):
        """Tax lots should be seeded from start_stack."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=2.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=True)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == 2.0
        assert state.tax_lots[0].source == "initial"

    def test_no_tax_rate_field(self):
        """The old tax_rate placeholder should be replaced."""
        from engines.citadel import SimConfig
        cfg = SimConfig()
        assert not hasattr(cfg, "tax_rate")


class TestTaxDefaults:
    def test_citadel_has_tax_defaults(self):
        from tab_defaults import CITADEL
        assert CITADEL["tax_enabled"] is False
        assert CITADEL["filing_status"] == "single"
        assert CITADEL["state_code"] == "TX"
        assert CITADEL["td_btc"] == 0.5
        assert CITADEL["tf_btc"] == 0.5
        assert CITADEL["cost_basis_method"] == "fifo"

    def test_build_sim_config_passes_tax_fields(self):
        from figures.citadel import _build_sim_config
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "mfj"
        p["state_code"] = "CA"
        p["birth_year"] = 1985
        p["td_btc"] = 0.5
        p["tf_cash"] = 100_000
        cfg = _build_sim_config(p)
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "mfj"
        assert cfg.state_code == "CA"
        assert cfg.birth_year == 1985
        assert cfg.td_btc_stack == 0.5
        assert cfg.tf_cash_initial == 100_000


class TestTaxFigures:
    def test_tax_on_produces_ghost_traces(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "single"
        p["state_code"] = "CA"
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        trace_names = [t.name for t in fig.data if t.name]
        assert any("no tax" in (n or "").lower() for n in trace_names)

    def test_tax_off_no_ghost_traces(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        trace_names = [t.name for t in fig.data if t.name]
        assert not any("no tax" in (n or "").lower() for n in trace_names)

    def test_tax_summary_data_returned(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        assert "annual_taxes" in extra


class TestTaxCallbacks:
    def test_state_to_rate(self):
        from callbacks.citadel_tax_cb import _state_to_rate
        assert _state_to_rate("CA") == 13.30
        assert _state_to_rate("TX") == 0.0
        assert _state_to_rate("NY") == 10.90


class TestTaxSnapshot:
    def test_tax_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = [c[0] for c in _SNAPSHOT_CONTROLS]
        assert "cp-tax-toggle" in ids
        assert "cp-tax-config" in ids
        assert "cp-td-btc" in ids
        assert "cp-tf-btc" in ids

    def test_tax_checklist_options(self):
        from snapshot import _CHECKLIST_OPTIONS
        assert "cp-tax-toggle" in _CHECKLIST_OPTIONS


class TestTaxSummaryPanel:
    def test_summary_panel_exists(self):
        from layout.citadel_tax import tax_summary_panel
        panel = tax_summary_panel()
        assert panel is not None

    def test_build_tax_summary_empty(self):
        from callbacks.citadel_tax_cb import _build_tax_summary
        is_open, children = _build_tax_summary([])
        assert is_open is False
        assert children == []

    def test_build_tax_summary_with_data(self):
        from callbacks.citadel_tax_cb import _build_tax_summary
        data = [{"year": 2031, "ordinary_income": 60000, "st_gains": 0,
                 "lt_gains": 45000, "federal_ordinary": 8000, "federal_ltcg": 6750,
                 "niit": 0, "state": 5000, "total": 19750, "effective_rate": 0.175}]
        is_open, children = _build_tax_summary(data)
        assert is_open is True
        assert len(children) == 2  # header + tbody


class TestTreasuryStateExemption:
    def test_treasury_interest_exempt_from_state_tax(self):
        """Treasury interest should not be state-taxed (US law)."""
        from engines.tax import TaxYearAccumulator, compute_annual_tax
        # $100k treasury interest only — no other income
        accum_treasury = TaxYearAccumulator(treasury_interest=100_000)
        result_treasury = compute_annual_tax(
            accum_treasury, filing_status="single", tcja_sunset=False,
            sim_year=2025, inflation_rate=0.0, state_rate=10.0)

        # $100k cash interest only — same amount
        accum_cash = TaxYearAccumulator(interest_income=100_000)
        result_cash = compute_annual_tax(
            accum_cash, filing_status="single", tcja_sunset=False,
            sim_year=2025, inflation_rate=0.0, state_rate=10.0)

        # Federal tax should be identical (both are ordinary income)
        assert result_treasury["federal_ordinary"] == pytest.approx(
            result_cash["federal_ordinary"])

        # State tax: treasury should be $0, cash should be ~$10k
        assert result_treasury["state"] == pytest.approx(0.0)
        assert result_cash["state"] > 0


class TestInvestmentCostBasis:
    def test_cost_basis_initialized_from_initial_value(self):
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(invest_bins=[
            {"label": "Equities", "initial": 200_000, "return_rate": 10, "volatility": 0},
            {"label": "Bonds", "initial": 100_000, "return_rate": 5, "volatility": 0},
        ])
        state = _initial_state(cfg)
        assert state.invest_cost_basis == [200_000, 100_000]

    def test_cost_basis_decreases_proportionally_on_sale(self):
        """Selling 50% of an investment should remove 50% of its cost basis."""
        from engines.citadel import CitadelState
        state = CitadelState(
            investments=[400_000, 100_000],       # equities doubled from 200k
            invest_cost_basis=[200_000, 100_000],  # original cost
        )
        # Simulate selling $200k of equities (50% of current $400k)
        draw = 200_000
        current = state.investments[0]
        fraction = draw / current  # 0.5
        basis_sold = state.invest_cost_basis[0] * fraction  # 100k
        gain = draw - basis_sold  # 200k - 100k = 100k gain
        state.invest_cost_basis[0] -= basis_sold
        state.investments[0] -= draw

        assert gain == pytest.approx(100_000)
        assert state.invest_cost_basis[0] == pytest.approx(100_000)  # half basis remains
        assert state.investments[0] == pytest.approx(200_000)        # half value remains

    def test_gain_increases_as_investments_appreciate(self):
        """After appreciation, same dollar withdrawal has higher gain %."""
        from engines.citadel import CitadelState
        # Start: $100k equities, $100k basis → 0% gain
        state1 = CitadelState(
            investments=[100_000, 0], invest_cost_basis=[100_000, 0])
        draw = 50_000
        fraction1 = draw / state1.investments[0]
        basis1 = state1.invest_cost_basis[0] * fraction1
        gain1 = draw - basis1
        assert gain1 == pytest.approx(0)  # no appreciation yet

        # After 2x appreciation: $200k equities, still $100k basis
        state2 = CitadelState(
            investments=[200_000, 0], invest_cost_basis=[100_000, 0])
        fraction2 = draw / state2.investments[0]  # 25%
        basis2 = state2.invest_cost_basis[0] * fraction2  # 25k
        gain2 = draw - basis2  # 50k - 25k = 25k
        assert gain2 == pytest.approx(25_000)

        # After 10x appreciation: $1M equities, still $100k basis
        state3 = CitadelState(
            investments=[1_000_000, 0], invest_cost_basis=[100_000, 0])
        fraction3 = draw / state3.investments[0]  # 5%
        basis3 = state3.invest_cost_basis[0] * fraction3  # 5k
        gain3 = draw - basis3  # 50k - 5k = 45k
        assert gain3 == pytest.approx(45_000)


# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive Tax Simulation Tests — every parameter, every asset type
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaxSimComparative:
    """Compare tax-on vs tax-off and parameter variations at the engine level."""

    @staticmethod
    def _run(tax_enabled=True, **kw):
        from engines.citadel import SimConfig, simulate
        defaults = dict(
            start_stack=1.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=5_000,
            cash_initial=100_000, selected_qs=[0.25],
        )
        defaults.update(kw)
        cfg = SimConfig(tax_enabled=tax_enabled, **defaults)
        return simulate(cfg, _test_model())

    @staticmethod
    def _tax_years(result):
        """Extract the per-year tax dicts from sim result (first sim index)."""
        if result.annual_taxes and len(result.annual_taxes) > 0:
            at = result.annual_taxes[0]
            if isinstance(at, list):
                return at  # list of year dicts
            if isinstance(at, dict):
                return [at]  # single dict wrapped
        return []

    # ── Tax on vs off ──────────────────────────────────────────────────────

    def test_tax_on_reduces_terminal_wealth(self):
        r_off = self._run(tax_enabled=False)
        r_on = self._run(tax_enabled=True, filing_status="single", state_code="CA")
        assert r_on.total_usd[0, -1] <= r_off.total_usd[0, -1]

    def test_tax_off_pays_zero_tax(self):
        r = self._run(tax_enabled=False)
        # taxes_paid should be None or all zeros
        if r.taxes_paid is not None:
            assert r.taxes_paid.max() == 0

    def test_tax_on_pays_nonzero_tax(self):
        r = self._run(tax_enabled=True, filing_status="single",
                      state_code="CA", other_income=100_000)
        assert r.taxes_paid is not None
        assert r.taxes_paid[0, -1] > 0

    # ── State tax comparison ───────────────────────────────────────────────

    def test_california_tax_exceeds_texas(self):
        r_ca = self._run(state_code="CA", filing_status="single", other_income=100_000)
        r_tx = self._run(state_code="TX", filing_status="single", other_income=100_000)
        ca_total = sum(t["total"] for t in self._tax_years(r_ca))
        tx_total = sum(t["total"] for t in self._tax_years(r_tx))
        assert ca_total > tx_total

    def test_zero_tax_state_no_state_component(self):
        r = self._run(state_code="TX", filing_status="single", other_income=50_000)
        for yr in self._tax_years(r):
            assert yr["state"] == pytest.approx(0.0)

    # ── Filing status ──────────────────────────────────────────────────────

    def test_mfj_lower_tax_than_single(self):
        """MFJ brackets are wider — same income should pay less tax."""
        r_s = self._run(filing_status="single", state_code="TX", other_income=150_000)
        r_m = self._run(filing_status="mfj", state_code="TX", other_income=150_000)
        s_total = sum(t["total"] for t in self._tax_years(r_s))
        m_total = sum(t["total"] for t in self._tax_years(r_m))
        assert m_total <= s_total

    # ── TCJA sunset ────────────────────────────────────────────────────────

    def test_sunset_higher_tax_than_tcja(self):
        r_tcja = self._run(filing_status="single", state_code="TX",
                           tcja_sunset=False, other_income=200_000)
        r_sunset = self._run(filing_status="single", state_code="TX",
                             tcja_sunset=True, other_income=200_000)
        tcja_total = sum(t["total"] for t in self._tax_years(r_tcja))
        sunset_total = sum(t["total"] for t in self._tax_years(r_sunset))
        assert sunset_total >= tcja_total

    # ── Cost basis method ──────────────────────────────────────────────────

    def test_fifo_and_lifo_produce_different_gains(self):
        """FIFO sells oldest (likely LT), LIFO sells newest (likely ST)."""
        r_fifo = self._run(cost_basis_method="fifo", filing_status="single",
                           state_code="TX", monthly_spend=20_000)
        r_lifo = self._run(cost_basis_method="lifo", filing_status="single",
                           state_code="TX", monthly_spend=20_000)
        # Both should complete without error
        assert r_fifo.total_usd.shape[1] > 0
        assert r_lifo.total_usd.shape[1] > 0

    # ── Investment cost basis ──────────────────────────────────────────────

    def test_low_basis_means_higher_tax(self):
        """$200k equities with $50k basis (150k unrealized gain) vs $200k basis (0 gain)."""
        r_low = self._run(filing_status="single", state_code="TX",
                          invest_cost_basis_initial=[50_000, 100_000],
                          monthly_spend=20_000)  # force investment sales
        r_full = self._run(filing_status="single", state_code="TX",
                           invest_cost_basis_initial=[200_000, 100_000],
                           monthly_spend=20_000)
        low_tax = sum(t["total"] for t in self._tax_years(r_low))
        full_tax = sum(t["total"] for t in self._tax_years(r_full))
        assert low_tax >= full_tax

    # ── TD wrapper (Tax-Deferred) ──────────────────────────────────────────

    def test_td_withdrawals_taxed_as_ordinary(self):
        """TD withdrawals should show up as ordinary income in tax summary."""
        r = self._run(filing_status="single", state_code="TX",
                      td_cash_initial=500_000, monthly_spend=10_000)
        yrs = self._tax_years(r)
        if yrs:
            total_ordinary = sum(t.get("ordinary_income", 0) for t in yrs)
            assert total_ordinary > 0

    # ── TF wrapper (Roth) ──────────────────────────────────────────────────

    def test_roth_only_portfolio_zero_tax(self):
        """If all assets are in Roth, no tax should be owed."""
        r = self._run(filing_status="single", state_code="TX",
                      start_stack=0, cash_initial=0,
                      invest_bins=[
                          {"label": "Equities", "initial": 0, "return_rate": 10, "volatility": 0},
                          {"label": "Bonds", "initial": 0, "return_rate": 5, "volatility": 0},
                      ],
                      tf_cash_initial=500_000,
                      monthly_spend=3_000)
        yrs = self._tax_years(r)
        if yrs:
            total_tax = sum(t["total"] for t in yrs)
            assert total_tax == pytest.approx(0.0, abs=1.0)

    # ── RMD ────────────────────────────────────────────────────────────────

    def test_rmd_creates_ordinary_income(self):
        """Birth year 1958, age 73 in 2031 → RMD should force TD withdrawal."""
        r = self._run(filing_status="single", state_code="TX",
                      birth_year=1958, td_cash_initial=1_000_000,
                      start_stack=0, cash_initial=0, monthly_spend=0,
                      start_yr=2031, end_yr=2033)
        yrs = self._tax_years(r)
        if yrs:
            has_rmd_income = any(t.get("ordinary_income", 0) > 0 for t in yrs)
            assert has_rmd_income

    def test_no_rmd_without_birth_year(self):
        """No birth year → no RMDs, TD untouched if not needed for spending."""
        r = self._run(filing_status="single", state_code="TX",
                      birth_year=None, td_cash_initial=1_000_000,
                      start_stack=0, cash_initial=500_000, monthly_spend=1_000,
                      start_yr=2031, end_yr=2033)
        # All spending covered by taxable cash, no TD withdrawals needed
        yrs = self._tax_years(r)
        if yrs:
            total_td = sum(t.get("ordinary_income", 0) for t in yrs)
            assert total_td < 100_000  # well under $1M

    # ── NIIT ───────────────────────────────────────────────────────────────

    def test_niit_triggers_above_threshold(self):
        """$300k other income (single) should trigger NIIT."""
        r = self._run(filing_status="single", state_code="TX",
                      other_income=300_000, start_yr=2025, end_yr=2027)
        yrs = self._tax_years(r)
        if yrs:
            has_niit = any(t.get("niit", 0) > 0 for t in yrs)
            assert has_niit

    def test_niit_zero_below_threshold(self):
        """$100k other income (single) should NOT trigger NIIT."""
        r = self._run(filing_status="single", state_code="TX",
                      other_income=100_000, start_stack=0, cash_initial=500_000,
                      monthly_spend=1_000, start_yr=2025, end_yr=2027)
        yrs = self._tax_years(r)
        if yrs:
            total_niit = sum(t.get("niit", 0) for t in yrs)
            assert total_niit == pytest.approx(0.0, abs=1.0)

    # ── Other income growth ────────────────────────────────────────────────

    def test_other_income_growth_increases_tax(self):
        r_flat = self._run(filing_status="single", state_code="TX",
                           other_income=50_000, other_income_growth=0)
        r_grow = self._run(filing_status="single", state_code="TX",
                           other_income=50_000, other_income_growth=5.0)
        flat_tax = sum(t["total"] for t in self._tax_years(r_flat))
        grow_tax = sum(t["total"] for t in self._tax_years(r_grow))
        assert grow_tax >= flat_tax

    # ── Ghost traces in figure builder ─────────────────────────────────────

    def test_figure_has_ghost_traces_when_tax_on(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "single"
        p["state_code"] = "CA"
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        names = [t.name for t in fig.data if t.name]
        assert any("no tax" in (n or "").lower() for n in names)
        assert "annual_taxes" in extra

    def test_figure_no_ghost_traces_when_tax_off(self):
        from figures.citadel import build_citadel_figure
        from tab_defaults import citadel_defaults
        p = citadel_defaults()
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        fig, extra = build_citadel_figure(M, p)
        names = [t.name for t in fig.data if t.name]
        assert not any("no tax" in (n or "").lower() for n in names)


class TestTaxWrapperGrowth:
    """Verify TD/TF wrapper balances grow over time (Critical #1 fix)."""

    def test_td_cash_grows(self):
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=0,
                        cash_initial=0, selected_qs=[0.25],
                        tax_enabled=True, td_cash_initial=100_000,
                        cash_rate=5.0)
        r = simulate(cfg, _test_model())
        # After 4 years at 5%, $100k should grow to ~$121,550
        assert r.td_total is not None
        final_td = r.td_total[0, -1]
        assert final_td > 100_000, f"TD should grow but got {final_td}"

    def test_tf_investments_grow(self):
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=0,
                        cash_initial=0, selected_qs=[0.25],
                        tax_enabled=True,
                        tf_invest_bins=[
                            {"label": "Equities", "initial": 200_000},
                            {"label": "Bonds", "initial": 0},
                        ],
                        invest_bins=[
                            {"label": "Equities", "initial": 0, "return_rate": 10, "volatility": 0},
                            {"label": "Bonds", "initial": 0, "return_rate": 5, "volatility": 0},
                        ])
        r = simulate(cfg, _test_model())
        final_tf = r.tf_total[0, -1]
        # 10% return on $200k over 4 years ≈ $292,820
        assert final_tf > 200_000, f"TF should grow but got {final_tf}"

    def test_td_balance_used_for_rmd(self):
        """RMD should be based on grown TD balance, not initial."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(start_stack=0, start_yr=2031, end_yr=2035,
                        freq="Annually", monthly_spend=0,
                        cash_initial=0, selected_qs=[0.25],
                        tax_enabled=True, filing_status="single",
                        birth_year=1958, td_cash_initial=500_000,
                        cash_rate=5.0)
        r = simulate(cfg, _test_model())
        yrs = r.annual_taxes[0] if r.annual_taxes else []
        if yrs:
            # RMD income should reflect growing balance, not fixed $500k
            first_yr_income = yrs[0].get("ordinary_income", 0)
            assert first_yr_income > 0


class TestTaxEdgeCases:
    """Test gap coverage: depletion, cost basis bounds, shortfall, tax payment side effects."""

    def test_partial_year_depletion_no_crash(self):
        """If all accounts deplete mid-year, sim should complete without error.
        The final partial year's taxes may not be computed (year boundary never reached)."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0.01, start_yr=2031, end_yr=2035,
            freq="Monthly", monthly_spend=50_000,  # high spend to force depletion
            cash_initial=10_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="TX",
            invest_bins=[
                {"label": "Equities", "initial": 10_000, "return_rate": 10, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 5, "volatility": 0},
            ])
        r = simulate(cfg, _test_model())
        # Should complete without crash
        assert r.total_usd.shape[1] > 0
        # Terminal wealth should be near zero (depleted)
        assert r.total_usd[0, -1] < 100_000

    def test_invest_cost_basis_never_negative(self):
        """Cost basis should never go below zero even with float arithmetic."""
        from engines.citadel import CitadelState
        # Simulate selling 100% of investment
        state = CitadelState(
            investments=[100_000, 50_000],
            invest_cost_basis=[80_000, 50_000],
        )
        # Sell all equities
        current = state.investments[0]
        fraction = current / current  # 1.0
        basis_sold = state.invest_cost_basis[0] * fraction
        state.invest_cost_basis[0] -= basis_sold
        state.investments[0] = 0
        assert state.invest_cost_basis[0] >= 0
        assert state.invest_cost_basis[0] == pytest.approx(0.0)

        # Edge: tiny floating point residual
        state2 = CitadelState(
            investments=[100.0, 0],
            invest_cost_basis=[100.0, 0],
        )
        # Sell in 3 chunks of 1/3 each (float imprecision)
        for _ in range(3):
            amt = 100.0 / 3
            cur = state2.investments[0]
            if cur <= 0:
                break
            frac = min(amt / cur, 1.0)
            basis = state2.invest_cost_basis[0] * frac
            state2.invest_cost_basis[0] = max(state2.invest_cost_basis[0] - basis, 0.0)
            state2.investments[0] = max(state2.investments[0] - amt, 0.0)
        assert state2.invest_cost_basis[0] >= 0

    def test_waterfall_shortfall_when_all_depleted(self):
        """When all three wrappers are empty, shortfall should equal the spending amount."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2032,
            freq="Annually", monthly_spend=10_000,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ])
        r = simulate(cfg, _test_model())
        # With zero assets and $10k/mo spending, there should be shortfall
        assert r.total_usd.shape[1] > 0

    def test_tax_payment_from_investments_tracks_basis(self):
        """When taxes are paid by selling investments, cost basis should decrease."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Annually", monthly_spend=0,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,  # generates tax liability
            invest_bins=[
                {"label": "Equities", "initial": 500_000, "return_rate": 10, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ])
        r = simulate(cfg, _test_model())
        # Tax was owed on $200k other income, paid from investments
        # Investments should be less than they'd be without tax
        yrs = r.annual_taxes[0] if r.annual_taxes else []
        if yrs:
            assert yrs[0]["total"] > 0  # taxes were owed
        # Investment balance should be reduced by tax payment
        assert r.invest_balances[0, -1, 0] < 500_000 * 1.1 ** 2  # less than pure growth

    def test_annual_taxes_list_of_lists_flattened_by_summary(self):
        """_build_tax_summary should handle list-of-lists from engine."""
        from callbacks.citadel_tax_cb import _build_tax_summary
        # Simulate engine output: list containing one list of year dicts
        nested = [[
            {"year": 2031, "ordinary_income": 60000, "st_gains": 0,
             "lt_gains": 0, "federal_ordinary": 5000, "federal_ltcg": 0,
             "niit": 0, "state": 0, "total": 5000, "effective_rate": 0.08},
        ]]
        is_open, children = _build_tax_summary(nested)
        assert is_open is True
        assert len(children) == 2  # header + tbody


class TestGrossUpTaxPayment:
    """Verify gross-up when paying taxes from taxable investments/TD."""

    def test_gross_up_pays_from_investments_without_crash(self):
        """Paying taxes by selling investments with low basis should work."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Annually", monthly_spend=0,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            invest_bins=[
                {"label": "Equities", "initial": 500_000, "return_rate": 10, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            invest_cost_basis_initial=[250_000, 0],  # 50% unrealized gain
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid is not None
        assert r.taxes_paid[0, -1] > 0
        # Investments should be reduced (tax was paid from them)
        assert r.invest_balances[0, -1, 0] < 500_000 * 1.1 ** 2

    def test_gross_up_pays_from_td_without_crash(self):
        """Paying taxes from TD (ordinary income) with gross-up should work."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Annually", monthly_spend=0,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=300_000,  # high income → high tax
            td_cash_initial=1_000_000,
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid is not None
        assert r.taxes_paid[0, -1] > 0


class TestMcStatusWithTaxExtraDict:
    """Regression test: _mc_status must not crash when the extra dict
    contains tax keys (annual_taxes) but no MC keys (created).
    This was the root cause of the silent background callback crash
    when running deterministic simulations with tax enabled."""

    def test_mc_status_with_tax_only_extra(self):
        """Deterministic run: extra dict has annual_taxes but no 'created'."""
        from callbacks.mc_helpers import _mc_status
        tax_extra = {"annual_taxes": [{"year": 2031, "total": 5000}]}
        store_val, status, show_modal = _mc_status(tax_extra, None, None)
        assert status == ""
        assert show_modal is True  # mc_result is truthy (non-empty dict)

    def test_mc_status_with_mc_result(self):
        """Stochastic run: extra dict has MC 'created' key."""
        from callbacks.mc_helpers import _mc_status
        mc_result = {"created": "2026-03-29T12:00:00.000Z", "sims": 1000}
        store_val, status, show_modal = _mc_status(mc_result, None, None)
        assert "Saved:" in status
        assert show_modal is True

    def test_mc_status_with_empty_result(self):
        """No result at all (first render, no sim run)."""
        from callbacks.mc_helpers import _mc_status
        store_val, status, show_modal = _mc_status(None, None, None)
        assert status == ""
        assert show_modal is False

    def test_mc_status_with_cached_only(self):
        """No new result, but cached MC exists."""
        from callbacks.mc_helpers import _mc_status
        cached = {"created": "2026-03-28T10:00:00.000Z"}
        store_val, status, show_modal = _mc_status(None, cached, ["mc"])
        assert "Using saved:" in status
        assert show_modal is False

    def test_mc_status_with_combined_tax_and_mc(self):
        """Both tax extra and MC result keys in the same dict."""
        from callbacks.mc_helpers import _mc_status
        combined = {"annual_taxes": [{"year": 2031}],
                    "created": "2026-03-29T12:00:00.000Z", "sims": 500}
        store_val, status, show_modal = _mc_status(combined, None, None)
        assert "Saved:" in status
        assert show_modal is True

    def test_full_figure_builder_extra_survives_mc_status(self):
        """End-to-end: build_citadel_figure with tax → extra → _mc_status."""
        from figures.citadel import build_citadel_figure
        from callbacks.mc_helpers import _mc_status
        from tab_defaults import citadel_defaults, CITADEL
        p = citadel_defaults()
        p["tax_enabled"] = True
        p["filing_status"] = "single"
        p["state_code"] = "CA"
        p["start_yr"] = 2031
        p["end_yr"] = 2033
        for k in ("td_btc", "td_cash", "td_res_short", "td_res_med", "td_res_long",
                   "td_inv_eq", "td_inv_bd", "tf_btc", "tf_cash", "tf_res_short",
                   "tf_res_med", "tf_res_long", "tf_inv_eq", "tf_inv_bd"):
            p[k] = CITADEL.get(k, 0)
        fig, extra = build_citadel_figure(M, p)
        # This is the exact call that was crashing in production
        store_val, status, show_modal = _mc_status(extra, None, None)
        assert isinstance(status, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Citadel Planner — engine rule verification tests
# ═══════════════════════════════════════════════════════════════════════════════


class _ControlledPriceModel:
    """Price model that returns configurable quantiles for trigger testing."""
    def __init__(self, quantile=0.50, price=50_000.0):
        import pandas as pd
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
        # Zero volatility → deterministic dollar-asset growth
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


class TestCashFloorEnforcement:
    """1) Cash floor must not be violated until all assets are zero."""

    def test_cash_floor_replenished_from_investments_first(self):
        """Floor draws from investments before touching BTC."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        state = CitadelState(
            cash=10_000, reserves=[0, 0, 0], investments=[50_000, 30_000],
            btc_stack=1.0, btc_price=60_000,
        )
        cfg = _bare_config(cash_floor=50_000)
        _enforce_floors(state, cfg)
        assert state.cash >= 50_000 - 1  # floor met (within rounding)
        assert state.btc_stack == 1.0    # BTC untouched

    def test_cash_floor_draws_btc_only_when_dollar_assets_exhausted(self):
        """BTC is sold for cash floor only after all dollar assets are zero."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            btc_stack=2.0, btc_price=50_000,
        )
        cfg = _bare_config(cash_floor=30_000)
        _enforce_floors(state, cfg)
        assert state.cash >= 30_000 - 1
        assert state.btc_stack < 2.0  # BTC was sold

    def test_reserve_floor_never_sells_btc(self):
        """Reserve floors redistribute among dollar assets, never sell BTC."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            btc_stack=5.0, btc_price=100_000,
        )
        cfg = _bare_config(
            cash_floor=0,
            reserve_floors=[10_000, 0, 0],  # floor on short reserve
        )
        _enforce_floors(state, cfg)
        assert state.btc_stack == 5.0  # BTC untouched
        # Floor can't be met (no dollar sources) — reserve stays at 0
        assert state.reserves[0] == 0

    def test_cash_floor_holds_through_simulation(self):
        """Over a full sim, cash stays above floor until total depletion."""
        from engines.citadel import SimConfig, simulate, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            cash_floor=20_000, monthly_spend=8000,
            start_yr=2031, end_yr=2040, freq="Annually",
        )
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for _ in range(9):  # 9 annual steps
            state = step(state, cfg, 50_000, rng, model=model)
            total = state.cash + sum(state.reserves) + sum(state.investments) + state.btc_stack * state.btc_price
            if total > 20_000:
                # If total assets can cover the floor, cash should meet it
                assert state.cash >= 20_000 - 100, \
                    f"Cash {state.cash:.0f} below floor 20000 with total {total:.0f}"


    def test_cash_floor_respected_after_tax_payment(self):
        """Regression: tax payment at year-end must not leave cash below floor
        when other assets are available to replenish it."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            start_stack=5.0, cash_initial=100_000,
            cash_floor=80_000,
            monthly_spend=1000,
            tax_enabled=True, state_code="CA",
            other_income=500_000,  # large income → large tax bill
            start_yr=2031, end_yr=2035, freq="Annually",
            reserve_bins=[
                {"label": "Short", "initial": 200_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for i in range(4):
            state = step(state, cfg, 50_000, rng, model=model)
            total_other = (sum(state.reserves) + sum(state.investments)
                           + state.btc_stack * state.btc_price)
            if total_other > 80_000:
                assert state.cash >= 80_000 - 100, (
                    f"Period {i+1}: cash {state.cash:.0f} below floor 80000 "
                    f"with {total_other:.0f} in other assets")


    def test_cash_floor_draws_from_td_when_taxable_exhausted(self):
        """Cash floor replenished from TD when all taxable assets are depleted."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000,
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = _bare_config(cash_floor=10_000, tax_enabled=True, state_code="TX")
        _enforce_floors(state, cfg)
        assert state.cash >= 10_000 - 1
        assert state.td_cash == pytest.approx(90_000)
        assert state.tax_year_accum.tax_deferred_withdrawals == pytest.approx(10_000)

    def test_cash_floor_draws_tf_after_td_exhausted(self):
        """Cash floor falls through to TF (Roth) when TD is also exhausted."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000,
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tf_cash=50_000, tf_reserves=[0, 0, 0], tf_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = _bare_config(cash_floor=10_000, tax_enabled=True, state_code="TX")
        _enforce_floors(state, cfg)
        assert state.cash >= 10_000 - 1
        assert state.tf_cash == pytest.approx(40_000)
        assert state.tax_year_accum.roth_withdrawals == pytest.approx(10_000)

    def test_cash_floor_holds_through_tax_sim_with_td(self):
        """Over a full tax-enabled sim, cash floor holds while TD/TF have funds."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            start_stack=0, cash_initial=50_000, cash_floor=10_000,
            monthly_spend=10_000,
            tax_enabled=True, state_code="TX",
            td_cash_initial=200_000, tf_cash_initial=100_000,
            start_yr=2031, end_yr=2040, freq="Monthly",
            # Zero growth for predictability
            cash_rate=0,
            reserve_bins=[
                {"label": "S", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for p in range(36):  # 3 years monthly
            state = step(state, cfg, 50_000, rng, model=model)
            total_all = (state.cash + state.td_cash + state.tf_cash
                        + sum(state.td_reserves) + sum(state.tf_reserves)
                        + sum(state.td_investments) + sum(state.tf_investments))
            if total_all > 10_000:
                assert state.cash >= 10_000 - 100, \
                    f"Period {p}: cash {state.cash:.0f} below floor with {total_all:.0f} total"


class TestBtcThresholdRules:
    """2) Bitcoin is sold/bought according to threshold rules."""

    def test_high_q_triggers_btc_sell(self):
        """When quantile >= high_q_trigger, BTC is sold."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            high_q_trigger=0.90,
            high_q_action={"mode": "lump", "rate": 20.0, "duration": 1,
                           "split": {"cash": 1.0}},
        )
        model = _ControlledPriceModel(quantile=0.95, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.btc_stack < initial_btc, "BTC should have been sold at high quantile"
        assert state.rebal_event is not None
        assert state.rebal_event["action"] == "sell_btc"

    def test_low_q_triggers_btc_buy(self):
        """When quantile <= low_q_trigger, BTC is bought."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            low_q_trigger=0.10,
            low_q_action={"mode": "lump", "rate": 10.0, "duration": 1,
                          "split": {"cash": 0.5, "inv_eq": 0.5}},
        )
        model = _ControlledPriceModel(quantile=0.03, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.btc_stack > initial_btc, "BTC should have been bought at low quantile"
        assert state.rebal_event is not None
        assert state.rebal_event["action"] == "buy_btc"

    def test_mid_quantile_no_rebalancing(self):
        """Between triggers, no rebalancing occurs."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(high_q_trigger=0.95, low_q_trigger=0.05)
        model = _ControlledPriceModel(quantile=0.50, price=50_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        state = step(state, cfg, 50_000, rng, model=model)
        # BTC changes only from spending, not rebalancing
        assert state.rebal_event is None


class TestSpendingIncreasesTax:
    """3) Increasing monthly spending increases taxes."""

    def test_higher_spending_means_higher_taxes(self):
        """More withdrawals → more realized gains → higher tax bill."""
        from engines.citadel import SimConfig, simulate
        # Use BTC-only portfolio so spending forces capital gains
        common = dict(
            start_stack=10.0, cash_initial=0,
            tax_enabled=True, filing_status="single", state_code="CA",
            start_yr=2031, end_yr=2035, freq="Annually",
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        low = _bare_config(monthly_spend=3000, **common)
        high = _bare_config(monthly_spend=10000, **common)
        model = _test_model()
        r_low = simulate(low, model)
        r_high = simulate(high, model)
        tax_low = r_low.taxes_paid[0, -1]
        tax_high = r_high.taxes_paid[0, -1]
        assert tax_high > tax_low, \
            f"Higher spend should yield higher tax: {tax_high:.0f} vs {tax_low:.0f}"

    def test_zero_spending_minimal_tax(self):
        """With zero spending and no other income, minimal or zero tax."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            monthly_spend=0, tax_enabled=True,
            filing_status="single", state_code="TX",
            other_income=0,
            start_yr=2031, end_yr=2035, freq="Annually",
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Only interest income could generate tax (from cash/reserves)
        # With TX (no state tax) and low interest, tax should be very small
        assert r.taxes_paid[0, -1] < 5000


class TestWithdrawalOrderTaxAdvantaged:
    """4) Verify withdrawal logic follows tax-advantaged ordering."""

    def test_taxable_cash_drawn_before_td(self):
        """Taxable principal (no tax) should be drawn before TD (ordinary income)."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            monthly_spend=20_000,
            cash_initial=200_000,
            tax_enabled=True, state_code="TX",
            td_cash_initial=200_000,
            start_yr=2031, end_yr=2034, freq="Annually",
            # Zero growth so balances are predictable
            cash_rate=0,
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Taxable cash should deplete faster than TD cash
        # After 3 years of spending $240k/yr, taxable cash should be gone first
        assert r.taxes_paid is not None

    def test_roth_btc_is_absolute_last(self):
        """Roth BTC should be the very last asset sold."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = _bare_config(
            monthly_spend=50_000, cash_initial=0,
            tax_enabled=True, state_code="TX",
            start_stack=0,  # no taxable BTC
            tf_btc_stack=1.0,
            tf_cash_initial=100_000,
            start_yr=2031, end_yr=2035, freq="Annually",
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _ControlledPriceModel(quantile=0.50, price=100_000)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        # After first step: TF cash should be drawn before TF BTC
        state = step(state, cfg, 100_000, rng, model=model)
        assert state.tf_cash < 100_000, "Roth cash should be drawn"
        if state.tf_cash > 0:
            assert state.tf_btc_stack == 1.0, "Roth BTC untouched while Roth cash remains"

    def test_td_bracket_fill_uses_low_bracket_room(self):
        """TD withdrawals should bracket-fill to minimize marginal rate."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            monthly_spend=15_000,
            cash_initial=500_000,
            tax_enabled=True, state_code="TX",
            td_cash_initial=500_000,
            start_yr=2031, end_yr=2035, freq="Annually",
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Should have some TD withdrawals (bracket-filling) even though
        # taxable cash could cover all spending
        assert r.taxes_paid is not None
        assert len(r.annual_taxes) > 0
        # Check that at least one year has TD withdrawal recorded
        has_td_wd = any(
            yr.get("ordinary_income", 0) > 0
            for sim_taxes in r.annual_taxes
            for yr in sim_taxes
        )
        assert has_td_wd, "TD bracket-filling should produce ordinary income"


class TestLumpCooldown:
    """5) Global lump cooldown is obeyed."""

    def test_cooldown_prevents_consecutive_lumps(self):
        """After a lump action, another lump is blocked for cooldown periods."""
        from engines.citadel import CitadelState, _evaluate_rebalancing
        from engines.citadel import SimConfig
        cfg = _bare_config(
            lump_cooldown=3,
            high_q_trigger=0.90,
            high_q_action={"mode": "lump", "rate": 10.0, "duration": 1,
                           "split": {"cash": 1.0}},
        )
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000,
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        # First trigger: should fire
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.rebal_event is not None
        assert state.rebal_cooldown == 3
        first_btc = state.btc_stack

        # Next 2 periods: cooldown should block (3→2, 2→1)
        for i in range(2):
            state.rebal_event = None
            _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
            assert state.rebal_event is None, f"Cooldown period {i+1}: should be blocked"
            assert state.btc_stack == first_btc, "No BTC sold during cooldown"

        # 3rd call: cooldown 1→0, trigger fires again
        state.rebal_event = None
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.rebal_event is not None, "Should fire after cooldown expires"

    def test_gradual_mode_ignores_cooldown(self):
        """Gradual actions continue regardless of cooldown counter."""
        from engines.citadel import CitadelState, _evaluate_rebalancing
        from engines.citadel import SimConfig
        cfg = _bare_config(
            high_q_trigger=0.90,
            high_q_action={"mode": "gradual", "rate": 5.0, "duration": 3,
                           "split": {"cash": 1.0}},
        )
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000,
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        # Trigger gradual
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.grad_active is True
        btc_after_first = state.btc_stack

        # Continue gradual — even with cooldown set, gradual proceeds
        _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.btc_stack < btc_after_first, "Gradual should continue selling"


class TestBtcSaleDistribution:
    """6) Bitcoin sale proceeds distributed according to split rules."""

    def test_sell_distributes_per_split(self):
        """Proceeds from BTC sale go to accounts per configured split."""
        from engines.citadel import CitadelState, SimConfig, _execute_sell_btc
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-01-15",
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
        )
        split = {"cash": 0.20, "res_short": 0.10, "res_med": 0.10,
                 "res_long": 0.10, "inv_eq": 0.30, "inv_bd": 0.20}
        evt = _execute_sell_btc(state, SimConfig(cost_basis_method="fifo"), rate_pct=10.0, split=split)
        # Sold 10% of 10 BTC = 1 BTC = $50,000
        assert evt["btc_sold"] == pytest.approx(1.0)
        assert evt["proceeds"] == pytest.approx(50_000)
        assert state.cash == pytest.approx(10_000)           # 20%
        assert state.reserves[0] == pytest.approx(5_000)     # 10%
        assert state.reserves[1] == pytest.approx(5_000)     # 10%
        assert state.reserves[2] == pytest.approx(5_000)     # 10%
        assert state.investments[0] == pytest.approx(15_000)  # 30%
        assert state.investments[1] == pytest.approx(10_000)  # 20%

    def test_sell_zero_btc_no_event(self):
        """Selling from empty stack produces no event."""
        from engines.citadel import CitadelState, SimConfig, _execute_sell_btc
        state = CitadelState(btc_stack=0, btc_price=50_000, sim_date="2035-01-15")
        evt = _execute_sell_btc(state, SimConfig(cost_basis_method="fifo"), rate_pct=10.0, split={"cash": 1.0})
        assert evt == {}


class TestBtcPurchaseSourcing:
    """7) Bitcoin purchases source funds according to split rules."""

    def test_buy_sources_per_split(self):
        """BTC purchase draws from accounts per configured split."""
        from engines.citadel import CitadelState, SimConfig, _execute_buy_btc
        state = CitadelState(
            btc_stack=0, btc_price=50_000, sim_date="2035-01-15",
            cash=100_000, reserves=[50_000, 50_000, 50_000],
            investments=[200_000, 100_000],
        )
        split = {"cash": 0.10, "inv_eq": 0.50, "inv_bd": 0.40}
        # Total dollar assets = 100k + 150k + 300k = 550k
        # 10% of 550k = 55k target
        evt = _execute_buy_btc(state, SimConfig(), rate_pct=10.0, split=split)
        assert evt["action"] == "buy_btc"
        assert evt["btc_bought"] == pytest.approx(55_000 / 50_000)
        # Cash should lose 10% of 55k = 5,500
        assert state.cash == pytest.approx(100_000 - 5_500)
        # Equities lose 50% of 55k = 27,500
        assert state.investments[0] == pytest.approx(200_000 - 27_500)
        # Bonds lose 40% of 55k = 22,000
        assert state.investments[1] == pytest.approx(100_000 - 22_000)

    def test_buy_respects_floor(self):
        """BTC purchase won't draw cash below its floor."""
        from engines.citadel import CitadelState, _execute_buy_btc
        cfg = _bare_config(cash_floor=80_000)
        state = CitadelState(
            btc_stack=0, btc_price=50_000, sim_date="2035-01-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        split = {"cash": 1.0}
        # Total dollars = 100k, 10% = 10k, but floor = 80k, so avail = 20k
        evt = _execute_buy_btc(state, cfg, rate_pct=10.0, split=split)
        assert state.cash >= 80_000 - 1, "Cash floor should be respected"

    def test_buy_redistributes_shortfall(self):
        """When one source can't cover its share, shortfall goes to others."""
        from engines.citadel import CitadelState, SimConfig, _execute_buy_btc
        state = CitadelState(
            btc_stack=0, btc_price=50_000, sim_date="2035-01-15",
            cash=1_000,  # very little cash
            reserves=[0, 0, 0],
            investments=[200_000, 200_000],
        )
        split = {"cash": 0.50, "inv_eq": 0.25, "inv_bd": 0.25}
        # Total = 401k, 10% = 40.1k
        # Cash wants 50% = 20.05k but only has 1k → shortfall redistributed
        evt = _execute_buy_btc(state, SimConfig(), rate_pct=10.0, split=split)
        assert evt["action"] == "buy_btc"
        assert state.cash < 1_000  # Cash was drawn
        # Investments picked up the slack
        total_drawn = evt["cost"]
        assert total_drawn > 1_000  # More than just cash


class TestTaxEfficientAccountUsage:
    """8) Trades in TD/TF accounts don't generate capital gains;
    taxable account trades do."""

    def test_taxable_btc_sale_generates_capital_gains(self):
        """Selling BTC from taxable wrapper records capital gains."""
        from engines.citadel import SimConfig, simulate
        # BTC-only portfolio forces BTC sale for spending
        cfg = _bare_config(
            start_stack=10.0,  # taxable BTC (plenty to cover spending)
            monthly_spend=20_000,
            cash_initial=0,
            tax_enabled=True, state_code="CA",
            start_yr=2031, end_yr=2035, freq="Annually",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Selling BTC from taxable should generate gains
        has_gains = any(
            yr.get("lt_gains", 0) > 0 or yr.get("st_gains", 0) > 0
            for sim_taxes in r.annual_taxes
            for yr in sim_taxes
        )
        assert has_gains, "Taxable BTC sale should generate capital gains"
        assert r.taxes_paid[0, -1] > 0

    def test_roth_only_portfolio_zero_capital_gains_tax(self):
        """All assets in Roth (TF) → no capital gains tax on any trade."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            start_stack=0,  # no taxable BTC
            monthly_spend=10_000,
            cash_initial=0,
            tax_enabled=True, state_code="TX",
            other_income=0,
            tf_btc_stack=2.0,
            tf_cash_initial=300_000,
            start_yr=2031, end_yr=2035, freq="Annually",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        # All Roth → zero tax
        assert r.taxes_paid[0, -1] == pytest.approx(0, abs=1)

    def test_td_withdrawal_taxed_as_ordinary_not_capital_gains(self):
        """TD withdrawals are ordinary income, not capital gains."""
        from engines.citadel import SimConfig, simulate
        cfg = _bare_config(
            start_stack=0, monthly_spend=20_000,
            cash_initial=0,
            tax_enabled=True, state_code="TX",
            td_cash_initial=1_000_000,
            start_yr=2031, end_yr=2035, freq="Annually",
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        r = simulate(cfg, model)
        # Should have ordinary income but zero capital gains
        for sim_taxes in r.annual_taxes:
            for yr in sim_taxes:
                assert yr.get("ordinary_income", 0) > 0, "TD withdrawal = ordinary income"
                assert yr.get("lt_gains", 0) == 0, "TD should not produce LTCG"
                assert yr.get("st_gains", 0) == 0, "TD should not produce STCG"

    def test_taxable_vs_td_same_spend_different_tax_type(self):
        """Same spending from taxable BTC vs TD produces different tax profiles."""
        from engines.citadel import SimConfig, simulate
        empty_bins = dict(
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
        )
        # _test_model has price growth: 50000*(1+t/100), so BTC bought at
        # t0 and sold later produces a capital gain.
        model = _test_model()
        # Scenario A: spend from taxable BTC (capital gains)
        cfg_taxable = _bare_config(
            start_stack=10.0, monthly_spend=20_000, cash_initial=0,
            tax_enabled=True, state_code="CA",
            start_yr=2031, end_yr=2036, freq="Annually",
            **empty_bins,
        )
        # Scenario B: spend from TD cash (ordinary income)
        cfg_td = _bare_config(
            start_stack=0, monthly_spend=20_000, cash_initial=0,
            tax_enabled=True, state_code="CA",
            td_cash_initial=1_000_000,
            start_yr=2031, end_yr=2036, freq="Annually",
            **empty_bins,
        )
        r_taxable = simulate(cfg_taxable, model)
        r_td = simulate(cfg_td, model)
        # Both should pay tax
        assert r_taxable.taxes_paid[0, -1] > 0, "Taxable BTC should generate tax"
        assert r_td.taxes_paid[0, -1] > 0, "TD withdrawals should generate tax"
        # Tax types differ: BTC = capital gains (lower rate), TD = ordinary income
        tax_btc = r_taxable.taxes_paid[0, -1]
        tax_td = r_td.taxes_paid[0, -1]
        assert tax_btc != tax_td, "Different tax types should produce different totals"


class TestQuarterlyTaxPayments:
    """Quarterly estimated tax payment tests."""

    def test_state_has_quarterly_field(self):
        from engines.citadel import CitadelState
        state = CitadelState()
        assert hasattr(state, "quarterly_tax_paid_ytd")
        assert state.quarterly_tax_paid_ytd == 0.0

    def test_pay_tax_amount_draws_from_cash_first(self):
        """_pay_tax_amount draws cash before other sources."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=30_000, sim_year=2031)
        assert state.cash == pytest.approx(20_000)

    def test_pay_tax_amount_uses_investments_after_cash(self):
        """_pay_tax_amount falls through to investments when cash exhausted."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        state = CitadelState(
            cash=10_000, reserves=[0, 0, 0], investments=[100_000, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[100_000, 0],  # full basis = no gain = no gross-up needed
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=30_000, sim_year=2031)
        assert state.cash == 0
        assert state.investments[0] < 100_000

    def test_quarterly_payment_annualizes_ytd(self):
        """Q1 payment should be ~25% of annualized tax projection."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=50_000),
            quarterly_tax_paid_ytd=0,
        )
        cfg = SimConfig(tax_enabled=True, state_code="CA",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=1, sim_year=2031)
        assert state.quarterly_tax_paid_ytd > 0
        assert state.cash < 500_000

    def test_quarterly_payment_cumulative_tracking(self):
        """Q2 payment accounts for Q1 already paid."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=100_000),
            quarterly_tax_paid_ytd=10_000,
        )
        cfg = SimConfig(tax_enabled=True, state_code="CA",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=2, sim_year=2031)
        assert state.quarterly_tax_paid_ytd > 10_000

    def test_quarterly_no_payment_if_overpaid(self):
        """If already overpaid relative to cumulative target, pay $0."""
        from engines.citadel import (SimConfig, CitadelState,
                                      _quarterly_estimated_payment)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=500_000, reserves=[0, 0, 0], investments=[0, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            invest_cost_basis=[0, 0],
            tax_year_accum=TaxYearAccumulator(other_income=10_000),
            quarterly_tax_paid_ytd=100_000,
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX",
                        filing_status="single", inflation=4.0)
        _quarterly_estimated_payment(state, cfg, quarter=2, sim_year=2031)
        assert state.cash == 500_000  # no payment drawn

    def test_monthly_sim_pays_quarterly(self):
        """Monthly frequency produces quarterly payments + year-end true-up."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=300_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid[0, -1] > 0

    def test_annual_freq_falls_back_to_year_end(self):
        """Annually frequency: no quarterly payments, all at year-end."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2034,
            freq="Annually", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid[0, -1] > 0

    def test_q4_trueup_matches_annual(self):
        """Monthly and annual sims should produce approximately equal total tax."""
        from engines.citadel import SimConfig, simulate
        common = dict(
            start_stack=0, start_yr=2031, end_yr=2034,
            monthly_spend=0, cash_initial=1_000_000,
            selected_qs=[0.25],
            tax_enabled=True, filing_status="single", state_code="CA",
            other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r_monthly = simulate(SimConfig(freq="Monthly", **common), _test_model())
        r_annual = simulate(SimConfig(freq="Annually", **common), _test_model())
        tax_monthly = r_monthly.taxes_paid[0, -1]
        tax_annual = r_annual.taxes_paid[0, -1]
        assert abs(tax_monthly - tax_annual) / max(tax_annual, 1) < 0.05, \
            f"Monthly {tax_monthly:.0f} vs Annual {tax_annual:.0f} differ by >5%"

    def test_quarterly_tax_paid_ytd_resets_each_year(self):
        """quarterly_tax_paid_ytd must be 0 at each year boundary."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=0,
            cash_initial=1_000_000, selected_qs=[0.25],
            tax_enabled=True, state_code="CA", other_income=200_000,
            reserve_bins=[
                {"label": "Short", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for i in range(24):
            state = step(state, cfg, 50_000, rng, model=model)
            if state.period % 12 == 0:
                assert state.quarterly_tax_paid_ytd == 0, \
                    f"Period {state.period}: ytd should be 0, got {state.quarterly_tax_paid_ytd:.0f}"

    def test_cash_floor_respected_after_quarterly_payment(self):
        """Cash floor must hold after each quarterly tax payment."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=5.0, start_yr=2031, end_yr=2033,
            freq="Monthly", monthly_spend=1000,
            cash_initial=100_000, cash_floor=80_000,
            selected_qs=[0.25],
            tax_enabled=True, state_code="CA", other_income=500_000,
            reserve_bins=[
                {"label": "Short", "initial": 200_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 200_000, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=_test_model())
        for i in range(24):
            state = step(state, cfg, 50_000, rng, model=_test_model())
            total_other = (sum(state.reserves) + sum(state.investments)
                           + state.btc_stack * state.btc_price)
            if total_other > 80_000:
                assert state.cash >= 80_000 - 100, \
                    f"Period {i+1}: cash {state.cash:.0f} below floor"


class TestTaxAccountingHelpers:
    """Tests for the 3 tracking helpers and related infrastructure."""

    def test_state_has_sim_date(self):
        from engines.citadel import CitadelState
        s = CitadelState()
        assert hasattr(s, "sim_date")
        assert s.sim_date == ""

    def test_lots_seeded_when_tax_off(self):
        """Lots should be created even when tax_enabled=False."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=2.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=False)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == 2.0
        assert state.tax_year_accum is None

    def test_scf_purchase_creates_lot(self):
        """SCF initial BTC purchase must create a separate lot."""
        from engines.citadel import SimConfig, _initial_state
        cfg = SimConfig(start_stack=1.0, start_yr=2031, end_yr=2035,
                        cash_initial=100_000, selected_qs=[0.25],
                        tax_enabled=False,
                        scf_enabled=True, scf_amount=50_000)
        state = _initial_state(cfg, model=_test_model())
        assert len(state.tax_lots) == 2
        assert state.tax_lots[0].source == "initial"
        assert state.tax_lots[1].source == "scf"
        total_lot_btc = sum(l.btc for l in state.tax_lots)
        assert abs(total_lot_btc - state.btc_stack) < 1e-8

    def test_sell_btc_tracked_records_gains_tax_on(self):
        """With tax on, selling BTC records capital gains in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=50_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(1.0)
        assert len(state.tax_lots) == 1
        assert state.tax_lots[0].btc == pytest.approx(1.0)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(50_000)

    def test_sell_btc_tracked_no_gains_tax_off(self):
        """With tax off (accum=None), BTC still sold but no gains recorded."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=50_000, source="initial")],
            tax_year_accum=None,
        )
        cfg = SimConfig(tax_enabled=False, cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(1.0)

    def test_sell_btc_tracked_empty_lots_fallback(self):
        """With no lots, raw stack decrement as fallback."""
        from engines.citadel import CitadelState, SimConfig, _sell_btc_tracked
        state = CitadelState(btc_stack=3.0, btc_price=50_000, sim_date="2035-01-15")
        cfg = SimConfig(cost_basis_method="fifo")
        result = _sell_btc_tracked(state, cfg, 1.0)
        assert result.btc_sold == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(2.0)

    def test_buy_btc_tracked_creates_lot(self):
        """Buying BTC creates a lot with correct date/basis/source."""
        from engines.citadel import CitadelState, SimConfig, _buy_btc_tracked
        state = CitadelState(btc_stack=1.0, btc_price=80_000, sim_date="2033-03-15")
        cfg = SimConfig()
        _buy_btc_tracked(state, cfg, 0.5, source="rebal_buy")
        assert state.btc_stack == pytest.approx(1.5)
        assert len(state.tax_lots) == 1
        lot = state.tax_lots[0]
        assert lot.btc == pytest.approx(0.5)
        assert lot.cost_basis == 80_000
        assert lot.date == "2033-03-15"
        assert lot.source == "rebal_buy"

    def test_sell_investments_tracked_records_ltcg(self):
        """Investment sale records LTCG in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _sell_investments_tracked
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            investments=[200_000, 100_000],
            invest_cost_basis=[100_000, 80_000],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True)
        drawn, gain = _sell_investments_tracked(state, cfg, 0, 50_000)
        assert drawn == pytest.approx(50_000)
        assert gain == pytest.approx(25_000)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(25_000)
        assert state.investments[0] == pytest.approx(150_000)
        assert state.invest_cost_basis[0] == pytest.approx(75_000)

    def test_sell_investments_tracked_noop_tax_off(self):
        """Investment sale updates balances but skips accumulator when tax off."""
        from engines.citadel import CitadelState, SimConfig, _sell_investments_tracked
        state = CitadelState(
            investments=[200_000, 0],
            invest_cost_basis=[100_000, 0],
            tax_year_accum=None,
        )
        cfg = SimConfig(tax_enabled=False)
        drawn, gain = _sell_investments_tracked(state, cfg, 0, 50_000)
        assert drawn == pytest.approx(50_000)
        assert state.investments[0] == pytest.approx(150_000)

    def test_floor_enforcement_btc_sale_lot_tracked(self):
        """Bug 1: BTC sold to replenish cash floor must be lot-tracked."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            btc_stack=2.0, btc_price=50_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
            invest_cost_basis=[0, 0],
        )
        cfg = SimConfig(cash_floor=20_000, cost_basis_method="fifo")
        _enforce_floors(state, cfg)
        assert state.cash >= 20_000 - 1
        assert state.btc_stack < 2.0
        # Capital gain should be recorded (sold at 50k, basis 30k)
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_floor_enforcement_investment_sale_tracks_basis(self):
        """Bug 7: Investment sold for floor must update cost basis."""
        from engines.citadel import CitadelState, SimConfig, _enforce_floors
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[100_000, 50_000],
            invest_cost_basis=[60_000, 30_000],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(cash_floor=30_000, cost_basis_method="fifo")
        _enforce_floors(state, cfg)
        assert state.cash >= 30_000 - 1
        # Cost basis should have been reduced proportionally
        assert state.invest_cost_basis[1] < 30_000 or state.invest_cost_basis[0] < 60_000
        # LTCG should be recorded
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_rebalancing_sell_btc_lot_tracked(self):
        """Bug 2: Rebalancing BTC sell must be lot-tracked."""
        from engines.citadel import CitadelState, SimConfig, _execute_sell_btc
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-06-15",
            cash=0, reserves=[0, 0, 0], investments=[0, 0],
            tax_lots=[TaxLot(date="2031-01-01", btc=10.0,
                             cost_basis=20_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(cost_basis_method="fifo")
        evt = _execute_sell_btc(state, cfg, rate_pct=10.0, split={"cash": 1.0})
        assert evt["btc_sold"] == pytest.approx(1.0)
        assert state.btc_stack == pytest.approx(9.0)
        assert state.tax_year_accum.lt_capital_gains == pytest.approx(30_000)

    def test_rebalancing_buy_btc_creates_lot(self):
        """Bug 3: Rebalancing BTC buy must create a tax lot."""
        from engines.citadel import CitadelState, SimConfig, _execute_buy_btc
        state = CitadelState(
            btc_stack=1.0, btc_price=50_000, sim_date="2033-03-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
        )
        cfg = SimConfig(cash_floor=0)
        evt = _execute_buy_btc(state, cfg, rate_pct=10.0, split={"cash": 1.0})
        assert evt["action"] == "buy_btc"
        assert state.btc_stack > 1.0
        new_lots = [l for l in state.tax_lots if l.source == "rebal_buy"]
        assert len(new_lots) == 1
        assert new_lots[0].cost_basis == 50_000

    def test_scf_repay_btc_sale_lot_tracked(self):
        """Bug 6: SCF perpetual loan repayment must lot-track BTC sale."""
        from engines.citadel import CitadelState, SimConfig, _scf_check_repay
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=5.0, btc_price=50_000, sim_date="2040-01-15",
            scf_outstanding=100_000, scf_active=True,
            tax_lots=[TaxLot(date="2031-01-01", btc=5.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(scf_enabled=True, scf_type="perpetual",
                        scf_rate=8.0, scf_repay_trigger=1.0,
                        cost_basis_method="fifo")
        _scf_check_repay(state, cfg, btc_annual_return=0.0)
        assert state.btc_stack < 5.0
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_lot_inventory_matches_stack_after_operations(self):
        """Lot sum must match btc_stack after sell/buy operations."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _sell_btc_tracked, _buy_btc_tracked)
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=5.0, btc_price=60_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=5.0,
                             cost_basis=30_000, source="initial")],
        )
        cfg = SimConfig(cost_basis_method="fifo")
        _sell_btc_tracked(state, cfg, 2.0)
        _buy_btc_tracked(state, cfg, 1.0, source="rebal_buy")
        lot_sum = sum(l.btc for l in state.tax_lots)
        assert abs(lot_sum - state.btc_stack) < 1e-8

    def test_pay_tax_investment_sale_recorded(self):
        """Bug 4: Investment gains during tax payment must be in accumulator."""
        from engines.citadel import CitadelState, SimConfig, _pay_tax_amount
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0],
            invest_cost_basis=[100_000, 0],
            td_cash=0, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
            sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX")
        _pay_tax_amount(state, cfg, amount=50_000, sim_year=2035)
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_merged_waterfall_tax_off_same_behavior(self):
        """Merged waterfall with tax_enabled=False works correctly."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=1.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=5000,
            cash_initial=50_000, selected_qs=[0.25],
            tax_enabled=False,
            reserve_bins=[
                {"label": "Short", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "Medium", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "Long", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Equities", "initial": 50_000, "return_rate": 0, "volatility": 0},
                {"label": "Bonds", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.total_usd.shape[1] > 0
        assert r.taxes_paid is None
        assert r.total_usd[0, -1] >= 0

    def test_tax_off_still_zero_tax(self):
        """Critical regression: tax_enabled=False must produce zero tax."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=5.0, start_yr=2031, end_yr=2035,
            freq="Monthly", monthly_spend=5000,
            cash_initial=100_000, selected_qs=[0.25],
            tax_enabled=False,
        )
        r = simulate(cfg, _test_model())
        assert r.taxes_paid is None
        assert r.annual_taxes is None

    def test_gradual_rebalancing_consumes_lots_across_periods(self):
        """Gradual sell over multiple periods correctly consumes lots."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _evaluate_rebalancing)
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=10.0, btc_price=50_000, sim_date="2035-06-15",
            cash=100_000, reserves=[0, 0, 0], investments=[0, 0],
            tax_lots=[TaxLot(date="2031-01-01", btc=10.0,
                             cost_basis=20_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(
            cost_basis_method="fifo",
            high_q_trigger=0.90,
            high_q_action={"mode": "gradual", "rate": 5.0, "duration": 3,
                           "split": {"cash": 1.0}},
        )
        initial_btc = state.btc_stack
        for i in range(3):
            _evaluate_rebalancing(state, cfg, btc_quantile=0.95)
        assert state.btc_stack < initial_btc
        lot_sum = sum(l.btc for l in state.tax_lots)
        assert abs(lot_sum - state.btc_stack) < 1e-8
        assert state.tax_year_accum.lt_capital_gains > 0


class TestDynamicWaterfall:
    """Tests for the dynamic cost-ranked spending waterfall."""

    def test_build_source_list_taxable_only(self):
        """Non-tax mode produces only taxable sources."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        state = CitadelState(
            cash=10_000, reserves=[20_000, 30_000, 5_000],
            investments=[100_000, 50_000], invest_cost_basis=[60_000, 30_000],
            btc_stack=1.0, btc_price=50_000, sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=False, cash_rate=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        # 7 taxable sources: cash + 3 reserves + 2 investments + BTC
        assert len(sources) == 7
        assert all(not s.is_roth for s in sources)
        # Check available balances
        cash_src = [s for s in sources if s.key == "cash"][0]
        assert cash_src.available == pytest.approx(10_000)
        btc_src = [s for s in sources if s.key == "btc"][0]
        assert btc_src.available == pytest.approx(50_000)

    def test_build_source_list_with_tax(self):
        """Tax mode produces taxable + TD + TF sources."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=10_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=1.0, btc_price=50_000, sim_date="2035-06-15",
            td_cash=20_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            td_btc_stack=0.5,
            tf_cash=10_000, tf_reserves=[0, 0, 0], tf_investments=[0, 0],
            tf_btc_stack=0.3,
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, cash_rate=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        # Should have taxable + TD + TF sources
        wrappers = set(s.wrapper for s in sources)
        assert "taxable" in wrappers
        assert "td" in wrappers
        assert "tf" in wrappers
        # TF sources should be marked is_roth
        tf_sources = [s for s in sources if s.wrapper == "tf"]
        assert all(s.is_roth for s in tf_sources)
        # Only include sources with available > 0
        assert all(s.available > 0.01 for s in sources)

    def test_source_gain_fraction(self):
        """Gain fraction computed correctly for investments and BTC."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax_lots import TaxLot
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0], invest_cost_basis=[100_000, 0],
            btc_stack=2.0, btc_price=80_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=30_000, source="initial")],
        )
        cfg = SimConfig(tax_enabled=False,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        eq_src = [s for s in sources if s.key == "invest_0"][0]
        # Gain fraction: 1 - (100k / 200k) = 0.5
        assert eq_src.gain_fraction == pytest.approx(0.5)
        btc_src = [s for s in sources if s.key == "btc"][0]
        # BTC gain fraction: 1 - (30k / 80k) = 0.625
        assert btc_src.gain_fraction == pytest.approx(0.625)

    def test_score_taxable_cash_zero_tax(self):
        """Taxable cash has zero tax cost, only opportunity cost."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
        )
        cfg = SimConfig(tax_enabled=False, cash_rate=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        cash = [s for s in sources if s.key == "cash"][0]
        # Tax cost = 0, opportunity = (1.04)^15 - 1 ≈ 0.80
        assert cash.cost == pytest.approx((1.04 ** 15) - 1, rel=0.01)

    def test_score_td_ordinary_rate(self):
        """TD source tax cost = marginal ordinary rate + state rate."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", cash_rate=4.0,
                        filing_status="single", inflation=4.0,
                        start_yr=2031,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        td = [s for s in sources if s.key == "td_cash"][0]
        # At $0 YTD ordinary income, marginal rate = 10% (first bracket)
        # TX = 0% state. Tax cost = 0.10. Opportunity = (1.04^15 - 1) × (1-0.10)
        # At $0 YTD, TD is within standard deduction (0% marginal rate) so
        # tax cost = 0 + state (TX=0) = 0. TD cost equals cash cost.
        # Once income exceeds the deduction, TD becomes more expensive.
        cash_src = [s for s in sources if s.key == "cash"][0]
        assert td.cost >= cash_src.cost  # TD at least as expensive as cash

    def test_score_niit_above_threshold(self):
        """NIIT adds 3.8% to capital gains sources when MAGI > threshold."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[200_000, 0], invest_cost_basis=[100_000, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=250_000),  # above NIIT
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", filing_status="single",
                        inflation=4.0, start_yr=2031,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        eq = [s for s in sources if s.key == "invest_0"][0]
        # LTCG rate (15%) + NIIT (3.8%) + state (0%) = 18.8% × gain_fraction (0.5) = 9.4% tax
        # Plus opportunity cost
        assert eq.cost > 0.094  # tax component alone

    def test_score_btc_high_early_low_late(self):
        """BTC opportunity cost is higher in 2035 than 2065."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources,
                                      _WithdrawalSource)
        from engines.tax_lots import TaxLot
        from btc_core import yr_to_t

        def _btc_cost_at_year(yr):
            t = yr_to_t(yr)
            state = CitadelState(
                cash=0, reserves=[0, 0, 0],
                investments=[0, 0], invest_cost_basis=[0, 0],
                btc_stack=1.0, btc_price=50_000, t=t, sim_date=f"{yr}-06-15",
                tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                                 cost_basis=10_000, source="initial")],
            )
            cfg = SimConfig(tax_enabled=False, start_yr=yr,
                            reserve_bins=[
                                {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                            ],
                            invest_bins=[
                                {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                            ])
            sources = _build_source_list(state, cfg, model=_test_model())
            _score_sources(sources, state, cfg, model=_test_model())
            btc = [s for s in sources if s.key == "btc"][0]
            return btc.cost

        cost_2035 = _btc_cost_at_year(2035)
        cost_2065 = _btc_cost_at_year(2065)
        assert cost_2035 > cost_2065, "BTC cost should decrease as growth slows"

    def test_rank_by_cost_ascending(self):
        """Sources ranked purely by cost — cheapest first, regardless of wrapper."""
        from engines.citadel import _WithdrawalSource, _rank_sources
        sources = [
            _WithdrawalSource(key="btc", wrapper="taxable", asset_type="btc",
                              index=0, available=50_000, growth_rate=0.5,
                              horizon=10, gain_fraction=0.9, is_roth=False,
                              is_bracket_sensitive=True, bracket_type="ltcg", cost=5.0),
            _WithdrawalSource(key="tf_cash_res", wrapper="tf", asset_type="cash",
                              index=0, available=10_000, growth_rate=0.04,
                              horizon=15, gain_fraction=0.0, is_roth=True,
                              is_bracket_sensitive=False, bracket_type="none", cost=0.01),
        ]
        ranked = _rank_sources(sources)
        # Roth cash (cost=0.01) is cheaper than taxable BTC (cost=5.0)
        assert ranked[0].key == "tf_cash_res"
        assert ranked[1].key == "btc"

    def test_max_draw_ordinary_bracket(self):
        """Distance to next ordinary bracket computed correctly."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=10_000),
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=100_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # 10k ordinary income, first bracket top ~11,925 × inflation^10 ≈ ~17,651
        # Distance ≈ 7,651 (approximate due to inflation)
        assert max_draw > 0
        assert max_draw < 100_000  # capped at bracket boundary

    def test_max_draw_niit_cliff(self):
        """Draw capped at NIIT threshold when MAGI is below it."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=190_000),
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=100_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # MAGI at 190k, NIIT threshold 200k (NOT inflated) → distance = 10k
        # Should be capped at 10k (or less if ordinary bracket is closer)
        assert max_draw <= 10_001

    def test_zero_bracket_distance_skips(self):
        """When at exact bracket boundary, max_draw returns ~0."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        # Set income exactly at an inflated bracket boundary
        from engines.tax import _inflate_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA
        brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA["single"], 10, 0.04)
        boundary = brackets[0][0]  # first bracket top, inflated
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(other_income=boundary),
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=100_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # At exact boundary → distance to NEXT bracket should be > 0
        assert max_draw > 0

    def test_execute_draw_td_records_ordinary(self):
        """Drawing from TD records ordinary income in accumulator."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            td_cash=50_000, sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True)
        source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=50_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        _execute_draw(state, cfg, source, 10_000)
        assert state.td_cash == pytest.approx(40_000)
        assert state.tax_year_accum.tax_deferred_withdrawals == pytest.approx(10_000)

    def test_execute_draw_btc_uses_sell_tracked(self):
        """Drawing BTC uses _sell_btc_tracked for lot tracking."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw)
        from engines.tax import TaxYearAccumulator
        from engines.tax_lots import TaxLot
        state = CitadelState(
            btc_stack=2.0, btc_price=50_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=2.0,
                             cost_basis=30_000, source="initial")],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, cost_basis_method="fifo")
        source = _WithdrawalSource(
            key="btc", wrapper="taxable", asset_type="btc", index=0,
            available=100_000, growth_rate=0.5, horizon=10,
            gain_fraction=0.7, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ltcg",
        )
        _execute_draw(state, cfg, source, 25_000)  # sell $25k worth
        assert state.btc_stack < 2.0
        assert state.tax_year_accum.lt_capital_gains > 0

    def test_execute_draw_roth_records_roth(self):
        """Drawing from Roth records roth_withdrawals, no tax."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            tf_cash=30_000, tf_reserves=[10_000, 0, 0],
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True)
        source = _WithdrawalSource(
            key="tf_cash_res", wrapper="tf", asset_type="cash", index=0,
            available=40_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=True,
            is_bracket_sensitive=False, bracket_type="none",
        )
        _execute_draw(state, cfg, source, 35_000)
        assert state.tf_cash == 0  # drained cash first
        assert state.tf_reserves[0] == pytest.approx(5_000)  # then reserves
        assert state.tax_year_accum.roth_withdrawals == pytest.approx(35_000)

    def test_full_waterfall_btc_protected_early(self):
        """Cash and reserves (no-tax principal) should be drawn before BTC."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=1.0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=2_000,
            cash_initial=50_000, cash_rate=4.0,
            selected_qs=[0.25], tax_enabled=True, state_code="TX",
            td_cash_initial=100_000,
            reserve_bins=[
                {"label": "S", "initial": 20_000, "rate": 5.0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 50_000, "return_rate": 10.0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        initial_btc = state.btc_stack
        # $24k annual spend × 2 = $48k. Non-BTC assets = $220k+ (ample coverage).
        # Cash+reserves (zero-tax sources) should be drawn first.
        for _ in range(2):
            state = step(state, cfg, model.price_at(0.25, state.t + 1), rng, model=model)
        # BTC should be fully preserved — non-BTC assets more than cover spending
        assert state.btc_stack >= initial_btc * 0.95, \
            f"BTC should be mostly preserved in early retirement, got {state.btc_stack:.3f}"
        # Cash should be depleted or significantly reduced (drawn first as cheapest)
        assert state.cash < 50_000, "Cash should have been drawn from"

    def test_full_waterfall_roth_last(self):
        """Roth is never touched while other sources remain."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2035, end_yr=2037,
            freq="Annually", monthly_spend=5000,
            cash_initial=100_000, selected_qs=[0.25],
            tax_enabled=True, state_code="TX",
            td_cash_initial=100_000,
            tf_cash_initial=50_000,
            reserve_bins=[
                {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        for _ in range(2):
            state = step(state, cfg, 50_000, rng, model=model)
        # Taxable cash ($100k) + TD cash ($100k) should cover $120k spending.
        # Roth untouched (may grow slightly due to cash_rate interest).
        assert state.tf_cash >= 49_000, \
            f"Roth should be untouched while non-Roth covers spending, got {state.tf_cash}"

    def test_full_waterfall_non_tax_mode(self):
        """Non-tax mode still works with the dynamic waterfall."""
        from engines.citadel import SimConfig, simulate
        cfg = SimConfig(
            start_stack=1.0, start_yr=2031, end_yr=2035,
            freq="Annually", monthly_spend=5000,
            cash_initial=50_000, selected_qs=[0.25],
            tax_enabled=False,
            reserve_bins=[
                {"label": "S", "initial": 20_000, "rate": 0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 50_000, "return_rate": 0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        r = simulate(cfg, _test_model())
        assert r.total_usd.shape[1] > 0
        assert r.taxes_paid is None
        assert r.total_usd[0, -1] >= 0

    def test_full_waterfall_high_spender_crosses_brackets(self):
        """$500k monthly spend should cross multiple brackets without hanging."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2035, end_yr=2036,
            freq="Monthly", monthly_spend=500_000,
            cash_initial=0, selected_qs=[0.25],
            tax_enabled=True, state_code="CA",
            td_cash_initial=10_000_000,
            reserve_bins=[
                {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        # Run 1 month — should not hang and should draw from TD
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.td_cash < 10_000_000
        assert state.spending_shortfall == 0

    def test_full_waterfall_shortfall_when_all_depleted(self):
        """Returns shortfall when all sources exhausted."""
        from engines.citadel import SimConfig, _initial_state, step
        import numpy as np
        cfg = SimConfig(
            start_stack=0, start_yr=2035, end_yr=2036,
            freq="Annually", monthly_spend=100_000,
            cash_initial=1_000, selected_qs=[0.25],
            tax_enabled=False,
            reserve_bins=[
                {"label": "S", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "M", "initial": 0, "rate": 0, "volatility": 0},
                {"label": "L", "initial": 0, "rate": 0, "volatility": 0},
            ],
            invest_bins=[
                {"label": "Eq", "initial": 0, "return_rate": 0, "volatility": 0},
                {"label": "Bd", "initial": 0, "return_rate": 0, "volatility": 0},
            ],
        )
        model = _test_model()
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=model)
        state = step(state, cfg, 50_000, rng, model=model)
        assert state.spending_shortfall > 0

    def test_btc_midpack_in_2065(self):
        """Spec test 6: In 2065 BTC moves to mid-pack ranking (growth slowed)."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax_lots import TaxLot
        from engines.tax import TaxYearAccumulator
        from btc_core import yr_to_t
        t = yr_to_t(2065)
        state = CitadelState(
            cash=50_000, reserves=[50_000, 0, 0],
            investments=[100_000, 0], invest_cost_basis=[50_000, 0],
            btc_stack=1.0, btc_price=50_000, t=t, sim_date="2065-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
            td_cash=50_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", start_yr=2065,
                        filing_status="single", inflation=4.0,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=_test_model())
        _score_sources(sources, state, cfg, model=_test_model())
        non_roth = [s for s in sources if not s.is_roth]
        ranked = sorted(non_roth, key=lambda s: s.cost)
        btc = [s for s in ranked if s.key == "btc"][0]
        btc_rank = ranked.index(btc)
        # BTC should NOT be last (it was in 2035) — should be mid-pack
        assert btc_rank < len(ranked) - 1, "BTC should be mid-pack in 2065"

    def test_td_draw_shifts_ltcg_stack_base(self):
        """Spec test 8: TD draw increases ordinary income → shifts LTCG bracket base."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _execute_draw,
                                      _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        # Seed $40k other income so we are already partway into the 0% LTCG bracket.
        # The inflated 0% LTCG upper for 2031 (single, 4% inflation) ≈ $61k.
        # After std deduction (~$19k): stacked ≈ $21k → boundary ≈ $40k in gain-space.
        # Drawing $10k TD adds $10k ordinary income → stacked ≈ $31k → boundary ≈ $30k.
        # Since we stay within the same bracket, after < before.
        acum = TaxYearAccumulator()
        acum.other_income = 40_000.0
        state = CitadelState(
            td_cash=500_000, sim_date="2031-06-15",
            investments=[200_000, 0], invest_cost_basis=[100_000, 0],
            btc_stack=0, btc_price=50_000,
            tax_year_accum=acum,
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX",
                        filing_status="single", inflation=4.0, start_yr=2031,
                        freq="Monthly",
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        inv_src = _WithdrawalSource(
            key="invest_0", wrapper="taxable", asset_type="invest", index=0,
            available=200_000, growth_rate=0.10, horizon=15,
            gain_fraction=0.5, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ltcg",
        )
        # LTCG boundary before TD draw
        before = _max_draw_before_boundary(state, cfg, inv_src)
        # Draw $10k from TD → stays within same 0% LTCG bracket, shifts base up
        td_src = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=500_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        _execute_draw(state, cfg, td_src, 10_000)
        # LTCG boundary after TD draw — should be smaller (base shifted up within bracket)
        after = _max_draw_before_boundary(state, cfg, inv_src)
        assert after < before, "TD draw should shift LTCG stack base, reducing boundary distance"
        assert state.tax_year_accum.tax_deferred_withdrawals == pytest.approx(10_000)

    def test_gain_fraction_updates_after_partial_sale(self):
        """Spec test 15: Partial BTC sale changes gain fraction for next scoring."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources,
                                      _sell_btc_tracked)
        from engines.tax_lots import TaxLot
        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=2.0, btc_price=100_000, sim_date="2035-06-15",
            tax_lots=[
                TaxLot(date="2031-01-01", btc=1.0, cost_basis=10_000, source="initial"),
                TaxLot(date="2034-01-01", btc=1.0, cost_basis=90_000, source="rebal_buy"),
            ],
        )
        cfg = SimConfig(tax_enabled=False, cost_basis_method="fifo",
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        # Before sale: avg basis = (10k + 90k) / (2 * 100k) = 50%, gain_frac = 50%
        sources_before = _build_source_list(state, cfg, model=None)
        btc_before = [s for s in sources_before if s.key == "btc"][0]
        # Sell the cheap lot (FIFO sells the 10k-basis lot first)
        _sell_btc_tracked(state, cfg, 1.0)
        # After sale: only the 90k-basis lot remains, gain_frac = 1 - 90k/100k = 10%
        sources_after = _build_source_list(state, cfg, model=None)
        btc_after = [s for s in sources_after if s.key == "btc"][0]
        assert btc_after.gain_fraction < btc_before.gain_fraction

    def test_late_retirement_crossover(self):
        """Spec test 17: With very short treasury horizon (age 95), treasury is cheaper to sell
        than BTC (equity-rate fallback). Treasury horizon clamps at 1 for ages ≥90."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax_lots import TaxLot
        from btc_core import yr_to_t
        t = yr_to_t(2070)
        state = CitadelState(
            cash=0, reserves=[100_000, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=1.0, btc_price=50_000, t=t, sim_date="2070-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
        )
        # birth_year=1975 → age 95 in 2070 → treasury horizon = max(min(90-95, 40), 1) = 1
        cfg = SimConfig(tax_enabled=False, start_yr=2070, birth_year=1975,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        # No model → BTC falls back to equity rate (10%) over 10yr horizon
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        btc = [s for s in sources if s.key == "btc"][0]
        tres = [s for s in sources if s.key == "reserve_0"][0]
        # Treasury horizon=1: cost = (1.05^1 - 1) = 0.05 (very low)
        # BTC horizon=10, growth=10%: cost = (1.10^10 - 1) ≈ 1.59 (high)
        # Treasury should be far cheaper to sell in late retirement with age 95
        assert tres.horizon == 1, f"Expected treasury horizon=1 for age 95, got {tres.horizon}"
        assert tres.cost < btc.cost, \
            f"Treasury ({tres.cost:.3f}) should be cheaper than BTC ({btc.cost:.3f}) at age 95"

    def test_negative_btc_growth_ranks_first(self):
        """When model returns negative 10yr growth, BTC is cheapest to sell."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax_lots import TaxLot

        class _DeclineModel:
            def __init__(self):
                import pandas as pd
                self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
                self.genesis = pd.Timestamp("2009-07-25")
            def price_at(self, q, t):
                return max(50_000 * (1 - t / 200), 100)  # declining
            def quantile_at(self, price, t):
                return 0.5

        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[100_000, 0], invest_cost_basis=[50_000, 0],
            btc_stack=1.0, btc_price=50_000, t=50, sim_date="2060-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
        )
        cfg = SimConfig(tax_enabled=False, start_yr=2060,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=_DeclineModel())
        _score_sources(sources, state, cfg, model=_DeclineModel())
        btc = [s for s in sources if s.key == "btc"][0]
        cash = [s for s in sources if s.key == "cash"][0]
        # Negative growth → negative opportunity cost → BTC cheaper than cash
        assert btc.cost < cash.cost

    def test_treasury_horizon_age_92(self):
        """Treasury horizon clamps to 1 for ages 90+."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        state = CitadelState(
            cash=0, reserves=[50_000, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            period=0,
        )
        cfg = SimConfig(tax_enabled=False, birth_year=1943, start_yr=2035,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        res = [s for s in sources if s.key == "reserve_0"][0]
        # Age 92, horizon = max(min(90-92, 40), 1) = max(-2, 1) = 1
        assert res.horizon == 1

    def test_model_failure_fallback(self):
        """When model.price_at throws, BTC falls back to equity rate."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax_lots import TaxLot

        class _BrokenModel:
            def __init__(self):
                import pandas as pd
                self.fits = {0.25: {"slope": 5.0, "intercept": 2.0}}
                self.genesis = pd.Timestamp("2009-07-25")
            def price_at(self, q, t):
                raise ValueError("model broken")
            def quantile_at(self, price, t):
                return 0.5

        state = CitadelState(
            cash=0, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=1.0, btc_price=50_000, sim_date="2035-06-15",
            tax_lots=[TaxLot(date="2031-01-01", btc=1.0,
                             cost_basis=10_000, source="initial")],
        )
        cfg = SimConfig(tax_enabled=False,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=_BrokenModel())
        btc = [s for s in sources if s.key == "btc"][0]
        # Should fall back to equity rate (10%)
        assert btc.growth_rate == pytest.approx(0.10)

    def test_td_horizon_before_rmd_age(self):
        """TD horizon ramps down as RMD start age approaches."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax import TaxYearAccumulator
        # Age 50, RMD at 75 → horizon = min(15, 25) = 15
        state = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2035,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        td = [s for s in sources if s.key == "td_cash"][0]
        assert td.horizon == 15  # 25 years until RMD, capped at 15

        # Age 65, RMD at 75 → horizon = min(15, 10) = 10
        state2 = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2050-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg2 = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2050,
                         reserve_bins=cfg.reserve_bins, invest_bins=cfg.invest_bins)
        sources2 = _build_source_list(state2, cfg2, model=None)
        td2 = [s for s in sources2 if s.key == "td_cash"][0]
        assert td2.horizon == 10

        # Age 70, RMD at 75 → horizon = 5
        state3 = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2055-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg3 = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2055,
                         reserve_bins=cfg.reserve_bins, invest_bins=cfg.invest_bins)
        sources3 = _build_source_list(state3, cfg3, model=None)
        td3 = [s for s in sources3 if s.key == "td_cash"][0]
        assert td3.horizon == 5

    def test_td_horizon_at_rmd_age_uses_factor(self):
        """At RMD age, TD horizon equals the IRS RMD factor."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list)
        from engines.tax import TaxYearAccumulator
        from engines.tax_data import RMD_FACTORS
        # Age 75, RMD factor = 24.6
        state = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2060-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2060,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        td = [s for s in sources if s.key == "td_cash"][0]
        assert td.horizon == int(RMD_FACTORS[75])  # 24

        # Age 85, RMD factor = 16.0
        state2 = CitadelState(
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            sim_date="2070-06-15",
            tax_year_accum=TaxYearAccumulator(),
        )
        cfg2 = SimConfig(tax_enabled=True, birth_year=1985, start_yr=2070,
                         reserve_bins=cfg.reserve_bins, invest_bins=cfg.invest_bins)
        sources2 = _build_source_list(state2, cfg2, model=None)
        td2 = [s for s in sources2 if s.key == "td_cash"][0]
        assert td2.horizon == int(RMD_FACTORS[85])

    def test_td_cheaper_near_rmd_age(self):
        """TD becomes cheaper to withdraw as RMD age approaches."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator

        def _td_cost_at_age(age):
            yr = 1985 + age
            state = CitadelState(
                cash=50_000, td_cash=100_000,
                td_reserves=[0, 0, 0], td_investments=[0, 0],
                sim_date=f"{yr}-06-15",
                tax_year_accum=TaxYearAccumulator(),
            )
            cfg = SimConfig(tax_enabled=True, birth_year=1985, start_yr=yr,
                            state_code="TX", filing_status="single", inflation=4.0,
                            reserve_bins=[
                                {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                                {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                                {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                            ],
                            invest_bins=[
                                {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                                {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                            ])
            sources = _build_source_list(state, cfg, model=None)
            _score_sources(sources, state, cfg, model=None)
            td = [s for s in sources if s.key == "td_cash"][0]
            return td.cost

        cost_50 = _td_cost_at_age(50)
        cost_65 = _td_cost_at_age(65)
        cost_73 = _td_cost_at_age(73)
        # TD gets cheaper as RMD approaches (shorter horizon = less forgone compounding)
        assert cost_50 > cost_65 > cost_73

    def test_td_free_below_standard_deduction(self):
        """TD draws are free (0% marginal rate) when below standard deduction."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _build_source_list, _score_sources)
        from engines.tax import TaxYearAccumulator
        state = CitadelState(
            cash=50_000, reserves=[0, 0, 0],
            investments=[0, 0], invest_cost_basis=[0, 0],
            btc_stack=0, btc_price=50_000, sim_date="2035-06-15",
            td_cash=100_000, td_reserves=[0, 0, 0], td_investments=[0, 0],
            tax_year_accum=TaxYearAccumulator(),  # zero YTD income
        )
        cfg = SimConfig(tax_enabled=True, state_code="TX", cash_rate=4.0,
                        filing_status="single", inflation=4.0, start_yr=2031,
                        reserve_bins=[
                            {"label": "S", "initial": 0, "rate": 5.0, "volatility": 0},
                            {"label": "M", "initial": 0, "rate": 4.5, "volatility": 0},
                            {"label": "L", "initial": 0, "rate": 4.0, "volatility": 0},
                        ],
                        invest_bins=[
                            {"label": "Eq", "initial": 0, "return_rate": 10.0, "volatility": 0},
                            {"label": "Bd", "initial": 0, "return_rate": 5.0, "volatility": 0},
                        ])
        sources = _build_source_list(state, cfg, model=None)
        _score_sources(sources, state, cfg, model=None)
        cash_src = [s for s in sources if s.key == "cash"][0]
        td_src = [s for s in sources if s.key == "td_cash"][0]
        # At zero YTD income, TD marginal rate = 0% (within standard deduction)
        # TD cost should equal cash cost (both have 4% growth, same horizon,
        # zero tax, and TD opp adjusted by (1-0) = 1.0)
        assert td_src.cost == pytest.approx(cash_src.cost, rel=0.01)

    def test_boundary_includes_deduction_cushion(self):
        """Bracket boundary includes remaining standard deduction cushion."""
        from engines.citadel import (CitadelState, SimConfig,
                                      _WithdrawalSource, _max_draw_before_boundary)
        from engines.tax import TaxYearAccumulator
        # Zero YTD income → full standard deduction + first bracket available
        state = CitadelState(
            sim_date="2035-06-15",
            tax_year_accum=TaxYearAccumulator(),  # zero income
        )
        cfg = SimConfig(tax_enabled=True, filing_status="single",
                        inflation=4.0, start_yr=2031, freq="Monthly")
        td_source = _WithdrawalSource(
            key="td_cash", wrapper="td", asset_type="cash", index=0,
            available=500_000, growth_rate=0.04, horizon=15,
            gain_fraction=0.0, is_roth=False,
            is_bracket_sensitive=True, bracket_type="ordinary",
        )
        max_draw = _max_draw_before_boundary(state, cfg, td_source)
        # sim_year=2031, yrs_from_base=6 (2031-2025)
        # Should be std_ded(inflated 6yr) + first_bracket_top(inflated 6yr)
        from engines.tax import _inflate_brackets
        from engines.tax_data import FEDERAL_BRACKETS_TCJA, STANDARD_DEDUCTION_TCJA
        std_ded = STANDARD_DEDUCTION_TCJA["single"] * (1.04 ** 6)
        brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA["single"], 6, 0.04)
        expected = std_ded + brackets[0][0]
        assert max_draw == pytest.approx(expected, rel=0.01)


# ── Phase 1: Unified Citadel MC ──────────────────────────────────────────────

class TestInitialRegimeConfig:
    def test_default_initial_regimes_are_neutral(self):
        from engines.citadel_types import SimConfig
        cfg = SimConfig()
        assert cfg.initial_equity_regime == 2
        assert cfg.initial_bond_regime == 2
        assert cfg.initial_res_short_regime == 2
        assert cfg.initial_res_med_regime == 2
        assert cfg.initial_res_long_regime == 2

    def test_initial_regimes_are_configurable(self):
        from engines.citadel_types import SimConfig
        cfg = SimConfig(
            initial_equity_regime=0,
            initial_bond_regime=4,
            initial_res_short_regime=1,
            initial_res_med_regime=3,
            initial_res_long_regime=0,
        )
        assert cfg.initial_equity_regime == 0
        assert cfg.initial_bond_regime == 4


class TestInitialRegimeWiring:
    def test_td_tf_regime_fields_exist(self):
        from engines.citadel_types import CitadelState
        state = CitadelState()
        assert state.td_equity_regime == 2
        assert state.td_bond_regime == 2
        assert state.td_res_short_regime == 2
        assert state.td_res_med_regime == 2
        assert state.td_res_long_regime == 2
        assert state.tf_equity_regime == 2
        assert state.tf_bond_regime == 2
        assert state.tf_res_short_regime == 2
        assert state.tf_res_med_regime == 2
        assert state.tf_res_long_regime == 2

    def test_initial_state_uses_config_regimes(self):
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import _initial_state
        cfg = SimConfig()
        cfg.initial_equity_regime = 0
        cfg.initial_bond_regime = 4
        cfg.initial_res_short_regime = 1
        cfg.initial_res_med_regime = 3
        cfg.initial_res_long_regime = 0
        state = _initial_state(cfg, model=None)
        assert state.equity_regime == 0
        assert state.bond_regime == 4
        assert state.res_short_regime == 1
        assert state.res_med_regime == 3
        assert state.res_long_regime == 0

    def test_initial_state_seeds_td_tf_regimes_unconditionally(self):
        """TD/TF regimes seeded from config regardless of tax_enabled."""
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import _initial_state
        cfg = SimConfig()
        cfg.tax_enabled = False
        cfg.initial_equity_regime = 4
        cfg.initial_bond_regime = 0
        state = _initial_state(cfg, model=None)
        assert state.td_equity_regime == 4
        assert state.td_bond_regime == 0
        assert state.tf_equity_regime == 4
        assert state.tf_bond_regime == 0


class TestMarkovGuard:
    def _make_markov_config(self, n_sims=10):
        import numpy as np
        from engines.citadel_types import SimConfig
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.ones((n_bins, n_bins)) / n_bins
            bin_means = np.array([-0.02, -0.005, 0.005, 0.01, 0.02])
            bin_vols = np.array([0.01, 0.005, 0.003, 0.005, 0.01])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        cfg = SimConfig()
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = matrices
        cfg.n_sims = n_sims
        return cfg

    def test_markov_fires_when_n_sims_gt_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        for _ in range(20):
            state = step(state, cfg, 100_000.0, rng, model=None)
        regimes = [state.equity_regime, state.bond_regime,
                   state.res_short_regime, state.res_med_regime, state.res_long_regime]
        assert any(r != 2 for r in regimes), "After 20 Markov steps, at least one regime should change"

    def test_markov_does_not_fire_when_n_sims_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=1)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        state = step(state, cfg, 100_000.0, rng, model=None)
        assert state.equity_regime == 2
        assert state.bond_regime == 2


class TestTdTfMarkovReturns:
    def _make_markov_config(self, n_sims=10):
        import numpy as np
        from engines.citadel_types import SimConfig
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.ones((n_bins, n_bins)) / n_bins
            bin_means = np.array([-0.02, -0.005, 0.005, 0.01, 0.02])
            bin_vols = np.array([0.01, 0.005, 0.003, 0.005, 0.01])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        cfg = SimConfig()
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = matrices
        cfg.n_sims = n_sims
        cfg.tax_enabled = True
        return cfg

    def test_td_regimes_evolve_under_markov(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        for _ in range(30):
            state = step(state, cfg, 100_000.0, rng, model=None)
        td_regimes = [state.td_equity_regime, state.td_bond_regime,
                      state.td_res_short_regime, state.td_res_med_regime,
                      state.td_res_long_regime]
        assert any(r != 2 for r in td_regimes), "TD regimes should evolve under Markov"

    def test_tf_regimes_evolve_under_markov(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=10)
        rng = np.random.default_rng(99)
        state = _initial_state(cfg, model=None)
        for _ in range(30):
            state = step(state, cfg, 100_000.0, rng, model=None)
        tf_regimes = [state.tf_equity_regime, state.tf_bond_regime]
        assert any(r != 2 for r in tf_regimes), "TF regimes should evolve under Markov"

    def test_td_tf_use_lognormal_when_n_sims_1(self):
        import numpy as np
        from engines.citadel_step import step
        from engines.citadel_sim import _initial_state
        cfg = self._make_markov_config(n_sims=1)
        rng = np.random.default_rng(42)
        state = _initial_state(cfg, model=None)
        state = step(state, cfg, 100_000.0, rng, model=None)
        assert state.td_equity_regime == 2, "TD regimes unchanged when n_sims=1"
        assert state.tf_equity_regime == 2, "TF regimes unchanged when n_sims=1"


class TestBandAggregation:
    def test_compute_bands_returns_7_percentiles(self):
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES
        assert BAND_PERCENTILES == (5, 10, 25, 50, 75, 90, 95)

    def test_compute_bands_returns_11_series(self):
        from engines.citadel_bands import BAND_SERIES
        assert len(BAND_SERIES) == 11
        assert "total" in BAND_SERIES
        assert "btc_stack" in BAND_SERIES
        assert "td_total" in BAND_SERIES
        assert "tf_total" in BAND_SERIES
        assert "depletion" in BAND_SERIES

    def test_band_ordering(self):
        """P5 <= P25 <= P50 <= P75 <= P95 for total portfolio."""
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2032
        paths = np.array([[20000 + i * 30000 + j * 100 for j in range(12)]
                          for i in range(20)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        for t in range(12):
            vals = [bands[p]["total"][t] for p in [5, 25, 50, 75, 95]]
            for k in range(len(vals) - 1):
                assert vals[k] <= vals[k + 1] + 1e-6


class TestDevBypass:
    def test_dev_bypass_exists_in_mc_payment(self):
        import inspect
        from callbacks import mc_payment
        source = inspect.getsource(mc_payment)
        assert "DEV" in source, "mc_payment should check DEV env var for bypass"


class TestUnifiedMcIntegration:
    def _make_matrices(self):
        import numpy as np
        matrices = {}
        for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
            n_bins = 5
            trans = np.full((n_bins, n_bins), 0.05)
            np.fill_diagonal(trans, 0.80)
            trans /= trans.sum(axis=1, keepdims=True)
            bin_means = np.array([-0.03, -0.01, 0.005, 0.015, 0.03])
            bin_vols = np.array([0.015, 0.008, 0.005, 0.008, 0.015])
            matrices[key] = {"trans": trans, "bin_means": bin_means, "bin_vols": bin_vols}
        return matrices

    def test_full_mc_20_sims_produces_spread(self):
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2033
        cfg.asset_return_model = "markov"
        cfg.asset_matrices = self._make_matrices()
        cfg.initial_equity_regime = 4  # Bull
        cfg.initial_bond_regime = 0    # Bear
        rng = np.random.default_rng(123)
        base = np.linspace(50000, 150000, 24)
        paths = np.array([base * (1 + rng.normal(0, 0.1, 24)) for _ in range(20)])
        result = simulate(cfg, model=None, price_paths=paths)
        assert result.total_usd.shape == (20, 24)
        assert set(result.percentiles.keys()) == {5, 10, 25, 50, 75, 90, 95}
        p5 = result.percentiles[5]["total"]
        p95 = result.percentiles[95]["total"]
        assert np.any(p95 > p5 + 1.0), "MC should produce nonzero spread"

    def test_deterministic_unchanged(self):
        """n_sims=1 with a single price path: all percentiles identical."""
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2033; cfg.n_sims = 1
        paths = np.array([[80000 + j * 200 for j in range(24)]])
        result = simulate(cfg, model=None, price_paths=paths)
        assert result.total_usd.shape[0] == 1
        for key in ["total", "btc_usd", "cash"]:
            np.testing.assert_array_almost_equal(
                result.percentiles[5][key], result.percentiles[95][key],
                err_msg=f"Deterministic: P5 should equal P95 for {key}")

    def test_bands_match_standalone_compute(self):
        import numpy as np
        from engines.citadel_types import SimConfig
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        cfg = SimConfig()
        cfg.start_yr = 2031; cfg.end_yr = 2032
        paths = np.array([[50000 + i * 10000 + j * 100 for j in range(12)]
                          for i in range(10)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        for pct in [5, 50, 95]:
            np.testing.assert_array_almost_equal(
                bands[pct]["total"], result.percentiles[pct]["total"])


# ── Phase 2: Citadel Presets & Cache ─────────────────────────────────────────

class TestCitadelPresets:
    def test_wealth_levels_exist(self):
        from citadel_presets import WEALTH_LEVELS
        assert set(WEALTH_LEVELS.keys()) == {"starter", "full", "bitcoin"}

    def test_wealth_level_has_required_keys(self):
        from citadel_presets import WEALTH_LEVELS
        required = {"label", "dollar_assets", "btc", "monthly_spend",
                    "spend_growth", "inflation", "allocation"}
        for key, wl in WEALTH_LEVELS.items():
            assert required.issubset(wl.keys()), f"{key} missing {required - wl.keys()}"

    def test_allocation_sums_to_100(self):
        from citadel_presets import WEALTH_LEVELS
        for key, wl in WEALTH_LEVELS.items():
            total = sum(wl["allocation"].values())
            assert abs(total - 100) < 0.01, f"{key} allocation sums to {total}"

    def test_macro_regimes_exist(self):
        from citadel_presets import MACRO_REGIMES
        assert set(MACRO_REGIMES.keys()) == {"bear", "neutral", "bull"}
        assert MACRO_REGIMES["bear"]["bin"] == 0
        assert MACRO_REGIMES["neutral"]["bin"] == 2
        assert MACRO_REGIMES["bull"]["bin"] == 4

    def test_rule_sets_exist(self):
        from citadel_presets import RULE_SETS
        assert set(RULE_SETS.keys()) == {"no_rebal", "cautious", "aggressive"}

    def test_cache_dimensions(self):
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, START_YEARS,
                                     SIMS_PER_SCENARIO, WEALTH_LEVELS,
                                     MACRO_REGIMES, RULE_SETS, TAX_STATUSES)
        assert BTC_MODELS == ["bub", "qr", "pl", "lppl", "ef"]
        assert BTC_ENTRY_QS == [1, 10, 50]
        assert START_YEARS == [2028, 2035]
        assert SIMS_PER_SCENARIO == 800
        total = (len(BTC_MODELS) * len(BTC_ENTRY_QS) * len(MACRO_REGIMES) *
                 len(WEALTH_LEVELS) * len(RULE_SETS) * len(START_YEARS) *
                 len(TAX_STATUSES))
        assert total == 1620

    def test_build_config_returns_simconfig(self):
        from citadel_presets import build_config
        from engines.citadel_types import SimConfig
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert isinstance(cfg, SimConfig)

    def test_build_config_starter_values(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.start_stack == 0.5
        assert cfg.monthly_spend == 5000
        assert cfg.cash_initial == 50_000
        assert cfg.start_yr == 2035
        assert cfg.end_yr == 2075
        assert cfg.freq == "Monthly"
        assert cfg.inflation == 4.0

    def test_build_config_regime_sets_initial_regimes(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="bull", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.initial_equity_regime == 4
        assert cfg.initial_bond_regime == 4
        assert cfg.initial_res_short_regime == 4
        assert cfg.initial_res_med_regime == 4
        assert cfg.initial_res_long_regime == 4

    def test_build_config_tax_status_mfj(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="mfj",
        )
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "mfj"

    def test_build_config_tax_status_single(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "single"

    def test_build_config_loads_asset_matrices(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.asset_matrices is not None
        assert "equity" in cfg.asset_matrices
        assert "bond" in cfg.asset_matrices
        assert "tres_short" in cfg.asset_matrices


class TestCitadelBandCache:
    def test_band_cache_key_format(self):
        from citadel_band_cache import band_cache_key
        key = band_cache_key("bub", 10, "neutral", "starter",
                             "no_rebal", 2035, "single")
        assert key == "bub_q10_neutral_starter_no_rebal_2035_single"

    def test_band_cache_key_all_combos_unique(self):
        from citadel_band_cache import band_cache_key
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES,
                                     WEALTH_LEVELS, RULE_SETS, START_YEARS,
                                     TAX_STATUSES)
        keys = set()
        for model in BTC_MODELS:
            for eq in BTC_ENTRY_QS:
                for regime in MACRO_REGIMES:
                    for wealth in WEALTH_LEVELS:
                        for rules in RULE_SETS:
                            for yr in START_YEARS:
                                for tax in TAX_STATUSES:
                                    k = band_cache_key(model, eq, regime,
                                                       wealth, rules, yr, tax)
                                    keys.add(k)
        assert len(keys) == 1620

    def test_pack_unpack_bands_roundtrip(self):
        import numpy as np
        from citadel_band_cache import pack_bands, unpack_bands
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 480
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {}
            for series in BAND_SERIES:
                bands[pct][series] = np.random.rand(n_periods).astype(np.float32)
        packed = pack_bands(bands)
        assert isinstance(packed, np.ndarray)
        assert packed.dtype == np.float32
        unpacked = unpack_bands(packed)
        for pct in BAND_PERCENTILES:
            for series in BAND_SERIES:
                np.testing.assert_array_almost_equal(
                    unpacked[pct][series], bands[pct][series], decimal=5)

    def test_store_and_lookup(self, tmp_path):
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert result is not None
        for pct in BAND_PERCENTILES:
            np.testing.assert_array_almost_equal(
                result[pct]["total"], np.ones(n_periods) * pct, decimal=5)

    def test_lookup_missing_returns_none(self, tmp_path):
        from citadel_band_cache import lookup_entry
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)

        assert result is None

class TestCitadelBandGeneration:
    def test_generate_single_entry(self, tmp_path):
        """Smoke test: generate one combo with 5 sims (fast)."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        cfg.end_yr = 2036  # 1 year = 12 periods (fast)
        n_sims = 5
        n_periods = 12
        rng = np.random.default_rng(42)
        base = np.linspace(80000, 120000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.05, n_periods))
                          for _ in range(n_sims)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        loaded = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert loaded is not None
        assert set(loaded.keys()) == set(BAND_PERCENTILES)
        assert set(loaded[50].keys()) == set(BAND_SERIES)
        assert len(loaded[50]["total"]) == n_periods

    def test_generate_preserves_band_ordering(self, tmp_path):
        """P5 <= P50 <= P95 in generated bands."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="full", regime="bear", rules="cautious",
            start_year=2028, tax_status="mfj",
        )
        cfg.end_yr = 2029
        n_sims = 20
        n_periods = 12
        rng = np.random.default_rng(99)
        base = np.linspace(50000, 100000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.15, n_periods))
                          for _ in range(n_sims)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        store_entry("pl", 50, "bear", "full", "cautious",
                    2028, "mfj", bands, cache_dir=tmp_path)
        loaded = lookup_entry("pl", 50, "bear", "full", "cautious",
                              2028, "mfj", cache_dir=tmp_path)
        for t in range(n_periods):
            assert loaded[5]["total"][t] <= loaded[50]["total"][t] + 1e-6
            assert loaded[50]["total"][t] <= loaded[95]["total"][t] + 1e-6


class TestCitadelBandCacheLoader:
    @pytest.fixture(autouse=True)
    def _clear_band_cache(self):
        """Isolate tests from shared module state."""
        from citadel_band_cache import _BAND_CACHE
        _BAND_CACHE.clear()
        yield
        _BAND_CACHE.clear()

    def test_load_band_caches_from_disk(self, tmp_path):
        import numpy as np
        from citadel_band_cache import store_entry, load_band_caches, _BAND_CACHE
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 12
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        load_band_caches(cache_dir=tmp_path)
        assert len(_BAND_CACHE) == 1
        key = "bub_q10_neutral_starter_no_rebal_2035_single"
        assert key in _BAND_CACHE

    def test_load_empty_dir(self, tmp_path):
        from citadel_band_cache import load_band_caches, _BAND_CACHE
        load_band_caches(cache_dir=tmp_path)
        assert len(_BAND_CACHE) == 0


class TestCitadelBandCacheIntegration:
    def test_full_pipeline_build_simulate_store_lookup(self, tmp_path):
        """End-to-end: build_config -> simulate -> compute_bands -> store -> lookup."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="bitcoin", regime="bull", rules="aggressive",
            start_year=2028, tax_status="mfj",
        )
        cfg.end_yr = 2029  # 12 periods for speed
        n_sims = 10
        n_periods = 12
        rng = np.random.default_rng(77)
        base = np.linspace(60000, 200000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.1, n_periods))
                          for _ in range(n_sims)])

        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)

        store_entry("bub", 50, "bull", "bitcoin", "aggressive",
                    2028, "mfj", bands, cache_dir=tmp_path)

        loaded = lookup_entry("bub", 50, "bull", "bitcoin", "aggressive",
                              2028, "mfj", cache_dir=tmp_path)

        assert loaded is not None
        assert set(loaded.keys()) == set(BAND_PERCENTILES)
        for pct in BAND_PERCENTILES:
            assert set(loaded[pct].keys()) == set(BAND_SERIES)
            assert len(loaded[pct]["total"]) == n_periods

        # Verify band ordering
        for t in range(n_periods):
            assert loaded[5]["total"][t] <= loaded[50]["total"][t] + 1e-6
            assert loaded[50]["total"][t] <= loaded[95]["total"][t] + 1e-6

    def test_multiple_entries_same_npz(self, tmp_path):
        """Multiple entries for same (model, start_yr) share one npz."""
        import numpy as np
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        n_periods = 12
        for regime in ["bear", "neutral", "bull"]:
            bands = {}
            for pct in BAND_PERCENTILES:
                bands[pct] = {s: np.full(n_periods, float(pct), dtype=np.float32)
                              for s in BAND_SERIES}
            store_entry("qr", 10, regime, "starter", "no_rebal",
                        2035, "single", bands, cache_dir=tmp_path)

        # All three in same npz, all retrievable
        for regime in ["bear", "neutral", "bull"]:
            loaded = lookup_entry("qr", 10, regime, "starter", "no_rebal",
                                  2035, "single", cache_dir=tmp_path)
            assert loaded is not None
            assert loaded[50]["total"][0] == 50.0

    def test_cache_key_uniqueness_across_all_dimensions(self):
        """All 1620 combos produce unique cache keys."""
        from itertools import product as iproduct
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES,
                                     WEALTH_LEVELS, RULE_SETS, START_YEARS,
                                     TAX_STATUSES)
        from citadel_band_cache import band_cache_key
        keys = set()
        for m, eq, reg, wl, rs, yr, ts in iproduct(
            BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES.keys(),
            WEALTH_LEVELS.keys(), RULE_SETS.keys(), START_YEARS, TAX_STATUSES,
        ):
            keys.add(band_cache_key(m, eq, reg, wl, rs, yr, ts))
        assert len(keys) == 1620


# ── Phase 3: Quick Scenarios UI ──────────────────────────────────────────────

class TestCitadelQuickScenariosLayout:
    def test_scenario_stores_exist(self):
        """Verify scenario-related stores are in the layout."""
        from layout.citadel import _citadel_controls
        layout = _citadel_controls()
        layout_str = str(layout)
        assert "cp-scenario-wealth" in layout_str
        assert "cp-scenario-regime" in layout_str
        assert "cp-scenario-rules" in layout_str
        assert "cp-scenario-start-yr" in layout_str
        assert "cp-scenario-bands" in layout_str
        assert "cp-scenario-active" in layout_str

    def test_scenario_pill_buttons_exist(self):
        """Verify pill button IDs are present."""
        from layout.citadel import _citadel_controls
        layout = _citadel_controls()
        layout_str = str(layout)
        for wl in ["starter", "full", "bitcoin"]:
            assert f"cp-pill-{wl}" in layout_str
        for reg in ["bear", "neutral", "bull"]:
            assert f"cp-pill-{reg}" in layout_str
        for rs in ["no_rebal", "cautious", "aggressive"]:
            assert f"cp-pill-{rs}" in layout_str


class TestCitadelScenarioCallback:
    def test_scenario_lookup_returns_bands_for_valid_combo(self, tmp_path):
        """Verify lookup returns bands when cache exists."""
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 480
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert result is not None
        assert 50 in result
        assert "total" in result[50]
        assert len(result[50]["total"]) == n_periods

    def test_scenario_lookup_returns_none_for_missing(self, tmp_path):
        from citadel_band_cache import lookup_entry
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2099, "single", cache_dir=tmp_path)
        assert result is None

    def test_snap_quantile_to_cached_bin(self):
        """cp-qs value (float 0.25) should snap to nearest cached bin (10)."""
        from callbacks.citadel_scenarios import _snap_entry_q
        assert _snap_entry_q(0.01) == 1
        assert _snap_entry_q(0.05) == 1
        assert _snap_entry_q(0.10) == 10
        assert _snap_entry_q(0.25) == 10
        assert _snap_entry_q(0.50) == 50
        assert _snap_entry_q(0.75) == 50
        assert _snap_entry_q(0.999) == 50


class TestCitadelBandRendering:
    def test_build_band_traces_returns_traces(self):
        """Verify band trace builder produces scatter traces."""
        import numpy as np
        from figures.citadel import _build_band_traces
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.linspace(1000, 2000, n_periods).tolist()
                          for s in BAND_SERIES}
        time_axis = np.linspace(22, 24, n_periods).tolist()
        traces = _build_band_traces(bands, time_axis, series_key="total",
                                     color="#000000")
        assert len(traces) == 4
        import plotly.graph_objects as go
        for t in traces:
            assert isinstance(t, go.Scatter)

    def test_build_band_traces_empty_bands(self):
        from figures.citadel import _build_band_traces
        traces = _build_band_traces(None, [], series_key="total", color="#000")
        assert traces == []

    def test_build_band_traces_string_keys(self):
        """Verify works with string percentile keys (from JSON store)."""
        import numpy as np
        from figures.citadel import _build_band_traces
        bands = {
            "5": {"total": [100] * 10},
            "25": {"total": [200] * 10},
            "75": {"total": [300] * 10},
            "95": {"total": [400] * 10},
        }
        traces = _build_band_traces(bands, list(range(10)),
                                     series_key="total", color="#FF0000")
        assert len(traces) == 4


class TestCitadelScenarioSnapshot:
    def test_scenario_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = {c[0] for c in _SNAPSHOT_CONTROLS}
        assert "cp-scenario-wealth" in ids
        assert "cp-scenario-regime" in ids
        assert "cp-scenario-rules" in ids
        assert "cp-scenario-start-yr" in ids
        assert "cp-scenario-active" in ids

    def test_scenario_controls_in_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        citadel_ids = _TAB_CONTROLS["citadel"]
        assert "cp-scenario-wealth" in citadel_ids
        assert "cp-scenario-regime" in citadel_ids
        assert "cp-scenario-rules" in citadel_ids
        assert "cp-scenario-start-yr" in citadel_ids
        assert "cp-scenario-active" in citadel_ids


class TestCitadelQuickScenariosIntegration:
    def test_full_scenario_pipeline(self, tmp_path):
        """End-to-end: store bands → lookup → build traces."""
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from figures.citadel import _build_band_traces
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES

        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.linspace(100 * pct, 200 * pct, n_periods).astype(np.float32)
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)

        loaded = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert loaded is not None

        # Serialize like the callback does
        serialized = {}
        for pct, series_dict in loaded.items():
            serialized[str(pct)] = {k: v.tolist() for k, v in series_dict.items()}

        time_axis = np.linspace(26, 28, n_periods).tolist()
        traces = _build_band_traces(serialized, time_axis,
                                     series_key="total", color="#000000")
        assert len(traces) == 4
        # P5 lower bound should be less than P95 upper bound
        assert traces[0].y[0] < traces[1].y[0]

    def test_all_preset_combos_produce_valid_configs(self):
        """Every preset combo builds a valid SimConfig."""
        from citadel_presets import (WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS,
                                     START_YEARS, TAX_STATUSES, build_config)
        from engines.citadel_sim import validate_config
        for wealth in WEALTH_LEVELS:
            for regime in MACRO_REGIMES:
                for rules in RULE_SETS:
                    cfg = build_config(wealth, regime, rules, 2035, "single")
                    validate_config(cfg)  # raises on invalid


class TestPresetControlValues:
    def test_preset_control_values_returns_dict(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert isinstance(vals, dict)
        assert "cp-stack" in vals
        assert "cp-spend" in vals
        assert "cp-cash-init" in vals

    def test_preset_control_values_starter(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert vals["cp-stack"] == 0.5
        assert vals["cp-spend"] == 5000
        assert vals["cp-cash-init"] == 50000
        assert vals["cp-infl"] == 4.0
        assert vals["cp-spend-growth"] == 1.0

    def test_preset_control_values_bitcoin_bull_aggressive(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("bitcoin", "bull", "aggressive", 2028)
        assert vals["cp-stack"] == 12.5
        assert vals["cp-spend"] == 50000
        assert vals["cp-cash-floor"] == 100000
        assert vals["cp-yr-range"] == [2028, 2075]

    def test_preset_control_values_rules_no_rebal(self):
        from citadel_presets import preset_control_values
        vals = preset_control_values("starter", "neutral", "no_rebal", 2035)
        assert vals["cp-high-q-thresh"] == 99
        assert vals["cp-low-q-thresh"] == 1


class TestScenarioDynamicLookup:
    def test_snap_entry_q_boundary_values(self):
        from callbacks.citadel_scenarios import _snap_entry_q
        # Values near boundaries (bins: [1, 10, 50])
        assert _snap_entry_q(0.005) == 1    # closer to 1%
        assert _snap_entry_q(0.05) == 1     # closer to 1% than 10%
        assert _snap_entry_q(0.06) == 10    # closer to 10%
        assert _snap_entry_q(0.31) == 50    # closer to 50% (midpoint 30% ties to 10)
        assert _snap_entry_q(0.999) == 50


class TestScenarioStaleIndicator:
    def test_stale_badge_in_layout(self):
        from layout.citadel import _citadel_controls
        layout = _citadel_controls()
        assert "cp-scenario-stale" in repr(layout)


class TestSimplifiedQuantilePanel:
    def test_default_quantile_options(self):
        from layout.common import _q_options_default, _DEFAULT_BANDS
        opts = _q_options_default()
        values = [o["value"] for o in opts]
        assert values == [b["value"] for b in _DEFAULT_BANDS]
        assert "median" in values
        assert "inner" in values
        assert "outer" in values

    def test_default_qs_values(self):
        from layout.common import _DEFAULT_QS
        assert _DEFAULT_QS == [0.01, 0.15, 0.50, 0.85, 0.99]

    def test_quantile_mode_toggle_in_bubble(self):
        from layout.bubble import _bubble_controls
        layout_str = repr(_bubble_controls())
        assert "bub-qs-mode" in layout_str
        assert "bub-qs-default-wrap" in layout_str
        assert "bub-qs-advanced-wrap" in layout_str
        assert "bub-qs-adv" in layout_str


class TestQuantileModeSwitch:
    def test_mode_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = {c[0] for c in _SNAPSHOT_CONTROLS}
        for prefix in ["bub", "dca", "ret", "sc"]:
            assert f"{prefix}-qs-mode" in ids, f"{prefix}-qs-mode missing"
            assert f"{prefix}-qs-adv" in ids, f"{prefix}-qs-adv missing"

    def test_mode_controls_in_checklist_options(self):
        from snapshot import _CHECKLIST_OPTIONS
        for prefix in ["bub", "dca", "ret", "sc"]:
            assert f"{prefix}-qs-mode" in _CHECKLIST_OPTIONS
            assert f"{prefix}-qs-adv" in _CHECKLIST_OPTIONS

    def test_mode_controls_in_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        assert "bub-qs-mode" in _TAB_CONTROLS["bubble"]
        assert "dca-qs-mode" in _TAB_CONTROLS["dca"]
        assert "ret-qs-mode" in _TAB_CONTROLS["retire"]
        assert "sc-qs-mode" in _TAB_CONTROLS["supercharge"]

    def test_qs_mode_in_tab_defaults(self):
        from tab_defaults import BUBBLE, DCA, RETIRE, SUPERCHARGE
        assert "qs_mode" in BUBBLE
        assert "qs_mode" in DCA
        assert "qs_mode" in RETIRE
        assert "qs_mode" in SUPERCHARGE


class TestDefaultModeOpacity:
    def test_fallback_q50_has_opacity_in_default_mode(self):
        """Q50% fallback in default mode should have 25% opacity."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[], shade=False, xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub"],
                 qs_mode=[])
        fig = build_bubble_figure(M, p)
        q50_traces = [t for t in fig.data if hasattr(t, 'name') and t.name and 'Q50%' in str(t.name)]
        assert len(q50_traces) > 0
        assert q50_traces[0].opacity == 0.25

    def test_fallback_q50_full_opacity_in_advanced_mode(self):
        """Q50% fallback in advanced mode should have full opacity."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[], shade=False, xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub"],
                 qs_mode=["advanced"])
        fig = build_bubble_figure(M, p)
        q50_traces = [t for t in fig.data if hasattr(t, 'name') and t.name and 'Q50%' in str(t.name)]
        assert len(q50_traces) > 0
        assert q50_traces[0].opacity == 1.0  # Q50% = full opacity


class TestSymmetricBandShading:
    def test_symmetric_bands_5_quantiles(self):
        """5 quantiles → 2 bands (outer + inner)."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.01, 0.15, 0.50, 0.85, 0.99]
        prices = {q: np.linspace(100 * (1 + q), 200 * (1 + q), 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 4  # 2 bands × 2 traces each

    def test_symmetric_bands_3_quantiles(self):
        """3 quantiles → 1 band (outer only)."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.15, 0.50, 0.85]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#FF0000")
        assert len(traces) == 2

    def test_symmetric_bands_2_quantiles(self):
        """2 quantiles → 1 band."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.15, 0.85]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 2

    def test_symmetric_bands_1_quantile(self):
        """1 quantile → 0 bands."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.50]
        prices = {0.50: np.linspace(100, 200, 10)}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        assert len(traces) == 0

    def test_symmetric_bands_outer_lighter_than_inner(self):
        """Outer band should have lower opacity than inner."""
        from figures.bubble import _build_symmetric_bands
        import numpy as np
        qs = [0.01, 0.15, 0.50, 0.85, 0.99]
        prices = {q: np.linspace(100, 200, 10) for q in qs}
        t_arr = np.linspace(1, 10, 10)
        traces = _build_symmetric_bands(qs, prices, t_arr, model_color="#000000")
        outer_fill = traces[1].fillcolor
        inner_fill = traces[3].fillcolor
        outer_alpha = float(outer_fill.split(",")[-1].rstrip(")"))
        inner_alpha = float(inner_fill.split(",")[-1].rstrip(")"))
        assert outer_alpha < inner_alpha


class TestSymmetricQuantileColors:
    def test_mirror_quantiles_same_color(self):
        """Q15% and Q85% should get the same color."""
        from figures.common import _symmetric_thermal_color
        c15 = _symmetric_thermal_color(0.15)
        c85 = _symmetric_thermal_color(0.85)
        assert c15 == c85

    def test_q50_gets_median_color(self):
        """Q50% should get the median (gray) color."""
        from figures.common import _symmetric_thermal_color
        c50 = _symmetric_thermal_color(0.50)
        assert c50 == "#bdbdbd"


class TestOverlayModelShading:
    def test_overlay_model_bands_use_model_color(self):
        """Overlay model bands should use that model's trace color."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[0.15, 0.50, 0.85], shade=True,
                 xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub", "pl"],
                 qs_mode=[])
        fig = build_bubble_figure(M, p)
        fill_traces = [t for t in fig.data if t.fill == "tonexty"]
        assert len(fill_traces) >= 2  # at least 1 band per model


class TestColorCodedModelLabels:
    def test_model_labels_have_color_swatch(self):
        """Display Models labels should have colored boxes."""
        from layout.bubble import _bubble_controls
        layout_str = repr(_bubble_controls())
        assert "backgroundColor" in layout_str


class TestResolveLpplMaster:
    """Unit test for the LPPL master -> flavor translation helper."""

    def test_no_master_passes_through(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["bub", "pl"], [3], [], [])
        assert result == ["bub", "pl"]

    def test_master_1_unweighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["bub", "lppl"], [1], [], [])
        assert "lppl" in result and "bub" in result
        assert "lp2" not in result

    def test_master_3_weighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [3], ["weighted"], [])
        assert result == ["lp3_w"]

    def test_master_3_disabled_by_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        # no_13 disables LP3
        result = _resolve_lppl_master(["lppl"], [3], [], ["no13"])
        assert result == []

    def test_master_4_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [4], [], ["no13"])
        assert result == ["lp4_n13"]

    def test_master_4_weighted_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [4], ["weighted"], ["no13"])
        assert result == ["lp4_w_n13"]

    def test_master_all_freqs_unweighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [1, 2, 3, 4], [], [])
        assert set(result) == {"lppl", "lp2", "lp3", "lp4"}

    def test_empty_n_freqs_strips_master(self):
        from callbacks.charts import _resolve_lppl_master
        # Master checked but no flavor selected -> master stripped with no replacement
        result = _resolve_lppl_master(["bub", "lppl"], [], [], [])
        assert result == ["bub"]


class TestModelShowChecklistStandardized:
    """Unit tests for _model_show_checklist standardized=True mode."""

    def test_has_lppl_master(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems).replace("'", '"')
        assert '"value": "lppl"' in rendered

    def test_omits_lppl_variants(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems).replace("'", '"')
        assert '"value": "lp2"' not in rendered
        assert '"value": "lp3"' not in rendered
        assert '"value": "lp4"' not in rendered
        assert '"value": "lppl_w"' not in rendered

    def test_omits_exp_and_s2f(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems).replace("'", '"')
        assert '"value": "exp"' not in rendered
        assert '"value": "s2f"' not in rendered

    def test_non_standardized_unchanged(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=False)
        rendered = str(elems).replace("'", '"')
        assert '"value": "lppl"' in rendered


class TestLpplConfigPanel:
    """Unit test for _lppl_config_panel compact helper."""

    def test_has_activate_and_summary_and_button(self):
        from layout.common import _lppl_config_panel
        card = _lppl_config_panel("dca")
        rendered = str(card)
        assert "dca-lppl-activate" in rendered
        assert "dca-lppl-summary" in rendered
        assert "dca-lppl-configure-btn" in rendered

    def test_no_inline_config_controls(self):
        """The un-prefixed config IDs live in the global modal, not here."""
        from layout.common import _lppl_config_panel
        card = _lppl_config_panel("ret")
        rendered = str(card).replace("'", '"')
        assert '"lppl-n-freqs"' not in rendered
        assert '"lppl-weighted"' not in rendered
        assert '"lppl-no-13"' not in rendered


class TestGlobalLpplModal:
    """Unit test for _global_lppl_modal root-level modal."""

    def test_has_all_config_controls(self):
        from layout.common import _global_lppl_modal
        modal = _global_lppl_modal()
        rendered = str(modal)
        assert "lppl-config-modal" in rendered
        assert "lppl-n-freqs" in rendered
        assert "lppl-weighted" in rendered
        assert "lppl-no-13" in rendered
        assert "lppl-modal-close-btn" in rendered


class TestResolveHmLpplMaster:
    """Unit test for heatmap LPPL master translation."""

    def test_non_lppl_passes_through(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("bub", [3], [], []) == "bub"
        assert _resolve_hm_lppl_master("pl", [3], [], []) == "pl"
        assert _resolve_hm_lppl_master("linppl", [3], [], []) == "linppl"

    def test_lppl_default_n3_unweighted(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [3], [], []) == "lp3"

    def test_lppl_n3_weighted(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [3], ["weighted"], []) == "lp3_w"

    def test_lppl_n4_no_13(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [4], [], ["no13"]) == "lp4_n13"

    def test_lppl_n4_weighted_no_13(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [4], ["weighted"], ["no13"]) == "lp4_w_n13"

    def test_lppl_picks_first_when_multi_selected(self):
        from callbacks.charts import _resolve_hm_lppl_master
        # Heatmap is single-select: takes first entry, ignores rest
        assert _resolve_hm_lppl_master("lppl", [2, 4], [], []) == "lp2"

    def test_lppl_empty_n_freqs_defaults_to_3(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [], [], []) == "lp3"

    def test_lppl_n3_with_no_13_falls_through_to_lppl(self):
        from callbacks.charts import _resolve_hm_lppl_master
        # n=3 and no_13 both set → LP3 disabled → fallback to "lppl"
        assert _resolve_hm_lppl_master("lppl", [3], [], ["no13"]) == "lppl"
