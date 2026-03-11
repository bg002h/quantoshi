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
                           update_effective_lots, restore_from_url,
                           _toggle_dca_sc_body, auto_bubble_yrange,
                           update_sc_info, toggle_sc_mode,
                           _TAB_CONTROLS, _TAB_TO_PATH)
    import _app_ctx
    _ALL_QS = _app_ctx._ALL_QS
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

from mc_overlay import (_mc_path_key, _mc_overlay_key, _mc_fan_to_lists,
                        _mc_fan_from_lists, _mc_paths_to_lists, _mc_paths_from_lists,
                        _MC_FAN_PCTS)
from figures import _apply_watermark, _interp_qr_price, _FREQ_STEP_DAYS
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
        """Unified key defaults mc_start_yr to 2026; callback sets tab-specific value."""
        key = _mc_path_key({}, "ret")
        assert key["mc_start_yr"] == 2026

    def test_sc_defaults(self):
        """Unified key defaults mc_start_yr to 2026; callback sets tab-specific value."""
        key = _mc_path_key({}, "sc")
        assert key["mc_start_yr"] == 2026

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
            mc_bins=None, mc_sims=None, mc_years=None,
            mc_freq=None, mc_window=None,
            mc_start_yr=None, mc_entry_q=None,
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
            mc_bins=None, mc_sims=None, mc_years=None,
            mc_freq=None, mc_window=None,
            mc_start_yr=None, mc_entry_q=None,
            mc_cached=None, mc_live_price=0,
            amount_default=5000, infl_default=0.0, start_yr_default=2031,
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
        """800 sims × Monthly must allow all 4 year options (10, 20, 30, 40)."""
        opts = _mc_years_options(800, "Monthly")
        values = [o["value"] for o in opts]
        assert values == [10, 20, 30, 40]


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
        assert abs(y_vals[-1] - expected) < 1e-8

    def test_start_stack_offset(self):
        """Starting stack should shift all values by a constant."""
        fig0, _ = build_dca_figure(M, self._dca_params(start_stack=0.0))
        fig1, _ = build_dca_figure(M, self._dca_params(start_stack=2.5))
        final0 = fig0.data[0].y[-1]
        final1 = fig1.data[0].y[-1]
        assert abs(final1 - final0 - 2.5) < 1e-8

    def test_usd_mode_equals_btc_times_price(self):
        """USD display mode = BTC balance × final price."""
        fig_btc, _ = build_dca_figure(M, self._dca_params(disp_mode="btc"))
        fig_usd, _ = build_dca_figure(M, self._dca_params(disp_mode="usd"))
        btc_final = fig_btc.data[0].y[-1]
        usd_final = fig_usd.data[0].y[-1]
        t_end = yr_to_t(2031, M.genesis)
        ts = np.arange(max(yr_to_t(2030, M.genesis), 1.0), t_end + (1 / 12) * 0.5, 1 / 12)
        final_price = float(qr_price(_Q50, max(ts[-1], 0.5), M.qr_fits))
        assert abs(usd_final - btc_final * final_price) < 1.0

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
        # Manual
        t_start = max(yr_to_t(2030, M.genesis), 1.0)
        t_end = yr_to_t(2035, M.genesis)
        ts = np.arange(t_start, t_end + 0.5, 1.0)
        stack = 1.0
        for i, t in enumerate(ts):
            price = float(qr_price(_Q50, max(t, 0.5), M.qr_fits))
            stack -= 50000.0 / price
            stack = max(stack, 0.0)
            assert abs(y_vals[i] - stack) < 1e-8

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
                and abs(ay - float(list(lt.y)[-1])) < 1e-6
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
    """Context manager that patches dash.ctx and dash.callback_context."""
    ctx_obj = _CallbackCtx(triggered_id)
    return patch.multiple("callbacks", ctx=ctx_obj)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateBubbleCallback:
    """Smoke-test the update_bubble callback."""

    def test_returns_figure(self):
        with _patch_ctx("bub-qs"):
            fig = update_bubble(
                sel_qs=[0.5], toggles=["show_data", "show_today"],
                bubble_toggles=[], xscale="log", yscale="log",
                xrange=[2012, 2030], yrange=[0, 7],
                n_future=3, ptsize=3, ptalpha=0.6,
                stack=0, show_stack=[], use_lots=[], legend_pos="outside", lots_data=[],
            )
        assert isinstance(fig, go.Figure)

    def test_empty_quantiles(self):
        with _patch_ctx("bub-qs"):
            fig = update_bubble(
                sel_qs=[], toggles=[], bubble_toggles=[],
                xscale="linear", yscale="log",
                xrange=[2015, 2028], yrange=[1, 6],
                n_future=0, ptsize=2, ptalpha=0.3,
                stack=0, show_stack=[], use_lots=[], legend_pos="outside", lots_data=[],
            )
        assert isinstance(fig, go.Figure)

    def test_with_stack(self):
        with _patch_ctx("bub-stack"):
            fig = update_bubble(
                sel_qs=[0.1, 0.5, 0.9], toggles=["show_legend"],
                bubble_toggles=["show_comp"], xscale="log", yscale="log",
                xrange=[2012, 2035], yrange=[0, 7],
                n_future=2, ptsize=4, ptalpha=0.5,
                stack=1.5, show_stack=["yes"], use_lots=[], legend_pos="outside", lots_data=[],
            )
        assert isinstance(fig, go.Figure)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateHeatmapCallback:
    """Smoke-test the update_heatmap callback."""

    def test_returns_figure(self):
        yr = pd.Timestamp.today().year
        with _patch_ctx("hm-entry-yr"):
            result = update_heatmap(
                active_tab="heatmap", entry_yr=yr, entry_q=50.0,
                exit_range=[yr, yr + 10],
                exit_qs=[0.01, 0.1, 0.5, 0.85, 0.99],
                mode=0, b1=0, b2=20,
                c_lo=None, c_mid1=None, c_mid2=None, c_hi=None,
                grad=32, vfmt="cagr", cell_fs=9,
                toggles=["colorbar"], stack=0, use_lots=[],
                lots_data=[],
                mc_enable=[], mc_amount=100, mc_infl=0,
                mc_bins=5, mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=[2010, yr],
                mc_start_yr=yr, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0,
                live_price=0, mc_cached=None, pay_token=None,
            )
        # Returns 9 outputs: qr_fig, mc_fig, store, status, panel_style, indicator_style, rendered_key, modal, tab
        assert len(result) == 9
        assert isinstance(result[0], go.Figure)

    def test_wrong_tab_prevents_update(self):
        with _patch_ctx("main-tabs"):
            with pytest.raises(Exception):  # dash.exceptions.PreventUpdate
                update_heatmap(
                    active_tab="dca", entry_yr=2025, entry_q=50,
                    exit_range=[2025, 2035], exit_qs=[0.5],
                    mode=0, b1=0, b2=20,
                    c_lo=None, c_mid1=None, c_mid2=None, c_hi=None,
                    grad=32, vfmt="cagr", cell_fs=9,
                    toggles=[], stack=0, use_lots=[], lots_data=[],
                    mc_enable=[], mc_amount=100, mc_infl=0,
                    mc_bins=5, mc_sims=800, mc_years=10,
                    mc_freq="Monthly", mc_window=None,
                    mc_start_yr=2025, mc_entry_q=50,
                    _mc_loaded=None, _pay_trigger=0,
                    live_price=0, mc_cached=None, pay_token=None,
                )


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateDcaCallback:
    """Smoke-test the update_dca callback."""

    def test_returns_figure_and_mc_outputs(self):
        with _patch_ctx("dca-amount"):
            result = update_dca(
                active_tab="dca", stack=0, use_lots=[], amount=200,
                freq="Monthly", yr_range=[2025, 2035],
                disp="btc", toggles=["show_legend"], legend_pos="outside",
                sel_qs=[0.5], lots_data=[],
                sc_enable=[], sc_loan=0, sc_rate=13, sc_term=12,
                sc_type="interest_only", sc_repeats=0,
                sc_entry_mode="live", sc_custom_price=80000,
                sc_tax=33, sc_rollover=[],
                mc_enable=[], mc_amount=100, mc_infl=4,
                mc_bins=5, mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=None,
                mc_start_yr=2026, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0,
                price_data=0, mc_cached=None, pay_token=None,
            )
        # 6 outputs: fig, mc_results, mc_status, rendered_key, mc_modal, mc_tab
        assert len(result) == 6
        assert isinstance(result[0], go.Figure)

    def test_wrong_tab_prevents_update(self):
        with _patch_ctx("main-tabs"):
            with pytest.raises(Exception):
                update_dca(
                    active_tab="bubble", stack=0, use_lots=[], amount=200,
                    freq="Monthly", yr_range=[2025, 2035],
                    disp="btc", toggles=[], legend_pos="outside", sel_qs=[0.5], lots_data=[],
                    sc_enable=[], sc_loan=0, sc_rate=13, sc_term=12,
                    sc_type="interest_only", sc_repeats=0,
                    sc_entry_mode="live", sc_custom_price=80000,
                    sc_tax=33, sc_rollover=[],
                    mc_enable=[], mc_amount=100, mc_infl=4,
                    mc_bins=5, mc_sims=800, mc_years=10,
                    mc_freq="Monthly", mc_window=None,
                    mc_start_yr=2026, mc_entry_q=50,
                    _mc_loaded=None, _pay_trigger=0,
                price_data=0, mc_cached=None, pay_token=None,
                )

    def test_with_sc_enabled(self):
        with _patch_ctx("dca-sc-enable"):
            result = update_dca(
                active_tab="dca", stack=0, use_lots=[], amount=500,
                freq="Monthly", yr_range=[2025, 2030],
                disp="btc", toggles=[], legend_pos="outside",
                sel_qs=[0.1, 0.5], lots_data=[],
                sc_enable=["yes"], sc_loan=10000, sc_rate=13, sc_term=12,
                sc_type="interest_only", sc_repeats=0,
                sc_entry_mode="custom", sc_custom_price=90000,
                sc_tax=33, sc_rollover=[],
                mc_enable=[], mc_amount=100, mc_infl=4,
                mc_bins=5, mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=None,
                mc_start_yr=2026, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0,
                price_data=0, mc_cached=None, pay_token=None,
            )
        assert isinstance(result[0], go.Figure)

    def test_usd_display_mode(self):
        with _patch_ctx("dca-disp"):
            result = update_dca(
                active_tab="dca", stack=0.5, use_lots=[], amount=300,
                freq="Weekly", yr_range=[2025, 2032],
                disp="usd", toggles=["log_y"], legend_pos="outside",
                sel_qs=[0.5, 0.85], lots_data=[],
                sc_enable=[], sc_loan=0, sc_rate=13, sc_term=12,
                sc_type="interest_only", sc_repeats=0,
                sc_entry_mode="live", sc_custom_price=80000,
                sc_tax=33, sc_rollover=[],
                mc_enable=[], mc_amount=100, mc_infl=4,
                mc_bins=5, mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=None,
                mc_start_yr=2026, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0,
                price_data=0, mc_cached=None, pay_token=None,
            )
        assert isinstance(result[0], go.Figure)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateRetireCallback:
    """Smoke-test the update_retire callback."""

    def test_returns_figure(self):
        with _patch_ctx("ret-wd"):
            result = update_retire(
                active_tab="retire", stack=1.0, use_lots=[], wd=5000,
                freq="Monthly", yr_range=[2031, 2075], infl=4,
                disp="btc", toggles=["log_y", "annotate"],
                legend_pos="outside",
                sel_qs=[0.01, 0.1, 0.25], lots_data=[],
                mc_enable=[], mc_amount=5000, mc_infl=4,
                mc_bins=5, mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=None,
                mc_stack=1.0, mc_start_yr=2031, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0,
                price_data=0, mc_cached=None, pay_token=None,
            )
        # 6 outputs: fig, mc_results, mc_status, rendered_key, mc_modal, mc_tab
        assert len(result) == 6
        assert isinstance(result[0], go.Figure)


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestUpdateSuperchargeCallback:
    """Smoke-test the update_supercharge callback."""

    def test_mode_a(self):
        with _patch_ctx("sc-mode"):
            result = update_supercharge(
                active_tab="supercharge", stack=1.0, use_lots=[],
                start_yr=2033, d0=0, d1=0, d2=0, d3=1, d4=2,
                freq="Annually", infl=4, sel_qs=[0.001, 0.1],
                mode="a", wd=100000, end_yr=2075, target_yr=2060,
                disp="usd",
                toggles=["annotate", "log_y", "show_legend"], legend_pos="outside",
                chart_layout=["shade"], display_q=0.5, lots_data=[],
                mc_enable=[], mc_amount=5000, mc_infl=4,
                mc_bins=5, mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=None,
                mc_stack=1.0, mc_start_yr=2031, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0,
                price_data=0, mc_cached=None, pay_token=None,
            )
        assert len(result) == 5
        assert isinstance(result[0], go.Figure)

    def test_mode_b(self):
        with _patch_ctx("sc-mode"):
            result = update_supercharge(
                active_tab="supercharge", stack=2.0, use_lots=[],
                start_yr=2030, d0=0, d1=1, d2=3, d3=5, d4=10,
                freq="Monthly", infl=3, sel_qs=[0.1, 0.5],
                mode="b", wd=50000, end_yr=2080, target_yr=2055,
                disp="usd", toggles=["show_legend"], legend_pos="outside",
                chart_layout=[], display_q=0.5, lots_data=[],
                mc_enable=[], mc_amount=5000, mc_infl=4,
                mc_bins=5, mc_sims=800, mc_years=10,
                mc_freq="Monthly", mc_window=None,
                mc_stack=1.0, mc_start_yr=2031, mc_entry_q=50,
                _mc_loaded=None, _pay_trigger=0,
                price_data=0, mc_cached=None, pay_token=None,
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
class TestToggleCallbacks:
    def test_sc_body_visible(self):
        assert _toggle_dca_sc_body(["yes"]) == {}

    def test_sc_body_hidden(self):
        assert _toggle_dca_sc_body([]) == {"display": "none"}

    def test_sc_body_falsy(self):
        assert _toggle_dca_sc_body(None) == {"display": "none"}


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
        result = restore_from_url("")
        # All no_update
        assert len(result) == len(_SNAPSHOT_CONTROLS) + 2

    def test_none_hash(self):
        result = restore_from_url(None)
        assert len(result) == len(_SNAPSHOT_CONTROLS) + 2

    def test_invalid_prefix(self):
        result = restore_from_url("#garbage")
        assert len(result) == len(_SNAPSHOT_CONTROLS) + 2

    def test_valid_roundtrip(self):
        """Encode a snapshot, then decode via restore_from_url."""
        state = {
            "bub-xscale:value": "log",
            "bub-yscale:value": "log",
            "main-tabs:active_tab": "bubble",
            "bub-qs:value": [0.5],
        }
        encoded = _encode_snapshot(state)
        hash_str = f"#q2:{encoded}"
        result = restore_from_url(hash_str)
        assert len(result) == len(_SNAPSHOT_CONTROLS) + 2
        # main-tabs should be restored
        main_tab_idx = next(i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
                           if cid == "main-tabs")
        assert result[main_tab_idx] == "bubble"
        # loaded-hash-store should be set
        assert result[-1] == hash_str


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestAutoBubbleYrange:
    def test_no_auto_prevents_update(self):
        with _patch_ctx("bub-xrange"):
            with pytest.raises(Exception):
                auto_bubble_yrange(
                    xrange=[2015, 2030], auto_y=[], yscale="log", sel_qs=[0.5],
                )

    def test_returns_yrange(self):
        import math as _m
        with _patch_ctx("bub-xrange"):
            result = auto_bubble_yrange(
                xrange=[2015, 2030], auto_y=["yes"], yscale="log", sel_qs=[0.5],
            )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] < result[1]


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestToggleScMode:
    def test_mode_a(self):
        result = toggle_sc_mode("a")
        assert result == (True, False, True)

    def test_mode_b(self):
        result = toggle_sc_mode("b")
        assert result == (False, True, False)


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
        """All quantiles selected → roundtrip through encode/decode."""
        all_qs = list(_ALL_QS)
        state = {"bub-qs:value": all_qs, "main-tabs:active_tab": "bubble"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        restored = decoded.get("bub-qs:value", [])
        assert set(restored) == set(all_qs)

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

    def test_dca_cached_10yr(self):
        assert btcpay.compute_price("dca", 10, True) == 100

    def test_dca_live_10yr(self):
        assert btcpay.compute_price("dca", 10, False) == 500

    def test_hm_discount(self):
        assert btcpay.compute_price("hm", 10, True) == 50

    def test_hm_live_40yr(self):
        assert btcpay.compute_price("hm", 40, False) == 1000

    def test_ret_cached_20yr(self):
        assert btcpay.compute_price("ret", 20, True) == 200

    def test_all_horizons(self):
        for yrs in (10, 20, 30, 40):
            c = btcpay.compute_price("dca", yrs, True)
            l = btcpay.compute_price("dca", yrs, False)
            assert l > c, f"{yrs}yr: live ({l}) should exceed cached ({c})"

    def test_free_tier_dca_default(self):
        assert btcpay.is_free_tier(10, 2026, 50)

    def test_free_tier_hm_default(self):
        assert btcpay.is_free_tier(10, 2026, 10)

    def test_free_tier_retire_default(self):
        assert btcpay.is_free_tier(10, 2031, 50)

    def test_free_tier_cache_aligned_entry_q(self):
        """Cache-aligned entry percentiles (10% bins) are free."""
        assert btcpay.is_free_tier(10, 2026, 20)
        assert btcpay.is_free_tier(10, 2026, 90)
        assert btcpay.is_free_tier(20, 2028, 40)
        assert btcpay.is_free_tier(20, 2031, 70)

    def test_not_free_non_aligned_entry_q(self):
        """Non-cache-aligned entry percentiles require payment."""
        assert not btcpay.is_free_tier(10, 2026, 25)
        assert not btcpay.is_free_tier(10, 2026, 4.3)
        assert not btcpay.is_free_tier(20, 2028, 5)
        assert not btcpay.is_free_tier(20, 2031, 75)

    def test_free_tier_20yr(self):
        assert btcpay.is_free_tier(20, 2026, 50)
        assert btcpay.is_free_tier(20, 2028, 10)
        assert btcpay.is_free_tier(20, 2031, 50)

    def test_free_tier_2028(self):
        assert btcpay.is_free_tier(10, 2028, 50)

    def test_not_free_30yr(self):
        assert not btcpay.is_free_tier(30, 2026, 50)

    def test_not_free_uncovered_yr(self):
        assert not btcpay.is_free_tier(10, 2035, 50)

    def test_is_cached_request(self):
        assert btcpay.is_cached_request(2026)
        assert btcpay.is_cached_request(2031)
        assert not btcpay.is_cached_request(2027)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
