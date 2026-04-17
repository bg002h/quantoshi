"""Callback smoke tests, MC cache, BTCPay, lots."""
import plotly.graph_objects as go
from mc_cache import (snap_to_bin, _path_key_str, _overlay_key_str,
                      CACHED_START_YRS, ENTRY_PCT_BINS, MC_YEARS_OPTIONS,
                      WD_AMOUNTS, INFL_OPTIONS, STACK_SIZES, FAN_PCTS,
                      is_cached, is_cached_year, _CACHED_MODEL_KEYS)
from mc_overlay import (_apply_bin_mask, _snap_start_pctile,
                        bin_regime_labels, _mc_path_key)
import btcpay
from conftest import (
    _MC_UPLOAD_FIELDS,
    _SNAPSHOT_CONTROLS,
    _app_ctx,
    _build_mc_params,
    _encode_snapshot,
    _lots_summary,
    _mc_years_options,
    _parse_mc_upload,
    _patch_ctx,
    _pk,
    _q3,
    apply_snapshot,
    base64,
    go,
    json,
    manage_lots,
    np,
    pd,
    preview_percentile,
    pytest,
    restore_from_url,
    update_bubble,
    update_dca,
    update_effective_lots,
    update_heatmap,
    update_retire,
    update_sc_info,
    update_supercharge,
)


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
        # _GHOST_BANDS is 2 bands (5-95%, 25-75%) × 2 traces each + 1 median = 5
        assert len(traces) == 5

    def test_no_median_when_disabled(self):
        ts = np.linspace(10, 20, 50)
        fan = {p: np.random.rand(50) * 100 for p in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)}
        traces = self._ghost(ts, fan, show_median=False)
        assert len(traces) == 4  # 2 bands × 2, no median

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
        # 2 ghost bands × 2 traces each + 1 median = 5
        assert len(traces) == 5

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
                lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[],
                hyb_a_nlog=1, hyb_a_ncal=1,
                hyb_a_log1d="d", hyb_a_log2d="d",
                hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0,
                hyb_b_log1d="d", hyb_b_log2d="d",
                hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1,
                ep_a_log1d="d", ep_a_log2d="d",
                ep_a_cal1d="u", ep_a_cal2d="u",
                ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0,
                ep_b_log1d="d", ep_b_log2d="d",
                ep_b_cal1d="u", ep_b_cal2d="u",
                decomp_model="", decomp_components=[], decomp_mode="individual",
                lots_data=[],
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
                lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[],
                hyb_a_nlog=1, hyb_a_ncal=1,
                hyb_a_log1d="d", hyb_a_log2d="d",
                hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0,
                hyb_b_log1d="d", hyb_b_log2d="d",
                hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1,
                ep_a_log1d="d", ep_a_log2d="d",
                ep_a_cal1d="u", ep_a_cal2d="u",
                ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0,
                ep_b_log1d="d", ep_b_log2d="d",
                ep_b_cal1d="u", ep_b_cal2d="u",
                decomp_model="", decomp_components=[], decomp_mode="individual",
                lots_data=[],
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
                lppl_n_freqs=[], lppl_weighted=[], lppl_no_13=[],
                hyb_a_nlog=1, hyb_a_ncal=1,
                hyb_a_log1d="d", hyb_a_log2d="d",
                hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0,
                hyb_b_log1d="d", hyb_b_log2d="d",
                hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1,
                ep_a_log1d="d", ep_a_log2d="d",
                ep_a_cal1d="u", ep_a_cal2d="u",
                ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0,
                ep_b_log1d="d", ep_b_log2d="d",
                ep_b_cal1d="u", ep_b_cal2d="u",
                decomp_model="", decomp_components=[], decomp_mode="individual",
                lots_data=[],
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
                sel_qs=[0.5], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
                hyb_a_nlog=1, hyb_a_ncal=1, hyb_a_log1d="d", hyb_a_log2d="d", hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0, hyb_b_log1d="d", hyb_b_log2d="d", hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1, ep_a_log1d="d", ep_a_log2d="d", ep_a_cal1d="u", ep_a_cal2d="u", ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0, ep_b_log1d="d", ep_b_log2d="d", ep_b_cal1d="u", ep_b_cal2d="u",
                _hybppl_commit=0, _eppl_commit=0,
                lots_data=[],
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
                sel_qs=[0.1, 0.5], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
                hyb_a_nlog=1, hyb_a_ncal=1, hyb_a_log1d="d", hyb_a_log2d="d", hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0, hyb_b_log1d="d", hyb_b_log2d="d", hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1, ep_a_log1d="d", ep_a_log2d="d", ep_a_cal1d="u", ep_a_cal2d="u", ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0, ep_b_log1d="d", ep_b_log2d="d", ep_b_cal1d="u", ep_b_cal2d="u",
                _hybppl_commit=0, _eppl_commit=0,
                lots_data=[],
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
                sel_qs=[0.5, 0.85], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
                hyb_a_nlog=1, hyb_a_ncal=1, hyb_a_log1d="d", hyb_a_log2d="d", hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0, hyb_b_log1d="d", hyb_b_log2d="d", hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1, ep_a_log1d="d", ep_a_log2d="d", ep_a_cal1d="u", ep_a_cal2d="u", ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0, ep_b_log1d="d", ep_b_log2d="d", ep_b_cal1d="u", ep_b_cal2d="u",
                _hybppl_commit=0, _eppl_commit=0,
                lots_data=[],
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
                sel_qs=[0.01, 0.1, 0.25], adv_qs=[], lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
                hyb_a_nlog=1, hyb_a_ncal=1, hyb_a_log1d="d", hyb_a_log2d="d", hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0, hyb_b_log1d="d", hyb_b_log2d="d", hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1, ep_a_log1d="d", ep_a_log2d="d", ep_a_cal1d="u", ep_a_cal2d="u", ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0, ep_b_log1d="d", ep_b_log2d="d", ep_b_cal1d="u", ep_b_cal2d="u",
                _hybppl_commit=0, _eppl_commit=0,
                lots_data=[],
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
                hyb_a_nlog=1, hyb_a_ncal=1, hyb_a_log1d="d", hyb_a_log2d="d", hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0, hyb_b_log1d="d", hyb_b_log2d="d", hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1, ep_a_log1d="d", ep_a_log2d="d", ep_a_cal1d="u", ep_a_cal2d="u", ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0, ep_b_log1d="d", ep_b_log2d="d", ep_b_cal1d="u", ep_b_cal2d="u",
                _hybppl_commit=0, _eppl_commit=0,
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
                hyb_a_nlog=1, hyb_a_ncal=1, hyb_a_log1d="d", hyb_a_log2d="d", hyb_a_cal1d="u", hyb_a_cal2d="u",
                hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0, hyb_b_log1d="d", hyb_b_log2d="d", hyb_b_cal1d="u", hyb_b_cal2d="u",
                ep_a_nlog=1, ep_a_ncal=1, ep_a_log1d="d", ep_a_log2d="d", ep_a_cal1d="u", ep_a_cal2d="u", ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0, ep_b_log1d="d", ep_b_log2d="d", ep_b_cal1d="u", ep_b_cal2d="u",
                _hybppl_commit=0, _eppl_commit=0,
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
        # BTCPay IDs are ~22 chars; regex requires >=16 to reject junk
        assert _INVOICE_ID_RE.match("X31XGHwugKcCpeF38GtGxM")
        assert _INVOICE_ID_RE.match("abc-123_DEF_016chars")

    def test_invalid_invoice_id_regex(self):
        from api import _INVOICE_ID_RE
        assert not _INVOICE_ID_RE.match("")
        assert not _INVOICE_ID_RE.match("../../../etc/passwd")
        assert not _INVOICE_ID_RE.match("id with spaces")
        assert not _INVOICE_ID_RE.match("a" * 65)  # too long
        assert not _INVOICE_ID_RE.match("short")  # <16 chars rejected

    def test_short_id_rejected(self):
        """Regex requires >=16 chars -- rejects obvious guesses."""
        from api import _INVOICE_ID_RE
        assert not _INVOICE_ID_RE.match("a")
        assert not _INVOICE_ID_RE.match("1")
        assert not _INVOICE_ID_RE.match("a" * 15)

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


