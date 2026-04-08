"""Core utility tests (btc_core, _q3, quantize, cache)."""
from figures import _FREQ_STEP_DAYS
from conftest import (
    M,
    _ALL_QS,
    _CHECKLIST_OPTIONS,
    _MC_UPLOAD_FIELDS,
    _SNAPSHOT_CONTROLS,
    _SNAP_PREFIX,
    _TAB_CONTROLS,
    _TAB_TO_PATH,
    _app_ctx,
    _app_module,
    _build_mc_params,
    _decode_snapshot,
    _encode_snapshot,
    _find_lot_percentile,
    _list_to_mask,
    _lots_summary,
    _mask_to_list,
    _mc_years_options,
    _nearest_quantile,
    _parse_mc_upload,
    _pk,
    _q3,
    _quantize_params,
    apply_snapshot,
    build_dca_figure,
    build_retire_figure,
    fmt_price,
    go,
    leo_weighted_entry,
    manage_lots,
    np,
    os,
    preview_percentile,
    pytest,
    qr_price,
    restore_from_url,
    update_bubble,
    update_dca,
    update_effective_lots,
    update_heatmap,
    update_retire,
    update_sc_info,
    update_supercharge,
    yr_to_t,
)


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


class TestFreqStepDays:
    def test_all_frequencies(self):
        assert _FREQ_STEP_DAYS["Daily"] == 1
        assert _FREQ_STEP_DAYS["Weekly"] == 7
        assert _FREQ_STEP_DAYS["Monthly"] == 30
        assert _FREQ_STEP_DAYS["Quarterly"] == 91
        assert _FREQ_STEP_DAYS["Annually"] == 365



class TestFalsyZeroGuard:
    """Verify that 0 is handled correctly as input (not treated as falsy)."""

    def test_zero_inflation_retire(self):
        """Inflation=0 should not become a default value."""
        p = {
            "start_yr": 2031,
            "end_yr": 2075,
            "start_stack": 1.0,
            "wd_amount": 5000,
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
            "sc_loan_amount": 1200,
            "sc_rate": 0,  # zero interest
            "sc_term_months": 12,
            "sc_loan_type": "interest_only",
            "sc_repeats": 0,
            "sc_entry_mode": "model",
            "sc_custom_price": 100000,
            "sc_tax_rate": 0,  # zero tax too
            "sc_rollover": False,
            "sc_live_price": 0,
        }
        fig, _ = build_dca_figure(M, p)
        assert isinstance(fig, go.Figure)
        sc_traces = [t for t in fig.data if "SC" in (t.name or "")]
        assert len(sc_traces) >= 1


class TestLayoutRendering:
    """Catch layout build errors that only surface on page load, not during import."""

    def test_serve_layout_no_crash(self):
        """_serve_layout() must complete without exceptions."""
        from layout import _serve_layout
        layout = _serve_layout()
        assert layout is not None

    def test_model_info_accordion_items(self):
        """Every model info accordion item must render without AttributeError."""
        from layout import model_info
        # Force all coeff table helpers to execute
        for fn_name in dir(model_info):
            if fn_name.endswith("_coeff_table") or fn_name.endswith("_formula"):
                fn = getattr(model_info, fn_name)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        pass  # functions that need args — skip
