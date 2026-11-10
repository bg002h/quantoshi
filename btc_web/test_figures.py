"""Figure builders, fuzz tests, financial math."""
import plotly.graph_objects as go
import pytest
from btc_core import BubbleModel
from figures import (_apply_watermark, _FREQ_STEP_DAYS,
                     build_bubble_figure, build_heatmap_figure,
                     build_dca_figure, build_retire_figure,
                     build_supercharge_figure,
                     _build_qr_config_text, _build_mc_config_text,
                     _apply_config_annotation)
from mc_overlay import (_mc_path_key, _mc_overlay_key, _mc_fan_to_lists,
                        _mc_fan_from_lists, _mc_paths_to_lists, _mc_paths_from_lists,
                        _MC_FAN_PCTS)
from conftest import (
    M,
    _FUZZ_Q1,
    _FUZZ_Q2,
    _FUZZ_QS,
    _Q50,
    _YR_NOW,
    _app_ctx,
    _assert_figure,
    _decode_snapshot,
    _encode_snapshot,
    _q3,
    build_bubble_figure,
    build_dca_figure,
    build_heatmap_figure,
    build_retire_figure,
    build_supercharge_figure,
    go,
    json,
    math,
    np,
    pd,
    pytest,
    qr_price,
    yr_to_t,
)


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
        fig, _ = build_bubble_figure(M, p)
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
        fig, _ = build_bubble_figure(M, p)
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
            "active_models": ["bub"],
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
        fig, _ = build_bubble_figure(M, p)
        _assert_figure(fig)


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
            "active_models": ["bub"],
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
    """Verify Stack-celerator loan cap via _compute_sc_loan."""

    def test_interest_only_cap(self):
        """Interest-only: principal capped at amount/r."""
        from _app_ctx import _compute_sc_loan
        principal, pmt, capped = _compute_sc_loan(
            principal=999_999, amount=500, r=0.01, term_periods=12,
            loan_type="interest_only")
        assert capped
        assert abs(principal - 50_000) < 0.01
        assert abs(pmt - 500) < 0.01

    def test_amortizing_cap(self):
        """Amortizing: principal capped so pmt == amount."""
        from _app_ctx import _compute_sc_loan
        principal, pmt, capped = _compute_sc_loan(
            principal=999_999, amount=500, r=0.01, term_periods=12,
            loan_type="amortizing")
        assert capped
        assert abs(pmt - 500) < 0.01

    def test_cap_prevents_negative_dca(self):
        """Huge loan should be capped; SC trace still generated."""
        p = {
            "start_yr": 2030, "end_yr": 2031,
            "start_stack": 0.0, "amount": 100, "freq": "Monthly",
            "disp_mode": "btc", "selected_qs": [_Q50],
            "active_models": ["bub"],
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
            "active_models": ["bub"],
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
    """Verify tax-on-gain behavior via _compute_sc_loan + build_dca_figure."""

    def _sc_params(self, tax_rate=0.33):
        _Q50 = min(M.qr_fits, key=lambda q: abs(q - 0.5))
        return {
            "start_yr": 2030, "end_yr": 2031,
            "start_stack": 0.0, "amount": 100, "freq": "Monthly",
            "disp_mode": "btc", "selected_qs": [_Q50],
            "active_models": ["bub"],
            "log_y": False, "show_today": False, "show_legend": True,
            "lots": [], "use_lots": False,
            "sc_enabled": True,
            "sc_loan_amount": 5000, "sc_rate": 12.0,
            "sc_term_months": 12, "sc_loan_type": "interest_only",
            "sc_repeats": 0, "sc_entry_mode": "model",
            "sc_custom_price": 0, "sc_tax_rate": tax_rate,
            "sc_rollover": False, "sc_live_price": 0,
        }

    def test_with_tax(self):
        """SC with 33% tax should produce SC traces."""
        fig, _ = build_dca_figure(M, self._sc_params(0.33))
        sc_traces = [t for t in fig.data if "SC" in (t.name or "")]
        assert len(sc_traces) >= 1

    def test_zero_tax(self):
        """SC with 0% tax should produce SC traces."""
        fig, _ = build_dca_figure(M, self._sc_params(0.0))
        sc_traces = [t for t in fig.data if "SC" in (t.name or "")]
        assert len(sc_traces) >= 1

    def test_loan_cap_not_capped(self):
        """Small loan at 12% should not be capped."""
        from _app_ctx import _compute_sc_loan
        _, _, capped = _compute_sc_loan(
            principal=5000, amount=100, r=0.01, term_periods=12,
            loan_type="interest_only")
        assert not capped

    def test_zero_rate_no_cap(self):
        """0% interest → no cap, payment = 0 for interest-only."""
        from _app_ctx import _compute_sc_loan
        principal, pmt, capped = _compute_sc_loan(
            principal=5000, amount=100, r=0, term_periods=12,
            loan_type="interest_only")
        assert not capped
        assert principal == 5000
        assert pmt == 0



class TestRetireMath:
    """Verify retirement depletion arithmetic."""

    def _retire_params(self, **overrides):
        p = {
            "start_yr": 2030, "end_yr": 2035,
            "start_stack": 1.0, "wd_amount": 50000, "freq": "Annually",
            "inflation": 0, "disp_mode": "btc",
            "selected_qs": [_Q50], "active_models": ["bub"],
            "log_y": False, "show_legend": True, "annotate": False,
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
                 disp_mode="btc", selected_qs=qs, active_models=["bub"],
                 log_y=True, annotate=True,
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
        fig, _ = build_bubble_figure(M, p)
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



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestAutoYGrid:
    """Auto-Y is now clientside — test the pre-computed grid data."""

    def test_grid_has_all_models(self):
        import _app_ctx
        grid = _app_ctx.AUTO_Y_GRID
        assert "t" in grid
        assert "models" in grid
        for key in ("bub", "pl", "lppl"):
            assert key in grid["models"], f"{key} missing from auto-y grid"

    def test_grid_envelope_shape(self):
        import _app_ctx
        grid = _app_ctx.AUTO_Y_GRID
        t_len = len(grid["t"])
        for key, env in grid["models"].items():
            assert len(env["lo"]) == t_len, f"{key} lo length mismatch"
            assert len(env["hi"]) == t_len, f"{key} hi length mismatch"

    def test_grid_hi_ge_lo(self):
        import _app_ctx
        grid = _app_ctx.AUTO_Y_GRID
        for key, env in grid["models"].items():
            for i in range(len(env["lo"])):
                assert env["hi"][i] >= env["lo"][i] - 0.01, (
                    f"{key} hi < lo at index {i}")


def test_add_mc_spaghetti_returns_n_traces():
    """100 sample paths from a 2000-path array, deterministic stride."""
    from figures.bubble import _add_mc_spaghetti
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    n_sims, n_steps = 2000, 60
    rng = np.random.default_rng(0)
    paths = np.cumsum(rng.normal(0, 0.05, size=(n_sims, n_steps)), axis=1)
    t_axis = np.arange(n_steps)

    n_initial = len(fig.data)
    _add_mc_spaghetti(fig, paths, t_axis, n_display=100)
    n_final = len(fig.data)
    assert n_final - n_initial == 100, \
        f"expected 100 new traces, got {n_final - n_initial}"


def test_add_mc_spaghetti_single_neutral_color():
    """Every example path is drawn in ONE dark neutral (colors.MC_PATH_COLOR).

    Was an RdYlGn gradient keyed on the terminal value; the pale-yellow middle
    of that scale vanished on the cream plot background and red/green is the
    axis a deuteranomalous reader cannot separate.
    """
    from figures.bubble import _add_mc_spaghetti
    from colors import MC_PATH_COLOR, _hex_to_rgb
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    paths = np.array([
        [1.0, 1.0, 1.0, 1.0],   # lowest final
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 3.0, 5.0, 7.0],   # highest final
    ])
    _add_mc_spaghetti(fig, paths, np.arange(4), n_display=3)
    colors = {t.line.color for t in fig.data}
    assert len(fig.data) == 3
    assert len(colors) == 1, f"expected one shared colour, got {colors}"
    r, g, b = _hex_to_rgb(MC_PATH_COLOR)
    assert colors.pop().startswith(f"rgba({r},{g},{b},"), \
        "example paths must use colors.MC_PATH_COLOR"
    # dark + neutral: low luminance, no hue (r == g == b within a few counts)
    assert max(r, g, b) <= 0x60, "MC_PATH_COLOR must be dark"
    assert max(r, g, b) - min(r, g, b) <= 8, "MC_PATH_COLOR must be neutral"


def test_add_mc_spaghetti_paths_stay_thin():
    """Example paths are hairlines flagged to KEEP their width.

    assets/chart_responsive.js restyles every line trace to the user's trace
    width (3.75 px on desktop). Applied to the example paths that buries the
    fan under a black slab, so each path carries meta.qs_keep_width and the
    script leaves it alone (test_chart_responsive.py covers the JS half).
    """
    from figures.bubble import _add_mc_spaghetti
    from colors import MC_PATH_WIDTH, TRACE_WIDTH
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    _add_mc_spaghetti(fig, np.ones((5, 4)), np.arange(4), n_display=5)

    assert len(fig.data) == 5
    assert 0 < MC_PATH_WIDTH <= TRACE_WIDTH / 2
    for t in fig.data:
        assert t.line.width == MC_PATH_WIDTH
        assert t.meta == {"qs_keep_width": True}


def _path_alpha(trace):
    return float(trace.line.color.rstrip(")").split(",")[-1])


def test_mc_path_alpha_fades_as_lines_are_added():
    """A few example paths are distinct lines; 200 of them are a density cloud.

    At a fixed alpha 200 paths paint a black slab over the fan. The summed
    alpha is capped at colors.MC_PATH_ALPHA_BUDGET instead, so the fan's amber
    shows through however many paths are drawn.
    """
    from figures.bubble import _mc_path_alpha
    from colors import MC_PATH_ALPHA, MC_PATH_ALPHA_BUDGET

    assert _mc_path_alpha(1) == MC_PATH_ALPHA
    assert _mc_path_alpha(8) == MC_PATH_ALPHA
    assert _mc_path_alpha(200) == pytest.approx(0.04)
    counts = (1, 8, 16, 32, 64, 128, 200, 400, 800, 1600, 3200)
    alphas = [_mc_path_alpha(n) for n in counts]
    assert alphas == sorted(alphas, reverse=True)
    assert all(a > 0 for a in alphas)
    for n, a in zip(counts, alphas):
        assert n * a <= MC_PATH_ALPHA_BUDGET + 1e-9


def test_add_mc_spaghetti_uses_the_count_dependent_alpha():
    from figures.bubble import _add_mc_spaghetti, _mc_path_alpha
    import numpy as np
    import plotly.graph_objects as go

    for n in (8, 200):
        fig = go.Figure()
        _add_mc_spaghetti(fig, np.ones((200, 4)), np.arange(4), n_display=n)
        assert len(fig.data) == n
        alphas = {_path_alpha(t) for t in fig.data}
        assert len(alphas) == 1
        assert alphas.pop() == pytest.approx(_mc_path_alpha(n), abs=5e-4)


def test_add_mc_spaghetti_ships_float32():
    """200 paths are the bulk of the Tab-1 MC payload; Dash sends numpy arrays
    as base64 of the raw bytes, so float32 halves it (rounding would not)."""
    from figures.bubble import _add_mc_spaghetti
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    paths = 50_000.0 * np.exp(np.linspace(0, 3, 12)).reshape(3, 4)
    t_axis = np.linspace(18.5, 22.5, 4)
    _add_mc_spaghetti(fig, paths, t_axis, n_display=3)
    for t, row in zip(fig.data, paths):
        assert t.x.dtype == np.float32 and t.y.dtype == np.float32
        np.testing.assert_allclose(t.y, row, rtol=1e-6)
        np.testing.assert_allclose(t.x, t_axis, rtol=1e-6)


def test_tab1_default_sims_is_the_whole_cached_scenario():
    """Tab 1 opens on the density-cloud look: all 200 cached paths."""
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    from tab_defaults import bubble_defaults
    from layout.bubble import MC_DEFAULT_SIMS

    assert MC_DEFAULT_SIMS == 200
    assert SNAPSHOT_DEFAULTS["bub-mc-sims:value"] == MC_DEFAULT_SIMS
    assert bubble_defaults()["mc_sims"] == MC_DEFAULT_SIMS


def test_add_mc_spaghetti_handles_empty_paths():
    """None / empty array → no-op, no exception."""
    from figures.bubble import _add_mc_spaghetti
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    _add_mc_spaghetti(fig, None, np.arange(5))
    _add_mc_spaghetti(fig, np.array([]).reshape(0, 5), np.arange(5))
    assert len(fig.data) == 0


# ── MC price fan on Tab 1 (_add_mc_fan) ──────────────────────────────────────

def _fan_fixture(n_sims=200, n_steps=48, seed=3):
    import numpy as np
    rng = np.random.default_rng(seed)
    paths = 50_000.0 * np.exp(
        np.cumsum(rng.normal(0.01, 0.08, size=(n_sims, n_steps)), axis=1))
    t_axis = np.linspace(18.5, 22.5, n_steps)   # years since genesis
    return paths, t_axis


def _by_name(fig):
    return {t.name: t for t in fig.data if t.name}


def test_add_mc_fan_draws_bands_and_median_from_all_paths():
    """5–95% and 25–75% bands plus a median, each the per-step percentile of
    EVERY path handed in (not of a display subsample)."""
    from figures.bubble import _add_mc_fan
    import numpy as np
    import plotly.graph_objects as go

    paths, t_axis = _fan_fixture()
    fig = go.Figure()
    _add_mc_fan(fig, paths, t_axis)
    named = _by_name(fig)
    assert set(named) == {"MC 5\u201395%", "MC 25\u201375%", "MC median"}
    np.testing.assert_allclose(
        np.asarray(named["MC median"].y, dtype=float),
        np.percentile(paths, 50, axis=0), rtol=1e-6)
    # each band = an unnamed upper edge followed by the named lower edge
    # filled up to it
    data = list(fig.data)
    for name, lo, hi in (("MC 5\u201395%", 5, 95), ("MC 25\u201375%", 25, 75)):
        i = data.index(named[name])
        assert data[i].fill == "tonexty"
        np.testing.assert_allclose(np.asarray(data[i].y, dtype=float),
                                   np.percentile(paths, lo, axis=0), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(data[i - 1].y, dtype=float),
                                   np.percentile(paths, hi, axis=0), rtol=1e-6)
    for t in fig.data:
        np.testing.assert_allclose(np.asarray(t.x, dtype=float), t_axis)


def test_add_mc_fan_is_amber_with_tab1_strength_fills():
    """Same amber and dotted median as the MC fan on the other tabs, but the
    fills use the Tab-1 alphas: at Tab 1's zoom (8 decades of price in one
    graph) the shared 0.08 / 0.15 fills on a 14-px band are invisible."""
    from figures.bubble import _add_mc_fan
    from colors import (MC_AMBER, MC_PRICE_BAND_OUTER_ALPHA,
                        MC_PRICE_BAND_INNER_ALPHA, MC_MEDIAN_ALPHA, _hex_alpha)
    import plotly.graph_objects as go

    paths, t_axis = _fan_fixture()
    fig = go.Figure()
    _add_mc_fan(fig, paths, t_axis)
    named = _by_name(fig)
    assert named["MC 5\u201395%"].fillcolor == _hex_alpha(MC_AMBER, MC_PRICE_BAND_OUTER_ALPHA)
    assert named["MC 25\u201375%"].fillcolor == _hex_alpha(MC_AMBER, MC_PRICE_BAND_INNER_ALPHA)
    assert named["MC median"].line.color == _hex_alpha(MC_AMBER, MC_MEDIAN_ALPHA)
    assert named["MC median"].line.dash == "dot"
    assert all(t.showlegend for t in named.values())


def test_tab1_fan_fills_are_stronger_than_the_shared_ones_and_graduated():
    from colors import (MC_BAND_OUTER_ALPHA, MC_BAND_INNER_ALPHA,
                        MC_PRICE_BAND_OUTER_ALPHA, MC_PRICE_BAND_INNER_ALPHA)
    assert MC_PRICE_BAND_OUTER_ALPHA >= 2 * MC_BAND_OUTER_ALPHA
    assert MC_PRICE_BAND_INNER_ALPHA >= 2 * MC_BAND_INNER_ALPHA
    assert MC_PRICE_BAND_OUTER_ALPHA < MC_PRICE_BAND_INNER_ALPHA < 0.5, \
        "outer lighter than inner, and the price line stays readable through it"


def test_tab1_fan_does_not_restyle_the_shared_band_lists():
    """The Tab-1 alphas are passed per call; the fan on the other tabs
    (mc_overlay._MC_BANDS / _GHOST_BANDS) keeps its deliberate 0.08 / 0.15."""
    from figures.bubble import _add_mc_fan
    import mc_overlay
    from colors import (MC_AMBER, MC_GHOST_GRAY, MC_BAND_OUTER_ALPHA,
                        MC_BAND_INNER_ALPHA, MC_GHOST_OUTER_ALPHA,
                        MC_GHOST_INNER_ALPHA, _hex_alpha)
    import plotly.graph_objects as go

    paths, t_axis = _fan_fixture()
    _add_mc_fan(go.Figure(), paths, t_axis)
    _add_mc_fan(go.Figure(), paths, t_axis, reference=True)
    assert [b[2] for b in mc_overlay._MC_BANDS] == [
        _hex_alpha(MC_AMBER, MC_BAND_OUTER_ALPHA),
        _hex_alpha(MC_AMBER, MC_BAND_INNER_ALPHA)]
    assert [b[2] for b in mc_overlay._GHOST_BANDS] == [
        _hex_alpha(MC_GHOST_GRAY, MC_GHOST_OUTER_ALPHA),
        _hex_alpha(MC_GHOST_GRAY, MC_GHOST_INNER_ALPHA)]


def test_add_mc_fan_reference_mode_is_gray_and_labelled():
    """reference=True (regime filter on a cached scenario): the fan is the
    all-regimes reference, drawn gray and NAMED in the legend so the reader
    can tell it is not the filtered result."""
    from figures.bubble import _add_mc_fan
    from colors import (MC_GHOST_GRAY, MC_PRICE_BAND_OUTER_ALPHA,
                        MC_PRICE_BAND_INNER_ALPHA, MC_GHOST_MEDIAN_ALPHA,
                        _hex_alpha)
    import plotly.graph_objects as go

    paths, t_axis = _fan_fixture()
    fig = go.Figure()
    _add_mc_fan(fig, paths, t_axis, reference=True)
    named = _by_name(fig)
    assert set(named) == {"MC ref 5\u201395%", "MC ref 25\u201375%", "MC ref median"}
    assert named["MC ref 5\u201395%"].fillcolor == _hex_alpha(MC_GHOST_GRAY, MC_PRICE_BAND_OUTER_ALPHA)
    assert named["MC ref 25\u201375%"].fillcolor == _hex_alpha(MC_GHOST_GRAY, MC_PRICE_BAND_INNER_ALPHA)
    assert named["MC ref median"].line.color == _hex_alpha(MC_GHOST_GRAY, MC_GHOST_MEDIAN_ALPHA)
    assert all(t.showlegend for t in named.values()), \
        "the reference fan must appear in the legend on Tab 1"


def test_add_mc_fan_band_edges_are_explicitly_transparent():
    """width=0 edge lines need an explicit transparent colour, or mobile
    browsers paint phantom hairlines in the default colour cycle."""
    from figures.bubble import _add_mc_fan
    from colors import BLACK_A0
    import plotly.graph_objects as go

    paths, t_axis = _fan_fixture()
    fig = go.Figure()
    _add_mc_fan(fig, paths, t_axis)
    edges = [t for t in fig.data if t.line.width == 0]
    assert len(edges) == 4
    assert all(t.line.color == BLACK_A0 for t in edges)
    assert all(t.hoverinfo == "skip" for t in edges)


def test_add_mc_fan_median_hover_names_the_calendar_date():
    """The chart's x is years-since-genesis; the median's hover must show a
    date, like every other line on Tab 1."""
    from figures.bubble import _add_mc_fan
    from figures.common import _HOVER_FMT_USD
    import plotly.graph_objects as go

    paths, t_axis = _fan_fixture()
    fig = go.Figure()
    _add_mc_fan(fig, paths, t_axis)
    med = _by_name(fig)["MC median"]
    assert med.hovertemplate == _HOVER_FMT_USD
    assert len(med.customdata) == len(t_axis)
    first = med.customdata[0]
    first = first if isinstance(first, str) else first[0]
    assert "2028" in first, f"t=18.5 is early 2028; got {first!r}"


def test_add_mc_fan_leaves_existing_traces_untouched():
    """Only the fan's own traces get hover data — never the chart's."""
    from figures.bubble import _add_mc_fan
    import plotly.graph_objects as go

    paths, t_axis = _fan_fixture()
    fig = go.Figure(go.Scatter(x=[18.6, 19.0], y=[1.0, 2.0], name="pre"))
    _add_mc_fan(fig, paths, t_axis)
    assert fig.data[0].name == "pre"
    assert fig.data[0].customdata is None
    assert fig.data[0].hovertemplate is None
    assert len(fig.data) == 1 + 5


def test_add_mc_fan_handles_empty_paths():
    """None / zero-row input is a no-op, never an exception."""
    from figures.bubble import _add_mc_fan
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    _add_mc_fan(fig, None, np.arange(5))
    _add_mc_fan(fig, np.array([]).reshape(0, 5), np.arange(5))
    assert len(fig.data) == 0

